"""Step-driven mixed decode-PoC support.

A decode-PoC request runs ONE decode token per scheduler step, mixed with chat
in the same forward (instead of running its whole decode loop inside one
pure-batch call). Its KV is allocated on demand by the KV manager exactly like
chat (pure dynamic KV, no reserved blocks) so its prefill KV persists and each
decode step reads it. ``sphere_k`` is chained across steps via the per-request
state held here.

Each PoC request carries a single nonce (routes fan out per-nonce via
``generate(poc_params)``), so the decode state is per-(request, nonce).

The slot/layout helpers at the top are pure (no torch) and unit-tested. The
model-runner helpers at the bottom (moved out of gpu_model_runner.py to keep the
core vLLM footprint minimal) take the GPUModelRunner as ``runner``. Validation
runs pure.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from gonka_poc.mixed.policy import poc_cfg

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import PoCOutput

logger = init_logger(__name__)

# SNAP MARGIN (validator-side). Every teacher-forced disagreement is counted, and the
# largest snap margin (top1-top2 cosine gap of the validator's OWN query) among the
# disagreeing steps is emitted as the nonce's distance. A tiny margin means the query
# sat on a codebook boundary, where cross-HW/backend fp jitter flips the snap; a
# structured fraud pushes the query decisively into a wrong cell. The threshold that
# separates the two (tau) is NOT applied here: it arrives with the validation request
# as stat_test.dist_threshold, exactly like the prefill L2 threshold, so a validator
# cannot move its own floor through the environment.

def _vector_artifact_cfg(runner) -> bool:
    """poc_vector_artifacts enabled? The dim is the PoC's own k_dim and the window
    is every step — no separate knobs (retired poc_vector_artifact_steps/_dim).
    Read from ``--additional-config`` (``gonka_poc.poc_vector_artifacts``);
    absent → disabled."""
    return bool(poc_cfg(getattr(runner, "vllm_config", None), "poc_vector_artifacts"))


_EMIT_STAT = {"n": 0, "k": 0.0, "q": 0.0, "q_steps": 0}
_VEC_STAT = {"n": 0, "mt": None, "len": None, "has_mgr": None}


def _diag_now():
    """POC_DIAG=1: monotonic time (sec) for emission timers; else None."""
    import os, time
    if os.environ.get("POC_DIAG", "") != "1":
        return None
    return time.monotonic()


def keep_q_step(step: int, debug: bool, va_on: bool) -> bool:
    """Which decode steps retain their pre-snap q for emission: every step under
    debug or poc_vector_artifacts (aligned to the PoC step count — one vector per
    step, all max_tokens of them). Pure."""
    return debug or va_on


def encode_sph_slices(q_host, block_hash, public_key, nonce, k_dim, debug):
    """Wire-encode each kept step's q (one fp16-LE base64 row, vector_b64 codec).
    Non-debug: a SEEDED k_dim-pick of the SPHERE_DIM coords per step — the SAME
    sampler as prefill (random_pick_indices), NOT a leading slice — so prover and
    validator select the identical coords from the (identical) step vector. The
    row's position in q_host IS its step (0 = prefill, 1..max_tokens = decode), so
    both sides seed the pick on the same (block_hash, public_key, nonce, step).
    Debug ships full width. Not renormalized; the scorer renormalizes both sides."""
    import torch

    from gonka_poc.poc.data import encode_vector
    from gonka_poc.poc.decode_random import random_pick_indices_decode_steps
    from gonka_poc.poc.sphere import SPHERE_DIM
    if debug:
        return [encode_vector(row) for row in q_host]
    cpu = torch.device("cpu")
    # All steps' indices in one batch; bitwise equal to per-step
    # random_pick_indices_decode (tests/unit/test_pick_indices_steps.py).
    idx_all = random_pick_indices_decode_steps(
        block_hash, public_key, nonce, SPHERE_DIM, k_dim, cpu,
        list(range(len(q_host)))).numpy()
    return [encode_vector(row[idx_all[step]]) for step, row in enumerate(q_host)]


def slice_sampling_metadata(sm, rows, device):
    """Restrict a SamplingMetadata to `rows` (input_batch indices), so the sampler
    runs on chat rows only. PoC rows have no sampling semantics; keeping them out
    avoids stale/oversized penalty tensors and the per-row param mismatch."""
    import dataclasses

    from gonka_poc.poc.decode_random import pinned_to_device

    idx = pinned_to_device(rows, torch.long, device)
    keep = set(rows)
    remap = {old: new for new, old in enumerate(rows)}

    def take_t(t):
        return None if t is None else t[idx]

    def take_list(lst):
        return [lst[i] for i in rows] if lst else lst

    def remap_dict(d):
        return None if d is None else {remap[k]: v for k, v in d.items() if k in keep}

    sliced = dataclasses.replace(
        sm,
        temperature=take_t(sm.temperature),
        top_p=take_t(sm.top_p),
        top_k=take_t(sm.top_k),
        generators=remap_dict(sm.generators) or {},
        logprob_token_ids=remap_dict(sm.logprob_token_ids),
        prompt_token_ids=take_t(sm.prompt_token_ids),
        frequency_penalties=take_t(sm.frequency_penalties),
        presence_penalties=take_t(sm.presence_penalties),
        repetition_penalties=take_t(sm.repetition_penalties),
        output_token_ids=take_list(sm.output_token_ids),
        spec_token_ids=take_list(sm.spec_token_ids),
        allowed_token_ids_mask=take_t(sm.allowed_token_ids_mask),
        bad_words_token_ids=remap_dict(sm.bad_words_token_ids),
    )
    # enforced_next_token_ids: 0.20-only field (inference-validation feature,
    # not decode-PoC). Slice it when the base engine carries it.
    if hasattr(sm, "enforced_next_token_ids"):
        sliced.enforced_next_token_ids = take_t(sm.enforced_next_token_ids)
    return sliced







@dataclass
class PoCDecodeState:
    """Per-request decode state, carried across scheduler steps."""
    nonce: int
    slot: int
    seq_len: int
    max_tokens: int
    # number of decode steps completed so far (0 == only prefill done).
    step: int = 0
    # previous step's sphere_k; seeds the next step's input embedding + the
    # per-step random dimension selection. -1 before the prefill sphere_k is set.
    prev_k: int = -1
    # full sphere_k trajectory: prefill k, then one per decode step.
    k_points_steps: list[int] = field(default_factory=list)
    # the prefill artifact vector (base64), set at the prefill step.
    vector_b64: str = ""
    n_sphere_mismatches: int = 0
    # validation reference trajectory (enforced_k_steps), or None for
    # generation. index 0 = prefill k, 1..N = decode-step k. Drives the per-step comparison.
    reference: list | None = None
    # --- GPU-native chaining: prev_k stays on device so the per-step host sync
    # disappears (-> async scheduling works). The trajectory accumulates on device
    # and is copied to host ONCE at end-of-sequence (emit-once), so no per-step
    # delta crosses the IPC boundary. ---
    base_seeds: "torch.Tensor | None" = None     # [1] int64 per-nonce base (set once)
    prev_k_t: "torch.Tensor | None" = None        # [1] int64, chained on device
    reference_t: "torch.Tensor | None" = None     # [R] int64 uploaded reference
    k_steps_t: list = field(default_factory=list)  # list of [1] int64; cat+tolist at end
    # margin trajectory, parallel to k_steps_t. The mismatch count is DEFERRED to
    # emit (one batched reduction over k+margin vs the reference) instead of a
    # per-step accumulate; n_nan is derived at emit from k == -1 (the snap marks
    # non-finite steps that way). So the hot decode loop does no counter ops.
    margin_steps_t: list = field(default_factory=list)
    # per-step pre-snap sphere slices (the q whose argmax is sphere_k) for
    # PoCOutput.sph_values_steps: every step under debug or poc_vector_artifacts
    # (the seeded k_dim-coord pick is applied at emit — see encode_sph_slices).
    # [1, SPHERE_DIM] floats, index 0 = prefill; device-accumulated like
    # k_steps_t (no per-step host sync), encoded once at emit.
    q_steps_t: list = field(default_factory=list)


class PoCMixedDecodeManager:
    """Per-request decode state for step-driven mixed decode-PoC (one instance
    per model runner). Unbounded: a row keeps its state across preemptions, so
    rows holding state can exceed the rows running (max_num_seqs), and a finite
    slot pool sized by max_num_seqs ran dry under preemption (MiniMax-M2.7,
    max_num_seqs 128, window 256: 53 of 1000 nonces came back without a
    trajectory). KV itself is paged via the scheduler — the state holds no blocks.
    """

    def __init__(self, poc_max_batch_size: int = 0):
        self._state: dict[str, PoCDecodeState] = {}
        self._next_slot = 0

    def get(self, req_id: str) -> PoCDecodeState | None:
        return self._state.get(req_id)

    def allocate(self, req_id: str, nonce: int, seq_len: int,
                 max_tokens: int) -> PoCDecodeState | None:
        existing = self._state.get(req_id)
        if existing is not None:
            return existing
        st = PoCDecodeState(
            nonce=nonce, slot=self._next_slot, seq_len=seq_len, max_tokens=max_tokens
        )
        self._next_slot += 1
        self._state[req_id] = st
        return st

    def free(self, req_id: str) -> None:
        self._state.pop(req_id, None)


def get_decode_manager(runner) -> "PoCMixedDecodeManager":
    """Lazily get/create the per-runner mixed-decode state store."""
    mgr = getattr(runner, "_poc_mixed_decode_mgr", None)
    if mgr is None:
        mgr = PoCMixedDecodeManager()
        runner._poc_mixed_decode_mgr = mgr
    return mgr


def setup_decode_poc(runner, poc_requests) -> bool:
    """Entry hook (called from gpu_model_runner before _prepare_inputs).

    For each decode-PoC request (max_tokens>0): grab a state slot + refresh its
    per-request decode step counter. KV is pure dynamic: the scheduler-allocated
    paged block-table row already drives decode (see below).

    Returns True if any decode-PoC is active this step, signalling the caller to
    route the batch through the unified step-driven path. Returns False when there
    are no decode-PoC requests. Generation and validation both run here; validation
    carries its reference trajectory in PoCDecodeState.reference (aligned compare).
    """
    decode_reqs = [r for r in poc_requests if r.poc_params.max_tokens > 0]
    if not decode_reqs:
        return False
    mgr = get_decode_manager(runner)
    # Pure dynamic KV: the scheduler allocated real (paged) blocks via the
    # manager, so _prepare_inputs already built the correct block-table row +
    # slot_mapping. We only track decode state (step + reference trajectory).
    for r in decode_reqs:
        pp = r.poc_params
        # Emit-once: overshoot steps (pipelined after finish) get no state.
        if max(0, r.num_computed_tokens - pp.seq_len) >= pp.max_tokens:
            continue
        st = mgr.allocate(r.req_id, pp.nonce, pp.seq_len, pp.max_tokens)
        if st is None:
            # Pool exhausted (scheduler caps to poc_max_batch_size; defensive).
            logger.warning("PoC mixed-decode slot pool exhausted for %s",
                           r.req_id)
            continue
        # decode step = tokens computed beyond prefill (0 during the prefill step)
        st.step = max(0, r.num_computed_tokens - pp.seq_len)
        st.reference = pp.enforced_k_steps  # None for generation
    return True


# ---------------------------------------------------------------------------
# Mixed-batch model-runner helpers (moved out of gpu_model_runner.py to keep the
# core vLLM footprint minimal). Each takes the GPUModelRunner as `runner`.
# ---------------------------------------------------------------------------

def _cat_prev_k(states, where: str) -> "torch.Tensor":
    """torch.cat of per-nonce prev_k, with the invariant stated explicitly.

    ``prev_k_t`` is published by the PREFILL snap. A row can only reach the decode
    path with it still None if its prefill output has not been processed yet —
    ``num_computed_tokens`` is advanced at schedule time, so it reads >= seq_len
    while the prefill forward is still in flight.

    The prefill snap that publishes prev_k runs in the worker's execute_model
    (bridge.extract) before the next step's inputs are built, so this can only
    trip when the nonce had no state slot at prefill time. The pool is sized by
    max_num_seqs (see get_decode_manager), which vLLM never exceeds; the
    scheduler-side one-step hold that used to guard this was removed on
    2026-09-05 after window-1 stress runs (prefill and first decode in adjacent
    steps) on B300 never tripped it. Detection stays: the batch fails loudly, its
    nonces come back empty, and the corpus guard catches them.

    Substituting a placeholder here would be WORSE than failing: prev_k < 0 makes
    the embedding wrapper fall back to the prefill embed, so the nonce would keep
    running with a silently wrong decode input and produce a plausible but invalid
    trajectory.
    """
    missing = [i for i, st in enumerate(states) if st.prev_k_t is None]
    if missing:
        raise RuntimeError(
            f"PoC {where}: {len(missing)} of {len(states)} decode rows have no "
            f"prev_k (prefill output not processed yet), rows {missing[:8]}. "
            "Known race, see this function's docstring; observed only at "
            "poc_max_batch_size=1.")
    return torch.cat([st.prev_k_t for st in states])


def build_unified_mixed_batch_inputs(
    runner,
    scheduler_output: "SchedulerOutput",
    chat_input_ids: torch.Tensor | None,
    chat_inputs_embeds: torch.Tensor | None,
    chat_positions: torch.Tensor,
    poc_req_ids: set,
    num_total_tokens: int,
    batch_view: tuple | None = None,
    req_views: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Build unified inputs for mixed batch (chat + PoC in same forward).

    CRITICAL: Preserves scheduler's token order to match slot_mapping.
    Tokens are built in the exact order of runner.input_batch.req_ids.

    Args:
        scheduler_output: The scheduler output with token counts
        chat_input_ids: Chat token IDs [num_total_tokens] or None
        chat_inputs_embeds: Chat embeddings [num_total_tokens, hidden] or None
        chat_positions: Chat positions [num_total_tokens]
        poc_req_ids: Set of PoC request IDs
        num_total_tokens: Total scheduled tokens (chat + PoC)

    Returns:
        Tuple of:
        - unified_embeds: [num_total_tokens, hidden_size]
        - unified_positions: [num_total_tokens]
        - poc_position_mask: [num_total_tokens] bool tensor (True = PoC)
        - poc_metadata: List of dicts with PoC request info
    """
    from gonka_poc.poc.gpu_random import generate_inputs

    hidden_size = runner.model_config.get_hidden_size()
    # batch_view: (num_reqs, req_ids) — plain metadata handed in by the caller
    # so this works on either runner (V1 exposes runner.input_batch; V2 keeps
    # the batch local to execute_model). No tensor copies.
    if batch_view is not None:
        num_reqs, req_ids = batch_view
    else:
        num_reqs = runner.input_batch.num_reqs
        req_ids = runner.input_batch.req_ids

    tokens_per_req = [scheduler_output.num_scheduled_tokens[req_id]
                      for req_id in req_ids]

    unified_embeds = torch.empty(
        (num_total_tokens, hidden_size),
        dtype=runner.dtype,
        device=runner.device,
    )
    unified_positions = torch.empty(
        num_total_tokens,
        dtype=chat_positions.dtype,
        device=runner.device,
    )
    poc_position_mask = torch.zeros(
        num_total_tokens,
        dtype=torch.bool,
        device=runner.device,
    )
    poc_metadata = []

    offset = 0
    # Decode-step embeddings are generated in ONE batched call after this loop
    # (was per-nonce). Each entry: (decode_state, decode_step, offset).
    decode_embed_jobs = []
    decode_positions: list[int] = []

    for req_idx in range(num_reqs):
        req_id = req_ids[req_idx]
        num_tokens = tokens_per_req[req_idx]

        if num_tokens <= 0:
            continue

        if req_id in poc_req_ids:
            # req_views: runner-agnostic per-request view (V1 keeps objects,
            # V2 columnar tensors); fall back to the V1 store when absent.
            req_state = (req_views[req_id] if req_views is not None
                         else runner.requests[req_id])
            poc_params = req_state.poc_params
            seq_len = poc_params.seq_len
            mgr = getattr(runner, "_poc_mixed_decode_mgr", None)
            st = mgr.get(req_id) if mgr is not None else None

            if st is not None and req_state.num_computed_tokens >= seq_len:
                # Decode step: one token, embed chained from prev sphere_k.
                # GPU-native: prev_k is a device tensor (set by the previous step's
                # output processing) -> no host sync -> async-scheduling safe. The
                # embedding itself is generated in ONE batched call after the loop.
                from gonka_poc.poc.decode_random import decode_base_seeds
                decode_step = req_state.num_computed_tokens - seq_len + 1
                if st.base_seeds is None:
                    st.base_seeds = decode_base_seeds(
                        poc_params.block_hash, poc_params.public_key,
                        [poc_params.nonce], runner.device)
                decode_embed_jobs.append(
                    (st, decode_step, offset,
                     (poc_params.block_hash, poc_params.public_key,
                      poc_params.nonce)))
                # Positions/mask for decode rows are written in ONE batched
                # index_copy_/index_fill_ after the loop: a per-row scalar
                # assignment is a separate H2D copy (~25 us/row/step at batch
                # 1024 it dominated the whole step).
                decode_positions.append(req_state.num_computed_tokens)
                poc_metadata.append({
                    'type': 'poc', 'req_id': req_id, 'start_idx': offset,
                    'length': 1, 'poc_params': poc_params,
                    'decode_state': st, 'decode_step': decode_step,
                })
                offset += 1
            elif (poc_params.max_tokens > 0
                  and req_state.num_computed_tokens >= seq_len):
                # Stale row: already emitted and freed (st is None),
                # but the async scheduler gave it one more step. Otherwise it would
                # take the prefill path (generate_inputs + host-sync per row).
                # No output needed: zeros, mask False, no metadata.
                unified_embeds[offset:offset + num_tokens].zero_()
                unified_positions[offset:offset + num_tokens] = (
                    chat_positions[offset:offset + num_tokens])
                offset += num_tokens
            else:
                # Prefill (prefill-only PoC, or the prefill step of a decode-PoC).
                poc_len = num_tokens
                poc_embeds = generate_inputs(
                    poc_params.block_hash,
                    poc_params.public_key,
                    [poc_params.nonce],
                    dim=hidden_size,
                    seq_len=poc_len,
                    device=runner.device,
                    dtype=runner.dtype,
                ).squeeze(0)  # [poc_len, hidden]
                unified_embeds[offset:offset + poc_len] = poc_embeds
                unified_positions[offset:offset + poc_len] = torch.arange(
                    poc_len, device=runner.device, dtype=chat_positions.dtype
                )
                poc_position_mask[offset:offset + poc_len] = True
                # Pseudo token ids for the prefill rows (token-id-routed models only):
                # same derivation as the prefill scheme, so both schemes see the
                # same ids for a nonce.
                native = getattr(runner, "_poc_native", None)
                if native is not None and getattr(native, "token_id_vocab", 0):
                    from gonka_poc.poc.gpu_random import derive_pseudo_input_ids
                    native.set_prefill_token_ids(
                        torch.arange(offset, offset + poc_len,
                                     device=runner.device, dtype=torch.long),
                        derive_pseudo_input_ids(
                            poc_params.block_hash, poc_params.public_key,
                            [poc_params.nonce], poc_len,
                            native.token_id_vocab, runner.device))
                poc_metadata.append({
                    'type': 'poc', 'req_id': req_id, 'start_idx': offset,
                    'length': poc_len, 'poc_params': poc_params,
                    'decode_state': st,
                })
                offset += poc_len

        else:
            if chat_inputs_embeds is not None:
                unified_embeds[offset:offset + num_tokens] = (
                    chat_inputs_embeds[offset:offset + num_tokens]
                )
            elif chat_input_ids is not None:
                token_ids = chat_input_ids[offset:offset + num_tokens]
                # v15 renamed get_input_embeddings -> embed_input_ids (and
                # the model is the cudagraph wrapper, which forwards it).
                chat_embeds = runner.model.embed_input_ids(input_ids=token_ids)
                unified_embeds[offset:offset + num_tokens] = chat_embeds

            unified_positions[offset:offset + num_tokens] = (
                chat_positions[offset:offset + num_tokens]
            )
            offset += num_tokens

    # Batched decode-step embeddings: one generate_decode_inputs_gpu call for the
    # whole nonce-batch (per-row identical to the old per-nonce calls).
    from gonka_poc.poc.decode_random import generate_decode_inputs_gpu, pinned_to_device
    decode_offs = (pinned_to_device([j[2] for j in decode_embed_jobs],
                                    torch.long, runner.device)
                   if decode_embed_jobs else None)
    if decode_offs is not None:
        unified_positions.index_copy_(
            0, decode_offs,
            pinned_to_device(decode_positions, unified_positions.dtype,
                             runner.device))
        poc_position_mask.index_fill_(0, decode_offs, True)
    # Chain tensors for this step. base_seeds is constant per nonce, and prev_k is
    # what the previous step's snap already produced for the SAME rows in the SAME
    # order — catting B one-element tensors every step was O(B) host work on the
    # critical path. Rebuilt only when the row set changes (key = nonces in row
    # order), or when any row is teacher-forced from a reference trajectory.
    chain = getattr(runner, "_poc_chain", None)
    if chain is None:
        chain = runner._poc_chain = {}
    # The key must carry EVERY field the cached value depends on:
    # base_seeds = decode_base_seeds(block_hash, public_key, nonce), plus the step
    # each row is on. Keying on the nonce tuple alone let a second round over the
    # same nonce range reuse the first round's seeds and prev_k — step 0 still
    # matched, every later step chained from the wrong base.
    chain_key = ((tuple(j[3] for j in decode_embed_jobs),
                  tuple(j[1] for j in decode_embed_jobs))
                 if decode_embed_jobs else ())
    base_cat = prev_cat = None
    if decode_embed_jobs:
        base_key = chain_key[0]
        if chain.get("base_key") == base_key:
            base_cat = chain["base"]
        else:
            base_cat = torch.cat([j[0].base_seeds for j in decode_embed_jobs])
            chain["base_key"], chain["base"] = base_key, base_cat
        if chain.get("prev_key") == chain_key and chain.get("prev") is not None:
            prev_cat = chain["prev"]
        else:
            prev_cat = _cat_prev_k([j[0] for j in decode_embed_jobs],
                                   "decode-embed")
        chain["prev"] = None            # consumed; the snap below publishes the next
        chain["step_key"] = chain_key
        chain["step_base"], chain["step_prev"] = base_cat, prev_cat
    _nat = getattr(runner, "_poc_native", None)
    if _nat is not None and getattr(_nat, "embed_base", None) is not None:
        # SYNTH = EMBEDDING: publish per-row (base, prev_k, step); the embedding
        # wrapper synths input[step] in-graph. No eager RNG, no [B,H] embed copy.
        if decode_embed_jobs:
            _nat.set_decode_chain(
                offs=decode_offs, base=base_cat, prev_k=prev_cat,
                step=pinned_to_device([j[1] for j in decode_embed_jobs], torch.int64, runner.device))
        else:
            _nat.set_decode_chain()          # no decode rows in this forward
    elif decode_embed_jobs:
        steps = pinned_to_device([j[1] for j in decode_embed_jobs], torch.int64, runner.device)
        embeds = generate_decode_inputs_gpu(
            base_cat, prev_cat, steps,
            dim=hidden_size, device=runner.device, dtype=runner.dtype)  # [B, 1, H]
        unified_embeds.index_copy_(0, decode_offs, embeds[:, 0])  # [B, H] -> rows

    return unified_embeds, unified_positions, poc_position_mask, poc_metadata


def process_poc_outputs_from_hidden(
    runner,
    hidden_states: torch.Tensor,
    poc_metadata: list[dict],
) -> dict[str, "PoCOutput"]:
    from vllm.v1.outputs import PoCOutput
    from gonka_poc.poc.gpu_random import apply_haar_rotation
    from gonka_poc.poc.decode_random import (
        random_pick_indices_decode, decode_base_seeds, random_pick_indices_gpu,
        pinned_to_device,
    )
    from gonka_poc.poc.data import encode_vector
    from gonka_poc.poc.sphere import (
        SPHERE_DIM, get_sphere_codebook, project_to_sphere, snap_with_margin,
    )

    poc_outputs = {}
    # Codebook is constant (device-only); cache on the runner instead of re-copying
    # it every step. nearest_sphere_index casts to float, so dtype here is moot.
    codebook = getattr(runner, "_poc_codebook", None)
    if codebook is None:
        codebook = get_sphere_codebook().to(device=runner.device)
        runner._poc_codebook = codebook
    # engine-static: safe to fetch once per forward
    va_on = _vector_artifact_cfg(runner)

    # Decode steps are the hot path; collect them and run ONE batched set of GPU
    # ops for the whole nonce-batch below (was a per-nonce Python loop = B× the
    # kernel launches). Prefill-only / prefill-step PoCs (rare, once per request)
    # stay inline.
    decode_metas = []

    for meta in poc_metadata:
        st = meta.get('decode_state')
        if st is not None and 'decode_step' in meta:
            decode_metas.append(meta)
            continue

        end = meta['start_idx'] + meta['length']
        poc_params = meta['poc_params']
        nonce = poc_params.nonce

        last_hidden = hidden_states[end - 1].float()
        last_hidden = last_hidden / (last_hidden.norm() + 1e-8)
        hidden_size = last_hidden.shape[-1]

        def _vector_b64():
            idx = random_pick_indices_decode(
                poc_params.block_hash, poc_params.public_key, [nonce],
                hidden_size, poc_params.k_dim, runner.device)
            xk = last_hidden[idx[0]]
            yk = apply_haar_rotation(
                poc_params.block_hash, poc_params.public_key, [nonce],
                xk.unsqueeze(0), runner.device)[0]
            yk = yk / (yk.norm() + 1e-8)
            return encode_vector(yk.half().cpu().numpy())

        def _sphere_from_idx(sph):
            """hidden -> (sphere index, non-finite mask, margin, pre-snap slice) as
            [1]/[1]/[1]/[1, SPHERE_DIM] TENSORs (no .item(), so the chain stays on
            GPU and async scheduling works)."""
            xk_sphere = project_to_sphere(torch.gather(last_hidden.unsqueeze(0), 1, sph))
            k_, bad_, margin_ = snap_with_margin(xk_sphere, codebook)
            return k_, bad_, margin_, xk_sphere

        if st is None:
            # Prefill-only PoC: just the vector_b64 artifact.
            if _diag_now() is not None:
                _VEC_STAT["n"] += 1
                _VEC_STAT["mt"] = poc_params.max_tokens
                _VEC_STAT["len"] = meta['length']
                _VEC_STAT["has_mgr"] = get_decode_manager(runner).get(meta['req_id']) is not None
            poc_outputs[meta['req_id']] = PoCOutput(
                nonce=nonce, vector_b64=_vector_b64())
            continue

        # Prefill step of a decode-PoC: compute the prefill sphere_k (k0) and start
        # the on-device trajectory. Decode is scored on k_points_steps; the chain
        # (prev_k_t) stays on device until end-of-sequence (emit-once).
        if st.base_seeds is None:
            st.base_seeds = decode_base_seeds(
                poc_params.block_hash, poc_params.public_key, [nonce], runner.device)
        sph0 = random_pick_indices_decode(
            poc_params.block_hash, poc_params.public_key, [nonce],
            hidden_size, SPHERE_DIM, runner.device)
        k0_t, _bad0, margin0, q0 = _sphere_from_idx(sph0)   # [1]/[1]/[1]/[1,SPHERE_DIM]
        # (nan is derived at emit from k == -1; _bad0 no longer accumulated per step)
        st.k_steps_t = [k0_t]
        st.margin_steps_t = [margin0]                       # for the deferred mismatch (emit)
        # emission only — forward unchanged (contract: q_steps_t field doc)
        st.q_steps_t = [q0.detach()] if (poc_params.debug or va_on) else []
        if st.reference is not None:
            st.reference_t = torch.tensor(
                st.reference, dtype=torch.int64, device=runner.device)
            st.prev_k_t = st.reference_t[0:1]               # aligned (teacher-forced)
        else:
            st.prev_k_t = k0_t

    # Batched decode step: one set of GPU ops (seed -> pick -> sphere) for the whole
    # nonce-batch. Per-row results are identical to the old per-nonce calls (same
    # seeds, murmur, topk, gather, argmax), so artifacts are unchanged.
    if decode_metas:
        device = runner.device
        H = hidden_states.shape[-1]
        _chain = getattr(runner, "_poc_chain", None)
        if _chain is None:
            _chain = runner._poc_chain = {}
        _metas_key = (tuple((m['poc_params'].block_hash,
                             m['poc_params'].public_key, m['poc_params'].nonce)
                            for m in decode_metas),
                      tuple(m['decode_step'] for m in decode_metas))
        _any_reference = False
        # Post-forward sphere-snap. The final-norm wrapper snapped every row IN-GRAPH
        # this forward, so we just index_select the decode rows (no per-step eager
        # tail). The eager path below is the fallback for models without native PoC
        # wrappers (_snap_active False).
        from torch.autograd.profiler import record_function
        _nat = getattr(runner, "_poc_native", None)
        _snap_active = (_nat is not None and getattr(_nat, "snap_k", None) is not None
                        and getattr(_nat, "embed_base", None) is not None)
        # Does ANY row keep its pre-snap q (vector artifacts / debug)? Needed before the
        # snap so the snap-active path can skip pulling snap_q when nothing keeps it — on
        # the honest hot path q is dead, so that index_select is a wasted launch/step.
        _keep_any = va_on or any(m['poc_params'].debug for m in decode_metas)
        if _snap_active:
            with record_function("poc_snap_in_forward"):
                # SNAP = SAMPLING: the final-norm wrapper already snapped every row
                # IN-GRAPH this forward; just index_select the decode rows. No tail-
                # graph feed (4 index_copy_/step), no separate replay. Same math/inputs
                # as the tail (embed_* == decode_state), so artifacts are identical.
                rows = pinned_to_device(
                    [m['start_idx'] + m['length'] - 1 for m in decode_metas],
                    torch.long, device)
                k_all = _nat.snap_k.index_select(0, rows)
                margin_all = _nat.snap_margin.index_select(0, rows)
                # snap_bad is never read here (n_nan is derived at emit from k == -1),
                # and snap_q only feeds kept q artifacts. Both were dead index_select
                # launches on the honest hot path -> pull snap_q only when kept, skip
                # snap_bad entirely. Artifacts byte-identical (unused values).
                q_all = _nat.snap_q.index_select(0, rows) if _keep_any else None
        else:
            with record_function("poc_tail_eager_path"):
                # sync-free host->device: hidden[py_list] and torch.tensor(list, device=cuda)
                # each block on the forward every step (see pinned_to_device); index_select
                # with a pinned index avoids it. Values (hence artifacts) are identical.
                idxs = [m['start_idx'] + m['length'] - 1 for m in decode_metas]
                lh = hidden_states.index_select(
                    0, pinned_to_device(idxs, torch.long, device)).float()   # [B, H]
                lh = lh / (lh.norm(dim=-1, keepdim=True) + 1e-8)
                # Same rows, same step: the batch builder already assembled these.
                if _chain.get("step_key") == _metas_key:
                    base_seeds, prev_k = _chain["step_base"], _chain["step_prev"]
                else:
                    base_seeds = torch.cat([m['decode_state'].base_seeds for m in decode_metas])
                    prev_k = _cat_prev_k([m['decode_state'] for m in decode_metas],
                                         "snap")
                steps = pinned_to_device([m['decode_step'] for m in decode_metas], torch.int64, device)
                sph = random_pick_indices_gpu(base_seeds, prev_k, steps, H, SPHERE_DIM, device)
                # snap_with_margin: argmax(NaN) is garbage -> non-finite rows return k=-1
                # (compute fault, NOT fraud). bad_all/margin_all stay on device (no per-step sync).
                q_all = project_to_sphere(torch.gather(lh, 1, sph))          # [B, SPHERE_DIM]
                k_all, bad_all, margin_all = snap_with_margin(q_all, codebook)  # [B] each

        # q_clone: one clone of q for the whole batch (the loop then only does cheap
        # views + list appends). Shared by this step's kept rows, alive until emit.
        # (_keep_any computed above, before the snap, to gate the snap_q pull.)
        q_clone = q_all.detach().clone() if _keep_any else None
        _chain_prev_candidate = k_all.detach()
        for i, meta in enumerate(decode_metas):
            st = meta['decode_state']
            step = meta['decode_step']
            k_t = k_all[i:i + 1]                               # [1] tensor (view)
            st.k_steps_t.append(k_t)
            st.margin_steps_t.append(margin_all[i:i + 1])     # deferred mismatch (emit)
            if keep_q_step(step, meta['poc_params'].debug, va_on):
                st.q_steps_t.append(q_clone[i:i + 1])         # view into the one per-step clone
            # chain only (teacher-forced ref, or free-running k) — no per-step counters
            if st.reference_t is not None and step < st.reference_t.shape[0]:
                st.prev_k_t = st.reference_t[step:step + 1]   # aligned (teacher-forced)
                _any_reference = True
            else:
                st.prev_k_t = k_t
            if step >= st.max_tokens:
                _emit_t0 = _diag_now()
                # End-of-sequence: ONE host copy of the trajectory + the DEFERRED
                # reductions (emit-once). n_nan = non-finite steps (snap marks k=-1);
                # mismatch = finite disagreements vs the reference over the whole
                # k trajectory, plus the largest margin among them — batched here so
                # the hot loop stays counter-free.
                k_traj = torch.cat(st.k_steps_t)              # [L] int64 on device
                k_points = k_traj.tolist()
                n_nan = int((k_traj == -1).sum().item())
                mismatch_margin_max = 0.0
                if st.reference_t is not None:
                    margin_traj = torch.cat(st.margin_steps_t)
                    L = min(k_traj.shape[0], st.reference_t.shape[0])
                    disagree = ((k_traj[:L] != st.reference_t[:L])
                                & (k_traj[:L] >= 0))
                    n_mismatches = int(disagree.sum().item())
                    if n_mismatches:
                        mismatch_margin_max = float(
                            margin_traj[:L][disagree].max().item())
                else:
                    n_mismatches = -1
                if n_nan:
                    logger.warning(
                        "PoC decode nonce %s: %d/%d non-finite hidden step(s) "
                        "(compute fault, NOT fraud; excluded from mismatch rate) — "
                        "trajectory suspect, re-run on a clean GPU",
                        meta['poc_params'].nonce, n_nan, len(k_points))
                _emit_t1 = _diag_now()
                sph_vals = []
                if st.q_steps_t:
                    # cat on device, then ONE host copy for the whole trajectory
                    # (a per-step .cpu() would sync T+1 times at emit).
                    q_host = torch.cat(st.q_steps_t).cpu().numpy()
                    pp = meta['poc_params']
                    sph_vals = encode_sph_slices(
                        q_host, pp.block_hash, pp.public_key, pp.nonce,
                        pp.k_dim, pp.debug)
                _emit_t2 = _diag_now()
                if _emit_t0 is not None:
                    _EMIT_STAT["n"] += 1
                    _EMIT_STAT["k"] += _emit_t1 - _emit_t0
                    _EMIT_STAT["q"] += _emit_t2 - _emit_t1
                    _EMIT_STAT["q_steps"] += len(st.q_steps_t)
                poc_outputs[meta['req_id']] = PoCOutput(
                    nonce=meta['poc_params'].nonce,
                    vector_b64="",
                    k_points_steps=k_points,
                    n_sphere_mismatches=(
                        n_mismatches if st.reference is not None else -1),
                    n_nan_steps=n_nan,
                    mismatch_margin_max=mismatch_margin_max,
                    sph_values_steps=sph_vals,
                )
                get_decode_manager(runner).free(meta['req_id'])

        # Publish this step's k as the next step's prev_k, still batched. Teacher-
        # forced rows chain from the reference instead, so the whole vector is only
        # valid when no row used one; otherwise the next build falls back to the cat.
        if not _any_reference:
            _chain["prev_key"], _chain["prev"] = (
                (_metas_key[0], tuple(s + 1 for s in _metas_key[1])),
                _chain_prev_candidate)

    if _VEC_STAT["n"]:
        logger.info("poc: vector_b64 (st is None branch) for %d rows in step: "
                    "max_tokens=%s length=%s slot in manager=%s",
                    _VEC_STAT["n"], _VEC_STAT["mt"], _VEC_STAT["len"], _VEC_STAT["has_mgr"])
        _VEC_STAT.update(n=0)
    if _EMIT_STAT["n"]:
        logger.info("poc: emitted %d rows: k/margin (cat+tolist+item) %.0f ms, "
                    "q (cat+cpu+encode) %.0f ms, q_steps per row %.0f",
                    _EMIT_STAT["n"], _EMIT_STAT["k"] * 1000, _EMIT_STAT["q"] * 1000,
                    _EMIT_STAT["q_steps"] / _EMIT_STAT["n"])
        _EMIT_STAT.update(n=0, k=0.0, q=0.0, q_steps=0)
    return poc_outputs

