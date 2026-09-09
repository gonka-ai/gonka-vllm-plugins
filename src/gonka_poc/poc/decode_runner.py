"""Decode-PoC loop for the plugin architecture — replaces the 0.20 engine
mixing with a self-contained step loop on the worker.

One chunk = ``len(nonces)`` sequences, each holding ``seq_len + max_tokens``
KV tokens for the whole trajectory:

    prefill (seq_len synthetic embeds, fresh buffer — never a reused scratch)
      -> snap step 0
      -> for t in 1..max_tokens:
           embeds_t = f(prev_k)            (chained, on-device)
           forward 1 token/row             (KV grows into slot seq_len+t-1)
           pick(t) -> project -> snap -> mismatch/teacher-force -> prev_k

Migration rules honoured (NOTES.md):
  * buffers are address-stable, the step body is pure tensor code — CAPTURE-
    READY; the initial revision replays it eagerly step-by-step, the capture
    of the step graph is the first perf iteration, not a redesign;
  * the prefill chunk must fit the pre-sized MoE workspace: chunk ≤
    ``POC_DECODE_PREFILL_CHUNK`` (default 128 nonces × 256 tokens = 32768);
  * shared engine buffers are never resized here (v0.1.3 lesson);
  * fresh prefill embeds always (parity with the 0.20 reference), the legacy
    KV-scratch quirk stays in the prefill-only path of tag v0.1.x.

Consensus arithmetic lives in gpu_random/sphere/decode_chain (layer A,
bit-parity-tested); this module only orchestrates it.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from .gpu_random import (
    decode_base_seeds,
    generate_decode_inputs_gpu,
    generate_inputs,
    random_pick_indices,
    random_pick_indices_gpu,
)
from .sphere import (
    SPHERE_DIM,
    get_sphere_codebook,
    project_to_sphere,
    snap_with_margin,
)
from .decode_chain import count_mismatch, keep_q_step, next_prev_k
from .native import attach_native_poc

logger = logging.getLogger(__name__)

_MARGIN_TAU = float(os.environ.get("VLLM_POC_MARGIN_TAU", "0") or "0")
# Compiled path is the default (0.20 bit reference); eager only as a debug
# fallback behind the flag (design rule: eager is not an execution mode).
_SKIP_COMPILED = os.environ.get("POC_DECODE_SKIP_COMPILED", "0") == "1"
# Per-step wall-clock breakdown (metadata/embeds/forward/snap) logged once per
# chunk; costs one cuda sync per step — diagnostics only, keep off in prod.
_PROFILE = os.environ.get("POC_DECODE_PROFILE", "0") == "1"
# Capture the decode-step forward into a private CUDA graph and replay it:
# the step is launch-bound (profile: forward 98% at ~55ms/step, hundreds of
# kernel launches through the compiled runner), the graph replays the same
# kernels at the same addresses. Bit-neutral by construction; the ladder
# (self-validation + golden tau-gate) re-verifies after every perf change.
_CAPTURE = os.environ.get("POC_DECODE_CAPTURE", "1") == "1"

# Process-lifetime graph cache, keyed by (batch, alloc_len). Two jobs:
#  * replays across RPCs skip the ~seconds-long re-capture per chunk;
#  * keeping graphs (and their private memory pools) alive prevents the
#    known hazard of a dying CUDAGraph freeing its pool while another
#    capture is in flight (observed as async illegal-memory-access when a
#    128-batch capture followed dead 4/32-batch graphs).
# Cached tensors (positions/ids/steps/slot clones/output) are the stable
# addresses the captured kernels read; callers refresh them in place.
_GRAPH_CACHE: Dict[tuple, dict] = {}
POC_DECODE_PREFILL_CHUNK = int(os.environ.get("POC_DECODE_PREFILL_CHUNK", "128"))
# Upper bound for one RPC's nonce count (the joint-decode batch). KV must
# hold batch x (seq_len+max_tokens) tokens; the reservation layer degrades
# to lease=None (abort-based in-place) when the pool cannot cover it.
POC_DECODE_MAX_BATCH = int(os.environ.get("POC_DECODE_MAX_BATCH", "512"))



def _build_step_meta(worker, sm, batch, kv_len, positions, alloc_len,
                     borrowed_block_ids, borrowed_stripe, decode_step):
    """Build decode-step attention metadata from STABLE tensors only.

    Every tensor handed to builder.build() or the forward context lives in
    ``sm`` (per-(batch,alloc_len) bundle) and is refreshed IN PLACE: with a
    captured graph, any kernel that ends up reading one of our tensors must
    find it at the address recorded at capture time. Rebuilding them per
    step worked for generation only by allocator coincidence (identical
    allocation sequences -> identical addresses); one extra allocation in
    the validation path shifted everything and the replays read garbage.
    """
    from gonka_poc._compat import current as _compat_current
    from .poc_model_runner import _borrowed_layout, _inplace_layout
    compat = _compat_current()

    sm["seq_gpu"].fill_(kv_len)
    sm["seq_cpu"].fill_(kv_len)
    sm["ncomp_cpu"].fill_(kv_len - 1)

    def _layout_for_block_size(g_block, m_block):
        key = (g_block, m_block)
        st = sm["groups"].get(key)
        if st is None or sm["lease_dirty"]:
            if borrowed_block_ids is not None:
                slot_all, table = _borrowed_layout(
                    batch, alloc_len, g_block, m_block,
                    borrowed_block_ids, int(borrowed_stripe or 0),
                    positions.device)
            else:
                slot_all, table = _inplace_layout(
                    batch, alloc_len, g_block, positions.device)
            if st is None:
                st = {"slot_all": slot_all.view(batch, alloc_len).clone(),
                      "table": table.clone(),
                      "slot": torch.empty(batch, dtype=slot_all.dtype,
                                          device=slot_all.device)}
                sm["groups"][key] = st
            else:
                st["slot_all"].copy_(slot_all.view(batch, alloc_len))
                st["table"].copy_(table)
        st["slot"].copy_(st["slot_all"][:, decode_step])
        return st["slot"], st["table"]

    def _common(slot_mapping, block_table):
        return compat.build_common_attention_metadata(
            positions=positions,
            query_start_loc=sm["qsl_gpu"],
            query_start_loc_cpu=sm["qsl_cpu"],
            seq_lens=sm["seq_gpu"],
            num_reqs=batch,
            num_actual_tokens=batch,
            max_query_len=1,
            max_seq_len=kv_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            _seq_lens_cpu=sm["seq_cpu"],
            seq_lens_cpu_upper_bound=sm["seq_cpu"],
            _num_computed_tokens_cpu=sm["ncomp_cpu"],
        )

    out = compat.build_attn_metadata_per_group(
        worker.model_runner,
        layout_for_block_size=_layout_for_block_size,
        common_metadata_for_layout=_common,
    )
    sm["lease_dirty"] = False
    return out


def _capture_step_graph(model, ids_step, positions_step, device, tp_group,
                        dist, replay_done, batch, alloc_len):
    """Capture the decode-step forward into a private CUDA graph on TP-safe
    rails. Returns (graph, hidden) or (None, None) on failure.

    THE TP FIX: model() contains TP all-reduces; capturing them raw corrupts
    the NCCL communicator (async illegal-memory-access on workers — TP=1 has
    no collectives, hence it worked there). vLLM's ``graph_capture`` context
    switches the TP/PP communicators to their graph-capturable path and runs
    on a private stream; capture MUST happen inside it. Barriers keep ranks
    in lockstep so a warmup collective on one rank never pairs with a capture
    collective on another.

    Discipline: the forward is a REPLAY even on the capture step (the
    capture-pass execution itself proved bit-divergent); warmup + capture +
    replay all write the same bytes (same inputs, same KV slot).
    """
    from vllm.distributed.parallel_state import graph_capture as _vllm_gc
    tp = tp_group.world_size > 1
    try:
        if tp:
            dist.barrier(group=tp_group.cpu_group)
        with _vllm_gc(device) as _gc:
            stream = _gc.stream
            with torch.cuda.stream(stream):
                # Second warmup on the capture stream: finishes DeepGEMM lazy
                # JIT and primes plan buffers before recording.
                model(input_ids=ids_step, positions=positions_step,
                      intermediate_tensors=None, inputs_embeds=None)
                torch.cuda.synchronize()
                if tp:
                    dist.barrier(group=tp_group.cpu_group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    g_hidden = model(input_ids=ids_step,
                                     positions=positions_step,
                                     intermediate_tensors=None,
                                     inputs_embeds=None)
        if isinstance(g_hidden, tuple):
            g_hidden = g_hidden[0]
        torch.cuda.synchronize()
        if tp:
            dist.barrier(group=tp_group.cpu_group)
        graph.replay()
        replay_done.record()
        logger.info("PoC decode: step graph captured (batch %d, alloc %d, "
                    "tp=%d)", batch, alloc_len, tp_group.world_size)
        return graph, g_hidden
    except Exception:
        logger.exception("PoC decode: step-graph capture failed; per-step "
                         "compiled fallback")
        return None, None




@torch.inference_mode()
def execute_poc_decode(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    max_tokens: int,
    hidden_size: int,
    route_window: int,
    enforced_k_steps: Optional[Dict[int, List[int]]] = None,
    debug: bool = False,
    va_steps: int = 0,
    per_nonce_reflection: bool = False,
    borrowed_block_ids: Optional[List[int]] = None,
    borrowed_stripe: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run one decode-PoC chunk on a V1 worker. Returns artifacts on the last
    PP rank of the TP driver; None elsewhere.

    Validation mode = ``enforced_k_steps`` given (teacher forcing, decision:
    the enforced k seeds the next step; every step an independent check).
    """
    import torch.distributed as dist
    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.distributed.communication_op import broadcast_tensor_dict
    from vllm.forward_context import set_forward_context
    from gonka_poc._compat import current as _compat_current
    from .poc_model_runner import _create_v1_attn_metadata

    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch = len(nonces)
    alloc_len = seq_len + max_tokens

    tp_group = get_tp_group()
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if tp_group.rank_in_group == 0:
            broadcast_tensor_dict({
                "poc_decode_go": True, "seq_len": seq_len,
                "max_tokens": max_tokens, "hidden_size": hidden_size,
                "nonces": nonces, "route_window": route_window,
                "debug": debug, "va_steps": va_steps,
                "per_nonce_reflection": per_nonce_reflection,
                "enforced": enforced_k_steps or {},
                "has_borrowed": borrowed_block_ids is not None,
                "borrowed_block_ids": borrowed_block_ids or [],
                "borrowed_stripe": int(borrowed_stripe or 0),
            }, src=0)
        else:
            bd = broadcast_tensor_dict(src=0)
            seq_len = int(bd["seq_len"]); max_tokens = int(bd["max_tokens"])
            hidden_size = int(bd["hidden_size"]); nonces = list(bd["nonces"])
            route_window = int(bd["route_window"]); debug = bool(bd["debug"])
            va_steps = int(bd["va_steps"])
            per_nonce_reflection = bool(bd["per_nonce_reflection"])
            enforced_k_steps = {int(k): v for k, v in bd["enforced"].items()} or None
            if bd.get("has_borrowed"):
                borrowed_block_ids = list(bd["borrowed_block_ids"])
                borrowed_stripe = int(bd["borrowed_stripe"])
            else:
                borrowed_block_ids = None; borrowed_stripe = None
            batch = len(nonces); alloc_len = seq_len + max_tokens

    pp_group = get_pp_group()
    if pp_group.world_size > 1:
        raise RuntimeError("decode-PoC: PP > 1 not supported in this revision")

    # --- TP consensus gate: validate inputs on EVERY rank and
    # agree via an all-reduce BEFORE the collective forward phase. A rank that
    # rejects (bad shapes / empty batch / lease too small) must not bail while
    # the others enter NCCL and hang forever. If any rank is unhealthy, all
    # raise together — collective_rpc then surfaces one clean error.
    local_ok = 1
    reason = ""
    if batch <= 0:
        local_ok, reason = 0, "empty batch"
    elif borrowed_block_ids is not None and len(borrowed_block_ids) < batch:
        local_ok, reason = 0, "lease smaller than batch"
    if tp_group.world_size > 1:
        flag = torch.tensor([local_ok], dtype=torch.int32, device=device)
        dist.all_reduce(flag, group=tp_group.device_group)
        if int(flag.item()) != tp_group.world_size:
            raise RuntimeError(
                f"decode-PoC: TP consensus gate failed (this rank: "
                f"{reason or 'ok'}); aborting on all ranks to avoid a "
                f"half-entered collective")
    elif not local_ok:
        raise RuntimeError(f"decode-PoC: input rejected: {reason}")

    # --- model wrappers: pre-compile (registry class) or late attach -------
    # The registry-wrapped model carries state from construction (wrappers are
    # INSIDE the compiled graph — 0.20 bit path). Late attach remains as a
    # fallback for unregistered architectures; its wrappers are only seen by
    # the eager path.
    state = getattr(model, "_poc_native_state", None)
    if state is None:
        max_rows = POC_DECODE_PREFILL_CHUNK * seq_len
        state = attach_native_poc(model, hidden_size, max_rows, device, dtype,
                                  route_window)
    elif int(route_window) != int(getattr(state, "route_window", route_window)):
        # The window is frozen into the compiled graph at first compilation
        # (0.20 semantics: an engine arg, not a request knob) — a mismatched
        # request must fail loud, silently serving the frozen window would
        # produce consensus-invalid artifacts.
        raise ValueError(
            f"route_window={route_window} requested but the process was "
            f"compiled with {state.route_window}; set POC_ROUTE_WINDOW and "
            f"restart")
    if per_nonce_reflection:
        raise NotImplementedError(
            "per_nonce_reflection: per-row reflection buffers deferred "
            "(off in the consensus configuration)")

    # --- chain state (address-stable, capture-ready) ------------------------
    codebook = get_sphere_codebook().to(device=device)
    base_seeds = decode_base_seeds(block_hash, public_key, nonces, device)
    prev_k = torch.zeros(batch, dtype=torch.int64, device=device)
    k_steps = torch.full((batch, max_tokens + 1), -1, dtype=torch.int64,
                         device=device)
    margin_steps = torch.zeros(batch, max_tokens + 1, device=device)
    n_nan = torch.zeros(batch, dtype=torch.int64, device=device)
    mismatch = torch.zeros(batch, dtype=torch.int64, device=device)
    validating = enforced_k_steps is not None
    if validating:
        ref_rows = [enforced_k_steps[n] for n in nonces]
        L = min(min(len(r) for r in ref_rows), max_tokens + 1)
        reference = torch.tensor([r[:L] for r in ref_rows], dtype=torch.int64,
                                 device=device)
    q_kept: Dict[int, torch.Tensor] = {}

    def snap_rows(last_hidden: torch.Tensor, sph_idx: torch.Tensor, step: int):
        """hidden [B,H] + pick idx [B,SPHERE_DIM] -> обновить цепочку шага."""
        nonlocal prev_k
        q = project_to_sphere(torch.gather(last_hidden.float(), 1, sph_idx))
        k, bad, margin = snap_with_margin(q, codebook)
        k_steps[:, step] = k
        margin_steps[:, step] = margin
        n_nan.add_(bad.to(torch.int64))
        if validating and step < reference.shape[1]:
            ref = reference[:, step]
            mismatch.add_(count_mismatch(k, ref, margin, _MARGIN_TAU))
            prev_k = next_prev_k(k, ref)
        else:
            prev_k = next_prev_k(k, None)
        if keep_q_step(step, debug, va_steps > 0, va_steps):
            q_kept[step] = q.detach().to(torch.float16).cpu()

    # ------------------------------------------------------------ prefill --
    # Split prefill / joint decode: the MoE workspace bound applies to the
    # PREFILL token count only (sub-chunks of <= POC_DECODE_PREFILL_CHUNK
    # nonces x seq_len rows); the decode phase then runs the WHOLE batch one
    # step at a time (a few hundred rows), which amortizes the full-scatter
    # expert-weight traffic exactly like the 0.20 engine did with its single
    # round-wide decode batch. Each sub-chunk writes the same blocks the
    # joint phase reads: lease slice for borrowed layouts, row_offset for
    # the in-place fallback.
    if not getattr(state, "has_embed_patch", False):
        raise RuntimeError(
            "decode-PoC: embed_tokens patch missing — the compiled engine "
            "path needs in-model embedding swap (eager is not a mode)")

    t_pf0 = time.perf_counter()
    last_pf = torch.empty(batch, hidden_size, device=device, dtype=dtype)
    stripe = int(borrowed_stripe or 0)
    for off in range(0, batch, POC_DECODE_PREFILL_CHUNK):
        sub_nonces = nonces[off:off + POC_DECODE_PREFILL_CHUNK]
        b = len(sub_nonces)
        positions_pf = torch.arange(seq_len, dtype=torch.int64,
                                    device=device).repeat(b)
        sub_lease = (borrowed_block_ids[off * stripe:(off + b) * stripe]
                     if borrowed_block_ids is not None else None)
        attn_pf, slots_pf = _create_v1_attn_metadata(
            b, seq_len, device, worker, positions_pf,
            borrowed_block_ids=sub_lease,
            borrowed_stripe=borrowed_stripe,
            alloc_len=alloc_len, row_offset=off)
        embeds_pf = generate_inputs(block_hash, public_key, sub_nonces,
                                    dim=hidden_size, seq_len=seq_len,
                                    device=device, dtype=dtype)
        # ENGINE signature: input_ids tensor + inputs_embeds None — the exact
        # shape @support_torch_compile was compiled under. Synthetic embeds
        # are staged and swapped in by the embed patch INSIDE the graph.
        ids_pf = torch.zeros(b * seq_len, dtype=torch.int64, device=device)
        state.set_rows(block_hash, b * seq_len)
        state.set_routing(block_hash, sub_nonces, seq_len, 0)
        state.set_embeds(embeds_pf.view(-1, hidden_size))
        with set_forward_context(attn_pf, vllm_config,
                                 num_tokens=b * seq_len,
                                 slot_mapping=slots_pf,
                                 skip_compiled=_SKIP_COMPILED):
            hidden = model(input_ids=ids_pf, positions=positions_pf,
                           intermediate_tensors=None, inputs_embeds=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_pf[off:off + b] = hidden.view(b, seq_len, -1)[:, -1, :]
    sph0 = random_pick_indices(block_hash, public_key, nonces,
                               hidden_size, SPHERE_DIM, device)
    snap_rows(last_pf, sph0, 0)
    torch.cuda.synchronize()
    t_pf = time.perf_counter() - t_pf0

    # ------------------------------------------------------------- decode --
    # Graph bundle: captured once per (batch, alloc_len) per process on the
    # first chunk's step 1 (after a warmup call; same inputs + same KV slot
    # -> idempotent), then replayed for every remaining step of every chunk.
    # All tensors the captured kernels read live at stable addresses:
    # positions/ids/steps (cached with the graph), state buffers (embeds,
    # mask, route), the builder's persistent cudagraph plan buffers, and the
    # per-group slot clones refreshed with copy_ each step (borrowed layouts
    # are not arithmetic in the step index, so copy_, never add_).
    capture_wanted = _CAPTURE and not _PROFILE
    cache_key = (device.index, batch, alloc_len)
    bundle = _GRAPH_CACHE.get(cache_key) if capture_wanted else None
    if bundle is not None:
        positions_step = bundle["positions"]
        steps_t = bundle["steps"]
        ids_step = bundle["ids"]
        step_meta = bundle["step_meta"]
    else:
        positions_step = torch.empty(batch, dtype=torch.int64, device=device)
        steps_t = torch.empty(batch, dtype=torch.int64, device=device)
        ids_step = torch.zeros(batch, dtype=torch.int64, device=device)
        step_meta = {
            "qsl_gpu": torch.arange(batch + 1, dtype=torch.int32,
                                    device=device),
            "qsl_cpu": torch.arange(batch + 1, dtype=torch.int32,
                                    device="cpu"),
            "seq_gpu": torch.empty(batch, dtype=torch.int32, device=device),
            "seq_cpu": torch.empty(batch, dtype=torch.int32, device="cpu"),
            "ncomp_cpu": torch.empty(batch, dtype=torch.int32, device="cpu"),
            "groups": {},
            "lease_dirty": True,
        }
    step_meta["lease_dirty"] = True   # lease changes between RPCs
    graph = bundle["graph"] if bundle else None
    g_hidden = bundle["hidden"] if bundle else None
    prof = [0.0, 0.0, 0.0, 0.0] if _PROFILE else None  # meta/embeds/fwd/snap
    # Replay fence: builder.build() copies the step's FlashInfer plan into
    # the SAME persistent device buffers every step. With graph replays the
    # CPU races ahead, and step t+1's plan copy can land before replay t
    # finished reading step t's plan — attention then sees one token too
    # many (a not-yet-written slot): flaky snap flips and occasional
    # illegal-memory-access. Fence the loop so the CPU never runs more than
    # one step ahead of the GPU. Costs ~1ms/step against a ~60ms step.
    replay_done = torch.cuda.Event()
    t_steps0 = time.perf_counter()
    for t in range(1, max_tokens + 1):
        if prof is not None:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        if graph is not None:
            replay_done.synchronize()
        steps_t.fill_(t)
        positions_step.fill_(seq_len + t - 1)
        embeds_t = generate_decode_inputs_gpu(base_seeds, prev_k, steps_t,
                                              hidden_size, device).view(
                                                  batch, hidden_size).to(dtype)
        if prof is not None:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            prof[1] += _t1 - _t0; _t0 = _t1
        attn_t, slots_t = _build_step_meta(
            worker, step_meta, batch, seq_len + t, positions_step,
            alloc_len, borrowed_block_ids, borrowed_stripe,
            decode_step=seq_len + t - 1)
        if prof is not None:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            prof[0] += _t1 - _t0; _t0 = _t1
        state.set_rows(block_hash, batch)
        state.set_routing(block_hash, nonces, 1, t)
        state.set_embeds(embeds_t)
        if graph is not None:
            graph.replay()
            replay_done.record()
            h = g_hidden
        else:
            with set_forward_context(attn_t, vllm_config, num_tokens=batch,
                                     slot_mapping=slots_t,
                                     skip_compiled=_SKIP_COMPILED):
                h = model(input_ids=ids_step, positions=positions_step,
                          intermediate_tensors=None, inputs_embeds=None)
                if capture_wanted and t == 1:
                    graph, g_hidden = _capture_step_graph(
                        model, ids_step, positions_step, device, tp_group,
                        dist, replay_done, batch, alloc_len)
                    if graph is not None:
                        h = g_hidden
                        _GRAPH_CACHE[cache_key] = {
                            "graph": graph, "hidden": g_hidden,
                            "positions": positions_step, "steps": steps_t,
                            "ids": ids_step, "step_meta": step_meta,
                        }
        if isinstance(h, tuple):
            h = h[0]
        if prof is not None:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            prof[2] += _t1 - _t0; _t0 = _t1
        sph_t = random_pick_indices_gpu(base_seeds, prev_k, steps_t,
                                        hidden_size, SPHERE_DIM, device)
        snap_rows(h.view(batch, -1), sph_t, t)
        if prof is not None:
            torch.cuda.synchronize()
            prof[3] += time.perf_counter() - _t0
    if prof is not None:
        tot = sum(prof) or 1.0
        logger.info(
            "PoC decode profile (%d steps, batch %d): metadata %.1fms/step "
            "(%.0f%%), embeds %.1fms (%.0f%%), forward %.1fms (%.0f%%), "
            "snap %.1fms (%.0f%%)", max_tokens, batch,
            1e3*prof[0]/max_tokens, 100*prof[0]/tot,
            1e3*prof[1]/max_tokens, 100*prof[1]/tot,
            1e3*prof[2]/max_tokens, 100*prof[2]/tot,
            1e3*prof[3]/max_tokens, 100*prof[3]/tot)

    torch.cuda.synchronize()
    t_steps = time.perf_counter() - t_steps0
    logger.info("PoC decode chunk: %d nonces, prefill %.2fs, %d steps %.2fs "
                "(%.1f ms/step)%s", batch, t_pf, max_tokens, t_steps,
                1e3 * t_steps / max(1, max_tokens),
                "" if _GRAPH_CACHE else " [no graph]")

    state.clear()

    if tp_group.rank_in_group != 0:
        return None

    # -------------------------------------------------------------- emit --
    from .data import encode_vector
    k_host = k_steps.cpu().tolist()
    nan_host = n_nan.cpu().tolist()
    mism_host = mismatch.cpu().tolist()
    artifacts = []
    for i, nonce in enumerate(nonces):
        art = {
            "nonce": nonce,
            "vector_b64": "",
            "k_points_steps": k_host[i],
            "n_sphere_mismatches": mism_host[i] if validating else -1,
            "n_nan_steps": nan_host[i],
        }
        if q_kept:
            art["sph_values_steps"] = [
                encode_vector(q_kept[s][i].numpy()) if s in q_kept else ""
                for s in range(max_tokens + 1)
            ]
        artifacts.append(art)
    return {"artifacts": artifacts,
            "steps_total": batch * (max_tokens + 1),
            "mismatch_total": int(sum(mism_host)) if validating else -1}
