"""Native PoC transform.

The per-layer Householder reflection applied as INLINE layer code, so vLLM's
native torch.compile + cudagraph capture it like any model op (prefill AND decode,
dynamic KV, any backend) — replacing the un-capturable Python forward-hook and the
hand-rolled CUDA-graph capture.

Each decoder layer is wrapped by ``PoCLayerWrapper``, which runs the original layer
then reflects the output (hidden AND residual) on PoC rows only, selected by a
shared boolean mask buffer. Chat rows pass through unchanged (mask False →
``where`` is identity), so one compiled model serves chat and PoC. The reflection
vectors (one per layer, seeded by block_hash) and the mask live in stable buffers
updated in-place each round/step, so replay reads the live values.
"""
import os

import torch

from torch import nn

import logging

import numpy as np

logger = logging.getLogger(__name__)

from gonka_poc.poc.gpu_random import (generate_householder_vector,
                                      _seed_from_string)
from gonka_poc.poc.decode_random import (decode_pseudo_token_ids,
                                         expert_logits_from_base, route_base_seed,
                                         pinned_to_device)

# Debug-only TP guard (VLLM_POC_DEBUG_TP=1): PoC reflection vectors / embeds are
# generated per rank from deterministic seeds and MUST be bit-identical across
# tensor-parallel ranks, else rows reflect/inject differently per rank -> corruption.
_DEBUG_TP = os.environ.get("VLLM_POC_DEBUG_TP") == "1"


# Block hashes kept in the router-seed memo. One scope holds a round's
# nonces; a new round makes the old scope unreachable, so a handful is
# enough to cover interleaved rounds while bounding dead entries.
_SEED_CACHE_MAX_SCOPES = 8


def _assert_replicated_across_tp(t: torch.Tensor, name: str) -> None:
    """No-op unless VLLM_POC_DEBUG_TP=1 and TP world size > 1. Fingerprints `t`
    (3 moments) and all-gathers across the TP group, asserting bit-equality so a
    per-rank RNG divergence is caught the moment a TP run hits it."""
    if not _DEBUG_TP:
        return
    try:
        import torch.distributed as dist
        from vllm.distributed import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_world_size,
        )
    except ImportError:
        return
    if not dist.is_initialized():
        return
    ws = get_tensor_model_parallel_world_size()
    if ws <= 1:
        return
    x = t.detach().to(torch.float64).reshape(-1)
    pos = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float64)
    fp = torch.stack([x.sum(), (x * x).sum(), (x * pos).sum()])
    gathered = [torch.empty_like(fp) for _ in range(ws)]
    dist.all_gather(gathered, fp, group=get_tensor_model_parallel_group().device_group)
    for r in range(1, ws):
        if not torch.equal(gathered[0], gathered[r]):
            raise AssertionError(
                f"PoC '{name}' diverged across TP ranks (rank0 vs rank{r}) — "
                "per-rank RNG non-determinism; PoC is not TP-safe in this setup")


def _reflect_torch(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Householder (reference): rows where mask is True -> x - 2*(x·v)*v; else x.
    Per-row independent, static-shape (no data-dependent control flow) -> the
    compiled graph captures it; cudagraph replays it reading live v/mask."""
    dot = (x * v).sum(-1, keepdim=True)
    transformed = x - 2.0 * dot * v
    return torch.where(mask, transformed, x)


def _reflect(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reflection on PoC rows — the reference path, and the only path: the
    reflection is consensus math, and an explicit torch expression behaves the
    same on every node (a fused Triton variant was removed on 2026-09-05: it
    sped PoC up ~12% on Hopper, did nothing for inference and rounded
    differently per architecture). v: [n, *pad, hidden], mask: [n, *pad, 1]."""
    return _reflect_torch(x, v, mask)


def _install_poc_patch(module: nn.Module, wrapper: nn.Module) -> None:
    """Class-level forward patch. torch.compile traces the CLASS forward, so
    instance ``obj.forward = ...`` assignments are ignored inside compiled
    regions (decode snaps read zeros exactly this way). The class forward
    dispatches to the instance's wrapper when present; untouched instances of
    the same class keep original behaviour. The wrapper calls the ORIGINAL
    class forward, captured once per class."""
    import functools

    cls = type(module)
    if "_poc_orig_forward" not in cls.__dict__:
        orig = cls.forward
        cls._poc_orig_forward = orig

        def _poc_forward(self, *args, **kwargs):
            w = getattr(self, "_poc_wrap", None)
            if w is not None:
                return w.forward(*args, **kwargs)
            return cls._poc_orig_forward(self, *args, **kwargs)

        cls.forward = _poc_forward
    wrapper._inner_call = functools.partial(
        cls.__dict__["_poc_orig_forward"], module)
    # object.__setattr__: keep the wrapper out of nn.Module submodule
    # registration (wrapper.inner already points back at module — a cycle).
    object.__setattr__(module, "_poc_wrap", wrapper)


class PoCLayerWrapper(nn.Module):
    """Wraps one decoder layer; reflects its output hidden + residual on PoC rows.
    ``v`` is this layer's reflection vector; ``mask`` is the shared per-row PoC mask
    (both stable buffers, updated in place)."""

    def __init__(self, inner: nn.Module, table: torch.Tensor,
                 row_group: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.inner = inner
        # table: this layer's [groups, hidden] reflection vectors (group 0 is the
        # zero vector); row_group: shared [max_tokens] int64 group id per row.
        # A row's vector is gathered from the table, so the state holds one copy
        # per (block_hash[, nonce]) instead of one per token per layer.
        self.register_buffer("poc_table", table, persistent=False)
        self.register_buffer("poc_row_group", row_group, persistent=False)
        self.register_buffer("poc_mask", mask, persistent=False)
        self._inner_call = inner.forward

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden states are [rows, hidden] on most models, but a model that keeps
        # several per-row copies carries them in between (DeepSeek V4 repeats the
        # row hc_mult times). Broadcast the per-row vector and mask over whatever
        # dimensions sit there; the reflection always runs along the hidden dim.
        n = x.shape[0]
        pad = (1,) * (x.dim() - 2)
        v = self.poc_table.index_select(0, self.poc_row_group[:n])
        return _reflect(x, v.view(n, *pad, -1).to(x.dtype),
                        self.poc_mask[:n].view(n, *pad, 1))

    def forward(self, *args, **kwargs):
        out = self._inner_call(*args, **kwargs)
        if not isinstance(out, tuple):
            return self._apply(out)
        rest = list(out[1:])
        if rest and rest[0] is not None:  # residual
            rest[0] = self._apply(rest[0])
        return (self._apply(out[0]), *rest)


class PoCEmbeddingWrapper(nn.Module):
    """Wraps the token embedding; for PoC rows, replaces the token embeds with the
    deterministic PoC embeds (from a stable buffer). PoC requests carry dummy token
    IDs so the graphed input_ids path runs; this injects the real PoC embeds INSIDE
    the graph. Chat rows keep their token embeds (mask False)."""

    def __init__(self, inner: nn.Module, embeds: torch.Tensor, mask: torch.Tensor,
                 embed_base: torch.Tensor = None, embed_prev_k: torch.Tensor = None,
                 embed_step: torch.Tensor = None, hidden_size: int = 0,
                 poc_token_ids: torch.Tensor = None):
        super().__init__()
        self.inner = inner
        self.hidden_size = hidden_size
        self.register_buffer("poc_embeds", embeds, persistent=False)
        self.register_buffer("poc_mask", mask, persistent=False)
        self.register_buffer("poc_token_ids", poc_token_ids, persistent=False)
        # SYNTH = EMBEDDING (in-graph): synth the decode input from the chain buffers.
        self._synth = embed_base is not None
        if self._synth:
            self.register_buffer("embed_base", embed_base, persistent=False)
            self.register_buffer("embed_prev_k", embed_prev_k, persistent=False)
            self.register_buffer("embed_step", embed_step, persistent=False)
        self._inner_call = inner.forward

    def forward(self, input_ids):
        # PoC rows carry a dummy token id whose embedding is overridden below, but
        # under async scheduling that id can be a stale/sentinel value (e.g. -1 from
        # the previous step's sampled-token plumbing) -> out-of-vocab gather crash.
        # Force masked (PoC) rows to a valid in-vocab id (0); their value is unused.
        n = input_ids.shape[0]
        m_rows = self.poc_mask[:n]
        # PoC rows: poc_token_ids (seeded id on hash-MoE, else zeros — the id
        # reaches the MoE router via tid2eid, see _TOKEN_ID_ROUTED_MODELS).
        input_ids = torch.where(m_rows, self.poc_token_ids[:n].to(input_ids.dtype),
                                input_ids)
        out = self._inner_call(input_ids)
        m = m_rows.unsqueeze(-1)
        if not self._synth:
            return torch.where(m, self.poc_embeds[:n].to(out.dtype), out)
        # DECODE rows: synth input[step] IN-GRAPH from (base, prev_k, step) — the SAME
        # derivation as gpu_random.generate_decode_inputs_gpu, so byte-identical, but
        # it rides the captured forward (no eager RNG on the host between steps).
        # prev_k<0 rows (prefill) fall back to the pre-filled embed.
        from gonka_poc.poc.decode_random import (
            _step_seeds, _batched_normal_t, _SALT_DECODE_EMBED)
        seeds = _step_seeds(self.embed_base[:n], self.embed_step[:n],
                            self.embed_prev_k[:n], _SALT_DECODE_EMBED)
        dec = _batched_normal_t(seeds, self.hidden_size, out.device).to(out.dtype)
        is_dec = (self.embed_prev_k[:n] >= 0).unsqueeze(-1)
        poc_e = torch.where(is_dec, dec, self.poc_embeds[:n].to(out.dtype))
        return torch.where(m, poc_e, out)


class PoCSnapWrapper(nn.Module):
    """Wraps the model's FINAL norm. Runs it, then SNAPS the normed last hidden ->
    sphere_k IN-GRAPH for every row (PoC's 'sampler'), reusing the embed_* seed
    buffers + codebook and writing per-row k/bad/margin/q to the state's snap_*
    buffers. Returns the norm output unchanged so the LM head still runs. The runner
    index_selects the decode rows post-forward — no per-step index_copy_ feed, no
    separate tail graph replay (this in-graph snap replaced the old eager tail)."""

    def __init__(self, inner: nn.Module, state):
        super().__init__()
        self.inner = inner
        self._st = state
        self._inner_call = inner.forward

    def forward(self, *args, **kwargs):
        out = self._inner_call(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        st = self._st
        n = h.shape[0]
        from gonka_poc.poc.decode_random import random_pick_indices_gpu
        from gonka_poc.poc.sphere import project_to_sphere, snap_with_margin
        lh = h.float()
        lh = lh / (lh.norm(dim=-1, keepdim=True) + 1e-8)
        sph = random_pick_indices_gpu(
            st.embed_base[:n], st.embed_prev_k[:n], st.embed_step[:n],
            st.hidden_size, st.sphere_dim, h.device)
        q = project_to_sphere(torch.gather(lh, 1, sph))
        k_all, bad_all, margin_all = snap_with_margin(q, st.codebook)
        st.snap_k[:n].copy_(k_all)
        st.snap_bad[:n].copy_(bad_all)
        st.snap_margin[:n].copy_(margin_all)
        st.snap_q[:n].copy_(q)
        return out


def _experts_meta(experts) -> tuple:
    """(num_experts, top_k, n_group, topk_group) across vLLM minors: 0.20
    exposes the first two directly on FusedMoE; 0.25 moved them into
    ``moe_config`` (``experts_per_token``) and keeps the grouped-top-k params
    (DeepSeek family) on the runner's router. Flat routers report
    n_group=topk_group=1 (the grouped formula degenerates away). A
    gate+experts module we cannot read is a HARD error — a silently unseeded
    router re-opens the MoE honest-floor hole."""
    # FusedMoE: global_num_experts; DeepSeek-V4 MegaMoE: num_experts
    # (num_local_experts is the EP shard; the router does not count by it).
    n_global = getattr(experts, "global_num_experts",
                       getattr(experts, "num_experts", None))
    if hasattr(experts, "top_k") and n_global is not None:
        return int(n_global), int(experts.top_k)
    cfg = getattr(experts, "moe_config", None)
    if cfg is not None:
        return int(cfg.num_experts), int(cfg.experts_per_token)
    raise RuntimeError(
        f"PoC seeded routing: cannot read expert meta from {type(experts)}; "
        "vLLM moved the FusedMoE attributes again — extend _experts_meta")


class PoCRouterWrapper(nn.Module):
    """Wraps an MoE gate (router Linear). For PoC rows (mask True) it REPLACES the
    router logits with deterministic, hidden-INDEPENDENT seeded logits, so MoE
    expert selection (and gate weights) no longer read the noise-prone hidden ->
    removes the routing nondeterminism that drives the decode-PoC honest floor.
    Chat rows (mask False) keep their natural logits untouched.

    The seeded logits are computed HERE, INSIDE the forward — i.e. INSIDE the
    captured cudagraph — from this layer's cached seed base ([max_tokens] int64)
    and the shared per-row decode ``step`` buffer. Both are address-stable and
    updated in place by ``set_routing`` (the graph reads live values), so per step
    the eager path only bumps a tiny [B] step scalar; the Fisher-Yates selection
    (pure integer, no topk/scores/ties -> bit-identical eager==graph) rides in the
    graph instead of stalling the decode pipeline as an eager tail. Static shape ->
    cudagraph-safe."""

    def __init__(self, inner: nn.Module, route_base: torch.Tensor,
                 route_step: torch.Tensor, n_experts: int, top_k: int,
                 mask: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.n_experts = n_experts
        self.top_k = top_k
        self.register_buffer("poc_route_base", route_base, persistent=False)  # [max_tokens] int64
        self.register_buffer("poc_route_step", route_step, persistent=False)  # [max_tokens] int64 (shared)
        self.register_buffer("poc_mask", mask, persistent=False)

    def __getattr__(self, name: str):
        # Delegate unknown attributes (e.g. `.weight`, quant scales) to the wrapped
        # gate, so backends that read gate attributes directly (FlashInfer MoE init)
        # still resolve them — FlashAttention doesn't, which is why it worked there.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, *args, **kwargs):
        out = self._inner_call(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        n = logits.shape[0]
        m = self.poc_mask[:n].unsqueeze(-1)
        forced = expert_logits_from_base(                   # in-graph seeded selection
            self.poc_route_base[:n], self.poc_route_step[:n],
            self.n_experts, self.top_k, logits.device).to(logits.dtype)
        logits = torch.where(m, forced, logits)
        return (logits, *out[1:]) if isinstance(out, tuple) else logits


class PoCSelectOverride:
    """Overrides the ROUTER SELECTION OUTPUT for PoC rows: seeded expert ids
    and ladder-softmax weights are written directly over whatever the engine
    selected. The engine's selection math (scoring functions, grouped stages,
    e_score_correction_bias, backend top-k kernels) still runs for chat rows
    but is DISCARDED for PoC rows — selection is skipped, not reproduced, so
    none of its numerics can perturb the seeded choice. In-graph: ids/weights
    derive from the live route buffers via integer topk over the forced
    ladder (distinct values — no ties, backend-independent)."""

    def __init__(self, route_base: torch.Tensor, route_step: torch.Tensor,
                 n_experts: int, top_k: int, mask: torch.Tensor):
        self.poc_route_base = route_base
        self.poc_route_step = route_step
        self.n_experts = n_experts
        self.top_k = top_k
        self.poc_mask = mask
        self._inner_call = None  # bound by _install_poc_select_patch

    def select_experts(self, *args, **kwargs):
        weights, ids = self._inner_call(*args, **kwargs)
        # router_logits for masked rows already carry the forced ladder (the
        # gate wrapper runs first) — derive ids/weights from THEM instead of
        # recomputing the seed math: ladder values are distinct, so topk is
        # deterministic; unmasked rows' results are discarded by the where.
        logits = kwargs.get("router_logits",
                            args[1] if len(args) > 1 else None)
        n = ids.shape[0]
        ladder, seed_ids = torch.topk(logits.float(), self.top_k)
        seed_w = torch.softmax(ladder, dim=-1)
        m = self.poc_mask[:n].unsqueeze(-1)
        ids = torch.where(m, seed_ids.to(ids.dtype), ids)
        weights = torch.where(m, seed_w.to(weights.dtype), weights)
        return weights, ids


def _install_poc_select_patch(router, override: "PoCSelectOverride") -> None:
    """Class-level patch of ``select_experts`` (same rationale as
    _install_poc_patch: compiled regions trace the class method)."""
    import functools

    cls = type(router)
    if "_poc_orig_select" not in cls.__dict__:
        orig = cls.select_experts
        cls._poc_orig_select = orig

        def _poc_select(self, *args, **kwargs):
            w = getattr(self, "_poc_sel", None)
            if w is not None:
                return w.select_experts(*args, **kwargs)
            return cls._poc_orig_select(self, *args, **kwargs)

        cls.select_experts = _poc_select
    override._inner_call = functools.partial(
        cls.__dict__["_poc_orig_select"], router)
    router._poc_sel = override


class PoCNativeState:
    """Per-model PoC transform state: PER-ROW reflection vectors per wrapped layer,
    a shared row mask, and a PoC-embeds buffer. Held on the runner; updated each
    round (block_hash) / step (mask, embeds) in place so the captured graph reads
    live values.

    The reflection vectors are per-row ([max_tokens, hidden]) so requests with
    DIFFERENT block_hashes can share one forward batch without stepping on each
    other (each row reflects with its own block's vectors).
    """

    def __init__(self, num_layers: int, hidden_size: int, max_tokens: int,
                 device, dtype, max_groups: int | None = None):
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        # ONE [layers, groups, hidden] table; self.table keeps the per-layer
        # entries as VIEWS into it. A group is one (block_hash[, nonce]) draw,
        # group 0 is the zero vector; rows point at their group through the
        # shared row_group index and the layer wrapper gathers. Per-row storage
        # ([layers, max_tokens, hidden]) cost 12.5 GB on MiniMax-M2.7 at
        # max_num_batched_tokens 32768; the table is a few tens of MB.
        self.max_groups = int(max_groups or max_tokens)
        self.table_t, self.table = self.alloc_table(
            num_layers, self.max_groups + 1, hidden_size, device, dtype)
        self.row_group = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.embeds = torch.zeros(max_tokens, hidden_size, device=device, dtype=dtype)
        # (block_hash, nonce-or-None) -> per-layer vectors. nonce=None is the
        # per-block scheme (one draw shared by every nonce of the block); an int
        # nonce is the per-nonce scheme (each nonce gets its own draw).
        self._seed_cache: dict = {}     # block_hash -> {nonce: per-layer bases}
        self._hash_cache: dict[tuple, list] = {}
        self._stack_cache: dict = {}   # (hash, nonce) -> [layers, hidden] stack
        self._last_refl_key: tuple | None = None    # skip redundant per-step rescatter
        # seeded-routing (MANDATORY for MoE; filled by attach_native_poc): per-MoE-layer
        # cached seed base [max_tokens] int64 (block_hash,nonce,layer, hashed once per
        # mapping) + a SHARED per-row decode-step buffer. The forced-logit selection
        # itself runs in-graph inside PoCRouterWrapper.forward; per step set_routing
        # only writes the step buffer (the graph reads base+step live). Static shape.
        self.route_step = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.router_meta: list = []                 # [(n_experts, top_k), ...]
        self._route_base: list = []                 # per-layer [max_tokens] int64 sha256 base (cached)
        self._base_key: tuple | None = None         # (hashes,nonces) the base was built for
        self._last_route_key: tuple | None = None   # skip refresh if (hashes,nonces,steps) unchanged
        # PoC-as-a-sampler, part 1: SYNTH = EMBEDDING. The next decode input is the
        # "embedding" of the sampled sphere_k — synthesized IN-GRAPH in the embedding
        # wrapper from these per-row buffers, so it rides vLLM's standard per-step
        # flow with ~zero eager CPU (like chat's token->embed). prev_k<0 = non-decode.
        self.embed_base = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.embed_prev_k = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self.embed_step = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        # Pseudo token ids for hash-MoE; token_id_vocab==0 — the model does not
        # route by id, buffer stays zero (attach_native_poc enables it).
        self.token_id_vocab = 0
        self.poc_token_ids = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        # PoC-as-a-sampler, part 2: SNAP = SAMPLING. A wrapper on the final norm snaps
        # the last hidden -> sphere_k IN-GRAPH (reusing the embed_* seed buffers + the
        # codebook), writing per-row k/bad/margin/q here. The runner index_selects the
        # decode rows post-forward — no separate tail-graph feed (4 index_copy_/step).
        from gonka_poc.poc.sphere import SPHERE_DIM, get_sphere_codebook
        self.sphere_dim = SPHERE_DIM
        self.snap_k = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.snap_bad = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.snap_margin = torch.zeros(max_tokens, dtype=torch.float32, device=device)
        self.snap_q = torch.zeros(max_tokens, SPHERE_DIM, dtype=torch.float32, device=device)
        self.codebook = get_sphere_codebook().to(device=device).float().contiguous()

    def set_embeds(self, row_embeds: torch.Tensor) -> None:
        """Write the PoC rows' input embeds into the buffer (in place)."""
        n = row_embeds.shape[0]
        self.embeds[:n].copy_(row_embeds)
        _assert_replicated_across_tp(self.embeds[:n], "embeds")

    def set_decode_chain(self, offs: torch.Tensor = None, base: torch.Tensor = None,
                         prev_k: torch.Tensor = None, step: torch.Tensor = None) -> None:
        """Publish per-row (base, prev_k, step) so the embedding wrapper synths the
        decode input IN-GRAPH (like set_routing does for the router). Cheap [n]
        uploads; rows with prev_k<0 are non-decode (prefill/chat).

        prev_k alone decides whether a row is synthesized, the buffer is persistent
        and the captured graph reads it whole. So clear it on EVERY forward before
        scattering this batch: rows left by an earlier batch, or by cudagraph
        padding, would otherwise be synthesized as decode. Call with no arguments
        when the forward has no decode rows."""
        self.embed_prev_k.fill_(-1)
        if offs is None:
            return
        self.embed_base.index_copy_(0, offs, base)
        self.embed_prev_k.index_copy_(0, offs, prev_k)
        self.embed_step.index_copy_(0, offs, step)
        if self.token_id_vocab:
            self.poc_token_ids.index_copy_(
                0, offs, decode_pseudo_token_ids(
                    base, step, prev_k, self.token_id_vocab))

    def set_prefill_token_ids(self, offs: torch.Tensor, ids: torch.Tensor) -> None:
        """Pseudo token ids for nonce prefill rows (hash-MoE only; the caller
        gates on token_id_vocab)."""
        self.poc_token_ids.index_copy_(0, offs, ids.to(torch.int32))

    # Device-side cache bound: per-nonce seeding adds one entry per (block_hash,
    # nonce), so a 128-nonce round is ~128 entries. Entries are stored in the
    # reflection-buffer dtype (num_layers x hidden, fp16/bf16 -> e.g. ~0.75 MB
    # for 94 x 4096), so the cap bounds the cache at ~200 MB worst case — memory
    # allocated AFTER vLLM's startup profiling, so it must stay small. Clear
    # wholesale past the cap (regeneration is cheap, seeded host murmur).
    _HASH_CACHE_MAX = 256

    @staticmethod
    def alloc_table(num_layers, n_groups, hidden_size, device, dtype):
        """Contiguous reflection table + per-layer views onto it."""
        buf = torch.zeros(num_layers, n_groups, hidden_size,
                          device=device, dtype=dtype)
        return buf, list(buf)

    def _stacked_vectors_for(self, block_hash: str, nonce: int | None = None):
        """[num_layers, hidden] stack of the per-layer vectors, cached alongside
        the list form so the batched scatter never re-stacks."""
        key = (block_hash, nonce)
        st = self._stack_cache.get(key)
        if st is None:
            st = torch.stack(self._vectors_for(block_hash, nonce))
            self._stack_cache[key] = st
        return st

    def _vectors_for(self, block_hash: str, nonce: int | None = None) -> list:
        """Per-layer reflection vectors for (block_hash, nonce) (cached across
        forwards). nonce=None -> the per-block seed (production default); an int
        nonce -> that nonce's own seed, so every nonce reflects with an
        independent draw."""
        key = (block_hash, nonce)
        vs = self._hash_cache.get(key)
        if vs is None:
            if len(self._hash_cache) >= self._HASH_CACHE_MAX:
                self._hash_cache.clear()
                self._stack_cache.clear()
            suffix = ("" if nonce is None else f"_nonce{nonce}")
            # Cache in the buffer dtype: halves the footprint vs the generator's
            # fp32 and matches what the scatter writes (copy_ casts identically).
            dt = self.table[0].dtype
            vs = [
                generate_householder_vector(
                    f"{block_hash}{suffix}_layer_{i}_householder",
                    self.hidden_size, self.device).to(dt)
                for i in range(self.num_layers)
            ]
            self._hash_cache[key] = vs
        return vs

    def set_row_block_hashes(self, row_hashes: list,
                             row_refl_nonces: list | None = None) -> None:
        """Write each row's reflection vectors from ITS OWN block_hash (in place),
        so requests with different block_hashes coexist in one forward. row_hashes[i]
        = block_hash for row i, or None (left zero; masked out). Generation of the
        vectors is cached per (block_hash, nonce); the scatter is cheap CPU-side setup.

        row_refl_nonces[i] (optional) = the row's nonce when its request runs
        per-nonce reflection seeding, else None (per-block seed, the default —
        omitting the argument reproduces the legacy behavior bit-exactly).

        The reflection vectors do not depend on the decode step, so for a stable
        batch the row->seed mapping is unchanged across all decode steps. Skip
        the zero + per-(row,layer) copy_ (num_layers x B kernels) when the mapping
        matches the last call; the buffers already hold the right values."""
        if row_refl_nonces is None:
            row_refl_nonces = [None] * len(row_hashes)
        refl_key = (tuple(row_hashes), tuple(row_refl_nonces))
        if refl_key == self._last_refl_key:
            return
        # Rows sharing (block_hash, refl_nonce) share one table slot; the
        # per-row index is written once for the whole batch. Rows with no
        # block_hash point at slot 0 (the zero vector; masked out anyway).
        groups: dict = {}
        for row, (bh, nz) in enumerate(zip(row_hashes, row_refl_nonces)):
            if bh is None:
                continue
            groups.setdefault((bh, nz), []).append(row)
        if len(groups) > self.max_groups:
            raise RuntimeError(
                f"PoC reflection table: {len(groups)} distinct (block_hash, nonce) "
                f"groups in one batch, table holds {self.max_groups} (max_num_seqs)")
        idx = np.zeros(len(row_hashes), dtype=np.int64)
        for gid, ((bh, nz), rows) in enumerate(groups.items(), start=1):
            # one write covers every layer: [L, hidden] into slot gid
            self.table_t[:, gid, :] = self._stacked_vectors_for(bh, nz)
            idx[rows] = gid
        if len(idx):
            self.row_group[:len(idx)].copy_(
                pinned_to_device(idx.tolist(), torch.int64, self.device))
        self._last_refl_key = refl_key
        _assert_replicated_across_tp(self.table[0], "reflection_table[0]")
        # (reflection vectors depend on block_hash [+ nonce when per-nonce seeded],
        # never the step; routing also depends on step so it is refreshed
        # separately, per step, via set_routing.)

    def set_routing(self, row_hashes, row_nonces, row_steps) -> None:
        """Refresh PER-ROW seeded router logits — MANDATORY for MoE. EFFICIENT (the
        K-calc discipline):
          * the sha256 BASE (block_hash,nonce,layer) is hashed ONCE per mapping and
            cached in [max_tokens] int64 buffers — NOT per step,
          * each step only folds `step` ON GPU (expert_logits_from_base): two integer
            murmur kernels per layer, batched [B, n_experts], copied GPU->GPU into the
            static buffer IN PLACE.
        So per step there is NO host string-hashing, NO device->host sync, and the
        captured graph (which only READS the buffer) needs no recapture. Rows with
        block_hash None get base 0 (masked out anyway)."""
        if not self._route_base:
            return
        base_key = (tuple(row_hashes), tuple(row_nonces))
        if base_key != self._base_key:                       # rebuild cached base (host, ONCE/mapping)
            # The sha256 base depends only on (block_hash, nonce, layer), never
            # on the row, so memoize it: prefill rows are tokens, thousands per
            # chunk over a few dozen nonces.
            cache = self._seed_cache
            # Table per unique (block_hash, nonce); the per-row expansion runs
            # on device via index_select. Column 0 holds rows with no
            # block_hash (the per-row form wrote 0 for those).
            n_rows = len(row_hashes)
            n_layers = len(self._route_base)
            uniq: dict = {}
            row_col = [0] * n_rows
            for _r, (bh, nz) in enumerate(zip(row_hashes, row_nonces)):
                if bh is None:
                    continue
                k = (bh, nz)
                j = uniq.get(k)
                if j is None:
                    j = len(uniq) + 1
                    uniq[k] = j
                row_col[_r] = j
            tab = [[0] * (len(uniq) + 1) for _ in range(n_layers)]
            # Scoped by block_hash, one entry per NONCE holding the per-layer
            # bases. A new round brings a new hash, which makes the previous
            # round's entries unreachable (the key contains the hash) — so bound
            # the memo by SCOPES rather than by a nonce count: garbage is at most
            # a few dead rounds, and no live scope is dropped. Evicting everything
            # absent from the current mapping would be tighter but thrashes, since
            # rows carry their own hash and two rounds can interleave step to step.
            for (bh, nz), j in uniq.items():
                scope = cache.get(bh)
                if scope is None:
                    scope = cache[bh] = {}
                vals = scope.get(nz)
                if vals is None:
                    vals = scope[nz] = tuple(
                        _seed_from_string(route_base_seed(bh, nz, i))
                        for i in range(n_layers))
                for i in range(n_layers):
                    tab[i][j] = vals[i]
            while len(cache) > _SEED_CACHE_MAX_SCOPES:
                cache.pop(next(iter(cache)))          # oldest round, now unreachable
            # One upload of [n_layers, n_unique+1]; the fan-out below is D2D.
            # Keep it blocking: the async variant corrupted the tail rows of
            # the mapping against the in-flight forward (cause not confirmed).
            tab_t = torch.tensor(tab, dtype=torch.int64, device=self.device)
            idx_t = torch.tensor(row_col, dtype=torch.long, device=self.device)
            for i, base_buf in enumerate(self._route_base):
                base_buf[:n_rows] = tab_t[i].index_select(0, idx_t)
            self._base_key = base_key
        key = (base_key, tuple(row_steps))
        if key == self._last_route_key:                      # nothing changed -> skip
            return
        b = len(row_steps)
        # Per step, ONLY publish the decode step into the shared buffer (tiny [B]
        # upload, sync-free; a direct torch.tensor(list, device=cuda) would block the
        # forward — see pinned_to_device). The Fisher-Yates forced-logit selection
        # runs in-graph in PoCRouterWrapper.forward, reading base+step live.
        self.route_step[:b].copy_(pinned_to_device(row_steps, torch.int64, self.device))
        self._last_route_key = key

    def set_mask(self, row_mask: torch.Tensor | None) -> None:
        """Set which rows are PoC this forward (in place). None -> all chat."""
        self.mask.zero_()
        if row_mask is not None:
            n = row_mask.shape[0]
            self.mask[:n].copy_(row_mask)


# Architectures whose MoE gate picks experts by token id (integer table on the
# gate; tid2eid on DeepSeek-V4). Their gates keep natural weights; pseudo token
# ids are enabled for all models. A model outside this list with such a table
# is rejected at attach — extend the list only after verification.
_TOKEN_ID_ROUTED_MODELS = ("deepseek_v4",)


def _token_id_table(gate) -> torch.Tensor | None:
    """Integer 2-D table on the gate (DeepSeek-V4: tid2eid[vocab, k]) marks
    token-id routing. Float gate parameters (weights, e_score_correction_bias)
    do not qualify."""
    for _, t in list(gate.named_parameters(recurse=False)) + list(gate.named_buffers(recurse=False)):
        if t is not None and t.dim() == 2 and t.dtype in (torch.int32, torch.int64):
            return t
    return None


_ABLATE = frozenset(x.strip() for x in os.environ.get("POC_ABLATE", "").lower().split(",") if x.strip())


def _ablated(part: str) -> bool:
    """POC_ABLATE=reflect,router,pseudo — DIAGNOSTIC: disable PoC intervention
    to isolate a defect (e.g. hang above ~200 PoC rows on Hopper). Breaks
    consensus — corpora produced in this mode are UNFIT for verdicts.
      reflect — Householder reflections on layers (no PoCLayerWrapper);
      router  — expert seeding (no PoCRouterWrapper);
      pseudo  — pseudo token ids on hash-MoE (dummy ids as before)."""
    return part in _ABLATE


def attach_native_poc(model: nn.Module, layers: list, embed_owner, max_tokens: int,
                      hidden_size: int, device, dtype,
                      hf_config=None, max_groups: int | None = None) -> PoCNativeState:
    """Wrap each decoder layer (Householder) AND the token embedding (PoC-embed
    injection) BEFORE compilation, sharing one mask. Returns the state to drive
    them. Idempotent: skipped if already wrapped."""
    if getattr(model, "_poc_native_state", None) is not None:
        return model._poc_native_state
    state = PoCNativeState(len(layers), hidden_size, max_tokens, device, dtype,
                           max_groups=max_groups)
    from gonka_poc.poc.decode_random import set_ladder_base_for_model
    _mt = getattr(hf_config, "model_type", None)
    logger.info("PoC seeded-routing ladder base: %d (model_type=%s)",
                set_ladder_base_for_model(_mt), _mt)
    vocab = int(getattr(hf_config, "vocab_size", 0) or 0)
    if vocab and not _ablated("pseudo"):
        state.token_id_vocab = vocab
        logger.info("PoC pseudo token ids ON (vocab=%d)", vocab)
    # Patch forward IN PLACE (never replace the module): 0.25 compiles the model
    # ahead of time and resolves parameters by qualified name, so re-parenting a
    # layer under a wrapper breaks the compiled graph's parameter map.
    if _ABLATE:
        logger.warning("POC_ABLATE=%s: diagnostic mode, artifacts NOT consensus-safe",
                       ",".join(sorted(_ABLATE)))
    if not _ablated("reflect"):
        for i, layer in enumerate(layers):
            _install_poc_patch(
                layer, PoCLayerWrapper(layer, state.table[i], state.row_group, state.mask))
    if embed_owner is not None and hasattr(embed_owner, "embed_tokens"):
        # Patch forward IN PLACE, never replace the module: wrapping renames
        # parameters (embed_tokens.weight -> embed_tokens.inner.weight) and
        # 0.25's ahead-of-time compiled graph looks them up by name.
        _emb = embed_owner.embed_tokens
        _wrap = PoCEmbeddingWrapper(
            _emb, state.embeds, state.mask,
            state.embed_base, state.embed_prev_k, state.embed_step, hidden_size,
            state.poc_token_ids)
        _install_poc_patch(_emb, _wrap)
    # SNAP = SAMPLING: patch the final norm in place, same reason as above.
    if embed_owner is not None and hasattr(embed_owner, "norm"):
        _nrm = embed_owner.norm
        _install_poc_patch(_nrm, PoCSnapWrapper(_nrm, state))
    # Seeded-routing is MANDATORY for MoE — part of the PoC algorithm, not a toggle.
    # Natural MoE top-k reads the noise-prone hidden, so cross-HW/backend drift flips
    # the k-th expert and inflates the honest floor; seeding the experts from
    # (block_hash,nonce,step,layer) removes that. There is NO non-seeded path. Wrap
    # every MoE gate, discovered generically (any submodule with .gate + a FusedMoE
    # .experts) -> no per-model code. Chat rows are masked out (natural router kept).
    skipped_hash = 0
    for wrapper in layers:
        inner_layer = getattr(wrapper, "inner", wrapper)
        moe = next(
            (m for m in inner_layer.modules()
             if hasattr(m, "gate") and hasattr(m, "experts")
             and not isinstance(m.gate, PoCRouterWrapper)),
            None)
        if moe is None:
            continue
        if _token_id_table(moe.gate) is not None:
            if getattr(hf_config, "model_type", None) not in _TOKEN_ID_ROUTED_MODELS:
                raise RuntimeError(
                    f"PoC: MoE gate {type(moe.gate).__name__} routes by token id "
                    f"(integer table on the gate), but model_type "
                    f"{getattr(hf_config, 'model_type', None)!r} is not in "
                    "_TOKEN_ID_ROUTED_MODELS — verify the model, then add it")
            # Hash-MoE (DeepSeek-V4): expert choice is tid2eid[input_ids], already
            # determined by pseudo token ids; logits only give weights. Forcing the
            # ladder would zero the table experts' weights — keep natural.
            skipped_hash += 1
            continue
        n_exp, top_k = _experts_meta(moe.experts)
        # Address-stable per-layer seed base [max_tokens] -> the wrapper folds the
        # shared route_step buffer into it and runs the Fisher-Yates selection
        # in-graph (see PoCRouterWrapper). cudagraph-safe (static shape).
        route_base = torch.zeros(state.max_tokens, dtype=torch.int64, device=device)
        state._route_base.append(route_base)
        state.router_meta.append((n_exp, top_k))
        _gate = moe.gate
        if _ablated("router"):
            continue
        _install_poc_patch(_gate, PoCRouterWrapper(
            _gate, route_base, state.route_step, n_exp, top_k, state.mask))
        # The gate-logit forcing above is the single routing seam, as in
        # 0.20. The selection override is NOT installed: replacing the
        # engine's expert weights with the ladder softmax collapses the
        # honest/fraud gap — AWQ fraud validates at ~75% mismatches instead
        # of ~11-13%, and any numeric input difference (quant, cross-GPU)
        # saturates at the same ceiling. Measured on 1xB300, MiniMax-M2.7
        # vs its AWQ checkpoint; disabling restores the 0.20-level gap.

    model._poc_native_state = state
    if skipped_hash:
        logger.info("PoC hash-MoE gates left natural: %d (selection by seeded "
                    "pseudo token ids, weights from natural logits)", skipped_hash)
    logger.info(
        "PoC native attach: %d layers, embed=%s, snap=%s, %d MoE routers "
        "seeded%s", len(layers), embed_owner is not None,
        embed_owner is not None and hasattr(embed_owner, "norm"),
        len(state.router_meta),
        (" (n_exp=%d top_k=%d)" % state.router_meta[0])
        if state.router_meta else "")
    return state
