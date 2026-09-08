"""Compat shim for vLLM 0.28.x.

Copy of ``v0_25.py``: every private surface it touches exists on the
GLM-5.3-Flash base under the same name and shape. Line numbers in the
docstrings below are from 0.25.1.

Every function below touches a vLLM private surface. Each docstring records:
  * the upstream symbol (with file path + line if stable);
  * the version constraint we're claiming;
  * the contract-test reference that must stay green.

If any of these shift in a future vLLM minor, copy this file to
``v0_29.py``, edit the relevant function, and register the new dispatch
mapping in ``gonka_poc/_compat/__init__.py``.

CommonAttentionMetadata import-path policy
------------------------------------------
v0.25 REVERSED the two paths relative to v0.23:

  * ``vllm.v1.attention.backend``       — canonical declaration site
    (class at vllm/v1/attention/backend.py:395 in v0.25.1).
  * ``vllm.v1.attention.backends.utils`` — convenience re-export
    (utils.py:33 imports from backend).

We import from the canonical ``vllm.v1.attention.backend`` path because that
is the one pinned by
``tests/contract/test_api_surface.py::test_common_attention_metadata_fields``
(shim and contract test must reference the declaration site; otherwise an
upstream re-export removal would break the shim silently while the contract
test stays green).
"""
from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Attention metadata (CommonAttentionMetadata)
# ---------------------------------------------------------------------------- #

def build_common_attention_metadata(
    *,
    query_start_loc: Any,
    query_start_loc_cpu: Any,
    seq_lens: Any,
    num_reqs: int,
    num_actual_tokens: int,
    max_query_len: int,
    max_seq_len: int,
    block_table_tensor: Any,
    slot_mapping: Any,
    causal: bool = True,
    seq_lens_cpu_upper_bound: Optional[Any] = None,
    _seq_lens_cpu: Optional[Any] = None,
    _num_computed_tokens_cpu: Optional[Any] = None,
    positions: Optional[Any] = None,
) -> Any:
    """Construct a ``CommonAttentionMetadata`` for a PoC forward pass.

    Upstream symbol: ``vllm.v1.attention.backend.CommonAttentionMetadata``
        (declaration at vllm/v1/attention/backend.py:395 in v0.25.1;
        re-exported via ``vllm.v1.attention.backends.utils``; private;
        carries ``seq_lens_cpu_upper_bound`` for MLA-style backends and the
        optional ``positions`` field — in v0.25.1 read by the DeepSeek-V4
        C128A sparse-MLA builder and the SWA compressor, ``None``-safe for
        every other backend).

    Version constraint: vllm == 0.28.*

    Contract test:
        tests/contract/test_api_surface.py::test_common_attention_metadata_fields

    The kwarg list mirrors the fork's ``_create_v1_attn_metadata`` call site at
    ``vllm/poc/poc_model_runner.py`` (branch mb/feat/port-pocv2-vllm-0.23.0):

        CommonAttentionMetadata(
            query_start_loc=...,
            query_start_loc_cpu=...,
            seq_lens=...,
            num_reqs=batch_size,
            num_actual_tokens=batch_size * seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=...,
            slot_mapping=...,
            causal=True,
            _seq_lens_cpu=...,
            seq_lens_cpu_upper_bound=...,
            _num_computed_tokens_cpu=torch.zeros(batch_size, ...),
        )

    Caller (poc_model_runner) is responsible for building the GPU/CPU tensors;
    this helper is the version-pinned constructor binding only.
    """
    # Import from the v0.25 canonical declaration site (backend.py) to keep
    # the contract test's pin point authoritative.
    from vllm.v1.attention.backend import CommonAttentionMetadata

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=causal,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=_seq_lens_cpu,
        _num_computed_tokens_cpu=_num_computed_tokens_cpu,
        positions=positions,
    )


# ---------------------------------------------------------------------------- #
# Per-group AttentionMetadata expansion (model_runner.attn_groups iteration)
# ---------------------------------------------------------------------------- #

def build_attn_metadata_per_group(
    model_runner: Any,
    *,
    layout_for_block_size: Any,
    common_metadata_for_layout: Any,
) -> tuple[dict, dict]:
    """Build per-layer ``AttentionMetadata``, one build per attention group.

    Models may register KV cache groups with DIFFERENT block sizes (e.g.
    DeepSeek-V4: sparse MLA / indexer at ``cache_config.block_size``, the SWA
    compressor at 8). Each group's metadata must therefore be built from a
    layout computed for THAT group's block size — sharing one layout hands
    out-of-range slot ids to the other pools (OOB writes: illegal memory
    access on sm_90, silent corruption elsewhere). For single-group models
    the loop reproduces the historical single-layout behaviour exactly.

    Upstream symbols (all version-pinned here, callers stay generic):
        * ``GPUModelRunner.attn_groups`` (list[list[AttentionGroup]]),
          vllm/v1/worker/gpu_model_runner.py:555 (v0.25.1)
        * ``AttentionGroup.get_metadata_builder(index)`` and
          ``AttentionGroup.layer_names`` / ``.kv_cache_spec``
          (vllm/v1/worker/utils.py:223)
        * ``builder.kv_cache_spec`` (vllm/v1/attention/backend.py:620) —
          preferred block-size source because it reflects
          ``kernel_block_size`` splits (``KVCacheSpec.copy_with_new_block_size``,
          applied in ``AttentionGroup.get_metadata_builder``); the group spec
          is the fallback. A literal 0 is treated as present (explicit
          ``is None`` checks), even though no current spec produces it.
        * ``builder.build(common_prefix_len, common_attn_metadata)`` — the v1
          entry point for materialising backend-specific metadata.

    Version constraint: vllm == 0.28.*

    Contract test:
        tests/contract/test_api_surface.py::test_kv_caches_attribute
        (covers the GPUModelRunner declaration site; attn_groups lives in
        the same class so any reshuffle that breaks one breaks the other)

    Args:
        model_runner: live ``GPUModelRunner`` instance from the worker.
        layout_for_block_size: callable ``(block_size:int,
            manager_block_size:int) -> (slot_mapping, block_table)``; the
            caller owns the tensor math and may cache per block size.
            ``block_size`` is the group's KERNEL block size (slot units),
            ``manager_block_size`` the pool-unit size used to expand a
            borrowed block lease.
        common_metadata_for_layout: callable ``(slot_mapping, block_table) ->
            CommonAttentionMetadata`` (normally a partial application of
            :func:`build_common_attention_metadata`).

    Returns:
        ``(attn_metadata_dict, slot_mapping_dict)`` — both keyed by layer
        name; the slot mapping for each layer is its group's. Pass straight
        into ``set_forward_context(attn_metadata=..., slot_mapping=...)``.
    """
    attn_metadata_dict: dict = {}
    slot_mapping_dict: dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            spec = getattr(builder, "kv_cache_spec", None)
            block_size = getattr(spec, "block_size", None)
            if block_size is None:
                block_size = attn_group.kv_cache_spec.block_size
            # kv_cache_spec.block_size = manager (pool-unit) size;
            # block_size above may be its kernel split (r = m/g).
            manager_block_size = attn_group.kv_cache_spec.block_size
            slot_mapping, block_table = layout_for_block_size(
                block_size, manager_block_size)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata_for_layout(
                    slot_mapping, block_table
                ),
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


# ---------------------------------------------------------------------------- #
# KV cache pool access
# ---------------------------------------------------------------------------- #

def get_kv_cache_pool(model_runner: Any) -> list:
    """Return the worker's KV-cache tensor list.

    Upstream symbol: ``GPUModelRunner.kv_caches`` (list[torch.Tensor])
        declared at vllm/v1/worker/gpu_model_runner.py:550 (v0.25.1).

    Version constraint: vllm == 0.28.*

    Contract test:
        tests/contract/test_api_surface.py::test_kv_caches_attribute

    The PoC forward reuses blocks starting at index 0 as scratch space; the
    99a372d4e fork commit ("safer kv cache reuse") added dtype/contiguity
    checks. That logic lives in gonka_poc.poc.poc_model_runner; this helper
    is just the access point.
    """
    kv = getattr(model_runner, "kv_caches", None)
    if kv is None:
        raise RuntimeError(
            "model_runner.kv_caches is not populated -- worker initialise_kv_caches() "
            "must run before PoC forward. See vllm/v1/worker/gpu_worker.py."
        )
    return kv


# ---------------------------------------------------------------------------- #
# Engine client abort (used by gating to stop in-flight inference)
# ---------------------------------------------------------------------------- #

def _list_in_flight_request_ids(engine_client: Any) -> list[str]:
    """Best-effort enumeration of in-flight request IDs on the EngineClient.

    v0.25 ``EngineClient`` ABC (vllm/engine/protocol.py:102) does NOT expose a
    ``list_requests`` / ``get_active_requests`` method. The concrete
    ``AsyncLLM`` impl (vllm/v1/engine/async_llm.py) keeps state on its
    ``output_processor.request_states`` dict[str, RequestState] -- that is
    the only stable hook in v0.25 (async_llm.py:138).

    Returns [] if no enumerable surface is available; caller then logs and
    relies on PoCGatingMiddleware to gate new requests instead.
    """
    # Path 1: AsyncLLM output_processor (v1 engine).
    op = getattr(engine_client, "output_processor", None)
    if op is not None:
        states = getattr(op, "request_states", None)
        if isinstance(states, dict):
            # Snapshot keys so concurrent mutation during abort doesn't raise.
            return list(states.keys())
    # Path 2: legacy v0 engine (kept for defensive fallback; should not hit
    # in a 0.25-only deployment but harmless if a wrapped client appears).
    engine = getattr(engine_client, "engine", None)
    if engine is not None:
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            # scheduler may be a list (per-vp) or a single instance.
            schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
            ids: list[str] = []
            for sch in schedulers:
                for queue_name in ("running", "waiting", "swapped"):
                    queue = getattr(sch, queue_name, None) or []
                    for seq_group in queue:
                        rid = getattr(seq_group, "request_id", None)
                        if rid is not None:
                            ids.append(str(rid))
            return ids
    return []


async def abort_all_requests(engine_client: Any) -> int:
    """Abort every in-flight inference request before PoC seizes the GPU.

    Upstream symbol: ``vllm.engine.protocol.EngineClient.abort`` (ABC method;
        ``async def abort(request_id: str | Iterable[str]) -> None``).

    Version constraint: vllm == 0.28.*

    Contract test:
        tests/contract/test_api_surface.py::test_engine_client_has_abort

    Returns: number of requests aborted (best-effort).

    CALLER NOTE: v0.23 ``EngineClient`` ABC has no public ``list_requests``
    API (unchanged from 0.23). We introspect the concrete ``AsyncLLM.output_processor.request_states``
    dict. If that surface vanishes in a future minor, this function returns 0
    and logs a warning -- PoCGatingMiddleware alone is sufficient to gate new
    requests (this abort path is belt-and-suspenders for already-admitted
    requests still mid-decode).
    """
    abort_fn = getattr(engine_client, "abort", None)
    if abort_fn is None:
        logger.warning(
            "abort_all_requests: engine_client has no .abort method "
            "(expected per EngineClient ABC v0.25) -- skipping."
        )
        return 0

    request_ids = _list_in_flight_request_ids(engine_client)
    if not request_ids:
        logger.warning(
            "abort_all_requests: no enumerable in-flight requests on "
            "engine_client (output_processor.request_states unavailable). "
            "PoCGatingMiddleware still blocks new admissions."
        )
        return 0

    # internal=True is required, not optional: the ids above come from
    # output_processor.request_states, whose keys are INTERNAL request ids.
    # With internal=False, abort() looks them up in the external->internal
    # map, finds nothing, and returns without aborting anything -- silently,
    # so the counter below still incremented and the log still claimed
    # success. The EngineClient ABC (vllm/engine/protocol.py:102) declares
    # abort() without the parameter, so probe rather than assume: a client
    # that is not AsyncLLM would otherwise raise TypeError on every id.
    try:
        supports_internal = "internal" in inspect.signature(abort_fn).parameters
    except (TypeError, ValueError):  # C-implemented or otherwise uninspectable
        supports_internal = False
    if not supports_internal:
        logger.warning(
            "abort_all_requests: %s.abort() has no 'internal' parameter; "
            "aborting by internal id may be a no-op on this client.",
            type(engine_client).__name__,
        )

    aborted = 0
    for rid in request_ids:
        try:
            # EngineClient.abort accepts a single id or an iterable; we issue
            # per-id calls so one bad id does not abort the rest of the loop.
            if supports_internal:
                await abort_fn(rid, internal=True)
            else:
                await abort_fn(rid)
            aborted += 1
        except Exception as exc:  # noqa: BLE001 -- best-effort, never raise
            logger.warning(
                "abort_all_requests: abort(%s) failed: %s", rid, exc
            )
    return aborted


# ---------------------------------------------------------------------------- #
# KV block borrowing (PoC validation without aborting inference)
# ---------------------------------------------------------------------------- #

_ENGINE_CORE_BORROW_FLAG = "_gonka_poc_borrow_installed"


def install_engine_core_poc_methods() -> bool:
    """Install ``gonka_poc_borrow_blocks``/``gonka_poc_return_blocks`` on
    ``vllm.v1.engine.core.EngineCore`` (class-level, idempotent).

    The BlockPool lives in the engine-core process; the plugin entry point
    runs there too (load_general_plugins in vllm/v1/engine/core.py:107-109, v0.25.1), so injecting
    the methods at plugin-register time makes them reachable over the
    UTILITY RPC (``call_utility_async`` -> ``getattr(self, method)``).

    Upstream symbols pinned here:
        * ``EngineCore`` (vllm/v1/engine/core.py:96) with ``self.scheduler``
          (core.py:150); ``Scheduler.kv_cache_manager``
          (vllm/v1/core/sched/scheduler.py:262)
        * ``KVCacheManager.block_pool`` = the ONE pool shared by every
          kv-cache group (vllm/v1/core/kv_cache_manager.py:161) and
          ``KVCacheManager.kv_cache_config`` (kv_cache_manager.py:162)
        * ``BlockPool.get_new_blocks`` (vllm/v1/core/block_pool.py:542) /
          ``free_blocks`` (block_pool.py:614) / ``get_num_free_blocks``
          (block_pool.py:692); the null block is id 0 and never in the
          free queue (block_pool.py:188-192)

    Sizing happens HERE, per kv-cache group -- the frontend-visible scalar
    ``cache_config.block_size`` undercounts by up to block_size/min_g on
    multi-group models (DeepSeek-V4 SWA compressor: 8 vs the main group).
    Because the block-id namespace is pool-global, ONE lease reserves the
    id's byte range in EVERY group's tensors simultaneously.

    Version constraint: vllm == 0.28.*

    Contract test:
        tests/contract/test_api_surface.py::test_kv_block_pool_borrow_surface
    """
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception as exc:  # import environment without the v1 engine
        logger.debug(
            "EngineCore import failed, PoC borrow methods not installed: %s",
            exc)
        return False

    if getattr(EngineCore, _ENGINE_CORE_BORROW_FLAG, False):
        return True

    def gonka_poc_borrow_blocks(self, num_nonces: int, seq_len: int):
        """Lease free KV blocks for a PoC validation forward.

        Returns ``{"block_ids": [...], "blocks_per_seq": int}`` or ``None``
        when the pool cannot spare them. ``blocks_per_seq`` (the per-sequence
        stripe) = ``max_g ceil(seq_len / manager_block_size_g)`` over every
        kv-cache group -- one lease covers ALL groups.
        """
        try:
            kv_mgr = self.scheduler.kv_cache_manager
            block_pool = kv_mgr.block_pool
            groups = kv_mgr.kv_cache_config.kv_cache_groups
            blocks_per_seq = max(
                math.ceil(int(seq_len) / int(g.kv_cache_spec.block_size))
                for g in groups)
        except Exception as exc:
            logger.warning(
                "gonka_poc_borrow_blocks: pool introspection failed: %s", exc)
            return None
        needed = int(num_nonces) * int(blocks_per_seq)
        if needed <= 0 or needed > block_pool.get_num_free_blocks():
            return None
        try:
            blocks = block_pool.get_new_blocks(needed)
        except Exception:
            return None
        block_ids = [b.block_id for b in blocks if not b.is_null]
        if len(block_ids) != needed:
            # A null/placeholder block slipped in -- roll back rather than
            # risk writing PoC K/V over the null block.
            block_pool.free_blocks(blocks)
            return None
        return {"block_ids": block_ids, "blocks_per_seq": blocks_per_seq}

    def gonka_poc_return_blocks(self, block_ids) -> None:
        """Return blocks previously leased via ``gonka_poc_borrow_blocks``."""
        if not block_ids:
            return
        block_pool = self.scheduler.kv_cache_manager.block_pool
        block_pool.free_blocks(
            [block_pool.blocks[int(bid)] for bid in block_ids])

    EngineCore.gonka_poc_borrow_blocks = gonka_poc_borrow_blocks
    EngineCore.gonka_poc_return_blocks = gonka_poc_return_blocks
    setattr(EngineCore, _ENGINE_CORE_BORROW_FLAG, True)
    logger.info("gonka-poc: EngineCore KV borrow/return methods installed")
    return True


def _utility_call(engine_client: Any):
    """Resolve the UTILITY RPC callable or raise (feature-unavailable)."""
    core = getattr(engine_client, "engine_core", None)
    call = getattr(core, "call_utility_async", None)
    if call is None:
        raise RuntimeError(
            "engine_core.call_utility_async unavailable on this EngineClient")
    return call


async def borrow_poc_blocks(
    engine_client: Any, num_nonces: int, seq_len: int,
) -> Optional[dict]:
    """Frontend wrapper: ask the engine core for a KV block lease.

    Upstream symbols: ``AsyncLLM.engine_core`` (vllm/v1/engine/async_llm.py:146) and
    ``call_utility_async(method, *args)`` on the MP client
    (vllm/v1/engine/core_client.py:1101/:1449, v0.25.1). Raises when that surface is missing or the RPC
    fails (callers treat it as feature-unavailable and fall back);
    returns ``None`` when the pool is merely busy.

    Version constraint: vllm == 0.28.*
    """
    parallel = getattr(
        getattr(engine_client, "vllm_config", None), "parallel_config", None)
    if parallel is not None and getattr(parallel, "data_parallel_size", 1) > 1:
        # The DP internal-LB client fans call_utility_async to EVERY engine
        # and returns only engine-0's result: each other engine would leak a
        # lease, and the paired return would free foreign ids on their pools
        # (assert-crash or cross-request corruption). Refuse -> callers fall
        # back to the abort-based legacy path.
        raise RuntimeError(
            "KV borrow is unsupported with data_parallel_size > 1")
    return await _utility_call(engine_client)(
        "gonka_poc_borrow_blocks", int(num_nonces), int(seq_len))


async def return_poc_blocks(engine_client: Any, block_ids: list) -> None:
    """Frontend wrapper: return a previously leased KV block set."""
    if not block_ids:
        return
    await _utility_call(engine_client)(
        "gonka_poc_return_blocks", list(block_ids))


__all__ = [
    "build_common_attention_metadata",
    "build_attn_metadata_per_group",
    "get_kv_cache_pool",
    "abort_all_requests",
    "install_engine_core_poc_methods",
    "borrow_poc_blocks",
    "return_poc_blocks",
]
