"""Compat shim for vLLM 0.30.x (checked against tag ced6857af).

Every private surface the 0.28 shim uses is unchanged in 0.30 except
``CommonAttentionMetadata``: ``_seq_lens_cpu`` and ``_num_computed_tokens_cpu``
are gone (0.30 derives the computed-token count from ``seq_lens`` and the query
lengths) and ``is_prefilling`` is new, indexed by the KDA and Mamba builders
without a ``None`` check.
"""
from __future__ import annotations

from typing import Any, Optional

from gonka_poc._compat.v0_28 import (
    abort_all_requests,
    borrow_poc_blocks,
    build_attn_metadata_per_group,
    get_kv_cache_pool,
    install_engine_core_poc_methods,
    return_poc_blocks,
)


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
    is_prefilling: Optional[Any] = None,
) -> Any:
    """``CommonAttentionMetadata`` for a PoC forward, in which every row is an
    atomic prefill. Same kwargs as the 0.28 shim; the two removed fields are
    accepted and ignored."""
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata

    if is_prefilling is None:
        is_prefilling = torch.ones(int(num_reqs), dtype=torch.bool)
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
        positions=positions,
        is_prefilling=is_prefilling,
    )


__all__ = [
    "build_common_attention_metadata",
    "build_attn_metadata_per_group",
    "get_kv_cache_pool",
    "abort_all_requests",
    "install_engine_core_poc_methods",
    "borrow_poc_blocks",
    "return_poc_blocks",
]
