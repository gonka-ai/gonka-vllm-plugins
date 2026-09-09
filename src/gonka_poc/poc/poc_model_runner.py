"""PoC model runner for the vLLM V1 architecture (supported minors per
``gonka_poc._compat``); version-specific surfaces live in the compat shim.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — processes all nonces in a single forward call.

Private-API touchpoint policy
-----------------------------
All ``vllm.v1.*`` private surfaces are routed through the version-dispatched
compat shim. The import binds the resolver function and each consumer calls
it to obtain the actual module: ``compat = _compat_current();
compat.build_common_attention_metadata(...)``. Touchpoints:
    * ``CommonAttentionMetadata`` construction
    * per-group ``AttentionMetadata`` construction (iteration over
      ``model_runner.attn_groups``, per-group ``kv_cache_spec.block_size``
      resolution, ``builder.build``) via ``build_attn_metadata_per_group``
    * ``model_runner.kv_caches`` access

The following vLLM imports REMAIN at module scope because they are public
(re-exported via the package root):
    * ``vllm.distributed.get_pp_group`` / ``get_tp_group``
      (pinned by ``tests/contract/test_api_surface.py::test_distributed_groups_present``)
    * ``vllm.distributed.communication_op.broadcast_tensor_dict``
      (pinned by ``::test_communication_op_broadcast``)
    * ``vllm.forward_context.set_forward_context``
      (pinned by ``::test_forward_context_set``)
    * ``vllm.sequence.IntermediateTensors``
      (pinned by ``::test_intermediate_tensors``)
    * ``vllm.logger.init_logger``

If a future minor reshuffles any of these into private namespaces, move
the import into the compat shim and add a contract-test pin.
"""
import math
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from gonka_poc._compat import current as _compat_current

from .gpu_random import (
    generate_inputs,
    generate_inputs_concat_murmur,
    derive_pseudo_input_ids,
    random_pick_indices,
    apply_haar_rotation,
)

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


def _borrowed_layout(
    batch_size: int,
    seq_len: int,
    g_block: int,
    m_block: int,
    borrowed_block_ids: List[int],
    stripe: int,
    device,
):
    """slot_mapping + block_table over a LEASED set of pool blocks.

    ``borrowed_block_ids`` are pool-unit (manager) block ids leased from the
    ONE engine-wide BlockPool; ``stripe`` is the per-sequence allotment
    (``max_g ceil(seq_len/manager_block_size_g)`` — computed by the engine
    core with full group knowledge). Sequence ``i`` uses the first
    ``ceil(seq_len/m_block)`` ids of its stripe
    ``borrowed_block_ids[i*stripe : (i+1)*stripe]``.

    Unit conversion: slot/table math runs in KERNEL units (``g_block`` from
    ``builder.kv_cache_spec`` — possibly a kernel split of the manager
    size). A pool block ``b`` covers kernel blocks ``b*r .. b*r+r-1``
    (``r = m_block//g_block``), i.e. the contiguous slot range
    ``[b*m_block, (b+1)*m_block)`` — so the slot for token ``t`` of
    sequence ``i`` is ``L[i][t//m_block]*m_block + t%m_block``, and the
    kernel block table entry ``k`` is ``L[i][k//r]*r + k%r``. Using a raw
    pool id as a kernel id WITHOUT the ×r expansion would address bytes
    inside pool block ``id//r`` — unleased, possibly live inference KV.

    Fail-loud guards (ValueError → RPC error → the chunk fails visibly,
    never a silent mis-write): non-divisible split, stripe too small,
    lease too small.
    """
    if m_block % g_block != 0:
        raise ValueError(
            f"PoC borrowed layout: manager block {m_block} is not a "
            f"multiple of kernel block {g_block}")
    r = m_block // g_block
    bps = math.ceil(seq_len / m_block)
    if bps > stripe:
        raise ValueError(
            f"PoC lease stripe {stripe} too small: group with manager "
            f"block {m_block} needs {bps} blocks/seq at seq_len {seq_len}")
    if batch_size * stripe > len(borrowed_block_ids):
        raise ValueError(
            f"PoC lease has {len(borrowed_block_ids)} blocks, needs "
            f"{batch_size * stripe} ({batch_size} seqs × stripe {stripe})")

    ids = torch.tensor(
        borrowed_block_ids[:batch_size * stripe],
        dtype=torch.long, device=device).view(batch_size, stripe)
    seq_blocks = ids[:, :bps]  # [batch, bps] pool-unit ids

    j = torch.arange(seq_len, dtype=torch.long, device=device) // m_block
    off = torch.arange(seq_len, dtype=torch.long, device=device) % m_block
    slot_mapping = (seq_blocks[:, j] * m_block + off).reshape(-1)

    kernel_ids = (
        seq_blocks.unsqueeze(-1) * r
        + torch.arange(r, dtype=torch.long, device=device)
    ).view(batch_size, bps * r)
    block_table = kernel_ids.to(torch.int32)
    return slot_mapping, block_table


def _inplace_layout(batch_size, seq_len, g_block, device, row_offset=0):
    """Legacy in-place slot_mapping + block_table over blocks ``0..N``.

    slot = (seq_idx*blocks_per_seq + t//g_block)*g_block + t%g_block
         = seq_idx*padded_len + t   (contiguous per sequence, padded to
    a block multiple), so the mapping vectorizes to two aranges.

    ``row_offset`` shifts the whole layout down by that many sequences —
    the split-prefill path runs sub-chunks of one big batch, and sub-chunk
    rows must land in the same blocks the joint decode phase will read
    (offset 0 == the historical layout, bit-path unchanged).
    """
    blocks_per_seq = math.ceil(seq_len / g_block)
    padded = blocks_per_seq * g_block
    base = ((torch.arange(batch_size, dtype=torch.long, device=device)
             + row_offset) * padded).repeat_interleave(seq_len)
    slot_mapping = base + torch.arange(
        seq_len, dtype=torch.long, device=device).repeat(batch_size)
    block_table = (torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ) + row_offset * blocks_per_seq).view(batch_size, blocks_per_seq)
    return slot_mapping, block_table


def _create_v1_attn_metadata(batch_size, seq_len, device, worker, positions,
                             borrowed_block_ids=None, borrowed_stripe=None,
                             alloc_len=None, decode_step=None, row_offset=0):
    """Create attention metadata, built independently for every attention group.

    Models may register KV cache groups with DIFFERENT block sizes (e.g.
    DeepSeek-V4: sparse MLA and indexer use ``cache_config.block_size``,
    the SWA compressor uses its own ``block_size`` — 8 at compress_ratio=128).
    Sharing one slot_mapping / block_table built for the main group hands
    out-of-range slot ids to the other pools: OOB writes cause an illegal
    memory access on sm_90 and silent memory corruption elsewhere. The
    layout is therefore computed per group from that group's
    ``kv_cache_spec.block_size``. For single-group models this reduces to
    exactly the previous behaviour.

    Two block sources:
      * ``borrowed_block_ids is None`` — legacy in-place layout over blocks
        ``0..N`` (mining and the abort-based fallback). BIT-PATH UNCHANGED.
      * lease (``borrowed_block_ids`` + ``borrowed_stripe`` from
        ``gonka_poc_borrow_blocks``) — validation runs on pool blocks that
        are provably disjoint from live inference; see
        :func:`_borrowed_layout`. Physical block ids enter ONLY the address
        translation (scatter targets / gather tables), never the attention
        math, so artifacts are invariant to block choice.

    ``positions`` is the shared per-token position tensor (also passed to the
    model forward); DeepSeek-V4's C128A metadata builder requires it, every
    other v0.23 backend ignores ``cm.positions``.

    Decode-PoC extensions (both None => the prefill behaviour above is
    BIT-PATH UNCHANGED):
      * ``alloc_len``   — reserve the layout for this many slots per sequence
        (``seq_len + max_tokens``; the lease is taken up-front, no growth);
      * ``decode_step`` — build metadata for ONE decode token per sequence at
        absolute position ``decode_step`` (query_len 1, seq_len
        ``decode_step+1``, slot ``decode_step`` of each row).
    """
    compat = _compat_current()
    layout_len = int(alloc_len) if alloc_len is not None else seq_len

    if decode_step is None:
        q_len, kv_len = seq_len, seq_len
        n_computed = 0
    else:
        q_len, kv_len = 1, int(decode_step) + 1
        n_computed = int(decode_step)
    total_tokens = batch_size * q_len

    query_start_loc_gpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * q_len)
    query_start_loc_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * q_len)
    seq_lens_gpu = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    seq_lens_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cpu")
    num_computed_cpu = torch.full((batch_size,), n_computed, dtype=torch.int32,
                                  device="cpu")

    def _layout(g_block, m_block):
        if borrowed_block_ids is not None:
            # Sub-chunk callers pass the SLICE of the lease covering their
            # rows, so no offset is needed here.
            slot_all, table = _borrowed_layout(
                batch_size, layout_len, g_block, m_block,
                borrowed_block_ids, int(borrowed_stripe or 0), device)
        else:
            slot_all, table = _inplace_layout(batch_size, layout_len, g_block,
                                              device, row_offset=row_offset)
        if alloc_len is None:
            return slot_all, table
        # slot_all covers layout_len slots per row; cut the piece this forward
        # actually writes: prefill -> first seq_len slots, decode -> slot
        # ``decode_step`` of each row.
        per_row = slot_all.view(batch_size, layout_len)
        if decode_step is None:
            return per_row[:, :seq_len].reshape(-1), table
        return per_row[:, int(decode_step)].reshape(-1), table

    layouts = {}

    def _layout_for_block_size(g_block, m_block):
        key = (g_block, m_block)
        if key not in layouts:
            layouts[key] = _layout(g_block, m_block)
        return layouts[key]

    def _common_metadata_for_layout(slot_mapping, block_table):
        return compat.build_common_attention_metadata(
            positions=positions,
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens_gpu,
            num_reqs=batch_size,
            num_actual_tokens=total_tokens,
            max_query_len=q_len,
            max_seq_len=kv_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_cpu,
        )

    return compat.build_attn_metadata_per_group(
        worker.model_runner,
        layout_for_block_size=_layout_for_block_size,
        common_metadata_for_layout=_common_metadata_for_layout,
    )


