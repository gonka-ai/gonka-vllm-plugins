"""Unit tests for the legacy in-place PoC KV layout (``_inplace_layout``).

Block 0 is vLLM's null block (``NULL_BLOCK_ID == 0``): the engine hands it out
as padding, and mamba-style state kernels skip any sequence whose state index
is 0. The in-place layout therefore starts at block 1. These tests pin that,
the slot<->table consistency invariant, and cross-sequence collision-freedom.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("vllm")  # poc_model_runner imports vllm.distributed
import torch  # noqa: E402

from gonka_poc.poc.poc_model_runner import _inplace_layout  # noqa: E402


@pytest.mark.parametrize("g_block", [8, 16, 64, 2304])
@pytest.mark.parametrize("seq_len", [1, 7, 256, 1024])
@pytest.mark.parametrize("batch", [1, 3, 32])
def test_never_uses_block_zero(g_block, seq_len, batch):
    slot_mapping, block_table = _inplace_layout(batch, seq_len, g_block, device="cpu")
    bps = math.ceil(seq_len / g_block)
    assert block_table.shape == (batch, bps)
    assert block_table.dtype == torch.int32
    assert block_table.min().item() == 1
    # Sequence i owns blocks 1+i*bps .. i*bps+bps, contiguous and disjoint.
    assert torch.equal(
        block_table.reshape(-1),
        torch.arange(1, batch * bps + 1, dtype=torch.int32))
    # No slot lands inside block 0.
    assert slot_mapping.min().item() >= g_block


@pytest.mark.parametrize("g_block,seq_len", [(16, 100), (64, 65), (2304, 1024), (8, 8)])
def test_slot_table_consistency(g_block, seq_len):
    batch = 4
    slot_mapping, block_table = _inplace_layout(batch, seq_len, g_block, device="cpu")
    assert slot_mapping.shape == (batch * seq_len,)
    for i in range(batch):
        for t in range(seq_len):
            slot = slot_mapping[i * seq_len + t].item()
            # slot // g_block must be the table entry for token t of seq i.
            assert slot // g_block == block_table[i, t // g_block].item()
            assert slot % g_block == t % g_block
    # Every slot is unique across the whole batch.
    assert slot_mapping.unique().numel() == slot_mapping.numel()
