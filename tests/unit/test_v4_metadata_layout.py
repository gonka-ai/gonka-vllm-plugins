"""Unit tests for the per-attention-group PoC metadata layout and the
DeepSeek-V4 pseudo input-ids derivation (``poc_model_runner``).

Motivation (DeepSeek-V4 bring-up): models may register KV cache groups with
DIFFERENT block sizes (V4: sparse MLA / indexer at ``cache_config.block_size``,
SWA compressor at 8). A single slot_mapping built for the main group hands
out-of-range slot ids to the other pools — OOB writes crash on sm_90 and
silently corrupt memory on sm_100. The runner therefore builds the layout per
group; these tests pin the layout math and the ids scheme, CPU-only. The
layout tests exercise ``poc_model_runner._inplace_layout`` directly and skip
when vllm is unavailable (``poc_model_runner`` imports it at module level);
the ids tests need only ``gpu_random`` and always run.
"""
from __future__ import annotations

import math

import pytest
import torch

from gonka_poc.poc.gpu_random import _seed_from_string, derive_pseudo_input_ids


# --------------------------------------------------------------------------- #
# Layout math: the vectorized mapping must equal the naive per-token walk.
# --------------------------------------------------------------------------- #

def _naive_layout(batch_size: int, seq_len: int, g_block: int) -> torch.Tensor:
    blocks_per_seq = math.ceil(seq_len / g_block)
    slots = []
    for seq_idx in range(batch_size):
        base_block = 1 + seq_idx * blocks_per_seq  # block 0 is the null block
        for t in range(seq_len):
            b = base_block + t // g_block
            slots.append(b * g_block + t % g_block)
    return torch.tensor(slots, dtype=torch.long)


def _inplace_layout(batch_size: int, seq_len: int, g_block: int):
    # The REAL implementation (shared with ``_create_v1_attn_metadata._layout``).
    runner = pytest.importorskip(
        "gonka_poc.poc.poc_model_runner", reason="requires vllm")
    return runner._inplace_layout(
        batch_size, seq_len, g_block, torch.device("cpu"))


def _vectorized_layout(batch_size: int, seq_len: int, g_block: int) -> torch.Tensor:
    return _inplace_layout(batch_size, seq_len, g_block)[0]


@pytest.mark.parametrize(
    "batch_size,seq_len,g_block",
    [
        (1, 256, 256),    # main sparse-MLA group, single sequence
        (16, 256, 256),
        (16, 256, 8),     # V4 SWA compressor group (compress_ratio=128)
        (16, 1024, 8),
        (3, 100, 8),      # seq_len not a multiple of block
        (2, 257, 256),    # seq_len barely over one block
        (16, 1024, 64),
    ],
)
def test_vectorized_layout_matches_naive(batch_size, seq_len, g_block):
    assert torch.equal(
        _naive_layout(batch_size, seq_len, g_block),
        _vectorized_layout(batch_size, seq_len, g_block),
    )


def _block_table(batch_size: int, seq_len: int, g_block: int) -> torch.Tensor:
    return _inplace_layout(batch_size, seq_len, g_block)[1]


def test_per_group_block_tables_differ():
    """The core of the OOB bug: the block TABLE must be built per group.

    For seq_len=256 a 256-block group needs 1 block per sequence while the
    8-block compressor group needs 32; a backend indexing ``bt[seq, t//8]``
    against a 1-column shared table reads past the table and addresses
    foreign blocks. Slot ids may coincide when padded extents match — the
    table shape is what diverges.
    """
    main_bt = _block_table(16, 256, 256)
    comp_bt = _block_table(16, 256, 8)
    assert main_bt.shape == (16, 1)
    assert comp_bt.shape == (16, 32)
    # Padded extents equal here (256 | 256), so slot ids coincide up to the
    # per-group skip of block 0 (one block of that group's size)...
    assert torch.equal(
        _vectorized_layout(16, 256, 256) - 256, _vectorized_layout(16, 256, 8) - 8)
    # ...but diverge as soon as the padded extents differ
    # (seq_len=260: ceil/8*8 = 264 vs ceil/256*256 = 512):
    assert not torch.equal(
        _vectorized_layout(2, 260, 8) - 8, _vectorized_layout(2, 260, 256) - 256)


# --------------------------------------------------------------------------- #
# Pseudo input-ids: deterministic, framework-independent, per-nonce distinct.
# --------------------------------------------------------------------------- #

def _derive_ids(block_hash, public_key, nonces, seq_len, vocab):
    # The REAL implementation (called by ``poc_model_runner._generate_poc_input_ids``).
    return derive_pseudo_input_ids(
        block_hash, public_key, nonces, seq_len, vocab, torch.device("cpu"))


def test_pseudo_ids_deterministic_and_in_range():
    a = _derive_ids("bh", "pk", [0, 1, 2], 128, vocab=163840)
    b = _derive_ids("bh", "pk", [0, 1, 2], 128, vocab=163840)
    assert torch.equal(a, b)
    assert a.dtype == torch.int32
    assert a.shape == (3 * 128,)
    assert int(a.min()) >= 0 and int(a.max()) < 163840


def test_pseudo_ids_vary_by_nonce_and_namespace():
    base = _derive_ids("bh", "pk", [0], 128, vocab=163840)
    other_nonce = _derive_ids("bh", "pk", [1], 128, vocab=163840)
    assert not torch.equal(base, other_nonce)
    # The "_input_ids" suffix namespaces ids away from the embedding stream:
    # same (bh, pk, nonce) but a different seed string must give different ids.
    alt_seed = _seed_from_string("bh_pk_nonce0")
    ids_seed = _seed_from_string("bh_pk_nonce0_input_ids")
    assert alt_seed != ids_seed


def test_pseudo_ids_reference_vectors():
    """Frozen reference vectors for the derivation convention (v1).

    Mirrors docs/pseudo-input-ids-convention.md. An independent
    implementation (in-band PoC line) must reproduce these before
    DeepSeek-V4 network activation; any change to the derivation breaks
    this test on purpose.
    """
    bh, pk, vocab = "poc_ids_convention_v1_block", "poc_ids_convention_v1_pubkey", 163840
    assert _seed_from_string(f"{bh}_{pk}_nonce0_input_ids") == 2980507924
    assert _seed_from_string(f"{bh}_{pk}_nonce1_input_ids") == 2917457512
    expected0 = [119046, 160019, 98337, 39450, 94909, 163782, 59011, 57361,
                 156377, 36469, 139643, 43988, 50299, 147011, 12130, 86013]
    expected1 = [137719, 146475, 40370, 95731, 105609, 72910, 88608, 153625,
                 112488, 156080, 83284, 17839, 70977, 48769, 15417, 65900]
    assert _derive_ids(bh, pk, [0], 16, vocab).tolist() == expected0
    assert _derive_ids(bh, pk, [1], 16, vocab).tolist() == expected1
    ids256 = _derive_ids(bh, pk, [0], 256, vocab)
    assert (int(ids256.sum()), int(ids256.min()), int(ids256.max())) == (20594062, 181, 163834)
