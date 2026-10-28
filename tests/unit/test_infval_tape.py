"""The tape is a sampler: its winners follow the sampling distribution, and
for a fixed seed and position the winner is fixed."""
from __future__ import annotations

import torch

from gonka_poc.infval.tape import gumbel, tape_key, truncate


def test_tape_winner_samples_the_distribution_and_is_reproducible():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0], dtype=torch.float64)
    n = 20000
    keys = torch.full((n,), tape_key(123456789012345), dtype=torch.int64)
    positions = torch.arange(n, dtype=torch.int32)
    winners = (logits + gumbel(keys, positions, 4)).argmax(dim=-1)
    freq = torch.bincount(winners, minlength=4).double() / n
    assert torch.allclose(freq, logits.softmax(dim=-1), atol=0.015)
    again = (logits + gumbel(keys, positions, 4)).argmax(dim=-1)
    assert torch.equal(winners, again)
    other = torch.full((n,), tape_key(1), dtype=torch.int64)
    assert not torch.equal(winners, (logits + gumbel(other, positions, 4)).argmax(-1))


def test_truncate_keeps_top_k_and_top_p_sets():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]] * 3)
    kept = torch.isfinite(truncate(probs.log(), torch.tensor([2, 0, 0]),
                                   torch.tensor([1.0, 0.75, 1.0])))
    assert kept.tolist() == [[True, True, False, False],
                             [True, True, False, False],
                             [True, True, True, True]]
