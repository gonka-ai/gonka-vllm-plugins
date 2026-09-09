# SPDX-License-Identifier: Apache-2.0
"""Two-stage seeded forcing for grouped top-k routers.

CPU-only. Three layers of protection:
  * PROPERTY tests against a naive reimplementation of the DeepSeek-style
    grouped selector (groups by top-2 sum, then top_k inside the survivors):
    the engine-side selection over our forced logits must recover exactly
    the seeded expert set, even under an e_score_correction_bias.
  * COVERAGE invariant: every seeded group holds >= 1 seeded expert (an
    empty picked group would tie at the -1e4 floor and hand the outcome to
    kernel ordering).
  * SNAPSHOT of the formula for a fixed seed — the consensus freeze: any
    drift in salts/derivation flips this test.
"""
import torch

from gonka_poc.poc import gpu_random as g

DEV = torch.device("cpu")


def _forced(seeds, n_experts, top_k, n_group, topk_group):
    base = torch.tensor(seeds, dtype=torch.int64)
    steps = torch.zeros(len(seeds), dtype=torch.int64)
    return g.expert_logits_from_base(base, steps, n_experts, top_k, DEV,
                                     n_group=n_group, topk_group=topk_group)


def _naive_grouped_topk(logits, n_group, topk_group, top_k, bias=None):
    """Reference DeepSeek-style selector: score groups by their top-2 sum,
    keep topk_group groups, mask the rest, then top_k over survivors."""
    scores = logits if bias is None else logits + bias
    b, n = scores.shape
    gsz = n // n_group
    grouped = scores.view(b, n_group, gsz)
    gscore = grouped.topk(2, dim=-1).values.sum(-1)          # [B, n_group]
    keep = gscore.topk(topk_group, dim=-1).indices           # [B, topk_group]
    mask = torch.zeros(b, n_group, dtype=torch.bool)
    mask.scatter_(1, keep, True)
    masked = scores.masked_fill(
        ~mask.unsqueeze(-1).expand(b, n_group, gsz).reshape(b, n), float("-inf"))
    return masked.topk(top_k, dim=-1).indices


def test_engine_selection_recovers_seeded_set():
    n_experts, top_k, n_group, topk_group = 256, 8, 8, 4
    logits = _forced(list(range(64)), n_experts, top_k, n_group, topk_group)
    seeded = (logits > 0).nonzero(as_tuple=False)
    chosen = _naive_grouped_topk(logits, n_group, topk_group, top_k)
    for row in range(logits.shape[0]):
        want = set(seeded[seeded[:, 0] == row][:, 1].tolist())
        got = set(chosen[row].tolist())
        assert got == want, f"row {row}: {got} != {want}"


def test_bias_cannot_change_the_selected_set():
    n_experts, top_k, n_group, topk_group = 256, 8, 8, 4
    logits = _forced(list(range(32)), n_experts, top_k, n_group, topk_group)
    gen = torch.Generator().manual_seed(7)
    bias = torch.randn(n_experts, generator=gen) * 3.0   # far above real |bias|
    chosen = _naive_grouped_topk(logits, n_group, topk_group, top_k,
                                 bias=bias.unsqueeze(0))
    want = _naive_grouped_topk(logits, n_group, topk_group, top_k)
    for row in range(logits.shape[0]):
        assert set(chosen[row].tolist()) == set(want[row].tolist())


def test_coverage_every_seeded_group_nonempty():
    for n_experts, n_group, topk_group, top_k in [
            (256, 8, 4, 8), (384, 12, 4, 8), (64, 8, 8, 8), (160, 10, 5, 6)]:
        logits = _forced(list(range(48)), n_experts, top_k, n_group,
                         topk_group)
        gsz = n_experts // n_group
        for row in range(logits.shape[0]):
            experts = (logits[row] > 0).nonzero().flatten()
            groups = torch.unique(experts // gsz)
            assert len(groups) == topk_group, (row, groups)
            assert len(experts) == top_k
            assert len(torch.unique(experts)) == top_k    # distinct


def test_flat_router_path_untouched_by_group_params():
    """n_group<=1 must be byte-identical to the historical flat formula —
    the MiniMax consensus does not move."""
    base = torch.tensor([11, 22, 33], dtype=torch.int64)
    steps = torch.tensor([5, 5, 5], dtype=torch.int64)
    a = g.expert_logits_from_base(base, steps, 256, 8, DEV)
    b = g.expert_logits_from_base(base, steps, 256, 8, DEV,
                                  n_group=1, topk_group=1)
    assert torch.equal(a, b)


def test_snapshot_consensus_freeze():
    """Fixed-seed snapshot of the two-stage formula. If this test moves, the
    consensus formula moved — that requires a coordinated network event, not
    a code review."""
    logits = _forced([0xC0FFEE], 256, 8, 8, 4)[0]
    chosen = tuple(sorted((logits > 0).nonzero().flatten().tolist()))
    ranked = sorted(range(256), key=lambda e: -logits[e])[:8]
    assert chosen == (41, 50, 134, 146, 152, 171, 179, 248)
    snap = (chosen, tuple(round(float(logits[e]), 1) for e in ranked))
    import hashlib, json
    digest = hashlib.sha256(json.dumps(snap).encode()).hexdigest()[:16]
    assert digest == "5c6952bfc2affd4a", (
        f"grouped-forcing formula drifted: digest {digest}, "
        f"experts {snap[0]}")
