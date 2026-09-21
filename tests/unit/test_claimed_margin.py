"""A disagreement is judged against the CLAIMED cell; a non-finite step is no verdict."""
import pytest
import torch

from gonka_poc.poc.sphere import (
    SPHERE_DIM, SPHERE_POINTS, claimed_margin, project_to_sphere, snap_with_scores,
)
from gonka_poc.poc.validation import run_validation, unscored_nonces

TAU = 0.025


def _codebook():
    g = torch.Generator().manual_seed(0)
    return project_to_sphere(torch.randn(SPHERE_POINTS, SPHERE_DIM, generator=g))


def _boundary_query(cb, a, b, eps=0.004):
    """A query almost equidistant from cells a and b (top1-top2 gap ~eps)."""
    return project_to_sphere((cb[a] * (1 + eps) + cb[b]).unsqueeze(0))


def test_runner_up_claim_is_jitter_far_claim_is_not():
    cb = _codebook()
    q = _boundary_query(cb, 3, 7)
    k, bad, scores = snap_with_scores(q, cb)
    assert k.item() == 3 and not bad.item()
    top2 = scores.topk(2).values[0]
    assert top2[0] - top2[1] <= TAU                   # boundary: top1 ~ top2
    runner_up = claimed_margin(scores, torch.tensor([7]))
    assert runner_up.item() == pytest.approx((top2[0] - top2[1]).item(), abs=1e-6)
    assert runner_up.item() <= TAU                    # may pass
    farthest = int(scores.argmin().item())
    far = claimed_margin(scores, torch.tensor([farthest]))
    assert far.item() > TAU                           # must fail
    assert claimed_margin(scores, k).item() == 0.0    # own snap: no gap


def test_claim_outside_codebook_is_maximal_gap():
    cb = _codebook()
    _, _, scores = snap_with_scores(_boundary_query(cb, 1, 2), cb)
    assert claimed_margin(scores, torch.tensor([SPHERE_POINTS])).item() == 2.0
    assert claimed_margin(scores, torch.tensor([-1])).item() == 2.0


def test_non_finite_row_snaps_to_minus_one():
    cb = _codebook()
    q = torch.cat([_boundary_query(cb, 1, 2), torch.full((1, SPHERE_DIM), float("nan"))])
    k, bad, scores = snap_with_scores(q, cb)
    assert k.tolist()[1] == -1 and bad.tolist() == [False, True]
    assert torch.all(scores[1] == 0)


def _artifact(nonce, m, d, n_nan=0, steps=5):
    return {"nonce": nonce, "vector_b64": "", "k_points_steps": [0] * steps,
            "n_sphere_mismatches": m, "n_nan_steps": n_nan, "mismatch_margin_max": d}


def test_run_validation_flags_by_claimed_margin():
    ok = run_validation([_artifact(1, 2, TAU / 2), _artifact(2, 1, TAU * 4)], {}, 2,
                        dist_threshold=TAU, use_trajectory=True)
    assert ok["n_mismatch"] == 1 and ok["mismatch_nonces"] == [2]


def test_unscored_nonces_leave_the_sample_up_to_a_cap():
    arts = [_artifact(n, 0, 0.0) for n in range(63)]          # nonce 63 never came back
    assert unscored_nonces(list(range(64)), arts) == [63]
    r = run_validation(arts, {}, 64, dist_threshold=TAU, use_trajectory=True,
                       requested_nonces=list(range(64)))
    assert r["n_total"] == 63 and r["excluded_nonces"] == [63] and not r["fraud_detected"]
    with pytest.raises(ValueError, match="produced no artifact"):
        unscored_nonces(list(range(64)), arts[:50])            # 14 of 64 missing
    assert unscored_nonces([1, 2, 3, 4, 5, 6, 7, 8], arts[1:8]) == [8]   # 1 of 8 ok
