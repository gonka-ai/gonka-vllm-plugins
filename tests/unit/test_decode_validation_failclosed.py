import pytest

from gonka_poc.poc.validation import run_validation

STEPS = 9  # max_tokens 8


def _computed(nonce, n_steps=STEPS, mismatches=0, margin=0.0):
    return {"nonce": nonce, "vector_b64": "", "k_points_steps": [0] * n_steps,
            "n_sphere_mismatches": mismatches, "mismatch_margin_max": margin}


def _run(computed, refs, nonces):
    return run_validation(
        computed_artifacts=computed, validation_map={n: "" for n in nonces},
        n_total=len(computed), requested_nonces=nonces,
        dist_threshold=0.025, p_mismatch=0.001, fraud_threshold=0.01,
        use_trajectory=True, references=refs, expected_steps=STEPS)


def test_honest_sample_passes():
    nonces = list(range(4))
    r = _run([_computed(n) for n in nonces], {n: [0] * STEPS for n in nonces}, nonces)
    assert not r["fraud_detected"] and r["n_total"] == 4 and r["n_excluded"] == 0


@pytest.mark.parametrize("ref", [[0, 0], [0] * (STEPS + 1), [0] * (STEPS - 1) + [16],
                                 [-1] + [0] * (STEPS - 1)])
def test_malformed_reference_is_a_mismatch(ref):
    nonces = list(range(4))
    r = _run([_computed(n) for n in nonces], {n: list(ref) for n in nonces}, nonces)
    assert r["n_mismatch"] == 4 and r["fraud_detected"]
    assert all(p["malformed_reference"] for p in r["per_nonce"])


@pytest.mark.parametrize("n_steps", [0, 1, STEPS - 1])
def test_short_computed_trajectory_leaves_the_sample(n_steps):
    nonces = list(range(4))
    computed = [_computed(0, n_steps=n_steps)] + [_computed(n) for n in nonces[1:]]
    r = _run(computed, {n: [3] * STEPS for n in nonces}, nonces)
    assert r["excluded_nonces"] == [0] and r["n_excluded"] == 1 and r["n_total"] == 3


def test_nothing_scorable_is_not_a_verdict():
    nonces = list(range(4))
    with pytest.raises(ValueError):
        _run([_computed(n, n_steps=0) for n in nonces], {n: [3] * STEPS for n in nonces},
             nonces)
