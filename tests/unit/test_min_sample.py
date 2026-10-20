from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="scipy required by gonka_poc.poc.data")

from gonka_poc.poc.data import encode_vector, fraud_test  # noqa: E402
from gonka_poc.poc.validation import run_validation  # noqa: E402

K_DIM = 12
CHAIN_STAT_TEST = {"dist_threshold": 0.4, "p_mismatch": 0.1, "fraud_threshold": 0.05}


def _unit(i: int) -> np.ndarray:
    v = np.zeros(K_DIM, dtype=np.float32)
    v[i % K_DIM] = 1.0
    return v


def _case(n_total: int, n_fabricated: int):
    computed = []
    received = {}
    for nonce in range(n_total):
        vec = _unit(nonce)
        computed.append({"nonce": nonce, "vector_b64": encode_vector(vec)})
        fabricated = nonce < n_fabricated
        received[nonce] = encode_vector(np.zeros(K_DIM, dtype=np.float32) if fabricated else vec)
    return computed, received


def _validate(n_total: int, n_fabricated: int, min_n: int = 0) -> dict:
    computed, received = _case(n_total, n_fabricated)
    return run_validation(computed, received, n_total=n_total, k_dim=K_DIM, min_n=min_n, **CHAIN_STAT_TEST)


def test_half_fabricated_small_sample_without_min_n():
    result = _validate(4, 2)
    assert result["n_mismatch"] == 2
    assert result["p_value"] == pytest.approx(0.0523, abs=1e-4)
    assert result["fraud_detected"] is False
    assert result["insufficient_sample"] is False

    assert _validate(4, 3)["fraud_detected"] is True


def test_half_fabricated_small_sample_with_min_n():
    result = _validate(4, 2, min_n=200)
    assert result["insufficient_sample"] is True
    assert result["fraud_detected"] is False
    assert result["p_value"] == 1.0
    assert result["n_mismatch"] == 2
    assert result["mismatch_nonces"] == [0, 1]
    assert result["n_total"] == 4


def test_min_n_never_reports_fraud_below_threshold():
    result = _validate(4, 4, min_n=200)
    assert result["insufficient_sample"] is True
    assert result["fraud_detected"] is False


@pytest.mark.parametrize("n_fabricated,fraud", [(27, False), (28, True)])
def test_boundary_at_min_n(n_fabricated: int, fraud: bool):
    result = _validate(200, n_fabricated, min_n=200)
    assert result["insufficient_sample"] is False
    assert result["fraud_detected"] is fraud


def test_fraud_test_min_n():
    assert fraud_test(2, 4, 0.1, 0.05, min_n=200) == (1.0, False)
    p_value, fraud = fraud_test(2, 4, 0.1, 0.05)
    assert p_value == pytest.approx(0.0523, abs=1e-4)
    assert fraud is False
    assert fraud_test(28, 200, 0.1, 0.05, min_n=200)[1] is True
    assert fraud_test(2, 4) == fraud_test(2, 4, min_n=0)


def test_lost_outputs_shrink_the_scored_sample():
    computed, received = _case(200, 0)
    result = run_validation(
        computed[:199], received, n_total=200, k_dim=K_DIM, min_n=200,
        requested_nonces=list(range(200)), **CHAIN_STAT_TEST,
    )
    assert result["n_total"] == 199
    assert result["n_excluded"] == 1
    assert result["excluded_nonces"] == [199]
    assert result["insufficient_sample"] is True
    assert result["fraud_detected"] is False


def test_empty_scored_sample_is_insufficient_without_min_n():
    _, received = _case(200, 0)
    result = run_validation(
        [], received, n_total=200, k_dim=K_DIM,
        requested_nonces=list(range(200)), **CHAIN_STAT_TEST,
    )
    assert result["n_total"] == 0
    assert result["insufficient_sample"] is True
    assert result["fraud_detected"] is False


def test_scored_sample_ignores_unrequested_and_duplicate_artifacts():
    computed, received = _case(200, 0)
    extra = {"nonce": 10_000, "vector_b64": computed[0]["vector_b64"]}
    result = run_validation(
        computed + [computed[0], extra], received, n_total=200, k_dim=K_DIM, min_n=200,
        requested_nonces=list(range(200)), **CHAIN_STAT_TEST,
    )
    assert result["n_total"] == 200
    assert result["n_excluded"] == 0
    assert result["insufficient_sample"] is False


def test_mismatch_rate_uses_scored_denominator():
    computed, received = _case(200, 4)
    result = run_validation(
        computed[:4], received, n_total=200, k_dim=K_DIM,
        requested_nonces=list(range(200)), **CHAIN_STAT_TEST,
    )
    assert result["n_total"] == 4
    assert result["n_mismatch"] == 4
    assert result["fraud_detected"] is True


def test_empty_reference_vectors_are_not_scored():
    computed, _ = _case(200, 0)
    empty_refs = {nonce: "" for nonce in range(200)}
    result = run_validation(
        computed, empty_refs, n_total=200, k_dim=K_DIM, min_n=200,
        requested_nonces=list(range(200)), **CHAIN_STAT_TEST,
    )
    assert result["n_total"] == 0
    assert result["n_excluded"] == 200
    assert result["insufficient_sample"] is True
    assert result["fraud_detected"] is False
