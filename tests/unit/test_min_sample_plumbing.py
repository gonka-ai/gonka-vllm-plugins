from __future__ import annotations

import pytest
from pydantic import ValidationError

pytest.importorskip("scipy", reason="scipy required by gonka_poc.poc.data")

from gonka_poc.poc.generate_queue import GenerateJob, GenerateQueue  # noqa: E402
from gonka_poc.poc.routes import StatTestModel  # noqa: E402

DAPI_STAT_TEST = {
    "dist_threshold": 0.4,
    "p_mismatch": 0.1,
    "fraud_threshold": 0.05,
    "min_count": 200,
}


def test_stat_test_model_reads_dapi_min_count():
    model = StatTestModel.model_validate(DAPI_STAT_TEST)
    assert model.min_count == 200
    assert model.dist_threshold == 0.4
    assert model.p_mismatch == 0.1
    assert model.fraud_threshold == 0.05


def test_stat_test_model_defaults_min_count_to_zero():
    payload = {k: v for k, v in DAPI_STAT_TEST.items() if k != "min_count"}
    assert StatTestModel.model_validate(payload).min_count == 0


def test_stat_test_model_rejects_negative_min_count():
    with pytest.raises(ValidationError):
        StatTestModel.model_validate({**DAPI_STAT_TEST, "min_count": -1})


class _RecordingCallbackQueue:
    def __init__(self):
        self.sent = []

    def enqueue(self, url, path, payload):
        self.sent.append((url, path, payload))


def _job(**overrides) -> GenerateJob:
    fields = dict(
        request_id="req-1",
        engine_client=None,
        app_id=0,
        block_hash="hash",
        block_height=100,
        public_key="02ab",
        node_id=1,
        node_count=1,
        nonces=[1, 2, 3, 4],
        seq_len=256,
        k_dim=12,
        batch_size=4,
        validation_artifacts={1: "AA=="},
        stat_test_min_count=200,
        callback_url="http://api:9100/v2/poc-batches/m/v/4/token",
    )
    fields.update(overrides)
    return GenerateJob(**fields)


@pytest.mark.parametrize("insufficient", [True, False])
def test_validated_callback_carries_insufficient_sample(insufficient: bool):
    queue = GenerateQueue()
    recorder = _RecordingCallbackQueue()
    queue._callback_queue = recorder

    queue._enqueue_callback(_job(), {
        "n_total": 4,
        "n_mismatch": 2,
        "mismatch_nonces": [1, 2],
        "p_value": 1.0,
        "fraud_detected": False,
        "insufficient_sample": insufficient,
    })

    assert len(recorder.sent) == 1
    url, path, payload = recorder.sent[0]
    assert url == "http://api:9100/v2/poc-batches/m/v/4/token"
    assert path == "validated"
    assert payload["insufficient_sample"] is insufficient
    assert payload["n_total"] == 4


def test_generate_job_min_count_defaults_to_zero():
    assert _job(stat_test_min_count=0).stat_test_min_count == 0
    fields = _job().__dict__.copy()
    fields.pop("stat_test_min_count")
    fields.pop("created_at")
    assert GenerateJob(**fields).stat_test_min_count == 0


@pytest.mark.parametrize("returned", [0, 4, 199])
def test_queue_abstains_when_runner_loses_sample(monkeypatch, returned: int):
    import contextlib

    import numpy as np

    from gonka_poc.poc import generate_queue, routes
    from gonka_poc.poc.data import encode_vector

    vectors = {}
    for nonce in range(200):
        v = np.zeros(12, dtype=np.float32)
        v[nonce % 12] = 1.0
        vectors[nonce] = encode_vector(v)

    async def fake_rpc(engine_client, nonces, **kwargs):
        kept = [n for n in nonces if n < returned]
        return {"artifacts": [{"nonce": n, "vector_b64": vectors[n]} for n in kept]}

    @contextlib.asynccontextmanager
    async def fake_reservation(*args, **kwargs):
        yield None

    monkeypatch.setattr(routes, "_execute_poc_forward_rpc", fake_rpc)
    monkeypatch.setattr(generate_queue, "poc_reservation", fake_reservation)

    queue = GenerateQueue()
    recorder = _RecordingCallbackQueue()
    queue._callback_queue = recorder
    job = _job(
        nonces=list(range(200)),
        batch_size=64,
        validation_artifacts=vectors,
        stat_test_dist_threshold=0.4,
        stat_test_p_mismatch=0.1,
        stat_test_fraud_threshold=0.05,
        stat_test_min_count=200,
    )

    import asyncio

    result = asyncio.run(queue._process_job(job))
    queue._enqueue_callback(job, result)

    _, _, payload = recorder.sent[0]
    assert payload["n_total"] == returned
    assert payload["n_excluded"] == 200 - returned
    assert payload["insufficient_sample"] is True
    assert payload["fraud_detected"] is False
