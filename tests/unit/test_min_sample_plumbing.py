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
        callback_url="http://api:9100/v2/poc-batches/m",
    )
    fields.update(overrides)
    return GenerateJob(**fields)


def _validation_result(insufficient: bool) -> dict:
    return {
        "n_total": 4,
        "n_mismatch": 2,
        "mismatch_nonces": [1, 2],
        "p_value": 1.0,
        "fraud_detected": False,
        "insufficient_sample": insufficient,
        "n_excluded": 196,
        "excluded_nonces": list(range(4, 200)),
    }


def test_validated_callback_sent_for_sufficient_sample():
    queue = GenerateQueue()
    recorder = _RecordingCallbackQueue()
    queue._callback_queue = recorder

    queue._enqueue_callback(_job(), _validation_result(False))

    assert len(recorder.sent) == 1
    url, path, payload = recorder.sent[0]
    assert url == "http://api:9100/v2/poc-batches/m"
    assert path == "validated"
    assert payload["n_total"] == 4
    assert payload["n_excluded"] == 196
    assert "insufficient_sample" not in payload


def test_no_validated_callback_for_insufficient_sample():
    queue = GenerateQueue()
    recorder = _RecordingCallbackQueue()
    queue._callback_queue = recorder

    queue._enqueue_callback(_job(), _validation_result(True))

    assert recorder.sent == []


def test_generate_job_min_count_defaults_to_zero():
    assert _job(stat_test_min_count=0).stat_test_min_count == 0
    fields = _job().__dict__.copy()
    fields.pop("stat_test_min_count")
    fields.pop("created_at")
    assert GenerateJob(**fields).stat_test_min_count == 0


@pytest.mark.parametrize("min_count,returned,callback", [
    (200, 0, False),
    (200, 4, False),
    (200, 199, False),
    (200, 200, True),
    (0, 0, False),
    (0, 4, True),
])
def test_queue_callback_when_runner_loses_sample(monkeypatch, min_count: int, returned: int, callback: bool):
    import asyncio
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
        stat_test_min_count=min_count,
    )

    result = asyncio.run(queue._process_job(job))
    queue._enqueue_callback(job, result)

    assert result["n_total"] == returned
    assert result["n_excluded"] == 200 - returned
    if not callback:
        assert recorder.sent == []
        return
    _, _, payload = recorder.sent[0]
    assert payload["n_total"] == returned
    assert payload["fraud_detected"] is False
