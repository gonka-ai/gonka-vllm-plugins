import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from gonka_poc.poc import generate_queue, routes
from gonka_poc.poc.data import encode_vector


@pytest.mark.parametrize("threshold_field", ["fraud_threshold", "p_value_threshold"])
@pytest.mark.parametrize("wait", [True, False])
@pytest.mark.parametrize("returned", [0, 199, 200])
def test_generate_route_honors_chain_statistics(monkeypatch, threshold_field, wait, returned):
    vectors = {}
    for nonce in range(200):
        vector = np.zeros(12, dtype=np.float32)
        vector[nonce % 12] = 1
        vectors[nonce] = encode_vector(vector)
    zero = encode_vector(np.zeros(12, dtype=np.float32))

    def artifacts(nonces):
        return [{"nonce": n, "vector_b64": vectors[n]} for n in nonces if n < returned]

    async def fake_chunk(engine_client, nonces, *args, **kwargs):
        return artifacts(nonces)

    async def fake_rpc(engine_client, nonces, **kwargs):
        return {"artifacts": artifacts(nonces)}

    @asynccontextmanager
    async def fake_reservation(*args, **kwargs):
        yield None

    class CapturingQueue:
        job = None

        def set_generation_active_check(self, check):
            pass

        async def enqueue(self, job):
            self.job = job
            return job.request_id

        async def ensure_worker_running(self, *args):
            pass

    captured = CapturingQueue()
    monkeypatch.setattr(routes, "get_queue", lambda: captured)
    monkeypatch.setattr(routes, "_compute_artifacts_chunk", fake_chunk)
    monkeypatch.setattr(routes, "_execute_poc_forward_rpc", fake_rpc)
    monkeypatch.setattr(routes, "poc_reservation", fake_reservation)
    monkeypatch.setattr(generate_queue, "poc_reservation", fake_reservation)

    body = routes.PoCGenerateRequest.model_validate({
        "block_hash": "hash", "block_height": 100, "public_key": "02ab",
        "node_id": 0, "node_count": 1, "nonces": list(range(200)),
        "params": {"model": "model-a", "seq_len": 256, "k_dim": 12},
        "batch_size": 64, "wait": wait,
        "url": "http://api/v2/poc-batches/model-a/v/200/200/token",
        "validation": {"artifacts": [
            {"nonce": n, "vector_b64": zero if n < 28 else vectors[n]}
            for n in range(200)
        ]},
        "stat_test": {
            "dist_threshold": 0.4, "p_mismatch": 0.1,
            threshold_field: 0.05, "min_count": 200,
        },
    })
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=object())))

    async def execute():
        result = await routes.generate(request, body)
        if not wait:
            assert captured.job.stat_test_fraud_threshold == 0.05
            assert captured.job.stat_test_min_count == 200
            queue = generate_queue.GenerateQueue()
            result = await queue._process_job(captured.job)
            sent = []

            class Recorder:
                def enqueue(self, url, path, payload):
                    sent.append(payload)

            queue._callback_queue = Recorder()
            queue._enqueue_callback(captured.job, result)
            result = sent[0]
        return result

    result = asyncio.run(execute())
    assert result["n_total"] == returned
    assert result["n_excluded"] == 200 - returned
    assert result["insufficient_sample"] is (returned < 200)
    assert result["fraud_detected"] is (returned == 200)
