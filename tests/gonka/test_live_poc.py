"""Live integration tests: decode-PoC (the only scheme of this release).

Requires a running vLLM server on port 18199.

Tests:
  1. /generate wait=true returns k-trajectories (max_tokens+1 snaps per nonce)
  2. Self-validation: teacher-forcing our own trajectories -> 0 mismatches
  3. Different block_hash -> different k-trajectories
  4. Batch generation covers every requested nonce exactly once
  5. Server-side validation via validation.artifacts[].k_points_steps
  6. max_tokens=0 degenerates to a single prefill snap through the same loop

Bit-exactness note: 0-mismatch self-validation holds WITHIN one server
process (strict determinism); across processes/nodes only the margin gate
(tau) is meaningful — see NOTES.md of the migration bundle.
"""
import httpx
import pytest

from tests.gonka.live_conftest import BASE_URL, MODEL, require_server, stop_poc

# Small profile: fast on any single GPU; route_window=256 matches the shipped
# engine profile (on 256-expert MoEs it selects legacy full scatter).
POC_PARAMS = {"model": MODEL, "seq_len": 64, "k_dim": 12,
              "max_tokens": 16, "route_window": 256}

POC_BASE = {
    "block_hash": "TEST_BLOCK",
    "block_height": 100,
    "public_key": "test_pub_keys",
    "node_id": 0,
    "node_count": 1,
}


@pytest.fixture(scope="module", autouse=True)
def server_ready():
    require_server()
    stop_poc()
    yield
    stop_poc()


def poc_generate(nonces, block_hash="TEST_BLOCK", wait=True, batch_size=4,
                 validation=None, enforced_k_steps=None, params=None,
                 timeout=300):
    body = {
        **POC_BASE,
        "block_hash": block_hash,
        "nonces": nonces,
        "params": params or POC_PARAMS,
        "batch_size": batch_size,
        "wait": wait,
    }
    if validation:
        body["validation"] = validation
    if enforced_k_steps is not None:
        body["enforced_k_steps"] = enforced_k_steps
    return httpx.post(
        f"{BASE_URL}/api/v1/pow/generate", json=body, timeout=timeout
    )


def _trajectories(response):
    arts = response.json()["artifacts"]
    return {a["nonce"]: a["k_points_steps"] for a in arts}


class TestDecodePoC:

    def test_01_generate_returns_trajectories(self):
        """wait=true returns a full k-trajectory per nonce."""
        r = poc_generate(nonces=[0, 1, 2, 3])
        assert r.status_code == 200, f"Generate failed: {r.text}"
        traj = _trajectories(r)
        assert set(traj) == {0, 1, 2, 3}
        want_len = POC_PARAMS["max_tokens"] + 1
        for nonce, k_steps in traj.items():
            assert len(k_steps) == want_len, (nonce, len(k_steps))
            assert all(0 <= k < 16 for k in k_steps), (nonce, k_steps[:5])
        enc = r.json()["encoding"]
        assert enc["route_window"] == POC_PARAMS["route_window"]
        assert enc["max_tokens"] == POC_PARAMS["max_tokens"]

    def test_02_self_validation_zero_mismatch(self):
        """Teacher-forcing our own trajectories must give 0 mismatches
        (strict in-process determinism)."""
        gen = poc_generate(nonces=[0, 1, 2, 3])
        assert gen.status_code == 200
        traj = _trajectories(gen)
        val = poc_generate(
            nonces=[0, 1, 2, 3],
            enforced_k_steps={str(n): t for n, t in traj.items()})
        assert val.status_code == 200, val.text
        data = val.json()
        assert data["n_mismatch"] == 0, data
        assert data["n_total"] == 4 * (POC_PARAMS["max_tokens"] + 1)
        assert data["fraud_detected"] is False

    def test_03_different_block_hash_different_trajectories(self):
        r1 = poc_generate(nonces=[0])
        r2 = poc_generate(nonces=[0], block_hash="OTHER_BLOCK")
        assert r1.status_code == 200 and r2.status_code == 200
        assert _trajectories(r1)[0] != _trajectories(r2)[0]

    def test_04_batch_generation(self):
        nonces = list(range(8))
        r = poc_generate(nonces=nonces, batch_size=4)
        assert r.status_code == 200
        traj = _trajectories(r)
        assert set(traj) == set(nonces)

    def test_05_server_side_validation_via_artifacts(self):
        """validation.artifacts[].k_points_steps is the wire form the
        validator sends; it must be picked up as the teacher-forcing
        reference."""
        gen = poc_generate(nonces=[0, 1])
        assert gen.status_code == 200
        traj = _trajectories(gen)
        val = poc_generate(
            nonces=[0, 1],
            validation={"artifacts": [
                {"nonce": n, "vector_b64": "", "k_points_steps": t}
                for n, t in traj.items()]})
        assert val.status_code == 200, val.text
        data = val.json()
        assert data["n_mismatch"] == 0, data
        assert data["fraud_detected"] is False

    def test_06_prefill_only_degenerate(self):
        """max_tokens=0: one snap per nonce through the same decode loop."""
        params = dict(POC_PARAMS, max_tokens=0)
        r = poc_generate(nonces=[0, 1], params=params)
        assert r.status_code == 200, r.text
        traj = _trajectories(r)
        assert all(len(t) == 1 for t in traj.values())
