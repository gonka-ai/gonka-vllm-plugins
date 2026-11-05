"""Call-order test of the decode-PoC prev_k hand-over on a REAL 2-rank PP group.

The fake group in test_decode_pp_sync.py cannot model ordering. The worker
(gpu_worker.execute_model) makes a non-first rank block in the receive of the
intermediate tensors BEFORE it runs the model runner, and the first rank sends
them only AFTER its runner returned. A collective anywhere inside the runner's
execute_model (pre_step) therefore deadlocks; the hand-over has to sit at sample
time, like vLLM's own sampled-token hand-over. This replays that order over gloo
with spawned processes.
"""
from __future__ import annotations

import multiprocessing as mp
import socket
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from gonka_poc.mixed.bridge import PoCRunnerBridge  # noqa: E402,F401

SEQ_LEN, MAX_TOKENS = 4, 4
TIMEOUT_S = 90
# last-rank prev_k per step for (req "a" nonce 7, req "b" nonce 3)
EXPECTED = {0: {"a": 41, "b": 12}, 1: {"a": 42, "b": 13}, 2: {"a": 43, "b": 14}}


def _params(nonce):
    return SimpleNamespace(nonce=nonce, seq_len=SEQ_LEN, max_tokens=MAX_TOKENS,
                           enforced_k_steps=None)


def _step(step):
    if step == 0:
        return SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="a", poc_params=_params(7),
                                num_computed_tokens=0),
                SimpleNamespace(req_id="b", poc_params=_params(3),
                                num_computed_tokens=0)],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], num_computed_tokens=[]),
            finished_req_ids=set(),
            num_scheduled_tokens={"a": SEQ_LEN, "b": SEQ_LEN})
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a", "b"],
            num_computed_tokens=[SEQ_LEN + step - 1] * 2),
        finished_req_ids=set(),
        num_scheduled_tokens={"a": 1, "b": 1})


def _worker(rank, port, q):
    from vllm.config import VllmConfig, set_current_vllm_config

    try:
        with set_current_vllm_config(VllmConfig()):
            _body(rank, port, q)
    except BaseException as e:  # report instead of dying silently
        q.put((rank, "error", repr(e)))
        raise


def _body(rank, port, q):
    import torch
    from vllm.distributed import parallel_state as ps
    from gonka_poc.mixed import runtime
    from gonka_poc.mixed.bridge import PoCRunnerBridge
    from gonka_poc.mixed.runtime import _cat_prev_k, get_decode_manager

    ps.init_distributed_environment(
        world_size=2, rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank, backend="gloo")
    ps.initialize_model_parallel(1, 2, backend="gloo")
    pp = ps.get_pp_group()
    bridge = PoCRunnerBridge(SimpleNamespace(device="cpu"))
    mgr = get_decode_manager(bridge.runner)
    current = {"step": 0}

    def fake_process(runner, hidden, metadata):
        # Stands in for the k computation: the last rank "computes" known k.
        for rid, k in EXPECTED[current["step"]].items():
            mgr.get(rid).prev_k_t = torch.tensor([k], dtype=torch.int64)
        return {}

    runtime.process_poc_outputs_from_hidden = fake_process

    send_handles = []
    for step in range(3):
        current["step"] = step
        so = _step(step)
        if pp.is_last_rank:
            # worker: receive BEFORE the runner; the model waits on the handles
            _, recv_handles, recv_post = pp.irecv_tensor_dict()
            bridge.pre_step(so)
            for h in recv_handles:
                h.wait()
            for fn in recv_post:
                fn()
            bridge._step["poc_metadata"] = []
            bridge.extract(None)  # sample_tokens on the last rank
        else:
            for h in send_handles:  # the worker reaps these at the next step
                h.wait()
            bridge.pre_step(so)  # inside the runner's execute_model
            if step:
                _cat_prev_k([mgr.get("b"), mgr.get("a")], "next-step inputs")
            send_handles = pp.isend_tensor_dict(  # AFTER the runner
                {"h": torch.zeros(2)})[1:]
            bridge.pp_recv_prev_k()  # sample_tokens early branch
            got = {rid: mgr.get(rid).prev_k_t.tolist() for rid in ("a", "b")}
            q.put((rank, "step", step, got))
    for h in send_handles:
        h.wait()
    q.put((rank, "done"))


def test_pp_real_group_worker_order_completes():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in (0, 1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(TIMEOUT_S)
    alive = [p.is_alive() for p in procs]
    for p in procs:
        if p.is_alive():
            p.kill()
    msgs = []
    while not q.empty():
        msgs.append(q.get())
    assert alive == [False, False], f"deadlock: alive={alive} msgs={msgs}"
    errors = [m for m in msgs if m[1] == "error"]
    assert not errors, errors
    assert sorted(m[0] for m in msgs if m[1] == "done") == [0, 1]
    steps = {m[2]: m[3] for m in msgs if m[1] == "step"}
    for step, expected in EXPECTED.items():
        assert steps[step] == {k: [v] for k, v in expected.items()}
