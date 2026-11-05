"""Unit tests for decode-PoC under pipeline parallelism.

prev_k (the chain value) is produced by the last rank only, but every rank
consumes it when it builds the decode-step inputs. The bridge
therefore broadcasts it from the last rank at sample time (``extract``), the
other ranks join through ``pp_recv_prev_k``, and those ranks, which never reach
``extract``, free a finished request's decode state themselves.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")  # the bridge imports vllm.logger / vllm.distributed
import torch  # noqa: E402

from gonka_poc.mixed.bridge import PoCRunnerBridge  # noqa: E402
from gonka_poc.mixed.runtime import _cat_prev_k, get_decode_manager  # noqa: E402

SEQ_LEN = 4
MAX_TOKENS = 3


class FakePPGroup:
    def __init__(self, rank: int, world_size: int, reply=None):
        self.world_size = world_size
        self.ranks = list(range(world_size))
        self.is_last_rank = rank == world_size - 1
        self.is_first_rank = rank == 0
        self.reply = reply
        self.calls: list[tuple[dict | None, int]] = []

    def broadcast_tensor_dict(self, tensor_dict=None, src=0):
        self.calls.append((tensor_dict, src))
        if self.is_last_rank:
            return tensor_dict
        return self.reply(len(self.calls))


def _bridge():
    return PoCRunnerBridge(SimpleNamespace(device="cpu"))


def _params(nonce, max_tokens=MAX_TOKENS):
    return SimpleNamespace(nonce=nonce, seq_len=SEQ_LEN, max_tokens=max_tokens,
                           enforced_k_steps=None)


def _new_step(rows):
    """rows: (req_id, nonce) pairs, all on their prefill step."""
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=rid, poc_params=_params(n),
                            num_computed_tokens=0)
            for rid, n in rows],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        finished_req_ids=set(),
        num_scheduled_tokens={rid: SEQ_LEN for rid, _ in rows},
    )


def _cached_step(req_ids, num_computed, finished=()):
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(req_ids), num_computed_tokens=list(num_computed)),
        finished_req_ids=set(finished),
        num_scheduled_tokens={rid: 1 for rid in req_ids},
    )


def _use_group(monkeypatch, group):
    import vllm.distributed
    from gonka_poc.mixed import runtime

    monkeypatch.setattr(vllm.distributed, "get_pp_group", lambda: group)
    # The k computation is not under test here.
    monkeypatch.setattr(runtime, "process_poc_outputs_from_hidden",
                        lambda runner, hidden, metadata: {})


def _sample(bridge, group):
    """The sample-time hook points: extract on the last rank, pp_recv_prev_k
    elsewhere (the runner calls the latter on every non-last rank)."""
    bridge._step["poc_metadata"] = []
    if group.is_last_rank:
        bridge.extract(None)
    bridge.pp_recv_prev_k()


def test_pp_first_rank_receives_prev_k(monkeypatch):
    group = FakePPGroup(
        rank=0, world_size=2,
        reply=lambda n: {"prev_k": torch.tensor([12, 41], dtype=torch.int64)})
    _use_group(monkeypatch, group)
    bridge = _bridge()
    mgr = get_decode_manager(bridge.runner)
    # Rows listed in reverse nonce order on purpose.
    bridge.pre_step(_new_step([("a", 7), ("b", 3)]))
    assert group.calls == []  # pre_step makes no collective
    bridge.pp_recv_prev_k()  # the prefill step hands over its prefill k
    assert len(group.calls) == 1
    payload, src = group.calls[0]
    assert payload is None and src == 1
    # nonce 3 ("b") is first in the payload, nonce 7 ("a") second.
    assert torch.equal(mgr.get("b").prev_k_t, torch.tensor([12]))
    assert torch.equal(mgr.get("a").prev_k_t, torch.tensor([41]))
    out = _cat_prev_k([mgr.get("b"), mgr.get("a")], "test")
    assert torch.equal(out, torch.tensor([12, 41]))
    bridge.pp_recv_prev_k()  # second call in one step: nothing left to receive
    assert len(group.calls) == 1


def test_pp_last_rank_broadcasts_decode_rows_in_nonce_order(monkeypatch):
    group = FakePPGroup(rank=1, world_size=2)
    _use_group(monkeypatch, group)
    bridge = _bridge()
    mgr = get_decode_manager(bridge.runner)
    bridge.pre_step(_new_step([("a", 7), ("b", 3)]))
    mgr.get("a").prev_k_t = torch.tensor([41])
    mgr.get("b").prev_k_t = torch.tensor([12])
    _sample(bridge, group)
    assert len(group.calls) == 1
    payload, src = group.calls[0]
    assert src == 1
    assert payload["prev_k"].dtype == torch.int64
    assert payload["prev_k"].tolist() == [12, 41]

    # The final decode step has no decode step after it: nothing to hand over.
    bridge.pre_step(_cached_step(["a", "b"], [SEQ_LEN + MAX_TOKENS - 1] * 2))
    _sample(bridge, group)
    assert len(group.calls) == 1


def test_pp_non_last_rank_frees_decode_state(monkeypatch):
    group = FakePPGroup(
        rank=0, world_size=2,
        reply=lambda n: {"prev_k": torch.tensor([5], dtype=torch.int64)})
    _use_group(monkeypatch, group)
    bridge = _bridge()
    mgr = get_decode_manager(bridge.runner)
    bridge.pre_step(_new_step([("a", 7)]))
    _sample(bridge, group)
    assert mgr.get("a") is not None
    for k in range(MAX_TOKENS):
        bridge.pre_step(_cached_step(["a"], [SEQ_LEN + k]))
        _sample(bridge, group)
        assert mgr.get("a") is not None
    # The scheduler reports the request finished one step later.
    bridge.pre_step(_cached_step([], [], finished={"a"}))
    assert mgr.get("a") is None
    assert "a" not in bridge._reqs
    # prefill step + the first MAX_TOKENS - 1 decode steps hand over.
    assert len(group.calls) == MAX_TOKENS


def test_single_rank_never_broadcasts(monkeypatch):
    group = FakePPGroup(rank=0, world_size=1)
    _use_group(monkeypatch, group)
    bridge = _bridge()
    mgr = get_decode_manager(bridge.runner)
    bridge.pre_step(_new_step([("a", 7)]))
    mgr.get("a").prev_k_t = torch.tensor([9])
    _sample(bridge, group)
    for k in range(MAX_TOKENS):
        bridge.pre_step(_cached_step(["a"], [SEQ_LEN + k]))
        _sample(bridge, group)
    assert group.calls == []
    assert torch.equal(mgr.get("a").prev_k_t, torch.tensor([9]))
    # With one rank extract() owns freeing; pre_step leaves the state alone.
    bridge.pre_step(_cached_step([], [], finished={"a"}))
    assert mgr.get("a") is not None
