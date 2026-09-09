"""Unit tests for the PoC KV reservation layer (``gonka_poc.poc.reservation``).

Uses a fake engine client exposing the surfaces the layer touches:
``collective_rpc`` (legacy surface, no longer probed), ``engine_core.
call_utility_async`` (borrow/return), ``abort`` + ``output_processor``
(escalation / legacy safety), ``reset_prefix_cache``. vllm must be
importable (the compat dispatch resolves by installed version) but no
engine is started.

Probe-first flow: the first ``poc_reservation`` per engine client runs
``poc_validation_available`` — a ``collective_rpc("execute_poc_borrow_
compat")`` gate died with the prefill scheme; the probe is now a zero-block borrow RPC —
and caches the verdict for the client's lifetime.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("vllm")

from gonka_poc.poc import reservation  # noqa: E402


class FakeCore:
    def __init__(self, script):
        # script: list of returns for NON-probe borrow calls; an Exception
        # instance is raised instead. The zero-block availability probe
        # (args == (0, 1)) is answered from ``probe_response`` so tests
        # control it independently of the lease script.
        self.script = list(script)
        self.calls = []
        self.probe_response = None  # value or Exception

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == "gonka_poc_return_blocks":
            return None
        if args == (0, 1):
            if isinstance(self.probe_response, Exception):
                raise self.probe_response
            return self.probe_response
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeEngine:
    def __init__(self, script):
        self.engine_core = FakeCore(script)
        self.aborted = 0
        self.prefix_resets = 0

        # abort_all_requests enumerates output_processor.request_states
        # and calls .abort(rid); give it one in-flight request.
        class _OP:
            request_states = {"req-1": object()}
        self.output_processor = _OP()

    async def abort(self, request_id, **kwargs):
        self.aborted += 1

    async def reset_prefix_cache(self, reset_running_requests=False):
        self.prefix_resets += 1
        return True


LEASE = {"block_ids": [7, 3, 12, 5], "blocks_per_seq": 2}


def _borrow_calls(eng):
    return [c for c in eng.engine_core.calls
            if c[0] == "gonka_poc_borrow_blocks" and c[1] != (0, 1)]


def test_lease_yielded_and_returned():
    eng = FakeEngine([LEASE])

    async def go():
        async with reservation.poc_reservation(eng, 2, 32) as lease:
            assert lease == LEASE
        methods = [m for m, _ in eng.engine_core.calls]
        # probe zero-borrow, real borrow, return — in order
        assert methods == ["gonka_poc_borrow_blocks"] * 2 + [
            "gonka_poc_return_blocks"]
        assert eng.engine_core.calls[-1][1] == (LEASE["block_ids"],)

    asyncio.run(go())
    assert eng.aborted == 0
    assert eng.prefix_resets == 0


def test_probe_cached_across_reservations():
    eng = FakeEngine([LEASE, LEASE])

    async def go():
        async with reservation.poc_reservation(eng, 2, 32):
            pass
        async with reservation.poc_reservation(eng, 2, 32):
            pass

    asyncio.run(go())
    probes = [c for c in eng.engine_core.calls if c[1] == (0, 1)]
    assert len(probes) == 1  # second reservation reused the cached verdict


def test_probe_failure_disables_borrowing():
    # The zero-block borrow probe raising means no borrow surface: validation
    # degrades to the legacy abort path (lease is None).
    eng = FakeEngine([])
    eng.engine_core.probe_response = RuntimeError("borrow surface missing")

    async def go():
        async with reservation.poc_reservation(eng, 2, 32) as lease:
            assert lease is None

    asyncio.run(go())
    assert _borrow_calls(eng) == []   # never asked the pool for a lease
    assert eng.aborted == 1           # legacy path aborted inference
    assert eng.prefix_resets == 1     # in-place round drops the prefix cache


def test_blocks_returned_on_body_exception():
    eng = FakeEngine([LEASE])

    async def go():
        with pytest.raises(RuntimeError, match="boom"):
            async with reservation.poc_reservation(eng, 2, 32):
                raise RuntimeError("boom")

    asyncio.run(go())
    assert eng.engine_core.calls[-1][0] == "gonka_poc_return_blocks"


def test_pool_busy_escalates_through_abort_then_succeeds():
    # Pool stays busy (None) UNTIL inference is aborted — only the abort
    # escalation can free blocks, so a lease before eng.aborted would be a
    # logic error, not a lucky poll.
    eng = FakeEngine([])
    orig_pop = eng.engine_core.script  # unused; responder below instead
    del orig_pop

    async def call(method, *args):
        eng.engine_core.calls.append((method, args))
        if method == "gonka_poc_return_blocks":
            return None
        if args == (0, 1):
            return None
        return LEASE if eng.aborted else None
    eng.engine_core.call_utility_async = call

    async def go():
        async with reservation.poc_reservation(eng, 2, 32, timeout_ms=60) as lease:
            assert lease == LEASE

    asyncio.run(go())
    assert eng.aborted == 1  # escalation drained the in-flight request
    assert eng.prefix_resets == 0


def test_rpc_broken_falls_back_with_abort_and_prefix_reset():
    # Borrow RPC raises mid-run (engine died after the probe): no re-borrow,
    # inference aborted for the legacy in-place path, prefix cache reset on
    # exit because blocks 0..N were clobbered.
    eng = FakeEngine([AttributeError("no gonka_poc_borrow_blocks")])

    async def go():
        async with reservation.poc_reservation(eng, 2, 32, timeout_ms=1) as lease:
            assert lease is None

    asyncio.run(go())
    assert len(_borrow_calls(eng)) == 1  # broken RPC is not retried
    assert eng.aborted == 1              # legacy fallback is abort-protected
    assert eng.prefix_resets == 1        # poisoned prefix cache dropped


def test_return_retries_transient_failures():
    eng = FakeEngine([LEASE])
    attempts = {"n": 0}
    real = eng.engine_core.call_utility_async

    async def flaky(method, *args):
        if method == "gonka_poc_return_blocks":
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise ConnectionError("transient")
        return await real(method, *args)
    eng.engine_core.call_utility_async = flaky

    async def go():
        async with reservation.poc_reservation(eng, 2, 32):
            pass

    asyncio.run(go())
    assert attempts["n"] == 3  # 2 failures + 1 success
    assert eng.engine_core.calls[-1][0] == "gonka_poc_return_blocks"


def test_failed_reset_is_not_reported_as_success(caplog):
    # reset_prefix_cache returning False must escalate to
    # reset_running_requests=True and, if still False, log an ERROR.
    eng = FakeEngine([])
    eng.engine_core.probe_response = RuntimeError("borrow surface missing")
    reset_calls = []

    async def stubborn_reset(reset_running_requests=False):
        reset_calls.append(reset_running_requests)
        return False
    eng.reset_prefix_cache = stubborn_reset

    async def go():
        async with reservation.poc_reservation(eng, 2, 32):
            pass

    import logging
    with caplog.at_level(logging.ERROR, logger="gonka_poc.poc.reservation"):
        asyncio.run(go())
    assert reset_calls == [False, True]  # plain call, then escalation
    assert any("NOT reset" in r.message for r in caplog.records)


def test_reservation_lock_serializes():
    eng = FakeEngine([LEASE, LEASE])
    order = []

    async def user(tag, hold):
        async with reservation.poc_reservation(eng, 1, 16):
            order.append(f"{tag}-in")
            await asyncio.sleep(hold)
            order.append(f"{tag}-out")

    async def go():
        await asyncio.gather(user("a", 0.05), user("b", 0))

    asyncio.run(go())
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_versions_reports_probe_result():
    from types import SimpleNamespace
    from gonka_poc.poc.routes import get_versions

    async def go(engine):
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(engine_client=engine)))
        return await get_versions(request)

    on = asyncio.run(go(FakeEngine([])))
    assert on["poc_validation_inference"] is True
    assert "vllm_version" in on and "gonka_poc_version" in on

    eng_off = FakeEngine([])
    eng_off.engine_core.probe_response = RuntimeError("borrow surface missing")
    off = asyncio.run(go(eng_off))
    assert off["poc_validation_inference"] is False
