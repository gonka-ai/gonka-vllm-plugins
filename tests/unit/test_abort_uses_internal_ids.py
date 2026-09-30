"""Aborting in-flight requests must actually abort them.

``abort_all_requests`` enumerates ids from
``output_processor.request_states``. Those keys are **internal** request ids
-- randomly generated when the EngineCoreRequest is built -- not the external
ids a caller passed to the API.

``AsyncLLM.abort(request_id, internal=False)`` resolves its argument through
the external->internal map. Handed an internal id it finds no entry, returns
normally, and aborts nothing. The failure is silent: no exception, so the
caller's success counter still incremented and the log still reported the
requests as aborted. PoC would then start its forward pass while inference
requests were still decoding on the same GPU.

CPU-only: the engine client is a stand-in that reproduces vLLM's id
resolution, so no GPU, no model, no vllm import.
"""

from __future__ import annotations

import asyncio

from gonka_poc._compat.v0_25 import abort_all_requests


def _run(coro):
    """Run a coroutine on a private loop.

    Matches tests/unit/test_exception_paths.py: pytest-asyncio is in the test
    extra but not assumed, so these run on a stripped-down install too.
    """
    return asyncio.run(coro)


class _FakeOutputProcessor:
    def __init__(self, external_to_internal: dict[str, list[str]]) -> None:
        self.external_req_ids = dict(external_to_internal)
        # request_states is keyed by INTERNAL id -- this is what the code
        # under test enumerates.
        self.request_states = {
            internal: object()
            for internals in external_to_internal.values()
            for internal in internals
        }


class _FakeAsyncLLM:
    """Reproduces AsyncLLM.abort's id resolution, including the no-op case."""

    def __init__(self, external_to_internal: dict[str, list[str]]) -> None:
        self.output_processor = _FakeOutputProcessor(external_to_internal)
        self.actually_aborted: list[str] = []
        self.calls = 0

    async def abort(self, request_id, internal: bool = False) -> None:
        self.calls += 1
        op = self.output_processor
        ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for rid in ids:
            if internal:
                if rid in op.request_states:
                    self.actually_aborted.append(rid)
                continue
            # External path: look the id up in the external->internal map. An
            # internal id is not a key there, so nothing is aborted -- quietly.
            for internal_id in op.external_req_ids.get(rid, []):
                self.actually_aborted.append(internal_id)


def test_in_flight_requests_are_really_aborted() -> None:
    """Every enumerated request must end up aborted, not just counted."""
    client = _FakeAsyncLLM({"ext-1": ["int-1a", "int-1b"], "ext-2": ["int-2a"]})

    aborted = _run(abort_all_requests(client))

    assert sorted(client.actually_aborted) == ["int-1a", "int-1b", "int-2a"], (
        "requests were reported as aborted but never left the engine"
    )
    assert aborted == 3, "the returned count must reflect real aborts"
    assert client.calls == 1, "all ids must go to the engine in one abort call"


def test_client_without_internal_parameter_still_works() -> None:
    """A client whose abort() predates the parameter must not raise.

    The EngineClient ABC declares abort() without it, so a non-AsyncLLM
    implementation is possible. Degrading is acceptable; a TypeError on
    every id is not.
    """

    class _LegacyClient(_FakeAsyncLLM):
        async def abort(self, request_id: str) -> None:  # no internal kwarg
            for internal_id in self.output_processor.external_req_ids.get(
                request_id, []
            ):
                self.actually_aborted.append(internal_id)

    client = _LegacyClient({"ext-1": ["int-1a"]})

    aborted = _run(abort_all_requests(client))

    assert isinstance(aborted, int)
