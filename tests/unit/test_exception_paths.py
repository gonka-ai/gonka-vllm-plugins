"""Exception-path unit tests for ``init_generate``'s gate-cleanup contract.

``init_generate`` flips the PoC gate ON *before* it has a generation task
whose done-callback would flip it OFF. If anything between
``gate.activate(...)`` and ``gen_task.add_done_callback(...)`` raises:

    gate.activate("init-generate")           # ON
    compat = _compat_current()               # <-- may raise
    aborted = await compat.abort_all_requests(...)  # <-- may raise
    gen_task = asyncio.create_task(...)      # <-- may raise

...the gate would latch ON forever -- no task exists yet, so no
done-callback ever fires the corresponding ``gate.deactivate()``. The HTTP
server would then 503 every chat/completion request from that point on,
with no way back. The production code therefore wraps the three offending
calls in a try/except that calls ``gate.deactivate()`` on any exception
before re-raising. These tests pin that contract:

  * test_init_generate_compat_raises_deactivates_gate
        ``gonka_poc.poc.routes._compat_current`` raises RuntimeError.
        Expected: gate ends up DEACTIVATED, exception propagates.

  * test_init_generate_abort_raises_deactivates_gate
        ``compat.abort_all_requests`` raises RuntimeError.
        Expected: gate ends up DEACTIVATED, exception propagates.

  * test_init_generate_spawn_raises_deactivates_gate
        ``asyncio.create_task`` raises RuntimeError (or the inner
        ``_generation_loop`` raises before yielding -- either path lands
        in the same try-block).
        Expected: gate ends up DEACTIVATED, exception propagates.

Why no ``vllm`` import / engine: the gate object is a pure-Python flag
defined in ``gonka_poc.entrypoint.gating``; we mount a minimal FastAPI
app with just enough state for ``_get_gate`` to find it. ``engine_client``
is a Mock that survives a single ``getattr(...)`` lookup. This keeps the
test CPU-only and < 1s.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

# ``gonka_poc.poc.routes`` transitively imports scipy via
# ``gonka_poc.poc.data`` (binomtest). scipy is a vllm transitive dep in
# production / CI, but a stripped-down dev install may skip it. Importing
# routes at module collection time would surface as a hard collection
# error -- skip the whole module instead so a thin install can still run
# the rest of ``tests/unit`` cleanly.
pytest.importorskip("scipy", reason="scipy required by gonka_poc.poc.data")

from gonka_poc.entrypoint.gating import PoCGate  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture: a wired app that ``init_generate`` can run against.
#
# init_generate's prerequisites (from src/gonka_poc/poc/routes.py):
#   * request.app.state.gonka_gate  -- PoCGate instance
#   * request.app.state.engine_client  -- anything truthy (we use AsyncMock)
#   * request.app.state.openai_serving_models / poc_deployed -- absent => skip
#     the check_params_match validation
# --------------------------------------------------------------------------- #


@pytest.fixture
def gate() -> PoCGate:
    """Fresh inactive gate for each test."""
    return PoCGate()


@pytest.fixture
def app(gate: PoCGate) -> FastAPI:
    """Minimal app carrying the state ``init_generate`` reads from request.app."""
    a = FastAPI()
    a.state.gonka_gate = gate
    # Truthy sentinel so ``get_engine_client`` returns it instead of 503ing.
    a.state.engine_client = AsyncMock()
    # Absent on purpose: ``check_params_match`` early-returns when
    # ``openai_serving_models`` and ``poc_deployed`` are both falsy.
    return a


def _make_request(app: FastAPI) -> Any:
    """Build a stand-in ``fastapi.Request`` whose only attribute we touch is
    ``.app``. ``init_generate`` reads ``request.app.state.*`` and nothing else
    from the Request object until generation actually runs, which these
    failure-mode tests never reach.
    """
    req = MagicMock()
    req.app = app
    return req


def _make_body() -> Any:
    """Mirror ``PoCInitGenerateRequest`` minimally -- a duck-typed dataclass
    is enough because ``check_params_match`` is a no-op against this app
    fixture (no ``openai_serving_models`` / ``poc_deployed`` set), so the
    body only has to expose the attributes the logging line reads BEFORE the
    gate flip:

        body.block_hash, body.block_height, body.public_key, body.node_id,
        body.node_count, body.group_id, body.n_groups, body.batch_size,
        body.params (with .model/.seq_len/.k_dim/.max_tokens/.route_window),
        body.url
    """
    params = MagicMock(model="m", seq_len=8, k_dim=12, max_tokens=4,
                       route_window=256)
    body = MagicMock(
        block_hash="bh",
        block_height=1,
        public_key="pk",
        node_id=0,
        node_count=1,
        group_id=0,
        n_groups=1,
        batch_size=1,
        params=params,
        url=None,
    )
    return body


def _run(coro):
    """Run an async coroutine to completion on a private event loop.

    We do NOT use ``pytest-asyncio`` so this file stays compatible with the
    ``pytest`` baseline declared in ``pyproject.toml``'s ``[project.optional-
    dependencies].test`` group (which DOES install pytest-asyncio but we want
    a hard floor: this test must run even on a stripped-down install).
    """
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_init_generate_compat_raises_deactivates_gate(
    app: FastAPI, gate: PoCGate
) -> None:
    """``_compat_current()`` raises => gate MUST be deactivated before the
    exception propagates.

    Failure here = the production try/except wrapper is missing or did not
    cover the compat-dispatch lookup. The HTTP server would 503 every chat
    request from this point forward.
    """
    from gonka_poc.poc import routes

    request = _make_request(app)
    body = _make_body()

    with patch.object(
        routes, "_compat_current", side_effect=RuntimeError("compat boom")
    ):
        with pytest.raises(RuntimeError, match="compat boom"):
            _run(routes.init_generate(request, body))

    assert gate.is_active() is False, (
        "Gate latched ON after _compat_current() raised. The production code "
        "must wrap activate -> _compat_current -> abort -> create_task in a "
        "try/except that calls gate.deactivate() on failure."
    )


def test_init_generate_abort_raises_deactivates_gate(
    app: FastAPI, gate: PoCGate
) -> None:
    """``compat.abort_all_requests`` raises => gate MUST be deactivated.

    Models: vllm's ``EngineClient.abort_request`` raising due to an engine
    shutdown race, OR the compat shim wrapping it with a guard that itself
    fails. Both surface as an exception out of ``abort_all_requests``.
    """
    from gonka_poc.poc import routes

    request = _make_request(app)
    body = _make_body()

    fake_compat = MagicMock()
    fake_compat.abort_all_requests = AsyncMock(
        side_effect=RuntimeError("abort boom")
    )

    with patch.object(routes, "_compat_current", return_value=fake_compat):
        with pytest.raises(RuntimeError, match="abort boom"):
            _run(routes.init_generate(request, body))

    assert gate.is_active() is False, (
        "Gate latched ON after abort_all_requests raised. "
        "Wrap activate -> abort -> create_task in try/except + deactivate."
    )


def test_init_generate_spawn_raises_deactivates_gate(
    app: FastAPI, gate: PoCGate
) -> None:
    """``asyncio.create_task`` raises => gate MUST be deactivated.

    Models the (rare) case where the running event loop is cancelled or
    the spawned coroutine raises synchronously during scheduling. Practically
    the same bug surface as the previous two tests, but tests the THIRD
    failure point in the activate -> abort -> spawn sequence -- the one
    closest to the happy path, where it is easiest to forget the wrapper.
    """
    from gonka_poc.poc import routes

    request = _make_request(app)
    body = _make_body()

    fake_compat = MagicMock()
    fake_compat.abort_all_requests = AsyncMock(return_value=0)

    with patch.object(routes, "_compat_current", return_value=fake_compat):
        with patch.object(
            routes.asyncio,
            "create_task",
            side_effect=RuntimeError("spawn boom"),
        ):
            with pytest.raises(RuntimeError, match="spawn boom"):
                _run(routes.init_generate(request, body))

    assert gate.is_active() is False, (
        "Gate latched ON after asyncio.create_task raised. The wrapper must "
        "cover the spawn call too -- not just the dispatch / abort steps."
    )


def test_init_generate_exception_cancels_callback_task(
    app: FastAPI, gate: PoCGate
) -> None:
    """body.url set + later step raises => callback_task MUST be torn down.

    ``init_generate`` must spawn ``CallbackSender.run()`` INSIDE the
    try-block and, in the except branch, signal stop_event + bounded-wait +
    cancel. Otherwise an orphan aiohttp loop keeps POSTing to ``body.url``
    until the process dies. This test pins that behaviour by asserting the
    callback task is ``done()`` (cancelled or naturally exited) after the
    exception unwinds. If the bug regresses the task will still be running
    when the assertion fires.

    Why patch ``CallbackSender``: a real ``CallbackSender.run()`` opens an
    ``aiohttp.ClientSession`` and starts hitting the network. We do not want
    real I/O in a unit test, so we replace the class with a tiny stand-in
    whose ``run()`` is a stop_event-respecting sleep loop -- enough to
    exercise the lifecycle the production fix protects.
    """
    from gonka_poc.poc import routes

    request = _make_request(app)
    body = _make_body()
    body.url = "http://test.invalid/cb"

    captured: dict = {}

    class _FakeCallbackSender:
        def __init__(self, callback_url, stop_event, k_dim=12, **kwargs):
            self.callback_url = callback_url
            self.stop_event = stop_event
            self.k_dim = k_dim
            captured["sender"] = self

        async def run(self):
            # Mirror the real run(): loop until stop_event fires. The
            # production except-branch sets stop_event then waits, so this
            # loop must terminate promptly when that happens.
            while not self.stop_event.is_set():
                await asyncio.sleep(0.01)

        def clear(self):
            pass

        def add_artifacts(self, *a, **kw):
            pass

    fake_compat = MagicMock()
    fake_compat.abort_all_requests = AsyncMock(
        side_effect=RuntimeError("abort boom")
    )

    with patch.object(routes, "CallbackSender", _FakeCallbackSender):
        with patch.object(routes, "_compat_current", return_value=fake_compat):
            with pytest.raises(RuntimeError, match="abort boom"):
                _run(routes.init_generate(request, body))

    assert gate.is_active() is False, (
        "Gate latched ON after abort_all_requests raised with body.url set."
    )

    # The fix sets stop_event in the except branch, so the FakeCallbackSender
    # run-loop must have exited. The wait_for inside init_generate awaits
    # the task, so by the time the exception propagates out, the task is
    # either cleanly done or cancelled.
    sender = captured.get("sender")
    assert sender is not None, (
        "CallbackSender was never instantiated -- the spawn moved out of "
        "the try-block? Production code must spawn callback_task INSIDE "
        "the try so the except path can tear it down."
    )
    assert sender.stop_event.is_set(), (
        "stop_event was not set in the except branch. The orphan aiohttp "
        "loop will keep POSTing to body.url until the process restarts. "
        "Fix: in the except, set stop_event before waiting on callback_task."
    )
