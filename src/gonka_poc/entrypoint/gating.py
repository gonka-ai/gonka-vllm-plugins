"""PoCGatingMiddleware -- 503 + abort-in-flight on PoC priority gate.

Modeled on ``vllm/entrypoints/serve/elastic_ep/middleware.py:ScalingMiddleware``
(v0.23.0). When PoC generation is active, new chat/completion requests get a
503 response BEFORE reaching any vLLM handler.

In-flight aborts: a Starlette middleware cannot directly cancel a
``StreamingResponse`` that has already begun yielding bytes. The companion
shutdown path is the ``compat.abort_all_requests(engine_client)`` calls
issued from ``gonka_poc.poc.routes`` (init_generate; per-chunk legacy
re-drain) and ``gonka_poc.poc.reservation`` before GPU work starts. The
middleware is purely a "don't accept new work" gate.

What it can do for requests already in flight is rewrite the reply: vLLM
returns an aborted request as a normal completion with finish_reason "abort",
and the middleware turns that into a 503 error envelope (see
``PoCGatingMiddleware``).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("gonka_poc.entrypoint.gating")


# Paths that PoC priority blocks. Keep this list narrow -- /gonka/poc/* and
# /api/v1/pow/* MUST always be reachable (otherwise we deadlock).
DEFAULT_BLOCKED_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
)

# vLLM finishes a request with this reason when it is aborted (PoC gate
# abort_all_requests, KV-lease abort) and still returns it as a normal
# completion. It is never a complete answer, so blocked paths turn it into an
# error envelope.
ABORT_FINISH_REASON = "abort"
ABORT_ERROR_TYPE = "poc_preempted"
_ABORT_ERROR_MESSAGE = (
    "request aborted because the node was preempted by PoC; retry later"
)


def _abort_envelope() -> dict[str, Any]:
    return {
        "error": {
            "message": _ABORT_ERROR_MESSAGE,
            "type": ABORT_ERROR_TYPE,
            "param": None,
            "code": 503,
        }
    }


def _has_abort_choice(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(c, dict) and c.get("finish_reason") == ABORT_FINISH_REASON
        for c in choices
    )


def _event_is_abort(event: bytes) -> bool:
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if _has_abort_choice(payload):
            return True
    return False


def _is_event_stream(start: Message) -> bool:
    for name, value in start.get("headers", []):
        if name.lower() == b"content-type":
            return value.lower().startswith(b"text/event-stream")
    return False


class PoCGate:
    """Process-local PoC priority flag.

    Toggled by the PoC router when generation starts / stops. Lives on
    ``app.state.gonka_gate`` so the middleware can read it without import
    cycles. Thread-safe enough for asyncio (single-writer pattern: the PoC
    router serialises start/stop).
    """

    # Single source of truth for the 503 retry hint. The HTTP ``Retry-After``
    # header is delta-seconds (RFC 9110 section 10.2.3) and the JSON body
    # carries the same value in milliseconds. Keeping both derived from one
    # constant prevents the 10x drift that confused orchestrators in the
    # original code (header "1" second vs body "100" ms).
    RETRY_AFTER_SECONDS: int = 1

    def __init__(self) -> None:
        self._active: bool = False
        self._reason: Optional[str] = None

    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str = "poc-generation") -> None:
        self._active = True
        self._reason = reason

    def deactivate(self) -> None:
        self._active = False
        self._reason = None

    @property
    def reason(self) -> Optional[str]:
        return self._reason


class PoCGatingMiddleware:
    """ASGI middleware. When the gate is active, return 503 for blocked paths.

    Starlette wraps middlewares in REVERSE insertion order, so installing this
    AFTER ``vllm.entrypoints.openai.api_server.build_app(...)`` returns puts it
    OUTERMOST -- it runs before any vLLM handler.

    Doubles as the carrier for the one-shot "plugin loaded but no gate
    attached" warning. We used to register that via ``@app.on_event("startup")``
    but vLLM's ``build_app`` constructs ``FastAPI(lifespan=lifespan)`` (see
    v0.23.0 ``api_server.py``), and Starlette silently ignores ``on_event``
    handlers when a ``lifespan`` is supplied (FastAPI has deprecated this
    pattern since 0.93). To keep the warning fail-loud across both
    ``vllm serve`` and ``gonka-vllm-serve``, we now perform the check on the
    first HTTP dispatch through this middleware.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        gate: PoCGate,
        blocked_prefixes: Iterable[str] = DEFAULT_BLOCKED_PREFIXES,
    ) -> None:
        self.app = app
        self.gate = gate
        self.blocked_prefixes = tuple(blocked_prefixes)
        self._gate_check_done: bool = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            # websockets / lifespan: pass through unchanged
            await self.app(scope, receive, send)
            return

        # One-shot gate-presence check on first HTTP request -- see the class
        # docstring for why this cannot be a startup event.
        if not self._gate_check_done:
            self._gate_check_done = True
            self._maybe_warn_missing_gate(scope)

        if self.gate.is_active():
            path: str = scope.get("path", "")
            if any(path.startswith(prefix) for prefix in self.blocked_prefixes):
                # Header (delta-seconds) and body (ms) both derive from
                # PoCGate.RETRY_AFTER_SECONDS -- see its comment for why.
                retry_after_seconds = PoCGate.RETRY_AFTER_SECONDS
                response = JSONResponse(
                    status_code=503,
                    content={
                        "error": "poc_generation_active",
                        "reason": self.gate.reason or "poc-generation",
                        "retry_after_ms": retry_after_seconds * 1000,
                    },
                    headers={"Retry-After": str(retry_after_seconds)},
                )
                await response(scope, receive, send)
                return

        path = scope.get("path", "")
        if any(path.startswith(prefix) for prefix in self.blocked_prefixes):
            await self.app(scope, receive, self._abort_rewriting_send(scope, send))
            return

        await self.app(scope, receive, send)

    def _abort_rewriting_send(self, scope: Scope, send: Send) -> Send:
        """Wrap ``send`` so an aborted completion reaches the client as an error.

        Runs regardless of gate state: the KV-lease abort can fire with the
        gate inactive. Non-stream replies are buffered until complete; stream
        replies are forwarded event by event and cut at the first aborted one.
        """
        path: str = scope.get("path", "")
        start: Optional[Message] = None
        is_stream = False
        pending: list[Message] = []  # non-stream: start + body messages
        tail = b""  # stream: incomplete trailing event
        closed = False

        async def error_response() -> None:
            retry_after = PoCGate.RETRY_AFTER_SECONDS
            response = JSONResponse(
                status_code=503,
                content=_abort_envelope(),
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, _no_receive, send)

        async def wrapped(message: Message) -> None:
            nonlocal start, is_stream, tail, closed
            if closed:
                return
            mtype = message["type"]

            if mtype == "http.response.start":
                start = message
                is_stream = _is_event_stream(message)
                if is_stream:
                    await send(message)
                else:
                    pending.append(message)
                return

            if mtype != "http.response.body" or start is None:
                await send(message)
                return

            more = message.get("more_body", False)
            body = message.get("body", b"")

            if not is_stream:
                pending.append(message)
                if more:
                    return
                raw = b"".join(m.get("body", b"") for m in pending[1:])
                try:
                    aborted = _has_abort_choice(json.loads(raw))
                except ValueError:
                    aborted = False
                if aborted:
                    logger.info(
                        "PoC abort: rewrote aborted json response on %s", path
                    )
                    pending.clear()
                    await error_response()
                    return
                replay = list(pending)
                pending.clear()
                for m in replay:
                    await send(m)
                return

            buf = tail + body
            parts = buf.split(b"\n\n")
            tail = parts.pop()
            out = []
            for part in parts:
                if _event_is_abort(part):
                    logger.info(
                        "PoC abort: rewrote aborted stream response on %s", path
                    )
                    envelope = json.dumps(_abort_envelope()).encode()
                    out.append(b"data: " + envelope + b"\n\ndata: [DONE]\n\n")
                    closed = True
                    tail = b""
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"".join(out),
                            "more_body": False,
                        }
                    )
                    return
                out.append(part + b"\n\n")
            if not more:
                out.append(tail)
                tail = b""
            data = b"".join(out)
            if data or not more:
                await send(
                    {
                        "type": "http.response.body",
                        "body": data,
                        "more_body": more,
                    }
                )

        return wrapped

    def _maybe_warn_missing_gate(self, scope: Scope) -> None:
        """Log a warning if the plugin loaded but no gate is attached.

        Reads ``app.state.gonka_gate`` via the ASGI scope. The middleware sits
        OUTERMOST so ``scope["app"]`` is the FastAPI instance built by
        ``vllm.entrypoints.openai.api_server.build_app``. In the
        ``gonka-vllm-serve`` happy path this is a no-op because the gate IS
        present; in the bare ``vllm serve`` accident it fires loudly.
        """
        try:
            import gonka_poc

            plugin_loaded = bool(getattr(gonka_poc, "PLUGIN_LOADED", False))
            if not plugin_loaded:
                return

            app = scope.get("app")
            state = getattr(app, "state", None)
            gate_present = getattr(state, "gonka_gate", None) is not None
            if not gate_present:
                logger.warning(
                    "gonka_poc plugin loaded but app.state.gonka_gate is missing -- "
                    "PoC chat-completion gating is DISABLED. "
                    "Did you launch with `vllm serve` instead of `gonka-vllm-serve`?"
                )
        except Exception:  # pragma: no cover - defensive
            # Never let the diagnostic crash a real request.
            logger.exception("gonka_poc: gate-presence warning check raised")


async def _no_receive() -> Message:  # pragma: no cover - never awaited
    return {"type": "http.disconnect"}


def install_gating_middleware(
    app: Starlette,
    gate: PoCGate,
    blocked_prefixes: Iterable[str],
) -> None:
    """Install :class:`PoCGatingMiddleware` on an already-built app.

    Starlette 1.3.x (shipped with vLLM 0.23) finalizes the middleware stack
    eagerly, so ``add_middleware()`` after upstream ``build_app`` returns
    raises "Cannot add middleware after an application has started". Reset the
    stack so Starlette rebuilds it (incorporating ``PoCGatingMiddleware``) on
    first dispatch -- safe because both callers run before ``serve_http``
    (no real request yet) and resetting a None stack is a no-op.
    """
    app.middleware_stack = None
    app.add_middleware(
        PoCGatingMiddleware, gate=gate, blocked_prefixes=blocked_prefixes
    )


__all__ = [
    "PoCGate",
    "PoCGatingMiddleware",
    "DEFAULT_BLOCKED_PREFIXES",
    "ABORT_FINISH_REASON",
    "ABORT_ERROR_TYPE",
    "install_gating_middleware",
]
