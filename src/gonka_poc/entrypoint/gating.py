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
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("gonka_poc.entrypoint.gating")


# Paths that PoC priority blocks. Keep this list narrow -- /gonka/poc/* and
# /api/v1/pow/* MUST always be reachable (otherwise we deadlock).
DEFAULT_BLOCKED_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
    "/api/v1/infval/",
)


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

        await self.app(scope, receive, send)

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
    "install_gating_middleware",
]
