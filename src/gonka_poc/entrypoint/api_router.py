"""``gonka-vllm-serve`` -- compose a FastAPI app on top of stock vLLM (0.25.x).

This is a thin wrapper around the upstream public API:

    vllm.entrypoints.openai.api_server.setup_server
    vllm.entrypoints.openai.api_server.build_async_engine_client
    vllm.entrypoints.openai.api_server.build_app
    vllm.entrypoints.openai.api_server.init_app_state
    vllm.entrypoints.launcher.serve_http
    vllm.entrypoints.openai.cli_args.make_arg_parser
    vllm.entrypoints.openai.cli_args.validate_parsed_serve_args

We do NOT patch any vLLM source file. We only:
  1. Build the stock FastAPI app via ``build_app(args, ...)``.
  2. Leave the PoC router alone: ``build_app`` registers it itself in a vllm
     build carrying the PoC engine seams.
  3. Install ``PoCGatingMiddleware`` AFTER ``build_app`` returns so it ends up
     OUTERMOST in Starlette's reverse-insertion order, gating the
     ``/v1/chat/completions`` and ``/v1/completions`` routes with 503 when PoC
     is active.
  4. Forward to ``serve_http`` exactly like upstream's ``run_server_worker``.

Middleware ordering note (verified against v0.23.0
``vllm/entrypoints/openai/api_server.py:156-300``): user-supplied
``--middleware`` are added at L287-297; we add OURS AFTER ``build_app`` so we
sit outside them too. The chat-completion handler is reached only if the gate
is open.
"""
from __future__ import annotations

import logging
import signal
import sys
from typing import Any, Iterable, Optional

# fastapi import is cheap; vllm imports are deferred to ``main`` so that
# ``--help`` / argparse error paths don't fork the engine.
from fastapi import FastAPI

from gonka_poc.entrypoint.gating import (
    DEFAULT_BLOCKED_PREFIXES,
    PoCGate,
    install_gating_middleware,
)

logger = logging.getLogger("gonka_poc.entrypoint")


def _interrupt_init(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    """SIGTERM handler mirroring upstream ``api_server._interrupt_init``.

    Translates SIGTERM into a KeyboardInterrupt so the uvloop event loop
    unwinds cleanly via the same path as Ctrl-C.
    """
    raise KeyboardInterrupt("gonka-vllm-serve received SIGTERM")


def build_gonka_app(
    app: FastAPI,
    *,
    gate: PoCGate,
    blocked_prefixes: Optional[Iterable[str]] = None,
) -> FastAPI:
    """Mutate the upstream-built FastAPI app: add the gating middleware.

    The PoC router is NOT attached here: ``build_app`` registers it itself in
    a vllm build carrying the PoC engine seams. Attaching it again would put
    a second copy of every ``/api/v1/pow/*`` route on the app, via
    ``include_router`` -- the exact path the seam avoids because FastAPI's
    ``_IncludedRouter`` breaks prometheus route-name lookup.

    Args:
        app: the FastAPI instance returned by
            ``vllm.entrypoints.openai.api_server.build_app(args, ...)``.
        gate: the shared :class:`PoCGate` flag toggled by the PoC router.
        blocked_prefixes: optional override for the path prefixes the gating
            middleware 503s while PoC is active. ``None`` uses
            :data:`gonka_poc.entrypoint.gating.DEFAULT_BLOCKED_PREFIXES`.

    Returns:
        The same ``app`` instance (mutated). Returned for chainability.
    """
    # State for both the gating middleware AND the PoC router to read.
    app.state.gonka_gate = gate

    # Install the gating middleware LAST (so it runs FIRST per Starlette's
    # reverse-insertion ordering).
    prefixes = (
        tuple(blocked_prefixes)
        if blocked_prefixes is not None
        else DEFAULT_BLOCKED_PREFIXES
    )
    # Stack-reset + add_middleware pair lives in install_gating_middleware
    # (see its docstring for the Starlette 1.3.x rationale).
    install_gating_middleware(app, gate=gate, blocked_prefixes=prefixes)

    # The "plugin loaded but no gate attached" warning is carried by
    # PoCGatingMiddleware._maybe_warn_missing_gate (one-shot on first dispatch).

    return app


async def _run_server(args: Any) -> None:
    """Async body equivalent to ``vllm.entrypoints.openai.api_server.run_server``,
    but inserting PoC composition between ``build_app`` and ``serve_http``.

    Mirrors v0.23.0 ``api_server.py:559-604`` (``build_and_serve``).
    """
    # Deferred imports: keep ``gonka-vllm-serve --help`` fast and isolated
    # from CUDA fork issues.
    import vllm.envs as envs
    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        init_app_state,
        setup_server,
    )
    from vllm.entrypoints.launcher import serve_http

    # Best-effort: upstream pulls this from
    # ``vllm.entrypoints.serve.utils.server_utils`` to honour
    # ``--log-config-file`` / ``--disable-access-log-for-endpoints``. If the
    # internal module path moves in a future vLLM release, fall back to None
    # (uvicorn defaults) rather than crashing serve.
    try:
        from vllm.entrypoints.serve.utils.server_utils import (  # type: ignore[import-not-found]
            get_uvicorn_log_config,
        )
    except Exception:  # pragma: no cover - vllm internal layout drift
        get_uvicorn_log_config = None  # type: ignore[assignment]

    listen_address, sock = setup_server(args, reuse_port=False)

    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        # Stock vLLM app + middleware/handlers.
        app = build_app(args, supported_tasks, model_config)

        # Gonka composition: PoC router + gating middleware.
        gate = PoCGate()
        build_gonka_app(
            app,
            gate=gate,
            blocked_prefixes=getattr(args, "gonka_poc_block_prefixes", None),
        )

        # Standard upstream state population (sets app.state.engine_client and
        # app.state.openai_serving_*). MUST run after build_app and after we
        # add our middleware (Starlette freezes the stack on startup, which
        # serve_http triggers).
        await init_app_state(engine_client, app.state, args, supported_tasks)

        # Mirror upstream ``build_and_serve`` (v0.23.0 api_server.py:586-602).
        # Every kwarg upstream forwards MUST be forwarded here too -- missing
        # any of these silently drops user-supplied TLS / HTTP-limit / log
        # config flags. ``getattr(args, "...", None)`` keeps us robust to
        # upstream argparse changes: a removed flag falls back to None, which
        # ``serve_http`` already tolerates.
        log_config = None
        if get_uvicorn_log_config is not None:
            try:
                log_config = get_uvicorn_log_config(args)
            except Exception:  # pragma: no cover - defensive
                log_config = None

        serve_http_kwargs: dict[str, Any] = {
            "sock": sock,
            "enable_ssl_refresh": getattr(args, "enable_ssl_refresh", False),
            "host": args.host,
            "port": args.port,
            "log_level": getattr(args, "uvicorn_log_level", "info"),
            # disable_uvicorn_access_log == True  =>  access_log = False.
            "access_log": not getattr(args, "disable_uvicorn_access_log", False),
            "timeout_keep_alive": envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            "ssl_keyfile": getattr(args, "ssl_keyfile", None),
            "ssl_certfile": getattr(args, "ssl_certfile", None),
            "ssl_ca_certs": getattr(args, "ssl_ca_certs", None),
            "ssl_cert_reqs": getattr(args, "ssl_cert_reqs", 0),
            "ssl_ciphers": getattr(args, "ssl_ciphers", None),
            "h11_max_incomplete_event_size": getattr(
                args, "h11_max_incomplete_event_size", None
            ),
            "h11_max_header_count": getattr(args, "h11_max_header_count", None),
        }
        if log_config is not None:
            serve_http_kwargs["log_config"] = log_config

        # Hand off to uvicorn via the stock launcher.
        shutdown_task = await serve_http(app, **serve_http_kwargs)
        await shutdown_task

    sock.close()


def main(argv: list[str] | None = None) -> int:
    """``gonka-vllm-serve`` entry point.

    Mirrors v0.23.0 ``api_server.py:__main__`` (L693-703) but routes through
    :func:`_run_server` so we own the composition step.
    """
    # Deferred to avoid pulling vllm at --help time on a system without it.
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    # ``FlexibleArgumentParser`` moved out of the flat ``vllm/utils.py`` module
    # into ``vllm.utils.argparse_utils`` in v0.22.0+ and is NOT re-exported from
    # the ``vllm.utils`` package ``__init__.py``. Try the canonical location
    # first, fall back to the legacy flat path for older wheels or any future
    # re-export. Mirrors the ``cli_env_setup`` try/except pattern below.
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:  # pragma: no cover - legacy/future re-export fallback
        from vllm.utils import FlexibleArgumentParser  # type: ignore[no-redef]

    # ``cli_env_setup`` MUST run before we hand off to uvloop. Upstream calls
    # it at the very top of ``vllm/entrypoints/openai/api_server.py:__main__``
    # (v0.23.0 L697) to set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` (default is
    # ``fork``). Skipping it crashes TP>1 / PP>1 launches with the classic
    # CUDA-in-forked-process error -- our gonka-vllm-serve entry was missing
    # this call entirely. Import path on v0.23.0:
    # ``vllm.entrypoints.serve.utils.api_utils.cli_env_setup``.
    try:
        from vllm.entrypoints.serve.utils.api_utils import (  # type: ignore[import-not-found]
            cli_env_setup,
        )
    except Exception:  # pragma: no cover - upstream layout drift fallback
        cli_env_setup = None  # type: ignore[assignment]

    if cli_env_setup is not None:
        cli_env_setup()
    else:
        logger.warning(
            "gonka-vllm-serve: could not import vllm.entrypoints.serve.utils."
            "api_utils.cli_env_setup -- VLLM_WORKER_MULTIPROC_METHOD may be "
            "left at the unsafe default. TP>1/PP>1 launches may crash on CUDA-fork."
        )

    parser = FlexibleArgumentParser(
        description="gonka-vllm-serve: vLLM OpenAI server with Gonka PoC v2 plugin",
    )
    parser = make_arg_parser(parser)

    # Gonka-local toggles (do NOT shadow upstream flag names).
    #
    # nargs="+" (not "*"): the bare flag without values used to silently parse
    # as an empty list, and ``build_gonka_app`` happily installed a tuple()
    # of blocked prefixes -- the gate became permanently disabled because
    # ``any(path.startswith(p) for p in ())`` is always False. ``nargs="+"``
    # turns a typo (``--gonka-poc-block-prefixes`` with no values) into an
    # argparse error instead of a silent gate-off.
    parser.add_argument(
        "--gonka-poc-block-prefixes",
        nargs="+",
        default=None,
        help="REPLACES the default list of path prefixes that PoC priority "
        "gates with 503 while generation is active. Requires at least one "
        "value if supplied. Default (when omitted): "
        "/v1/chat/completions /v1/completions.",
    )

    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)

    # Mirror upstream ``api_server.run_server``: translate SIGTERM into the
    # same shutdown path as Ctrl-C so the event loop unwinds the engine
    # cleanly instead of being torn down mid-step.
    signal.signal(signal.SIGTERM, _interrupt_init)

    # uvloop.run is the upstream parity choice (matches
    # ``vllm.entrypoints.openai.api_server.__main__``). Deferred so the
    # ``--help`` path stays light on a system without uvloop.
    import uvloop  # type: ignore[import-not-found]

    try:
        uvloop.run(_run_server(args))
    except KeyboardInterrupt:
        logger.info("gonka-vllm-serve interrupted; shutting down")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
