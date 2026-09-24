"""``gonka-vllm-serve`` -- compose a FastAPI app on top of stock vLLM.

This is a thin wrapper around the upstream server entry. The upstream symbols
moved in vLLM 0.28.1 and again in 0.30.0, so each one is resolved through a
short list of known locations (newest first):

    setup_server, serve_http     vllm.entrypoints.launchers.launcher
                                 (before 0.28.1: vllm.entrypoints.openai.api_server /
                                 vllm.entrypoints.launcher)
    build_async_engine_client    vllm.entrypoints.launchers.api_server.entry
    build_app                    vllm.entrypoints.launchers.app
    init_app_state               vllm.entrypoints.launchers.api_server.app_state
                                 (before 0.30.0 the three above lived in
                                 vllm.entrypoints.openai.api_server)
    make_arg_parser,
    validate_parsed_serve_args   vllm.entrypoints.launchers.cli_args
                                 (before 0.28.1: vllm.entrypoints.openai.cli_args)
    get_uvicorn_log_config       vllm.entrypoints.launchers.utils.server_utils
                                 (before 0.30.0: vllm.entrypoints.serve.utils.server_utils)

We do NOT patch any vLLM source file. We only:
  1. Build the stock FastAPI app via ``build_app(args, ...)``.
  2. Leave the PoC router alone: ``build_app`` registers it itself in a vllm
     build carrying the PoC engine seams.
  3. Install ``PoCGatingMiddleware`` AFTER ``build_app`` returns so it ends up
     OUTERMOST in Starlette's reverse-insertion order, gating the
     ``/v1/chat/completions`` and ``/v1/completions`` routes with 503 when PoC
     is active -- unless the engine build already installed the gate inside
     ``build_app`` (the 0.28/0.30 residual does), in which case the app is left
     as built.
  4. Forward to ``serve_http`` exactly like upstream's ``build_and_serve``.

Middleware ordering note (verified against v0.23.0
``vllm/entrypoints/openai/api_server.py:156-300`` and v0.30.0
``vllm/entrypoints/launchers/app.py``): user-supplied ``--middleware`` are
added inside ``build_app``; we add OURS AFTER ``build_app`` so we sit outside
them too. The chat-completion handler is reached only if the gate is open.
"""
from __future__ import annotations

import importlib
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


def _resolve(symbol: str, *module_names: str) -> Any:
    """Import ``symbol`` from the first of ``module_names`` that provides it.

    The candidates are ordered newest layout first. Raises ``ImportError``
    naming every location tried, so a future move shows up as one clear
    message instead of a bare ``ModuleNotFoundError`` deep in the launcher.
    """
    tried: list[str] = []
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            tried.append(name)
            continue
        if hasattr(module, symbol):
            return getattr(module, symbol)
        tried.append(f"{name} (no {symbol})")
    raise ImportError(
        f"gonka-vllm-serve: cannot import {symbol}; tried {', '.join(tried)}"
    )


def _interrupt_init(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    """SIGTERM handler mirroring upstream ``run_server``'s ``_interrupt_init``.

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

    An engine build whose ``build_app`` already installed a gate (the 0.28 and
    0.30 residuals set ``app.state.gonka_gate`` and add the middleware there)
    is left untouched: a second middleware would gate the same routes twice
    and the router would toggle a gate the middleware does not read.

    Args:
        app: the FastAPI instance returned by ``build_app(args, ...)``.
        gate: the shared :class:`PoCGate` flag toggled by the PoC router.
        blocked_prefixes: optional override for the path prefixes the gating
            middleware 503s while PoC is active. ``None`` uses
            :data:`gonka_poc.entrypoint.gating.DEFAULT_BLOCKED_PREFIXES`.

    Returns:
        The same ``app`` instance (mutated). Returned for chainability.
    """
    existing = getattr(app.state, "gonka_gate", None)
    if existing is not None:
        logger.info(
            "gonka-vllm-serve: build_app already installed the PoC gate (%s); "
            "not installing a second one",
            type(existing).__name__,
        )
        return app

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


def _import_parser_plugins(args: Any) -> None:
    """Mirror upstream ``run_server_worker``: load ``--tool-parser-plugin`` and
    ``--reasoning-parser-plugin`` before the app is built (0.28+ flags; absent
    on older argparsers, hence ``getattr``)."""
    tool_plugin = getattr(args, "tool_parser_plugin", None)
    if tool_plugin and len(tool_plugin) > 3:
        from vllm.tool_parsers import ToolParserManager

        ToolParserManager.import_tool_parser(tool_plugin)
    reasoning_plugin = getattr(args, "reasoning_parser_plugin", None)
    if reasoning_plugin and len(reasoning_plugin) > 3:
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.import_reasoning_parser(reasoning_plugin)


async def _run_server(args: Any) -> None:
    """Async body equivalent to upstream ``run_server`` + ``build_and_serve``,
    but inserting PoC composition between ``build_app`` and ``init_app_state``.

    Mirrors v0.30.0 ``vllm/entrypoints/launchers/api_server/entry.py``
    (``build_and_serve`` / ``run_server_worker``).
    """
    # Deferred imports: keep ``gonka-vllm-serve --help`` fast and isolated
    # from CUDA fork issues.
    import vllm.envs as envs

    setup_server = _resolve(
        "setup_server",
        "vllm.entrypoints.launchers.launcher",
        "vllm.entrypoints.openai.api_server",
    )
    serve_http = _resolve(
        "serve_http",
        "vllm.entrypoints.launchers.launcher",
        "vllm.entrypoints.launcher",
    )
    build_async_engine_client = _resolve(
        "build_async_engine_client",
        "vllm.entrypoints.launchers.api_server.entry",
        "vllm.entrypoints.openai.api_server",
    )
    build_app = _resolve(
        "build_app",
        "vllm.entrypoints.launchers.app",
        "vllm.entrypoints.openai.api_server",
    )
    init_app_state = _resolve(
        "init_app_state",
        "vllm.entrypoints.launchers.api_server.app_state",
        "vllm.entrypoints.openai.api_server",
    )

    # Best-effort: honours ``--log-config-file`` /
    # ``--disable-access-log-for-endpoints``. If the internal module path
    # moves again, fall back to None (uvicorn defaults) rather than crashing.
    try:
        get_uvicorn_log_config = _resolve(
            "get_uvicorn_log_config",
            "vllm.entrypoints.launchers.utils.server_utils",
            "vllm.entrypoints.serve.utils.server_utils",
        )
    except ImportError:  # pragma: no cover - vllm internal layout drift
        get_uvicorn_log_config = None

    _import_parser_plugins(args)

    listen_address, sock = setup_server(args, reuse_port=False)

    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        # Stock vLLM app + middleware/handlers.
        app = build_app(args, supported_tasks, model_config)

        # Gonka composition: gating middleware (a no-op when the engine build
        # installed the gate inside build_app).
        build_gonka_app(
            app,
            gate=PoCGate(),
            blocked_prefixes=getattr(args, "gonka_poc_block_prefixes", None),
        )

        # Standard upstream state population (sets app.state.engine_client and
        # app.state.openai_serving_*). MUST run after build_app and after we
        # add our middleware (Starlette freezes the stack on startup, which
        # serve_http triggers).
        await init_app_state(engine_client, app.state, args, supported_tasks)

        # Mirror upstream ``build_and_serve``. Every kwarg upstream forwards
        # MUST be forwarded here too -- missing any of these silently drops
        # user-supplied TLS / HTTP-limit / log config flags.
        # ``getattr(args, "...", None)`` keeps us robust to upstream argparse
        # changes: a removed flag falls back to None, which ``serve_http``
        # already tolerates.
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

        logger.info("gonka-vllm-serve: starting on %s", listen_address)
        # Hand off to uvicorn via the stock launcher.
        shutdown_task = await serve_http(app, **serve_http_kwargs)
        await shutdown_task

    sock.close()


def main(argv: list[str] | None = None) -> int:
    """``gonka-vllm-serve`` entry point.

    Mirrors upstream ``api_server`` ``main`` but routes through
    :func:`_run_server` so we own the composition step.
    """
    # Deferred to avoid pulling vllm at --help time on a system without it.
    make_arg_parser = _resolve(
        "make_arg_parser",
        "vllm.entrypoints.launchers.cli_args",
        "vllm.entrypoints.openai.cli_args",
    )
    validate_parsed_serve_args = _resolve(
        "validate_parsed_serve_args",
        "vllm.entrypoints.launchers.cli_args",
        "vllm.entrypoints.openai.cli_args",
    )
    # ``FlexibleArgumentParser`` moved out of the flat ``vllm/utils.py`` module
    # into ``vllm.utils.argparse_utils`` in v0.22.0+ and is NOT re-exported from
    # the ``vllm.utils`` package ``__init__.py``.
    FlexibleArgumentParser = _resolve(
        "FlexibleArgumentParser", "vllm.utils.argparse_utils", "vllm.utils"
    )

    # ``cli_env_setup`` MUST run before we hand off to uvloop. Upstream calls
    # it at the top of its ``main`` to set ``VLLM_WORKER_MULTIPROC_METHOD=spawn``
    # (default is ``fork``). Skipping it crashes TP>1 / PP>1 launches with the
    # classic CUDA-in-forked-process error.
    try:
        cli_env_setup = _resolve(
            "cli_env_setup", "vllm.entrypoints.serve.utils.api_utils"
        )
    except ImportError:  # pragma: no cover - upstream layout drift fallback
        cli_env_setup = None

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

    # Mirror upstream ``run_server``: translate SIGTERM into the same shutdown
    # path as Ctrl-C so the event loop unwinds the engine cleanly instead of
    # being torn down mid-step.
    signal.signal(signal.SIGTERM, _interrupt_init)

    # uvloop.run is the upstream parity choice. Deferred so the ``--help``
    # path stays light on a system without uvloop.
    import uvloop  # type: ignore[import-not-found]

    try:
        uvloop.run(_run_server(args))
    except KeyboardInterrupt:
        logger.info("gonka-vllm-serve interrupted; shutting down")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
