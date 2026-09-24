"""``gonka-vllm-serve`` -- the stock vLLM OpenAI server plus the PoC gate.

The upstream entry symbols moved in 0.28.1 and again in 0.30.0; ``_resolve``
looks each one up in its known locations, newest first, so one file serves the
0.25, 0.28 and 0.30 lines.

No vLLM source is patched. The app comes from ``build_app(args, ...)``. An
engine build carrying the PoC seams registers the PoC router there, and the
0.28/0.30 residuals install the gate there too, in which case the app is left
as built. Otherwise the gating middleware is added after ``build_app`` so it
sits outermost (Starlette adds middleware in reverse order) and answers 503 on
``/v1/chat/completions`` and ``/v1/completions`` while PoC is active. Then
``serve_http``, as in upstream's ``build_and_serve``.
"""
from __future__ import annotations

import importlib
import logging
import signal
import sys
from typing import Any, Iterable, Optional

# vllm imports are deferred to ``main`` so ``--help`` does not fork the engine.
from fastapi import FastAPI

from gonka_poc.entrypoint.gating import (
    DEFAULT_BLOCKED_PREFIXES,
    PoCGate,
    install_gating_middleware,
)

logger = logging.getLogger("gonka_poc.entrypoint")


def _resolve(symbol: str, *module_names: str) -> Any:
    """Import ``symbol`` from the first of ``module_names`` that has it;
    the ImportError names every location tried."""
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
    """SIGTERM -> KeyboardInterrupt, the same shutdown path as Ctrl-C."""
    raise KeyboardInterrupt("gonka-vllm-serve received SIGTERM")


def build_gonka_app(
    app: FastAPI,
    *,
    gate: PoCGate,
    blocked_prefixes: Optional[Iterable[str]] = None,
) -> FastAPI:
    """Add the gating middleware to the upstream-built app.

    The PoC router is not attached here: ``build_app`` registers it in an engine
    build carrying the PoC seams, and a second ``include_router`` copy breaks
    prometheus route-name lookup. When ``build_app`` already installed a gate
    (``app.state.gonka_gate``, the 0.28 and 0.30 residuals), the app is
    returned untouched: a second middleware would gate the routes twice and
    the router would toggle a gate the middleware does not read.
    """
    existing = getattr(app.state, "gonka_gate", None)
    if existing is not None:
        logger.info(
            "gonka-vllm-serve: build_app already installed the PoC gate (%s)",
            type(existing).__name__,
        )
        return app

    app.state.gonka_gate = gate
    prefixes = (
        tuple(blocked_prefixes)
        if blocked_prefixes is not None
        else DEFAULT_BLOCKED_PREFIXES
    )
    # Added last, so it runs first (Starlette's reverse insertion order).
    install_gating_middleware(app, gate=gate, blocked_prefixes=prefixes)
    return app


def _import_parser_plugins(args: Any) -> None:
    """``--tool-parser-plugin`` / ``--reasoning-parser-plugin`` before the app
    is built, as upstream's ``run_server_worker`` does (0.28+ flags)."""
    tool_plugin = getattr(args, "tool_parser_plugin", None)
    if tool_plugin and len(tool_plugin) > 3:
        from vllm.tool_parsers import ToolParserManager

        ToolParserManager.import_tool_parser(tool_plugin)
    reasoning_plugin = getattr(args, "reasoning_parser_plugin", None)
    if reasoning_plugin and len(reasoning_plugin) > 3:
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.import_reasoning_parser(reasoning_plugin)


async def _run_server(args: Any) -> None:
    """Upstream ``run_server`` + ``build_and_serve`` (v0.30.0
    ``vllm/entrypoints/launchers/api_server/entry.py``) with the PoC
    composition between ``build_app`` and ``init_app_state``."""
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
    try:
        get_uvicorn_log_config = _resolve(
            "get_uvicorn_log_config",
            "vllm.entrypoints.launchers.utils.server_utils",
            "vllm.entrypoints.serve.utils.server_utils",
        )
    except ImportError:  # uvicorn defaults instead of a crash
        get_uvicorn_log_config = None

    _import_parser_plugins(args)

    listen_address, sock = setup_server(args, reuse_port=False)

    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        app = build_app(args, supported_tasks, model_config)
        build_gonka_app(
            app,
            gate=PoCGate(),
            blocked_prefixes=getattr(args, "gonka_poc_block_prefixes", None),
        )
        # After build_app and our middleware: serve_http freezes the stack.
        await init_app_state(engine_client, app.state, args, supported_tasks)

        log_config = None
        if get_uvicorn_log_config is not None:
            try:
                log_config = get_uvicorn_log_config(args)
            except Exception:  # pragma: no cover - defensive
                log_config = None

        # Every kwarg upstream forwards; getattr keeps a removed flag at None.
        serve_http_kwargs: dict[str, Any] = {
            "sock": sock,
            "enable_ssl_refresh": getattr(args, "enable_ssl_refresh", False),
            "host": args.host,
            "port": args.port,
            "log_level": getattr(args, "uvicorn_log_level", "info"),
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
        shutdown_task = await serve_http(app, **serve_http_kwargs)
        await shutdown_task

    sock.close()


def main(argv: list[str] | None = None) -> int:
    """``gonka-vllm-serve`` entry point."""
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
    FlexibleArgumentParser = _resolve(
        "FlexibleArgumentParser", "vllm.utils.argparse_utils", "vllm.utils"
    )
    # Sets VLLM_WORKER_MULTIPROC_METHOD=spawn; without it TP>1 / PP>1 launches
    # crash on CUDA-in-forked-process.
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
            "gonka-vllm-serve: cli_env_setup not found; "
            "VLLM_WORKER_MULTIPROC_METHOD keeps its default"
        )

    parser = FlexibleArgumentParser(
        description="gonka-vllm-serve: vLLM OpenAI server with Gonka PoC v2 plugin",
    )
    parser = make_arg_parser(parser)
    # nargs="+": a bare flag is an argparse error, not an empty prefix list
    # that would silently disable the gate.
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

    signal.signal(signal.SIGTERM, _interrupt_init)

    import uvloop  # type: ignore[import-not-found]

    try:
        uvloop.run(_run_server(args))
    except KeyboardInterrupt:
        logger.info("gonka-vllm-serve interrupted; shutting down")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
