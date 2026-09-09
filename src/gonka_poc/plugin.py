"""``vllm.general_plugins`` entry point for the gonka-poc package.

Registration is declared in ``pyproject.toml``:

    [project.entry-points."vllm.general_plugins"]
    gonka_poc = "gonka_poc.plugin:register"

vLLM 0.23.0 calls :func:`register` in every process that touches the model
class: process0, the V1 engine-core process, every worker process, and the
registry inspection subprocess. The function MUST therefore be:

  * re-entrant (idempotent across multiple calls in one process),
  * cheap (no CUDA init, no big imports at module scope),
  * exception-safe (vllm.plugins.load_general_plugins swallows exceptions
    but logs them at exception level -- we still want to avoid crashing).

Verified call sites (vllm 0.23.0):
  - vllm/engine/arg_utils.py:747   (sync engine startup)
  - vllm/v1/engine/core.py:109     (V1 engine-core process)
  - vllm/v1/worker/worker_base.py:247   (every worker process)
  - vllm/model_executor/models/registry.py:1411   (inspection subprocess)
"""
from __future__ import annotations

import logging

logger = logging.getLogger("gonka_poc.plugin")

_registered: bool = False


def register() -> None:
    """vllm.general_plugins entry point. Idempotent.

    Tasks performed (each guarded for repeated calls):
      1. Set the ``gonka_poc.PLUGIN_LOADED`` process-local flag so
         ``PoCGatingMiddleware`` can detect "plugin loaded but no gate
         attached" (operator likely ran plain ``vllm serve`` instead of
         ``gonka-vllm-serve``).
      2. Install a one-shot warning wrapper around
         ``vllm.entrypoints.openai.api_server.build_app`` so the same
         gate-presence check fires for both ``vllm serve`` and our own
         ``gonka-vllm-serve`` path.
      3. Install the KV borrow/return UTILITY methods on
         ``vllm.v1.engine.core.EngineCore`` (via the version-dispatched
         compat shim). ``load_general_plugins()`` runs inside the
         engine-core process (pinned with version + contract test in the
         shim's ``install_engine_core_poc_methods`` docstring), which is
         the only process that owns the BlockPool — class-level injection
         here is what makes ``call_utility_async("gonka_poc_borrow_blocks",
         ...)`` from the API server resolve. Harmless no-op in every other
         process.

    NOTE: we do NOT install the worker extension here -- that lives behind
    the ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` CLI
    flag, which vLLM consumes during ParallelConfig parsing.
    """
    global _registered
    if _registered:
        return

    # PoC-wrapped model classes: registered FIRST so the engine compiles the
    # wrapped module (transforms inside the compiled graph — 0.20 bit path).
    # ONE model-agnostic factory for every architecture (call agreement
    # 2026-08-19): subclasses are built dynamically against whatever base
    # classes this vLLM build ships; extra archs via POC_ARCHITECTURES env.
    try:
        from vllm import ModelRegistry
        from gonka_poc.models.factory import build_poc_subclasses
        for arch, sub in build_poc_subclasses():
            ModelRegistry.register_model(arch, sub)
            logger.info("gonka_poc: %s overridden with PoC-wrapped class", arch)
    except Exception:  # pragma: no cover — non-vllm import contexts
        logger.exception("gonka_poc: model registry override failed")

    # Expose a process-local flag so the gate-presence check in
    # ``gonka_poc.entrypoint.gating.PoCGatingMiddleware`` can detect "plugin
    # loaded but no gate attached" (operator likely ran plain ``vllm serve``
    # instead of ``gonka-vllm-serve``). Set BEFORE the inner registration
    # steps so the flag survives even if a sub-step raises.
    import gonka_poc as _pkg  # local import: avoid cycles at module import

    _pkg.PLUGIN_LOADED = True

    _adopt_vllm_logging()

    try:
        _install_build_app_warning_wrapper()
    except Exception:  # pragma: no cover
        logger.exception(
            "gonka_poc.plugin.register: build_app warning wrapper install failed"
        )

    try:
        from gonka_poc._compat import current as _compat_current

        _compat_current().install_engine_core_poc_methods()
    except Exception as exc:
        # Unsupported vllm minor / import quirk: validation degrades to the
        # legacy abort-based path, never a crash at plugin load.
        logger.debug(
            "gonka_poc.plugin.register: EngineCore borrow install skipped: %s",
            exc)

    _registered = True


def _adopt_vllm_logging() -> None:
    """Route every ``gonka_poc.*`` record through vLLM's log handler.

    vLLM's logging config covers only the ``vllm`` namespace, so plugin
    records propagate to the root logger, which has no handler: INFO
    vanishes and WARNING+ comes out bare via ``logging.lastResort``.

    When vLLM's logger has no handlers the operator disabled
    VLLM_CONFIGURE_LOGGING and owns the logging tree; keep propagating to
    root rather than second-guessing their setup.
    """
    vllm_logger = logging.getLogger("vllm")
    if not vllm_logger.handlers:
        return
    pkg_logger = logging.getLogger("gonka_poc")
    pkg_logger.handlers = list(vllm_logger.handlers)
    pkg_logger.setLevel(vllm_logger.level)
    pkg_logger.propagate = False


_build_app_wrapped: bool = False


def _install_build_app_warning_wrapper() -> None:
    """Wrap ``vllm.entrypoints.openai.api_server.build_app`` with a one-shot
    startup-event registrar.

    Purpose: catch the operator footgun where ``vllm serve`` runs instead of
    ``gonka-vllm-serve``. The plugin's :func:`register` still executes (so
    ``PLUGIN_LOADED`` is set) but no ``app.state.gonka_gate`` is ever
    attached -- the chat endpoint would NOT be 503-gated while PoC seizes
    the GPU.

    We mutate the upstream module so the wrapper is in effect for both the
    bare ``vllm serve`` invocation and our own ``gonka-vllm-serve`` path
    (where the wrapper just no-ops on top of an already-registered hook).

    Best-effort: if vllm or its api_server module is not importable in this
    process (e.g. a worker subprocess where the entrypoint isn't even
    referenced), we silently skip.
    """
    global _build_app_wrapped
    if _build_app_wrapped:
        return

    try:
        from vllm.entrypoints.openai import api_server as _api_server  # type: ignore
    except Exception:
        # No api_server in this process -- engine-core / worker / inspection
        # subprocesses don't need this hook. Not an error.
        return

    original_build_app = getattr(_api_server, "build_app", None)
    if original_build_app is None:
        return

    def _wrapped_build_app(*args, **kwargs):  # type: ignore[no-untyped-def]
        app = original_build_app(*args, **kwargs)
        try:
            # Install a tiny ASGI middleware that fires the warning on first
            # HTTP dispatch (see PoCGatingMiddleware's docstring for why a
            # startup event cannot work). ``gonka-vllm-serve`` installs
            # ``PoCGatingMiddleware`` itself, so this wrapper covers ONLY the
            # bare ``vllm serve`` accident path.
            from gonka_poc.entrypoint.gating import PoCGate, install_gating_middleware

            # Sentinel gate that never activates -- the middleware acts purely
            # as the warning carrier here. ``PoCGate().is_active()`` is False
            # by construction so no real request is ever 503'd by this shim.
            sentinel_gate = PoCGate()
            # install_gating_middleware carries the Starlette 1.3.x
            # stack-reset workaround (see its docstring).
            install_gating_middleware(app, gate=sentinel_gate, blocked_prefixes=())
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "gonka_poc: failed to attach gate-presence warning to FastAPI app"
            )
        return app

    _api_server.build_app = _wrapped_build_app  # type: ignore[assignment]
    _build_app_wrapped = True


__all__ = ["register"]
