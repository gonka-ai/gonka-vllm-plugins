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
      1. Route ``gonka_poc.*`` log records through vLLM's handler.
      2. Install the KV borrow/return UTILITY methods on
         ``vllm.v1.engine.core.EngineCore`` (via the version-dispatched
         compat shim). ``load_general_plugins()`` runs inside the
         engine-core process (pinned with version + contract test in the
         shim's ``install_engine_core_poc_methods`` docstring), which is
         the only process that owns the BlockPool -- class-level injection
         here is what makes ``call_utility_async("gonka_poc_borrow_blocks",
         ...)`` from the API server resolve. Harmless no-op in every other
         process.

    NOTE: we do NOT install the worker extension here -- that lives behind
    the ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` CLI
    flag, which vLLM consumes during ParallelConfig parsing.

    The PoC API routes are registered by the engine itself, in
    ``vllm.entrypoints.openai.api_server.build_app``; the plugin does not
    wrap or patch that function.
    """
    global _registered
    if _registered:
        return

    _adopt_vllm_logging()

    try:
        # Before anything compiles: knobs that change the traced forward get
        # their own compile cache (see compile_cache.py). No-op at defaults.
        from gonka_poc.compile_cache import scope_compile_cache

        scope_compile_cache()
    except Exception as exc:
        logger.warning("gonka_poc.plugin.register: compile cache scoping skipped: %s", exc)

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


__all__ = ["register"]
