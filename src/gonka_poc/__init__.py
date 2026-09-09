"""gonka-poc: out-of-tree vLLM plugin for Gonka Proof-of-Compute (PoC v2).

This package ships as a standalone pip-installable plugin that targets a stock
vllm wheel (0.25.x). It provides three integration surfaces:

1. ``vllm.general_plugins`` entry point (:func:`gonka_poc.plugin.register`)
   that sets a process-local ``PLUGIN_LOADED`` flag, installs a one-shot
   wrapper around ``vllm.entrypoints.openai.api_server.build_app`` so a
   gate-presence warning fires when the operator runs ``vllm serve`` instead
   of ``gonka-vllm-serve``, and installs the EngineCore KV borrow/return
   utility methods for leased-block validation (ADR-0015).
2. ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` exposing
   ``execute_poc_decode`` to vLLM's ``collective_rpc``.
3. ``gonka-vllm-serve`` console script that composes a FastAPI app on top of
   the stock vLLM OpenAI-compatible server, attaching the PoC API router and a
   503/abort gating middleware.

The fork-residual changes (sampler-stack + structured-output) are NOT shipped
here -- see ``MIGRATION_FROM_FORK.md`` for the disposition of every commit
from the source branch.
"""

# Single source of truth for the version is pyproject.toml; derive it so a
# version bump cannot leave a stale literal behind.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("gonka-poc")
except Exception:  # pragma: no cover - not installed (e.g. source checkout)
    __version__ = "unknown"

# Set to True by :func:`gonka_poc.plugin.register` (the
# ``vllm.general_plugins`` entry point) so ``PoCGatingMiddleware`` can detect
# "plugin loaded but no gate attached" (operator likely ran plain
# ``vllm serve`` instead of ``gonka-vllm-serve``).
PLUGIN_LOADED: bool = False
