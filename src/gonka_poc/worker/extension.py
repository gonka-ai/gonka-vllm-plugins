"""PoCWorkerExtension -- mixed into the vLLM V1 GPU Worker via ``--worker-extension-cls``
(supported minors per ``gonka_poc._compat``; version-specific surfaces live in
the compat shim).

Activation:
    vllm serve <model> --worker-extension-cls gonka_poc.worker.PoCWorkerExtension

How vLLM wires this in (verified on the 0.25.1 line):
    ``vllm/v1/worker/worker_base.py:263-284`` (WorkerWrapperBase.init_worker)
    resolves the qualname, asserts no attribute collisions with the concrete
    Worker, then appends ``PoCWorkerExtension`` to ``worker_class.__bases__``.
    There is NO __init__ -- methods just become attributes on the live Worker.

Inside any method on this class, ``self`` is the live GPU Worker. Available
attributes:
    self.model_runner           -- GPUModelRunner (gpu_model_runner.py)
    self.model_runner.model     -- the nn.Module
    self.model_runner.kv_caches -- list[torch.Tensor]   (declared L525)
    self.model_runner.attn_groups -- list[list[AttentionGroup]] (L530)
    self.device, self.rank, self.vllm_config

Invocation (from the API server / async engine):
    await async_llm.collective_rpc(
        "execute_poc_decode",
        args=(),
        kwargs={"block_hash": ..., "public_key": ..., "nonces": [...],
                "seq_len": int, "max_tokens": int, "route_window": int, ...},
        timeout=POC_RPC_TIMEOUT_MS / 1000,
    )

    (``collective_rpc`` takes seconds; the env knob is ``POC_RPC_TIMEOUT_MS``,
    milliseconds, in ``gonka_poc.poc.config``.)

CONTRACT WARNINGS:
- Method names MUST NOT collide with any public Worker attribute -- vLLM
  asserts ``not hasattr(worker_class, attr)`` at init_worker time. Keep the
  ``execute_poc_*`` prefix unique.
- Return values must be msgpack-serialisable; do NOT return tensors. Return
  digests / dicts of bytes / ints (artifacts carry vectors as base64 strings
  via :func:`gonka_poc.poc.data.encode_vector`).
- Every TP/PP rank executes the method; the API server aggregates results
  across ranks (PP non-last ranks return ``{"artifacts": [], "rank": ...}``
  because the underlying forward returns None for them).
"""
from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional

# NOTE: keep imports light at module scope -- this file is imported in every
# worker process during init_worker. Heavy imports (torch, gonka_poc.poc.*)
# are deferred into method bodies.
#
import logging

logger = logging.getLogger(__name__)

class PoCWorkerExtension:
    """Add-only methods reachable from ``collective_rpc``.

    See module docstring for the full contract.
    """

    # ------------------------------------------------------------------ #
    # PoC forward (the actual GPU work)
    # ------------------------------------------------------------------ #

    def execute_poc_decode(
        self,
        block_hash: str,
        public_key: str,
        nonces,
        seq_len: int,
        max_tokens: int,
        route_window: int = 256,
        enforced_k_steps=None,
        debug: bool = False,
        va_steps: int = 0,
        per_nonce_reflection: bool = False,
        borrowed_block_ids=None,
        borrowed_stripe=None,
    ):
        """Decode-PoC chunk (prefill + max_tokens chained steps) — the new
        scheme's worker entry: every rank runs it, the TP driver returns
        artifacts, other ranks return None and the API server aggregates.
        """
        from gonka_poc.poc.decode_runner import execute_poc_decode as _run
        hidden_size = int(self.vllm_config.model_config.get_hidden_size())
        try:
            return _run(
                self,
                block_hash=block_hash,
                public_key=public_key,
                nonces=list(nonces),
                seq_len=int(seq_len),
                max_tokens=int(max_tokens),
                hidden_size=int(hidden_size),
                route_window=int(route_window),
                enforced_k_steps=(
                    {int(k): list(v) for k, v in enforced_k_steps.items()}
                    if enforced_k_steps else None),
                debug=bool(debug),
                va_steps=int(va_steps),
                per_nonce_reflection=bool(per_nonce_reflection),
                borrowed_block_ids=(
                    list(borrowed_block_ids)
                    if borrowed_block_ids is not None else None),
                borrowed_stripe=(
                    int(borrowed_stripe)
                    if borrowed_stripe is not None else None),
            )
        except Exception:  # noqa: BLE001 — surface via RPC, never hang the rank
            logger.exception("execute_poc_decode failed")
            raise

# Public alias used in the ``--worker-extension-cls`` CLI string.
__all__ = ["PoCWorkerExtension"]
