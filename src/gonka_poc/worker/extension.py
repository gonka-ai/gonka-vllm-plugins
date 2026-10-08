"""PoCWorkerExtension -- mixed into the vLLM V1 GPU Worker via ``--worker-extension-cls``
(supported minors per ``gonka_poc._compat``; version-specific surfaces live in
the compat shim).

Activation:
    vllm serve <model> --worker-extension-cls gonka_poc.worker.PoCWorkerExtension

How vLLM wires this in (v0.23.0, verified):
    ``vllm/v1/worker/worker_base.py:261-287`` (WorkerWrapperBase.init_worker)
    resolves the qualname, asserts no attribute collisions with the concrete
    Worker, then does ``worker_class.__bases__ += (PoCWorkerExtension,)``.
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
        "execute_poc_forward",
        args=(),
        kwargs={"block_hash": ..., "public_key": ..., "nonces": [...],
                "seq_len": int, "k_dim": int, "poc_stronger_rng": bool},
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

from typing import Any, Dict, List, Optional

# NOTE: keep imports light at module scope -- this file is imported in every
# worker process during init_worker. Heavy imports (torch, gonka_poc.poc.*)
# are deferred into method bodies.
#
# ``gonka_poc._compat`` is intentionally light (pure-Python dispatcher) so
# it's safe to import at module scope; routing kv_caches access through the
# shim keeps the documented private-API touchpoint policy honest.

class PoCWorkerExtension:
    """Add-only methods reachable from ``collective_rpc``.

    See module docstring for the full contract.
    """

    # ------------------------------------------------------------------ #
    # PoC forward (the actual GPU work)
    # ------------------------------------------------------------------ #

    def execute_poc_forward(
        self,
        *,
        block_hash: str,
        public_key: str,
        nonces: List[int],
        seq_len: int,
        k_dim: int = 12,
        poc_stronger_rng: bool = False,
        borrowed_block_ids: Optional[List[int]] = None,
        borrowed_stripe: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute one PoC forward pass on this worker rank.

        Args:
            block_hash, public_key: PoC scope tags fed into the seeded RNG.
            nonces: list[int] of nonces to compute artifacts for; one batched
                forward processes all of them.
            seq_len: sequence length per nonce.
            k_dim: artifact vector dimensionality (default 12).
            poc_stronger_rng: if True, use murmur-concat RNG path; default
                False (legacy seeded normal path).
            borrowed_block_ids / borrowed_stripe: KV block lease from
                ``gonka_poc_borrow_blocks`` (validation-without-abort path).
                ``None`` = legacy in-place layout over blocks 1..N. Physical
                block choice does not affect artifact values (address-only).

        Returns:
            ``{"artifacts": [{"nonce": int, "vector_b64": str}, ...],
               "rank": int}``

            On PP non-last ranks the underlying forward returns ``None``
            (intermediate tensors were forwarded inter-rank); we mirror that
            with an empty artifact list so the caller can aggregate uniformly.

            Keep the payload msgpack-friendly. Do NOT return torch tensors.
        """
        # Deferred imports: pulling gonka_poc.poc.* requires a configured
        # vllm runtime (vllm.logger), which is only available inside the
        # worker process.
        from gonka_poc.poc.data import encode_vector
        from gonka_poc.poc.poc_model_runner import (
            DEFAULT_K_DIM,
            execute_poc_forward as _execute_poc_forward,
        )

        if not nonces:
            return {"artifacts": [], "rank": int(getattr(self, "rank", -1))}

        # vllm_config.model_config.get_hidden_size() is the canonical
        # accessor on all supported minors; an unmapped minor already
        # hard-fails in gonka_poc._compat, so no fallback is needed.
        hidden_size = int(self.vllm_config.model_config.get_hidden_size())

        result = _execute_poc_forward(
            self,  # the live Worker; matches the ``worker`` param in poc_model_runner
            block_hash,
            public_key,
            list(nonces),
            int(seq_len),
            int(hidden_size),
            k_dim=int(k_dim) if k_dim is not None else DEFAULT_K_DIM,
            poc_stronger_rng=bool(poc_stronger_rng),
            borrowed_block_ids=(
                list(borrowed_block_ids)
                if borrowed_block_ids is not None else None),
            borrowed_stripe=(
                int(borrowed_stripe)
                if borrowed_stripe is not None else None),
        )

        rank = int(getattr(self, "rank", -1))

        # PP non-last ranks return None from the underlying forward.
        if result is None:
            return {"artifacts": [], "rank": rank}

        vectors = result.get("vectors")
        result_nonces = result.get("nonces", [])

        artifacts: List[Dict[str, Any]] = []
        if vectors is not None and len(result_nonces) > 0:
            for i, nonce in enumerate(result_nonces):
                artifacts.append({
                    "nonce": int(nonce),
                    "vector_b64": encode_vector(vectors[i]),
                })

        return {"artifacts": artifacts, "rank": rank}

    def execute_poc_forward_multi(
        self,
        *,
        block_hash: str,
        public_key: str,
        nonce_batches: List[List[int]],
        seq_len: int,
        k_dim: int = 12,
        poc_stronger_rng: bool = False,
    ) -> Dict[str, Any]:
        """Several prefill batches in one RPC, back to back on every rank.

        Under pipeline parallelism the first rank runs its layers for batch
        k+1 while the last rank runs its layers for batch k: the send of one
        batch's hidden state only has to match the next rank's receive, so
        the ranks settle one batch apart and both stay busy. A single-batch
        RPC leaves each rank idle while the other works.
        """
        import time

        from vllm.logger import init_logger
        log = init_logger(__name__)
        artifacts: List[Dict[str, Any]] = []
        times = []
        for batch in nonce_batches:
            t0 = time.time()
            res = self.execute_poc_forward(
                block_hash=block_hash, public_key=public_key, nonces=list(batch),
                seq_len=seq_len, k_dim=k_dim, poc_stronger_rng=poc_stronger_rng)
            times.append(time.time() - t0)
            artifacts.extend(res.get("artifacts", []) or [])
        log.info("PoC multi-batch forward: %d batches x %d nonces, per batch %s s",
                 len(nonce_batches), len(nonce_batches[0]) if nonce_batches else 0,
                 " ".join(f"{t:.2f}" for t in times))
        return {"artifacts": artifacts, "rank": int(getattr(self, "rank", -1))}

    # ------------------------------------------------------------------ #
    # Worker-side continuous prefill mining: a thread per rank keeps the
    # pipeline fed between the API's polls, so no RPC boundary drains it.
    # ------------------------------------------------------------------ #

    def execute_poc_mine_start(
        self,
        *,
        block_hash: str,
        public_key: str,
        seq_len: int,
        k_dim: int,
        poc_stronger_rng: bool,
        node_id: int,
        n_nodes: int,
        group_id: int,
        n_groups: int,
        sub_batch: int,
    ) -> Dict[str, Any]:
        import threading
        import time

        from vllm.logger import init_logger
        log = init_logger(__name__)
        st = getattr(self, "_poc_mine", None)
        if st is not None and st["thread"].is_alive():
            return {"rank": int(getattr(self, "rank", -1)), "error": "already mining"}
        st = {"stop_at": None, "started": 0, "done": 0, "buf": [], "lock": threading.Lock(),
              "error": None, "times": [], "thread": None}
        offset = node_id + group_id * n_nodes
        step = n_groups * n_nodes

        def loop():
            import torch
            torch.cuda.set_device(self.device)  # per-thread CUDA device, else NCCL sees device 0
            x = 0
            try:
                while True:
                    with st["lock"]:
                        if st["stop_at"] is not None and st["started"] >= st["stop_at"]:
                            break
                        st["started"] += 1
                    nonces = [offset + (x + i) * step for i in range(sub_batch)]
                    x += sub_batch
                    t0 = time.time()
                    res = self.execute_poc_forward(
                        block_hash=block_hash, public_key=public_key, nonces=nonces,
                        seq_len=seq_len, k_dim=k_dim, poc_stronger_rng=poc_stronger_rng)
                    with st["lock"]:
                        st["buf"].extend(res.get("artifacts", []) or [])
                        st["done"] += 1
                        st["times"].append(time.time() - t0)
            except Exception as e:  # reported through poll
                log.error("PoC worker mining loop failed: %r", e, exc_info=True)
                with st["lock"]:
                    st["error"] = repr(e)

        st["thread"] = threading.Thread(target=loop, name="poc-mine", daemon=True)
        self._poc_mine = st
        st["thread"].start()
        return {"rank": int(getattr(self, "rank", -1))}

    def execute_poc_mine_poll(self) -> Dict[str, Any]:
        st = getattr(self, "_poc_mine", None)
        if st is None:
            return {"artifacts": [], "started": 0, "done": 0, "alive": False, "error": "not mining"}
        with st["lock"]:
            arts, st["buf"] = st["buf"], []
            return {"artifacts": arts, "started": st["started"], "done": st["done"],
                    "alive": st["thread"].is_alive(), "error": st["error"],
                    "rank": int(getattr(self, "rank", -1))}

    def execute_poc_mine_flag(self) -> Dict[str, Any]:
        """Stop taking new batches once ``started`` reaches the number returned
        here; the caller then sets the same target on every rank so that a
        batch one PP rank already started is still received by the next."""
        st = getattr(self, "_poc_mine", None)
        if st is None:
            return {"started": 0}
        with st["lock"]:
            st["stop_at"] = st["started"]
            return {"started": st["started"]}

    def execute_poc_mine_stop(self, *, stop_at: int) -> Dict[str, Any]:
        st = getattr(self, "_poc_mine", None)
        if st is None:
            return {"artifacts": [], "done": 0}
        with st["lock"]:
            st["stop_at"] = max(int(stop_at), st["started"])
        st["thread"].join(timeout=300)
        # Give the mining activations back to the driver: live inference (and
        # NCCL's own cudaMalloc) must not find the card full after a round.
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with st["lock"]:
            arts, st["buf"] = st["buf"], []
            t = st["times"]
            return {"artifacts": arts, "done": st["done"], "alive": st["thread"].is_alive(),
                    "error": st["error"], "rank": int(getattr(self, "rank", -1)),
                    "per_batch_s": (sum(t[2:]) / max(1, len(t) - 2)) if len(t) > 2 else None}

    def execute_poc_borrow_compat(self) -> Dict[str, Any]:
        """Report whether borrowed-lease validation is bit-safe on this rank.

        Always ``scratch_capable=False``: PoC inputs are derived in a fresh
        buffer on every path, so a leased forward derives exactly what the
        in-place forward derives. Kept as an RPC so older API-server code
        that probes it keeps working.
        """
        return {"scratch_capable": False,
                "rank": int(getattr(self, "rank", -1))}


# Public alias used in the ``--worker-extension-cls`` CLI string.
__all__ = ["PoCWorkerExtension"]
