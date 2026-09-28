# SPDX-License-Identifier: Apache-2.0
"""The v0.1.x prefill scheme, kept intact and selected at launch.

This is the derivation the deployed fleet validates: one forward over the
synthetic prompt, Householder reflections seeded per layer from the block
hash alone, the MoE router left untouched, and the k_dim pick seeded without
the decode salt. Its artifacts are bit-identical to the shipped MLNode image.

``--poc-decode`` switches the whole node to the decode scheme instead. The two
never share a process: the decode stack forces the MoE router and seeds
reflections per nonce, so a hidden state produced under one is not comparable
with the other. Selecting per request would mean two derivations behind one
endpoint, which is why this is a launch flag.
"""
import logging
from typing import Any, Dict, List, Optional

from gonka_poc._compat import current as _compat_current

logger = logging.getLogger(__name__)

POC_RPC_TIMEOUT_MS = 60000

async def execute_poc_forward_rpc(
    engine_client: Any,
    *,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    poc_stronger_rng: bool = False,
    timeout_ms: int = POC_RPC_TIMEOUT_MS,
    lease: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run ``execute_poc_forward`` on every worker rank and aggregate.

    ``lease`` is a KV block lease from :func:`poc_reservation`
    (``{"block_ids": [...], "blocks_per_seq": int}``) — when present the
    forward writes ONLY leased blocks and live inference stays intact;
    ``None`` selects the legacy in-place layout (blocks 1..N — callers must
    have aborted inference first).

    Uses ``EngineClient.collective_rpc`` (vllm/engine/protocol.py) to invoke
    :meth:`gonka_poc.worker.PoCWorkerExtension.execute_poc_forward` on each
    rank. Each rank returns ``{"artifacts": [...], "rank": int}``;
    PP non-last ranks return an empty list. We aggregate the union (in
    practice only the PP last rank produces non-empty artifacts, but a
    union is safe and handles non-PP topologies uniformly).

    Args / kwargs mirror what ``PoCWorkerExtension.execute_poc_forward``
    accepts. The vectors are already base64-encoded FP16 in the per-rank
    result; we do not need to decode here -- the API response forwards the
    ``vector_b64`` strings unchanged.

    Returns: ``{"artifacts": [{"nonce": int, "vector_b64": str}, ...]}``.
    """
    if not nonces:
        return {"artifacts": []}

    if lease is None:
        # In-place layout writes blocks 1..N unconditionally, and nothing
        # gates new admissions on the validation path — re-drain in-flight
        # inference before EVERY legacy chunk (upstream donor behaviour;
        # requests admitted between chunks would otherwise be silently
        # clobbered). During mining the gate keeps the in-flight set empty,
        # so this is a cheap no-op there.
        try:
            await _compat_current().abort_all_requests(engine_client)
        except Exception as exc:
            logger.warning("PoC pre-chunk abort failed: %s", exc)

    timeout_sec = timeout_ms / 1000.0
    results = await engine_client.collective_rpc(
        "execute_poc_forward",
        timeout=timeout_sec,
        kwargs={
            "block_hash": block_hash,
            "public_key": public_key,
            "nonces": list(nonces),
            "seq_len": int(seq_len),
            "k_dim": int(k_dim),
            "poc_stronger_rng": bool(poc_stronger_rng),
            "borrowed_block_ids": (
                list(lease["block_ids"]) if lease else None),
            "borrowed_stripe": (
                int(lease["blocks_per_seq"]) if lease else None),
        },
    )

    # Aggregate per-rank artifacts. In a PP topology only the last rank
    # populates artifacts; in TP-only it's typically the driver rank
    # (whichever ran the forward to completion). De-duplicate by nonce so a
    # buggy worker that doubles up doesn't corrupt the API response.
    seen: set = set()
    artifacts: List[Dict[str, Any]] = []
    for rank_result in results:
        if not rank_result:
            continue
        for art in rank_result.get("artifacts", []) or []:
            nonce = art.get("nonce")
            if nonce is None or nonce in seen:
                continue
            seen.add(nonce)
            artifacts.append(art)

    return {"artifacts": artifacts}


async def compute_prefill_artifacts(
    engine_client: Any,
    *,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    poc_stronger_rng: bool = False,
    chunk: int = 0,
    timeout_ms: int = POC_RPC_TIMEOUT_MS,
    lease: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Prefill artifacts for `nonces`, chunked the way v0.1.x chunks them.

    ``chunk`` 0 submits every nonce in one RPC (the engine batches them);
    a positive value pins in-flight nonces to that number.
    """
    step = chunk if chunk > 0 else len(nonces)
    out: List[dict] = []
    for i in range(0, len(nonces), max(step, 1)):
        res = await execute_poc_forward_rpc(
            engine_client,
            nonces=nonces[i:i + step],
            block_hash=block_hash,
            public_key=public_key,
            seq_len=seq_len,
            k_dim=k_dim,
            poc_stronger_rng=poc_stronger_rng,
            timeout_ms=timeout_ms,
            lease=lease,
        )
        out.extend(res.get("artifacts", []))
    return out
