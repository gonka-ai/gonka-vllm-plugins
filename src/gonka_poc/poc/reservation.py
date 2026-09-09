"""KV block reservation for PoC validation — inference keeps running.

Port of gonka-ai/vllm branch ``qd/combine-poc-and-inference`` into the
plugin architecture — commits, design, and rationale in ADR-0015.

Multi-group ("shared lease"): the block-id namespace is pool-global, so one
lease of ``num_nonces * max_g ceil(seq_len/block_size_g)`` blocks reserves
every group's tensors at once; the engine core sizes the stripe, the worker
expands ids per group (ADR-0015 Decisions 1-3,
``poc_model_runner._borrowed_layout``).

Safety contract:
    * Borrowed path: the only KV bytes the PoC forward writes belong to
      leased blocks (``ref_cnt`` held by the lease) — disjoint from every
      live request and from the prefix cache (``get_new_blocks`` evicts
      cached hashes on lease).
    * Legacy fallback (no lease obtainable): inference is ABORTED first —
      restoring the abort-before-overwrite invariant that the plugin's
      ``/generate`` path silently lacked — and the prefix cache is reset
      on exit, because blocks ``0..N`` were overwritten without hash
      invalidation (a prefix hit would otherwise serve poisoned KV).

Concurrency: the FIFO ``asyncio.Lock`` serializes validations in this
API-server process, so queued validations WAIT instead of each tripping
the abort-inference escalation, and the peak KV lease stays 1×. With
``--api-server-count > 1`` the bound degrades to P×; a cross-process cap
would have to live in EngineCore (the single pool owner). Single process
today. Mining (``/init/generate``) never takes this lock: its priority is
enforced by the callers' ``_is_generation_active`` spins.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple

from gonka_poc._compat import current as _compat_current

logger = logging.getLogger(__name__)

# Total budget for obtaining a lease (poll + escalation re-poll each get
# one window). Matches the upstream default order of magnitude.
POC_BORROW_TIMEOUT_MS = int(os.getenv("POC_BORROW_TIMEOUT_MS", "3000"))

_BORROW_POLL_SEC = 0.05
_ABORT_SETTLE_SEC = 0.05

# FIFO per-process lock — see module docstring.
_poc_reservation_lock = asyncio.Lock()

# Lazily-cached borrow availability per engine client. Weakly keyed so a
# dead client cannot alias a new object at the same address.
_borrow_available: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


async def poc_validation_available(engine_client: Any) -> bool:
    """Probe (once per engine client) whether borrowed-lease validation is on.

    Two gates, both required (the legacy KV-scratch gate died with the
    prefill scheme — decode always prefills into a fresh buffer, decision
    #12, so a leased forward is bit-safe by construction):
      * the borrow RPC surface answers (a zero-block borrow returns None
        without raising — proves the injected EngineCore methods and the
        utility transport);
      * not DP>1 (the compat wrapper refuses — the internal LB fans the
        utility to every engine and returns only engine-0's lease).

    Probe failures report False (validation degrades to the abort-based
    legacy path); the result is cached for the engine client's lifetime.
    """
    try:
        cached = _borrow_available.get(engine_client)
    except TypeError:  # non-weakref-able client
        cached = None
    if cached is not None:
        return cached
    compat = _compat_current()
    available = False
    try:
        # Zero-block borrow: exercises the full RPC path; returns None
        # (needed <= 0) iff the surface is present and reachable.
        await compat.borrow_poc_blocks(engine_client, 0, 1)
        available = True
    except Exception as exc:
        logger.warning(
            "PoC borrow availability probe failed — validation will use the "
            "legacy abort-based path: %s", exc)
    try:
        _borrow_available[engine_client] = available
    except TypeError:
        pass
    logger.info("PoC borrowed-lease validation available: %s", available)
    return available


async def _borrow_with_retry(
    engine_client: Any,
    num_nonces: int,
    seq_len: int,
    deadline: float,
) -> Tuple[Optional[dict], bool]:
    """Poll the engine core for a lease until ``deadline``.

    Returns ``(lease, rpc_broken)``. ``rpc_broken=True`` means the RPC
    surface itself failed (feature not installed / transport error) — the
    caller must NOT keep polling or escalating for a better answer.
    """
    compat = _compat_current()
    while True:
        try:
            lease = await compat.borrow_poc_blocks(
                engine_client, num_nonces, seq_len)
        except Exception as exc:
            logger.warning(
                "PoC borrow RPC failed (engine lacks the borrow methods?): %s",
                exc)
            return None, True
        if lease is not None:
            return lease, False
        if time.monotonic() > deadline:
            return None, False
        await asyncio.sleep(_BORROW_POLL_SEC)


async def reserve_poc_blocks(
    engine_client: Any,
    num_nonces: int,
    seq_len: int,
    timeout_ms: int = POC_BORROW_TIMEOUT_MS,
) -> Optional[dict]:
    """Lease KV blocks, escalating once through an inference abort.

    Order: (1) poll-borrow within ``timeout_ms``; (2) on failure abort all
    in-flight inference — this is BOTH the escalation (freed blocks make the
    re-borrow succeed) AND the safety precondition for the legacy fallback
    (which overwrites blocks ``0..N`` in place); (3) re-poll unless the RPC
    surface itself is broken. Returns the lease dict or ``None`` — by the
    time ``None`` is returned, inference has been aborted, so the caller may
    safely run the legacy in-place path.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    lease, rpc_broken = await _borrow_with_retry(
        engine_client, num_nonces, seq_len, deadline)
    if lease is not None:
        return lease

    compat = _compat_current()
    try:
        aborted = await compat.abort_all_requests(engine_client)
    except Exception as exc:
        aborted = 0
        logger.error("PoC reservation: abort escalation failed: %s", exc)
    logger.warning(
        "PoC reservation: no lease within %dms%s — aborted %d in-flight "
        "request(s), %s",
        timeout_ms,
        " (borrow RPC broken)" if rpc_broken else "",
        aborted,
        "falling back to the legacy in-place path" if rpc_broken
        else "retrying the borrow once")
    await asyncio.sleep(_ABORT_SETTLE_SEC)

    if not rpc_broken:
        lease, _ = await _borrow_with_retry(
            engine_client, num_nonces, seq_len,
            time.monotonic() + timeout_ms / 1000.0)
        if lease is not None:
            return lease
    return None


@asynccontextmanager
async def poc_reservation(
    engine_client: Any,
    num_nonces: int,
    seq_len: int,
    timeout_ms: int = POC_BORROW_TIMEOUT_MS,
):
    """Serialized lease → (caller's forwards) → return, as a context manager.

    Yields the lease dict (``{"block_ids": [...], "blocks_per_seq": int}``)
    or ``None`` for the legacy fallback — inference has already been aborted
    by then. Block return is structurally guaranteed (``finally``); on the
    legacy path the prefix cache is reset instead (see module docstring).
    """
    async with _poc_reservation_lock:
        if await poc_validation_available(engine_client):
            lease = await reserve_poc_blocks(
                engine_client, num_nonces, seq_len, timeout_ms)
        else:
            # Legacy path from the start: abort once here; the per-chunk
            # re-abort in _execute_poc_decode_rpc keeps later admissions
            # out of the clobber range.
            lease = None
            try:
                await _compat_current().abort_all_requests(engine_client)
            except Exception as exc:
                logger.error("PoC legacy-path abort failed: %s", exc)
        try:
            yield lease
        finally:
            if lease is not None:
                await _return_lease_with_retry(engine_client, lease)
            else:
                await reset_prefix_cache_after_inplace_poc(engine_client)


async def _return_lease_with_retry(engine_client: Any, lease: dict) -> None:
    """Return leased blocks; a lost return shrinks the pool until restart,
    so retry transient RPC failures a few times before giving up loudly."""
    block_ids = lease["block_ids"]
    for attempt in range(3):
        try:
            await _compat_current().return_poc_blocks(engine_client, block_ids)
            return
        except Exception as exc:
            logger.warning(
                "PoC lease return attempt %d/3 failed: %s", attempt + 1, exc)
            await asyncio.sleep(0.2 * (attempt + 1))
    logger.error(
        "PoC LEAKED %d leased KV blocks (return failed 3x) — pool stays "
        "smaller and prefix-cache resets will fail until engine restart",
        len(block_ids))


async def reset_prefix_cache_after_inplace_poc(engine_client: Any) -> None:
    """Drop the prefix cache after an in-place (blocks ``0..N``) PoC round.

    The legacy path overwrites cached blocks WITHOUT evicting their hashes
    (``free_blocks`` keeps ``block_hash`` for reuse), so a later prefix hit
    would silently serve PoC garbage as KV. Best-effort: the engine may not
    expose ``reset_prefix_cache`` (then the operator must not enable prefix
    caching with PoC — pre-existing behaviour).
    """
    reset = getattr(engine_client, "reset_prefix_cache", None)
    if reset is None:
        logger.warning(
            "engine client lacks reset_prefix_cache — prefix cache may hold "
            "PoC-clobbered blocks after an in-place PoC round")
        return
    # BlockPool.reset_prefix_cache returns False (dropping NO hashes) unless
    # every block is free — a plain call can silently no-op while blocks are
    # held (e.g. an outstanding lease or live requests). Escalate once with
    # reset_running_requests=True (preempts running requests so they recompute
    # their KV) and report failure truthfully.
    try:
        ok = await reset()
    except Exception as exc:
        ok = False
        logger.warning("reset_prefix_cache failed: %s", exc)
    if ok is False:
        try:
            ok = await reset(reset_running_requests=True)
        except TypeError:
            pass  # older signature without the kwarg
        except Exception as exc:
            logger.warning(
                "reset_prefix_cache(reset_running_requests=True) failed: %s",
                exc)
    if ok is False:
        logger.error(
            "prefix cache NOT reset after in-place PoC round (blocks still "
            "held) — cached hashes of clobbered blocks survive; prefix hits "
            "may serve PoC garbage until a later successful reset")
    else:
        logger.info("prefix cache reset after in-place PoC round")
