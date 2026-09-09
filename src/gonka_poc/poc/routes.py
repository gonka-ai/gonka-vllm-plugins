"""PoC API routes for vLLM server."""
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict

from vllm.logger import init_logger
from gonka_poc._compat import current as _compat_current
from .config import (
    GENERATION_ACTIVE_POLL_SEC,
    POC_BATCH_SIZE_DEFAULT,
    POC_GENERATE_CHUNK_TIMEOUT_SEC,
    POC_MAX_QUEUED_NONCES,
    POC_RPC_TIMEOUT_MS,
    PoCState,
)
from .data import Artifact, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD, wire_encoding
from .callbacks import CallbackSender
from .generate_queue import GenerateJob, get_queue, clear_queue
from .reservation import (
    poc_reservation,
    poc_validation_available,
    reset_prefix_cache_after_inplace_poc,
)

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Backoff after a collective_rpc timeout in the mining loop.
POC_RPC_TIMEOUT_BACKOFF_SEC = 0.1

_poc_tasks: Dict[int, Dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12
    # Decode-PoC (the canonical scheme of this release). max_tokens == 0 keeps
    # the prefill-only artifact of the DECODE seed scheme; max_tokens > 0
    # selects decode (a request property — there is no server-side flag).
    max_tokens: int = 0
    # Routing window: consensus value ships in the request from the Go node
    # reading the on-chain config; recorded in the artifact encoding
    # (decisions #6/#11, release value 256).
    route_window: int = 256   # shipped profile value (golden collected under it); 256 on 256 experts = legacy full scatter


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    batch_size: int = POC_BATCH_SIZE_DEFAULT
    params: PoCParamsModel
    url: Optional[str] = None
    # accepted-and-ignored: legacy prefill-scheme knob, kept so old callers
    # don't 422 during the rollout window
    poc_stronger_rng: bool = False


@dataclass
class NonceIterator:
    """Iterator for nonces with multi-node and multi-group support.

    Binding contract: the offset/step formula
    (nonce = node_id + group_id*n_nodes + x*(n_groups*n_nodes)) is the
    network-wide disjoint nonce-partition scheme — frozen; changing it
    breaks disjoint coverage across nodes/groups.
    """
    node_id: int
    n_nodes: int
    group_id: int
    n_groups: int
    _current_x: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value

    def take(self, n: int) -> List[int]:
        """Take the next n nonces."""
        return [next(self) for _ in range(n)]


class ArtifactModel(BaseModel):
    nonce: int
    vector_b64: str = ""
    # Decode-PoC trajectory (reference for teacher-forced validation).
    k_points_steps: Optional[List[int]] = None


class ValidationModel(BaseModel):
    artifacts: List[ArtifactModel]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = POC_BATCH_SIZE_DEFAULT
    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None
    # accepted-and-ignored: legacy prefill-scheme knob, kept so old callers
    # don't 422 during the rollout window
    poc_stronger_rng: bool = False
    # Decode-PoC: reference trajectories for teacher forcing (0.20 wire shape:
    # {nonce: [k0..kN]}); alternatively taken from validation.artifacts'
    # k_points_steps. Emission of pre-snap slices: debug (all steps) or the
    # leading va_steps window (a request parameter).
    enforced_k_steps: Optional[Dict[int, List[int]]] = None
    debug: bool = False
    poc_vector_artifact_steps: int = 0
    per_nonce_reflection: bool = False


# =============================================================================
# Helpers
# =============================================================================

async def get_engine_client(request: Request):
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed config. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, 'openai_serving_models', None)
    if serving_models and hasattr(serving_models, 'base_model_paths'):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim},
                        "deployed": {"model": list(valid_models), "seq_len": None, "k_dim": None},
                    }
                )
    
    # Optional integrator pin — set app.state.poc_deployed =
    # {'model': ..., 'seq_len': ..., 'k_dim': ...} to enforce full param
    # matching; nothing in this repo sets it, so by default only the model
    # name is checked.
    deployed = getattr(request.app.state, 'poc_deployed', None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")
        
        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim},
                    "deployed": deployed,
                }
            )


def _is_generation_active(app_id: int) -> bool:
    tasks = _poc_tasks.get(app_id)
    if not tasks:
        return False
    gen_task = tasks.get("gen_task")
    return gen_task is not None and not gen_task.done()


def _get_api_status(app_id: int) -> dict:
    tasks = _poc_tasks.get(app_id)
    
    if not tasks or not _is_generation_active(app_id):
        return {"status": PoCState.IDLE.value, "config": None, "stats": None}
    
    config = tasks.get("config", {})
    stats = tasks.get("stats", {})
    start_time = stats.get("start_time", 0)
    total_processed = stats.get("total_processed", 0)
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0
    
    return {
        "status": PoCState.GENERATING.value,
        "config": {
            "block_hash": config.get("block_hash"),
            "block_height": config.get("block_height"),
            "public_key": config.get("public_key"),
            "node_id": config.get("node_id"),
            "node_count": config.get("node_count"),
            "group_id": config.get("group_id"),
            "n_groups": config.get("n_groups"),
            "seq_len": config.get("seq_len"),
            "k_dim": config.get("k_dim"),
        },
        "stats": {
            "total_processed": total_processed,
            "nonces_per_second": nonces_per_second,
        },
    }


async def _cancel_poc_tasks(app_id: int):
    tasks = _poc_tasks.pop(app_id, None)
    if tasks:
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
            try:
                await tasks["gen_task"]
            except asyncio.CancelledError:
                pass
        # Cancel the callback aiohttp loop too -- on re-init / shutdown it
        # would otherwise keep POSTing until the process died. stop_event is
        # already set above; that should let run() exit cleanly, but cancel
        # as a belt-and-braces measure for the mid-POST case.
        if tasks.get("callback_task"):
            cb_task = tasks["callback_task"]
            if not cb_task.done():
                cb_task.cancel()
                try:
                    await cb_task
                except (asyncio.CancelledError, Exception):
                    pass
        if tasks.get("callback_sender"):
            tasks["callback_sender"].clear()



async def _execute_poc_decode_rpc(
    engine_client,
    *,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    max_tokens: int,
    route_window: int,
    enforced_k_steps: Optional[Dict[int, List[int]]] = None,
    debug: bool = False,
    va_steps: int = 0,
    per_nonce_reflection: bool = False,
    lease: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One decode-PoC chunk via collective_rpc; returns the driver's result."""
    kwargs = dict(
        block_hash=block_hash, public_key=public_key, nonces=nonces,
        seq_len=seq_len, max_tokens=max_tokens, route_window=route_window,
        enforced_k_steps=enforced_k_steps, debug=debug, va_steps=va_steps,
        per_nonce_reflection=per_nonce_reflection,
    )
    if lease is not None:
        kwargs["borrowed_block_ids"] = lease["block_ids"]
        kwargs["borrowed_stripe"] = lease["blocks_per_seq"]
    results = await asyncio.wait_for(
        engine_client.collective_rpc("execute_poc_decode", kwargs=kwargs),
        timeout=POC_RPC_TIMEOUT_MS / 1000.0 * max(1, max_tokens // 32),
    )
    result = next((r for r in results if r), None)
    if result is None:
        raise RuntimeError("decode chunk returned no result")
    return result


# =============================================================================
# Generation Loop
# =============================================================================

async def _generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    callback_sender: Optional[CallbackSender],
    config: dict,
    stats: dict,
):
    nonce_iter = NonceIterator(
        node_id=config["node_id"],
        n_nodes=config["node_count"],
        group_id=config["group_id"],
        n_groups=config["n_groups"],
    )
    batch_size = config["batch_size"]
    
    start_time = time.time()
    stats["start_time"] = start_time
    stats["total_processed"] = 0
    last_report_time = start_time
    
    logger.info(f"PoC generation started (node {config['node_id']}/{config['node_count']}, group {config['group_id']}/{config['n_groups']})")
    timeout_count = 0
    pending_nonces = None

    try:
        while not stop_event.is_set():
            nonces = pending_nonces if pending_nonces else nonce_iter.take(batch_size)

            try:
                result = await _execute_poc_decode_rpc(
                    engine_client,
                    nonces=nonces,
                    block_hash=config["block_hash"],
                    public_key=config["public_key"],
                    seq_len=config["seq_len"],
                    max_tokens=config["max_tokens"],
                    route_window=config["route_window"],
                )
                timeout_count = 0
            except (TimeoutError, asyncio.TimeoutError):
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(f"PoC timed out (#{timeout_count}), engine busy")
                pending_nonces = nonces
                await asyncio.sleep(POC_RPC_TIMEOUT_BACKOFF_SEC)
                continue

            pending_nonces = None
            artifacts = result.get("artifacts", [])
            
            if artifacts and callback_sender:
                artifact_objs = [Artifact(nonce=a["nonce"], vector_b64=a["vector_b64"]) for a in artifacts]
                callback_sender.add_artifacts(artifact_objs, {
                    "public_key": config["public_key"],
                    "block_hash": config["block_hash"],
                    "block_height": config["block_height"],
                    "node_id": config["node_id"],
                })
            
            stats["total_processed"] += len(nonces)
            
            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = stats["total_processed"] / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {stats['total_processed']} nonces ({rate:.0f}/min)")
                last_report_time = current_time
            
    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"PoC stopped: {stats['total_processed']} nonces in {elapsed_min:.2f}min")
    except Exception as e:
        logger.error(f"PoC generation crashed: {e}", exc_info=True)
        raise
    finally:
        # Mining wrote blocks 0..N in place without evicting their cached
        # hashes — drop the prefix cache so later hits cannot serve
        # PoC-clobbered KV (best-effort; see reservation module docstring).
        await reset_prefix_cache_after_inplace_poc(engine_client)


# =============================================================================
# API Endpoints
# =============================================================================

def _get_gate(request: Request):
    """Return the per-app PoCGate.

    The gate is installed on ``app.state.gonka_gate`` by
    :func:`gonka_poc.entrypoint.api_router.build_gonka_app`. If it's
    missing the API server was not composed via that helper -- raise
    500 so the operator notices the wiring bug immediately.
    """
    gate = getattr(request.app.state, "gonka_gate", None)
    if gate is None:
        raise HTTPException(
            status_code=500,
            detail="PoCGate not installed on app.state.gonka_gate "
            "(gonka_poc.entrypoint.api_router.build_gonka_app must run "
            "before the PoC router accepts traffic).",
        )
    return gate


@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    logger.info(
        f"PoC /init/generate: block_hash={body.block_hash} "
        f"block_height={body.block_height} node={body.node_id}/{body.node_count} "
        f"group={body.group_id}/{body.n_groups} batch_size={body.batch_size} "
        f"url={bool(body.url)}"
    )
    logger.debug(f"PoC /init/generate full body: {body}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    gate = _get_gate(request)

    app_id = id(request.app)

    if _is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")

    await _cancel_poc_tasks(app_id)

    config = {
        "block_hash": body.block_hash,
        "block_height": body.block_height,
        "public_key": body.public_key,
        "node_id": body.node_id,
        "node_count": body.node_count,
        "group_id": body.group_id,
        "n_groups": body.n_groups,
        "batch_size": body.batch_size,
        "seq_len": body.params.seq_len,
        "k_dim": body.params.k_dim,
        "max_tokens": body.params.max_tokens,
        "route_window": body.params.route_window,
    }

    stats = {"start_time": 0, "total_processed": 0}
    stop_event = asyncio.Event()

    callback_sender = None
    callback_task = None

    # Activate the gate BEFORE creating the task so PoCGatingMiddleware
    # starts returning 503 to /v1/chat/completions and /v1/completions
    # immediately. The gate is the single source of truth for "PoC is
    # currently running" -- no module-level flag.
    #
    # CRITICAL: gate.activate() flips ON before any guarded call. If the
    # compat lookup or abort_all_requests raises, the gate would latch ON
    # forever (no done-callback registered yet because no task spawned).
    # Wrap activate -> abort -> spawn in try/except that deactivates the
    # gate on any exception before re-raising. The done-callback below
    # handles deactivation for the post-spawn happy path.
    #
    # Also: spawn callback_task INSIDE the try-block so an exception between
    # spawn and the _poc_tasks store does not orphan the aiohttp loop --
    # without this, CallbackSender.run() would keep hammering body.url
    # forever (no stop_event signal, no cancel, no termination short of
    # a process restart).
    gate.activate("init-generate")
    try:
        if body.url:
            callback_sender = CallbackSender(body.url, stop_event, body.params.k_dim)
            callback_task = asyncio.create_task(callback_sender.run())

        # Abort any already-admitted chat/completions requests that snuck in
        # before the gate flipped. PoCGatingMiddleware blocks NEW admissions;
        # abort_all_requests() drains the in-flight set so PoC forwards run on
        # an exclusively-owned GPU. Ordering contract (ADR-0013): gate.activate
        # -> abort_all_requests -> spawn gen task. This depends on the compat
        # dispatch shim for the `current()` module lookup.
        compat = _compat_current()
        aborted = await compat.abort_all_requests(engine_client)
        logger.info(
            "PoC init: aborted %d in-flight requests before generation", aborted
        )

        gen_task = asyncio.create_task(
            _generation_loop(engine_client, stop_event, callback_sender, config, stats)
        )

        def _on_generation_done(task: asyncio.Task):
            gate.deactivate()
            if task.cancelled():
                logger.info("PoC generation task cancelled, gate deactivated")
            elif task.exception():
                logger.warning("PoC generation task failed, gate deactivated: %s",
                               task.exception())
            else:
                logger.info("PoC generation task completed, gate deactivated")

        gen_task.add_done_callback(_on_generation_done)
    except Exception:
        # Anything between activate() and add_done_callback() failing means
        # the done-callback path will never deactivate. Deactivate here so
        # the gate does not latch ON across operator retries.
        #
        # If callback_task was spawned before the failure (body.url set +
        # compat / abort / spawn raised), tear it down too: set the
        # stop_event so CallbackSender.run() exits its loop cleanly, wait a
        # bounded time for in-flight aiohttp POST to wrap up, then cancel
        # if it overran. 5.0s is a pragmatic ceiling -- long enough for a
        # mid-flight POST + one backoff sleep, short enough that operator
        # retries (which call this same path) do not stack.
        if callback_task is not None:
            stop_event.set()
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                callback_task.cancel()
                try:
                    await callback_task
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass
        gate.deactivate()
        logger.exception(
            "PoC init: failed between gate.activate() and task spawn; "
            "gate deactivated (init-generate-failed) before re-raising"
        )
        raise
    
    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "callback_task": callback_task,
        "callback_sender": callback_sender,
        "stop_event": stop_event,
        "config": config,
        "stats": stats,
    }
    
    return {"status": "OK", "pow_status": {"status": "GENERATING"}}


@router.post("/generate")
async def generate(request: Request, body: PoCGenerateRequest) -> dict:
    logger.info(
        f"PoC /generate: block_hash={body.block_hash} "
        f"block_height={body.block_height} node={body.node_id}/{body.node_count} "
        f"batch_size={body.batch_size} nonces={len(body.nonces)} "
        f"wait={body.wait} validation={bool(body.validation)} "
        f"url={bool(body.url)}"
    )
    logger.debug(f"PoC /generate full body: {body}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        if validation_nonces != set(body.nonces):
            raise HTTPException(status_code=400, detail="validation.artifacts nonces must match nonces field")
    
    stat_test = body.stat_test or StatTestModel()
    
    if not body.wait:
        queue = get_queue()
        queue.set_generation_active_check(_is_generation_active)

        enforced = body.enforced_k_steps
        if enforced is None and body.validation:
            traj = {a.nonce: a.k_points_steps
                    for a in body.validation.artifacts if a.k_points_steps}
            enforced = traj or None
        job = GenerateJob(
            request_id=str(uuid.uuid4()),
            engine_client=engine_client,
            app_id=app_id,
            block_hash=body.block_hash,
            block_height=body.block_height,
            public_key=body.public_key,
            node_id=body.node_id,
            node_count=body.node_count,
            nonces=body.nonces,
            seq_len=body.params.seq_len,
            k_dim=body.params.k_dim,
            batch_size=body.batch_size,
            max_tokens=body.params.max_tokens,
            route_window=body.params.route_window,
            enforced_k_steps=enforced,
            stat_test_p_mismatch=stat_test.p_mismatch,
            stat_test_fraud_threshold=stat_test.fraud_threshold,
            callback_url=body.url,
        )
        
        request_id = await queue.enqueue(job)
        if request_id is None:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full: {queue.queued_nonces} nonces queued, limit is {POC_MAX_QUEUED_NONCES}"
            )
        
        await queue.ensure_worker_running(engine_client, app_id)
        
        return {"status": "queued", "request_id": request_id, "queued_count": len(body.nonces)}
    
    # Decode-PoC is the ONLY scheme of this release (the prefill
    # scheme lives in the v0.1.x tags). max_tokens == 0 degenerates to a
    # prefill-only trajectory (one snap step) through the same decode loop.
    return await _generate_decode(request, body, engine_client, app_id)


async def _generate_decode(request: Request, body: PoCGenerateRequest,
                           engine_client, app_id: int) -> dict:
    """Decode-PoC wait-path: chunked prefill+steps loop on the workers.

    Chunk size: min(batch_size or AUTO, POC_DECODE_PREFILL_CHUNK) — the
    prefill of a chunk must fit the pre-sized MoE workspace (v0.1.3 lesson);
    batch_size == 0 means AUTO.
    """
    from gonka_poc.poc.decode_runner import POC_DECODE_MAX_BATCH
    from .data import fraud_test

    seq_len = body.params.seq_len
    max_tokens = body.params.max_tokens
    alloc_len = seq_len + max_tokens

    # teacher forcing reference: explicit field wins, else validation.artifacts
    enforced = body.enforced_k_steps
    if enforced is None and body.validation:
        traj = {a.nonce: a.k_points_steps for a in body.validation.artifacts
                if a.k_points_steps}
        enforced = traj or None
    validating = enforced is not None
    if validating:
        missing = [n for n in body.nonces if n not in enforced]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"enforced_k_steps missing for nonces {missing[:5]}...")

    # batch_size caps ONE RPC's nonce count (prefill sub-chunking happens
    # inside the runner); 0 = AUTO. The joint decode batch is what amortizes
    # MoE weight traffic — bigger is faster until KV runs out.
    chunk_size = body.batch_size if body.batch_size > 0 else POC_DECODE_MAX_BATCH
    chunk_size = min(chunk_size, POC_DECODE_MAX_BATCH)

    total = len(body.nonces)
    start_time = time.time()
    artifacts: List[dict] = []
    mismatch_total = 0
    steps_total = 0

    # Equal-size chunks (ceil split): a trailing 16-nonce chunk pays the same
    # 257 sequential steps as a full one, and uniform batches reuse one
    # captured step graph across the whole round.
    n_chunks = max(1, -(-total // chunk_size))
    base, rem = divmod(total, n_chunks)
    bounds = []
    off = 0
    for ci in range(n_chunks):
        size = base + (1 if ci < rem else 0)
        bounds.append((off, off + size))
        off += size
    async with poc_reservation(engine_client, chunk_size, alloc_len) as lease:
        for lo, hi in bounds:
            chunk = body.nonces[lo:hi]
            while _is_generation_active(app_id):
                await asyncio.sleep(GENERATION_ACTIVE_POLL_SEC)
            kwargs = dict(
                block_hash=body.block_hash,
                public_key=body.public_key,
                nonces=chunk,
                seq_len=seq_len,
                max_tokens=max_tokens,
                route_window=body.params.route_window,
                enforced_k_steps=(
                    {n: enforced[n] for n in chunk} if validating else None),
                debug=body.debug,
                va_steps=body.poc_vector_artifact_steps,
                per_nonce_reflection=body.per_nonce_reflection,
            )
            if lease is not None:
                kwargs["borrowed_block_ids"] = lease["block_ids"]
                kwargs["borrowed_stripe"] = lease["blocks_per_seq"]
            results = await asyncio.wait_for(
                engine_client.collective_rpc("execute_poc_decode",
                                             kwargs=kwargs),
                timeout=POC_RPC_TIMEOUT_MS / 1000.0 * max(1, max_tokens // 32),
            )
            result = next((r for r in results if r), None)
            if result is None:
                raise HTTPException(status_code=500,
                                    detail="decode chunk returned no result")
            artifacts.extend(result["artifacts"])
            steps_total += int(result.get("steps_total") or 0)
            if validating:
                mismatch_total += int(result.get("mismatch_total") or 0)

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    logger.info("PoC /generate decode: %d nonces x %d steps in %.2fs (%.2f/s)",
                total, max_tokens + 1, elapsed, rate)

    encoding = wire_encoding(body.params.k_dim)
    encoding["route_window"] = body.params.route_window   # echoed so the validator replays the same window
    encoding["seq_len"] = seq_len
    encoding["max_tokens"] = max_tokens

    if not validating:
        return {"status": "completed", "request_id": str(uuid.uuid4()),
                "artifacts": artifacts, "encoding": encoding}

    st = body.stat_test or StatTestModel()
    p_value, fraud = fraud_test(
        n_mismatch=mismatch_total, n_total=steps_total,
        p_mismatch=st.p_mismatch, fraud_threshold=st.fraud_threshold)
    rate_mism = mismatch_total / steps_total if steps_total else 0.0
    return {
        "status": "completed", "request_id": str(uuid.uuid4()),
        "artifacts": artifacts, "encoding": encoding,
        "n_mismatch": mismatch_total, "n_total": steps_total,
        "mismatch_rate": rate_mism, "p_value": p_value,
        "fraud_detected": fraud,
    }


@router.get("/generate/{request_id}")
async def get_generate_result(request: Request, request_id: str) -> dict:
    queue = get_queue()
    record = queue.get_result(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    response = {"status": record.status, "request_id": request_id}
    
    if record.status == "completed" and record.result:
        response.update(record.result)
    elif record.status == "failed" and record.error:
        response["error"] = record.error
    
    return response


@router.get("/versions")
async def get_versions(request: Request) -> dict:
    """Feature-detection handshake for the network node.

    ``poc_validation_inference`` advertises that ``/generate`` validation
    runs on leased KV blocks concurrently with live inference (port of
    gonka-ai/vllm qd/combine-poc-and-inference). It reflects an actual
    probe (borrow RPC reachable AND the config is not scratch-capable) —
    never a hardcoded literal.
    """
    from vllm import __version__ as vllm_version
    try:
        import importlib.metadata as _md
        gonka_poc_version = _md.version("gonka-poc")
    except Exception:
        gonka_poc_version = "unknown"
    engine_client = await get_engine_client(request)
    return {
        "vllm_version": vllm_version,
        "gonka_poc_version": gonka_poc_version,
        "poc_validation_inference":
            await poc_validation_available(engine_client),
    }


@router.get("/status")
async def get_status(request: Request) -> dict:
    return _get_api_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    app_id = id(request.app)

    await _cancel_poc_tasks(app_id)
    await clear_queue()

    # Deactivate the gate after task cancellation so the chat endpoint
    # cannot squeeze a request in between cancellation and gate clear.
    # NOTE: the gen_task done-callback also calls deactivate(); this is
    # idempotent (PoCGate.deactivate clears the flag unconditionally).
    gate = getattr(request.app.state, "gonka_gate", None)
    if gate is not None:
        gate.deactivate()
    return {"status": "OK", "pow_status": {"status": PoCState.STOPPED.value}}
