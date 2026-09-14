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
from .validation import run_validation

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Backoff after a collective_rpc timeout in the mining loop.
POC_RPC_TIMEOUT_BACKOFF_SEC = 0.1

_poc_tasks: Dict[int, Dict[str, Any]] = {}


async def _execute_poc_forward_rpc(
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


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12


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
    vector_b64: str


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
    poc_stronger_rng: bool = False


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



async def _compute_artifacts_chunk(
    engine_client,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    poc_stronger_rng: bool = False,
    timeout_sec: float = POC_GENERATE_CHUNK_TIMEOUT_SEC,
    lease: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Compute artifacts for a chunk via collective_rpc.

    There is no longer a "skipped" backoff path: with a ``lease`` the
    forward runs on KV blocks disjoint from live inference, and on the
    legacy fallback (``lease is None``) the reservation layer has already
    aborted in-flight inference.
    """
    result = await _execute_poc_forward_rpc(
        engine_client,
        nonces=nonces,
        block_hash=block_hash,
        public_key=public_key,
        seq_len=seq_len,
        k_dim=k_dim,
        poc_stronger_rng=poc_stronger_rng,
        timeout_ms=int(timeout_sec * 1000),
        lease=lease,
    )
    return result.get("artifacts", [])


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
                result = await _execute_poc_forward_rpc(
                    engine_client,
                    nonces=nonces,
                    block_hash=config["block_hash"],
                    public_key=config["public_key"],
                    seq_len=config["seq_len"],
                    k_dim=config["k_dim"],
                    poc_stronger_rng=config["poc_stronger_rng"],
                    timeout_ms=POC_RPC_TIMEOUT_MS,
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
        # Mining wrote blocks 1..N in place without evicting their cached
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
        f"url={bool(body.url)} poc_stronger_rng={body.poc_stronger_rng}"
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
        "poc_stronger_rng": body.poc_stronger_rng,
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
        f"url={bool(body.url)} poc_stronger_rng={body.poc_stronger_rng}"
    )
    logger.debug(f"PoC /generate full body: {body}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        if validation_nonces != set(body.nonces):
            raise HTTPException(status_code=400, detail="validation.artifacts nonces must match nonces field")
    
    validation_map = {a.nonce: a.vector_b64 for a in body.validation.artifacts} if body.validation else None
    stat_test = body.stat_test or StatTestModel()
    
    if not body.wait:
        queue = get_queue()
        queue.set_generation_active_check(_is_generation_active)

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
            poc_stronger_rng=body.poc_stronger_rng,
            validation_artifacts=validation_map,
            stat_test_dist_threshold=stat_test.dist_threshold,
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
    
    while _is_generation_active(app_id):
        await asyncio.sleep(GENERATION_ACTIVE_POLL_SEC)

    total_nonces = len(body.nonces)
    n_chunks = (total_nonces + body.batch_size - 1) // body.batch_size
    logger.info(f"PoC /generate: {total_nonces} nonces, batch_size={body.batch_size}, chunks={n_chunks}")

    start_time = time.time()
    computed_artifacts = []

    # One lease per request, reused across chunks; lease=None ⇒ inference
    # already aborted, legacy in-place layout — see poc_reservation.
    async with poc_reservation(
        engine_client, body.batch_size, body.params.seq_len,
    ) as lease:
        for i in range(0, total_nonces, body.batch_size):
            chunk = body.nonces[i:i + body.batch_size]
            chunk_idx = i // body.batch_size

            while _is_generation_active(app_id):
                await asyncio.sleep(GENERATION_ACTIVE_POLL_SEC)

            try:
                artifacts = await _compute_artifacts_chunk(
                    engine_client, chunk, body.block_hash, body.public_key,
                    body.params.seq_len, body.params.k_dim, body.poc_stronger_rng,
                    POC_GENERATE_CHUNK_TIMEOUT_SEC,
                    lease=lease,
                )
                computed_artifacts.extend(artifacts)
                logger.debug(f"PoC /generate: chunk {chunk_idx+1}/{n_chunks} done ({len(chunk)} nonces)")
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
    
    elapsed = time.time() - start_time
    rate = total_nonces / elapsed if elapsed > 0 else 0
    logger.info(f"PoC /generate completed: {total_nonces} nonces in {elapsed:.2f}s ({rate:.0f}/s)")
    
    if not body.validation:
        return {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "artifacts": computed_artifacts,
            "encoding": wire_encoding(body.params.k_dim),
        }
    
    validation_result = run_validation(
        computed_artifacts=computed_artifacts,
        validation_map=validation_map,
        n_total=len(body.nonces),
        dist_threshold=stat_test.dist_threshold,
        p_mismatch=stat_test.p_mismatch,
        fraud_threshold=stat_test.fraud_threshold,
        k_dim=body.params.k_dim,
    )
    
    return {
        "status": "completed",
        "request_id": str(uuid.uuid4()),
        **validation_result,
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
    probe (borrow RPC reachable AND every rank derives inputs scratch-free) —
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
