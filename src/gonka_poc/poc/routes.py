"""PoC API routes for vLLM server."""
import asyncio
import contextlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator

import logging
from gonka_poc.mixed.policy import poc_cfg
from gonka_poc.poc.config import PoCState
from gonka_poc.poc.data import (
    Artifact, DEFAULT_DIST_THRESHOLD, DEFAULT_MARGIN_TAU, DEFAULT_P_MISMATCH,
    DEFAULT_FRAUD_THRESHOLD,
)
from gonka_poc.poc.callbacks import CallbackSender
from gonka_poc.poc.generate_queue import (
    GenerateJob, get_queue, clear_queue, POC_MAX_QUEUED_NONCES,
    compute_nonce_artifacts, drain_poc,
)
from gonka_poc.poc.reservation import poc_reservation
from gonka_poc.poc.validation import run_validation
from gonka_poc._compat import current as _compat_current

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

def _server_engine() -> dict:
    """Engine identity of the SERVING box: version/commit, attention backend,
    cudagraph mode. Server truth — never recorded client-side."""
    out = {}
    try:
        import vllm
        out["vllm_version"] = getattr(vllm, "__version__", "?")
    except Exception:
        pass
    try:
        import os
        out["attention_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
        out["v2_runner"] = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "?")
    except Exception:
        pass
    return out


def _server_gpu() -> str:
    """The SERVING box's GPU — provenance must name the machine that computed
    the artifacts, not whatever client collected them."""
    try:
        import torch
        n = torch.cuda.device_count()
        return f"{n}x{torch.cuda.get_device_name(0)}" if n else "cpu"
    except Exception:
        return "?"



POC_CALLBACK_INTERVAL_SEC = float(os.environ.get("POC_CALLBACK_INTERVAL_SEC", "5"))
POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))
POC_CHAT_BUSY_BACKOFF_SEC = 0.05
POC_RPC_TIMEOUT_MS = int(os.environ.get("POC_RPC_TIMEOUT_MS", "60000"))
# Request default when the chain sends no batch_size: 32, as in 3.0.16.
POC_BATCH_SIZE_DEFAULT = int(os.environ.get("POC_BATCH_SIZE_DEFAULT", "32"))

_poc_tasks: Dict[int, Dict[str, Any]] = {}

def resolve_mining_round(configured: int, engine_client, seq_len: int = 0,
                         prefill: bool = True) -> int:
    """How many nonces continuous mining pulls per iteration.

    The chain's batch_size, verbatim, as in 3.0.16 (the request default is
    POC_BATCH_SIZE_DEFAULT = 32). The decode scheme alone treats 0 as AUTO —
    poc_max_batch_size from ``--additional-config``, then max_num_seqs — since its
    rows are ordinary scheduler requests capped by max_num_seqs. A prefill-scheme
    round is one forward; whether batch_size x seq_len fits the node is the
    operator's configuration, the node does not resize it."""
    if configured or prefill:
        return configured
    vc = getattr(engine_client, "vllm_config", None)
    sc = getattr(vc, "scheduler_config", None)
    resolved = int(poc_cfg(vc, "poc_max_batch_size") or 0) or getattr(sc, "max_num_seqs", 0)
    if resolved:
        return resolved
    logger.warning("PoC mining: engine config unreadable, defaulting round to 32")
    return 32


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12
    # Which proof the chain is asking for. Explicit, because the two are
    # different derivations and a node must never guess: "prefill" is the
    # v0.1.x scheme whose artifacts the deployed fleet validates, "decode"
    # is the chained sphere_k trajectory. Absent => prefill, so a chain that
    # knows nothing about decode keeps working unchanged.
    scheme: Literal["prefill", "decode"] = "prefill"
    # The same switch as a flag, mirroring PoCParams.poc_decode: the chain
    # sends {"decode": true, "max_tokens": N}. Either form selects decode.
    decode: bool = False
    # Decode steps. Only read for the decode scheme.
    max_tokens: int = 0

    @model_validator(mode="after")
    def _decode_flag(self):
        if self.decode:
            self.scheme = "decode"
        elif self.scheme == "decode":
            self.decode = True
        return self


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
    """Iterator for nonces with multi-node and multi-group support."""
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
    k_points_steps: Optional[List[int]] = None
    n_sphere_mismatches: Optional[int] = None
    sph_indices_steps: Optional[List[List[int]]] = None
    sph_values_steps: Optional[List[str]] = None


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
    enforced_k_steps: Optional[Dict[int, List[int]]] = None
    debug: bool = False
    # Per-nonce Householder seeding (see PoCParams.per_nonce_reflection).
    # Forward-affecting: a validation request MUST carry the same value the
    # reference artifacts were generated with, or every chain diverges.
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
                        "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim, "max_tokens": params.max_tokens},
                        "deployed": {"model": list(valid_models), "seq_len": None, "k_dim": None, "max_tokens": None},
                    }
                )

    deployed = getattr(request.app.state, 'poc_deployed', None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")
        # max_tokens defines the decode trajectory length -> artifact-defining,
        # so it must match the deployed config like seq_len/k_dim. Use "is not
        # None" since max_tokens=0 (prefill-only) is a valid configured value.
        if deployed.get("max_tokens") is not None and params.max_tokens != deployed["max_tokens"]:
            mismatches.append("max_tokens")

        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim, "max_tokens": params.max_tokens},
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
    poc_decode: bool = False,
    max_tokens: int = 0,
    enforced_k_steps: Optional[Dict[int, List[int]]] = None,
    debug: bool = False,
    per_nonce_reflection: bool = False,
    timeout_sec: float = POC_GENERATE_CHUNK_TIMEOUT_SEC,
    block_height: int = 0,
    lease: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Compute artifacts for a chunk of nonces via the scheduler.

    Thin wrapper over generate_queue.compute_nonce_artifacts (the single source
    of truth for PoC artifact computation). ``timeout_sec`` is accepted only for
    call compatibility; the scheduler handles queuing/backoff.
    """
    return await compute_nonce_artifacts(
        engine_client, nonces, block_hash, public_key, block_height,
        seq_len, k_dim,
        poc_decode=poc_decode,
        max_tokens=max_tokens,
        enforced_k_steps=enforced_k_steps,
        debug=debug,
        per_nonce_reflection=per_nonce_reflection,
        poc_stronger_rng=poc_stronger_rng,
        lease=lease,
    )


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
    # Continuous mining pulls a round of nonces per iteration. 0 = AUTO -> ask the ENGINE
    # how many PoC sequences it can hold (poc_max_batch_size, auto-scaled to max_num_seqs)
    # instead of a client-side constant, so a bigger machine mines a bigger round.
    poc_decode = config.get("scheme", "prefill") == "decode"
    batch_size = resolve_mining_round(config["batch_size"], engine_client,
                                      config["seq_len"], prefill=not poc_decode)

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
                # Continuous generation: prefill-only when max_tokens==0 (default),
                # or decode-PoC (sphere_k trajectory) when max_tokens>0. PoC rides
                # through the scheduler alongside chat; no collective_rpc.
                mt = config.get("max_tokens", 0)
                artifacts = await _compute_artifacts_chunk(
                    engine_client, nonces,
                    config["block_hash"], config["public_key"],
                    config["seq_len"], config["k_dim"],
                    config["poc_stronger_rng"],
                    poc_decode=(config.get("scheme", "prefill") == "decode"),
                    max_tokens=mt,
                    block_height=config["block_height"],
                )
            except (TimeoutError, asyncio.TimeoutError):
                # Engine busy: retry the same chunk, as in 3.0.16. Any other
                # error ends the round (the task's done-callback logs it and
                # releases the gate).
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(f"PoC timed out (#{timeout_count}), engine busy")
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue

            timeout_count = 0
            pending_nonces = None

            if artifacts and callback_sender:
                artifact_objs = [Artifact(nonce=a["nonce"], vector_b64=a["vector_b64"],
                                          k_points_steps=a.get("k_points_steps"),
                                          sph_values_steps=a.get("sph_values_steps"))
                                 for a in artifacts]
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


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    logger.info(f"PoC /init/generate: {body.block_hash}, {body.block_height}, {body.public_key}, {body.node_id}, {body.node_count}, {body.group_id}, {body.n_groups}, {body.batch_size}, {body.params}, {body.url}, {body.poc_stronger_rng}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)

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
        "max_tokens": body.params.max_tokens,
        "k_dim": body.params.k_dim,
        "scheme": body.params.scheme,
        "poc_stronger_rng": body.poc_stronger_rng,
    }
    
    stats = {"start_time": 0, "total_processed": 0}
    stop_event = asyncio.Event()
    
    # A mining round owns the node, whichever scheme: live inference is gated
    # off (503) and drained first, as in 0.1.3 (ADR-0013 ordering: activate ->
    # abort -> spawn), and /stop or the round's end re-opens it. The prefill
    # scheme needs this for correctness (its forward writes KV blocks 1..N in
    # place); the decode scheme could share the scheduler with chat, but a
    # round next to live chat starves both sides, and the chain's UX is
    # "PoC runs, inference pauses, artifacts get published".
    gate = getattr(request.app.state, "gonka_gate", None)
    if gate is None:
        raise HTTPException(
            status_code=503,
            detail="PoCGate not installed on app.state.gonka_gate; a mining round "
                   "cannot run next to live inference")
    gate.activate("init-generate")

    callback_sender = None
    callback_task = None
    try:
        if body.url:
            callback_sender = CallbackSender(body.url, stop_event, body.params.k_dim)
            callback_task = asyncio.create_task(callback_sender.run())

        if gate is not None:
            aborted = await _compat_current().abort_all_requests(engine_client)
            logger.info("PoC init: aborted %d in-flight requests before generation",
                        aborted)

        gen_task = asyncio.create_task(
            _generation_loop(engine_client, stop_event, callback_sender, config, stats)
        )

        def _on_generation_done(task: asyncio.Task):
            if gate is not None:
                gate.deactivate()
            if task.cancelled():
                logger.info("PoC generation task cancelled, gate released")
            elif task.exception():
                logger.warning("PoC generation task failed, gate released: %s",
                               task.exception())
            else:
                logger.info("PoC generation task completed, gate released")

        gen_task.add_done_callback(_on_generation_done)
    except Exception:
        # Nothing between activate() and add_done_callback() may leave the
        # gate latched ON across operator retries.
        if callback_task is not None:
            stop_event.set()
            callback_task.cancel()
        if gate is not None:
            gate.deactivate()
        logger.exception("PoC init: failed before the generation task was spawned")
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
    # Summarize validation in the log: with debug or poc_vector_artifacts
    # refs its artifacts carry
    # per-step sph_values_steps (multi-MB) — never dump the body wholesale.
    val_log = (f"validation[{len(body.validation.artifacts)} artifacts]"
               if body.validation else None)
    logger.info(
        f"PoC /generate: {body.block_hash}, {body.block_height}, "
        f"{body.public_key}, {body.node_id}, {body.node_count}, {body.nonces}, "
        f"{body.params}, {body.batch_size}, {body.wait}, {body.url}, "
        f"{val_log}, {body.stat_test}, {body.poc_stronger_rng}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)

    app_id = id(request.app)

    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        if validation_nonces != set(body.nonces):
            raise HTTPException(status_code=400, detail="validation.artifacts nonces must match nonces field")

    enforced_k_steps = body.enforced_k_steps
    if (body.validation and body.params.scheme == "decode"
            and enforced_k_steps is None):
        # The wire form carries the reference trajectory inside each artifact.
        # Without teacher-forcing it, every computed artifact comes back with
        # n_sphere_mismatches=-1 (nothing compared) and the verdict is vacuous.
        missing = [a.nonce for a in body.validation.artifacts
                   if not a.k_points_steps]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=("decode validation needs k_points_steps for every "
                        f"artifact, missing for nonces {missing[:8]}"))
        enforced_k_steps = {a.nonce: list(a.k_points_steps)
                            for a in body.validation.artifacts}

    validation_map = {a.nonce: a.vector_b64 for a in body.validation.artifacts} if body.validation else None
    # prover-side pre-snap slices (reference generated with debug or
    # poc_vector_artifacts):
    # lets run_validation attach the continuous vector-channel score as evidence.
    ref_vectors = {a.nonce: a.sph_values_steps for a in body.validation.artifacts
                   if a.sph_values_steps} if body.validation else None
    stat_test = body.stat_test or StatTestModel(
        dist_threshold=(DEFAULT_MARGIN_TAU if body.params.scheme == "decode"
                        else DEFAULT_DIST_THRESHOLD))
    
    if not body.wait:
        queue = get_queue()
        queue.set_generation_active_check(_is_generation_active)
        
        if queue.queued_nonces + len(body.nonces) > POC_MAX_QUEUED_NONCES:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full: {queue.queued_nonces} nonces queued, limit is {POC_MAX_QUEUED_NONCES}"
            )
        
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
            poc_decode=(body.params.scheme == "decode"),
            max_tokens=body.params.max_tokens,
            enforced_k_steps=enforced_k_steps,
            debug=body.debug,
            per_nonce_reflection=body.per_nonce_reflection,
            validation_artifacts=validation_map,
            ref_vectors=ref_vectors,
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
        await asyncio.sleep(0.1)
    
    total_nonces = len(body.nonces)
    step = body.batch_size or total_nonces or 1   # 0 = submit all; engine batches
    n_chunks = (total_nonces + step - 1) // step
    logger.info(f"PoC /generate: {total_nonces} nonces, batch_size={body.batch_size}, chunks={n_chunks}")

    start_time = time.time()
    computed_artifacts = []
    poc_decode = body.params.scheme == "decode"

    # One lease per request, reused across chunks; lease=None => inference has
    # already been aborted and the forward falls back to the legacy in-place
    # layout over blocks 1..N -- see poc_reservation. Decode is deliberately
    # left outside: it shares the scheduler with live chat by design.
    async with contextlib.AsyncExitStack() as _stack:
        lease = None if poc_decode else await _stack.enter_async_context(
            poc_reservation(engine_client, step, body.params.seq_len))
        for i in range(0, total_nonces, step):
            chunk = body.nonces[i:i + step]
            chunk_idx = i // step

            while _is_generation_active(app_id):
                await asyncio.sleep(0.1)

            chunk_inference_steps = None
            if enforced_k_steps:
                chunk_inference_steps = {n: enforced_k_steps[n]
                                         for n in chunk if n in enforced_k_steps}

            try:
                artifacts = await _compute_artifacts_chunk(
                    engine_client, chunk, body.block_hash, body.public_key,
                    body.params.seq_len, body.params.k_dim, body.poc_stronger_rng,
                    poc_decode=poc_decode,
                    max_tokens=body.params.max_tokens,
                    enforced_k_steps=chunk_inference_steps,
                    debug=body.debug,
                    per_nonce_reflection=body.per_nonce_reflection,
                    timeout_sec=POC_GENERATE_CHUNK_TIMEOUT_SEC,
                    block_height=body.block_height,
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
            "encoding": {"dtype": "f16", "k_dim": body.params.k_dim, "endian": "le"},
            "server_gpu": _server_gpu(),
            "server_engine": _server_engine(),
        }
    
    if len(computed_artifacts) != len(body.nonces):
        # A verdict is only meaningful over the full requested nonce set. Missing
        # artifacts (dead engine, timeout) are NOT evidence of honesty: the
        # mismatch counters simply never see those nonces, so the rate collapses
        # toward zero and a broken validator would clear everyone it fails on.
        raise HTTPException(
            status_code=503,
            detail=(f"validation aborted: {len(computed_artifacts)} of "
                    f"{len(body.nonces)} nonces produced an artifact"))

    try:
        validation_result = run_validation(
            computed_artifacts=computed_artifacts,
            validation_map=validation_map,
            n_total=len(body.nonces),
            dist_threshold=stat_test.dist_threshold,
            p_mismatch=stat_test.p_mismatch,
            fraud_threshold=stat_test.fraud_threshold,
            k_dim=body.params.k_dim,
            # decode flow (max_tokens>0) → per-nonce snap margin vs dist_threshold
            # (tau); prefill flow → vector-L2 vs dist_threshold. Same binomial,
            # same response shape.
            use_trajectory=body.params.max_tokens > 0,
            ref_vectors=ref_vectors,
        )
    except ValueError as e:
        # No comparison happened for some artifact: not a verdict.
        raise HTTPException(status_code=503, detail=f"validation aborted: {e}")

    return {
        "status": "completed",
        "request_id": str(uuid.uuid4()),
        # debug: expose the validator's own artifacts (incl. sph_values_steps) so a
        # client can pair them with the prover's for offline vector-channel analysis;
        # verdict-only response otherwise (unchanged).
        "artifacts": computed_artifacts if body.debug else [],
        "server_gpu": _server_gpu(),
            "server_engine": _server_engine(),
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
    """Feature-detection handshake for the network node (ADR-0015 §6).

    ``poc_validation_inference`` used to reflect a borrow-RPC probe: whether
    validation could run on leased KV blocks while inference kept serving.
    Mixed decode-PoC removes the lease mechanism because coexistence is no
    longer conditional — PoC and chat share the scheduler and the batch, so
    validation always runs alongside inference. The field stays because the
    node reads it to decide whether it may keep serving during a round; it is
    now unconditionally true rather than a probe.
    """
    from vllm import __version__ as vllm_version
    try:
        import importlib.metadata as _md
        gonka_poc_version = _md.version("gonka-poc")
    except Exception:
        gonka_poc_version = "unknown"
    return {
        "vllm_version": vllm_version,
        "gonka_poc_version": gonka_poc_version,
        "poc_validation_inference": True,
    }


@router.get("/status")
async def get_status(request: Request) -> dict:
    return _get_api_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    app_id = id(request.app)

    await _cancel_poc_tasks(app_id)
    # The generation task's done-callback releases the gate; clear it here too
    # so a stop always re-opens inference (deactivate is idempotent).
    gate = getattr(request.app.state, "gonka_gate", None)
    if gate is not None:
        gate.deactivate()
    await clear_queue()
    # Cancelling the task does not evict requests already inside the engine:
    # a round started while they drain shares a forward with them and every
    # trajectory in it comes out different. Report STOPPED only once idle.
    await drain_poc()
    return {"status": "OK", "pow_status": {"status": "STOPPED"}}
