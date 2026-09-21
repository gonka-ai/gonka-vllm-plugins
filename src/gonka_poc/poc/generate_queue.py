"""PoC generate queue with bounded nonce cap and result store."""
import asyncio
import contextlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import logging
from gonka_poc.poc.validation import run_validation
from gonka_poc.poc.callbacks import get_callback_queue, clear_callback_queue
from gonka_poc.poc.data import DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD
from gonka_poc.poc.poc_params import PoCParams
from gonka_poc.poc.reservation import poc_reservation

logger = logging.getLogger(__name__)

# Decode-PoC request ids still inside the engine. /stop cancels the round's
# task, but requests already admitted keep draining; a round started while they
# drain shares a forward with them, and the different batch composition changes
# every trajectory in it (measured: 0/8 agreement on the overlapping probe).
_inflight: set = set()


async def drain_poc(timeout: float = 30.0) -> int:
    """Wait for in-flight PoC requests to finish. Returns how many were left."""
    deadline = time.monotonic() + timeout
    while _inflight and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    if _inflight:
        logger.warning("PoC drain timed out with %d request(s) in flight",
                       len(_inflight))
    return len(_inflight)


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
    """The SERVING box's GPU — provenance names the machine that computed the
    artifacts, never the client that collected them."""
    try:
        import torch
        n = torch.cuda.device_count()
        return f"{n}x{torch.cuda.get_device_name(0)}" if n else "cpu"
    except Exception:
        return "?"



# vLLM schedules lower values first under --scheduling-policy priority (chat
# requests carry 0): PoC rows go ahead of chat, and chat is what gets preempted
# when KV runs short. Under the default FCFS policy the value is ignored.
POC_REQUEST_PRIORITY = -1


async def compute_nonce_artifacts(
    engine_client,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    block_height: int,
    seq_len: int,
    k_dim: int,
    poc_decode: bool = False,
    max_tokens: int = 0,
    enforced_k_steps: Optional[Dict[int, List[int]]] = None,
    debug: bool = False,
    per_nonce_reflection: bool = False,
    poc_stronger_rng: bool = False,
    lease: Optional[dict] = None,
) -> List[dict]:
    """Compute PoC artifacts for a set of nonces.

    Both schemes are live in one process; ``params.scheme`` picks per request:

    * "prefill" — the v0.1.x scheme over ``collective_rpc``. It never sets the
      in-model PoC mask, so the wrappers stay at their identity branch and the
      artifacts are bit-identical to the shipped MLNode image. This is what
      the deployed fleet validates, so it is the default a chain gets when it
      sends nothing new.
    * "decode" — one PoC request per nonce through
      ``engine_client.generate(poc_params=...)``: the scheduler mixes them
      with live chat and the decode chain produces a sphere_k trajectory.

    This is the single source of truth for PoC artifact computation; both the
    /generate endpoint and the queue worker call it.
    """
    if not poc_decode:
        from gonka_poc.poc.prefill_path import compute_prefill_artifacts
        return await compute_prefill_artifacts(
            engine_client,
            nonces=nonces,
            block_hash=block_hash,
            public_key=public_key,
            seq_len=seq_len,
            k_dim=k_dim,
            poc_stronger_rng=poc_stronger_rng,
            lease=lease,
        )

    async def compute_one(nonce: int) -> Optional[dict]:
        inf_steps = (enforced_k_steps.get(nonce)
                     if enforced_k_steps else None)
        poc_params = PoCParams(
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonce=nonce,
            seq_len=seq_len,
            k_dim=k_dim,
            poc_decode=poc_decode,
            max_tokens=max_tokens,
            enforced_k_steps=inf_steps,
            debug=debug,
            per_nonce_reflection=per_nonce_reflection,
        )
        request_id = f"poc-{uuid.uuid4()}"
        _inflight.add(request_id)
        # PoC emits its artifact ONCE (emit-once): a single finished output
        # carrying the full trajectory (decode) or vector (prefill).
        try:
            async for output in engine_client.generate(
                prompt=None,
                sampling_params=None,
                poc_params=poc_params,
                request_id=request_id,
                priority=POC_REQUEST_PRIORITY,
            ):
                if not output.finished:
                    continue
                poc_out = output.poc_output
                if not poc_out:
                    # PoC ran but emitted no artifact. Silent until 31.08: the
                    # nonce was dropped from the result list and the caller saw
                    # a short batch with no reason given. The dominant cause is
                    # the KV capacity limit — nonces admitted beyond what the pool holds
                    # finish without a trajectory, and the ENGINE logs nothing:
                    # no preemption, no allocation failure. Say it here.
                    logger.warning(
                        "PoC nonce %s: no artifact emitted (request finished "
                        "with empty poc_output). Usually the KV capacity limit — the "
                        "batch asked for more nonces than the pool holds.", nonce)
                    return None
                get = poc_out.get if isinstance(poc_out, dict) else (
                    lambda k, d=None: getattr(poc_out, k, d))
                # sph_indices_steps is debug-only; sph_values_steps is emitted
                # under debug (full) or poc_vector_artifacts (windowed slice).
                artifact = {
                    "nonce": get("nonce", nonce),
                    "vector_b64": get("vector_b64", ""),
                    "k_points_steps": get("k_points_steps", []),
                    "n_sphere_mismatches": get("n_sphere_mismatches", -1),
                    "n_nan_steps": get("n_nan_steps", 0),
                    "mismatch_margin_max": get("mismatch_margin_max", 0.0),
                }
                if debug:
                    artifact["sph_indices_steps"] = get("sph_indices_steps", [])
                    artifact["sph_values_steps"] = get("sph_values_steps", [])
                else:
                    sph_vals = get("sph_values_steps", [])
                    if sph_vals:
                        artifact["sph_values_steps"] = sph_vals
                # Second silent path: the artifact IS emitted but its trajectory
                # is empty or short. The caller then gets a chain of length 0
                # among full ones, which a benchmark counts as work.
                # Only a length check catches it, and it costs one len() on data
                # already in hand — nothing on the happy path.
                if poc_decode and max_tokens:
                    got = len(artifact["k_points_steps"])
                    if got < max_tokens + 1:
                        logger.warning(
                            "PoC nonce %s: trajectory %d of %d steps%s. A short "
                            "chain is NOT work — it must not be scored.",
                            nonce, got, max_tokens + 1,
                            " (EMPTY)" if got == 0 else "")
                return artifact
        except Exception as e:
            logger.error("Error computing nonce %s: %r", nonce, e, exc_info=True)
        finally:
            _inflight.discard(request_id)
        return None

    # Every nonce goes to the engine at once; the scheduler admits them by
    # max_num_seqs and KV like a burst of chat requests (measured 10.09 on B300:
    # identical to a client-side window sized to the cudagraph capture).
    results = await asyncio.gather(*[compute_one(n) for n in nonces])
    out = [r for r in results if r is not None]
    # Batch-level summary. Per-nonce warnings would flood the log in a 500-nonce round;
    # this line states the shortfall once, in the terms an operator acts on.
    dropped = len(nonces) - len(out)
    short = sum(1 for r in out if poc_decode and max_tokens
                and len(r.get("k_points_steps") or []) < max_tokens + 1)
    if dropped or short:
        logger.warning(
            "PoC batch of %d: %d nonces produced no artifact, %d returned a "
            "short trajectory. %d of %d are usable. Lower the batch or raise "
            "KV — see the per-nonce warnings above.",
            len(nonces), dropped, short, len(nonces) - dropped - short, len(nonces))
    return out

POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))
POC_CHAT_BUSY_BACKOFF_SEC = 0.05
POC_GENERATE_RESULT_TTL_SEC = float(os.environ.get("POC_GENERATE_RESULT_TTL_SEC", "300"))
POC_MAX_QUEUED_NONCES = int(os.environ.get("POC_MAX_QUEUED_NONCES", "100000"))


@dataclass
class GenerateJob:
    """A queued /generate request."""
    request_id: str
    engine_client: Any
    app_id: int
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    seq_len: int
    k_dim: int
    batch_size: int
    poc_stronger_rng: bool = False
    poc_decode: bool = False
    max_tokens: int = 0
    enforced_k_steps: Optional[Dict[int, List[int]]] = None
    debug: bool = False
    per_nonce_reflection: bool = False
    validation_artifacts: Optional[Dict[int, str]] = None
    # nonce -> reference sph_values_steps (debug or poc_vector_artifacts refs):
    # enables the continuous vector_score on the queued path, same as the
    # inline wait=true path.
    ref_vectors: Optional[Dict[int, List[str]]] = None
    stat_test_dist_threshold: float = DEFAULT_DIST_THRESHOLD
    stat_test_p_mismatch: float = DEFAULT_P_MISMATCH
    stat_test_fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD
    callback_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class GenerateResult:
    """Result record for a queued /generate request."""
    status: str  # "queued", "running", "completed", "failed", "cancelled"
    nonce_count: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GenerateQueue:
    """Bounded queue for /generate jobs with result tracking."""
    
    def __init__(self):
        self._queue: asyncio.Queue[GenerateJob] = asyncio.Queue()
        self._results: Dict[str, GenerateResult] = {}
        self._queued_nonces: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._is_generation_active: Optional[Callable[[int], bool]] = None
        self._callback_queue = None  # Initialized lazily
    
    def set_generation_active_check(self, fn: Callable[[int], bool]):
        """Set callback to check if /init/generate is active."""
        self._is_generation_active = fn
    
    @property
    def queued_nonces(self) -> int:
        return self._queued_nonces
    
    async def enqueue(self, job: GenerateJob) -> Optional[str]:
        """Enqueue a job. Returns None if cap exceeded."""
        async with self._lock:
            new_total = self._queued_nonces + len(job.nonces)
            if new_total > POC_MAX_QUEUED_NONCES:
                return None
            
            self._queued_nonces = new_total
            self._results[job.request_id] = GenerateResult(
                status="queued",
                nonce_count=len(job.nonces)
            )
            await self._queue.put(job)
            return job.request_id
    
    def get_result(self, request_id: str) -> Optional[GenerateResult]:
        """Get result for a request_id."""
        return self._results.get(request_id)
    
    async def clear_all(self):
        """Clear queue and results."""
        async with self._lock:
            while not self._queue.empty():
                try:
                    job = self._queue.get_nowait()
                    if job.request_id in self._results:
                        self._results[job.request_id].status = "cancelled"
                        self._results[job.request_id].completed_at = time.time()
                except asyncio.QueueEmpty:
                    break
            
            self._queued_nonces = 0
            self._results.clear()
            self._stop_event.set()
    
    def cleanup_old_results(self):
        """Remove completed/failed results older than TTL."""
        now = time.time()
        expired = [
            rid for rid, rec in self._results.items()
            if rec.status in ("completed", "failed", "cancelled")
            and rec.completed_at
            and now - rec.completed_at > POC_GENERATE_RESULT_TTL_SEC
        ]
        for rid in expired:
            del self._results[rid]
    
    async def ensure_worker_running(self, engine_client, app_id: int):
        """Ensure the worker task is running."""
        if self._worker_task is None or self._worker_task.done():
            self._stop_event.clear()
            self._worker_task = asyncio.create_task(
                self._worker_loop(engine_client, app_id)
            )
    
    async def stop_worker(self):
        """Stop the worker task and callback queue."""
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        # Stop callback queue and clear global singleton
        if self._callback_queue:
            await self._callback_queue.stop()
            self._callback_queue = None
        await clear_callback_queue()
    
    async def _worker_loop(self, engine_client, app_id: int):
        """Background worker that processes queued jobs."""
        # Initialize callback queue with bounded concurrency
        self._callback_queue = get_callback_queue(self._stop_event)
        await self._callback_queue.start()

        logger.info("Generate queue worker started")
        
        while not self._stop_event.is_set():
            try:
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if job.request_id in self._results:
                    self._results[job.request_id].status = "running"
                
                try:
                    if self._is_generation_active:
                        while self._is_generation_active(job.app_id):
                            if self._stop_event.is_set():
                                break
                            await asyncio.sleep(0.1)
                    
                    if self._stop_event.is_set():
                        break
                    
                    result = await self._process_job(job)
                    
                    if job.request_id in self._results:
                        self._results[job.request_id].status = "completed"
                        self._results[job.request_id].completed_at = time.time()
                        self._results[job.request_id].result = result
                    
                    if job.callback_url:
                        self._enqueue_callback(job, result)
                    
                except Exception as e:
                    logger.error(f"Generate job {job.request_id} failed: {e}", exc_info=True)
                    if job.request_id in self._results:
                        self._results[job.request_id].status = "failed"
                        self._results[job.request_id].completed_at = time.time()
                        self._results[job.request_id].error = str(e)
                
                finally:
                    async with self._lock:
                        self._queued_nonces -= len(job.nonces)
                        self._queued_nonces = max(0, self._queued_nonces)
                
                self.cleanup_old_results()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Generate worker error: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info("Generate queue worker stopped")
    
    async def _process_job(self, job: GenerateJob) -> Dict[str, Any]:
        """Process a single generate job."""
        total_nonces = len(job.nonces)
        # batch_size 0 = no client-side chunking: submit every nonce at once and let the
        # ENGINE schedule them like chat. Chunking here awaits each chunk SEQUENTIALLY,
        # pinning in-flight nonces to the chunk size no matter what the engine can serve.
        step = job.batch_size or total_nonces
        n_chunks = (total_nonces + step - 1) // step
        logger.info(f"PoC queue job {job.request_id[:8]}: {total_nonces} nonces, batch_size={job.batch_size}, chunks={n_chunks}")

        start_time = time.time()
        computed_artifacts = []

        # One lease per job, reused across chunks -- see poc_reservation.
        # Decode stays outside it: it shares the scheduler with live chat.
        async with contextlib.AsyncExitStack() as _stack:
            lease = None if job.poc_decode else await _stack.enter_async_context(
                poc_reservation(job.engine_client, step, job.seq_len))
            for i in range(0, total_nonces, step):
                chunk = job.nonces[i:i + step]
                chunk_idx = i // step

                if self._stop_event.is_set():
                    raise RuntimeError("Job cancelled")

                chunk_inference_steps = None
                if job.enforced_k_steps:
                    chunk_inference_steps = {
                        n: job.enforced_k_steps[n]
                        for n in chunk if n in job.enforced_k_steps
                    }

                try:
                    artifacts = await compute_nonce_artifacts(
                        job.engine_client, chunk,
                        job.block_hash, job.public_key, job.block_height,
                        job.seq_len, job.k_dim,
                        poc_decode=job.poc_decode,
                        max_tokens=job.max_tokens,
                        enforced_k_steps=chunk_inference_steps,
                        debug=job.debug,
                        per_nonce_reflection=job.per_nonce_reflection,
                        poc_stronger_rng=job.poc_stronger_rng,
                        lease=lease,
                    )
                except asyncio.CancelledError:
                    logger.info(f"PoC queue job {job.request_id[:8]}: cancelled")
                    raise RuntimeError("Job cancelled")

                computed_artifacts.extend(artifacts)
                logger.debug(f"PoC queue job {job.request_id[:8]}: chunk {chunk_idx+1}/{n_chunks} done ({len(chunk)} nonces)")

        elapsed = time.time() - start_time
        rate = total_nonces / elapsed if elapsed > 0 else 0
        logger.info(f"PoC queue job {job.request_id[:8]} completed: {total_nonces} nonces in {elapsed:.2f}s ({rate:.0f}/s)")
        
        if job.validation_artifacts is None:
            return {
                "status": "completed",
                "request_id": job.request_id,
                "artifacts": computed_artifacts,
                "encoding": {"dtype": "f16", "k_dim": job.k_dim, "endian": "le"},
                "server_gpu": _server_gpu(),
            "server_engine": _server_engine(),
            }
        
        if len(computed_artifacts) != len(job.nonces):
            # Same rule as the wait path: a verdict over a partial nonce set is
            # not evidence of honesty, it is a failed job.
            raise RuntimeError(
                f"validation aborted: {len(computed_artifacts)} of "
                f"{len(job.nonces)} nonces produced an artifact")

        validation_result = run_validation(
            computed_artifacts=computed_artifacts,
            validation_map=job.validation_artifacts,
            n_total=len(job.nonces),
            dist_threshold=job.stat_test_dist_threshold,
            p_mismatch=job.stat_test_p_mismatch,
            fraud_threshold=job.stat_test_fraud_threshold,
            k_dim=job.k_dim,
            use_trajectory=job.max_tokens > 0,
            ref_vectors=job.ref_vectors,
        )
        
        return {
            "status": "completed",
            "request_id": job.request_id,
            "server_gpu": _server_gpu(),
            "server_engine": _server_engine(),
            **validation_result,
            # parity with the inline wait=true path (routes.py): debug requests
            # get the validator-side artifacts (sph_values_steps) back too.
            "artifacts": computed_artifacts if job.debug else [],
        }
    
    def _enqueue_callback(self, job: GenerateJob, result: Dict[str, Any]):
        """Enqueue callback for delivery via bounded callback queue."""
        if self._callback_queue is None:
            logger.warning(f"Callback queue not initialized, skipping callback for {job.request_id}")
            return

        if job.validation_artifacts is None:
            payload = {
                "request_id": job.request_id,
                "block_hash": job.block_hash,
                "block_height": job.block_height,
                "public_key": job.public_key,
                "node_id": job.node_id,
                "artifacts": result.get("artifacts", []),
                "encoding": result.get("encoding", {}),
            }
            self._callback_queue.enqueue(job.callback_url, "generated", payload)
        else:
            payload = {
                "request_id": job.request_id,
                "block_hash": job.block_hash,
                "block_height": job.block_height,
                "public_key": job.public_key,
                "node_id": job.node_id,
                "n_total": result.get("n_total", 0),
                "n_mismatch": result.get("n_mismatch", 0),
                "mismatch_nonces": result.get("mismatch_nonces", []),
                "p_value": result.get("p_value", 1.0),
                "fraud_detected": result.get("fraud_detected", False),
                # continuous vector-channel evidence (present when the reference
                # artifacts carried sph_values_steps); None otherwise.
                "vector_score": result.get("vector_score"),
            }
            self._callback_queue.enqueue(job.callback_url, "validated", payload)


_queue_instance: Optional[GenerateQueue] = None


def get_queue() -> GenerateQueue:
    """Get or create singleton queue instance."""
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = GenerateQueue()
    return _queue_instance


async def clear_queue():
    """Clear the queue singleton."""
    global _queue_instance
    if _queue_instance:
        await _queue_instance.clear_all()
        await _queue_instance.stop_worker()
        _queue_instance = None
