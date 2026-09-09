"""PoC generate queue with bounded nonce cap and result store."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from vllm.logger import init_logger
from .config import (
    GENERATION_ACTIVE_POLL_SEC,
    POC_GENERATE_CHUNK_TIMEOUT_SEC,
    POC_GENERATE_RESULT_TTL_SEC,
    POC_MAX_QUEUED_NONCES,
)
from .reservation import poc_reservation
from .callbacks import get_callback_queue, clear_callback_queue
from .data import DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD, fraud_test, wire_encoding

logger = init_logger(__name__)


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
    max_tokens: int = 0
    route_window: int = 256
    enforced_k_steps: Optional[Dict[int, List[int]]] = None
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
        """Clear queue and results; also signals the worker's stop_event
        (it calls self._stop_event.set())."""
        async with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
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
                            await asyncio.sleep(GENERATION_ACTIVE_POLL_SEC)
                    
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
        """Process a single generate job (decode-PoC chunks)."""
        from .decode_runner import POC_DECODE_MAX_BATCH
        chunk_size = min(job.batch_size or POC_DECODE_MAX_BATCH,
                         POC_DECODE_MAX_BATCH)
        total_nonces = len(job.nonces)
        n_chunks = (total_nonces + chunk_size - 1) // chunk_size
        logger.info(f"PoC queue job {job.request_id[:8]}: {total_nonces} nonces, batch_size={job.batch_size}, chunks={n_chunks}")
        
        start_time = time.time()
        computed_artifacts = []

        # One lease per job, reused across chunks; lease=None ⇒ inference
        # already aborted, legacy in-place layout — see poc_reservation.
        # The reservation lock also arbitrates with concurrent wait=True requests.
        mismatch_total = 0
        steps_total = 0
        validating = job.enforced_k_steps is not None
        n_eq = max(1, -(-total_nonces // chunk_size))
        base, rem = divmod(total_nonces, n_eq)
        bounds = []
        off = 0
        for ci in range(n_eq):
            size = base + (1 if ci < rem else 0)
            bounds.append((off, off + size))
            off += size
        async with poc_reservation(
            job.engine_client, chunk_size, job.seq_len + job.max_tokens,
        ) as lease:
            for chunk_idx, (lo, hi) in enumerate(bounds):
                chunk = job.nonces[lo:hi]

                while True:
                    if self._stop_event.is_set():
                        raise RuntimeError("Job cancelled")

                    if self._is_generation_active and self._is_generation_active(job.app_id):
                        await asyncio.sleep(GENERATION_ACTIVE_POLL_SEC)
                        continue

                    try:
                        # Lazy import: routes.py imports GenerateJob/queue
                        # from this module.
                        from .routes import _execute_poc_decode_rpc
                        result = await asyncio.wait_for(
                            _execute_poc_decode_rpc(
                                job.engine_client,
                                nonces=chunk,
                                block_hash=job.block_hash,
                                public_key=job.public_key,
                                seq_len=job.seq_len,
                                max_tokens=job.max_tokens,
                                route_window=job.route_window,
                                enforced_k_steps=(
                                    {n: job.enforced_k_steps[n] for n in chunk}
                                    if validating else None),
                                lease=lease,
                            ),
                            timeout=POC_GENERATE_CHUNK_TIMEOUT_SEC
                        )
                    except asyncio.CancelledError:
                        logger.info(f"PoC queue job {job.request_id[:8]}: cancelled during RPC")
                        raise RuntimeError("Job cancelled")
                    except asyncio.TimeoutError:
                        raise RuntimeError(f"Timeout waiting for engine RPC: chunk {chunk_idx}")

                    computed_artifacts.extend(result.get("artifacts", []))
                    steps_total += int(result.get("steps_total") or 0)
                    if validating:
                        mismatch_total += int(result.get("mismatch_total") or 0)
                    logger.debug(f"PoC queue job {job.request_id[:8]}: chunk {chunk_idx+1}/{n_chunks} done ({len(chunk)} nonces)")
                    break
        
        elapsed = time.time() - start_time
        rate = total_nonces / elapsed if elapsed > 0 else 0
        logger.info(f"PoC queue job {job.request_id[:8]} completed: {total_nonces} nonces in {elapsed:.2f}s ({rate:.0f}/s)")
        
        encoding = wire_encoding(job.k_dim)
        encoding["route_window"] = job.route_window
        encoding["seq_len"] = job.seq_len
        encoding["max_tokens"] = job.max_tokens
        if not validating:
            return {
                "status": "completed",
                "request_id": job.request_id,
                "artifacts": computed_artifacts,
                "encoding": encoding,
            }

        p_value, fraud = fraud_test(
            n_mismatch=mismatch_total, n_total=steps_total,
            p_mismatch=job.stat_test_p_mismatch,
            fraud_threshold=job.stat_test_fraud_threshold)
        return {
            "status": "completed",
            "request_id": job.request_id,
            "artifacts": computed_artifacts,
            "encoding": encoding,
            "n_mismatch": mismatch_total, "n_total": steps_total,
            "mismatch_rate": (mismatch_total / steps_total) if steps_total else 0.0,
            "p_value": p_value, "fraud_detected": fraud,
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
