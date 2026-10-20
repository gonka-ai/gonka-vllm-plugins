"""PoC callback sender with retry-until-stop and bounded buffer."""
import asyncio
import time
from collections import deque
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import aiohttp

from vllm.logger import init_logger
from .config import (
    POC_CALLBACK_INTERVAL_SEC,
    POC_CALLBACK_MAX_ARTIFACTS,
    POC_CALLBACK_MAX_CONCURRENT,
    POC_CALLBACK_MAX_RETRIES,
    POC_CALLBACK_QUEUE_SIZE,
)
from .data import Artifact, wire_encoding

logger = init_logger(__name__)

POC_CALLBACK_RETRY_BACKOFF_SEC = 1.0
POC_CALLBACK_RETRY_MAX_BACKOFF_SEC = 30.0


def _callback_log_target(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in ("http", "https") or not host:
            return "<callback>"
        if ":" in host:
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return "<callback>"


class CallbackSender:
    """Manages callback sending with retry and bounded buffer.

    Mining path: owned by /init/generate; streams cadence-batched artifacts
    to {url}/generated.
    """
    
    def __init__(
        self,
        callback_url: str,
        stop_event: asyncio.Event,
        k_dim: int = 12,
        max_artifacts: int = POC_CALLBACK_MAX_ARTIFACTS,
    ):
        self.callback_url = callback_url
        self._log_target = _callback_log_target(callback_url)
        self.stop_event = stop_event
        self.k_dim = k_dim
        self.max_artifacts = max_artifacts
        
        self._buffer: deque[Artifact] = deque()
        self._metadata: Dict[str, Any] = {}
        self._pending_payload: Optional[Dict] = None
    
    def add_artifacts(self, artifacts: List[Artifact], metadata: Dict[str, Any]):
        """Add artifacts to buffer, dropping oldest if cap exceeded."""
        self._metadata = metadata
        for artifact in artifacts:
            self._buffer.append(artifact)
        
        while len(self._buffer) > self.max_artifacts:
            self._buffer.popleft()
    
    def clear(self):
        """Clear all buffered artifacts."""
        self._buffer.clear()
        self._pending_payload = None

    async def run(self):
        """Main sender loop - batches and sends with retry-until-stop."""
        last_send_time = time.time()
        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
        retry_attempt = 0
        
        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)
                
                current_time = time.time()
                should_send = (
                    (self._buffer or self._pending_payload) and
                    (current_time - last_send_time >= POC_CALLBACK_INTERVAL_SEC)
                )
                
                if not should_send:
                    continue
                
                if self._pending_payload is None and self._buffer:
                    artifacts_to_send = list(self._buffer)
                    self._buffer.clear()
                    self._pending_payload = {
                        **self._metadata,
                        "artifacts": [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in artifacts_to_send],
                        "encoding": wire_encoding(self.k_dim),
                    }
                    retry_attempt = 0
                
                if self._pending_payload:
                    retry_attempt += 1
                    success = await self._send_callback(session, self._pending_payload)
                    if success:
                        if retry_attempt > 1:
                            logger.info(f"Callback to {self._log_target} succeeded after {retry_attempt} attempts")
                        self._pending_payload = None
                        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
                        retry_attempt = 0
                        last_send_time = current_time
                    elif retry_attempt >= POC_CALLBACK_MAX_RETRIES:
                        n_artifacts = len(self._pending_payload.get('artifacts', []))
                        logger.error(f"Callback to {self._log_target} failed after {retry_attempt} attempts, dropping {n_artifacts} artifacts")
                        self._pending_payload = None
                        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
                        retry_attempt = 0
                        last_send_time = current_time
                    else:
                        if retry_attempt == 1 or retry_attempt % 10 == 0:
                            logger.warning(f"Callback to {self._log_target} failed (attempt {retry_attempt}/{POC_CALLBACK_MAX_RETRIES}, next backoff {backoff:.1f}s)")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, POC_CALLBACK_RETRY_MAX_BACKOFF_SEC)
    
    async def _send_callback(self, session: aiohttp.ClientSession, payload: Dict) -> bool:
        """Send callback, return True on success."""
        try:
            async with session.post(
                f"{self.callback_url}/generated",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status < 400:
                    logger.debug(f"Callback sent: {len(payload.get('artifacts', []))} artifacts")
                    return True
                return False
        except Exception:
            return False


class CallbackQueue:
    """Queue for reliable callback delivery with bounded concurrency.

    Queued-/generate path: owned by GenerateQueue's worker; delivers one
    payload per job to {url}/generated (artifacts) or {url}/validated (verdict).
    """

    def __init__(
        self,
        stop_event: asyncio.Event,
        max_concurrent: int = POC_CALLBACK_MAX_CONCURRENT,
        max_queue_size: int = POC_CALLBACK_QUEUE_SIZE,
    ):
        self.stop_event = stop_event
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size

        self._queue: deque[tuple] = deque(maxlen=max_queue_size)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: set[asyncio.Task] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._dropped_count = 0

    def enqueue(self, url: str, path: str, payload: Dict[str, Any]):
        """Add callback to queue. Drops oldest if queue is full."""
        was_full = len(self._queue) >= self.max_queue_size
        self._queue.append((url, path, payload))
        if was_full:
            self._dropped_count += 1
            if self._dropped_count == 1 or self._dropped_count % 100 == 0:
                logger.warning(f"Callback queue full, dropped {self._dropped_count} callbacks total")

    async def start(self):
        """Start the callback worker."""
        if self._worker_task is None or self._worker_task.done():
            self._session = aiohttp.ClientSession()
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(f"Callback queue started (max_concurrent={self.max_concurrent}, max_queue={self.max_queue_size})")

    async def stop(self):
        """Stop the callback worker and cleanup."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

        if self._session:
            await self._session.close()
            self._session = None

        remaining = len(self._queue)
        if remaining > 0:
            logger.warning(f"Callback queue stopped with {remaining} pending callbacks")
        self._queue.clear()

    async def _worker_loop(self):
        """Main worker loop - dispatches callbacks with bounded concurrency."""
        logger.info("Callback worker loop starting")
        try:
            while not self.stop_event.is_set():
                self._active_tasks = {t for t in self._active_tasks if not t.done()}

                if not self._queue:
                    await asyncio.sleep(0.05)
                    continue

                url, path, payload = self._queue.popleft()
                task = asyncio.create_task(
                    self._send_with_retry(url, path, payload)
                )
                self._active_tasks.add(task)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Callback worker loop crashed: {e}", exc_info=True)
        logger.info("Callback worker loop exited")

    async def _send_with_retry(self, url: str, path: str, payload: Dict) -> bool:
        """Send callback with exponential backoff retry."""
        async with self._semaphore:
            backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
            attempt = 0
            url_path = f"{url}/{path}"
            log_target = _callback_log_target(url)

            while attempt < POC_CALLBACK_MAX_RETRIES:
                attempt += 1
                if self.stop_event.is_set():
                    return False

                try:
                    async with self._session.post(
                        url_path,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status < 400:
                            if attempt > 1:
                                logger.info(f"Callback to {log_target} succeeded after {attempt} attempts")
                            return True
                        logger.warning(f"Callback to {log_target} HTTP {resp.status} (attempt {attempt}/{POC_CALLBACK_MAX_RETRIES})")
                except Exception as e:
                    logger.warning(f"Callback to {log_target} failed: {type(e).__name__} (attempt {attempt}/{POC_CALLBACK_MAX_RETRIES})")

                if attempt < POC_CALLBACK_MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, POC_CALLBACK_RETRY_MAX_BACKOFF_SEC)

            logger.error(f"Callback to {log_target} failed after {POC_CALLBACK_MAX_RETRIES} attempts, giving up")
            return False


_callback_queue: Optional[CallbackQueue] = None


def get_callback_queue(stop_event: asyncio.Event) -> CallbackQueue:
    """Get or create singleton callback queue."""
    global _callback_queue
    if _callback_queue is None:
        _callback_queue = CallbackQueue(stop_event)
    return _callback_queue


async def clear_callback_queue():
    """Stop and clear the callback queue singleton."""
    global _callback_queue
    if _callback_queue:
        await _callback_queue.stop()
        _callback_queue = None
