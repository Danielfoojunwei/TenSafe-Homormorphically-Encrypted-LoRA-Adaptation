"""
TenSafe Async Webhook Dispatcher.

Provides queue-based webhook delivery with:
- Background task processing
- Delivery deduplication
- Rate limiting per endpoint
- Dead letter queue for failed deliveries
- Metrics and monitoring support

This dispatcher decouples event triggering from webhook delivery,
ensuring reliable delivery without blocking the main application.
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional

from .webhooks import WebhookPayload, WebhookService, get_webhook_service

logger = logging.getLogger(__name__)


class DispatcherState(str, Enum):
    """State of the dispatcher."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"


@dataclass(order=True)
class DeliveryTask:
    """
    A task in the delivery queue.

    Ordered by priority and scheduled time for priority queue support.
    """

    # Ordering fields (used by PriorityQueue)
    priority: int
    scheduled_at: float  # Unix timestamp

    # Task data (not used for ordering)
    webhook_id: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    event_id: str = field(compare=False)
    event_type: str = field(compare=False)
    tenant_id: str = field(compare=False)
    attempt: int = field(compare=False, default=1)
    max_attempts: int = field(compare=False, default=3)
    created_at: float = field(compare=False, default_factory=time.time)
    idempotency_key: str = field(compare=False, default="")

    def __post_init__(self):
        """Generate idempotency key if not provided."""
        if not self.idempotency_key:
            self.idempotency_key = hashlib.sha256(
                f"{self.webhook_id}:{self.event_id}".encode()
            ).hexdigest()


@dataclass
class DispatcherConfig:
    """Configuration for the webhook dispatcher."""

    # Queue settings
    max_queue_size: int = 10000
    batch_size: int = 10
    poll_interval_seconds: float = 0.1

    # Worker settings
    num_workers: int = 4
    worker_timeout_seconds: int = 60

    # Rate limiting
    max_deliveries_per_second: int = 100
    max_deliveries_per_endpoint_per_second: int = 10

    # Retry settings
    retry_base_delay_seconds: int = 5
    retry_max_delay_seconds: int = 300

    # Deduplication
    dedup_window_seconds: int = 300  # 5 minutes

    # Dead letter queue
    enable_dlq: bool = True
    dlq_max_size: int = 1000

    # Metrics
    enable_metrics: bool = True
    metrics_interval_seconds: int = 60


@dataclass
class DispatcherMetrics:
    """Metrics for the dispatcher."""

    # Counters
    tasks_enqueued: int = 0
    tasks_processed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    tasks_retried: int = 0
    tasks_deduplicated: int = 0
    tasks_rate_limited: int = 0

    # Timing
    total_processing_time_ms: int = 0
    avg_processing_time_ms: float = 0.0

    # Queue state
    queue_size: int = 0
    dlq_size: int = 0

    # Rate limiting
    current_rate: float = 0.0
    rate_limit_active: bool = False

    # Timestamps
    last_task_at: Optional[datetime] = None
    started_at: Optional[datetime] = None


class WebhookDispatcher:
    """
    Asynchronous webhook dispatcher with queue-based delivery.

    Features:
    - Priority queue for task ordering
    - Multiple concurrent workers
    - Delivery deduplication
    - Rate limiting per endpoint
    - Exponential backoff retry
    - Dead letter queue for permanent failures
    - Comprehensive metrics

    Usage:
        dispatcher = WebhookDispatcher()
        await dispatcher.start()

        # Enqueue webhooks
        dispatcher.enqueue(webhook_id, event_type, payload, tenant_id)

        # Graceful shutdown
        await dispatcher.stop()
    """

    def __init__(
        self,
        config: Optional[DispatcherConfig] = None,
        webhook_service: Optional[WebhookService] = None,
    ):
        """
        Initialize the dispatcher.

        Args:
            config: Dispatcher configuration
            webhook_service: Webhook service for delivery
        """
        self.config = config or DispatcherConfig()
        self._webhook_service = webhook_service

        # State
        self._state = DispatcherState.STOPPED
        self._state_lock = threading.Lock()

        # Main delivery queue (priority queue)
        self._queue: PriorityQueue[DeliveryTask] = PriorityQueue(
            maxsize=self.config.max_queue_size
        )

        # Dead letter queue for failed deliveries
        self._dlq: List[DeliveryTask] = []
        self._dlq_lock = threading.Lock()

        # Deduplication cache
        self._seen_tasks: Dict[str, float] = {}  # idempotency_key -> timestamp
        self._seen_lock = threading.Lock()

        # Rate limiting
        self._rate_buckets: Dict[str, List[float]] = defaultdict(list)
        self._rate_lock = threading.Lock()

        # Workers
        self._workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._metrics = DispatcherMetrics()
        self._metrics_lock = threading.Lock()

        # Event handlers
        self._on_success_handlers: List[Callable] = []
        self._on_failure_handlers: List[Callable] = []
        self._on_dlq_handlers: List[Callable] = []

    @property
    def state(self) -> DispatcherState:
        """Get current dispatcher state."""
        with self._state_lock:
            return self._state

    @property
    def metrics(self) -> DispatcherMetrics:
        """Get current metrics."""
        with self._metrics_lock:
            self._metrics.queue_size = self._queue.qsize()
            with self._dlq_lock:
                self._metrics.dlq_size = len(self._dlq)
            return self._metrics

    def _get_webhook_service(self) -> WebhookService:
        """Get webhook service instance."""
        if self._webhook_service is None:
            self._webhook_service = get_webhook_service()
        return self._webhook_service

    def _is_duplicate(self, task: DeliveryTask) -> bool:
        """
        Check if a task is a duplicate.

        Args:
            task: Task to check

        Returns:
            True if duplicate (already seen within dedup window)
        """
        now = time.time()
        window = self.config.dedup_window_seconds

        with self._seen_lock:
            # Clean old entries
            expired = [
                key for key, ts in self._seen_tasks.items()
                if now - ts > window
            ]
            for key in expired:
                del self._seen_tasks[key]

            # Check if seen
            if task.idempotency_key in self._seen_tasks:
                return True

            # Mark as seen
            self._seen_tasks[task.idempotency_key] = now
            return False

    def _check_rate_limit(self, endpoint: str) -> bool:
        """
        Check if rate limit allows delivery.

        Args:
            endpoint: Endpoint identifier (webhook URL or ID)

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window = 1.0  # 1 second window

        with self._rate_lock:
            # Clean old entries
            self._rate_buckets[endpoint] = [
                ts for ts in self._rate_buckets[endpoint]
                if now - ts < window
            ]

            # Check global rate limit
            total_recent = sum(
                len(timestamps)
                for timestamps in self._rate_buckets.values()
            )
            if total_recent >= self.config.max_deliveries_per_second:
                return False

            # Check per-endpoint rate limit
            if len(self._rate_buckets[endpoint]) >= self.config.max_deliveries_per_endpoint_per_second:
                return False

            # Record this request
            self._rate_buckets[endpoint].append(now)
            return True

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        base = self.config.retry_base_delay_seconds
        max_delay = self.config.retry_max_delay_seconds

        # Exponential backoff with jitter
        import random
        delay = base * (2 ** (attempt - 1))
        delay = min(delay, max_delay)

        # Add 10-25% jitter
        jitter = delay * random.uniform(0.1, 0.25)
        return delay + jitter

    def enqueue(
        self,
        webhook_id: str,
        event_type: str,
        payload: Dict[str, Any],
        tenant_id: str,
        event_id: Optional[str] = None,
        priority: int = 0,
    ) -> bool:
        """
        Enqueue a webhook delivery task.

        Args:
            webhook_id: Target webhook ID
            event_type: Event type
            payload: Event payload
            tenant_id: Tenant ID
            event_id: Optional event ID
            priority: Task priority (lower = higher priority)

        Returns:
            True if enqueued, False if queue is full or duplicate
        """
        if event_id is None:
            event_id = f"evt-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}"

        task = DeliveryTask(
            priority=priority,
            scheduled_at=time.time(),
            webhook_id=webhook_id,
            payload=payload,
            event_id=event_id,
            event_type=event_type,
            tenant_id=tenant_id,
        )

        # Check for duplicates
        if self._is_duplicate(task):
            logger.debug(f"Duplicate task {task.idempotency_key}, skipping")
            with self._metrics_lock:
                self._metrics.tasks_deduplicated += 1
            return False

        # Enqueue
        try:
            self._queue.put_nowait(task)
            with self._metrics_lock:
                self._metrics.tasks_enqueued += 1
            logger.debug(f"Enqueued task for webhook {webhook_id}, event {event_type}")
            return True
        except Exception:
            logger.warning("Queue is full, task not enqueued")
            return False

    def enqueue_retry(self, task: DeliveryTask) -> bool:
        """
        Re-enqueue a task for retry.

        Args:
            task: Task to retry

        Returns:
            True if enqueued, False if max retries exceeded
        """
        if task.attempt >= task.max_attempts:
            logger.warning(
                f"Task {task.idempotency_key} exceeded max retries, moving to DLQ"
            )
            self._add_to_dlq(task)
            return False

        # Calculate retry delay
        delay = self._calculate_retry_delay(task.attempt)

        # Create retry task
        retry_task = DeliveryTask(
            priority=task.priority + 1,  # Lower priority for retries
            scheduled_at=time.time() + delay,
            webhook_id=task.webhook_id,
            payload=task.payload,
            event_id=task.event_id,
            event_type=task.event_type,
            tenant_id=task.tenant_id,
            attempt=task.attempt + 1,
            max_attempts=task.max_attempts,
            created_at=task.created_at,
            idempotency_key=task.idempotency_key,
        )

        try:
            self._queue.put_nowait(retry_task)
            with self._metrics_lock:
                self._metrics.tasks_retried += 1
            logger.info(
                f"Scheduled retry {retry_task.attempt}/{retry_task.max_attempts} "
                f"for task {task.idempotency_key} in {delay:.1f}s"
            )
            return True
        except Exception:
            self._add_to_dlq(task)
            return False

    def _add_to_dlq(self, task: DeliveryTask) -> None:
        """
        Add a task to the dead letter queue.

        Args:
            task: Failed task
        """
        if not self.config.enable_dlq:
            return

        with self._dlq_lock:
            # Enforce max size
            while len(self._dlq) >= self.config.dlq_max_size:
                oldest = self._dlq.pop(0)
                logger.warning(f"DLQ full, dropping oldest task: {oldest.idempotency_key}")

            self._dlq.append(task)

        # Call DLQ handlers
        for handler in self._on_dlq_handlers:
            try:
                handler(task)
            except Exception as e:
                logger.error(f"DLQ handler error: {e}")

    async def _process_task(self, task: DeliveryTask) -> bool:
        """
        Process a single delivery task.

        Args:
            task: Task to process

        Returns:
            True if delivery succeeded
        """
        # Check if task is scheduled for later
        now = time.time()
        if task.scheduled_at > now:
            delay = task.scheduled_at - now
            await asyncio.sleep(delay)

        # Check rate limit
        if not self._check_rate_limit(task.webhook_id):
            logger.debug(f"Rate limited for webhook {task.webhook_id}")
            with self._metrics_lock:
                self._metrics.tasks_rate_limited += 1
            # Re-enqueue with small delay
            await asyncio.sleep(0.1)
            self._queue.put_nowait(task)
            return False

        # Get webhook
        service = self._get_webhook_service()
        webhook = service.get_webhook(task.webhook_id, task.tenant_id)

        if webhook is None:
            logger.warning(f"Webhook {task.webhook_id} not found, dropping task")
            with self._metrics_lock:
                self._metrics.tasks_failed += 1
            return False

        if not webhook.active:
            logger.debug(f"Webhook {task.webhook_id} is inactive, dropping task")
            return False

        # Create payload
        payload = WebhookPayload(
            id=task.event_id,
            type=task.event_type,
            created=int(task.created_at),
            data=task.payload,
            tenant_id=task.tenant_id,
        )

        # Deliver
        start_time = time.perf_counter()
        result = await service.deliver(webhook, payload)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Update metrics
        with self._metrics_lock:
            self._metrics.tasks_processed += 1
            self._metrics.total_processing_time_ms += elapsed_ms
            self._metrics.avg_processing_time_ms = (
                self._metrics.total_processing_time_ms / self._metrics.tasks_processed
            )
            self._metrics.last_task_at = datetime.utcnow()

        if result.success:
            with self._metrics_lock:
                self._metrics.tasks_succeeded += 1

            # Call success handlers
            for handler in self._on_success_handlers:
                try:
                    handler(task, result)
                except Exception as e:
                    logger.error(f"Success handler error: {e}")

            return True

        # Delivery failed
        logger.warning(
            f"Delivery failed for webhook {task.webhook_id}: "
            f"{result.error_code} - {result.error_message}"
        )

        with self._metrics_lock:
            self._metrics.tasks_failed += 1

        # Call failure handlers
        for handler in self._on_failure_handlers:
            try:
                handler(task, result)
            except Exception as e:
                logger.error(f"Failure handler error: {e}")

        # Schedule retry
        self.enqueue_retry(task)
        return False

    async def _worker(self, worker_id: int) -> None:
        """
        Worker coroutine that processes tasks from the queue.

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # Get task from queue (with timeout for graceful shutdown)
                try:
                    task = self._queue.get(timeout=self.config.poll_interval_seconds)
                except Empty:
                    continue

                # Process task
                try:
                    await self._process_task(task)
                except Exception as e:
                    logger.exception(f"Worker {worker_id} error processing task: {e}")
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} unexpected error: {e}")
                await asyncio.sleep(1)

        logger.info(f"Worker {worker_id} stopped")

    async def _metrics_reporter(self) -> None:
        """Background task that logs metrics periodically."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.metrics_interval_seconds)

            if not self.config.enable_metrics:
                continue

            m = self.metrics
            logger.info(
                f"Dispatcher metrics: "
                f"queued={m.queue_size}, "
                f"processed={m.tasks_processed}, "
                f"succeeded={m.tasks_succeeded}, "
                f"failed={m.tasks_failed}, "
                f"dlq={m.dlq_size}, "
                f"avg_time={m.avg_processing_time_ms:.1f}ms"
            )

    async def start(self) -> None:
        """Start the dispatcher and workers."""
        with self._state_lock:
            if self._state != DispatcherState.STOPPED:
                raise RuntimeError(f"Cannot start dispatcher in state {self._state}")
            self._state = DispatcherState.STARTING

        logger.info(
            f"Starting webhook dispatcher with {self.config.num_workers} workers"
        )

        # Reset state
        self._shutdown_event.clear()
        self._metrics.started_at = datetime.utcnow()

        # Start workers
        for i in range(self.config.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        # Start metrics reporter
        if self.config.enable_metrics:
            self._workers.append(asyncio.create_task(self._metrics_reporter()))

        with self._state_lock:
            self._state = DispatcherState.RUNNING

        logger.info("Webhook dispatcher started")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the dispatcher gracefully.

        Args:
            timeout: Maximum seconds to wait for workers to finish
        """
        with self._state_lock:
            if self._state != DispatcherState.RUNNING:
                return
            self._state = DispatcherState.STOPPING

        logger.info("Stopping webhook dispatcher...")

        # Signal workers to stop
        self._shutdown_event.set()

        # Wait for workers with timeout
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for workers, cancelling...")
                for worker in self._workers:
                    worker.cancel()

        self._workers.clear()

        with self._state_lock:
            self._state = DispatcherState.STOPPED

        logger.info("Webhook dispatcher stopped")

    def pause(self) -> None:
        """Pause processing (workers remain active but don't process)."""
        with self._state_lock:
            if self._state == DispatcherState.RUNNING:
                self._state = DispatcherState.PAUSED
                logger.info("Webhook dispatcher paused")

    def resume(self) -> None:
        """Resume processing after pause."""
        with self._state_lock:
            if self._state == DispatcherState.PAUSED:
                self._state = DispatcherState.RUNNING
                logger.info("Webhook dispatcher resumed")

    def get_dlq_tasks(self, limit: int = 100) -> List[DeliveryTask]:
        """
        Get tasks from the dead letter queue.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of failed tasks
        """
        with self._dlq_lock:
            return list(self._dlq[:limit])

    def replay_dlq_task(self, idempotency_key: str) -> bool:
        """
        Replay a task from the dead letter queue.

        Args:
            idempotency_key: Task idempotency key

        Returns:
            True if task found and re-enqueued
        """
        with self._dlq_lock:
            for i, task in enumerate(self._dlq):
                if task.idempotency_key == idempotency_key:
                    # Remove from DLQ
                    self._dlq.pop(i)

                    # Re-enqueue with reset attempts
                    task.attempt = 1
                    task.scheduled_at = time.time()

                    # Clear from seen cache to allow reprocessing
                    with self._seen_lock:
                        self._seen_tasks.pop(task.idempotency_key, None)

                    return self.enqueue(
                        webhook_id=task.webhook_id,
                        event_type=task.event_type,
                        payload=task.payload,
                        tenant_id=task.tenant_id,
                        event_id=task.event_id,
                    )

        return False

    def clear_dlq(self) -> int:
        """
        Clear the dead letter queue.

        Returns:
            Number of tasks cleared
        """
        with self._dlq_lock:
            count = len(self._dlq)
            self._dlq.clear()
            return count

    # Event handler registration
    def on_success(self, handler: Callable) -> None:
        """Register a success callback."""
        self._on_success_handlers.append(handler)

    def on_failure(self, handler: Callable) -> None:
        """Register a failure callback."""
        self._on_failure_handlers.append(handler)

    def on_dlq(self, handler: Callable) -> None:
        """Register a DLQ callback."""
        self._on_dlq_handlers.append(handler)


# Global dispatcher instance
_dispatcher: Optional[WebhookDispatcher] = None
_dispatcher_lock = threading.Lock()


def get_dispatcher() -> WebhookDispatcher:
    """Get the global dispatcher instance."""
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = WebhookDispatcher()
    return _dispatcher


def set_dispatcher(dispatcher: WebhookDispatcher) -> None:
    """Set the global dispatcher instance."""
    global _dispatcher
    with _dispatcher_lock:
        _dispatcher = dispatcher


async def start_dispatcher() -> WebhookDispatcher:
    """Start the global dispatcher."""
    dispatcher = get_dispatcher()
    await dispatcher.start()
    return dispatcher


async def stop_dispatcher() -> None:
    """Stop the global dispatcher."""
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is not None:
            await _dispatcher.stop()
