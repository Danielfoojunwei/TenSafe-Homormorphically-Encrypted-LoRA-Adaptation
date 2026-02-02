"""
Bulkhead Isolation Pattern.

Provides failure isolation between components:
- Semaphore-based concurrency limiting
- Thread pool isolation
- Queue-based request buffering
- Metrics and monitoring

Usage:
    # Semaphore bulkhead
    bulkhead = SemaphoreBulkhead(max_concurrent=10)

    @bulkhead
    async def call_service():
        return await service.request()

    # Thread pool bulkhead
    pool_bulkhead = ThreadPoolBulkhead(max_workers=5)

    result = pool_bulkhead.execute(blocking_operation, arg1, arg2)
"""

import asyncio
import functools
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BulkheadFullError(Exception):
    """Raised when bulkhead capacity is exhausted."""

    def __init__(
        self,
        message: str,
        bulkhead_name: str,
        max_capacity: int,
        current: int,
        queued: int = 0,
    ):
        self.bulkhead_name = bulkhead_name
        self.max_capacity = max_capacity
        self.current = current
        self.queued = queued
        super().__init__(message)


@dataclass
class BulkheadMetrics:
    """Metrics for bulkhead monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    rejected_calls: int = 0
    timed_out_calls: int = 0
    current_concurrent: int = 0
    current_queued: int = 0
    max_concurrent_seen: int = 0

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "rejected_calls": self.rejected_calls,
            "timed_out_calls": self.timed_out_calls,
            "current_concurrent": self.current_concurrent,
            "current_queued": self.current_queued,
            "max_concurrent_seen": self.max_concurrent_seen,
            "success_rate": self.successful_calls / self.total_calls
            if self.total_calls > 0
            else 1.0,
        }


class Bulkhead:
    """Base class for bulkhead implementations."""

    def __init__(self, name: str):
        self.name = name
        self._metrics = BulkheadMetrics()
        self._lock = threading.RLock()

    @property
    def metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        with self._lock:
            return self._metrics

    def get_status(self) -> dict:
        """Get bulkhead status."""
        raise NotImplementedError


class SemaphoreBulkhead(Bulkhead):
    """
    Semaphore-based bulkhead for limiting concurrency.

    Uses a semaphore to limit the number of concurrent
    operations, preventing resource exhaustion.
    """

    def __init__(
        self,
        name: str = "default",
        max_concurrent: int = 10,
        max_queue: int = 0,
        timeout: float = 0.0,
    ):
        """
        Initialize semaphore bulkhead.

        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent operations
            max_queue: Maximum queued operations (0 = no queue)
            timeout: Timeout for acquiring semaphore (0 = no wait)
        """
        super().__init__(name)
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.timeout = timeout

        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._queue_semaphore = (
            threading.Semaphore(max_queue) if max_queue > 0 else None
        )

    async def _ensure_async_semaphore(self) -> asyncio.Semaphore:
        """Ensure async semaphore is created."""
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._async_semaphore

    def _try_acquire(self) -> bool:
        """Try to acquire semaphore."""
        if self.timeout > 0:
            return self._semaphore.acquire(timeout=self.timeout)
        else:
            return self._semaphore.acquire(blocking=False)

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function within bulkhead.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            BulkheadFullError: If bulkhead is full
        """
        with self._lock:
            self._metrics.total_calls += 1

        if not self._try_acquire():
            with self._lock:
                self._metrics.rejected_calls += 1
            raise BulkheadFullError(
                f"Bulkhead '{self.name}' is full",
                bulkhead_name=self.name,
                max_capacity=self.max_concurrent,
                current=self.max_concurrent,
            )

        with self._lock:
            self._metrics.current_concurrent += 1
            self._metrics.max_concurrent_seen = max(
                self._metrics.max_concurrent_seen,
                self._metrics.current_concurrent,
            )

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._metrics.successful_calls += 1
            return result
        finally:
            self._semaphore.release()
            with self._lock:
                self._metrics.current_concurrent -= 1

    async def execute_async(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute async function within bulkhead.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            BulkheadFullError: If bulkhead is full
        """
        semaphore = await self._ensure_async_semaphore()

        with self._lock:
            self._metrics.total_calls += 1

        if self.timeout > 0:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                with self._lock:
                    self._metrics.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' timed out",
                    bulkhead_name=self.name,
                    max_capacity=self.max_concurrent,
                    current=self.max_concurrent,
                )
        else:
            # Try without waiting
            if semaphore.locked():
                with self._lock:
                    self._metrics.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' is full",
                    bulkhead_name=self.name,
                    max_capacity=self.max_concurrent,
                    current=self.max_concurrent,
                )
            await semaphore.acquire()

        with self._lock:
            self._metrics.current_concurrent += 1
            self._metrics.max_concurrent_seen = max(
                self._metrics.max_concurrent_seen,
                self._metrics.current_concurrent,
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            with self._lock:
                self._metrics.successful_calls += 1
            return result
        finally:
            semaphore.release()
            with self._lock:
                self._metrics.current_concurrent -= 1

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for wrapping functions with bulkhead."""
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await self.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return self.execute(func, *args, **kwargs)

            return sync_wrapper

    def get_status(self) -> dict:
        """Get bulkhead status."""
        with self._lock:
            return {
                "name": self.name,
                "type": "semaphore",
                "max_concurrent": self.max_concurrent,
                "max_queue": self.max_queue,
                "timeout": self.timeout,
                "metrics": self._metrics.to_dict(),
            }


class ThreadPoolBulkhead(Bulkhead):
    """
    Thread pool-based bulkhead for isolating blocking operations.

    Uses a dedicated thread pool to isolate blocking operations,
    preventing them from affecting other components.
    """

    def __init__(
        self,
        name: str = "default",
        max_workers: int = 5,
        queue_size: int = 100,
        timeout: float = 30.0,
    ):
        """
        Initialize thread pool bulkhead.

        Args:
            name: Bulkhead name
            max_workers: Maximum worker threads
            queue_size: Maximum queue size
            timeout: Default timeout for operations
        """
        super().__init__(name)
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.default_timeout = timeout

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"bulkhead-{name}",
        )
        self._pending = 0

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute function in thread pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Operation timeout
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            BulkheadFullError: If pool queue is full
        """
        timeout = timeout or self.default_timeout

        with self._lock:
            self._metrics.total_calls += 1

            if self._pending >= self.queue_size:
                self._metrics.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' queue is full",
                    bulkhead_name=self.name,
                    max_capacity=self.max_workers,
                    current=self.max_workers,
                    queued=self._pending,
                )

            self._pending += 1
            self._metrics.current_queued = self._pending

        try:
            future = self._executor.submit(func, *args, **kwargs)

            try:
                result = future.result(timeout=timeout)
                with self._lock:
                    self._metrics.successful_calls += 1
                return result
            except FutureTimeoutError:
                with self._lock:
                    self._metrics.timed_out_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' operation timed out",
                    bulkhead_name=self.name,
                    max_capacity=self.max_workers,
                    current=self.max_workers,
                )
        finally:
            with self._lock:
                self._pending -= 1
                self._metrics.current_queued = self._pending

    async def execute_async(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute function in thread pool asynchronously.

        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Operation timeout
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        timeout = timeout or self.default_timeout

        with self._lock:
            self._metrics.total_calls += 1

            if self._pending >= self.queue_size:
                self._metrics.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' queue is full",
                    bulkhead_name=self.name,
                    max_capacity=self.max_workers,
                    current=self.max_workers,
                    queued=self._pending,
                )

            self._pending += 1
            self._metrics.current_queued = self._pending

        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self._executor,
                functools.partial(func, *args, **kwargs),
            )

            try:
                result = await asyncio.wait_for(future, timeout=timeout)
                with self._lock:
                    self._metrics.successful_calls += 1
                return result
            except asyncio.TimeoutError:
                with self._lock:
                    self._metrics.timed_out_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' operation timed out",
                    bulkhead_name=self.name,
                    max_capacity=self.max_workers,
                    current=self.max_workers,
                )
        finally:
            with self._lock:
                self._pending -= 1
                self._metrics.current_queued = self._pending

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)
        logger.info(f"Bulkhead '{self.name}' thread pool shut down")

    def get_status(self) -> dict:
        """Get bulkhead status."""
        with self._lock:
            return {
                "name": self.name,
                "type": "thread_pool",
                "max_workers": self.max_workers,
                "queue_size": self.queue_size,
                "default_timeout": self.default_timeout,
                "pending": self._pending,
                "metrics": self._metrics.to_dict(),
            }
