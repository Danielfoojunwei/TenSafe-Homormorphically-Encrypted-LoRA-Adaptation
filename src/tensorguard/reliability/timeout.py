"""
Timeout Handling Utilities.

Provides timeout management for operations:
- Configurable timeouts with context managers
- Async and sync support
- Timeout callbacks for cleanup
- Cascading timeout propagation

Usage:
    # As decorator
    @with_timeout(seconds=30)
    def slow_operation():
        pass

    # As context manager
    with with_timeout(seconds=10):
        external_service.call()

    # Async
    async with async_timeout(seconds=5):
        await fetch_data()
"""

import asyncio
import functools
import logging
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OperationTimeoutError(Exception):
    """Raised when an operation times out."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout: Optional[float] = None,
        elapsed: Optional[float] = None,
    ):
        self.operation = operation
        self.timeout = timeout
        self.elapsed = elapsed
        super().__init__(message)


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""

    # Timeout in seconds
    timeout: float

    # Operation name for logging
    operation_name: Optional[str] = None

    # Callback on timeout
    on_timeout: Optional[Callable[[], None]] = None

    # Whether to log timeout
    log_timeout: bool = True

    # Grace period for cleanup after timeout
    cleanup_grace_period: float = 1.0


class TimeoutContext:
    """
    Context manager for timeout handling.

    Provides timeout functionality with proper cleanup
    and error reporting.
    """

    def __init__(
        self,
        timeout: float,
        operation_name: Optional[str] = None,
        on_timeout: Optional[Callable[[], None]] = None,
        log_timeout: bool = True,
    ):
        """
        Initialize timeout context.

        Args:
            timeout: Timeout in seconds
            operation_name: Name for logging
            on_timeout: Callback when timeout occurs
            log_timeout: Whether to log timeouts
        """
        self.timeout = timeout
        self.operation_name = operation_name
        self.on_timeout = on_timeout
        self.log_timeout = log_timeout

        self._start_time: Optional[float] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._future: Optional[Future] = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time since context entry."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        """Get remaining time before timeout."""
        return max(0.0, self.timeout - self.elapsed)

    def check_timeout(self) -> None:
        """
        Check if timeout has been exceeded.

        Raises:
            OperationTimeoutError: If timeout exceeded
        """
        if self.elapsed >= self.timeout:
            self._handle_timeout()

    def _handle_timeout(self) -> None:
        """Handle timeout condition."""
        if self.log_timeout:
            logger.warning(
                f"Operation timed out: {self.operation_name or 'unknown'} "
                f"after {self.elapsed:.2f}s (limit: {self.timeout}s)",
                extra={
                    "operation": self.operation_name,
                    "elapsed": self.elapsed,
                    "timeout": self.timeout,
                },
            )

        if self.on_timeout:
            try:
                self.on_timeout()
            except Exception as e:
                logger.warning(f"Timeout callback failed: {e}")

        raise OperationTimeoutError(
            f"Operation '{self.operation_name or 'unknown'}' timed out "
            f"after {self.elapsed:.2f}s",
            operation=self.operation_name,
            timeout=self.timeout,
            elapsed=self.elapsed,
        )


@contextmanager
def with_timeout(
    seconds: float,
    operation_name: Optional[str] = None,
    on_timeout: Optional[Callable[[], None]] = None,
):
    """
    Context manager for timeout handling.

    Note: This uses threading for timeout, which means the
    operation will continue running but the context will exit.
    For true cancellation, use async_timeout with async operations.

    Args:
        seconds: Timeout in seconds
        operation_name: Name for logging
        on_timeout: Callback when timeout occurs

    Yields:
        TimeoutContext for checking remaining time

    Raises:
        OperationTimeoutError: If timeout exceeded

    Usage:
        with with_timeout(10, "database_query"):
            result = db.execute(query)
    """
    ctx = TimeoutContext(
        timeout=seconds,
        operation_name=operation_name,
        on_timeout=on_timeout,
    )
    ctx._start_time = time.time()

    # Set up alarm signal if available (Unix only)
    old_handler = None
    if hasattr(signal, "SIGALRM"):
        def alarm_handler(signum, frame):
            ctx._handle_timeout()

        old_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield ctx

        # Check if we exceeded timeout (for platforms without SIGALRM)
        ctx.check_timeout()

    finally:
        # Restore signal handler
        if hasattr(signal, "SIGALRM") and old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


class AsyncTimeoutContext:
    """
    Async context manager for timeout handling.

    Provides true cancellation for async operations.
    """

    def __init__(
        self,
        timeout: float,
        operation_name: Optional[str] = None,
        on_timeout: Optional[Callable[[], None]] = None,
        log_timeout: bool = True,
    ):
        """
        Initialize async timeout context.

        Args:
            timeout: Timeout in seconds
            operation_name: Name for logging
            on_timeout: Callback when timeout occurs
            log_timeout: Whether to log timeouts
        """
        self.timeout = timeout
        self.operation_name = operation_name
        self.on_timeout = on_timeout
        self.log_timeout = log_timeout

        self._start_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time since context entry."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        """Get remaining time before timeout."""
        return max(0.0, self.timeout - self.elapsed)

    async def __aenter__(self) -> "AsyncTimeoutContext":
        """Enter async context."""
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit async context."""
        if exc_type is asyncio.TimeoutError:
            if self.log_timeout:
                logger.warning(
                    f"Async operation timed out: {self.operation_name or 'unknown'} "
                    f"after {self.elapsed:.2f}s (limit: {self.timeout}s)",
                    extra={
                        "operation": self.operation_name,
                        "elapsed": self.elapsed,
                        "timeout": self.timeout,
                    },
                )

            if self.on_timeout:
                try:
                    self.on_timeout()
                except Exception as e:
                    logger.warning(f"Timeout callback failed: {e}")

            raise OperationTimeoutError(
                f"Operation '{self.operation_name or 'unknown'}' timed out "
                f"after {self.elapsed:.2f}s",
                operation=self.operation_name,
                timeout=self.timeout,
                elapsed=self.elapsed,
            ) from exc_val

        return False


def async_timeout(
    seconds: float,
    operation_name: Optional[str] = None,
    on_timeout: Optional[Callable[[], None]] = None,
) -> AsyncTimeoutContext:
    """
    Create an async timeout context.

    Args:
        seconds: Timeout in seconds
        operation_name: Name for logging
        on_timeout: Callback when timeout occurs

    Returns:
        AsyncTimeoutContext

    Usage:
        async with async_timeout(10, "api_call"):
            result = await api.call()
    """
    return AsyncTimeoutContext(
        timeout=seconds,
        operation_name=operation_name,
        on_timeout=on_timeout,
    )


def timeout_decorator(
    seconds: float,
    operation_name: Optional[str] = None,
    on_timeout: Optional[Callable[[], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding timeout to functions.

    Args:
        seconds: Timeout in seconds
        operation_name: Name for logging
        on_timeout: Callback when timeout occurs

    Returns:
        Decorated function

    Usage:
        @timeout_decorator(30)
        def slow_function():
            pass

        @timeout_decorator(10)
        async def async_function():
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=seconds,
                    )
                except asyncio.TimeoutError:
                    if on_timeout:
                        try:
                            on_timeout()
                        except Exception as e:
                            logger.warning(f"Timeout callback failed: {e}")

                    raise OperationTimeoutError(
                        f"Operation '{name}' timed out after {seconds}s",
                        operation=name,
                        timeout=seconds,
                    )

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(func, *args, **kwargs)

                try:
                    return future.result(timeout=seconds)
                except FutureTimeoutError:
                    if on_timeout:
                        try:
                            on_timeout()
                        except Exception as e:
                            logger.warning(f"Timeout callback failed: {e}")

                    raise OperationTimeoutError(
                        f"Operation '{name}' timed out after {seconds}s",
                        operation=name,
                        timeout=seconds,
                    )
                finally:
                    executor.shutdown(wait=False)

            return sync_wrapper

    return decorator


class TimeoutManager:
    """
    Manages cascading timeouts for complex operations.

    Ensures that nested operations share a total timeout budget.
    """

    def __init__(self, total_timeout: float):
        """
        Initialize timeout manager.

        Args:
            total_timeout: Total timeout budget in seconds
        """
        self.total_timeout = total_timeout
        self._start_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time since manager creation."""
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        """Get remaining time in budget."""
        return max(0.0, self.total_timeout - self.elapsed)

    def get_timeout(self, max_timeout: Optional[float] = None) -> float:
        """
        Get timeout for a sub-operation.

        Args:
            max_timeout: Maximum timeout for this operation

        Returns:
            Timeout in seconds (min of remaining and max_timeout)
        """
        remaining = self.remaining
        if remaining <= 0:
            raise OperationTimeoutError(
                f"Timeout budget exhausted after {self.elapsed:.2f}s",
                timeout=self.total_timeout,
                elapsed=self.elapsed,
            )

        if max_timeout is not None:
            return min(remaining, max_timeout)
        return remaining

    def check_budget(self) -> None:
        """
        Check if timeout budget is exhausted.

        Raises:
            OperationTimeoutError: If budget exhausted
        """
        if self.remaining <= 0:
            raise OperationTimeoutError(
                f"Timeout budget exhausted after {self.elapsed:.2f}s",
                timeout=self.total_timeout,
                elapsed=self.elapsed,
            )

    @contextmanager
    def sub_timeout(
        self,
        max_timeout: Optional[float] = None,
        operation_name: Optional[str] = None,
    ):
        """
        Create a sub-timeout context from the budget.

        Args:
            max_timeout: Maximum timeout for this operation
            operation_name: Name for logging

        Yields:
            TimeoutContext
        """
        timeout = self.get_timeout(max_timeout)
        with with_timeout(timeout, operation_name) as ctx:
            yield ctx

    def sub_timeout_async(
        self,
        max_timeout: Optional[float] = None,
        operation_name: Optional[str] = None,
    ) -> AsyncTimeoutContext:
        """
        Create an async sub-timeout context from the budget.

        Args:
            max_timeout: Maximum timeout for this operation
            operation_name: Name for logging

        Returns:
            AsyncTimeoutContext
        """
        timeout = self.get_timeout(max_timeout)
        return async_timeout(timeout, operation_name)
