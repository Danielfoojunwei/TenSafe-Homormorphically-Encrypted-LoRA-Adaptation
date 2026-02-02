"""
Retry with Exponential Backoff Implementation.

Provides robust retry logic for transient failures:
- Exponential backoff with jitter
- Configurable retry conditions
- Async and sync support
- Retry budget management

Usage:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def call_external_service():
        return requests.get(url)

    # Or with custom configuration
    config = RetryConfig(
        max_retries=5,
        base_delay=0.5,
        max_delay=30.0,
        exponential_base=2,
        jitter=True,
    )

    @retry_with_backoff(config=config)
    async def call_async_service():
        return await client.request()
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Set, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
    ):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(message)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Maximum number of retry attempts
    max_retries: int = 3

    # Base delay between retries (seconds)
    base_delay: float = 1.0

    # Maximum delay between retries (seconds)
    max_delay: float = 60.0

    # Exponential base for backoff calculation
    exponential_base: float = 2.0

    # Add random jitter to prevent thundering herd
    jitter: bool = True

    # Jitter factor (0.0 to 1.0)
    jitter_factor: float = 0.25

    # Exceptions to retry on
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)

    # Exceptions to NOT retry on (takes precedence)
    no_retry_exceptions: Tuple[Type[Exception], ...] = ()

    # Custom retry condition function
    retry_condition: Optional[Callable[[Exception], bool]] = None

    # Log retry attempts
    log_retries: bool = True

    # Callback on each retry
    on_retry: Optional[Callable[[int, Exception, float], None]] = None

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """Aggressive retry configuration for critical operations."""
        return cls(
            max_retries=10,
            base_delay=0.1,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        )

    @classmethod
    def conservative(cls) -> "RetryConfig":
        """Conservative retry configuration to avoid overload."""
        return cls(
            max_retries=3,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=True,
        )

    @classmethod
    def for_network(cls) -> "RetryConfig":
        """Configuration optimized for network operations."""
        return cls(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            retry_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
            ),
        )

    @classmethod
    def for_database(cls) -> "RetryConfig":
        """Configuration optimized for database operations."""
        return cls(
            max_retries=3,
            base_delay=0.1,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=True,
        )


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_attempts: int = 0
    successful_first_try: int = 0
    successful_after_retry: int = 0
    exhausted_retries: int = 0
    total_retry_delay: float = 0.0

    def record_success(self, attempt: int) -> None:
        """Record a successful operation."""
        self.total_attempts += 1
        if attempt == 1:
            self.successful_first_try += 1
        else:
            self.successful_after_retry += 1

    def record_exhausted(self, delay_sum: float) -> None:
        """Record exhausted retries."""
        self.total_attempts += 1
        self.exhausted_retries += 1
        self.total_retry_delay += delay_sum

    def record_delay(self, delay: float) -> None:
        """Record retry delay."""
        self.total_retry_delay += delay

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "total_attempts": self.total_attempts,
            "successful_first_try": self.successful_first_try,
            "successful_after_retry": self.successful_after_retry,
            "exhausted_retries": self.exhausted_retries,
            "total_retry_delay_seconds": round(self.total_retry_delay, 2),
            "success_rate": (self.successful_first_try + self.successful_after_retry)
            / self.total_attempts
            if self.total_attempts > 0
            else 1.0,
        }


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay for a retry attempt.

    Uses exponential backoff: delay = base * (exponential_base ^ attempt)
    with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = config.base_delay * (config.exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(0.0, delay)

    return delay


def should_retry(
    exception: Exception,
    config: RetryConfig,
) -> bool:
    """
    Determine if an exception should trigger a retry.

    Args:
        exception: The exception that occurred
        config: Retry configuration

    Returns:
        True if should retry
    """
    # Check no-retry exceptions first (takes precedence)
    if config.no_retry_exceptions and isinstance(exception, config.no_retry_exceptions):
        return False

    # Check custom retry condition
    if config.retry_condition is not None:
        return config.retry_condition(exception)

    # Check retry exceptions
    return isinstance(exception, config.retry_exceptions)


def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        config: Full retry configuration
        **kwargs: Additional config options

    Returns:
        Decorated function
    """
    # Build config from parameters
    if config is None:
        config_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if max_retries is not None:
            config_kwargs["max_retries"] = max_retries
        if base_delay is not None:
            config_kwargs["base_delay"] = base_delay
        config = RetryConfig(**config_kwargs)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            return _async_retry_wrapper(func, config)
        else:
            return _sync_retry_wrapper(func, config)

    return decorator


def _sync_retry_wrapper(
    func: Callable[..., T],
    config: RetryConfig,
) -> Callable[..., T]:
    """Create synchronous retry wrapper."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None
        total_delay = 0.0

        for attempt in range(config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if config.log_retries and attempt > 0:
                    logger.info(
                        f"Retry successful for {func.__name__} on attempt {attempt + 1}",
                        extra={"function": func.__name__, "attempt": attempt + 1},
                    )
                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not should_retry(e, config):
                    raise

                # Check if we have retries left
                if attempt >= config.max_retries:
                    break

                # Calculate and apply delay
                delay = calculate_delay(attempt, config)
                total_delay += delay

                if config.log_retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__}: "
                        f"{type(e).__name__}: {e}. Waiting {delay:.2f}s",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": config.max_retries,
                            "delay": delay,
                            "exception": str(e),
                        },
                    )

                # Call retry callback
                if config.on_retry:
                    try:
                        config.on_retry(attempt + 1, e, delay)
                    except Exception as callback_error:
                        logger.warning(f"Retry callback failed: {callback_error}")

                time.sleep(delay)

        # All retries exhausted
        raise RetryExhaustedError(
            f"All {config.max_retries + 1} attempts failed for {func.__name__}",
            attempts=config.max_retries + 1,
            last_exception=last_exception,
        )

    return wrapper


def _async_retry_wrapper(
    func: Callable[..., T],
    config: RetryConfig,
) -> Callable[..., T]:
    """Create asynchronous retry wrapper."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None
        total_delay = 0.0

        for attempt in range(config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if config.log_retries and attempt > 0:
                    logger.info(
                        f"Retry successful for {func.__name__} on attempt {attempt + 1}",
                        extra={"function": func.__name__, "attempt": attempt + 1},
                    )
                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not should_retry(e, config):
                    raise

                # Check if we have retries left
                if attempt >= config.max_retries:
                    break

                # Calculate and apply delay
                delay = calculate_delay(attempt, config)
                total_delay += delay

                if config.log_retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__}: "
                        f"{type(e).__name__}: {e}. Waiting {delay:.2f}s",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": config.max_retries,
                            "delay": delay,
                            "exception": str(e),
                        },
                    )

                # Call retry callback
                if config.on_retry:
                    try:
                        config.on_retry(attempt + 1, e, delay)
                    except Exception as callback_error:
                        logger.warning(f"Retry callback failed: {callback_error}")

                await asyncio.sleep(delay)

        # All retries exhausted
        raise RetryExhaustedError(
            f"All {config.max_retries + 1} attempts failed for {func.__name__}",
            attempts=config.max_retries + 1,
            last_exception=last_exception,
        )

    return wrapper


async def async_retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments
        config: Retry configuration
        **kwargs: Keyword arguments

    Returns:
        Function result

    Usage:
        result = await async_retry_with_backoff(
            fetch_data,
            url,
            config=RetryConfig(max_retries=3),
        )
    """
    config = config or RetryConfig()
    wrapped = _async_retry_wrapper(func, config)
    return await wrapped(*args, **kwargs)


class RetryBudget:
    """
    Manages a retry budget to prevent retry storms.

    Limits the percentage of requests that can be retried
    within a time window to prevent cascading failures.
    """

    def __init__(
        self,
        max_retry_ratio: float = 0.2,
        window_seconds: float = 60.0,
        min_requests_for_budget: int = 10,
    ):
        """
        Initialize retry budget.

        Args:
            max_retry_ratio: Max ratio of retries to total requests
            window_seconds: Time window for measuring ratio
            min_requests_for_budget: Min requests before budget applies
        """
        self.max_retry_ratio = max_retry_ratio
        self.window_seconds = window_seconds
        self.min_requests_for_budget = min_requests_for_budget

        self._requests: list[float] = []
        self._retries: list[float] = []

    def _cleanup_old(self) -> None:
        """Remove old entries outside the window."""
        cutoff = time.time() - self.window_seconds
        self._requests = [t for t in self._requests if t > cutoff]
        self._retries = [t for t in self._retries if t > cutoff]

    def record_request(self) -> None:
        """Record a request."""
        self._cleanup_old()
        self._requests.append(time.time())

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self._cleanup_old()
        self._retries.append(time.time())

    def can_retry(self) -> bool:
        """Check if retry budget allows a retry."""
        self._cleanup_old()

        # Always allow retry if not enough data
        if len(self._requests) < self.min_requests_for_budget:
            return True

        # Calculate current retry ratio
        current_ratio = len(self._retries) / len(self._requests)
        return current_ratio < self.max_retry_ratio

    def get_stats(self) -> dict:
        """Get budget statistics."""
        self._cleanup_old()
        total_requests = len(self._requests)
        total_retries = len(self._retries)

        return {
            "requests_in_window": total_requests,
            "retries_in_window": total_retries,
            "retry_ratio": total_retries / total_requests if total_requests > 0 else 0.0,
            "budget_available": self.can_retry(),
            "max_retry_ratio": self.max_retry_ratio,
        }
