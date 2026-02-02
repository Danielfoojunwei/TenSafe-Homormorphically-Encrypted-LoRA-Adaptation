"""
Circuit Breaker Pattern Implementation.

Provides failure isolation to prevent cascading failures:
- Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
- Configurable failure thresholds and recovery timeouts
- Async and sync support
- Metrics and monitoring

Usage:
    breaker = CircuitBreaker(
        name="kms",
        failure_threshold=5,
        recovery_timeout=30.0,
    )

    @breaker
    def call_kms():
        return kms_client.encrypt(data)

    # Or with context manager
    with breaker:
        result = kms_client.encrypt(data)
"""

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(
        self,
        name: str,
        state: CircuitState,
        remaining_time: float,
        message: Optional[str] = None,
    ):
        self.name = name
        self.state = state
        self.remaining_time = remaining_time
        super().__init__(
            message or f"Circuit breaker '{name}' is {state.value}, "
            f"retry in {remaining_time:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold to open circuit
    failure_threshold: int = 5

    # Success threshold to close circuit (in half-open state)
    success_threshold: int = 2

    # Time in seconds before attempting recovery
    recovery_timeout: float = 30.0

    # Time window for counting failures
    failure_window: float = 60.0

    # Exceptions to consider as failures
    failure_exceptions: tuple = (Exception,)

    # Exceptions to exclude from failure count
    exclude_exceptions: tuple = ()

    # Half-open max concurrent requests
    half_open_max_calls: int = 3


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Failure timestamps within window
    failure_timestamps: List[float] = field(default_factory=list)

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(timezone.utc)

    def record_failure(self, window: float) -> None:
        """Record a failed call."""
        now = time.time()
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now(timezone.utc)
        self.failure_timestamps.append(now)

        # Clean old failures outside window
        cutoff = now - window
        self.failure_timestamps = [t for t in self.failure_timestamps if t > cutoff]

    def record_rejected(self) -> None:
        """Record a rejected call (circuit open)."""
        self.total_calls += 1
        self.rejected_calls += 1

    def failures_in_window(self, window: float) -> int:
        """Count failures within the time window."""
        now = time.time()
        cutoff = now - window
        return sum(1 for t in self.failure_timestamps if t > cutoff)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time
            else None,
            "last_success_time": self.last_success_time.isoformat()
            if self.last_success_time
            else None,
            "success_rate": self.successful_calls / self.total_calls
            if self.total_calls > 0
            else 1.0,
        }


class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.

    The circuit breaker has three states:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Service is failing, calls are rejected immediately
    - HALF_OPEN: Testing if service has recovered

    Transitions:
    - CLOSED -> OPEN: When failure_threshold reached within failure_window
    - OPEN -> HALF_OPEN: After recovery_timeout expires
    - HALF_OPEN -> CLOSED: After success_threshold consecutive successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        failure_window: float = 60.0,
        failure_exceptions: tuple = (Exception,),
        exclude_exceptions: tuple = (),
        half_open_max_calls: int = 3,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            failure_threshold: Failures to trigger open state
            success_threshold: Successes to close from half-open
            recovery_timeout: Seconds before trying half-open
            failure_window: Window for counting failures
            failure_exceptions: Exception types to count as failures
            exclude_exceptions: Exception types to ignore
            half_open_max_calls: Max concurrent calls in half-open
            on_state_change: Callback for state transitions
        """
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            failure_window=failure_window,
            failure_exceptions=failure_exceptions,
            exclude_exceptions=exclude_exceptions,
            half_open_max_calls=half_open_max_calls,
        )

        self._state = CircuitState.CLOSED
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._metrics = CircuitMetrics()
        self._on_state_change = on_state_change

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        """Get circuit metrics."""
        return self._metrics

    def _check_state_transition(self) -> None:
        """Check if state should transition (must hold lock)."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._opened_at and time.time() - self._opened_at >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state (must hold lock)."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._metrics.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None

        logger.info(
            f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}",
            extra={
                "circuit_breaker": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed (must hold lock)."""
        self._check_state_transition()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            return False

        # HALF_OPEN: allow limited requests
        if self._half_open_calls < self.config.half_open_max_calls:
            self._half_open_calls += 1
            return True

        return False

    def _record_success(self) -> None:
        """Record successful call (must hold lock)."""
        self._metrics.record_success()

        if self._state == CircuitState.HALF_OPEN:
            if self._metrics.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, exception: Exception) -> None:
        """Record failed call (must hold lock)."""
        # Check if this exception should be excluded
        if isinstance(exception, self.config.exclude_exceptions):
            return

        # Check if this is a failure exception
        if not isinstance(exception, self.config.failure_exceptions):
            return

        self._metrics.record_failure(self.config.failure_window)

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open opens the circuit
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            # Check if threshold reached
            failures = self._metrics.failures_in_window(self.config.failure_window)
            if failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _get_remaining_time(self) -> float:
        """Get remaining time until recovery (for error message)."""
        if self._opened_at is None:
            return 0.0
        elapsed = time.time() - self._opened_at
        return max(0.0, self.config.recovery_timeout - elapsed)

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if not self._should_allow_request():
                self._metrics.record_rejected()
                raise CircuitBreakerError(
                    name=self.name,
                    state=self._state,
                    remaining_time=self._get_remaining_time(),
                )

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result
        except Exception as e:
            with self._lock:
                self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute async function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        # Create async lock if needed
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            with self._lock:
                if not self._should_allow_request():
                    self._metrics.record_rejected()
                    raise CircuitBreakerError(
                        name=self.name,
                        state=self._state,
                        remaining_time=self._get_remaining_time(),
                    )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            with self._lock:
                self._record_success()
            return result
        except Exception as e:
            with self._lock:
                self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for wrapping functions with circuit breaker.

        Usage:
            @circuit_breaker
            def my_function():
                pass
        """
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await self.call_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return self.call(func, *args, **kwargs)

            return sync_wrapper

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        with self._lock:
            if not self._should_allow_request():
                self._metrics.record_rejected()
                raise CircuitBreakerError(
                    name=self.name,
                    state=self._state,
                    remaining_time=self._get_remaining_time(),
                )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        with self._lock:
            if exc_val is not None:
                self._record_failure(exc_val)
            else:
                self._record_success()

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            with self._lock:
                if not self._should_allow_request():
                    self._metrics.record_rejected()
                    raise CircuitBreakerError(
                        name=self.name,
                        state=self._state,
                        remaining_time=self._get_remaining_time(),
                    )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        with self._lock:
            if exc_val is not None:
                self._record_failure(exc_val)
            else:
                self._record_success()

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._metrics.consecutive_failures = 0
            self._metrics.failure_timestamps.clear()

    def force_open(self) -> None:
        """Manually force the circuit breaker open."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            self._check_state_transition()
            return {
                "name": self.name,
                "state": self._state.value,
                "remaining_recovery_time": self._get_remaining_time()
                if self._state == CircuitState.OPEN
                else None,
                "metrics": self._metrics.to_dict(),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "failure_window": self.config.failure_window,
                },
            }


# Global registry for circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    **kwargs: Any,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    Args:
        name: Circuit breaker name
        **kwargs: Configuration options (only used on creation)

    Returns:
        Circuit breaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered circuit breakers."""
    with _registry_lock:
        return {name: cb.get_status() for name, cb in _circuit_breakers.items()}
