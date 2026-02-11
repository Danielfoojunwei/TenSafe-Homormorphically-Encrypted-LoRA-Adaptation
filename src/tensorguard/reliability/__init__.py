"""
TensorGuard Reliability Module.

Provides production-grade reliability patterns for stable performance:
- Circuit breaker for failure isolation
- Retry with exponential backoff
- Timeout handling
- Health check aggregation
- Graceful shutdown
- Resource pool management
- Bulkhead isolation
- Fallback and degradation utilities

Usage:
    from tensorguard.reliability import (
        CircuitBreaker,
        retry_with_backoff,
        with_timeout,
        HealthAggregator,
        GracefulShutdown,
        ResourcePool,
        Bulkhead,
        Fallback,
    )
"""

from .bulkhead import (
    Bulkhead,
    BulkheadFullError,
    SemaphoreBulkhead,
    ThreadPoolBulkhead,
)
from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState
from .fallback import (
    Fallback,
    FallbackChain,
    cached_fallback,
    default_fallback,
)
from .health import (
    ComponentHealth,
    HealthAggregator,
    HealthCheck,
    HealthStatus,
)
from .resource_pool import (
    PooledResource,
    PoolExhaustedError,
    ResourcePool,
)
from .retry import (
    RetryConfig,
    RetryExhaustedError,
    async_retry_with_backoff,
    retry_with_backoff,
)
from .shutdown import (
    GracefulShutdown,
    ShutdownHandler,
    ShutdownPhase,
)
from .timeout import (
    OperationTimeoutError,
    TimeoutConfig,
    async_timeout,
    with_timeout,
)

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerError",
    # Retry
    "retry_with_backoff",
    "RetryConfig",
    "RetryExhaustedError",
    "async_retry_with_backoff",
    # Timeout
    "with_timeout",
    "TimeoutConfig",
    "OperationTimeoutError",
    "async_timeout",
    # Health
    "HealthAggregator",
    "HealthCheck",
    "HealthStatus",
    "ComponentHealth",
    # Shutdown
    "GracefulShutdown",
    "ShutdownHandler",
    "ShutdownPhase",
    # Resource pool
    "ResourcePool",
    "PoolExhaustedError",
    "PooledResource",
    # Bulkhead
    "Bulkhead",
    "BulkheadFullError",
    "SemaphoreBulkhead",
    "ThreadPoolBulkhead",
    # Fallback
    "Fallback",
    "FallbackChain",
    "cached_fallback",
    "default_fallback",
]
