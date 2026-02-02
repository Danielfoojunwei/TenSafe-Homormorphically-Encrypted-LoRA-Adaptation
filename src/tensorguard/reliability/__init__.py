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

from .circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerError
from .retry import (
    retry_with_backoff,
    RetryConfig,
    RetryExhaustedError,
    async_retry_with_backoff,
)
from .timeout import (
    with_timeout,
    TimeoutConfig,
    OperationTimeoutError,
    async_timeout,
)
from .health import (
    HealthAggregator,
    HealthCheck,
    HealthStatus,
    ComponentHealth,
)
from .shutdown import (
    GracefulShutdown,
    ShutdownHandler,
    ShutdownPhase,
)
from .resource_pool import (
    ResourcePool,
    PoolExhaustedError,
    PooledResource,
)
from .bulkhead import (
    Bulkhead,
    BulkheadFullError,
    SemaphoreBulkhead,
    ThreadPoolBulkhead,
)
from .fallback import (
    Fallback,
    FallbackChain,
    cached_fallback,
    default_fallback,
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
