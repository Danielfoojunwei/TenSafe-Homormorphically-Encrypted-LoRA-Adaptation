"""TenSafe Observability Module.

Provides comprehensive observability through OpenTelemetry:
- Metrics: Prometheus-compatible metrics export
- Tracing: Distributed tracing with Jaeger/Tempo
- Logging: Structured logging with correlation IDs
- Privacy-aware: Redacts sensitive data, tracks privacy budget
"""

from .middleware import TenSafeTracingMiddleware
from .setup import TenSafeMetrics, setup_observability

__all__ = [
    "setup_observability",
    "TenSafeMetrics",
    "TenSafeTracingMiddleware",
]
