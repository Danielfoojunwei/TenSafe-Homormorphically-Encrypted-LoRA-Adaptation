"""
TenSafe Analytics Module

Provides user behavior tracking, business metrics, and operational analytics
for founders and operators.
"""

from .events import EventTracker, EventType, track_event
from .metrics import BusinessMetrics, UserMetrics, OperationalMetrics
from .dashboards import FounderDashboard
from .health_checker import (
    ProductHealthChecker,
    HealthCheck,
    HealthReport,
    HealthStatus,
    CheckCategory,
    run_health_check,
)

__all__ = [
    "EventTracker",
    "EventType",
    "track_event",
    "BusinessMetrics",
    "UserMetrics",
    "OperationalMetrics",
    "FounderDashboard",
    "ProductHealthChecker",
    "HealthCheck",
    "HealthReport",
    "HealthStatus",
    "CheckCategory",
    "run_health_check",
]
