"""
TenSafe Analytics Module

Provides user behavior tracking, business metrics, and operational analytics
for founders and operators.
"""

from .dashboards import FounderDashboard
from .events import EventTracker, EventType, track_event
from .health_checker import (
    CheckCategory,
    HealthCheck,
    HealthReport,
    HealthStatus,
    ProductHealthChecker,
    run_health_check,
)
from .metrics import BusinessMetrics, OperationalMetrics, UserMetrics

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
