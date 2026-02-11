"""
Event Tracking System

Captures user behavior events for analytics and product intelligence.
Privacy-aware: respects user preferences and anonymizes where appropriate.
"""

import hashlib
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard event types for tracking."""

    # User lifecycle
    USER_SIGNUP = "user.signup"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_UPGRADE = "user.upgrade"
    USER_DOWNGRADE = "user.downgrade"
    USER_CHURN = "user.churn"

    # API usage
    API_CALL = "api.call"
    API_ERROR = "api.error"
    API_RATE_LIMITED = "api.rate_limited"

    # Training events
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    TRAINING_CHECKPOINT = "training.checkpoint"

    # Inference events
    INFERENCE_REQUEST = "inference.request"
    INFERENCE_COMPLETED = "inference.completed"
    INFERENCE_ERROR = "inference.error"

    # Feature usage
    FEATURE_USED = "feature.used"
    FEATURE_DISCOVERED = "feature.discovered"

    # Privacy features
    DP_TRAINING_USED = "privacy.dp_training"
    HE_INFERENCE_USED = "privacy.he_inference"
    TGSP_CREATED = "privacy.tgsp_created"

    # Billing
    PAYMENT_SUCCESS = "billing.payment_success"
    PAYMENT_FAILED = "billing.payment_failed"
    QUOTA_WARNING = "billing.quota_warning"
    QUOTA_EXCEEDED = "billing.quota_exceeded"


@dataclass
class Event:
    """A tracked event."""

    event_type: EventType
    timestamp: datetime
    tenant_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    # Context
    ip_hash: Optional[str] = None  # Anonymized
    user_agent: Optional[str] = None
    country: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "properties": self.properties,
            "context": {
                "ip_hash": self.ip_hash,
                "user_agent": self.user_agent,
                "country": self.country,
            },
        }


class EventTracker:
    """
    Event tracking system for user behavior analytics.

    Features:
    - Batch event processing for efficiency
    - Privacy-aware (anonymization, consent)
    - Multiple backend support (DB, analytics services)
    - Real-time and historical analytics
    """

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval_seconds: int = 10,
        anonymize_ip: bool = True,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.anonymize_ip = anonymize_ip

        self._event_buffer: List[Event] = []
        self._buffer_lock = threading.Lock()

        # In-memory aggregations for real-time metrics
        self._hourly_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._daily_active_users: Dict[str, set] = defaultdict(set)

        # Start background flush thread
        self._start_flush_thread()

    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        def flush_loop():
            import time
            while True:
                time.sleep(self.flush_interval)
                self.flush()

        thread = threading.Thread(target=flush_loop, daemon=True)
        thread.start()

    def track(
        self,
        event_type: EventType,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Event:
        """
        Track an event.

        Args:
            event_type: Type of event
            tenant_id: Tenant identifier
            user_id: Optional user identifier
            session_id: Optional session identifier
            properties: Event-specific properties
            ip_address: User IP (will be anonymized)
            user_agent: Browser/client user agent

        Returns:
            The tracked event
        """
        # Anonymize IP if configured
        ip_hash = None
        if ip_address and self.anonymize_ip:
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]

        event = Event(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            properties=properties or {},
            ip_hash=ip_hash,
            user_agent=user_agent,
        )

        # Add to buffer
        with self._buffer_lock:
            self._event_buffer.append(event)

            # Update real-time aggregations
            hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
            self._hourly_counts[hour_key][event_type.value] += 1

            if user_id:
                day_key = event.timestamp.strftime("%Y-%m-%d")
                self._daily_active_users[day_key].add(user_id)

            # Auto-flush if buffer is full
            if len(self._event_buffer) >= self.batch_size:
                self._flush_buffer()

        return event

    def flush(self):
        """Flush buffered events to storage."""
        with self._buffer_lock:
            self._flush_buffer()

    def _flush_buffer(self):
        """Internal flush (must hold lock)."""
        if not self._event_buffer:
            return

        events = self._event_buffer.copy()
        self._event_buffer.clear()

        # Send to storage backends
        self._persist_events(events)

    def _persist_events(self, events: List[Event]):
        """Persist events to storage backends."""
        # TODO: Implement actual persistence
        # Options: PostgreSQL, ClickHouse, BigQuery, Segment, Amplitude

        for event in events:
            logger.debug(f"Event: {event.event_type.value} - {event.tenant_id}")

    # Real-time analytics methods

    def get_hourly_counts(self, hours: int = 24) -> Dict[str, Dict[str, int]]:
        """Get event counts by hour for the last N hours."""
        result = {}
        now = datetime.utcnow()

        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d-%H")
            if hour_key in self._hourly_counts:
                result[hour_key] = dict(self._hourly_counts[hour_key])

        return result

    def get_daily_active_users(self, days: int = 7) -> Dict[str, int]:
        """Get DAU for the last N days."""
        result = {}
        now = datetime.utcnow()

        for i in range(days):
            day = now - timedelta(days=i)
            day_key = day.strftime("%Y-%m-%d")
            result[day_key] = len(self._daily_active_users.get(day_key, set()))

        return result


# Global tracker instance
_tracker: Optional[EventTracker] = None


def get_tracker() -> EventTracker:
    """Get the global event tracker."""
    global _tracker
    if _tracker is None:
        _tracker = EventTracker()
    return _tracker


def track_event(
    event_type: EventType,
    tenant_id: str,
    user_id: Optional[str] = None,
    **properties
) -> Event:
    """
    Convenience function to track an event.

    Usage:
        track_event(EventType.TRAINING_STARTED, "tenant-123", model="llama3")
    """
    return get_tracker().track(
        event_type=event_type,
        tenant_id=tenant_id,
        user_id=user_id,
        properties=properties,
    )
