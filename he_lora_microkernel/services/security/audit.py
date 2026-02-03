"""
Security Audit Logging

Provides audit logging for security-relevant events in MSS and HAS.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication/Authorization
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    ACCESS_DENIED = "access_denied"

    # Key operations
    KEY_GENERATED = "key_generated"
    KEY_ACCESSED = "key_accessed"
    KEY_DESTROYED = "key_destroyed"

    # Adapter operations
    ADAPTER_LOADED = "adapter_loaded"
    ADAPTER_UNLOADED = "adapter_unloaded"

    # Request operations
    REQUEST_RECEIVED = "request_received"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"

    # Security events
    VALIDATION_FAILED = "validation_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

    # System events
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    CONFIG_CHANGED = "config_changed"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """An audit event record."""
    event_type: AuditEventType
    timestamp: float
    severity: AuditSeverity = AuditSeverity.INFO

    # Event context
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    service: str = "mss"

    # Event details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Success/failure
    success: bool = True
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.timestamp)),
            'severity': self.severity.value,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'source_ip': self.source_ip,
            'service': self.service,
            'message': self.message,
            'details': self.details,
            'success': self.success,
            'error_code': self.error_code,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class SecurityAuditLog:
    """
    Security audit log for HE-LoRA services.

    Features:
    - Structured audit events
    - Multiple output destinations (file, stdout, syslog)
    - Async writing for minimal latency impact
    - Retention and rotation support
    """

    def __init__(
        self,
        service_name: str = "mss",
        log_file: Optional[str] = None,
        max_events: int = 10000,
        enable_stdout: bool = False,
        enable_file: bool = True,
    ):
        """
        Initialize audit log.

        Args:
            service_name: Name of the service
            log_file: Path to audit log file
            max_events: Maximum events to retain in memory
            enable_stdout: Whether to log to stdout
            enable_file: Whether to log to file
        """
        self._service_name = service_name
        self._log_file = log_file or f"/var/log/helora/{service_name}_audit.log"
        self._max_events = max_events
        self._enable_stdout = enable_stdout
        self._enable_file = enable_file

        # Event storage
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()

        # File handle
        self._file_handle = None

        # Callbacks for external integrations
        self._callbacks: List[Callable[[AuditEvent], None]] = []

    def start(self) -> None:
        """Start the audit log."""
        if self._enable_file:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
                self._file_handle = open(self._log_file, 'a')
            except OSError as e:
                logger.warning(f"Failed to open audit log file: {e}")
                self._enable_file = False

        self.log(AuditEvent(
            event_type=AuditEventType.SERVICE_STARTED,
            timestamp=time.time(),
            service=self._service_name,
            message=f"{self._service_name} audit log started",
        ))

    def stop(self) -> None:
        """Stop the audit log."""
        self.log(AuditEvent(
            event_type=AuditEventType.SERVICE_STOPPED,
            timestamp=time.time(),
            service=self._service_name,
            message=f"{self._service_name} audit log stopped",
        ))

        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: Event to log
        """
        # Ensure service is set
        if not event.service:
            event.service = self._service_name

        # Store in memory
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        # Write to outputs
        event_json = event.to_json()

        if self._enable_stdout:
            print(f"AUDIT: {event_json}")

        if self._enable_file and self._file_handle:
            try:
                self._file_handle.write(event_json + '\n')
                self._file_handle.flush()
            except OSError as e:
                logger.warning(f"Failed to write audit event: {e}")

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Audit callback error: {e}")

    def log_request(
        self,
        request_id: str,
        event_type: AuditEventType,
        message: str = "",
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict] = None,
    ) -> None:
        """Helper to log request-related events."""
        self.log(AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            request_id=request_id,
            user_id=user_id,
            source_ip=source_ip,
            message=message,
            success=success,
            details=details or {},
        ))

    def log_security_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        source_ip: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Helper to log security-related events."""
        self.log(AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            severity=severity,
            source_ip=source_ip,
            message=message,
            details=details or {},
            success=False,
        ))

    def log_key_operation(
        self,
        event_type: AuditEventType,
        key_id: str,
        message: str = "",
    ) -> None:
        """Helper to log key-related events."""
        self.log(AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            severity=AuditSeverity.INFO,
            message=message or f"Key operation: {event_type.value}",
            details={'key_id': key_id},
        ))

    def add_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add callback for audit events."""
        self._callbacks.append(callback)

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
    ) -> List[AuditEvent]:
        """Get recent audit events."""
        with self._lock:
            events = self._events[-count:]

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]

        return events

    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security-related events."""
        with self._lock:
            events = self._events[:]

        security_events = [
            e for e in events
            if e.event_type in (
                AuditEventType.AUTH_FAILURE,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.VALIDATION_FAILED,
                AuditEventType.RATE_LIMIT_EXCEEDED,
                AuditEventType.SUSPICIOUS_ACTIVITY,
            )
        ]

        # Count by type
        by_type = {}
        for e in security_events:
            by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1

        # Recent IPs with issues
        suspicious_ips = {}
        for e in security_events:
            if e.source_ip:
                suspicious_ips[e.source_ip] = suspicious_ips.get(e.source_ip, 0) + 1

        return {
            'total_security_events': len(security_events),
            'by_type': by_type,
            'suspicious_ips': suspicious_ips,
            'recent': [e.to_dict() for e in security_events[-10:]],
        }
