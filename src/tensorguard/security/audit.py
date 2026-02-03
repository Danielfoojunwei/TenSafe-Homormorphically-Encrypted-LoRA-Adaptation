"""
Security Audit Logging Module.

Provides comprehensive security event auditing:
- Authentication events (login, logout, failed attempts)
- Authorization events (access granted/denied)
- Data access events (sensitive data viewed/modified)
- Configuration changes
- Security incidents
- Compliance-relevant events

Supports multiple backends:
- File-based audit logs (default)
- Database storage
- External SIEM integration
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of security audit events."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILED = "auth.login.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_ISSUED = "auth.token.issued"
    AUTH_TOKEN_REFRESHED = "auth.token.refreshed"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    AUTH_MFA_SUCCESS = "auth.mfa.success"
    AUTH_MFA_FAILED = "auth.mfa.failed"
    AUTH_PASSWORD_CHANGED = "auth.password.changed"
    AUTH_PASSWORD_RESET = "auth.password.reset"

    # Authorization events
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_ROLE_CHANGED = "authz.role.changed"
    AUTHZ_PERMISSION_CHANGED = "authz.permission.changed"

    # Data access events
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_ENCRYPTED = "data.encrypted"
    DATA_DECRYPTED = "data.decrypted"

    # Key management events
    KEY_GENERATED = "key.generated"
    KEY_ROTATED = "key.rotated"
    KEY_REVOKED = "key.revoked"
    KEY_ACCESSED = "key.accessed"
    KEY_EXPORTED = "key.exported"

    # Configuration events
    CONFIG_CHANGED = "config.changed"
    CONFIG_SECURITY_CHANGED = "config.security.changed"
    CONFIG_PRIVACY_CHANGED = "config.privacy.changed"

    # Security incidents
    SECURITY_RATE_LIMITED = "security.rate_limited"
    SECURITY_BLOCKED = "security.blocked"
    SECURITY_SUSPICIOUS = "security.suspicious"
    SECURITY_VIOLATION = "security.violation"
    SECURITY_INTRUSION_ATTEMPT = "security.intrusion_attempt"

    # Privacy events
    PRIVACY_CONSENT_GRANTED = "privacy.consent.granted"
    PRIVACY_CONSENT_REVOKED = "privacy.consent.revoked"
    PRIVACY_DATA_REQUESTED = "privacy.data.requested"
    PRIVACY_DATA_DELETED = "privacy.data.deleted"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

    # Training events (ML-specific)
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_DP_BUDGET_CONSUMED = "training.dp_budget.consumed"
    TRAINING_HE_OPERATION = "training.he.operation"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """A security audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity

    # Actor information
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Action details
    action: Optional[str] = None
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    outcome: str = "success"

    # Context
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    # Compliance
    compliance_relevant: bool = False
    retention_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "action": self.action,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "outcome": self.outcome,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "details": self._filter_sensitive(self.details),
            "compliance_relevant": self.compliance_relevant,
        }

    def _filter_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive fields from details."""
        sensitive_keys = {
            "password",
            "secret",
            "token",
            "api_key",
            "private_key",
            "credential",
        }

        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive(value)
            else:
                filtered[key] = value
        return filtered

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), default=str)


class SecurityAuditLog:
    """
    Security audit log manager.

    Provides tamper-evident logging with support for:
    - File-based logs with rotation
    - Hash chaining for integrity verification
    - Multiple output destinations
    - Async processing for performance
    """

    HASH_ALGORITHM = "sha256"

    def __init__(
        self,
        log_path: Optional[str] = None,
        enable_hash_chain: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize security audit log.

        Args:
            log_path: Path for audit log files
            enable_hash_chain: Enable hash chaining for tamper detection
            buffer_size: Number of events to buffer before flushing
            flush_interval: Seconds between automatic flushes
        """
        self.log_path = Path(log_path or os.getenv("TG_AUDIT_LOG_PATH", "/var/log/tensafe/audit"))
        self.enable_hash_chain = enable_hash_chain
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        # State
        self._buffer: List[SecurityEvent] = []
        self._lock = asyncio.Lock()
        self._last_hash: Optional[str] = None
        self._event_count = 0
        self._flush_task: Optional[asyncio.Task] = None

        # Callbacks for external integrations
        self._callbacks: List[Callable[[SecurityEvent], None]] = []

        # Ensure log directory exists
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Load last hash from existing log
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the last hash from the most recent log file."""
        if not self.enable_hash_chain:
            return

        try:
            log_files = sorted(self.log_path.glob("audit-*.jsonl"), reverse=True)
            if log_files:
                with open(log_files[0]) as f:
                    lines = f.readlines()
                    if lines:
                        last_event = json.loads(lines[-1])
                        self._last_hash = last_event.get("_hash")
        except Exception as e:
            logger.warning(f"Could not load last audit hash: {e}")

    def _compute_hash(self, event: SecurityEvent) -> str:
        """Compute hash for event, chained with previous hash."""
        event_data = event.to_json()

        if self.enable_hash_chain and self._last_hash:
            data_to_hash = f"{self._last_hash}:{event_data}"
        else:
            data_to_hash = event_data

        return hashlib.sha256(data_to_hash.encode()).hexdigest()

    async def log(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        request: Optional[Request] = None,
        **kwargs,
    ) -> SecurityEvent:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            severity: Event severity
            user_id: User identifier
            tenant_id: Tenant identifier
            request: FastAPI request for context extraction
            **kwargs: Additional event fields

        Returns:
            Created SecurityEvent
        """
        # Create event
        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            user_id=user_id,
            tenant_id=tenant_id,
            **kwargs,
        )

        # Extract context from request
        if request:
            event.ip_address = self._get_client_ip(request)
            event.user_agent = request.headers.get("User-Agent", "")[:500]
            event.request_id = request.headers.get("X-Request-ID", "")

            # Extract user from request state if available
            if hasattr(request.state, "user"):
                user = request.state.user
                event.user_id = event.user_id or getattr(user, "id", None)
                event.tenant_id = event.tenant_id or getattr(user, "tenant_id", None)

        # Add to buffer
        async with self._lock:
            self._buffer.append(event)
            self._event_count += 1

            # Flush if buffer is full
            if len(self._buffer) >= self.buffer_size:
                await self._flush_buffer()

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Audit callback error: {e}")

        # Log to standard logger for immediate visibility
        log_level = getattr(logging, severity.value.upper(), logging.INFO)
        logger.log(
            log_level,
            f"AUDIT: {event_type.value} | user={user_id} | outcome={event.outcome}",
            extra={"audit_event": event.to_dict()},
        )

        return event

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _flush_buffer(self) -> None:
        """Flush buffered events to storage."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        # Write to file
        log_file = self.log_path / f"audit-{datetime.now().strftime('%Y-%m-%d')}.jsonl"

        try:
            with open(log_file, "a") as f:
                for event in events:
                    # Compute and add hash
                    event_hash = self._compute_hash(event)
                    self._last_hash = event_hash

                    record = event.to_dict()
                    record["_hash"] = event_hash
                    record["_previous_hash"] = self._last_hash

                    f.write(json.dumps(record, default=str) + "\n")

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            # Re-add events to buffer for retry
            async with self._lock:
                self._buffer = events + self._buffer

    async def start_flush_task(self) -> None:
        """Start background flush task."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop_flush_task(self) -> None:
        """Stop background flush task."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

            # Final flush
            async with self._lock:
                await self._flush_buffer()

            self._flush_task = None

    async def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                async with self._lock:
                    await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audit flush error: {e}")

    def add_callback(self, callback: Callable[[SecurityEvent], None]) -> None:
        """Add a callback for external integrations (SIEM, etc.)."""
        self._callbacks.append(callback)

    async def verify_integrity(self, log_file: Optional[Path] = None) -> bool:
        """
        Verify the integrity of the audit log using hash chain.

        Args:
            log_file: Specific log file to verify (default: latest)

        Returns:
            True if integrity verified, False if tampering detected
        """
        if not self.enable_hash_chain:
            logger.warning("Hash chain not enabled, cannot verify integrity")
            return True

        if log_file is None:
            log_files = sorted(self.log_path.glob("audit-*.jsonl"), reverse=True)
            if not log_files:
                return True
            log_file = log_files[0]

        try:
            with open(log_file) as f:
                previous_hash = None

                for line_num, line in enumerate(f, 1):
                    record = json.loads(line)

                    # Reconstruct event
                    stored_hash = record.pop("_hash", None)
                    stored_prev_hash = record.pop("_previous_hash", None)

                    if previous_hash and stored_prev_hash != previous_hash:
                        logger.error(
                            f"Audit log integrity violation at line {line_num}: "
                            f"previous hash mismatch"
                        )
                        return False

                    # Verify hash
                    event_json = json.dumps(record, default=str)
                    if previous_hash:
                        data_to_hash = f"{previous_hash}:{event_json}"
                    else:
                        data_to_hash = event_json

                    computed_hash = hashlib.sha256(data_to_hash.encode()).hexdigest()

                    if computed_hash != stored_hash:
                        logger.error(
                            f"Audit log integrity violation at line {line_num}: "
                            f"hash mismatch"
                        )
                        return False

                    previous_hash = stored_hash

            logger.info(f"Audit log integrity verified: {log_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to verify audit log integrity: {e}")
            return False

    # Convenience methods for common events

    async def log_login_success(
        self,
        user_id: str,
        request: Optional[Request] = None,
        **kwargs,
    ) -> SecurityEvent:
        """Log successful login."""
        return await self.log(
            AuditEventType.AUTH_LOGIN_SUCCESS,
            AuditSeverity.INFO,
            user_id=user_id,
            request=request,
            action="login",
            outcome="success",
            compliance_relevant=True,
            **kwargs,
        )

    async def log_login_failed(
        self,
        user_id: Optional[str],
        request: Optional[Request] = None,
        reason: str = "invalid_credentials",
        **kwargs,
    ) -> SecurityEvent:
        """Log failed login attempt."""
        return await self.log(
            AuditEventType.AUTH_LOGIN_FAILED,
            AuditSeverity.WARNING,
            user_id=user_id,
            request=request,
            action="login",
            outcome="failure",
            details={"reason": reason},
            compliance_relevant=True,
            **kwargs,
        )

    async def log_access_denied(
        self,
        user_id: str,
        resource: str,
        request: Optional[Request] = None,
        reason: str = "insufficient_permissions",
        **kwargs,
    ) -> SecurityEvent:
        """Log access denied event."""
        return await self.log(
            AuditEventType.AUTHZ_ACCESS_DENIED,
            AuditSeverity.WARNING,
            user_id=user_id,
            request=request,
            action="access",
            resource=resource,
            outcome="denied",
            details={"reason": reason},
            compliance_relevant=True,
            **kwargs,
        )

    async def log_security_violation(
        self,
        event_details: Dict[str, Any],
        request: Optional[Request] = None,
        **kwargs,
    ) -> SecurityEvent:
        """Log security violation."""
        return await self.log(
            AuditEventType.SECURITY_VIOLATION,
            AuditSeverity.CRITICAL,
            request=request,
            action="security_violation",
            outcome="blocked",
            details=event_details,
            compliance_relevant=True,
            **kwargs,
        )

    async def log_key_operation(
        self,
        operation: str,
        key_id: str,
        user_id: Optional[str] = None,
        request: Optional[Request] = None,
        **kwargs,
    ) -> SecurityEvent:
        """Log key management operation."""
        event_type = {
            "generate": AuditEventType.KEY_GENERATED,
            "rotate": AuditEventType.KEY_ROTATED,
            "revoke": AuditEventType.KEY_REVOKED,
            "access": AuditEventType.KEY_ACCESSED,
            "export": AuditEventType.KEY_EXPORTED,
        }.get(operation, AuditEventType.KEY_ACCESSED)

        return await self.log(
            event_type,
            AuditSeverity.INFO,
            user_id=user_id,
            request=request,
            action=operation,
            resource="key",
            resource_id=key_id,
            compliance_relevant=True,
            **kwargs,
        )


class AuditMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request/response auditing.

    Logs all API requests for security analysis and compliance.
    """

    def __init__(
        self,
        app,
        audit_log: Optional[SecurityAuditLog] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        """
        Initialize audit middleware.

        Args:
            app: FastAPI application
            audit_log: Security audit log instance
            exclude_paths: Paths to exclude from auditing
        """
        super().__init__(app)
        self.audit_log = audit_log or SecurityAuditLog()
        self.exclude_paths = set(exclude_paths or ["/health", "/ready", "/live", "/metrics"])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Audit request/response."""
        path = request.url.path

        # Skip excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Determine event type based on response
        if response.status_code >= 500:
            event_type = AuditEventType.SYSTEM_ERROR
            severity = AuditSeverity.ERROR
        elif response.status_code == 401:
            event_type = AuditEventType.AUTH_LOGIN_FAILED
            severity = AuditSeverity.WARNING
        elif response.status_code == 403:
            event_type = AuditEventType.AUTHZ_ACCESS_DENIED
            severity = AuditSeverity.WARNING
        elif response.status_code == 429:
            event_type = AuditEventType.SECURITY_RATE_LIMITED
            severity = AuditSeverity.WARNING
        else:
            event_type = AuditEventType.DATA_READ
            severity = AuditSeverity.DEBUG

        # Log the event
        await self.audit_log.log(
            event_type=event_type,
            severity=severity,
            request=request,
            action=request.method,
            resource=path,
            outcome="success" if response.status_code < 400 else "failure",
            details={
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "query_params": dict(request.query_params),
            },
        )

        return response


# Singleton instance
_default_audit_log: Optional[SecurityAuditLog] = None


def get_audit_log() -> SecurityAuditLog:
    """Get or create the default security audit log."""
    global _default_audit_log
    if _default_audit_log is None:
        _default_audit_log = SecurityAuditLog()
    return _default_audit_log
