"""
Compliance Event Telemetry

Structured compliance events for ISO/IEC 27701, ISO/IEC 27001, and SOC 2 evidence collection.
All events are designed to be machine-readable, reproducible, and privacy-preserving.

Usage:
    from tensorguard.telemetry import ComplianceEventEmitter, ComplianceEventType

    emitter = ComplianceEventEmitter(environment="smoke")
    emitter.emit(
        event_type=ComplianceEventType.PII_SCAN,
        outcome="pass",
        details={"count": 0, "scope": "logs"},
        artifact_refs=["reports/compliance/abc123/pii_scan.json"]
    )
"""

import os
import json
import time
import hashlib
import threading
import logging
import subprocess
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ComplianceEventType(str, Enum):
    """Compliance event types mapped to control frameworks."""

    # Access Control (ISO27001 A.9, SOC2 CC6)
    AUTH = "AUTH"                         # Authentication events
    ACCESS = "ACCESS"                     # Authorization/access control events

    # Cryptography (ISO27001 A.10, SOC2 CC6.7)
    ENCRYPTION = "ENCRYPTION"             # Encryption posture events
    KEY_MANAGEMENT = "KEY_MANAGEMENT"     # Key rotation, KEK/DEK events

    # Logging & Monitoring (ISO27001 A.12.4, SOC2 CC7)
    AUDIT_LOG = "AUDIT_LOG"               # Audit log integrity events
    LOG_COVERAGE = "LOG_COVERAGE"         # Event coverage analysis

    # Data Retention (ISO27701, SOC2 Privacy)
    RETENTION = "RETENTION"               # Data retention enforcement
    DATA_DISPOSAL = "DATA_DISPOSAL"       # Secure data deletion

    # Privacy (ISO27701, SOC2 Privacy)
    PII_SCAN = "PII_SCAN"                 # PII detection scan results
    DATA_CLASSIFICATION = "DATA_CLASSIFICATION"  # Data classification events
    CONSENT = "CONSENT"                   # Consent tracking
    DATA_MINIMIZATION = "DATA_MINIMIZATION"  # Data minimization events

    # Incident Response (ISO27001 A.16, SOC2 CC7.4)
    INCIDENT = "INCIDENT"                 # Incident detection/response
    ALERT = "ALERT"                       # Alert generation

    # Change Management (ISO27001 A.12.1, SOC2 CC8)
    CHANGE = "CHANGE"                     # Change management events
    DEPLOYMENT = "DEPLOYMENT"             # Deployment events

    # Backup & Recovery (ISO27001 A.12.3, SOC2 A1)
    BACKUP = "BACKUP"                     # Backup events
    RECOVERY = "RECOVERY"                 # Recovery events

    # Processing Integrity (SOC2 PI1)
    VALIDATION = "VALIDATION"             # Input/output validation
    INTEGRITY = "INTEGRITY"               # Data integrity checks

    # Supply Chain (ISO27001 A.15)
    DEPENDENCY = "DEPENDENCY"             # Dependency scanning
    SECRETS_SCAN = "SECRETS_SCAN"         # Secrets detection

    # Availability (SOC2 A1)
    HEALTH_CHECK = "HEALTH_CHECK"         # Health/availability checks
    DEGRADATION = "DEGRADATION"           # Graceful degradation events


class Severity(str, Enum):
    """Event severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Outcome(str, Enum):
    """Event outcome status."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceEvent:
    """
    Structured compliance event for evidence collection.

    All fields are designed to be machine-readable and privacy-preserving.
    Sensitive data is never stored - only counts, hashes, and references.
    """

    # Temporal
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    timestamp_unix: float = field(default_factory=time.time)

    # Identity
    git_sha: str = ""
    run_id: str = ""
    component: str = ""
    environment: str = "development"  # smoke/full/development/production

    # Tenant (hashed for privacy)
    tenant_id_hash: Optional[str] = None

    # Event Core
    event_type: str = ""
    severity: str = Severity.INFO.value
    outcome: str = Outcome.PASS.value

    # Details (redacted/sanitized)
    details: Dict[str, Any] = field(default_factory=dict)

    # Evidence References (file paths, not content)
    artifact_refs: List[str] = field(default_factory=list)

    # Control Mapping
    control_refs: Dict[str, List[str]] = field(default_factory=dict)

    # Integrity
    event_hash: str = ""

    def __post_init__(self):
        """Compute event hash for integrity verification."""
        if not self.event_hash:
            self.event_hash = self._compute_hash()
        if not self.control_refs:
            self.control_refs = self._map_controls()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of event content for integrity."""
        content = f"{self.timestamp}:{self.git_sha}:{self.event_type}:{self.outcome}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _map_controls(self) -> Dict[str, List[str]]:
        """Map event type to control framework references."""
        CONTROL_MAP = {
            ComplianceEventType.AUTH.value: {
                "iso27001": ["A.9 Access Control"],
                "iso27701": [],
                "soc2": ["CC6.1 Logical Access"]
            },
            ComplianceEventType.ACCESS.value: {
                "iso27001": ["A.9 Access Control"],
                "iso27701": [],
                "soc2": ["CC6.1 Logical Access", "CC6.3 Role-Based Access"]
            },
            ComplianceEventType.ENCRYPTION.value: {
                "iso27001": ["A.10 Cryptography"],
                "iso27701": [],
                "soc2": ["CC6.7 Encryption", "C1 Confidentiality"]
            },
            ComplianceEventType.KEY_MANAGEMENT.value: {
                "iso27001": ["A.10 Cryptography"],
                "iso27701": [],
                "soc2": ["CC6.7 Encryption"]
            },
            ComplianceEventType.AUDIT_LOG.value: {
                "iso27001": ["A.12.4 Logging"],
                "iso27701": [],
                "soc2": ["CC7.2 Monitoring"]
            },
            ComplianceEventType.LOG_COVERAGE.value: {
                "iso27001": ["A.12.4 Logging"],
                "iso27701": [],
                "soc2": ["CC7.2 Monitoring"]
            },
            ComplianceEventType.RETENTION.value: {
                "iso27001": [],
                "iso27701": ["PIM-3 Retention"],
                "soc2": ["P6 Privacy - Retention"]
            },
            ComplianceEventType.DATA_DISPOSAL.value: {
                "iso27001": [],
                "iso27701": ["PIM-3 Retention"],
                "soc2": ["P7 Privacy - Disposal"]
            },
            ComplianceEventType.PII_SCAN.value: {
                "iso27001": [],
                "iso27701": ["PIM-4 Privacy by Design"],
                "soc2": ["P1 Privacy Notice", "P3 Privacy - Collection"]
            },
            ComplianceEventType.DATA_CLASSIFICATION.value: {
                "iso27001": ["A.8.2 Information Classification"],
                "iso27701": ["PIM-1 Purpose Limitation"],
                "soc2": ["C1 Confidentiality"]
            },
            ComplianceEventType.CONSENT.value: {
                "iso27001": [],
                "iso27701": ["PIM-1 Consent"],
                "soc2": ["P2 Privacy - Choice"]
            },
            ComplianceEventType.DATA_MINIMIZATION.value: {
                "iso27001": [],
                "iso27701": ["PIM-2 Data Minimization"],
                "soc2": ["P3 Privacy - Collection"]
            },
            ComplianceEventType.INCIDENT.value: {
                "iso27001": ["A.16 Incident Management"],
                "iso27701": [],
                "soc2": ["CC7.4 Response", "CC7.5 Recovery"]
            },
            ComplianceEventType.ALERT.value: {
                "iso27001": ["A.16 Incident Management"],
                "iso27701": [],
                "soc2": ["CC7.3 Detection"]
            },
            ComplianceEventType.CHANGE.value: {
                "iso27001": ["A.12.1 Change Management", "A.14.2 Development Security"],
                "iso27701": [],
                "soc2": ["CC8.1 Change Management"]
            },
            ComplianceEventType.DEPLOYMENT.value: {
                "iso27001": ["A.12.1 Change Management"],
                "iso27701": [],
                "soc2": ["CC8.1 Change Management"]
            },
            ComplianceEventType.BACKUP.value: {
                "iso27001": ["A.12.3 Backup", "A.17 Business Continuity"],
                "iso27701": [],
                "soc2": ["A1.2 Availability"]
            },
            ComplianceEventType.RECOVERY.value: {
                "iso27001": ["A.17 Business Continuity"],
                "iso27701": [],
                "soc2": ["A1.2 Availability"]
            },
            ComplianceEventType.VALIDATION.value: {
                "iso27001": [],
                "iso27701": [],
                "soc2": ["PI1.1 Processing Integrity"]
            },
            ComplianceEventType.INTEGRITY.value: {
                "iso27001": ["A.12.2 Protection from Malware"],
                "iso27701": [],
                "soc2": ["PI1.4 Processing Integrity"]
            },
            ComplianceEventType.DEPENDENCY.value: {
                "iso27001": ["A.15 Supplier Relationships"],
                "iso27701": [],
                "soc2": ["CC9.2 Vendor Management"]
            },
            ComplianceEventType.SECRETS_SCAN.value: {
                "iso27001": ["A.9.4 System Access Control"],
                "iso27701": [],
                "soc2": ["CC6.1 Logical Access"]
            },
            ComplianceEventType.HEALTH_CHECK.value: {
                "iso27001": [],
                "iso27701": [],
                "soc2": ["A1.1 Availability"]
            },
            ComplianceEventType.DEGRADATION.value: {
                "iso27001": ["A.17 Business Continuity"],
                "iso27701": [],
                "soc2": ["A1.2 Availability"]
            },
        }
        return CONTROL_MAP.get(self.event_type, {"iso27001": [], "iso27701": [], "soc2": []})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ComplianceEventEmitter:
    """
    Emits compliance events to structured log files.

    Thread-safe, batches events, and maintains event chain integrity.
    """

    _instance: Optional["ComplianceEventEmitter"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        output_dir: str = "reports/compliance",
        environment: str = "development",
        run_id: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.environment = environment
        self.run_id = run_id or f"run-{int(time.time())}"
        self.git_sha = self._get_git_sha()
        self._events: List[ComplianceEvent] = []
        self._events_lock = threading.Lock()
        self._chain_hash = ""

    @classmethod
    def get_instance(
        cls,
        output_dir: str = "reports/compliance",
        environment: str = "development",
    ) -> "ComplianceEventEmitter":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(output_dir, environment)
        return cls._instance

    def _get_git_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return "unknown"

    def _get_git_dirty(self) -> bool:
        """Check if git tree is dirty."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def emit(
        self,
        event_type: ComplianceEventType,
        outcome: Outcome = Outcome.PASS,
        severity: Severity = Severity.INFO,
        component: str = "",
        details: Optional[Dict[str, Any]] = None,
        artifact_refs: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> ComplianceEvent:
        """
        Emit a compliance event.

        Args:
            event_type: Type of compliance event
            outcome: Event outcome (pass/fail/partial/skipped)
            severity: Event severity level
            component: Component that generated the event
            details: Additional details (must not contain PII)
            artifact_refs: References to evidence artifacts
            tenant_id: Optional tenant ID (will be hashed)

        Returns:
            The emitted ComplianceEvent
        """
        # Hash tenant ID for privacy
        tenant_hash = None
        if tenant_id:
            tenant_hash = hashlib.sha256(tenant_id.encode()).hexdigest()[:16]

        event = ComplianceEvent(
            git_sha=self.git_sha,
            run_id=self.run_id,
            component=component,
            environment=self.environment,
            tenant_id_hash=tenant_hash,
            event_type=event_type.value if isinstance(event_type, ComplianceEventType) else event_type,
            severity=severity.value if isinstance(severity, Severity) else severity,
            outcome=outcome.value if isinstance(outcome, Outcome) else outcome,
            details=details or {},
            artifact_refs=artifact_refs or [],
        )

        # Update chain hash for integrity
        with self._events_lock:
            self._chain_hash = hashlib.sha256(
                f"{self._chain_hash}:{event.event_hash}".encode()
            ).hexdigest()
            self._events.append(event)

        logger.debug(f"Compliance event emitted: {event.event_type} [{event.outcome}]")
        return event

    def emit_auth_event(
        self,
        authn_method: str,
        authn_enabled: bool,
        authz_model: str,
        default_deny: bool,
        **kwargs
    ) -> ComplianceEvent:
        """Emit an authentication/authorization posture event."""
        return self.emit(
            event_type=ComplianceEventType.AUTH,
            component="auth",
            details={
                "authn_method": authn_method,
                "authn_enabled": authn_enabled,
                "authz_model": authz_model,
                "default_deny": default_deny,
            },
            **kwargs
        )

    def emit_encryption_event(
        self,
        at_rest_enabled: bool,
        in_transit_enabled: bool,
        kek_present: bool,
        dek_per_tenant: bool,
        rotation_configured: bool,
        **kwargs
    ) -> ComplianceEvent:
        """Emit an encryption posture event."""
        return self.emit(
            event_type=ComplianceEventType.ENCRYPTION,
            component="crypto",
            details={
                "at_rest_encryption_enabled": at_rest_enabled,
                "in_transit_encryption_enabled": in_transit_enabled,
                "kek_present": kek_present,
                "dek_per_tenant": dek_per_tenant,
                "key_rotation_configured": rotation_configured,
            },
            **kwargs
        )

    def emit_pii_scan_event(
        self,
        scope: str,
        scan_count: int,
        email_count: int = 0,
        phone_count: int = 0,
        ssn_count: int = 0,
        other_count: int = 0,
        **kwargs
    ) -> ComplianceEvent:
        """
        Emit a PII scan result event.

        Note: Only counts are stored, never the actual PII values.
        """
        total = email_count + phone_count + ssn_count + other_count
        outcome = Outcome.PASS if total == 0 else Outcome.FAIL

        return self.emit(
            event_type=ComplianceEventType.PII_SCAN,
            outcome=outcome,
            severity=Severity.HIGH if total > 0 else Severity.INFO,
            component="pii_scanner",
            details={
                "scope": scope,
                "files_scanned": scan_count,
                "pii_counts": {
                    "email": email_count,
                    "phone": phone_count,
                    "ssn": ssn_count,
                    "other": other_count,
                    "total": total,
                },
            },
            **kwargs
        )

    def emit_audit_log_event(
        self,
        log_enabled: bool,
        integrity_verified: bool,
        event_coverage_pct: float,
        hash_chain_valid: bool,
        **kwargs
    ) -> ComplianceEvent:
        """Emit an audit log integrity event."""
        outcome = Outcome.PASS if (integrity_verified and hash_chain_valid) else Outcome.FAIL

        return self.emit(
            event_type=ComplianceEventType.AUDIT_LOG,
            outcome=outcome,
            component="audit",
            details={
                "audit_log_enabled": log_enabled,
                "integrity_verified": integrity_verified,
                "event_coverage_pct": event_coverage_pct,
                "hash_chain_valid": hash_chain_valid,
            },
            **kwargs
        )

    def emit_retention_event(
        self,
        policy_days: int,
        enforced: bool,
        files_cleaned: int = 0,
        **kwargs
    ) -> ComplianceEvent:
        """Emit a data retention enforcement event."""
        return self.emit(
            event_type=ComplianceEventType.RETENTION,
            outcome=Outcome.PASS if enforced else Outcome.FAIL,
            component="retention",
            details={
                "retention_policy_days": policy_days,
                "retention_enforced": enforced,
                "files_cleaned": files_cleaned,
            },
            **kwargs
        )

    def emit_secrets_scan_event(
        self,
        secrets_found: int,
        files_scanned: int,
        patterns_checked: List[str],
        **kwargs
    ) -> ComplianceEvent:
        """Emit a secrets scan result event."""
        outcome = Outcome.PASS if secrets_found == 0 else Outcome.FAIL

        return self.emit(
            event_type=ComplianceEventType.SECRETS_SCAN,
            outcome=outcome,
            severity=Severity.CRITICAL if secrets_found > 0 else Severity.INFO,
            component="secrets_scanner",
            details={
                "secrets_found": secrets_found,
                "files_scanned": files_scanned,
                "patterns_checked": patterns_checked,
            },
            **kwargs
        )

    def emit_change_event(
        self,
        git_sha: str,
        dirty_tree: bool,
        ci_run_id: Optional[str] = None,
        lockfile_present: bool = True,
        **kwargs
    ) -> ComplianceEvent:
        """Emit a change management event."""
        return self.emit(
            event_type=ComplianceEventType.CHANGE,
            component="ci",
            details={
                "git_sha": git_sha,
                "dirty_tree": dirty_tree,
                "ci_run_id": ci_run_id,
                "dependency_lockfile_present": lockfile_present,
            },
            **kwargs
        )

    def emit_integrity_event(
        self,
        dataset_hash: str,
        adapter_hash: Optional[str] = None,
        determinism_score: Optional[float] = None,
        **kwargs
    ) -> ComplianceEvent:
        """Emit a processing integrity event."""
        return self.emit(
            event_type=ComplianceEventType.INTEGRITY,
            component="training",
            details={
                "dataset_hash": dataset_hash,
                "adapter_hash": adapter_hash,
                "determinism_score": determinism_score,
            },
            **kwargs
        )

    def get_events(self) -> List[ComplianceEvent]:
        """Get all emitted events."""
        with self._events_lock:
            return list(self._events)

    def get_chain_hash(self) -> str:
        """Get the current event chain hash for integrity verification."""
        return self._chain_hash

    def save_events(self, filepath: Optional[str] = None) -> str:
        """
        Save all events to a JSON file.

        Args:
            filepath: Optional custom path. Default: reports/compliance/<sha>/events.json

        Returns:
            Path to the saved file
        """
        if filepath is None:
            output_dir = self.output_dir / self.git_sha
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(output_dir / "events.json")

        events_data = {
            "metadata": {
                "git_sha": self.git_sha,
                "run_id": self.run_id,
                "environment": self.environment,
                "event_count": len(self._events),
                "chain_hash": self._chain_hash,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "events": [e.to_dict() for e in self._events]
        }

        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2, default=str)

        logger.info(f"Saved {len(self._events)} compliance events to {filepath}")
        return filepath

    def clear_events(self):
        """Clear all stored events."""
        with self._events_lock:
            self._events.clear()
            self._chain_hash = ""


# Module-level convenience function
def get_compliance_emitter(
    environment: str = "development",
    output_dir: str = "reports/compliance",
) -> ComplianceEventEmitter:
    """Get or create the global compliance event emitter."""
    return ComplianceEventEmitter.get_instance(output_dir, environment)
