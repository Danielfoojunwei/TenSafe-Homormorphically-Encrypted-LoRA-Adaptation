"""
TG-Tinker audit logging with hash chaining.

Provides tamper-evident append-only audit log.

Integration Points:
- Hash chaining for tamper detection
- Optional PQC signatures (Ed25519 + Dilithium3) for non-repudiation
- Compatible with tensorguard.identity.audit for enterprise audit trails

IMPORTANT: This module uses the canonical serialization from tensorguard.evidence.canonical
to ensure consistent hash computation across all TenSafe components.
"""

import hashlib
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tensorguard.evidence.canonical import canonical_json

from .models import TinkerAuditLog, generate_audit_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Genesis hash for the first log entry
GENESIS_HASH = "sha256:0000000000000000000000000000000000000000000000000000000000000000"


class AuditLogger:
    """
    Tamper-evident audit logger with hash chaining.

    Each log entry includes the hash of the previous entry, creating
    an append-only chain that detects tampering.
    """

    def __init__(self):
        """Initialize audit logger."""
        self._lock = threading.RLock()
        self._logs: List[TinkerAuditLog] = []
        self._sequence = 0
        self._prev_hash = GENESIS_HASH

    def log_operation(
        self,
        tenant_id: str,
        training_client_id: str,
        operation: str,
        request_hash: str,
        request_size_bytes: int,
        artifact_ids_produced: Optional[List[str]] = None,
        artifact_ids_consumed: Optional[List[str]] = None,
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        dp_metrics: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> TinkerAuditLog:
        """
        Log a primitive operation.

        Args:
            tenant_id: Tenant ID
            training_client_id: Training client ID
            operation: Operation type
            request_hash: Hash of the request payload
            request_size_bytes: Size of request in bytes
            artifact_ids_produced: IDs of artifacts created
            artifact_ids_consumed: IDs of artifacts used
            success: Whether operation succeeded
            error_code: Error code if failed
            error_message: Error message if failed
            dp_metrics: DP metrics if applicable
            started_at: Operation start time
            completed_at: Operation completion time

        Returns:
            Created audit log entry
        """
        with self._lock:
            now = datetime.utcnow()
            started_at = started_at or now
            completed_at = completed_at or now

            # Calculate duration
            duration_ms = None
            if started_at and completed_at:
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            # Increment sequence
            self._sequence += 1

            # Create entry
            entry_id = generate_audit_id()
            artifact_ids_produced = artifact_ids_produced or []
            artifact_ids_consumed = artifact_ids_consumed or []

            # Compute record hash with ALL security-relevant fields
            # This ensures the audit chain includes artifact lineage, request size, and DP metrics
            record_hash = self._compute_hash(
                entry_id=entry_id,
                tenant_id=tenant_id,
                training_client_id=training_client_id,
                operation=operation,
                request_hash=request_hash,
                artifact_ids_produced=artifact_ids_produced,
                started_at=started_at,
                completed_at=completed_at,
                success=success,
                prev_hash=self._prev_hash,
                artifact_ids_consumed=artifact_ids_consumed,
                request_size_bytes=request_size_bytes,
                dp_metrics=dp_metrics,
            )

            # Create log entry
            entry = TinkerAuditLog(
                id=entry_id,
                tenant_id=tenant_id,
                training_client_id=training_client_id,
                operation=operation,
                request_hash=request_hash,
                request_size_bytes=request_size_bytes,
                artifact_ids_produced=artifact_ids_produced,
                artifact_ids_consumed=artifact_ids_consumed,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                success=success,
                error_code=error_code,
                error_message=error_message,
                prev_hash=self._prev_hash,
                record_hash=record_hash,
                dp_metrics_json=dp_metrics,
                sequence=self._sequence,
            )

            # Update chain
            self._prev_hash = record_hash
            self._logs.append(entry)

            logger.debug(
                f"Audit log: {operation} on {training_client_id} (seq={self._sequence}, hash={record_hash[:20]}...)"
            )

            return entry

    def get_logs(
        self,
        tenant_id: Optional[str] = None,
        training_client_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TinkerAuditLog]:
        """
        Retrieve audit logs with optional filtering.

        Args:
            tenant_id: Filter by tenant ID
            training_client_id: Filter by training client ID
            operation: Filter by operation type
            limit: Maximum entries to return
            offset: Number of entries to skip

        Returns:
            List of matching audit log entries
        """
        with self._lock:
            filtered = self._logs

            if tenant_id:
                filtered = [e for e in filtered if e.tenant_id == tenant_id]
            if training_client_id:
                filtered = [e for e in filtered if e.training_client_id == training_client_id]
            if operation:
                filtered = [e for e in filtered if e.operation == operation]

            # Apply pagination
            return filtered[offset : offset + limit]

    def verify_chain(
        self,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Verify the integrity of the hash chain.

        Args:
            tenant_id: Optional tenant to verify (None = all)

        Returns:
            True if chain is valid, False if tampered

        Raises:
            ValueError: If chain is corrupted
        """
        with self._lock:
            logs = self._logs
            if tenant_id:
                logs = [e for e in logs if e.tenant_id == tenant_id]

            if not logs:
                return True

            # Verify each entry
            prev_hash = GENESIS_HASH

            for entry in logs:
                # Check prev_hash link
                if entry.prev_hash != prev_hash:
                    logger.error(
                        f"Chain broken at seq={entry.sequence}: expected prev_hash={prev_hash}, got {entry.prev_hash}"
                    )
                    return False

                # Recompute and verify record hash with ALL security-relevant fields
                computed_hash = self._compute_hash(
                    entry_id=entry.id,
                    tenant_id=entry.tenant_id,
                    training_client_id=entry.training_client_id,
                    operation=entry.operation,
                    request_hash=entry.request_hash,
                    artifact_ids_produced=entry.artifact_ids_produced,
                    started_at=entry.started_at,
                    completed_at=entry.completed_at,
                    success=entry.success,
                    prev_hash=entry.prev_hash,
                    artifact_ids_consumed=entry.artifact_ids_consumed,
                    request_size_bytes=entry.request_size_bytes,
                    dp_metrics=entry.dp_metrics_json,
                )

                if computed_hash != entry.record_hash:
                    logger.error(
                        f"Hash mismatch at seq={entry.sequence}: expected {entry.record_hash}, computed {computed_hash}"
                    )
                    return False

                prev_hash = entry.record_hash

            return True

    def _compute_hash(
        self,
        entry_id: str,
        tenant_id: str,
        training_client_id: str,
        operation: str,
        request_hash: str,
        artifact_ids_produced: List[str],
        started_at: datetime,
        completed_at: Optional[datetime],
        success: bool,
        prev_hash: str,
        artifact_ids_consumed: Optional[List[str]] = None,
        request_size_bytes: Optional[int] = None,
        dp_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Compute the hash for an audit log entry.

        IMPORTANT: Uses canonical_json from tensorguard.evidence.canonical to ensure
        consistent hash computation across all TenSafe components. This fixes the
        canonical vs empirical gap where different serialization methods were used.

        All security-relevant fields are now included in the hash:
        - artifact_ids_consumed: Critical for tracking data lineage
        - request_size_bytes: For detecting payload tampering
        - dp_metrics: Privacy-sensitive metrics must be immutable
        """
        # Normalize artifact lists to ensure deterministic ordering
        artifact_ids_consumed = artifact_ids_consumed or []

        # Build the canonical data structure with ALL security-relevant fields
        data = {
            "entry_id": entry_id,
            "tenant_id": tenant_id,
            "training_client_id": training_client_id,
            "operation": operation,
            "request_hash": request_hash,
            "request_size_bytes": request_size_bytes,
            "artifact_ids_produced": sorted(artifact_ids_produced),
            "artifact_ids_consumed": sorted(artifact_ids_consumed),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat() if completed_at else None,
            "success": success,
            "prev_hash": prev_hash,
        }

        # Include DP metrics in hash if present - these are privacy-critical
        # and must not be modifiable post-hoc
        if dp_metrics:
            data["dp_metrics"] = dp_metrics

        # Use canonical JSON serialization for consistent cross-module hashing
        canonical_data = canonical_json(data)
        hash_bytes = hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()
        return f"sha256:{hash_bytes}"


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set the global audit logger instance."""
    global _audit_logger
    _audit_logger = logger
