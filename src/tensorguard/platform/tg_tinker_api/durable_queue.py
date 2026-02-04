"""
TG-Tinker Durable Job Queue.

Provides a database-backed job queue with:
- Durable storage that survives API restarts
- Idempotency keys for deduplication
- Dead letter queue for failed jobs
- At-least-once processing semantics
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlmodel import JSON, Column, Field, Session, SQLModel, select

logger = logging.getLogger(__name__)


# ==============================================================================
# Database Models
# ==============================================================================


def generate_job_id() -> str:
    """Generate a job ID."""
    return f"job-{uuid.uuid4()}"


def generate_dlq_id() -> str:
    """Generate a dead letter queue entry ID."""
    return f"dlq-{uuid.uuid4()}"


class DurableJobStatus(str, Enum):
    """Status of a durable job."""

    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD = "dead"  # Moved to DLQ


class DurableJob(SQLModel, table=True):
    """A durable job persisted to the database."""

    __tablename__ = "tinker_durable_jobs"

    id: str = Field(default_factory=generate_job_id, primary_key=True)
    tenant_id: str = Field(index=True)
    training_client_id: str = Field(index=True)
    operation: str
    idempotency_key: Optional[str] = Field(default=None, index=True)

    # Payload stored as JSON
    payload_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    payload_hash: str

    # Status and priority
    status: str = Field(default=DurableJobStatus.PENDING.value, index=True)
    priority: int = Field(default=0, index=True)

    # Retry tracking
    attempt_count: int = Field(default=0)
    max_attempts: int = Field(default=3)
    last_error: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    claimed_at: Optional[datetime] = None
    claimed_by: Optional[str] = None  # Worker ID
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Visibility timeout (for at-least-once processing)
    visible_after: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Result (for completed jobs)
    result_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class IdempotencyRecord(SQLModel, table=True):
    """Idempotency record for deduplication."""

    __tablename__ = "tinker_idempotency_records"

    idempotency_key: str = Field(primary_key=True)
    tenant_id: str = Field(index=True)
    job_id: str = Field(index=True)
    operation: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # Records expire after a period


class DeadLetterEntry(SQLModel, table=True):
    """Dead letter queue entry for failed jobs."""

    __tablename__ = "tinker_dead_letter_queue"

    id: str = Field(default_factory=generate_dlq_id, primary_key=True)
    original_job_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    training_client_id: str
    operation: str

    # Original payload
    payload_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    payload_hash: str

    # Failure info
    attempt_count: int
    failure_reason: str
    failure_details: Optional[str] = None

    # Timestamps
    original_created_at: datetime
    moved_to_dlq_at: datetime = Field(default_factory=datetime.utcnow)

    # Status for DLQ processing
    acknowledged: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


# ==============================================================================
# Durable Queue Implementation
# ==============================================================================


class DurableJobQueue:
    """
    Database-backed job queue with durability guarantees.

    Features:
    - Jobs persist across API restarts
    - Idempotency keys prevent duplicate processing
    - Visibility timeout for at-least-once semantics
    - Dead letter queue for jobs that exceed max retries
    """

    def __init__(
        self,
        session: Session,
        worker_id: Optional[str] = None,
        visibility_timeout_seconds: int = 300,
        idempotency_ttl_hours: int = 24,
        max_pending_per_tenant: int = 100,
    ):
        """
        Initialize durable job queue.

        Args:
            session: Database session
            worker_id: Unique identifier for this worker (for claim tracking)
            visibility_timeout_seconds: How long before uncompleted jobs become visible again
            idempotency_ttl_hours: How long to keep idempotency records
            max_pending_per_tenant: Maximum pending jobs per tenant
        """
        self.session = session
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.visibility_timeout = timedelta(seconds=visibility_timeout_seconds)
        self.idempotency_ttl = timedelta(hours=idempotency_ttl_hours)
        self.max_pending_per_tenant = max_pending_per_tenant

    def submit(
        self,
        tenant_id: str,
        training_client_id: str,
        operation: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> DurableJob:
        """
        Submit a job to the queue.

        Args:
            tenant_id: Tenant ID
            training_client_id: Training client ID
            operation: Operation type
            payload: Job payload
            idempotency_key: Optional key for deduplication
            priority: Job priority (higher = more important)
            max_attempts: Maximum retry attempts

        Returns:
            Created or existing job (if idempotency key matches)

        Raises:
            ValueError: If idempotency key already used for different operation
            RuntimeError: If tenant has too many pending jobs
        """
        # Check idempotency
        if idempotency_key:
            existing = self._check_idempotency(tenant_id, idempotency_key, operation)
            if existing:
                logger.info(f"Idempotency hit for key {idempotency_key}, returning existing job {existing.id}")
                return existing

        # Check pending count
        pending_count = self._get_pending_count(tenant_id)
        if pending_count >= self.max_pending_per_tenant:
            raise RuntimeError(
                f"Tenant {tenant_id} has too many pending jobs ({pending_count}). "
                f"Maximum allowed: {self.max_pending_per_tenant}"
            )

        # Compute payload hash
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = f"sha256:{hashlib.sha256(payload_json.encode()).hexdigest()}"

        # Create job
        job = DurableJob(
            tenant_id=tenant_id,
            training_client_id=training_client_id,
            operation=operation,
            idempotency_key=idempotency_key,
            payload_json=payload,
            payload_hash=payload_hash,
            priority=priority,
            max_attempts=max_attempts,
        )

        self.session.add(job)

        # Create idempotency record
        if idempotency_key:
            idem_record = IdempotencyRecord(
                idempotency_key=idempotency_key,
                tenant_id=tenant_id,
                job_id=job.id,
                operation=operation,
                expires_at=datetime.utcnow() + self.idempotency_ttl,
            )
            self.session.add(idem_record)

        self.session.commit()
        self.session.refresh(job)

        logger.info(f"Submitted job {job.id} for tenant {tenant_id}, operation={operation}")
        return job

    def claim_next(self) -> Optional[DurableJob]:
        """
        Claim the next available job for processing.

        Uses visibility timeout to prevent other workers from claiming
        the same job. Job must be completed or failed within the
        visibility timeout, or it becomes available again.

        Returns:
            Claimed job, or None if queue is empty
        """
        now = datetime.utcnow()

        # Find next visible pending job (ordered by priority desc, then created_at asc)
        statement = (
            select(DurableJob)
            .where(
                DurableJob.status.in_([DurableJobStatus.PENDING.value, DurableJobStatus.CLAIMED.value]),
                DurableJob.visible_after <= now,
            )
            .order_by(DurableJob.priority.desc(), DurableJob.created_at)
            .limit(1)
        )

        result = self.session.exec(statement)
        job = result.first()

        if job is None:
            return None

        # Claim the job
        job.status = DurableJobStatus.CLAIMED.value
        job.claimed_at = now
        job.claimed_by = self.worker_id
        job.visible_after = now + self.visibility_timeout
        job.attempt_count += 1

        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        logger.info(f"Claimed job {job.id}, attempt {job.attempt_count}/{job.max_attempts}")
        return job

    def start(self, job_id: str) -> Optional[DurableJob]:
        """
        Mark a job as running.

        Args:
            job_id: Job ID

        Returns:
            Updated job, or None if not found
        """
        job = self.session.get(DurableJob, job_id)
        if job is None:
            return None

        job.status = DurableJobStatus.RUNNING.value
        job.started_at = datetime.utcnow()

        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        return job

    def complete(self, job_id: str, result: Dict[str, Any]) -> Optional[DurableJob]:
        """
        Mark a job as completed.

        Args:
            job_id: Job ID
            result: Job result

        Returns:
            Updated job, or None if not found
        """
        job = self.session.get(DurableJob, job_id)
        if job is None:
            return None

        job.status = DurableJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        job.result_json = result

        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        logger.info(f"Completed job {job_id}")
        return job

    def fail(self, job_id: str, error: str, details: Optional[str] = None) -> Optional[DurableJob]:
        """
        Mark a job as failed.

        If the job has exceeded max attempts, it will be moved to the
        dead letter queue. Otherwise, it will be made visible again
        for retry.

        Args:
            job_id: Job ID
            error: Error message
            details: Optional error details

        Returns:
            Updated job, or None if not found
        """
        job = self.session.get(DurableJob, job_id)
        if job is None:
            return None

        job.last_error = error

        if job.attempt_count >= job.max_attempts:
            # Move to DLQ
            self._move_to_dlq(job, error, details)
            job.status = DurableJobStatus.DEAD.value
            job.completed_at = datetime.utcnow()
            logger.warning(f"Job {job_id} moved to DLQ after {job.attempt_count} attempts")
        else:
            # Mark as pending for retry
            job.status = DurableJobStatus.PENDING.value
            job.visible_after = datetime.utcnow() + self._get_retry_delay(job.attempt_count)
            logger.info(f"Job {job_id} will retry (attempt {job.attempt_count}/{job.max_attempts})")

        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        return job

    def cancel(self, job_id: str, tenant_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: Job ID
            tenant_id: Tenant ID (for authorization)

        Returns:
            True if cancelled, False if not found or not cancellable
        """
        job = self.session.get(DurableJob, job_id)
        if job is None or job.tenant_id != tenant_id:
            return False

        if job.status not in (DurableJobStatus.PENDING.value, DurableJobStatus.CLAIMED.value):
            return False

        job.status = DurableJobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()

        self.session.add(job)
        self.session.commit()

        logger.info(f"Cancelled job {job_id}")
        return True

    def get_job(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[DurableJob]:
        """
        Get a job by ID.

        Args:
            job_id: Job ID
            tenant_id: Optional tenant ID for authorization check

        Returns:
            Job if found (and tenant matches if specified), None otherwise
        """
        job = self.session.get(DurableJob, job_id)
        if job is None:
            return None
        if tenant_id and job.tenant_id != tenant_id:
            return None
        return job

    def get_pending_count(self, tenant_id: Optional[str] = None) -> int:
        """Get count of pending jobs."""
        return self._get_pending_count(tenant_id)

    # ==========================================================================
    # Dead Letter Queue Operations
    # ==========================================================================

    def get_dlq_entries(
        self,
        tenant_id: str,
        include_acknowledged: bool = False,
        limit: int = 100,
    ) -> List[DeadLetterEntry]:
        """
        Get dead letter queue entries for a tenant.

        Args:
            tenant_id: Tenant ID
            include_acknowledged: Whether to include acknowledged entries
            limit: Maximum entries to return

        Returns:
            List of DLQ entries
        """
        statement = select(DeadLetterEntry).where(DeadLetterEntry.tenant_id == tenant_id)

        if not include_acknowledged:
            statement = statement.where(DeadLetterEntry.acknowledged == False)  # noqa: E712

        statement = statement.order_by(DeadLetterEntry.moved_to_dlq_at.desc()).limit(limit)

        return list(self.session.exec(statement))

    def acknowledge_dlq_entry(self, dlq_id: str, tenant_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge a DLQ entry (mark as handled).

        Args:
            dlq_id: DLQ entry ID
            tenant_id: Tenant ID
            acknowledged_by: Who acknowledged it

        Returns:
            True if acknowledged, False if not found
        """
        entry = self.session.get(DeadLetterEntry, dlq_id)
        if entry is None or entry.tenant_id != tenant_id:
            return False

        entry.acknowledged = True
        entry.acknowledged_at = datetime.utcnow()
        entry.acknowledged_by = acknowledged_by

        self.session.add(entry)
        self.session.commit()

        logger.info(f"Acknowledged DLQ entry {dlq_id}")
        return True

    def replay_dlq_entry(self, dlq_id: str, tenant_id: str) -> Optional[DurableJob]:
        """
        Replay a dead letter entry by creating a new job.

        Args:
            dlq_id: DLQ entry ID
            tenant_id: Tenant ID

        Returns:
            New job if replayed, None if not found
        """
        entry = self.session.get(DeadLetterEntry, dlq_id)
        if entry is None or entry.tenant_id != tenant_id:
            return None

        # Create new job from DLQ entry
        new_job = self.submit(
            tenant_id=entry.tenant_id,
            training_client_id=entry.training_client_id,
            operation=entry.operation,
            payload=entry.payload_json,
            # No idempotency key for replays
            priority=0,
            max_attempts=3,
        )

        # Mark DLQ entry as acknowledged
        entry.acknowledged = True
        entry.acknowledged_at = datetime.utcnow()
        entry.acknowledged_by = "replay"
        self.session.add(entry)
        self.session.commit()

        logger.info(f"Replayed DLQ entry {dlq_id} as job {new_job.id}")
        return new_job

    # ==========================================================================
    # Maintenance Operations
    # ==========================================================================

    def cleanup_expired_idempotency_records(self) -> int:
        """
        Remove expired idempotency records.

        Returns:
            Number of records deleted
        """
        now = datetime.utcnow()
        statement = select(IdempotencyRecord).where(IdempotencyRecord.expires_at < now)
        expired = list(self.session.exec(statement))

        for record in expired:
            self.session.delete(record)

        self.session.commit()

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired idempotency records")

        return len(expired)

    def recover_stale_jobs(self, stale_threshold_seconds: int = 600) -> int:
        """
        Recover jobs that were claimed but never completed.

        Jobs that have been claimed for longer than the stale threshold
        are made visible again for processing.

        Args:
            stale_threshold_seconds: How long before a claimed job is considered stale

        Returns:
            Number of jobs recovered
        """
        now = datetime.utcnow()
        stale_threshold = now - timedelta(seconds=stale_threshold_seconds)

        statement = select(DurableJob).where(
            DurableJob.status.in_([DurableJobStatus.CLAIMED.value, DurableJobStatus.RUNNING.value]),
            DurableJob.claimed_at < stale_threshold,
        )

        stale_jobs = list(self.session.exec(statement))

        for job in stale_jobs:
            job.status = DurableJobStatus.PENDING.value
            job.visible_after = now
            job.claimed_at = None
            job.claimed_by = None
            self.session.add(job)

        self.session.commit()

        if stale_jobs:
            logger.warning(f"Recovered {len(stale_jobs)} stale jobs")

        return len(stale_jobs)

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _check_idempotency(
        self,
        tenant_id: str,
        idempotency_key: str,
        operation: str,
    ) -> Optional[DurableJob]:
        """Check if idempotency key has been used."""
        record = self.session.get(IdempotencyRecord, idempotency_key)

        if record is None:
            return None

        # Check expiration
        if record.expires_at < datetime.utcnow():
            # Expired, delete and allow new job
            self.session.delete(record)
            self.session.commit()
            return None

        # Verify tenant and operation match
        if record.tenant_id != tenant_id:
            raise ValueError(
                f"Idempotency key {idempotency_key} belongs to a different tenant"
            )

        if record.operation != operation:
            raise ValueError(
                f"Idempotency key {idempotency_key} was used for operation "
                f"'{record.operation}', not '{operation}'"
            )

        # Return existing job
        return self.session.get(DurableJob, record.job_id)

    def _get_pending_count(self, tenant_id: Optional[str] = None) -> int:
        """Get count of pending jobs."""
        statement = select(DurableJob).where(
            DurableJob.status.in_([
                DurableJobStatus.PENDING.value,
                DurableJobStatus.CLAIMED.value,
                DurableJobStatus.RUNNING.value,
            ])
        )

        if tenant_id:
            statement = statement.where(DurableJob.tenant_id == tenant_id)

        result = list(self.session.exec(statement))
        return len(result)

    def _move_to_dlq(self, job: DurableJob, error: str, details: Optional[str] = None) -> DeadLetterEntry:
        """Move a job to the dead letter queue."""
        entry = DeadLetterEntry(
            original_job_id=job.id,
            tenant_id=job.tenant_id,
            training_client_id=job.training_client_id,
            operation=job.operation,
            payload_json=job.payload_json,
            payload_hash=job.payload_hash,
            attempt_count=job.attempt_count,
            failure_reason=error,
            failure_details=details,
            original_created_at=job.created_at,
        )

        self.session.add(entry)
        return entry

    def _get_retry_delay(self, attempt_count: int) -> timedelta:
        """Get exponential backoff delay for retries."""
        # Exponential backoff: 5s, 10s, 20s, 40s, ...
        base_delay = 5
        delay_seconds = base_delay * (2 ** (attempt_count - 1))
        # Cap at 5 minutes
        delay_seconds = min(delay_seconds, 300)
        return timedelta(seconds=delay_seconds)


# ==============================================================================
# Factory and Global Access
# ==============================================================================


def create_durable_queue(session: Session, worker_id: Optional[str] = None) -> DurableJobQueue:
    """
    Create a durable job queue.

    Args:
        session: Database session
        worker_id: Optional worker ID

    Returns:
        Configured DurableJobQueue
    """
    visibility_timeout = int(os.getenv("TG_JOB_VISIBILITY_TIMEOUT", "300"))
    idempotency_ttl = int(os.getenv("TG_IDEMPOTENCY_TTL_HOURS", "24"))
    max_pending = int(os.getenv("TG_MAX_PENDING_JOBS", "100"))

    return DurableJobQueue(
        session=session,
        worker_id=worker_id,
        visibility_timeout_seconds=visibility_timeout,
        idempotency_ttl_hours=idempotency_ttl,
        max_pending_per_tenant=max_pending,
    )
