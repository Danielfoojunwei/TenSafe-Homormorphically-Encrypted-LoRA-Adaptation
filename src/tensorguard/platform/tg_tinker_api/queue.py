"""
TG-Tinker job queue.

Provides an in-memory queue (with pluggable backends) for async job execution.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job in the queue."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A job in the queue."""

    job_id: str
    tenant_id: str
    training_client_id: str
    operation: str
    payload: Dict[str, Any]
    payload_hash: str
    status: JobStatus = JobStatus.PENDING
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __lt__(self, other: "Job") -> bool:
        """Compare jobs by priority (lower is higher priority)."""
        return (self.priority, self.created_at) < (other.priority, other.created_at)


class JobQueueBackend(ABC):
    """Abstract base class for job queue backends."""

    @abstractmethod
    def enqueue(self, job: Job) -> None:
        """Add a job to the queue."""
        pass

    @abstractmethod
    def dequeue(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Remove and return the next job from the queue."""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        pass

    @abstractmethod
    def update_job(self, job: Job) -> None:
        """Update a job's status."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        pass

    @abstractmethod
    def get_pending_count(self, tenant_id: Optional[str] = None) -> int:
        """Get count of pending jobs."""
        pass


class InMemoryJobQueue(JobQueueBackend):
    """In-memory job queue with priority support."""

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize in-memory queue.

        Args:
            max_queue_size: Maximum number of pending jobs
        """
        self._queue: PriorityQueue[Job] = PriorityQueue(maxsize=max_queue_size)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def enqueue(self, job: Job) -> None:
        """Add a job to the queue."""
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"Job {job.job_id} already exists")

            try:
                self._queue.put_nowait(job)
            except Exception:
                raise RuntimeError("Queue is full")

            self._jobs[job.job_id] = job
            self._condition.notify()

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Remove and return the next job."""
        try:
            job = self._queue.get(block=True, timeout=timeout)
            return job
        except Empty:
            return None

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job: Job) -> None:
        """Update a job's status."""
        with self._lock:
            self._jobs[job.job_id] = job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status != JobStatus.PENDING:
                return False

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            return True

    def get_pending_count(self, tenant_id: Optional[str] = None) -> int:
        """Get count of pending jobs."""
        with self._lock:
            if tenant_id is None:
                return sum(
                    1 for job in self._jobs.values() if job.status == JobStatus.PENDING
                )
            return sum(
                1
                for job in self._jobs.values()
                if job.status == JobStatus.PENDING and job.tenant_id == tenant_id
            )


class JobQueue:
    """
    High-level job queue manager.

    Handles job creation, status tracking, and integrates with the backend.
    """

    def __init__(
        self,
        backend: Optional[JobQueueBackend] = None,
        max_pending_per_tenant: int = 100,
    ):
        """
        Initialize job queue.

        Args:
            backend: Queue backend (defaults to in-memory)
            max_pending_per_tenant: Max pending jobs per tenant
        """
        self.backend = backend or InMemoryJobQueue()
        self.max_pending_per_tenant = max_pending_per_tenant

    def submit(
        self,
        job_id: str,
        tenant_id: str,
        training_client_id: str,
        operation: str,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> Job:
        """
        Submit a job to the queue.

        Args:
            job_id: Unique job ID
            tenant_id: Tenant ID
            training_client_id: Training client ID
            operation: Operation type
            payload: Job payload
            priority: Job priority (lower is higher)

        Returns:
            Created Job

        Raises:
            RuntimeError: If tenant has too many pending jobs
        """
        # Check pending count
        pending = self.backend.get_pending_count(tenant_id)
        if pending >= self.max_pending_per_tenant:
            raise RuntimeError(
                f"Tenant {tenant_id} has too many pending jobs ({pending})"
            )

        # Compute payload hash (for audit logging, privacy-preserving)
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = f"sha256:{hashlib.sha256(payload_json.encode()).hexdigest()}"

        # Create job
        job = Job(
            job_id=job_id,
            tenant_id=tenant_id,
            training_client_id=training_client_id,
            operation=operation,
            payload=payload,
            payload_hash=payload_hash,
            priority=priority,
        )

        # Enqueue
        self.backend.enqueue(job)

        return job

    def get_next(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Get the next job to process."""
        job = self.backend.dequeue(timeout)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            self.backend.update_job(job)
        return job

    def complete(
        self,
        job_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Mark a job as completed."""
        job = self.backend.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.result = result
        self.backend.update_job(job)

    def fail(
        self,
        job_id: str,
        error: str,
    ) -> None:
        """Mark a job as failed."""
        job = self.backend.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        job.status = JobStatus.FAILED
        job.completed_at = datetime.utcnow()
        job.error = error
        self.backend.update_job(job)

    def get_status(self, job_id: str) -> Optional[Job]:
        """Get job status."""
        return self.backend.get_job(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        return self.backend.cancel_job(job_id)


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


def set_job_queue(queue: JobQueue) -> None:
    """Set the global job queue instance."""
    global _job_queue
    _job_queue = queue
