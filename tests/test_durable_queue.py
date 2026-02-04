"""
Durable Queue Tests

Verifies that:
1. Jobs persist across "restarts" (new session)
2. Idempotency keys prevent duplicate submissions
3. Dead letter queue receives failed jobs after max retries
4. Visibility timeout allows job recovery
5. Retry with exponential backoff works correctly
"""

import sys
import time
from datetime import datetime, timedelta

import pytest


# Skip all tests if sqlmodel not available
sqlmodel = pytest.importorskip("sqlmodel")
from sqlmodel import Session, SQLModel, create_engine

# Import directly to avoid tensorguard's crypto imports
sys.path.insert(0, "src")


def import_durable_queue():
    """Import durable queue module avoiding crypto conflicts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "durable_queue",
        "src/tensorguard/platform/tg_tinker_api/durable_queue.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["durable_queue"] = module
    spec.loader.exec_module(module)
    return module


durable_queue_module = import_durable_queue()


class TestDurableQueue:
    """Test the durable job queue implementation."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a test database engine."""
        db_path = tmp_path / "test_queue.db"
        engine = create_engine(f"sqlite:///{db_path}")

        # Import models to register them
        DeadLetterEntry = durable_queue_module.DeadLetterEntry
        DurableJob = durable_queue_module.DurableJob
        IdempotencyRecord = durable_queue_module.IdempotencyRecord

        SQLModel.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session(self, engine):
        """Create a test database session."""
        with Session(engine) as session:
            yield session

    @pytest.fixture
    def queue(self, session):
        """Create a durable job queue."""
        DurableJobQueue = durable_queue_module.DurableJobQueue

        return DurableJobQueue(
            session=session,
            worker_id="test-worker",
            visibility_timeout_seconds=10,
            idempotency_ttl_hours=1,
            max_pending_per_tenant=5,
        )

    def test_submit_job(self, queue):
        """Test basic job submission."""
        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"batch": {"input_ids": [[1, 2, 3]]}},
        )

        assert job.id.startswith("job-")
        assert job.tenant_id == "tenant-1"
        assert job.training_client_id == "tc-1"
        assert job.operation == "forward_backward"
        assert job.status == "pending"
        assert job.attempt_count == 0
        assert "sha256:" in job.payload_hash

    def test_submit_with_idempotency_key(self, queue):
        """Test that idempotency keys prevent duplicates."""
        job1 = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"batch": {"input_ids": [[1, 2, 3]]}},
            idempotency_key="unique-key-123",
        )

        # Submit again with same idempotency key
        job2 = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"different": "payload"},  # Different payload
            idempotency_key="unique-key-123",
        )

        # Should return the same job
        assert job1.id == job2.id

    def test_idempotency_key_different_operation_fails(self, queue):
        """Test that idempotency keys reject different operations."""
        queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
            idempotency_key="key-1",
        )

        with pytest.raises(ValueError) as exc_info:
            queue.submit(
                tenant_id="tenant-1",
                training_client_id="tc-1",
                operation="optim_step",  # Different operation
                payload={},
                idempotency_key="key-1",
            )

        assert "was used for operation" in str(exc_info.value)

    def test_idempotency_key_different_tenant_fails(self, queue):
        """Test that idempotency keys reject different tenants."""
        queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
            idempotency_key="key-2",
        )

        with pytest.raises(ValueError) as exc_info:
            queue.submit(
                tenant_id="tenant-2",  # Different tenant
                training_client_id="tc-1",
                operation="forward_backward",
                payload={},
                idempotency_key="key-2",
            )

        assert "belongs to a different tenant" in str(exc_info.value)

    def test_max_pending_jobs_enforced(self, queue):
        """Test that max pending jobs per tenant is enforced."""
        # Submit max jobs
        for i in range(5):
            queue.submit(
                tenant_id="tenant-1",
                training_client_id="tc-1",
                operation="forward_backward",
                payload={"i": i},
            )

        # Next submission should fail
        with pytest.raises(RuntimeError) as exc_info:
            queue.submit(
                tenant_id="tenant-1",
                training_client_id="tc-1",
                operation="forward_backward",
                payload={},
            )

        assert "too many pending jobs" in str(exc_info.value)

        # But different tenant should work
        job = queue.submit(
            tenant_id="tenant-2",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
        )
        assert job is not None

    def test_claim_and_complete_job(self, queue):
        """Test claiming and completing a job."""
        submitted = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"data": "test"},
        )

        # Claim the job
        claimed = queue.claim_next()
        assert claimed is not None
        assert claimed.id == submitted.id
        assert claimed.status == "claimed"
        assert claimed.attempt_count == 1
        assert claimed.claimed_by == "test-worker"

        # Complete the job
        result = {"loss": 0.5, "grad_norm": 1.2}
        completed = queue.complete(claimed.id, result)

        assert completed.status == "completed"
        assert completed.result_json == result
        assert completed.completed_at is not None

    def test_fail_and_retry_job(self, queue):
        """Test that failed jobs are retried."""
        DurableJob = durable_queue_module.DurableJob

        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
            max_attempts=3,
        )

        # Claim and fail
        claimed = queue.claim_next()
        failed = queue.fail(claimed.id, "Test error")

        assert failed.status == "pending"  # Back to pending for retry
        assert failed.last_error == "Test error"
        assert failed.attempt_count == 1
        # Should have visibility delay
        assert failed.visible_after > datetime.utcnow()

    def test_dead_letter_queue_after_max_attempts(self, queue, session):
        """Test that jobs move to DLQ after max attempts."""
        DeadLetterEntry = durable_queue_module.DeadLetterEntry
        DurableJob = durable_queue_module.DurableJob

        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"test": "data"},
            max_attempts=2,
        )

        # Simulate max attempts
        for i in range(2):
            # Manually reset visibility for testing
            db_job = session.get(DurableJob, job.id)
            db_job.visible_after = datetime.utcnow()
            db_job.status = "pending"
            session.add(db_job)
            session.commit()

            claimed = queue.claim_next()
            assert claimed is not None
            queue.fail(claimed.id, f"Error {i + 1}")

        # Job should be dead
        final_job = queue.get_job(job.id)
        assert final_job.status == "dead"

        # Check DLQ
        dlq_entries = queue.get_dlq_entries("tenant-1")
        assert len(dlq_entries) == 1
        assert dlq_entries[0].original_job_id == job.id
        assert dlq_entries[0].attempt_count == 2
        assert "Error 2" in dlq_entries[0].failure_reason

    def test_dlq_acknowledge(self, queue, session):
        """Test acknowledging DLQ entries."""
        DurableJob = durable_queue_module.DurableJob

        # Create a job and push to DLQ
        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
            max_attempts=1,
        )

        claimed = queue.claim_next()
        queue.fail(claimed.id, "Fatal error")

        # Get and acknowledge
        entries = queue.get_dlq_entries("tenant-1")
        assert len(entries) == 1

        result = queue.acknowledge_dlq_entry(
            entries[0].id,
            "tenant-1",
            "admin@example.com",
        )
        assert result is True

        # Should not appear in unacknowledged list
        unacked = queue.get_dlq_entries("tenant-1", include_acknowledged=False)
        assert len(unacked) == 0

        # But should appear if we include acknowledged
        all_entries = queue.get_dlq_entries("tenant-1", include_acknowledged=True)
        assert len(all_entries) == 1
        assert all_entries[0].acknowledged is True

    def test_dlq_replay(self, queue, session):
        """Test replaying DLQ entries."""
        DurableJob = durable_queue_module.DurableJob

        # Create a job and push to DLQ
        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"important": "data"},
            max_attempts=1,
        )

        claimed = queue.claim_next()
        queue.fail(claimed.id, "Temporary error")

        # Replay from DLQ
        entries = queue.get_dlq_entries("tenant-1")
        new_job = queue.replay_dlq_entry(entries[0].id, "tenant-1")

        assert new_job is not None
        assert new_job.id != job.id  # New job ID
        assert new_job.payload_json == {"important": "data"}  # Same payload
        assert new_job.status == "pending"

        # Original entry should be acknowledged
        updated_entries = queue.get_dlq_entries("tenant-1", include_acknowledged=True)
        assert updated_entries[0].acknowledged is True

    def test_cancel_job(self, queue):
        """Test cancelling a pending job."""
        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
        )

        result = queue.cancel(job.id, "tenant-1")
        assert result is True

        cancelled = queue.get_job(job.id)
        assert cancelled.status == "cancelled"

    def test_cancel_wrong_tenant_fails(self, queue):
        """Test that cancelling with wrong tenant fails."""
        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
        )

        result = queue.cancel(job.id, "tenant-2")  # Wrong tenant
        assert result is False

    def test_jobs_persist_across_sessions(self, engine):
        """Test that jobs survive across database sessions (simulating restart)."""
        DurableJobQueue = durable_queue_module.DurableJobQueue

        # Session 1: Submit a job
        with Session(engine) as session1:
            queue1 = DurableJobQueue(session=session1, worker_id="worker-1")
            job = queue1.submit(
                tenant_id="tenant-1",
                training_client_id="tc-1",
                operation="forward_backward",
                payload={"persisted": True},
            )
            job_id = job.id

        # Session 2: Verify job exists and claim it
        with Session(engine) as session2:
            queue2 = DurableJobQueue(session=session2, worker_id="worker-2")
            retrieved = queue2.get_job(job_id)

            assert retrieved is not None
            assert retrieved.payload_json["persisted"] is True

            # Can claim it
            claimed = queue2.claim_next()
            assert claimed is not None
            assert claimed.id == job_id

    def test_recover_stale_jobs(self, queue, session):
        """Test recovering jobs that were claimed but never completed."""
        DurableJob = durable_queue_module.DurableJob

        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
        )

        # Claim the job
        claimed = queue.claim_next()
        assert claimed is not None

        # Manually make it stale
        db_job = session.get(DurableJob, job.id)
        db_job.claimed_at = datetime.utcnow() - timedelta(hours=1)
        session.add(db_job)
        session.commit()

        # Recover stale jobs (threshold = 60 seconds)
        recovered = queue.recover_stale_jobs(stale_threshold_seconds=60)
        assert recovered == 1

        # Job should be claimable again
        session.refresh(db_job)
        assert db_job.status == "pending"
        assert db_job.claimed_by is None

    def test_cleanup_expired_idempotency(self, queue, session):
        """Test cleanup of expired idempotency records."""
        IdempotencyRecord = durable_queue_module.IdempotencyRecord

        # Submit with idempotency key
        job = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
            idempotency_key="key-expire-test",
        )

        # Manually expire the record
        record = session.get(IdempotencyRecord, "key-expire-test")
        record.expires_at = datetime.utcnow() - timedelta(hours=1)
        session.add(record)
        session.commit()

        # Run cleanup
        cleaned = queue.cleanup_expired_idempotency_records()
        assert cleaned == 1

        # Should be able to use the key again
        job2 = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={},
            idempotency_key="key-expire-test",
        )
        assert job2.id != job.id

    def test_priority_ordering(self, queue, session):
        """Test that higher priority jobs are claimed first."""
        DurableJob = durable_queue_module.DurableJob

        # Submit jobs with different priorities
        low = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"priority": "low"},
            priority=0,
        )

        high = queue.submit(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            payload={"priority": "high"},
            priority=10,
        )

        # High priority should be claimed first
        claimed1 = queue.claim_next()
        assert claimed1.id == high.id

        # Complete it
        queue.complete(claimed1.id, {})

        # Low priority should be claimed next
        claimed2 = queue.claim_next()
        assert claimed2.id == low.id


class TestDurableQueueFactory:
    """Test the durable queue factory."""

    def test_create_durable_queue(self, tmp_path):
        """Test creating a durable queue with factory."""
        create_durable_queue = durable_queue_module.create_durable_queue
        DeadLetterEntry = durable_queue_module.DeadLetterEntry
        DurableJob = durable_queue_module.DurableJob
        IdempotencyRecord = durable_queue_module.IdempotencyRecord

        db_path = tmp_path / "test_factory.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            queue = create_durable_queue(session, worker_id="test-worker")

            assert queue is not None
            assert queue.worker_id == "test-worker"

            # Should be functional
            job = queue.submit(
                tenant_id="tenant-1",
                training_client_id="tc-1",
                operation="test",
                payload={},
            )
            assert job is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
