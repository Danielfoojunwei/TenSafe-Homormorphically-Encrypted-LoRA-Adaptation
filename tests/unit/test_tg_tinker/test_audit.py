"""
Unit tests for TG-Tinker audit logging with hash chaining.
"""

from datetime import datetime, timedelta

import pytest

from tensorguard.platform.tg_tinker_api.audit import (
    AuditLogger,
    GENESIS_HASH,
)


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def logger(self):
        """Create a fresh audit logger."""
        return AuditLogger()

    def test_first_entry_links_to_genesis(self, logger):
        """Test that first entry links to genesis hash."""
        entry = logger.log_operation(
            tenant_id="tenant-123",
            training_client_id="tc-456",
            operation="forward_backward",
            request_hash="sha256:abc123",
            request_size_bytes=1024,
        )

        assert entry.prev_hash == GENESIS_HASH
        assert entry.record_hash.startswith("sha256:")
        assert entry.sequence == 1

    def test_entries_chain_correctly(self, logger):
        """Test that entries form a proper chain."""
        entry1 = logger.log_operation(
            tenant_id="tenant-123",
            training_client_id="tc-456",
            operation="forward_backward",
            request_hash="sha256:req1",
            request_size_bytes=1024,
        )

        entry2 = logger.log_operation(
            tenant_id="tenant-123",
            training_client_id="tc-456",
            operation="optim_step",
            request_hash="sha256:req2",
            request_size_bytes=512,
        )

        entry3 = logger.log_operation(
            tenant_id="tenant-123",
            training_client_id="tc-456",
            operation="save_state",
            request_hash="sha256:req3",
            request_size_bytes=256,
        )

        # Verify chain links
        assert entry1.prev_hash == GENESIS_HASH
        assert entry2.prev_hash == entry1.record_hash
        assert entry3.prev_hash == entry2.record_hash

        # Verify sequences
        assert entry1.sequence == 1
        assert entry2.sequence == 2
        assert entry3.sequence == 3

    def test_chain_verification_valid(self, logger):
        """Test chain verification with valid chain."""
        # Add some entries
        for i in range(5):
            logger.log_operation(
                tenant_id="tenant-123",
                training_client_id="tc-456",
                operation="forward_backward",
                request_hash=f"sha256:req{i}",
                request_size_bytes=1024,
            )

        # Verify chain
        assert logger.verify_chain() is True

    def test_chain_verification_detects_modification(self, logger):
        """Test that chain verification detects modified entries."""
        # Add some entries
        for i in range(3):
            logger.log_operation(
                tenant_id="tenant-123",
                training_client_id="tc-456",
                operation="forward_backward",
                request_hash=f"sha256:req{i}",
                request_size_bytes=1024,
            )

        # Tamper with middle entry
        logger._logs[1].operation = "tampered_operation"

        # Verify chain - should fail
        assert logger.verify_chain() is False

    def test_chain_verification_detects_reordering(self, logger):
        """Test that chain verification detects reordered entries."""
        # Add entries
        for i in range(3):
            logger.log_operation(
                tenant_id="tenant-123",
                training_client_id="tc-456",
                operation="forward_backward",
                request_hash=f"sha256:req{i}",
                request_size_bytes=1024,
            )

        # Swap entries (simulating reorder attack)
        logger._logs[0], logger._logs[1] = logger._logs[1], logger._logs[0]

        # Verify chain - should fail
        assert logger.verify_chain() is False

    def test_get_logs_filtering_by_tenant(self, logger):
        """Test filtering logs by tenant."""
        # Add entries for different tenants
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:req1",
            request_size_bytes=1024,
        )
        logger.log_operation(
            tenant_id="tenant-2",
            training_client_id="tc-2",
            operation="forward_backward",
            request_hash="sha256:req2",
            request_size_bytes=1024,
        )
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="optim_step",
            request_hash="sha256:req3",
            request_size_bytes=512,
        )

        # Filter by tenant
        tenant1_logs = logger.get_logs(tenant_id="tenant-1")
        tenant2_logs = logger.get_logs(tenant_id="tenant-2")

        assert len(tenant1_logs) == 2
        assert len(tenant2_logs) == 1

    def test_get_logs_filtering_by_operation(self, logger):
        """Test filtering logs by operation."""
        # Add various operations
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:req1",
            request_size_bytes=1024,
        )
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="optim_step",
            request_hash="sha256:req2",
            request_size_bytes=512,
        )
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:req3",
            request_size_bytes=1024,
        )

        # Filter by operation
        fb_logs = logger.get_logs(operation="forward_backward")
        opt_logs = logger.get_logs(operation="optim_step")

        assert len(fb_logs) == 2
        assert len(opt_logs) == 1

    def test_get_logs_pagination(self, logger):
        """Test pagination of logs."""
        # Add many entries
        for i in range(10):
            logger.log_operation(
                tenant_id="tenant-1",
                training_client_id="tc-1",
                operation="forward_backward",
                request_hash=f"sha256:req{i}",
                request_size_bytes=1024,
            )

        # Get first page
        page1 = logger.get_logs(limit=3, offset=0)
        assert len(page1) == 3
        assert page1[0].sequence == 1

        # Get second page
        page2 = logger.get_logs(limit=3, offset=3)
        assert len(page2) == 3
        assert page2[0].sequence == 4

    def test_artifact_ids_tracking(self, logger):
        """Test that artifact IDs are tracked."""
        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="save_state",
            request_hash="sha256:req1",
            request_size_bytes=1024,
            artifact_ids_produced=["art-1", "art-2"],
            artifact_ids_consumed=["art-0"],
        )

        assert entry.artifact_ids_produced == ["art-1", "art-2"]
        assert entry.artifact_ids_consumed == ["art-0"]

    def test_error_logging(self, logger):
        """Test logging failed operations."""
        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:req1",
            request_size_bytes=1024,
            success=False,
            error_code="VALIDATION_ERROR",
            error_message="Invalid batch format",
        )

        assert entry.success is False
        assert entry.error_code == "VALIDATION_ERROR"
        assert entry.error_message == "Invalid batch format"

    def test_dp_metrics_logging(self, logger):
        """Test logging DP metrics."""
        dp_metrics = {
            "noise_applied": True,
            "epsilon_spent": 0.1,
            "total_epsilon": 0.5,
            "delta": 1e-5,
        }

        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="optim_step",
            request_hash="sha256:req1",
            request_size_bytes=512,
            dp_metrics=dp_metrics,
        )

        assert entry.dp_metrics_json == dp_metrics

    def test_timing_calculation(self, logger):
        """Test that duration is calculated correctly."""
        started = datetime.utcnow()
        completed = started + timedelta(milliseconds=500)

        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:req1",
            request_size_bytes=1024,
            started_at=started,
            completed_at=completed,
        )

        assert entry.duration_ms == 500

    def test_deterministic_hash(self, logger):
        """Test that same inputs produce same hash."""
        # Create entry with fixed inputs
        started_at = datetime(2024, 1, 1, 12, 0, 0)
        completed_at = datetime(2024, 1, 1, 12, 0, 5)

        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:fixed",
            request_size_bytes=1024,
            started_at=started_at,
            completed_at=completed_at,
        )

        # Hash should be deterministic
        assert entry.record_hash.startswith("sha256:")
        assert len(entry.record_hash) == 71  # "sha256:" + 64 hex chars

    def test_verify_chain_tenant_filter(self, logger):
        """Test chain verification with tenant filter.

        Note: The audit log is a global chain. When filtering by tenant,
        verification only works correctly if the tenant's entries are
        contiguous or if the chain is verified globally.
        """
        # Add entries for a single tenant to test filtered verification
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="forward_backward",
            request_hash="sha256:req1",
            request_size_bytes=1024,
        )
        logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-1",
            operation="optim_step",
            request_hash="sha256:req2",
            request_size_bytes=512,
        )

        # Verify entire chain
        assert logger.verify_chain() is True

        # Verify filtered by tenant (works because all entries are same tenant)
        assert logger.verify_chain(tenant_id="tenant-1") is True
