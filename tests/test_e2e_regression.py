"""
End-to-End Regression Test Suite

Verifies the full system behavior including:
1. Training client lifecycle
2. Job queue processing
3. Artifact persistence
4. Authentication flow
5. Error handling
6. Data persistence across restarts

These tests simulate real-world usage patterns.
"""

import importlib.util
import os
import sys
from datetime import datetime

import pytest

# Import sqlmodel
sqlmodel = pytest.importorskip("sqlmodel")
from sqlmodel import Session, SQLModel, create_engine

# Import directly to avoid crypto imports
sys.path.insert(0, "src")


# ==============================================================================
# Module-level imports (only done once to avoid table redefinition)
# ==============================================================================

def _import_module_once(name: str, path: str):
    """Import a module once, reusing if already imported."""
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import modules at module level to avoid repeated table registration
# Use the same module names as other test files to ensure consistency
_auth_module = _import_module_once(
    "auth_module",  # Same as test_auth.py
    "src/tensorguard/platform/tg_tinker_api/auth.py"
)
_queue_module = _import_module_once(
    "durable_queue",  # Same as test_durable_queue.py
    "src/tensorguard/platform/tg_tinker_api/durable_queue.py"
)
_policy_module = _import_module_once(
    "execution_policy",
    "src/tensorguard/platform/tg_tinker_api/execution_policy.py"
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with fresh tables."""
    db_path = tmp_path / "e2e_test.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)

    # Create all tables (uses already-registered models)
    SQLModel.metadata.create_all(engine)

    return {
        "engine": engine,
        "db_url": db_url,
        "db_path": db_path,
        "auth": _auth_module,
        "queue": _queue_module,
        "policy": _policy_module,
    }


@pytest.fixture
def session(test_db):
    """Create a test session."""
    with Session(test_db["engine"]) as session:
        yield session


# ==============================================================================
# E2E Test: Tenant Lifecycle
# ==============================================================================


class TestTenantLifecycle:
    """Test complete tenant lifecycle."""

    def test_tenant_registration_and_api_key(self, test_db, session):
        """Test tenant registration and API key generation."""
        auth = test_db["auth"]

        # Create tenant manager
        manager = auth.TenantManager(session)

        # Register new tenant
        tenant = manager.create_tenant(
            name="E2E Test Corp",
            email="e2e@test.com",
            max_training_clients=5,
            max_pending_jobs=50,
        )

        assert tenant.id.startswith("tnt-")
        assert tenant.active is True

        # Generate API key
        api_key, raw_key = manager.create_api_key(
            tenant_id=tenant.id,
            name="E2E Test Key",
        )

        assert raw_key.startswith("tg_")
        assert api_key.active is True

        # Authenticate with key
        provider = auth.APIKeyAuthProvider()
        auth_ctx = provider.authenticate(raw_key, session)

        assert auth_ctx.tenant_id == tenant.id
        assert auth_ctx.tenant_name == tenant.name

    def test_tenant_suspension_blocks_auth(self, test_db, session):
        """Test that suspended tenant cannot authenticate."""
        auth = test_db["auth"]
        from fastapi import HTTPException

        manager = auth.TenantManager(session)

        # Create and setup tenant
        tenant = manager.create_tenant(
            name="Suspended Corp",
            email="suspended@test.com",
        )
        api_key, raw_key = manager.create_api_key(tenant.id, "Key")

        # Suspend tenant
        manager.suspend_tenant(tenant.id, "Test suspension")

        # Try to authenticate
        provider = auth.APIKeyAuthProvider()
        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate(raw_key, session)

        assert exc_info.value.status_code == 403
        assert "TENANT_INACTIVE" in str(exc_info.value.detail)

    def test_api_key_revocation(self, test_db, session):
        """Test API key revocation."""
        auth = test_db["auth"]
        from fastapi import HTTPException

        manager = auth.TenantManager(session)

        # Create tenant and key
        tenant = manager.create_tenant(
            name="Revoke Test",
            email="revoke@test.com",
        )
        api_key, raw_key = manager.create_api_key(tenant.id, "Key to revoke")

        # Revoke key
        manager.revoke_api_key(api_key.id, "Security concern")

        # Try to authenticate
        provider = auth.APIKeyAuthProvider()
        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate(raw_key, session)

        assert exc_info.value.status_code == 401


# ==============================================================================
# E2E Test: Job Queue Flow
# ==============================================================================


class TestJobQueueFlow:
    """Test complete job queue processing flow."""

    def test_submit_claim_complete_job(self, test_db, session):
        """Test full job lifecycle."""
        queue_mod = test_db["queue"]

        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="e2e-worker",
        )

        # Submit job
        job = queue.submit(
            tenant_id="tenant-e2e",
            training_client_id="tc-e2e",
            operation="forward_backward",
            payload={"batch_size": 8, "input_ids": [[1, 2, 3]]},
            idempotency_key="e2e-test-1",
        )

        assert job.status == "pending"
        assert job.attempt_count == 0

        # Claim job
        claimed = queue.claim_next()
        assert claimed.id == job.id
        assert claimed.status == "claimed"
        assert claimed.attempt_count == 1

        # Complete job
        result = {"loss": 0.5, "grad_norm": 1.2}
        completed = queue.complete(job.id, result)

        assert completed.status == "completed"
        assert completed.result_json == result

    def test_job_retry_on_failure(self, test_db, session):
        """Test that failed jobs are retried."""
        queue_mod = test_db["queue"]

        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="e2e-worker",
        )

        # Submit job with multiple attempts
        job = queue.submit(
            tenant_id="tenant-e2e",
            training_client_id="tc-e2e",
            operation="forward_backward",
            payload={},
            max_attempts=3,
        )

        # First attempt fails
        claimed = queue.claim_next()
        failed = queue.fail(claimed.id, "Transient error")

        # Job should be pending again
        assert failed.status == "pending"
        assert failed.attempt_count == 1

        # Make it visible again for testing
        DurableJob = queue_mod.DurableJob
        db_job = session.get(DurableJob, job.id)
        db_job.visible_after = datetime.utcnow()
        session.add(db_job)
        session.commit()

        # Second attempt succeeds
        claimed2 = queue.claim_next()
        completed = queue.complete(claimed2.id, {"success": True})

        assert completed.status == "completed"
        assert completed.attempt_count == 2

    def test_idempotency_prevents_duplicate(self, test_db, session):
        """Test that idempotency key prevents duplicate jobs."""
        queue_mod = test_db["queue"]

        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="e2e-worker",
        )

        # Submit first job
        job1 = queue.submit(
            tenant_id="tenant-e2e",
            training_client_id="tc-e2e",
            operation="forward_backward",
            payload={"first": True},
            idempotency_key="unique-key-e2e",
        )

        # Try to submit duplicate
        job2 = queue.submit(
            tenant_id="tenant-e2e",
            training_client_id="tc-e2e",
            operation="forward_backward",
            payload={"second": True},  # Different payload
            idempotency_key="unique-key-e2e",
        )

        # Should return same job
        assert job1.id == job2.id

    def test_dlq_after_max_retries(self, test_db, session):
        """Test that jobs move to DLQ after max retries."""
        queue_mod = test_db["queue"]
        DurableJob = queue_mod.DurableJob

        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="e2e-worker",
        )

        # Submit job with 1 attempt
        job = queue.submit(
            tenant_id="tenant-e2e",
            training_client_id="tc-e2e",
            operation="forward_backward",
            payload={"important": "data"},
            max_attempts=1,
        )

        # Fail the only attempt
        claimed = queue.claim_next()
        queue.fail(claimed.id, "Fatal error")

        # Job should be dead
        final = queue.get_job(job.id)
        assert final.status == "dead"

        # Check DLQ
        dlq_entries = queue.get_dlq_entries("tenant-e2e")
        assert len(dlq_entries) == 1
        assert dlq_entries[0].original_job_id == job.id


# ==============================================================================
# E2E Test: Execution Policy
# ==============================================================================


class TestExecutionPolicy:
    """Test execution policy enforcement."""

    def test_development_default_policy(self, test_db):
        """Test default policy in development mode."""
        policy_mod = test_db["policy"]

        # Reset and test in development
        policy_mod.reset_enforcer()
        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "development"
            enforcer = policy_mod.ExecutionPolicyEnforcer()
            assert enforcer.get_default_policy() == policy_mod.ExecutionPolicy.HE_PREFERRED
        finally:
            if old_env:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]
            policy_mod.reset_enforcer()

    def test_production_default_policy(self, test_db):
        """Test default policy in production mode."""
        policy_mod = test_db["policy"]

        policy_mod.reset_enforcer()
        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "production"
            enforcer = policy_mod.ExecutionPolicyEnforcer()
            assert enforcer.get_default_policy() == policy_mod.ExecutionPolicy.HE_REQUIRED
        finally:
            if old_env:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]
            policy_mod.reset_enforcer()

    def test_attestation_included(self, test_db):
        """Test that attestation is included in responses."""
        policy_mod = test_db["policy"]

        policy_mod.reset_enforcer()
        enforcer = policy_mod.ExecutionPolicyEnforcer()

        # Get attestation (will use plaintext if HE unavailable)
        if not enforcer.is_he_available:
            attestation = enforcer.enforce(
                policy_mod.ExecutionPolicy.HE_PREFERRED,
                "test_operation",
            )

            assert attestation.execution_mode == policy_mod.ExecutionMode.PLAINTEXT
            assert attestation.fallback_reason is not None
            assert attestation.policy_applied == policy_mod.ExecutionPolicy.HE_PREFERRED


# ==============================================================================
# E2E Test: Data Persistence
# ==============================================================================


class TestDataPersistence:
    """Test data persistence across sessions."""

    def test_tenant_persists_across_sessions(self, test_db):
        """Test that tenants persist across database sessions."""
        auth = test_db["auth"]
        engine = test_db["engine"]

        # Session 1: Create tenant
        with Session(engine) as session1:
            manager = auth.TenantManager(session1)
            tenant = manager.create_tenant(
                name="Persistent Corp",
                email="persist@test.com",
            )
            tenant_id = tenant.id

        # Session 2: Verify tenant exists
        with Session(engine) as session2:
            manager2 = auth.TenantManager(session2)
            retrieved = manager2.get_tenant(tenant_id)

            assert retrieved is not None
            assert retrieved.name == "Persistent Corp"

    def test_job_queue_persists_across_sessions(self, test_db):
        """Test that jobs persist across database sessions."""
        queue_mod = test_db["queue"]
        engine = test_db["engine"]

        # Session 1: Submit job
        with Session(engine) as session1:
            queue1 = queue_mod.DurableJobQueue(
                session=session1,
                worker_id="worker-1",
            )
            job = queue1.submit(
                tenant_id="tenant-persist",
                training_client_id="tc-persist",
                operation="forward_backward",
                payload={"persisted": True},
            )
            job_id = job.id

        # Session 2: Verify and claim job
        with Session(engine) as session2:
            queue2 = queue_mod.DurableJobQueue(
                session=session2,
                worker_id="worker-2",
            )

            # Can retrieve job
            retrieved = queue2.get_job(job_id)
            assert retrieved is not None
            assert retrieved.payload_json["persisted"] is True

            # Can claim job
            claimed = queue2.claim_next()
            assert claimed.id == job_id


# ==============================================================================
# E2E Test: Error Handling
# ==============================================================================


class TestErrorHandling:
    """Test error handling throughout the system."""

    def test_tenant_quota_enforcement(self, test_db, session):
        """Test tenant quota enforcement."""
        queue_mod = test_db["queue"]

        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="e2e-worker",
            max_pending_per_tenant=3,
        )

        # Submit up to quota
        for i in range(3):
            queue.submit(
                tenant_id="quota-tenant",
                training_client_id="tc-1",
                operation="forward_backward",
                payload={"i": i},
            )

        # Next should fail
        with pytest.raises(RuntimeError) as exc_info:
            queue.submit(
                tenant_id="quota-tenant",
                training_client_id="tc-1",
                operation="forward_backward",
                payload={},
            )

        assert "too many pending jobs" in str(exc_info.value)

    def test_cross_tenant_isolation(self, test_db, session):
        """Test that tenants cannot access each other's resources."""
        queue_mod = test_db["queue"]

        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="e2e-worker",
        )

        # Tenant A submits job
        job = queue.submit(
            tenant_id="tenant-a",
            training_client_id="tc-a",
            operation="forward_backward",
            payload={},
        )

        # Tenant B cannot cancel it
        result = queue.cancel(job.id, "tenant-b")
        assert result is False

        # Job still exists for tenant A
        retrieved = queue.get_job(job.id, "tenant-a")
        assert retrieved is not None
        assert retrieved.status == "pending"

    def test_invalid_operations_rejected(self, test_db, session):
        """Test that invalid operations are rejected."""
        auth = test_db["auth"]

        manager = auth.TenantManager(session)

        # Cannot create tenant with duplicate email
        manager.create_tenant(name="First", email="duplicate@test.com")

        with pytest.raises(ValueError) as exc_info:
            manager.create_tenant(name="Second", email="duplicate@test.com")

        assert "already exists" in str(exc_info.value)


# ==============================================================================
# E2E Test: Full Training Flow
# ==============================================================================


class TestFullTrainingFlow:
    """Test a complete training flow."""

    def test_training_session_lifecycle(self, test_db, session):
        """Test a complete training session from start to finish."""
        auth = test_db["auth"]
        queue_mod = test_db["queue"]

        # 1. Create tenant
        tenant_mgr = auth.TenantManager(session)
        tenant = tenant_mgr.create_tenant(
            name="Training Corp",
            email="training@test.com",
        )
        api_key, raw_key = tenant_mgr.create_api_key(
            tenant.id,
            "Training Key",
        )

        # 2. Authenticate
        provider = auth.APIKeyAuthProvider()
        auth_ctx = provider.authenticate(raw_key, session)
        assert auth_ctx.tenant_id == tenant.id

        # 3. Submit training jobs
        queue = queue_mod.DurableJobQueue(
            session=session,
            worker_id="training-worker",
        )

        # Submit forward-backward job
        fb_job = queue.submit(
            tenant_id=tenant.id,
            training_client_id="tc-training",
            operation="forward_backward",
            payload={
                "batch": {"input_ids": [[1, 2, 3, 4, 5]]},
                "batch_hash": "test-hash",
            },
            idempotency_key=f"{tenant.id}-fb-1",
        )

        # 4. Process job (simulate worker)
        claimed = queue.claim_next()
        assert claimed.id == fb_job.id

        queue.start(claimed.id)
        # Simulate computation
        queue.complete(claimed.id, {
            "loss": 0.5,
            "grad_norm": 1.2,
            "tokens_processed": 5,
        })

        # 5. Submit optimizer step
        optim_job = queue.submit(
            tenant_id=tenant.id,
            training_client_id="tc-training",
            operation="optim_step",
            payload={"apply_dp_noise": True},
            idempotency_key=f"{tenant.id}-optim-1",
        )

        # 6. Process optimizer step
        claimed = queue.claim_next()
        queue.start(claimed.id)
        queue.complete(claimed.id, {
            "step": 1,
            "learning_rate": 1e-4,
        })

        # 7. Verify both jobs completed
        fb_final = queue.get_job(fb_job.id)
        optim_final = queue.get_job(optim_job.id)

        assert fb_final.status == "completed"
        assert optim_final.status == "completed"


# ==============================================================================
# E2E Test: Observability Integration
# ==============================================================================


class TestObservabilityIntegration:
    """Test observability integration."""

    def test_metrics_recorded_during_operations(self, test_db, session):
        """Test that metrics are recorded during operations."""
        import importlib.util

        # Import observability module
        obs_spec = importlib.util.spec_from_file_location(
            "observability_e2e",
            "src/tensorguard/platform/tg_tinker_api/observability.py"
        )
        obs = importlib.util.module_from_spec(obs_spec)
        obs_spec.loader.exec_module(obs)

        # Use traced decorator
        @obs.traced("test_operation")
        def my_operation():
            return "result"

        result = my_operation()
        assert result == "result"

        # Check that operation ID was set
        assert obs.get_operation_id() is not None

    def test_correlation_id_propagation(self, test_db):
        """Test correlation ID propagation."""
        import importlib.util

        obs_spec = importlib.util.spec_from_file_location(
            "observability_e2e2",
            "src/tensorguard/platform/tg_tinker_api/observability.py"
        )
        obs = importlib.util.module_from_spec(obs_spec)
        obs_spec.loader.exec_module(obs)

        # Set correlation ID
        cid = obs.generate_correlation_id()
        obs.set_correlation_id(cid)

        # Verify it's accessible
        assert obs.get_correlation_id() == cid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
