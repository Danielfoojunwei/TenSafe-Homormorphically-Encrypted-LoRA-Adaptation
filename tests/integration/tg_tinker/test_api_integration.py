"""
Integration tests for TG-Tinker API.

Tests the full workflow from creating a training client to running
training steps and saving checkpoints.
"""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorguard.platform.tg_tinker_api import router, start_worker, stop_worker


@pytest.fixture(scope="module")
def app():
    """Create FastAPI app with TG-Tinker routes."""
    app = FastAPI()
    app.include_router(router)

    # Start background worker
    start_worker()

    yield app

    # Cleanup
    stop_worker()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_header():
    """Create authorization header."""
    return {"Authorization": "Bearer test-api-key-12345"}


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tg-tinker"


class TestTrainingClientEndpoints:
    """Tests for training client endpoints."""

    def test_create_training_client(self, client, auth_header):
        """Test creating a training client."""
        response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={
                "model_ref": "test-model/llama-7b",
                "lora_config": {
                    "rank": 16,
                    "alpha": 32.0,
                },
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 1e-4,
                },
                "batch_size": 4,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["model_ref"] == "test-model/llama-7b"
        assert data["status"] == "ready"
        assert data["step"] == 0
        assert "training_client_id" in data
        assert data["training_client_id"].startswith("tc-")

    def test_create_training_client_with_dp(self, client, auth_header):
        """Test creating a training client with DP enabled."""
        response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={
                "model_ref": "test-model/llama-7b",
                "dp_config": {
                    "enabled": True,
                    "noise_multiplier": 1.0,
                    "max_grad_norm": 1.0,
                    "target_epsilon": 8.0,
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["config"]["dp_config"]["enabled"] is True

    def test_list_training_clients(self, client, auth_header):
        """Test listing training clients."""
        # Create a client first
        client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={"model_ref": "test-model/for-listing"},
        )

        response = client.get("/v1/training_clients", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_training_client(self, client, auth_header):
        """Test getting a specific training client."""
        # Create a client
        create_response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={"model_ref": "test-model/for-get"},
        )
        tc_id = create_response.json()["training_client_id"]

        # Get the client
        response = client.get(f"/v1/training_clients/{tc_id}", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["training_client_id"] == tc_id

    def test_get_nonexistent_training_client(self, client, auth_header):
        """Test getting a nonexistent training client returns 404."""
        response = client.get("/v1/training_clients/tc-nonexistent", headers=auth_header)
        assert response.status_code == 404


class TestTrainingPrimitives:
    """Tests for training primitive endpoints."""

    @pytest.fixture
    def training_client_id(self, client, auth_header):
        """Create a training client for testing."""
        response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={"model_ref": "test-model/for-primitives"},
        )
        return response.json()["training_client_id"]

    def test_forward_backward(self, client, auth_header, training_client_id):
        """Test forward_backward returns a future."""
        response = client.post(
            f"/v1/training_clients/{training_client_id}/forward_backward",
            headers=auth_header,
            json={
                "batch": {
                    "input_ids": [[1, 2, 3], [4, 5, 6]],
                    "attention_mask": [[1, 1, 1], [1, 1, 1]],
                    "labels": [[2, 3, -100], [5, 6, -100]],
                }
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "future_id" in data
        assert data["future_id"].startswith("fut-")
        assert data["status"] == "pending"
        assert data["operation"] == "forward_backward"

    def test_optim_step(self, client, auth_header, training_client_id):
        """Test optim_step returns a future."""
        response = client.post(
            f"/v1/training_clients/{training_client_id}/optim_step",
            headers=auth_header,
            json={"apply_dp_noise": True},
        )
        assert response.status_code == 202
        data = response.json()
        assert "future_id" in data
        assert data["operation"] == "optim_step"

    def test_sample(self, client, auth_header, training_client_id):
        """Test sample returns completions (synchronous)."""
        response = client.post(
            f"/v1/training_clients/{training_client_id}/sample",
            headers=auth_header,
            json={
                "prompts": ["Test prompt"],
                "max_tokens": 50,
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert len(data["samples"]) == 1
        assert data["samples"][0]["prompt"] == "Test prompt"
        assert "completion" in data["samples"][0]

    def test_save_state(self, client, auth_header, training_client_id):
        """Test save_state creates encrypted artifact."""
        response = client.post(
            f"/v1/training_clients/{training_client_id}/save_state",
            headers=auth_header,
            json={
                "include_optimizer": True,
                "metadata": {"checkpoint_name": "test"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "artifact_id" in data
        assert data["artifact_id"].startswith("art-")
        assert data["encryption"]["algorithm"] == "AES-256-GCM"
        assert "content_hash" in data
        assert data["content_hash"].startswith("sha256:")

    def test_load_state(self, client, auth_header, training_client_id):
        """Test load_state restores from artifact."""
        # First save state
        save_response = client.post(
            f"/v1/training_clients/{training_client_id}/save_state",
            headers=auth_header,
            json={"include_optimizer": True},
        )
        artifact_id = save_response.json()["artifact_id"]

        # Then load state
        response = client.post(
            f"/v1/training_clients/{training_client_id}/load_state",
            headers=auth_header,
            json={"artifact_id": artifact_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["loaded_artifact_id"] == artifact_id


class TestFutureEndpoints:
    """Tests for future endpoints."""

    @pytest.fixture
    def training_client_id(self, client, auth_header):
        """Create a training client for testing."""
        response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={"model_ref": "test-model/for-futures"},
        )
        return response.json()["training_client_id"]

    def test_get_future_status(self, client, auth_header, training_client_id):
        """Test getting future status."""
        # Create a future
        fb_response = client.post(
            f"/v1/training_clients/{training_client_id}/forward_backward",
            headers=auth_header,
            json={
                "batch": {
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                }
            },
        )
        future_id = fb_response.json()["future_id"]

        # Get status
        response = client.get(f"/v1/futures/{future_id}", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["future_id"] == future_id
        assert data["status"] in ["pending", "running", "completed"]

    def test_get_future_result(self, client, auth_header, training_client_id):
        """Test getting future result after completion."""
        # Create and wait for future to complete
        fb_response = client.post(
            f"/v1/training_clients/{training_client_id}/forward_backward",
            headers=auth_header,
            json={
                "batch": {
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                }
            },
        )
        future_id = fb_response.json()["future_id"]

        # Poll until complete (with timeout)
        for _ in range(50):  # 5 seconds max
            status_response = client.get(f"/v1/futures/{future_id}", headers=auth_header)
            if status_response.json()["status"] == "completed":
                break
            time.sleep(0.1)

        # Get result
        response = client.get(f"/v1/futures/{future_id}/result", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "result" in data
        assert "loss" in data["result"]

    def test_cancel_pending_future(self, client, auth_header, training_client_id):
        """Test cancelling a pending future."""
        # Create a future
        fb_response = client.post(
            f"/v1/training_clients/{training_client_id}/forward_backward",
            headers=auth_header,
            json={
                "batch": {
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                }
            },
        )
        future_id = fb_response.json()["future_id"]

        # Cancel (may or may not succeed depending on timing)
        response = client.post(f"/v1/futures/{future_id}/cancel", headers=auth_header)
        assert response.status_code == 200


class TestAuditLogEndpoints:
    """Tests for audit log endpoints."""

    @pytest.fixture
    def training_client_id(self, client, auth_header):
        """Create a training client and run some operations."""
        response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={"model_ref": "test-model/for-audit"},
        )
        tc_id = response.json()["training_client_id"]

        # Run some operations to generate audit logs
        client.post(
            f"/v1/training_clients/{tc_id}/sample",
            headers=auth_header,
            json={"prompts": ["test"], "max_tokens": 10},
        )

        return tc_id

    def test_get_audit_logs(self, client, auth_header, training_client_id):
        """Test retrieving audit logs."""
        response = client.get(
            f"/v1/audit_logs?training_client_id={training_client_id}",
            headers=auth_header,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Check structure
        entry = data[0]
        assert "entry_id" in entry
        assert "operation" in entry
        assert "request_hash" in entry
        assert "record_hash" in entry
        assert "prev_hash" in entry


class TestFullWorkflow:
    """Test complete training workflow."""

    def test_training_loop(self, client, auth_header):
        """Test a complete training loop with async operations."""
        # 1. Create training client
        create_response = client.post(
            "/v1/training_clients",
            headers=auth_header,
            json={
                "model_ref": "test-model/workflow",
                "lora_config": {"rank": 8},
            },
        )
        assert create_response.status_code == 201
        tc_id = create_response.json()["training_client_id"]

        # 2. Run 2 training steps with async operations
        for step in range(2):
            # Forward-backward
            fb_response = client.post(
                f"/v1/training_clients/{tc_id}/forward_backward",
                headers=auth_header,
                json={
                    "batch": {
                        "input_ids": [[1, 2, 3, 4]],
                        "attention_mask": [[1, 1, 1, 1]],
                    }
                },
            )
            assert fb_response.status_code == 202
            fb_future_id = fb_response.json()["future_id"]

            # Optim step (can be queued before waiting)
            opt_response = client.post(
                f"/v1/training_clients/{tc_id}/optim_step",
                headers=auth_header,
                json={"apply_dp_noise": False},
            )
            assert opt_response.status_code == 202
            opt_future_id = opt_response.json()["future_id"]

            # Wait for both to complete
            for future_id in [fb_future_id, opt_future_id]:
                for _ in range(50):
                    status = client.get(f"/v1/futures/{future_id}", headers=auth_header)
                    if status.json()["status"] == "completed":
                        break
                    time.sleep(0.1)

        # 3. Save checkpoint
        save_response = client.post(
            f"/v1/training_clients/{tc_id}/save_state",
            headers=auth_header,
            json={"include_optimizer": True},
        )
        assert save_response.status_code == 200
        artifact_id = save_response.json()["artifact_id"]

        # 4. Sample from model
        sample_response = client.post(
            f"/v1/training_clients/{tc_id}/sample",
            headers=auth_header,
            json={"prompts": ["Once upon a time"], "max_tokens": 20},
        )
        assert sample_response.status_code == 200

        # 5. Verify audit log has entries
        audit_response = client.get(
            f"/v1/audit_logs?training_client_id={tc_id}",
            headers=auth_header,
        )
        assert audit_response.status_code == 200
        logs = audit_response.json()
        assert len(logs) > 0  # Should have multiple operations logged
