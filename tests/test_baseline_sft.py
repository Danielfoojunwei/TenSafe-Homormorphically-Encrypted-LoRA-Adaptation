"""
Baseline SFT Tests

Tests for the baseline SFT smoke script to ensure core training functionality works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path for importing the smoke test module
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import from the smoke test script (self-contained, no external deps)
from baseline_sft_smoke import (
    MinimalTrainingClient,
    MockMLBackend,
    SmokeTestMetrics,
    create_synthetic_batch,
    run_smoke_test,
)


class TestSyntheticBatch:
    """Tests for synthetic batch creation."""

    def test_batch_shape(self):
        """Test that synthetic batch has correct shape."""
        batch = create_synthetic_batch(batch_size=4, seq_len=128)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        assert len(batch["input_ids"]) == 4
        assert len(batch["input_ids"][0]) == 128
        assert len(batch["attention_mask"]) == 4
        assert len(batch["attention_mask"][0]) == 128

    def test_batch_values(self):
        """Test that batch values are valid."""
        batch = create_synthetic_batch(batch_size=2, seq_len=64)

        # All attention mask values should be 1
        for row in batch["attention_mask"]:
            assert all(v == 1 for v in row)

        # Input IDs and labels should be in valid range
        for row in batch["input_ids"]:
            assert all(0 <= v < 1000 for v in row)


class TestMockMLBackend:
    """Tests for the mock ML backend."""

    def test_initialization(self):
        """Test backend initializes correctly."""
        backend = MockMLBackend()
        backend.initialize_model(
            training_client_id="test-tc",
            model_ref="test-model",
            config={"optimizer": {"learning_rate": 1e-4}},
        )

        assert "test-tc" in backend._models
        assert backend._models["test-tc"]["step"] == 0

    def test_forward_backward_returns_expected_fields(self):
        """Test forward-backward returns required fields."""
        backend = MockMLBackend()
        backend.initialize_model("test-tc", "test-model", {})

        result = backend.forward_backward(
            "test-tc",
            {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]},
        )

        assert "loss" in result
        assert "grad_norm" in result
        assert "tokens_processed" in result

    def test_optim_step_increments_step(self):
        """Test optim_step increments the training step."""
        backend = MockMLBackend()
        backend.initialize_model("test-tc", "test-model", {})

        assert backend._models["test-tc"]["step"] == 0

        result = backend.optim_step("test-tc")
        assert result["step"] == 1
        assert backend._models["test-tc"]["step"] == 1

    def test_loss_decreases_with_steps(self):
        """Test that loss decreases as steps increase."""
        backend = MockMLBackend()
        backend.initialize_model("test-tc", "test-model", {})
        batch = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

        losses = []
        for _ in range(10):
            result = backend.forward_backward("test-tc", batch)
            losses.append(result["loss"])
            backend.optim_step("test-tc")

        # Loss should generally decrease (mock backend: loss = 2.5 - step * 0.01)
        assert losses[-1] < losses[0]


class TestMinimalTrainingClient:
    """Tests for the minimal training client."""

    def test_initialization(self):
        """Test client initializes correctly."""
        tc = MinimalTrainingClient(model_ref="test-model")

        assert tc.model_ref == "test-model"
        assert tc.step == 0
        assert tc.training_client_id.startswith("tc-smoke-")

    def test_forward_backward(self):
        """Test forward-backward pass returns expected fields."""
        tc = MinimalTrainingClient()
        batch = create_synthetic_batch(batch_size=2, seq_len=64)

        result = tc.forward_backward(batch)

        assert "loss" in result
        assert "grad_norm" in result
        assert "tokens_processed" in result
        assert isinstance(result["loss"], float)
        assert result["loss"] > 0

    def test_optim_step_increments_step(self):
        """Test that optim_step increments the training step."""
        tc = MinimalTrainingClient()
        batch = create_synthetic_batch()

        assert tc.step == 0

        tc.forward_backward(batch)
        tc.optim_step()

        assert tc.step == 1

        tc.forward_backward(batch)
        tc.optim_step()

        assert tc.step == 2

    def test_sample_returns_completions(self):
        """Test that sample returns completions."""
        tc = MinimalTrainingClient()

        result = tc.sample(prompts=["Hello world"])

        assert "samples" in result
        assert len(result["samples"]) == 1
        assert "prompt" in result["samples"][0]
        assert "completion" in result["samples"][0]

    def test_save_load_state(self):
        """Test checkpoint save and load."""
        tc = MinimalTrainingClient()
        batch = create_synthetic_batch()

        # Train a few steps
        for _ in range(3):
            tc.forward_backward(batch)
            tc.optim_step()

        assert tc.step == 3

        # Save state
        save_result = tc.save_state()

        assert "artifact_id" in save_result
        assert "size_bytes" in save_result
        assert save_result["size_bytes"] > 0

        # Load state
        load_result = tc.load_state(
            artifact_id=save_result["artifact_id"],
            state_bytes=save_result["_state_bytes"],
        )

        assert load_result["step"] == 3


class TestSmokeTest:
    """Tests for the full smoke test."""

    def test_smoke_test_passes(self):
        """Test that the smoke test passes with default parameters."""
        metrics = run_smoke_test(
            num_steps=10,
            batch_size=2,
            seq_len=64,
            verbose=False,
        )

        # Check all fields populated
        assert metrics.steps_completed == 10
        assert metrics.initial_loss > 0
        assert metrics.final_loss > 0
        assert len(metrics.loss_history) == 10
        assert len(metrics.grad_norm_history) == 10

        # Check success conditions
        assert metrics.final_loss < metrics.initial_loss, "Loss should decrease"
        assert metrics.sample_generated, "Sampling should work"
        assert metrics.checkpoint_saved, "Checkpoint save should work"
        assert metrics.checkpoint_loaded, "Checkpoint load should work"

    def test_loss_decreases(self):
        """Test that loss monotonically decreases (with mock backend)."""
        metrics = run_smoke_test(
            num_steps=20,
            verbose=False,
        )

        # With mock backend, loss should steadily decrease
        # (loss = 2.5 - step * 0.01)
        for i in range(1, len(metrics.loss_history)):
            assert metrics.loss_history[i] <= metrics.loss_history[i - 1] + 0.01

    def test_metrics_serializable(self):
        """Test that metrics can be serialized to JSON."""
        metrics = run_smoke_test(num_steps=5, verbose=False)

        # Should not raise
        json_str = json.dumps(metrics.to_dict())
        loaded = json.loads(json_str)

        assert loaded["steps_completed"] == 5
        assert len(loaded["loss_history"]) == 5


class TestDeterminism:
    """Tests for reproducibility."""

    def test_same_batch_same_result(self):
        """Test that same batch produces consistent results."""
        tc = MinimalTrainingClient()
        batch = create_synthetic_batch(batch_size=2, seq_len=64)

        # Note: Mock backend has some randomness in grad_norm
        # but loss should be deterministic based on step
        result1 = tc.forward_backward(batch)
        tc.optim_step()

        # Create new client at same state
        tc2 = MinimalTrainingClient()
        result2 = tc2.forward_backward(batch)

        # Loss should be same (step 0 -> loss = 2.5 - 0*0.01 = 2.5, clamped to max(0.1, loss))
        assert result1["loss"] == result2["loss"]


class TestGoldenArtifacts:
    """Tests that compare against golden artifacts."""

    @pytest.fixture
    def golden_dir(self) -> Path:
        return PROJECT_ROOT / "tests" / "golden"

    def test_loss_curve_consistency(self, golden_dir: Path):
        """Test that loss curve matches golden artifact (if exists)."""
        golden_path = golden_dir / "baseline_loss_curve.json"

        if not golden_path.exists():
            pytest.skip("Golden artifact not found - run with --save-golden first")

        with open(golden_path) as f:
            golden = json.load(f)

        # Run smoke test with same parameters
        metrics = run_smoke_test(
            num_steps=golden["steps"],
            verbose=False,
        )

        # Loss values should match within tolerance
        # (Mock backend is deterministic based on step)
        for i, (actual, expected) in enumerate(
            zip(metrics.loss_history, golden["loss_history"])
        ):
            assert abs(actual - expected) < 0.01, (
                f"Loss mismatch at step {i}: {actual} vs {expected}"
            )


class TestIntegration:
    """Integration tests for the training stack."""

    @pytest.mark.slow
    def test_extended_training(self):
        """Test longer training run (50 steps)."""
        metrics = run_smoke_test(
            num_steps=50,
            batch_size=4,
            seq_len=128,
            verbose=False,
        )

        assert metrics.steps_completed == 50
        assert metrics.final_loss < metrics.initial_loss

        # Loss should have decreased significantly
        assert metrics.final_loss < metrics.initial_loss * 0.95

    def test_multiple_checkpoints(self):
        """Test multiple save/load cycles within same client."""
        tc = MinimalTrainingClient()
        batch = create_synthetic_batch()
        checkpoints = []

        for i in range(5):
            tc.forward_backward(batch)
            tc.optim_step()

            if i % 2 == 0:
                save_result = tc.save_state()
                checkpoints.append(save_result)

        # Should have 3 checkpoints (at steps 1, 3, 5)
        assert len(checkpoints) == 3

        # Verify checkpoints have correct steps recorded
        assert checkpoints[0]["step"] == 1  # After first optim_step
        assert checkpoints[1]["step"] == 3  # After third optim_step
        assert checkpoints[2]["step"] == 5  # After fifth optim_step

        # Verify all checkpoints have state bytes
        for cp in checkpoints:
            assert len(cp["_state_bytes"]) > 0
            assert "artifact_id" in cp

        # Reload a checkpoint within the same client
        # (simulates resuming training from an earlier point)
        load_result = tc.load_state(
            artifact_id=checkpoints[1]["artifact_id"],
            state_bytes=checkpoints[1]["_state_bytes"],
        )

        # The loaded step should match what was saved
        assert load_result["step"] == 3
        assert tc.step == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
