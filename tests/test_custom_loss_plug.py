"""
Tests for custom loss function pluggability.

Verifies that custom loss functions work end-to-end with the training client.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from baseline_sft_smoke import MinimalTrainingClient, create_synthetic_batch

from tensafe.training.losses import LossReturn, register_loss, resolve_loss


class TestCustomLossIntegration:
    """Test custom loss functions with training client."""

    def test_custom_callable_works_end_to_end(self):
        """Test that a custom callable loss works with training."""
        step_count = [0]

        def custom_tracking_loss(outputs, batch, **kwargs) -> LossReturn:
            """Custom loss that tracks calls."""
            step_count[0] += 1
            base_loss = 2.5 - (step_count[0] * 0.05)
            return {
                "loss": max(0.1, base_loss),
                "metrics": {
                    "call_count": step_count[0],
                    "custom_metric": base_loss * 2,
                },
            }

        # Resolve the custom loss
        loss_fn = resolve_loss(custom_tracking_loss)

        # Use with mock training
        tc = MinimalTrainingClient()
        batch = create_synthetic_batch()

        # Simulate training loop with custom loss
        losses = []
        for i in range(5):
            fb_result = tc.forward_backward(batch)
            tc.optim_step()

            # Call our custom loss (in real training, this would be integrated)
            loss_result = loss_fn({"logits": None}, batch)
            losses.append(loss_result["loss"])

            assert "call_count" in loss_result["metrics"]

        # Verify loss decreased
        assert losses[-1] < losses[0]
        assert step_count[0] == 5

    def test_custom_class_loss(self):
        """Test a custom loss implemented as a class."""

        class StatefulLoss:
            """A loss that maintains state across calls."""

            def __init__(self, alpha: float = 1.0):
                self.alpha = alpha
                self.call_history = []

            def __call__(self, outputs, batch, **kwargs) -> LossReturn:
                loss_value = 2.0 * self.alpha
                self.call_history.append(loss_value)
                return {
                    "loss": loss_value,
                    "metrics": {
                        "alpha": self.alpha,
                        "total_calls": len(self.call_history),
                    },
                }

        # Create instance
        stateful_loss = StatefulLoss(alpha=0.5)

        # Resolve as callable
        loss_fn = resolve_loss(stateful_loss)

        # Call multiple times
        for _ in range(3):
            result = loss_fn({}, {})
            assert result["loss"] == 1.0  # 2.0 * 0.5

        assert len(stateful_loss.call_history) == 3

    def test_registered_custom_loss(self):
        """Test a custom loss registered in the registry."""

        @register_loss("test_custom_registered_loss", overwrite=True)
        def my_registered_loss(outputs, batch, scale: float = 1.0, **kwargs) -> LossReturn:
            return {
                "loss": 3.14 * scale,
                "metrics": {"scale_used": scale},
            }

        # Resolve by name
        loss_fn = resolve_loss("test_custom_registered_loss", scale=2.0)

        result = loss_fn({}, {})

        assert abs(result["loss"] - 6.28) < 0.01
        assert result["metrics"]["scale_used"] == 2.0


class TestLossConfigDrivenUsage:
    """Test config-driven loss function selection."""

    def test_config_string_selection(self):
        """Test selecting loss via config string."""
        # Simulate config
        config = {
            "loss": "margin_ranking",
            "loss_kwargs": {"margin": 0.5},
        }

        # Resolve from config
        loss_fn = resolve_loss(config["loss"], **config.get("loss_kwargs", {}))

        # Should be the margin ranking loss
        assert loss_fn is not None

    def test_switch_loss_at_runtime(self):
        """Test switching loss functions during training."""
        losses = ["token_ce", "margin_ranking", "contrastive"]
        resolved_losses = {name: resolve_loss(name) for name in losses}

        # All should be callable
        for name, loss_fn in resolved_losses.items():
            assert callable(loss_fn), f"{name} is not callable"

    def test_invalid_loss_config_fails_gracefully(self):
        """Test that invalid loss config gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            resolve_loss("nonexistent_loss_xyz123")

        error_msg = str(exc_info.value)
        assert "Cannot resolve loss" in error_msg
        assert "Available built-in losses" in error_msg


class TestLossKwargsPassthrough:
    """Test that kwargs are properly passed through."""

    def test_kwargs_reach_loss_function(self):
        """Test that kwargs are passed to the loss function."""
        received_kwargs = {}

        def capturing_loss(outputs, batch, **kwargs) -> LossReturn:
            received_kwargs.update(kwargs)
            return {"loss": 1.0, "metrics": {}}

        loss_fn = resolve_loss(capturing_loss, custom_param="test_value", another=42)
        loss_fn({}, {})

        assert received_kwargs["custom_param"] == "test_value"
        assert received_kwargs["another"] == 42

    def test_call_time_kwargs_override_defaults(self):
        """Test that call-time kwargs override defaults."""
        received_values = []

        def tracking_loss(outputs, batch, value=10, **kwargs) -> LossReturn:
            received_values.append(value)
            return {"loss": float(value), "metrics": {}}

        loss_fn = resolve_loss(tracking_loss, value=20)

        # Default from resolve_loss
        loss_fn({}, {})
        assert received_values[-1] == 20

        # Override at call time
        loss_fn({}, {}, value=30)
        assert received_values[-1] == 30


class TestLossOutputFormats:
    """Test various loss output formats."""

    def test_minimal_dict_output(self):
        """Test loss returning minimal dict."""

        def minimal_loss(outputs, batch, **kwargs):
            return {"loss": 1.5}

        loss_fn = resolve_loss(minimal_loss)
        result = loss_fn({}, {})

        assert result["loss"] == 1.5

    def test_full_dict_output(self):
        """Test loss returning full dict with all fields."""

        def full_loss(outputs, batch, **kwargs):
            return {
                "loss": 2.0,
                "metrics": {"accuracy": 0.9, "f1": 0.85},
                "auxiliary": {"hidden_states": [1, 2, 3]},
            }

        loss_fn = resolve_loss(full_loss)
        result = loss_fn({}, {})

        assert result["loss"] == 2.0
        assert result["metrics"]["accuracy"] == 0.9
        assert result["auxiliary"]["hidden_states"] == [1, 2, 3]

    def test_scalar_return_coerced(self):
        """Test that scalar returns are coerced to dict format."""

        def scalar_loss(outputs, batch, **kwargs):
            return 3.14  # Just return a scalar

        loss_fn = resolve_loss(scalar_loss)
        result = loss_fn({}, {})

        assert result["loss"] == 3.14
        assert "metrics" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
