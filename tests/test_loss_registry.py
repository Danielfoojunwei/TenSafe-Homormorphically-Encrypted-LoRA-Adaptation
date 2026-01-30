"""
Tests for the loss function registry.

Tests string import, registry lookup, and custom callable support.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add tensafe to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.training.losses.base import (
    LossFn,
    LossOutput,
    LossReturn,
    validate_loss_return,
    wrap_simple_loss,
)
from tensafe.training.losses.registry import (
    LOSS_REGISTRY,
    _bind_kwargs,
    _import_from_path,
    _wrap_callable,
    get_registered_losses,
    register_loss,
    resolve_loss,
)


class TestLossReturn:
    """Tests for LossReturn validation."""

    def test_valid_dict_return(self):
        """Test validation of valid dict return."""
        result = {"loss": 1.5, "metrics": {"accuracy": 0.9}}
        validated = validate_loss_return(result)

        assert validated["loss"] == 1.5
        assert validated["metrics"]["accuracy"] == 0.9

    def test_minimal_return(self):
        """Test validation of minimal dict return."""
        result = {"loss": 2.0}
        validated = validate_loss_return(result)

        assert validated["loss"] == 2.0
        assert "metrics" not in validated or validated.get("metrics") == {}

    def test_loss_output_dataclass(self):
        """Test validation of LossOutput dataclass."""
        output = LossOutput(
            loss=1.0,
            metrics={"perplexity": 5.0},
            auxiliary={"hidden_states": "..."},
        )
        validated = validate_loss_return(output)

        assert validated["loss"] == 1.0
        assert validated["metrics"]["perplexity"] == 5.0

    def test_scalar_coercion(self):
        """Test that scalar values are coerced to proper format."""
        # Float
        validated = validate_loss_return(1.5, strict=False)
        assert validated["loss"] == 1.5

        # Int
        validated = validate_loss_return(2, strict=False)
        assert validated["loss"] == 2

    def test_missing_loss_raises(self):
        """Test that missing 'loss' key raises error."""
        with pytest.raises(ValueError, match="must return dict with 'loss' key"):
            validate_loss_return({"metrics": {}})

    def test_strict_mode(self):
        """Test strict mode raises on invalid types."""
        with pytest.raises(ValueError):
            validate_loss_return("invalid", strict=True)


class TestWrapSimpleLoss:
    """Tests for wrapping simple loss functions."""

    def test_wrap_function(self):
        """Test wrapping a simple function."""

        def simple_loss(outputs, batch, **kwargs):
            return outputs["pred"] - batch["target"]

        wrapped = wrap_simple_loss(simple_loss, name="test_loss")

        result = wrapped({"pred": 1.0}, {"target": 0.5})

        assert "loss" in result
        assert result["loss"] == 0.5
        assert wrapped.__name__ == "test_loss"

    def test_preserves_docstring(self):
        """Test that wrapping preserves docstring."""

        def documented_loss(outputs, batch, **kwargs):
            """This is a documented loss function."""
            return 1.0

        wrapped = wrap_simple_loss(documented_loss)

        assert wrapped.__doc__ == "This is a documented loss function."


class TestRegisterLoss:
    """Tests for loss registration."""

    def test_register_as_function(self):
        """Test registering a loss as a function."""

        def my_test_loss(outputs, batch, **kwargs):
            return {"loss": 1.0, "metrics": {}}

        register_loss("test_loss_func", my_test_loss)

        assert "test_loss_func" in LOSS_REGISTRY
        assert LOSS_REGISTRY["test_loss_func"] is my_test_loss

    def test_register_as_decorator(self):
        """Test registering a loss as a decorator."""

        @register_loss("test_loss_decorator")
        def another_loss(outputs, batch, **kwargs):
            return {"loss": 2.0, "metrics": {}}

        assert "test_loss_decorator" in LOSS_REGISTRY

    def test_overwrite_protection(self):
        """Test that overwriting is prevented by default."""

        def loss1(outputs, batch, **kwargs):
            return {"loss": 1.0, "metrics": {}}

        def loss2(outputs, batch, **kwargs):
            return {"loss": 2.0, "metrics": {}}

        register_loss("test_no_overwrite", loss1)

        with pytest.raises(ValueError, match="already registered"):
            register_loss("test_no_overwrite", loss2)

    def test_overwrite_allowed(self):
        """Test that overwriting works when explicitly allowed."""

        def loss1(outputs, batch, **kwargs):
            return {"loss": 1.0, "metrics": {}}

        def loss2(outputs, batch, **kwargs):
            return {"loss": 2.0, "metrics": {}}

        register_loss("test_overwrite_ok", loss1)
        register_loss("test_overwrite_ok", loss2, overwrite=True)

        assert LOSS_REGISTRY["test_overwrite_ok"] is loss2


class TestGetRegisteredLosses:
    """Tests for getting registered losses."""

    def test_returns_copy(self):
        """Test that get_registered_losses returns a copy."""
        losses = get_registered_losses()

        # Should have some built-in losses
        assert len(losses) > 0

        # Modifying shouldn't affect the registry
        losses["fake"] = lambda x, y: None
        assert "fake" not in LOSS_REGISTRY


class TestImportFromPath:
    """Tests for dotted path import."""

    def test_colon_format(self):
        """Test import with colon separator."""
        # Import a known module
        obj = _import_from_path("os.path:join")

        import os.path

        assert obj is os.path.join

    def test_dot_format(self):
        """Test import with dot separator."""
        obj = _import_from_path("os.path.join")

        import os.path

        assert obj is os.path.join

    def test_invalid_module_raises(self):
        """Test that invalid module raises ImportError."""
        with pytest.raises(ImportError, match="Cannot import module"):
            _import_from_path("nonexistent_module:func")

    def test_invalid_attribute_raises(self):
        """Test that invalid attribute raises AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            _import_from_path("os.path:nonexistent_func")


class TestResolveLoss:
    """Tests for loss resolution."""

    def test_resolve_from_registry(self):
        """Test resolving a loss from the registry."""
        # Register a test loss
        @register_loss("test_resolve_registry")
        def test_loss(outputs, batch, **kwargs):
            return {"loss": 42.0, "metrics": {}}

        resolved = resolve_loss("test_resolve_registry")
        result = resolved({}, {})

        assert result["loss"] == 42.0

    def test_resolve_callable(self):
        """Test resolving a callable directly."""

        def my_loss(outputs, batch, **kwargs):
            return {"loss": outputs["value"], "metrics": {}}

        resolved = resolve_loss(my_loss)
        result = resolved({"value": 3.14}, {})

        assert abs(result["loss"] - 3.14) < 0.001

    def test_resolve_with_kwargs(self):
        """Test resolving with default kwargs."""

        def param_loss(outputs, batch, multiplier=1.0, **kwargs):
            return {"loss": outputs["base"] * multiplier, "metrics": {}}

        register_loss("test_param_loss", param_loss)

        resolved = resolve_loss("test_param_loss", multiplier=2.0)
        result = resolved({"base": 5.0}, {})

        assert result["loss"] == 10.0

    def test_resolve_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Cannot resolve loss"):
            resolve_loss("completely_nonexistent_loss_function_xyz")

    def test_resolve_wraps_simple_return(self):
        """Test that simple returns are wrapped properly."""

        def simple_loss(outputs, batch, **kwargs):
            # Returns just a float, not a dict
            return outputs["loss_value"]

        resolved = resolve_loss(simple_loss)
        result = resolved({"loss_value": 1.5}, {})

        # Should be wrapped into proper format
        assert "loss" in result
        assert result["loss"] == 1.5


class TestBindKwargs:
    """Tests for kwarg binding."""

    def test_bind_defaults(self):
        """Test binding default kwargs."""

        def loss_with_params(outputs, batch, alpha=1.0, beta=2.0, **kwargs):
            return {
                "loss": outputs["x"] * alpha + beta,
                "metrics": {"alpha": alpha, "beta": beta},
            }

        bound = _bind_kwargs(loss_with_params, {"alpha": 5.0})
        result = bound({"x": 2.0}, {})

        assert result["loss"] == 2.0 * 5.0 + 2.0  # alpha=5.0, beta=default 2.0

    def test_explicit_overrides_defaults(self):
        """Test that explicit kwargs override bound defaults."""

        def loss_fn(outputs, batch, value=10, **kwargs):
            return {"loss": value, "metrics": {}}

        bound = _bind_kwargs(loss_fn, {"value": 20})

        # Bound default
        result1 = bound({}, {})
        assert result1["loss"] == 20

        # Explicit override
        result2 = bound({}, {}, value=30)
        assert result2["loss"] == 30


class TestBuiltinLossesRegistered:
    """Test that built-in losses are properly registered."""

    def test_token_ce_registered(self):
        """Test that token_ce is registered."""
        assert "token_ce" in LOSS_REGISTRY

    def test_margin_ranking_registered(self):
        """Test that margin_ranking is registered."""
        assert "margin_ranking" in LOSS_REGISTRY

    def test_contrastive_registered(self):
        """Test that contrastive is registered."""
        assert "contrastive" in LOSS_REGISTRY

    def test_mse_registered(self):
        """Test that mse is registered."""
        assert "mse" in LOSS_REGISTRY


class TestLossFnProtocol:
    """Tests for the LossFn Protocol."""

    def test_function_is_loss_fn(self):
        """Test that a function with correct signature is a LossFn."""

        def my_loss(outputs, batch, **kwargs) -> LossReturn:
            return {"loss": 1.0, "metrics": {}}

        # Runtime check using isinstance (with runtime_checkable)
        assert isinstance(my_loss, LossFn)

    def test_callable_class_is_loss_fn(self):
        """Test that a callable class is a LossFn."""

        class MyLossClass:
            def __call__(self, outputs, batch, **kwargs) -> LossReturn:
                return {"loss": 1.0, "metrics": {}}

        loss_obj = MyLossClass()
        assert isinstance(loss_obj, LossFn)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
