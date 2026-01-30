"""
Base types and protocols for pluggable loss functions.

This module defines the contract that all loss functions must follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, TypedDict, Union, runtime_checkable

# Try to import torch for type hints, but allow running without it
try:
    import torch
    Tensor = torch.Tensor
except ImportError:
    # Create a placeholder for environments without torch
    Tensor = Any  # type: ignore


class LossReturn(TypedDict, total=False):
    """
    Return type for loss functions.

    Required fields:
        loss: The scalar loss value (torch.Tensor or float)

    Optional fields:
        metrics: Dictionary of additional metrics to log
        auxiliary: Any auxiliary outputs (e.g., intermediate values for debugging)
    """

    loss: Union[Tensor, float]  # Required: the loss value
    metrics: Dict[str, float]  # Optional: additional metrics to log
    auxiliary: Dict[str, Any]  # Optional: auxiliary outputs


@dataclass
class LossOutput:
    """
    Structured output from loss computation.

    This is an alternative to LossReturn for users who prefer dataclasses.
    Both formats are supported by the training loop.
    """

    loss: Union[Tensor, float]
    metrics: Dict[str, float] = field(default_factory=dict)
    auxiliary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> LossReturn:
        """Convert to dictionary format."""
        result: LossReturn = {"loss": self.loss}
        if self.metrics:
            result["metrics"] = self.metrics
        if self.auxiliary:
            result["auxiliary"] = self.auxiliary
        return result


@runtime_checkable
class LossFn(Protocol):
    """
    Protocol for loss functions.

    All loss functions must be callable with the following signature:
        loss_fn(outputs, batch, **kwargs) -> LossReturn

    Args:
        outputs: Model outputs (typically contains logits, hidden_states, etc.)
                 This can be a dict, tuple, or model-specific output object.
        batch: Training batch (typically contains input_ids, labels, etc.)
               This should be a dict with string keys.
        **kwargs: Additional keyword arguments (e.g., reduction, label_smoothing)

    Returns:
        LossReturn dict with at least 'loss' key, optionally 'metrics' and 'auxiliary'

    Example implementation:
        def my_loss(outputs, batch, **kwargs) -> LossReturn:
            logits = outputs["logits"]
            labels = batch["labels"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {
                "loss": loss,
                "metrics": {"perplexity": torch.exp(loss).item()}
            }
    """

    def __call__(
        self,
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> LossReturn:
        """Compute the loss."""
        ...


# Type alias for loss function (can be callable or string path)
LossSpec = Union[str, Callable[..., LossReturn], LossFn]


def validate_loss_return(result: Any, strict: bool = False) -> LossReturn:
    """
    Validate and normalize a loss function return value.

    Args:
        result: The return value from a loss function
        strict: If True, raise on invalid types; if False, try to coerce

    Returns:
        Normalized LossReturn dict

    Raises:
        ValueError: If result cannot be converted to valid LossReturn
    """
    # Handle LossOutput dataclass
    if isinstance(result, LossOutput):
        return result.to_dict()

    # Handle dict-like results
    if isinstance(result, dict):
        if "loss" not in result:
            raise ValueError("Loss function must return dict with 'loss' key")

        validated: LossReturn = {"loss": result["loss"]}

        if "metrics" in result:
            if not isinstance(result["metrics"], dict):
                raise ValueError("'metrics' must be a dict")
            validated["metrics"] = result["metrics"]

        if "auxiliary" in result:
            validated["auxiliary"] = result["auxiliary"]

        return validated

    # Handle bare tensor/float returns (for simple loss functions)
    if not strict:
        try:
            # Try to use it as a scalar loss
            if hasattr(result, "item"):
                # It's a tensor-like object
                return {"loss": result, "metrics": {}}
            elif isinstance(result, (int, float)):
                return {"loss": result, "metrics": {}}
        except Exception:
            pass

    raise ValueError(
        f"Invalid loss return type: {type(result)}. "
        "Expected dict with 'loss' key, LossOutput, or scalar tensor."
    )


def wrap_simple_loss(
    fn: Callable[..., Union[Tensor, float]],
    name: Optional[str] = None,
) -> LossFn:
    """
    Wrap a simple loss function (returning just a tensor) into a LossFn.

    Args:
        fn: A function that returns just the loss value
        name: Optional name for the wrapped function

    Returns:
        A LossFn-compatible callable

    Example:
        @wrap_simple_loss
        def my_simple_loss(outputs, batch):
            return F.mse_loss(outputs["pred"], batch["target"])
    """

    def wrapped(
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> LossReturn:
        loss = fn(outputs, batch, **kwargs)
        return {"loss": loss, "metrics": {}}

    wrapped.__name__ = name or getattr(fn, "__name__", "wrapped_loss")
    wrapped.__doc__ = fn.__doc__

    return wrapped  # type: ignore
