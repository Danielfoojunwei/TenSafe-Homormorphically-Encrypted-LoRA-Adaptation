"""
TenSafe Pluggable Loss Function Module

This module provides a flexible loss function interface for training.
Users can provide:
- A Python callable
- A dotted-path string import (e.g., "my_pkg.losses:token_ce_loss")
- A built-in registry name ("token_ce", "margin_ranking", "contrastive")

Example usage:
    from tensafe.training.losses import resolve_loss, LossFn, LossReturn

    # Using a built-in loss
    loss_fn = resolve_loss("token_ce")

    # Using a dotted path
    loss_fn = resolve_loss("my_module.losses:custom_loss")

    # Using a callable directly
    loss_fn = resolve_loss(my_custom_loss_function)

    # Computing loss
    result = loss_fn(outputs, batch)
    print(f"Loss: {result['loss']}, Metrics: {result['metrics']}")
"""

from .base import LossFn, LossOutput, LossReturn
from .builtin import (
    contrastive_loss,
    margin_ranking_loss,
    mse_loss,
    token_cross_entropy_loss,
)
from .registry import (
    LOSS_REGISTRY,
    get_registered_losses,
    register_loss,
    resolve_loss,
)

__all__ = [
    # Types
    "LossFn",
    "LossReturn",
    "LossOutput",
    # Registry
    "resolve_loss",
    "register_loss",
    "get_registered_losses",
    "LOSS_REGISTRY",
    # Built-in losses
    "token_cross_entropy_loss",
    "margin_ranking_loss",
    "contrastive_loss",
    "mse_loss",
]
