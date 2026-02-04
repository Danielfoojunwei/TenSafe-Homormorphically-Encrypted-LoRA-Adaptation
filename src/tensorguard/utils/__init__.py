"""TensorGuard Utilities Module."""

from .production_gates import is_production, require_env

__all__ = ["is_production", "require_env"]
