"""
Loss function registry and resolution.

Provides utilities for registering and resolving loss functions by name or path.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Union

from .base import LossFn, LossReturn, LossSpec, validate_loss_return

logger = logging.getLogger(__name__)

# Global registry of loss functions
LOSS_REGISTRY: Dict[str, LossFn] = {}


def register_loss(
    name: str,
    loss_fn: Optional[LossFn] = None,
    *,
    overwrite: bool = False,
) -> Union[LossFn, Callable[[LossFn], LossFn]]:
    """
    Register a loss function in the global registry.

    Can be used as a decorator or a regular function.

    Args:
        name: Name to register the loss under
        loss_fn: The loss function (if not using as decorator)
        overwrite: If True, allow overwriting existing registrations

    Returns:
        The loss function (for decorator chaining)

    Example usage:
        # As a decorator
        @register_loss("my_custom_loss")
        def my_loss(outputs, batch, **kwargs):
            ...

        # As a function
        register_loss("another_loss", my_other_loss_fn)
    """

    def decorator(fn: LossFn) -> LossFn:
        if name in LOSS_REGISTRY and not overwrite:
            raise ValueError(
                f"Loss '{name}' already registered. Use overwrite=True to replace."
            )
        LOSS_REGISTRY[name] = fn
        logger.debug(f"Registered loss function: {name}")
        return fn

    if loss_fn is not None:
        return decorator(loss_fn)
    return decorator


def get_registered_losses() -> Dict[str, LossFn]:
    """
    Get all registered loss functions.

    Returns:
        Dictionary mapping names to loss functions
    """
    return dict(LOSS_REGISTRY)


def _import_from_path(dotted_path: str) -> Any:
    """
    Import an object from a dotted path string.

    Supports two formats:
    - "module.submodule:object_name" (explicit)
    - "module.submodule.object_name" (auto-detect)

    Args:
        dotted_path: Path to the object

    Returns:
        The imported object

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If object cannot be found in module
    """
    if ":" in dotted_path:
        # Explicit format: "module.path:object_name"
        module_path, object_name = dotted_path.rsplit(":", 1)
    else:
        # Auto-detect format: "module.path.object_name"
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(
                f"Invalid import path: {dotted_path}. "
                "Expected 'module.path:object' or 'module.path.object'"
            )
        module_path, object_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}") from e

    try:
        obj = getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{object_name}'"
        ) from e

    return obj


def _wrap_callable(fn: Callable[..., Any]) -> LossFn:
    """
    Wrap a callable to ensure it returns proper LossReturn format.

    Args:
        fn: A callable that computes loss

    Returns:
        A LossFn-compatible callable
    """

    def wrapped(
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> LossReturn:
        result = fn(outputs, batch, **kwargs)
        return validate_loss_return(result, strict=False)

    # Preserve original function metadata
    wrapped.__name__ = getattr(fn, "__name__", "wrapped_loss")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__module__ = getattr(fn, "__module__", __name__)

    return wrapped  # type: ignore


def resolve_loss(
    loss_spec: LossSpec,
    **default_kwargs: Any,
) -> LossFn:
    """
    Resolve a loss specification to a callable loss function.

    Args:
        loss_spec: One of:
            - A string name from the registry (e.g., "token_ce")
            - A dotted path string (e.g., "my_module.losses:custom_loss")
            - A callable that computes loss
        **default_kwargs: Default keyword arguments to bind to the loss function

    Returns:
        A LossFn callable

    Raises:
        ValueError: If loss_spec is invalid or cannot be resolved

    Example:
        # Using a registry name
        loss_fn = resolve_loss("token_ce")

        # Using a dotted path
        loss_fn = resolve_loss("my_pkg.losses:custom_ce", label_smoothing=0.1)

        # Using a callable
        loss_fn = resolve_loss(my_custom_loss_fn)

        # With default kwargs
        loss_fn = resolve_loss("token_ce", ignore_index=-100, reduction="mean")
    """
    # If it's already a callable, wrap and return
    if callable(loss_spec) and not isinstance(loss_spec, str):
        fn = _wrap_callable(loss_spec)
        if default_kwargs:
            return _bind_kwargs(fn, default_kwargs)
        return fn

    # It must be a string
    if not isinstance(loss_spec, str):
        raise ValueError(
            f"loss_spec must be a string or callable, got {type(loss_spec)}"
        )

    # Check the registry first
    if loss_spec in LOSS_REGISTRY:
        fn = LOSS_REGISTRY[loss_spec]
        logger.debug(f"Resolved loss from registry: {loss_spec}")
        if default_kwargs:
            return _bind_kwargs(fn, default_kwargs)
        return fn

    # Try to import from dotted path
    try:
        imported = _import_from_path(loss_spec)
        if not callable(imported):
            raise ValueError(f"Imported object is not callable: {loss_spec}")

        fn = _wrap_callable(imported)
        logger.debug(f"Resolved loss from import path: {loss_spec}")
        if default_kwargs:
            return _bind_kwargs(fn, default_kwargs)
        return fn

    except (ImportError, AttributeError) as e:
        # Provide helpful error message
        available = list(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Cannot resolve loss '{loss_spec}': {e}\n"
            f"Available built-in losses: {available}\n"
            f"Or provide a dotted import path like 'module.path:function_name'"
        ) from e


def _bind_kwargs(fn: LossFn, default_kwargs: Dict[str, Any]) -> LossFn:
    """
    Bind default keyword arguments to a loss function.

    Args:
        fn: The loss function
        default_kwargs: Default kwargs to bind

    Returns:
        A new function with bound defaults
    """

    def bound(
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> LossReturn:
        # Merge with defaults (explicit kwargs take precedence)
        merged = {**default_kwargs, **kwargs}
        return fn(outputs, batch, **merged)

    bound.__name__ = getattr(fn, "__name__", "bound_loss")
    bound.__doc__ = fn.__doc__

    return bound  # type: ignore


# Import built-in losses to register them
# This is done at the end to avoid circular imports
def _register_builtin_losses() -> None:
    """Register built-in loss functions."""
    try:
        from . import builtin  # noqa: F401
    except ImportError as e:
        logger.warning(f"Could not load built-in losses: {e}")


# Auto-register built-ins when module is imported
_register_builtin_losses()
