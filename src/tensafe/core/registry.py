"""
TenSafe Unified Function Registry.

This module provides a unified registry for pluggable functions:
- Loss functions (for SFT training)
- Reward functions (for RLVR training)
- Metric functions (for evaluation)
- Custom callbacks

The registry supports:
- Registration by decorator or explicit call
- Resolution by name or dotted import path
- Default argument binding
- Type validation

This unifies the separate loss and reward registries into a single
consistent pattern.

Usage:
    from tensafe.core.registry import (
        register_function,
        resolve_function,
        get_loss_registry,
        get_reward_registry,
    )

    # Register a loss function
    @register_function("my_loss", registry="loss")
    def my_loss(outputs, batch, **kwargs):
        return {"loss": outputs["logits"].mean()}

    # Register a reward function
    @register_function("my_reward", registry="reward")
    def my_reward(prompt, response, meta=None, **kwargs):
        return 1.0 if "good" in response else -0.5

    # Resolve functions
    loss_fn = resolve_function("my_loss", registry="loss")
    reward_fn = resolve_function("my_reward", registry="reward")
"""

from __future__ import annotations

import functools
import importlib
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Callable[..., Any])


# ==============================================================================
# Protocols for function types
# ==============================================================================


@runtime_checkable
class LossFnProtocol(Protocol):
    """Protocol for loss functions."""

    def __call__(
        self,
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute loss.

        Args:
            outputs: Model outputs (logits, hidden_states, etc.)
            batch: Training batch (input_ids, labels, etc.)
            **kwargs: Additional arguments

        Returns:
            Dict with at least 'loss' key, optionally 'metrics' and 'auxiliary'
        """
        ...


@runtime_checkable
class RewardFnProtocol(Protocol):
    """Protocol for reward functions."""

    def __call__(
        self,
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward.

        Args:
            prompt: Input prompt
            response: Generated response
            meta: Optional metadata
            **kwargs: Additional arguments

        Returns:
            Scalar reward value
        """
        ...


@runtime_checkable
class MetricFnProtocol(Protocol):
    """Protocol for metric functions."""

    def __call__(
        self,
        predictions: Any,
        references: Any,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute metrics.

        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional arguments

        Returns:
            Dict of metric name to value
        """
        ...


@runtime_checkable
class CallbackFnProtocol(Protocol):
    """Protocol for callback functions."""

    def __call__(
        self,
        event: str,
        payload: Dict[str, Any],
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle an event.

        Args:
            event: Event name
            payload: Event data
            **kwargs: Additional arguments

        Returns:
            Optional modified payload or None
        """
        ...


# Type for function specifications
FunctionSpec = Union[str, Callable[..., Any]]


# ==============================================================================
# Registry implementation
# ==============================================================================


@dataclass
class RegisteredFunction(Generic[T]):
    """A registered function with metadata."""

    name: str
    fn: T
    description: str = ""
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    deprecation_message: Optional[str] = None

    def __call__(self, *args, **kwargs):
        if self.deprecated:
            logger.warning(
                f"Function '{self.name}' is deprecated. "
                f"{self.deprecation_message or 'Use alternative.'}"
            )
        return self.fn(*args, **kwargs)


class FunctionRegistry(Generic[T]):
    """
    A generic registry for pluggable functions.

    Supports registration, resolution, and metadata tracking.

    Example:
        registry = FunctionRegistry[LossFnProtocol]("loss")

        @registry.register("my_loss")
        def my_loss(outputs, batch):
            return {"loss": ...}

        loss_fn = registry.resolve("my_loss")
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        allow_overwrite: bool = False,
    ):
        """
        Initialize registry.

        Args:
            name: Registry name (e.g., "loss", "reward")
            description: Registry description
            allow_overwrite: Allow overwriting existing registrations
        """
        self.name = name
        self.description = description
        self.allow_overwrite = allow_overwrite
        self._registry: Dict[str, RegisteredFunction[T]] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        fn: Optional[T] = None,
        *,
        overwrite: bool = False,
        description: str = "",
        version: str = "1.0",
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Union[T, Callable[[T], T]]:
        """
        Register a function.

        Can be used as decorator or direct call.

        Args:
            name: Function name
            fn: Function to register (if not using as decorator)
            overwrite: Override existing registration
            description: Function description
            version: Function version
            author: Function author
            tags: Function tags for filtering
            aliases: Alternative names

        Returns:
            Function or decorator
        """

        def decorator(func: T) -> T:
            if name in self._registry and not (overwrite or self.allow_overwrite):
                raise ValueError(
                    f"Function '{name}' already registered in {self.name} registry. "
                    f"Use overwrite=True to replace."
                )

            registered = RegisteredFunction(
                name=name,
                fn=func,
                description=description or getattr(func, "__doc__", "") or "",
                version=version,
                author=author,
                tags=tags or [],
            )

            self._registry[name] = registered
            logger.debug(f"Registered '{name}' in {self.name} registry")

            # Add aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name

            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    def unregister(self, name: str) -> bool:
        """
        Unregister a function.

        Args:
            name: Function name

        Returns:
            True if function was registered
        """
        if name in self._registry:
            del self._registry[name]
            # Remove aliases
            self._aliases = {k: v for k, v in self._aliases.items() if v != name}
            return True
        return False

    def resolve(
        self,
        spec: FunctionSpec,
        **default_kwargs: Any,
    ) -> T:
        """
        Resolve a function specification to a callable.

        Args:
            spec: Function name, dotted path, or callable
            **default_kwargs: Default arguments to bind

        Returns:
            Resolved function

        Raises:
            ValueError: If function cannot be resolved
        """
        # Already a callable
        if callable(spec) and not isinstance(spec, str):
            fn = self._wrap_callable(spec)
            if default_kwargs:
                return self._bind_kwargs(fn, default_kwargs)
            return fn

        # Must be a string
        if not isinstance(spec, str):
            raise ValueError(f"spec must be string or callable, got {type(spec)}")

        # Check aliases
        name = self._aliases.get(spec, spec)

        # Check registry
        if name in self._registry:
            fn = self._registry[name].fn
            if default_kwargs:
                return self._bind_kwargs(fn, default_kwargs)
            return fn

        # Try import from dotted path
        try:
            fn = self._import_from_path(spec)
            fn = self._wrap_callable(fn)
            if default_kwargs:
                return self._bind_kwargs(fn, default_kwargs)
            return fn
        except (ImportError, AttributeError) as e:
            available = list(self._registry.keys())
            raise ValueError(
                f"Cannot resolve '{spec}' in {self.name} registry: {e}\n"
                f"Available: {available}\n"
                f"Or provide a dotted import path like 'module.path:function_name'"
            ) from e

    def get(self, name: str) -> Optional[RegisteredFunction[T]]:
        """Get registered function metadata."""
        name = self._aliases.get(name, name)
        return self._registry.get(name)

    def list_all(self) -> List[str]:
        """List all registered function names."""
        return list(self._registry.keys())

    def list_by_tag(self, tag: str) -> List[str]:
        """List functions with a specific tag."""
        return [
            name for name, reg in self._registry.items()
            if tag in reg.tags
        ]

    def search(self, query: str) -> List[str]:
        """Search functions by name or description."""
        query = query.lower()
        results = []
        for name, reg in self._registry.items():
            if query in name.lower() or query in reg.description.lower():
                results.append(name)
        return results

    def deprecate(
        self,
        name: str,
        message: str = "",
    ) -> None:
        """Mark a function as deprecated."""
        if name in self._registry:
            self._registry[name].deprecated = True
            self._registry[name].deprecation_message = message

    def _wrap_callable(self, fn: Callable[..., Any]) -> T:
        """Wrap a callable to ensure proper return type."""
        # For now, just return as-is
        # Subclasses can override for type-specific wrapping
        return fn  # type: ignore

    def _bind_kwargs(self, fn: T, default_kwargs: Dict[str, Any]) -> T:
        """Bind default keyword arguments to a function."""

        @functools.wraps(fn)
        def bound(*args, **kwargs):
            merged = {**default_kwargs, **kwargs}
            return fn(*args, **merged)

        return bound  # type: ignore

    @staticmethod
    def _import_from_path(dotted_path: str) -> Any:
        """Import an object from a dotted path."""
        if ":" in dotted_path:
            module_path, object_name = dotted_path.rsplit(":", 1)
        else:
            parts = dotted_path.rsplit(".", 1)
            if len(parts) != 2:
                raise ImportError(
                    f"Invalid import path: {dotted_path}. "
                    "Expected 'module.path:object' or 'module.path.object'"
                )
            module_path, object_name = parts

        module = importlib.import_module(module_path)
        return getattr(module, object_name)

    def __contains__(self, name: str) -> bool:
        name = self._aliases.get(name, name)
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)


# ==============================================================================
# Specialized registries
# ==============================================================================


class LossRegistry(FunctionRegistry[LossFnProtocol]):
    """Registry specifically for loss functions."""

    def __init__(self):
        super().__init__(
            name="loss",
            description="Registry for training loss functions",
        )

    def _wrap_callable(self, fn: Callable[..., Any]) -> LossFnProtocol:
        """Wrap to ensure proper LossReturn format."""

        @functools.wraps(fn)
        def wrapped(
            outputs: Union[Dict[str, Any], Any],
            batch: Dict[str, Any],
            **kwargs: Any,
        ) -> Dict[str, Any]:
            result = fn(outputs, batch, **kwargs)
            return _validate_loss_return(result)

        return wrapped  # type: ignore


class RewardRegistry(FunctionRegistry[RewardFnProtocol]):
    """Registry specifically for reward functions."""

    def __init__(self):
        super().__init__(
            name="reward",
            description="Registry for RLVR reward functions",
        )


class MetricRegistry(FunctionRegistry[MetricFnProtocol]):
    """Registry specifically for metric functions."""

    def __init__(self):
        super().__init__(
            name="metric",
            description="Registry for evaluation metrics",
        )


class CallbackRegistry(FunctionRegistry[CallbackFnProtocol]):
    """Registry specifically for callback functions."""

    def __init__(self):
        super().__init__(
            name="callback",
            description="Registry for training callbacks",
        )


# ==============================================================================
# Global registry instances
# ==============================================================================

_LOSS_REGISTRY: Optional[LossRegistry] = None
_REWARD_REGISTRY: Optional[RewardRegistry] = None
_METRIC_REGISTRY: Optional[MetricRegistry] = None
_CALLBACK_REGISTRY: Optional[CallbackRegistry] = None


def get_loss_registry() -> LossRegistry:
    """Get the global loss function registry."""
    global _LOSS_REGISTRY
    if _LOSS_REGISTRY is None:
        _LOSS_REGISTRY = LossRegistry()
        _register_builtin_losses(_LOSS_REGISTRY)
    return _LOSS_REGISTRY


def get_reward_registry() -> RewardRegistry:
    """Get the global reward function registry."""
    global _REWARD_REGISTRY
    if _REWARD_REGISTRY is None:
        _REWARD_REGISTRY = RewardRegistry()
        _register_builtin_rewards(_REWARD_REGISTRY)
    return _REWARD_REGISTRY


def get_metric_registry() -> MetricRegistry:
    """Get the global metric function registry."""
    global _METRIC_REGISTRY
    if _METRIC_REGISTRY is None:
        _METRIC_REGISTRY = MetricRegistry()
    return _METRIC_REGISTRY


def get_callback_registry() -> CallbackRegistry:
    """Get the global callback function registry."""
    global _CALLBACK_REGISTRY
    if _CALLBACK_REGISTRY is None:
        _CALLBACK_REGISTRY = CallbackRegistry()
    return _CALLBACK_REGISTRY


def _get_registry(registry: str) -> FunctionRegistry:
    """Get registry by name."""
    if registry == "loss":
        return get_loss_registry()
    elif registry == "reward":
        return get_reward_registry()
    elif registry == "metric":
        return get_metric_registry()
    elif registry == "callback":
        return get_callback_registry()
    else:
        raise ValueError(f"Unknown registry: {registry}")


# ==============================================================================
# Convenience functions
# ==============================================================================


def register_function(
    name: str,
    registry: str = "loss",
    **kwargs: Any,
) -> Callable[[T], T]:
    """
    Register a function in the specified registry.

    Args:
        name: Function name
        registry: Registry name ("loss", "reward", "metric", "callback")
        **kwargs: Additional registration options

    Returns:
        Decorator

    Example:
        @register_function("my_loss", registry="loss")
        def my_loss(outputs, batch):
            return {"loss": ...}
    """
    reg = _get_registry(registry)
    return reg.register(name, **kwargs)


def resolve_function(
    spec: FunctionSpec,
    registry: str = "loss",
    **default_kwargs: Any,
) -> Callable[..., Any]:
    """
    Resolve a function from the specified registry.

    Args:
        spec: Function name, path, or callable
        registry: Registry name
        **default_kwargs: Default arguments to bind

    Returns:
        Resolved function
    """
    reg = _get_registry(registry)
    return reg.resolve(spec, **default_kwargs)


def list_functions(registry: str = "loss") -> List[str]:
    """List all functions in a registry."""
    return _get_registry(registry).list_all()


# ==============================================================================
# Validation helpers
# ==============================================================================


def _validate_loss_return(result: Any) -> Dict[str, Any]:
    """Validate and normalize loss function return value."""
    # Handle dataclass with to_dict
    if hasattr(result, 'to_dict'):
        result = result.to_dict()

    # Handle dict
    if isinstance(result, dict):
        if "loss" not in result:
            raise ValueError("Loss function must return dict with 'loss' key")
        return result

    # Handle bare tensor/scalar
    if hasattr(result, 'item') or isinstance(result, (int, float)):
        return {"loss": result, "metrics": {}}

    raise ValueError(
        f"Invalid loss return type: {type(result)}. "
        "Expected dict with 'loss' key or scalar."
    )


# ==============================================================================
# Built-in functions
# ==============================================================================


def _register_builtin_losses(registry: LossRegistry) -> None:
    """Register built-in loss functions."""

    @registry.register(
        "token_ce",
        description="Token-level cross-entropy loss for language modeling",
        tags=["sft", "causal_lm"],
    )
    def token_ce(
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Token-level cross-entropy loss."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            # Return mock for environments without torch
            return {"loss": 0.0, "metrics": {}}

        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        labels = batch.get("labels", batch.get("input_ids"))

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

        # Compute perplexity
        with torch.no_grad():
            ppl = torch.exp(loss).item()

        return {
            "loss": loss,
            "metrics": {
                "perplexity": ppl,
                "ce_loss": loss.item(),
            },
        }

    @registry.register(
        "mse",
        description="Mean squared error loss",
        tags=["regression"],
    )
    def mse_loss(
        outputs: Union[Dict[str, Any], Any],
        batch: Dict[str, Any],
        *,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Mean squared error loss."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return {"loss": 0.0, "metrics": {}}

        predictions = outputs.get("predictions") if isinstance(outputs, dict) else outputs
        targets = batch.get("targets", batch.get("labels"))

        loss = F.mse_loss(predictions, targets, reduction=reduction)

        return {
            "loss": loss,
            "metrics": {"mse": loss.item()},
        }


def _register_builtin_rewards(registry: RewardRegistry) -> None:
    """Register built-in reward functions."""

    @registry.register(
        "keyword_contains",
        description="Reward based on keyword presence in response",
        tags=["simple", "keyword"],
    )
    def keyword_contains(
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        keywords: Optional[List[str]] = None,
        positive_reward: float = 1.0,
        negative_reward: float = -0.5,
        **kwargs: Any,
    ) -> float:
        """Reward if response contains target keywords."""
        target_keywords = keywords or []
        if meta and "keywords" in meta:
            target_keywords = meta["keywords"]

        if not target_keywords:
            return positive_reward if response.strip() else negative_reward

        response_lower = response.lower()
        for kw in target_keywords:
            if kw.lower() in response_lower:
                return positive_reward

        return negative_reward

    @registry.register(
        "length_penalty",
        description="Reward based on response length",
        tags=["simple", "length"],
    )
    def length_penalty(
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        target_length: int = 50,
        penalty_scale: float = 0.01,
        **kwargs: Any,
    ) -> float:
        """Reward closer to target length."""
        words = response.split()
        length = len(words)

        if length <= target_length:
            return 1.0 - (target_length - length) * penalty_scale * 0.5
        else:
            return 1.0 - (length - target_length) * penalty_scale

    @registry.register(
        "format_compliance",
        description="Reward based on format compliance (JSON, list, etc.)",
        tags=["format", "structured"],
    )
    def format_compliance(
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        required_format: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Reward if response follows expected format."""
        fmt = required_format or (meta.get("format") if meta else None)

        if fmt == "json":
            response_stripped = response.strip()
            if response_stripped.startswith(("{", "[")) and response_stripped.endswith(("}", "]")):
                return 1.0
            return -0.5

        elif fmt == "numbered_list":
            lines = response.strip().split("\n")
            numbered = sum(1 for l in lines if l.strip() and l.strip()[0].isdigit())
            if numbered > 0 and numbered >= len(lines) * 0.5:
                return 1.0
            return 0.0

        return 0.0

    @registry.register(
        "constant",
        description="Constant reward (for testing)",
        tags=["test"],
    )
    def constant_reward(
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        value: float = 0.0,
        **kwargs: Any,
    ) -> float:
        """Return constant reward value."""
        return value
