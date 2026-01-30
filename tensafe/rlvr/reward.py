"""
RLVR Reward Functions

Provides a pluggable interface for reward functions, similar to the loss
function registry.

Reward functions can be:
- Python callables
- Dotted path imports
- Registered built-in rewards
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RewardFn(Protocol):
    """
    Protocol for reward functions.

    Reward functions take a prompt, response, and optional metadata,
    and return a scalar reward value.

    Example implementation:
        def my_reward(prompt: str, response: str, meta: dict = None) -> float:
            if "target_word" in response:
                return 1.0
            return -0.5
    """

    def __call__(
        self,
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward for a single (prompt, response) pair."""
        ...


class BatchRewardFn(Protocol):
    """
    Protocol for batch reward functions.

    Batch reward functions process multiple (prompt, response) pairs
    at once for efficiency (e.g., when using a reward model).
    """

    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """Compute rewards for a batch of (prompt, response) pairs."""
        ...


# Type alias for reward specifications
RewardSpec = Union[str, Callable[..., float], RewardFn, BatchRewardFn]

# Global reward registry
REWARD_REGISTRY: Dict[str, RewardFn] = {}


def register_reward(
    name: str,
    reward_fn: Optional[RewardFn] = None,
    *,
    overwrite: bool = False,
) -> Union[RewardFn, Callable[[RewardFn], RewardFn]]:
    """
    Register a reward function.

    Can be used as a decorator or regular function.

    Args:
        name: Name to register under
        reward_fn: The reward function
        overwrite: Allow overwriting existing registrations
    """

    def decorator(fn: RewardFn) -> RewardFn:
        if name in REWARD_REGISTRY and not overwrite:
            raise ValueError(f"Reward '{name}' already registered")
        REWARD_REGISTRY[name] = fn
        logger.debug(f"Registered reward function: {name}")
        return fn

    if reward_fn is not None:
        return decorator(reward_fn)
    return decorator


def _import_from_path(dotted_path: str) -> Any:
    """Import an object from a dotted path."""
    if ":" in dotted_path:
        module_path, object_name = dotted_path.rsplit(":", 1)
    else:
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(f"Invalid import path: {dotted_path}")
        module_path, object_name = parts

    module = importlib.import_module(module_path)
    return getattr(module, object_name)


def resolve_reward(
    reward_spec: RewardSpec,
    **default_kwargs: Any,
) -> RewardFn:
    """
    Resolve a reward specification to a callable reward function.

    Args:
        reward_spec: One of:
            - A string name from the registry
            - A dotted path string
            - A callable
        **default_kwargs: Default kwargs to bind

    Returns:
        A callable reward function

    Example:
        reward_fn = resolve_reward("keyword_contains")
        reward_fn = resolve_reward("my_module:custom_reward")
        reward_fn = resolve_reward(my_function)
    """
    # Already a callable
    if callable(reward_spec) and not isinstance(reward_spec, str):
        if default_kwargs:
            return _bind_kwargs(reward_spec, default_kwargs)  # type: ignore
        return reward_spec  # type: ignore

    # Must be a string
    if not isinstance(reward_spec, str):
        raise ValueError(f"reward_spec must be string or callable, got {type(reward_spec)}")

    # Check registry
    if reward_spec in REWARD_REGISTRY:
        fn = REWARD_REGISTRY[reward_spec]
        if default_kwargs:
            return _bind_kwargs(fn, default_kwargs)
        return fn

    # Try import
    try:
        imported = _import_from_path(reward_spec)
        if not callable(imported):
            raise ValueError(f"Imported object is not callable: {reward_spec}")
        if default_kwargs:
            return _bind_kwargs(imported, default_kwargs)  # type: ignore
        return imported  # type: ignore
    except (ImportError, AttributeError) as e:
        available = list(REWARD_REGISTRY.keys())
        raise ValueError(
            f"Cannot resolve reward '{reward_spec}': {e}\n"
            f"Available rewards: {available}"
        ) from e


def _bind_kwargs(fn: RewardFn, kwargs: Dict[str, Any]) -> RewardFn:
    """Bind default kwargs to a reward function."""

    def bound(
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        return fn(prompt, response, meta, **kwargs)  # type: ignore

    return bound  # type: ignore


def get_registered_rewards() -> List[str]:
    """Get list of all registered reward function names."""
    return list(REWARD_REGISTRY.keys())


# ==============================================================================
# Built-in Reward Functions
# ==============================================================================


@register_reward("keyword_contains")
def keyword_contains_reward(
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    keywords: Optional[List[str]] = None,
    positive_reward: float = 1.0,
    negative_reward: float = -0.5,
    **kwargs: Any,
) -> float:
    """
    Reward based on whether response contains target keywords.

    Args:
        prompt: The prompt (may contain keywords)
        response: The generated response
        meta: Metadata dict, may contain 'keywords' list
        keywords: List of keywords to check for
        positive_reward: Reward if keywords found
        negative_reward: Reward if keywords not found
    """
    # Get keywords from meta or kwarg
    target_keywords = keywords or []
    if meta and "keywords" in meta:
        target_keywords = meta["keywords"]

    if not target_keywords:
        # Default: check if response is non-empty
        return positive_reward if len(response.strip()) > 0 else negative_reward

    # Check for any keyword
    response_lower = response.lower()
    for kw in target_keywords:
        if kw.lower() in response_lower:
            return positive_reward

    return negative_reward


@register_reward("length_penalty")
def length_penalty_reward(
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    target_length: int = 50,
    penalty_scale: float = 0.01,
    **kwargs: Any,
) -> float:
    """
    Reward based on response length relative to target.

    Shorter responses get higher rewards (penalizes verbosity).

    Args:
        prompt: The prompt
        response: The generated response
        meta: Metadata
        target_length: Target response length in tokens/words
        penalty_scale: Scale for length penalty
    """
    words = response.split()
    length = len(words)

    # Reward closer to target, penalize longer
    if length <= target_length:
        return 1.0 - (target_length - length) * penalty_scale * 0.5
    else:
        return 1.0 - (length - target_length) * penalty_scale


@register_reward("format_compliance")
def format_compliance_reward(
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    required_format: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """
    Reward based on format compliance.

    Checks if response follows expected format patterns.

    Args:
        prompt: The prompt
        response: The generated response
        meta: Metadata, may contain 'format' specification
        required_format: Format specification (e.g., "json", "numbered_list")
    """
    fmt = required_format or (meta.get("format") if meta else None)

    if fmt == "json":
        # Check for JSON-like structure
        response_stripped = response.strip()
        if response_stripped.startswith("{") and response_stripped.endswith("}"):
            return 1.0
        elif response_stripped.startswith("[") and response_stripped.endswith("]"):
            return 1.0
        return -0.5

    elif fmt == "numbered_list":
        # Check for numbered list format
        lines = response.strip().split("\n")
        numbered_lines = sum(1 for line in lines if line.strip() and line.strip()[0].isdigit())
        if numbered_lines > 0 and numbered_lines >= len(lines) * 0.5:
            return 1.0
        return 0.0

    # Default: no format requirement
    return 0.0


@register_reward("composite")
def composite_reward(
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    rewards: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> float:
    """
    Combine multiple reward functions with weights.

    Args:
        prompt: The prompt
        response: The generated response
        meta: Metadata
        rewards: List of reward specs with weights:
            [{"name": "keyword_contains", "weight": 0.5, "kwargs": {...}}, ...]
    """
    if not rewards:
        return 0.0

    total_weight = 0.0
    total_reward = 0.0

    for spec in rewards:
        name = spec.get("name")
        weight = spec.get("weight", 1.0)
        fn_kwargs = spec.get("kwargs", {})

        try:
            fn = resolve_reward(name, **fn_kwargs)
            r = fn(prompt, response, meta)
            total_reward += weight * r
            total_weight += weight
        except Exception as e:
            logger.warning(f"Error computing reward '{name}': {e}")

    if total_weight > 0:
        return total_reward / total_weight
    return 0.0


# ==============================================================================
# Reward Model Wrapper (for learned reward models)
# ==============================================================================


class RewardModelWrapper:
    """
    Wrapper for learned reward models.

    Provides a RewardFn-compatible interface for neural reward models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """
        Initialize reward model wrapper.

        Args:
            model_path: Path to the reward model
            device: Device to run model on
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load the reward model (lazy loading)."""
        logger.info(f"Loading reward model from {self.model_path}")
        # In a real implementation, this would load a transformer model
        # For now, we just mark it as loaded
        self._model = "mock_model"
        self._tokenizer = "mock_tokenizer"

    def __call__(
        self,
        prompt: str,
        response: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward using the model."""
        if self._model is None:
            self.load_model()

        # Mock reward computation
        # In reality, this would:
        # 1. Tokenize prompt + response
        # 2. Forward through reward model
        # 3. Return scalar reward
        combined = prompt + response
        reward = len(combined) % 10 / 10.0 - 0.5  # Mock: -0.5 to 0.4

        return reward

    def batch_reward(
        self,
        prompts: List[str],
        responses: List[str],
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """Compute rewards for a batch."""
        return [
            self(p, r, m)
            for p, r, m in zip(
                prompts,
                responses,
                meta or [None] * len(prompts),
            )
        ]
