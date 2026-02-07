"""
Attention Projection Hooks for vLLM HE-LoRA Integration

This module provides the hook mechanism for intercepting and modifying
attention projection outputs in vLLM models.

The hooks wrap linear projection modules and inject HE-LoRA deltas
after the projection computation (POST_PROJECTION mode).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import functools
import weakref

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@dataclass
class HookContext:
    """Context passed to delta callback."""
    layer_idx: int
    projection_type: str  # "q", "k", "v", "o"
    batch_size: int
    seq_len: int
    hidden_size: int


@dataclass
class AttentionProjectionHook:
    """
    Hook for an attention projection module.

    Wraps the forward method of a linear projection (q_proj, k_proj, etc.)
    and applies HE-LoRA delta after the original computation.
    """
    layer_idx: int
    projection_type: str  # "q", "k", "v", "o"
    module: Any  # The original nn.Module
    delta_callback: Optional[Callable] = None
    enabled: bool = True

    # Original forward method - set in __post_init__, not passed to __init__
    original_forward: Callable = field(init=False, default=None)

    # Statistics
    call_count: int = 0
    total_delta_time_ms: float = 0.0

    def __post_init__(self):
        # Save original forward
        self.original_forward = self.module.forward


    def hooked_forward(self, hidden_states: 'torch.Tensor', *args, **kwargs) -> 'torch.Tensor':
        """
        Hooked forward method that applies delta after projection.

        Args:
            hidden_states: Input hidden states
            *args, **kwargs: Additional arguments passed to original forward

        Returns:
            Projected output + delta (if callback provided and enabled)
        """
        # Call original projection
        output = self.original_forward(hidden_states, *args, **kwargs)

        # Apply delta if enabled and callback provided
        if self.enabled and self.delta_callback is not None:
            import time
            start = time.perf_counter()

            delta = self.delta_callback(
                self.layer_idx,
                self.projection_type,
                hidden_states,
            )

            if delta is not None:
                output = output + delta

            self.total_delta_time_ms += (time.perf_counter() - start) * 1000

        self.call_count += 1
        return output

    def install(self) -> None:
        """Install the hook by replacing the module's forward method."""
        self.module.forward = self.hooked_forward

    def remove(self) -> None:
        """Remove the hook by restoring the original forward method."""
        self.module.forward = self.original_forward

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.call_count = 0
        self.total_delta_time_ms = 0.0


def create_projection_hooks(
    model: Any,
    layer_indices: List[int],
    projections: List[str],  # ["q", "k", "v"] or ["q", "k", "v", "o"]
    delta_callback: Optional[Callable] = None,
) -> Dict[Tuple[int, str], AttentionProjectionHook]:
    """
    Create hooks for attention projections in a model.

    This function identifies the attention projection modules and creates
    hooks for them. Works with common transformer architectures.

    Args:
        model: The model (vLLM engine's model)
        layer_indices: Which layers to hook
        projections: Which projections to hook ("q", "k", "v", "o")
        delta_callback: Optional callback for computing deltas

    Returns:
        Dict mapping (layer_idx, projection_type) to hook
    """
    hooks = {}

    # Common module naming patterns
    proj_name_patterns = {
        'q': ['q_proj', 'query', 'q'],
        'k': ['k_proj', 'key', 'k'],
        'v': ['v_proj', 'value', 'v'],
        'o': ['o_proj', 'out_proj', 'dense', 'output'],
    }

    # Try to find layers
    layers = None
    for attr in ['model.layers', 'transformer.h', 'layers', 'decoder.layers']:
        try:
            parts = attr.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError("Could not find transformer layers in model")

    # Create hooks for each layer and projection
    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(layers)} layers)")

        layer = layers[layer_idx]

        # Find attention module
        attn = None
        for attr in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break

        if attn is None:
            raise ValueError(f"Could not find attention module in layer {layer_idx}")

        # Create hooks for requested projections
        for proj_type in projections:
            proj_module = None

            # Try to find the projection module
            for pattern in proj_name_patterns.get(proj_type, []):
                if hasattr(attn, pattern):
                    proj_module = getattr(attn, pattern)
                    break

            # Also check QKV packed projection
            if proj_module is None and proj_type in ['q', 'k', 'v']:
                if hasattr(attn, 'qkv_proj'):
                    # QKV packed - need special handling
                    # For now, skip and log warning
                    continue

            if proj_module is None:
                raise ValueError(
                    f"Could not find {proj_type}_proj in layer {layer_idx} attention"
                )

            hook = AttentionProjectionHook(
                layer_idx=layer_idx,
                projection_type=proj_type,
                module=proj_module,
                delta_callback=delta_callback,
            )
            hooks[(layer_idx, proj_type)] = hook

    return hooks


def install_hooks(hooks: Dict[Tuple[int, str], AttentionProjectionHook]) -> None:
    """Install all hooks."""
    for hook in hooks.values():
        hook.install()


def remove_hooks(hooks: Dict[Tuple[int, str], AttentionProjectionHook]) -> None:
    """Remove all hooks."""
    for hook in hooks.values():
        hook.remove()


def set_hooks_enabled(
    hooks: Dict[Tuple[int, str], AttentionProjectionHook],
    enabled: bool,
) -> None:
    """Enable or disable all hooks."""
    for hook in hooks.values():
        hook.enabled = enabled


def set_delta_callback(
    hooks: Dict[Tuple[int, str], AttentionProjectionHook],
    callback: Optional[Callable],
) -> None:
    """Set delta callback for all hooks."""
    for hook in hooks.values():
        hook.delta_callback = callback


def get_hook_statistics(
    hooks: Dict[Tuple[int, str], AttentionProjectionHook],
) -> Dict[str, Any]:
    """Get aggregated statistics from all hooks."""
    total_calls = sum(h.call_count for h in hooks.values())
    total_time = sum(h.total_delta_time_ms for h in hooks.values())

    per_layer = {}
    for (layer_idx, proj_type), hook in hooks.items():
        if layer_idx not in per_layer:
            per_layer[layer_idx] = {}
        per_layer[layer_idx][proj_type] = {
            'calls': hook.call_count,
            'time_ms': hook.total_delta_time_ms,
        }

    return {
        'total_calls': total_calls,
        'total_time_ms': total_time,
        'avg_time_per_call_ms': total_time / max(total_calls, 1),
        'per_layer': per_layer,
    }


def reset_hook_statistics(
    hooks: Dict[Tuple[int, str], AttentionProjectionHook],
) -> None:
    """Reset statistics for all hooks."""
    for hook in hooks.values():
        hook.reset_stats()
