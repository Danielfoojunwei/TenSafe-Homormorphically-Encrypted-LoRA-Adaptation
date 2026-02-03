"""
SGLang Hook Implementation for HE-LoRA Delta Injection

SGLang provides model execution through its runtime. This module implements
hooks that intercept hidden states before QKV projections and apply
encrypted deltas from the HE Adapter Service.
"""

import logging
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class SGLangModelHook:
    """
    Hook for intercepting SGLang model execution.

    SGLang's model runner executes forward passes through the model.
    This hook wraps the linear projections in attention layers to
    inject HE-LoRA deltas.

    Attributes:
        layer_idx: Index of the transformer layer
        projection_type: Type of projection ("q", "k", "v", "o")
        module: Reference to the hooked module
        original_forward: Original forward method
        delta_callback: Callback to get delta from HAS
        enabled: Whether the hook is active
        pre_hook: Whether to apply delta before projection (True) or after (False)
    """

    layer_idx: int
    projection_type: str
    module: Any
    original_forward: Callable
    delta_callback: Optional[Callable] = None
    enabled: bool = True
    pre_hook: bool = False  # SGLang default: post-projection injection

    _hook_handle: Optional[Any] = field(default=None, repr=False)

    def hooked_forward(self, hidden_states: 'torch.Tensor', *args, **kwargs) -> 'torch.Tensor':
        """
        Hooked forward pass with delta injection.

        For post-projection (default):
            output = projection(hidden_states) + delta

        For pre-projection:
            output = projection(hidden_states + delta)
        """
        if not self.enabled:
            return self.original_forward(hidden_states, *args, **kwargs)

        if self.pre_hook and self.delta_callback is not None:
            # Pre-projection: apply delta to input hidden states
            delta = self.delta_callback(
                self.layer_idx,
                self.projection_type,
                hidden_states,
            )
            if delta is not None:
                hidden_states = hidden_states + delta

        # Run original projection
        output = self.original_forward(hidden_states, *args, **kwargs)

        if not self.pre_hook and self.delta_callback is not None:
            # Post-projection: apply delta to output
            delta = self.delta_callback(
                self.layer_idx,
                self.projection_type,
                hidden_states,
            )
            if delta is not None:
                output = output + delta

        return output

    def install(self) -> None:
        """Install the hook by replacing the module's forward method."""
        if hasattr(self.module, 'forward'):
            self.module.forward = self.hooked_forward
            logger.debug(
                f"Installed hook for layer {self.layer_idx} "
                f"projection {self.projection_type}"
            )

    def uninstall(self) -> None:
        """Restore the original forward method."""
        if hasattr(self.module, 'forward'):
            self.module.forward = self.original_forward
            logger.debug(
                f"Uninstalled hook for layer {self.layer_idx} "
                f"projection {self.projection_type}"
            )


class HookRegistry:
    """
    Registry for managing SGLang hooks across the model.

    Maintains a collection of hooks and provides utilities for
    batch operations (enable/disable all, cleanup).
    """

    def __init__(self):
        self._hooks: Dict[Tuple[int, str], SGLangModelHook] = {}
        self._model_ref: Optional[weakref.ref] = None
        self._delta_callback: Optional[Callable] = None

    def set_model(self, model: Any) -> None:
        """Set the model reference for hook installation."""
        self._model_ref = weakref.ref(model)

    def set_delta_callback(self, callback: Callable) -> None:
        """
        Set the delta callback for all hooks.

        The callback signature should be:
            callback(layer_idx: int, projection_type: str, hidden_states: Tensor) -> Optional[Tensor]
        """
        self._delta_callback = callback
        # Update existing hooks
        for hook in self._hooks.values():
            hook.delta_callback = callback

    def register_hook(
        self,
        layer_idx: int,
        projection_type: str,
        module: Any,
        pre_hook: bool = False,
    ) -> SGLangModelHook:
        """
        Register and install a hook for a projection module.

        Args:
            layer_idx: Transformer layer index
            projection_type: "q", "k", "v", or "o"
            module: The projection module to hook
            pre_hook: Whether to apply delta before projection

        Returns:
            The created hook instance
        """
        key = (layer_idx, projection_type)

        # Remove existing hook if present
        if key in self._hooks:
            self._hooks[key].uninstall()

        # Create and install new hook
        hook = SGLangModelHook(
            layer_idx=layer_idx,
            projection_type=projection_type,
            module=module,
            original_forward=module.forward,
            delta_callback=self._delta_callback,
            pre_hook=pre_hook,
        )
        hook.install()
        self._hooks[key] = hook

        return hook

    def unregister_hook(self, layer_idx: int, projection_type: str) -> bool:
        """
        Unregister and uninstall a hook.

        Returns:
            True if hook was found and removed, False otherwise
        """
        key = (layer_idx, projection_type)
        if key in self._hooks:
            self._hooks[key].uninstall()
            del self._hooks[key]
            return True
        return False

    def get_hook(self, layer_idx: int, projection_type: str) -> Optional[SGLangModelHook]:
        """Get a specific hook by layer and projection type."""
        return self._hooks.get((layer_idx, projection_type))

    def enable_all(self) -> None:
        """Enable all registered hooks."""
        for hook in self._hooks.values():
            hook.enabled = True
        logger.info(f"Enabled {len(self._hooks)} hooks")

    def disable_all(self) -> None:
        """Disable all registered hooks (but don't uninstall)."""
        for hook in self._hooks.values():
            hook.enabled = False
        logger.info(f"Disabled {len(self._hooks)} hooks")

    def uninstall_all(self) -> None:
        """Uninstall and remove all hooks."""
        for hook in self._hooks.values():
            hook.uninstall()
        self._hooks.clear()
        logger.info("Uninstalled all hooks")

    def get_hooked_layers(self) -> List[int]:
        """Get list of layer indices with active hooks."""
        return sorted(set(k[0] for k in self._hooks.keys()))

    def get_layer_projections(self, layer_idx: int) -> List[str]:
        """Get list of hooked projections for a layer."""
        return [k[1] for k in self._hooks.keys() if k[0] == layer_idx]

    @property
    def hook_count(self) -> int:
        """Total number of registered hooks."""
        return len(self._hooks)

    def get_statistics(self) -> Dict[str, Any]:
        """Get hook registry statistics."""
        layers = self.get_hooked_layers()
        return {
            'total_hooks': len(self._hooks),
            'hooked_layers': layers,
            'layer_count': len(layers),
            'projections_per_layer': {
                layer: self.get_layer_projections(layer)
                for layer in layers
            },
            'callback_set': self._delta_callback is not None,
        }


class RadixAttentionHook:
    """
    Specialized hook for SGLang's RadixAttention.

    RadixAttention uses prefix caching and batched attention.
    This hook handles the specific execution pattern of RadixAttention
    while maintaining compatibility with HE-LoRA delta injection.
    """

    def __init__(
        self,
        layer_idx: int,
        attention_module: Any,
        delta_callback: Optional[Callable] = None,
    ):
        self.layer_idx = layer_idx
        self.attention_module = attention_module
        self.delta_callback = delta_callback
        self._original_forward = None
        self._enabled = True

    def install(self) -> None:
        """Install hook on the attention module."""
        if hasattr(self.attention_module, 'forward'):
            self._original_forward = self.attention_module.forward
            self.attention_module.forward = self._hooked_forward

    def uninstall(self) -> None:
        """Restore original forward method."""
        if self._original_forward is not None:
            self.attention_module.forward = self._original_forward
            self._original_forward = None

    def _hooked_forward(
        self,
        hidden_states: 'torch.Tensor',
        *args,
        **kwargs,
    ) -> 'torch.Tensor':
        """
        Hooked forward that applies deltas to QKV projections.

        SGLang's attention typically computes Q, K, V projections internally.
        This hook intercepts before and after the attention computation.
        """
        if not self._enabled or self._original_forward is None:
            return self._original_forward(hidden_states, *args, **kwargs)

        # Get deltas for this layer's projections
        deltas = {}
        if self.delta_callback is not None:
            for proj in ['q', 'k', 'v']:
                delta = self.delta_callback(self.layer_idx, proj, hidden_states)
                if delta is not None:
                    deltas[proj] = delta

        # Store deltas in kwargs for internal use
        # (The actual injection happens during projection)
        kwargs['_helora_deltas'] = deltas

        # Run attention
        output = self._original_forward(hidden_states, *args, **kwargs)

        # Apply output projection delta if available
        if self.delta_callback is not None:
            o_delta = self.delta_callback(self.layer_idx, 'o', hidden_states)
            if o_delta is not None:
                output = output + o_delta

        return output

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
