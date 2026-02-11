"""
Extended Hooks for Non-Linear Adapter Support.

This module extends the base attention hooks with support for:

- Non-linear adapter types (DoRA, AdaLoRA, Gated, etc.)
- Fused QKV projection handling with split/recombine
- MLP layer targeting
- Input-dependent gating
- Original weight/output access for advanced adapters

Building on the base hooks from he_lora_microkernel, this provides
production-grade injection points for all adapter types.

Author: TenSafe Team
"""

import functools
import logging
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .adapter_types import AdapterType, BaseAdapter
from .placement import (
    FusedProjectionHandler,
    LayerType,
    ProjectionTarget,
    ProjectionType,
)

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# HOOK TYPES AND CONFIGURATION
# =============================================================================

class InjectionMode(Enum):
    """How the adapter delta is injected."""
    ADDITIVE = "additive"        # output = output + delta
    MULTIPLICATIVE = "mult"      # output = output * (1 + delta)
    REPLACEMENT = "replace"       # output = delta (for special cases)
    GATED = "gated"              # output = output + gate * delta


@dataclass
class HookConfig:
    """Configuration for an adapter hook."""
    # Adapter configuration
    adapter_type: AdapterType = AdapterType.LORA

    # Injection mode
    injection_mode: InjectionMode = InjectionMode.ADDITIVE

    # Whether this hook needs original weight for computation
    needs_original_weight: bool = False

    # Whether this hook needs original output (computed without adapter)
    needs_original_output: bool = False

    # Scaling factor (can be overridden per-hook)
    scaling: float = 1.0

    # For gated injection
    gate_activation: str = "sigmoid"  # sigmoid, tanh, none

    # Debug/profiling
    enable_profiling: bool = False


@dataclass
class HookStatistics:
    """Statistics for a hook."""
    call_count: int = 0
    total_time_ms: float = 0.0
    delta_compute_time_ms: float = 0.0
    injection_time_ms: float = 0.0

    # Delta statistics
    delta_norm_sum: float = 0.0
    delta_max_abs: float = 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.call_count = 0
        self.total_time_ms = 0.0
        self.delta_compute_time_ms = 0.0
        self.injection_time_ms = 0.0
        self.delta_norm_sum = 0.0
        self.delta_max_abs = 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.call_count, 1)

    @property
    def avg_delta_norm(self) -> float:
        return self.delta_norm_sum / max(self.call_count, 1)


# =============================================================================
# DELTA CALLBACK TYPES
# =============================================================================

# Standard delta callback: (layer_idx, proj_type, hidden_states) -> delta
StandardDeltaCallback = Callable[[int, str, Any], Optional[Any]]

# Extended delta callback: includes original weight and output
ExtendedDeltaCallback = Callable[
    [int, str, Any, Optional[Any], Optional[Any]],  # layer, proj, x, weight, output
    Optional[Any]
]


# =============================================================================
# BASE HOOK CLASS
# =============================================================================

class BaseAdapterHook:
    """
    Base class for adapter injection hooks.

    Provides common functionality for wrapping model modules and
    injecting adapter computations.
    """

    def __init__(
        self,
        layer_idx: int,
        projection_type: str,
        module: Any,
        config: Optional[HookConfig] = None,
    ):
        """
        Initialize the hook.

        Args:
            layer_idx: Layer index in the model
            projection_type: Projection type ("q", "k", "v", "o", etc.)
            module: The module to hook (e.g., nn.Linear)
            config: Hook configuration
        """
        self.layer_idx = layer_idx
        self.projection_type = projection_type
        self.module = module
        self.config = config or HookConfig()

        # Save original forward
        self._original_forward = module.forward

        # Delta callback
        self._delta_callback: Optional[Union[StandardDeltaCallback, ExtendedDeltaCallback]] = None
        self._use_extended_callback = False

        # Adapter instance (for non-linear adapters)
        self._adapter: Optional[BaseAdapter] = None

        # State
        self.enabled = True
        self._installed = False

        # Statistics
        self.stats = HookStatistics()

        # Original weight cache (for DoRA, etc.)
        self._original_weight: Optional[np.ndarray] = None

    def set_delta_callback(
        self,
        callback: Union[StandardDeltaCallback, ExtendedDeltaCallback],
        extended: bool = False,
    ) -> None:
        """Set the delta computation callback."""
        self._delta_callback = callback
        self._use_extended_callback = extended

    def set_adapter(self, adapter: BaseAdapter) -> None:
        """Set an adapter instance for direct computation."""
        self._adapter = adapter

    def cache_original_weight(self) -> None:
        """Cache the original weight matrix from the module."""
        if hasattr(self.module, 'weight'):
            weight = self.module.weight
            if hasattr(weight, 'detach'):
                # PyTorch tensor
                self._original_weight = weight.detach().cpu().numpy()
            else:
                self._original_weight = np.array(weight)

    def install(self) -> None:
        """Install the hook by replacing the module's forward method."""
        if self._installed:
            return

        # Cache original weight if needed
        if self.config.needs_original_weight:
            self.cache_original_weight()

        # Replace forward
        self.module.forward = self._hooked_forward
        self._installed = True

        logger.debug(f"Installed hook on layer {self.layer_idx} {self.projection_type}")

    def remove(self) -> None:
        """Remove the hook by restoring the original forward method."""
        if not self._installed:
            return

        self.module.forward = self._original_forward
        self._installed = False

        logger.debug(f"Removed hook from layer {self.layer_idx} {self.projection_type}")

    def _hooked_forward(self, hidden_states: Any, *args, **kwargs) -> Any:
        """
        Hooked forward method.

        Calls original forward, then applies adapter delta if enabled.
        """
        start_time = time.perf_counter() if self.config.enable_profiling else 0

        # Call original forward
        output = self._original_forward(hidden_states, *args, **kwargs)

        # Apply adapter if enabled
        if self.enabled:
            output = self._apply_adapter(hidden_states, output)

        # Update statistics
        if self.config.enable_profiling:
            self.stats.call_count += 1
            self.stats.total_time_ms += (time.perf_counter() - start_time) * 1000

        return output

    def _apply_adapter(self, hidden_states: Any, output: Any) -> Any:
        """Apply adapter delta to the output."""
        delta = self._compute_delta(hidden_states, output)

        if delta is None:
            return output

        # Apply injection based on mode
        if self.config.injection_mode == InjectionMode.ADDITIVE:
            return output + delta
        elif self.config.injection_mode == InjectionMode.MULTIPLICATIVE:
            return output * (1 + delta)
        elif self.config.injection_mode == InjectionMode.REPLACEMENT:
            return delta
        elif self.config.injection_mode == InjectionMode.GATED:
            gate = self._compute_gate(hidden_states)
            return output + gate * delta
        else:
            return output + delta

    def _compute_delta(self, hidden_states: Any, output: Any) -> Optional[Any]:
        """Compute the adapter delta."""
        start_time = time.perf_counter() if self.config.enable_profiling else 0

        delta = None

        # Try adapter instance first
        if self._adapter is not None:
            delta = self._adapter.forward(
                self._to_numpy(hidden_states),
                original_weight=self._original_weight,
                original_output=self._to_numpy(output) if self.config.needs_original_output else None,
            )
            if delta is not None:
                delta = self._from_numpy(delta, hidden_states)

        # Fall back to callback
        elif self._delta_callback is not None:
            if self._use_extended_callback:
                delta = self._delta_callback(
                    self.layer_idx,
                    self.projection_type,
                    hidden_states,
                    self._original_weight,
                    output if self.config.needs_original_output else None,
                )
            else:
                delta = self._delta_callback(
                    self.layer_idx,
                    self.projection_type,
                    hidden_states,
                )

        # Update statistics
        if self.config.enable_profiling and delta is not None:
            self.stats.delta_compute_time_ms += (time.perf_counter() - start_time) * 1000
            self._update_delta_stats(delta)

        return delta

    def _compute_gate(self, hidden_states: Any) -> Any:
        """Compute gating value for gated injection."""
        # Default: return 1.0 (no gating)
        # Can be overridden in subclasses for input-dependent gating
        if hasattr(hidden_states, 'new_ones'):
            return hidden_states.new_ones(1)
        return 1.0

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def _from_numpy(self, array: np.ndarray, reference: Any) -> Any:
        """Convert numpy array back to tensor type matching reference."""
        if isinstance(reference, np.ndarray):
            return array
        if hasattr(reference, 'new_tensor'):
            return reference.new_tensor(array)
        # Try torch
        try:
            import torch
            return torch.tensor(array, device=reference.device, dtype=reference.dtype)
        except:
            return array

    def _update_delta_stats(self, delta: Any) -> None:
        """Update delta statistics."""
        try:
            if hasattr(delta, 'detach'):
                delta_np = delta.detach().cpu().numpy()
            else:
                delta_np = np.array(delta)

            self.stats.delta_norm_sum += np.linalg.norm(delta_np)
            self.stats.delta_max_abs = max(
                self.stats.delta_max_abs,
                np.max(np.abs(delta_np))
            )
        except:
            pass


# =============================================================================
# SPECIALIZED HOOKS
# =============================================================================

class LinearAdapterHook(BaseAdapterHook):
    """
    Hook for standard linear adapters (LoRA, rsLoRA, LoRA-FA).

    Optimized for the common case of additive delta injection.
    """

    def __init__(
        self,
        layer_idx: int,
        projection_type: str,
        module: Any,
        lora_a: Optional[np.ndarray] = None,
        lora_b: Optional[np.ndarray] = None,
        scaling: float = 1.0,
        config: Optional[HookConfig] = None,
    ):
        config = config or HookConfig(adapter_type=AdapterType.LORA)
        super().__init__(layer_idx, projection_type, module, config)

        # Direct weight storage for fast access
        self._lora_a = lora_a
        self._lora_b = lora_b
        self._scaling = scaling

    def set_weights(
        self,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: Optional[float] = None,
    ) -> None:
        """Set LoRA weights directly."""
        self._lora_a = lora_a.astype(np.float32)
        self._lora_b = lora_b.astype(np.float32)
        if scaling is not None:
            self._scaling = scaling

    def _compute_delta(self, hidden_states: Any, output: Any) -> Optional[Any]:
        """Optimized delta computation for linear adapters."""
        # Use direct weights if available
        if self._lora_a is not None and self._lora_b is not None:
            x = self._to_numpy(hidden_states)

            # delta = scaling * (x @ A.T) @ B.T
            intermediate = np.matmul(x, self._lora_a.T)
            delta = np.matmul(intermediate, self._lora_b.T)
            delta = self._scaling * delta

            return self._from_numpy(delta, hidden_states)

        # Fall back to base implementation
        return super()._compute_delta(hidden_states, output)


class DoRAHook(BaseAdapterHook):
    """
    Hook for DoRA (Weight-Decomposed Low-Rank Adaptation).

    DoRA decomposes weights into magnitude and direction, requiring
    access to the original weight matrix for proper computation.
    """

    def __init__(
        self,
        layer_idx: int,
        projection_type: str,
        module: Any,
        config: Optional[HookConfig] = None,
    ):
        config = config or HookConfig(
            adapter_type=AdapterType.DORA,
            needs_original_weight=True,
        )
        super().__init__(layer_idx, projection_type, module, config)

        # DoRA-specific state
        self._lora_a: Optional[np.ndarray] = None
        self._lora_b: Optional[np.ndarray] = None
        self._magnitude: Optional[np.ndarray] = None
        self._scaling: float = 1.0

    def set_weights(
        self,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        magnitude: Optional[np.ndarray] = None,
        scaling: float = 1.0,
    ) -> None:
        """Set DoRA weights."""
        self._lora_a = lora_a.astype(np.float32)
        self._lora_b = lora_b.astype(np.float32)
        self._scaling = scaling

        if magnitude is not None:
            self._magnitude = magnitude.astype(np.float32)
        elif self._original_weight is not None:
            # Initialize magnitude from original weight
            self._magnitude = np.linalg.norm(
                self._original_weight, axis=1, keepdims=True
            ).T.astype(np.float32)

    def _compute_delta(self, hidden_states: Any, output: Any) -> Optional[Any]:
        """Compute DoRA delta."""
        if self._lora_a is None or self._lora_b is None:
            return super()._compute_delta(hidden_states, output)

        if self._original_weight is None:
            # Fall back to standard LoRA if no original weight
            x = self._to_numpy(hidden_states)
            intermediate = np.matmul(x, self._lora_a.T)
            delta = np.matmul(intermediate, self._lora_b.T)
            return self._from_numpy(self._scaling * delta, hidden_states)

        x = self._to_numpy(hidden_states)

        # Compute LoRA update
        lora_update = self._scaling * (self._lora_b @ self._lora_a)

        # Updated weight
        updated_weight = self._original_weight + lora_update

        # Normalize by column
        column_norm = np.linalg.norm(updated_weight, axis=1, keepdims=True)
        column_norm = np.maximum(column_norm, 1e-8)
        normalized_direction = updated_weight / column_norm

        # Apply magnitude
        magnitude = self._magnitude
        if magnitude is None:
            magnitude = np.ones((1, updated_weight.shape[0]), dtype=np.float32)

        effective_weight = magnitude.T * normalized_direction

        # Compute delta: difference from original
        delta_weight = effective_weight - self._original_weight

        # Apply to input
        delta = np.matmul(x, delta_weight.T)

        return self._from_numpy(delta, hidden_states)


class GatedAdapterHook(BaseAdapterHook):
    """
    Hook for gated adapters with input-dependent gating.

    The gate controls how much of the adapter output is applied,
    allowing dynamic adapter influence based on input.
    """

    def __init__(
        self,
        layer_idx: int,
        projection_type: str,
        module: Any,
        gate_type: str = "sigmoid",
        config: Optional[HookConfig] = None,
    ):
        config = config or HookConfig(
            adapter_type=AdapterType.GATED_LORA,
            injection_mode=InjectionMode.GATED,
        )
        super().__init__(layer_idx, projection_type, module, config)

        self.gate_type = gate_type

        # Gate computation weights
        self._gate_proj: Optional[np.ndarray] = None
        self._gate_bias: Optional[np.ndarray] = None

        # LoRA weights
        self._lora_a: Optional[np.ndarray] = None
        self._lora_b: Optional[np.ndarray] = None
        self._scaling: float = 1.0

    def set_weights(
        self,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        gate_proj: Optional[np.ndarray] = None,
        gate_bias: Optional[np.ndarray] = None,
        scaling: float = 1.0,
    ) -> None:
        """Set gated adapter weights."""
        self._lora_a = lora_a.astype(np.float32)
        self._lora_b = lora_b.astype(np.float32)
        self._gate_proj = gate_proj
        self._gate_bias = gate_bias
        self._scaling = scaling

    def _compute_gate(self, hidden_states: Any) -> Any:
        """Compute input-dependent gate."""
        x = self._to_numpy(hidden_states)

        if self._gate_proj is not None:
            # Learned gating: gate = σ(x @ W_g + b_g)
            gate_logits = np.matmul(x, self._gate_proj.T)
            if self._gate_bias is not None:
                gate_logits = gate_logits + self._gate_bias
        else:
            # No gate projection, use mean of input
            gate_logits = np.mean(x, axis=-1, keepdims=True)

        # Apply activation
        if self.gate_type == "sigmoid":
            gate = 1.0 / (1.0 + np.exp(-gate_logits))
        elif self.gate_type == "tanh":
            gate = np.tanh(gate_logits)
        else:
            gate = np.ones_like(gate_logits)

        return self._from_numpy(gate, hidden_states)

    def _compute_delta(self, hidden_states: Any, output: Any) -> Optional[Any]:
        """Compute gated adapter delta."""
        if self._lora_a is None or self._lora_b is None:
            return super()._compute_delta(hidden_states, output)

        x = self._to_numpy(hidden_states)

        # Standard LoRA computation
        intermediate = np.matmul(x, self._lora_a.T)
        delta = np.matmul(intermediate, self._lora_b.T)

        return self._from_numpy(self._scaling * delta, hidden_states)


class FusedQKVHook(BaseAdapterHook):
    """
    Hook for fused QKV projections.

    Handles architectures (Falcon, GPT-NeoX, etc.) that combine Q, K, V
    into a single projection, allowing separate adapters for each.
    """

    def __init__(
        self,
        layer_idx: int,
        module: Any,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        config: Optional[HookConfig] = None,
    ):
        super().__init__(layer_idx, "qkv", module, config)

        # Initialize fused projection handler
        self._handler = FusedProjectionHandler(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        # Separate adapters for Q, K, V
        self._q_adapter: Optional[Callable] = None
        self._k_adapter: Optional[Callable] = None
        self._v_adapter: Optional[Callable] = None

        # Direct weights (for linear adapters)
        self._q_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._k_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._v_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._scaling: float = 1.0

    def set_adapter(
        self,
        q_adapter: Optional[Callable] = None,
        k_adapter: Optional[Callable] = None,
        v_adapter: Optional[Callable] = None,
    ) -> None:
        """Set adapter callbacks for Q, K, V."""
        self._q_adapter = q_adapter
        self._k_adapter = k_adapter
        self._v_adapter = v_adapter

    def set_weights(
        self,
        q_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        k_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        v_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        scaling: float = 1.0,
    ) -> None:
        """Set direct weights for Q, K, V adapters."""
        self._q_weights = q_weights
        self._k_weights = k_weights
        self._v_weights = v_weights
        self._scaling = scaling

    def _hooked_forward(self, hidden_states: Any, *args, **kwargs) -> Any:
        """Hooked forward for fused QKV."""
        # Call original forward
        qkv_output = self._original_forward(hidden_states, *args, **kwargs)

        if not self.enabled:
            return qkv_output

        # Split QKV
        q, k, v = self._handler.split_qkv(qkv_output)

        x = hidden_states

        # Apply adapters to each component
        if self._q_weights is not None:
            q = q + self._compute_linear_delta(x, self._q_weights)
        elif self._q_adapter is not None:
            q_delta = self._q_adapter(x)
            if q_delta is not None:
                q = q + q_delta

        if self._k_weights is not None:
            k = k + self._compute_linear_delta(x, self._k_weights)
        elif self._k_adapter is not None:
            k_delta = self._k_adapter(x)
            if k_delta is not None:
                k = k + k_delta

        if self._v_weights is not None:
            v = v + self._compute_linear_delta(x, self._v_weights)
        elif self._v_adapter is not None:
            v_delta = self._v_adapter(x)
            if v_delta is not None:
                v = v + v_delta

        # Recombine
        return self._handler.combine_qkv(q, k, v)

    def _compute_linear_delta(
        self,
        x: Any,
        weights: Tuple[np.ndarray, np.ndarray],
    ) -> Any:
        """Compute linear delta for one projection."""
        lora_a, lora_b = weights
        x_np = self._to_numpy(x)

        intermediate = np.matmul(x_np, lora_a.T)
        delta = np.matmul(intermediate, lora_b.T)

        return self._from_numpy(self._scaling * delta, x)


# =============================================================================
# HOOK FACTORY AND MANAGEMENT
# =============================================================================

class HookManager:
    """
    Manages adapter hooks across the model.

    Provides centralized control for:
    - Creating and installing hooks
    - Updating hook weights during hot-swap
    - Enabling/disabling hooks
    - Collecting statistics
    """

    def __init__(self):
        self._hooks: Dict[Tuple[int, str], BaseAdapterHook] = {}
        self._fused_hooks: Dict[int, FusedQKVHook] = {}
        self._enabled = True

    def create_hook(
        self,
        target: ProjectionTarget,
        module: Any,
        config: Optional[HookConfig] = None,
        **kwargs,
    ) -> BaseAdapterHook:
        """
        Create an appropriate hook for a target.

        Args:
            target: Projection target
            module: Module to hook
            config: Hook configuration
            **kwargs: Additional arguments passed to hook constructor

        Returns:
            Created hook (not yet installed)
        """
        key = (target.layer_idx, target.projection_type.value)

        # Choose hook type based on target and config
        if target.is_fused and target.projection_type == ProjectionType.QKV_FUSED:
            hook = FusedQKVHook(
                layer_idx=target.layer_idx,
                module=module,
                hidden_size=target.in_features,
                num_heads=kwargs.get('num_heads', 32),
                num_kv_heads=kwargs.get('num_kv_heads'),
                config=config,
            )
            self._fused_hooks[target.layer_idx] = hook
        elif config and config.adapter_type == AdapterType.DORA:
            hook = DoRAHook(
                layer_idx=target.layer_idx,
                projection_type=target.projection_type.value,
                module=module,
                config=config,
            )
        elif config and config.adapter_type == AdapterType.GATED_LORA:
            hook = GatedAdapterHook(
                layer_idx=target.layer_idx,
                projection_type=target.projection_type.value,
                module=module,
                config=config,
            )
        else:
            # Default to linear adapter hook
            hook = LinearAdapterHook(
                layer_idx=target.layer_idx,
                projection_type=target.projection_type.value,
                module=module,
                config=config,
            )

        self._hooks[key] = hook
        return hook

    def get_hook(
        self,
        layer_idx: int,
        projection_type: str,
    ) -> Optional[BaseAdapterHook]:
        """Get hook by layer and projection type."""
        return self._hooks.get((layer_idx, projection_type))

    def get_fused_hook(self, layer_idx: int) -> Optional[FusedQKVHook]:
        """Get fused QKV hook for a layer."""
        return self._fused_hooks.get(layer_idx)

    def install_all(self) -> None:
        """Install all hooks."""
        for hook in self._hooks.values():
            hook.install()
        for hook in self._fused_hooks.values():
            hook.install()

    def remove_all(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks.values():
            hook.remove()
        for hook in self._fused_hooks.values():
            hook.remove()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all hooks."""
        self._enabled = enabled
        for hook in self._hooks.values():
            hook.enabled = enabled
        for hook in self._fused_hooks.values():
            hook.enabled = enabled

    def update_weights(
        self,
        weights: Dict[str, Tuple[np.ndarray, np.ndarray]],
        scaling: float = 1.0,
    ) -> None:
        """
        Update weights for all hooks (for hot-swap).

        Args:
            weights: module_name -> (lora_a, lora_b)
            scaling: Scaling factor
        """
        for (layer_idx, proj_type), hook in self._hooks.items():
            # Find matching weights
            for module_name, (lora_a, lora_b) in weights.items():
                if proj_type in module_name or module_name.endswith(proj_type):
                    if hasattr(hook, 'set_weights'):
                        hook.set_weights(lora_a, lora_b, scaling)
                    break

    def set_delta_callback(
        self,
        callback: StandardDeltaCallback,
    ) -> None:
        """Set delta callback for all hooks."""
        for hook in self._hooks.values():
            hook.set_delta_callback(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics from all hooks."""
        total_calls = 0
        total_time = 0.0
        per_layer = {}

        for (layer_idx, proj_type), hook in self._hooks.items():
            total_calls += hook.stats.call_count
            total_time += hook.stats.total_time_ms

            if layer_idx not in per_layer:
                per_layer[layer_idx] = {}

            per_layer[layer_idx][proj_type] = {
                'calls': hook.stats.call_count,
                'time_ms': hook.stats.total_time_ms,
                'avg_delta_norm': hook.stats.avg_delta_norm,
            }

        return {
            'total_calls': total_calls,
            'total_time_ms': total_time,
            'avg_time_per_call_ms': total_time / max(total_calls, 1),
            'per_layer': per_layer,
            'num_hooks': len(self._hooks),
            'num_fused_hooks': len(self._fused_hooks),
        }

    def reset_statistics(self) -> None:
        """Reset statistics for all hooks."""
        for hook in self._hooks.values():
            hook.stats.reset()
        for hook in self._fused_hooks.values():
            hook.stats.reset()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hooks_for_model(
    model: Any,
    targets: List[ProjectionTarget],
    config: Optional[HookConfig] = None,
    weights: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    scaling: float = 1.0,
) -> HookManager:
    """
    Create and install hooks for a model.

    Args:
        model: The model to hook
        targets: List of projection targets
        config: Hook configuration
        weights: Optional weights to set
        scaling: Scaling factor

    Returns:
        HookManager with installed hooks
    """
    manager = HookManager()

    for target in targets:
        # Get the module
        module = model
        for part in target.module_path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        # Create hook
        hook = manager.create_hook(target, module, config)

        # Set weights if provided
        if weights and hasattr(hook, 'set_weights'):
            for module_name, (lora_a, lora_b) in weights.items():
                if target.module_name in module_name or module_name.endswith(target.projection_type.value):
                    hook.set_weights(lora_a, lora_b, scaling)
                    break

    # Install all hooks
    manager.install_all()

    return manager


__all__ = [
    # Enums
    "InjectionMode",

    # Config
    "HookConfig",
    "HookStatistics",

    # Callbacks
    "StandardDeltaCallback",
    "ExtendedDeltaCallback",

    # Hooks
    "BaseAdapterHook",
    "LinearAdapterHook",
    "DoRAHook",
    "GatedAdapterHook",
    "FusedQKVHook",

    # Manager
    "HookManager",

    # Functions
    "create_hooks_for_model",
]
