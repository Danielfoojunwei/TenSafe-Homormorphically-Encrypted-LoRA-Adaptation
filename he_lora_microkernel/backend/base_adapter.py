"""
Base Runtime Adapter Interface for HE-LoRA Integration

This module defines the abstract interface that all inference backend
adapters (vLLM, TensorRT-LLM, SGLang) must implement.

The adapter provides hooks for:
  1. Model initialization and metadata extraction
  2. Prefill and decode steps with layer-level access
  3. Delta injection at configurable insertion points
  4. Proper cleanup and resource management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    TYPE_CHECKING
)
import numpy as np

if TYPE_CHECKING:
    import torch


# =============================================================================
# CONFIGURATION TYPES
# =============================================================================

class LoRATargets(Enum):
    """Which attention projections to patch."""
    QKV = "qkv"      # Query, Key, Value only
    QKVO = "qkvo"    # Query, Key, Value, Output


class InsertionPoint(Enum):
    """Where to apply the delta."""
    PRE_PROJECTION = "pre_projection"   # Add delta to x before W (rare)
    POST_PROJECTION = "post_projection"  # Add delta to projected output (standard)


class LayerScope(Enum):
    """Which layers to target."""
    SELF_ATTN_ONLY = "self_attn_only"  # Only self-attention (current milestone)


@dataclass(frozen=True)
class InsertionConfig:
    """
    Configuration for HE-LoRA insertion points.

    Specifies which layers and projections to patch with HE-LoRA deltas.
    """
    # Target projections
    targets: LoRATargets = LoRATargets.QKV

    # Layer indices to patch (None = all layers)
    layers: Optional[List[int]] = None

    # Per-layer target overrides: {layer_idx: targets}
    per_layer_targets: Dict[int, LoRATargets] = field(default_factory=dict)

    # Insertion point (where delta is applied)
    insertion_point: InsertionPoint = InsertionPoint.POST_PROJECTION

    # Layer scope
    layer_scope: LayerScope = LayerScope.SELF_ATTN_ONLY

    def get_targets_for_layer(self, layer_idx: int) -> LoRATargets:
        """Get targets for a specific layer."""
        return self.per_layer_targets.get(layer_idx, self.targets)

    def should_patch_layer(self, layer_idx: int, num_layers: int) -> bool:
        """Check if a layer should be patched."""
        if self.layers is None:
            return True  # Patch all layers
        return layer_idx in self.layers

    def validate(self, num_layers: int) -> List[str]:
        """Validate configuration against model architecture."""
        errors = []

        if self.layers is not None:
            for layer_idx in self.layers:
                if layer_idx < 0 or layer_idx >= num_layers:
                    errors.append(
                        f"Layer index {layer_idx} out of range [0, {num_layers})"
                    )

        for layer_idx in self.per_layer_targets.keys():
            if layer_idx < 0 or layer_idx >= num_layers:
                errors.append(
                    f"Per-layer target layer {layer_idx} out of range"
                )

        return errors


@dataclass
class BatchConfig:
    """Batch configuration for inference."""
    max_batch_size: int = 8
    max_context_length: int = 2048
    dtype: str = "float16"  # Must be FP16 - no quantization


class AttentionType(Enum):
    """Type of attention mechanism used by the model."""
    STANDARD = "standard"      # Standard multi-head attention (Q, K, V projections)
    MLA = "mla"               # Multi-head Latent Attention (Kimi, DeepSeek)
    GQA = "gqa"               # Grouped Query Attention
    MQA = "mqa"               # Multi-Query Attention


@dataclass
class ModelMetadata:
    """
    Metadata extracted from the model.

    Used to validate adapter compatibility and configure insertion points.
    """
    model_id: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int

    # Layer name mapping for debugging
    layer_name_template: str = "model.layers.{idx}"

    # Architecture specifics
    architecture: str = "unknown"
    has_output_projection: bool = True  # Does attention have o_proj?

    # Attention mechanism type
    attention_type: AttentionType = AttentionType.STANDARD

    # MoE (Mixture of Experts) specific fields
    is_moe: bool = False
    num_experts: Optional[int] = None           # Total number of experts
    num_selected_experts: Optional[int] = None  # Experts selected per token
    num_shared_experts: Optional[int] = None    # Shared experts (always active)

    # MLA (Multi-head Latent Attention) specific fields
    # Used by Kimi K2.5, DeepSeek, etc.
    kv_lora_rank: Optional[int] = None          # Latent dimension for KV projection
    q_lora_rank: Optional[int] = None           # Latent dimension for Q projection
    qk_nope_head_dim: Optional[int] = None      # Non-positional embedding head dim
    qk_rope_head_dim: Optional[int] = None      # RoPE head dimension

    # Trust remote code requirement
    requires_trust_remote_code: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'model_id': self.model_id,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'head_dim': self.head_dim,
            'vocab_size': self.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'architecture': self.architecture,
            'has_output_projection': self.has_output_projection,
            'attention_type': self.attention_type.value,
        }

        # Add MoE fields if applicable
        if self.is_moe:
            result.update({
                'is_moe': True,
                'num_experts': self.num_experts,
                'num_selected_experts': self.num_selected_experts,
                'num_shared_experts': self.num_shared_experts,
            })

        # Add MLA fields if applicable
        if self.attention_type == AttentionType.MLA:
            result.update({
                'kv_lora_rank': self.kv_lora_rank,
                'q_lora_rank': self.q_lora_rank,
                'qk_nope_head_dim': self.qk_nope_head_dim,
                'qk_rope_head_dim': self.qk_rope_head_dim,
            })

        return result

    @property
    def is_kimi_architecture(self) -> bool:
        """Check if this is a Kimi-style MLA+MoE architecture."""
        return self.attention_type == AttentionType.MLA and self.is_moe

    @property
    def effective_kv_dim(self) -> int:
        """Get effective KV dimension (latent for MLA, hidden for standard)."""
        if self.attention_type == AttentionType.MLA and self.kv_lora_rank:
            return self.kv_lora_rank
        return self.hidden_size


# =============================================================================
# DELTA CALLBACK TYPE
# =============================================================================

# Delta callback signature:
# (layer_idx, projection_type, hidden_states) -> delta
# projection_type: "q", "k", "v", "o"
DeltaCallback = Callable[[int, str, 'torch.Tensor'], 'torch.Tensor']


@dataclass
class LayerDeltas:
    """
    Deltas for a single attention layer.

    Contains delta tensors for Q, K, V, and optionally O projections.
    """
    layer_idx: int
    dq: Optional['torch.Tensor'] = None  # Delta for query projection
    dk: Optional['torch.Tensor'] = None  # Delta for key projection
    dv: Optional['torch.Tensor'] = None  # Delta for value projection
    do: Optional['torch.Tensor'] = None  # Delta for output projection (if QKVO)

    def has_deltas(self) -> bool:
        """Check if any deltas are set."""
        return any(d is not None for d in [self.dq, self.dk, self.dv, self.do])


# =============================================================================
# BASE RUNTIME ADAPTER
# =============================================================================

class BaseRuntimeAdapter(ABC):
    """
    Abstract base class for inference runtime adapters.

    Each adapter integrates with a specific inference backend (vLLM, TensorRT-LLM,
    SGLang) and provides hooks for HE-LoRA delta injection.
    """

    def __init__(
        self,
        model_id: str,
        batch_config: BatchConfig,
        device: str = "cuda:0",
    ):
        """
        Initialize the adapter.

        Args:
            model_id: HuggingFace model ID or path
            batch_config: Batch configuration
            device: Device to run inference on
        """
        self._model_id = model_id
        self._batch_config = batch_config
        self._device = device
        self._initialized = False
        self._metadata: Optional[ModelMetadata] = None
        self._insertion_config: Optional[InsertionConfig] = None
        self._delta_callback: Optional[DeltaCallback] = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    @abstractmethod
    def init(self) -> ModelMetadata:
        """
        Initialize the model and extract metadata.

        Returns:
            ModelMetadata with model configuration

        Raises:
            RuntimeError: If initialization fails
            ValueError: If model uses quantization (not allowed)
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    # -------------------------------------------------------------------------
    # INSERTION POINT CONFIGURATION
    # -------------------------------------------------------------------------

    def set_insertion_config(self, config: InsertionConfig) -> None:
        """
        Configure which layers and projections to patch.

        Args:
            config: Insertion configuration

        Raises:
            ValueError: If configuration is invalid for this model
        """
        if self._metadata is None:
            raise RuntimeError("Model not initialized. Call init() first.")

        errors = config.validate(self._metadata.num_layers)
        if errors:
            raise ValueError(f"Invalid insertion config: {errors}")

        # Validate QKVO targets only if model has output projection
        if config.targets == LoRATargets.QKVO and not self._metadata.has_output_projection:
            raise ValueError(
                f"Model {self._model_id} does not have output projection; "
                f"cannot use QKVO targets"
            )

        self._insertion_config = config

    def set_delta_callback(self, callback: DeltaCallback) -> None:
        """
        Set callback for computing deltas.

        The callback is invoked for each projection at each patched layer.

        Args:
            callback: Function (layer_idx, projection_type, hidden_states) -> delta
        """
        self._delta_callback = callback

    def get_patched_layers(self) -> List[int]:
        """Get list of layer indices that will be patched."""
        if self._insertion_config is None or self._metadata is None:
            return []

        layers = []
        for i in range(self._metadata.num_layers):
            if self._insertion_config.should_patch_layer(i, self._metadata.num_layers):
                layers.append(i)
        return layers

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------

    @abstractmethod
    def prefill(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None,
    ) -> Any:
        """
        Run prefill phase (process prompt).

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            KV cache handle for subsequent decode steps
        """
        pass

    @abstractmethod
    def decode_one_step(
        self,
        input_ids_last: 'torch.Tensor',
        kv_cache_handle: Any,
    ) -> Tuple['torch.Tensor', Dict[int, Any]]:
        """
        Run one decode step (generate one token).

        During decode, the adapter invokes the delta callback for each
        patched layer and applies the returned deltas.

        Args:
            input_ids_last: Last token IDs (batch_size, 1)
            kv_cache_handle: KV cache from prefill or previous decode

        Returns:
            Tuple of:
              - logits: Output logits (batch_size, vocab_size)
              - layer_states: Dict mapping layer_idx to intermediate states
                              (for debugging/verification)
        """
        pass

    @abstractmethod
    def apply_deltas(
        self,
        layer_idx: int,
        deltas: LayerDeltas,
    ) -> None:
        """
        Apply precomputed deltas to a layer's projections.

        This is an alternative to using a callback - deltas can be
        precomputed and applied directly.

        Args:
            layer_idx: Layer index
            deltas: Delta tensors for Q, K, V, (O)
        """
        pass

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_layer_module(self, layer_idx: int) -> Any:
        """
        Get the attention layer module by index.

        Useful for inspection and testing.

        Args:
            layer_idx: Layer index

        Returns:
            The layer module (backend-specific type)
        """
        pass

    def validate_adapter_compatibility(
        self,
        hidden_size: int,
        rank: int,
    ) -> List[str]:
        """
        Validate that an adapter is compatible with this model.

        Args:
            hidden_size: Adapter's hidden size
            rank: Adapter's rank

        Returns:
            List of validation errors (empty if compatible)
        """
        if self._metadata is None:
            return ["Model not initialized"]

        errors = []
        if hidden_size != self._metadata.hidden_size:
            errors.append(
                f"Hidden size mismatch: adapter={hidden_size}, "
                f"model={self._metadata.hidden_size}"
            )

        return errors

    def check_quantization(self) -> bool:
        """
        Check if model uses quantization.

        Returns:
            True if quantization detected (which is NOT allowed)

        Raises:
            ValueError: If quantization is detected
        """
        # Subclasses should implement specific checks
        return False


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

_ADAPTER_REGISTRY: Dict[str, type] = {}


def register_adapter(backend_name: str):
    """Decorator to register an adapter implementation."""
    def decorator(cls):
        _ADAPTER_REGISTRY[backend_name.lower()] = cls
        return cls
    return decorator


def get_adapter(
    backend_name: str,
    model_id: str,
    batch_config: BatchConfig,
    device: str = "cuda:0",
) -> BaseRuntimeAdapter:
    """
    Get an adapter instance for the specified backend.

    Args:
        backend_name: "vllm", "tensorrt_llm", or "sglang"
        model_id: Model identifier
        batch_config: Batch configuration
        device: Device string

    Returns:
        Adapter instance

    Raises:
        ValueError: If backend not registered
    """
    backend_key = backend_name.lower()
    if backend_key not in _ADAPTER_REGISTRY:
        available = list(_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: {available}"
        )

    adapter_cls = _ADAPTER_REGISTRY[backend_key]
    return adapter_cls(model_id, batch_config, device)


def list_available_adapters() -> List[str]:
    """List registered adapter backends."""
    return list(_ADAPTER_REGISTRY.keys())
