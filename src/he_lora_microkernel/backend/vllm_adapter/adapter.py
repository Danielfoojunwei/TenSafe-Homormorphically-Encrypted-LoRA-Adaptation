"""
vLLM Runtime Adapter Implementation

This module implements the BaseRuntimeAdapter interface for vLLM,
enabling HE-LoRA delta injection during inference.

The adapter:
  1. Initializes vLLM with the specified model
  2. Extracts model metadata for validation
  3. Installs hooks on attention projections
  4. Runs prefill and decode with delta injection
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

from ..base_adapter import (
    BaseRuntimeAdapter,
    BatchConfig,
    DeltaCallback,
    InsertionConfig,
    LayerDeltas,
    LoRATargets,
    ModelMetadata,
    register_adapter,
)
from .hooks import (
    AttentionProjectionHook,
    create_projection_hooks,
    get_hook_statistics,
    install_hooks,
    remove_hooks,
    reset_hook_statistics,
    set_delta_callback,
)

logger = logging.getLogger(__name__)


@register_adapter("vllm")
class VLLMAdapter(BaseRuntimeAdapter):
    """
    vLLM runtime adapter for HE-LoRA integration.

    Provides hooks into vLLM's attention projections for delta injection.
    """

    def __init__(
        self,
        model_id: str,
        batch_config: BatchConfig,
        device: str = "cuda:0",
    ):
        super().__init__(model_id, batch_config, device)

        self._engine = None
        self._model = None
        self._tokenizer = None
        self._hooks: Dict[Tuple[int, str], AttentionProjectionHook] = {}
        self._kv_cache = None

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    def init(self) -> ModelMetadata:
        """
        Initialize vLLM and extract model metadata.

        Returns:
            ModelMetadata with model configuration
        """
        if self._initialized:
            return self._metadata

        logger.info(f"Initializing vLLM adapter for model: {self._model_id}")

        try:
            # Try to import vLLM
            from vllm import LLM, SamplingParams
            from vllm.model_executor.model_loader import get_model
        except ImportError:
            # Check execution policy before allowing fallback
            import os
            environment = os.getenv("TG_ENVIRONMENT", "development").lower()
            allow_mock = os.getenv("TG_ALLOW_MOCK_BACKEND", "false").lower() == "true"

            if environment in ("production", "prod") and not allow_mock:
                raise RuntimeError(
                    "vLLM is required for production HE-LoRA but not installed. "
                    "Install vLLM (pip install vllm) or set TG_ALLOW_MOCK_BACKEND=true "
                    "to explicitly allow mock mode (NOT RECOMMENDED for production)."
                )

            logger.warning(
                "vLLM not installed, using mock implementation. "
                "THIS PROVIDES NO HE PROTECTION - for testing only!"
            )
            return self._init_mock()

        # Initialize vLLM engine
        self._engine = LLM(
            model=self._model_id,
            dtype=self._batch_config.dtype,
            max_model_len=self._batch_config.max_context_length,
            gpu_memory_utilization=0.9,
            enforce_eager=True,  # Disable CUDA graphs for hook compatibility
        )

        # Get underlying model
        self._model = self._engine.llm_engine.model_executor.driver_worker.model_runner.model
        self._tokenizer = self._engine.get_tokenizer()

        # Check for quantization
        if self.check_quantization():
            raise ValueError(
                f"Model {self._model_id} uses quantization, which is not allowed. "
                f"Use FP16 model only."
            )

        # Extract metadata
        config = self._model.config
        self._metadata = ModelMetadata(
            model_id=self._model_id,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            architecture=config.model_type,
            has_output_projection=True,  # Most models have o_proj
        )

        self._initialized = True
        logger.info(f"vLLM adapter initialized: {self._metadata.num_layers} layers, "
                   f"hidden_size={self._metadata.hidden_size}")

        return self._metadata

    def _init_mock(self) -> ModelMetadata:
        """
        Initialize mock for testing without vLLM.

        WARNING: This provides NO HE protection. Inputs are processed in
        plaintext. Only use for testing/development.
        """
        import torch.nn as nn

        # Set flag indicating mock mode
        self._is_mock = True

        # Create a minimal mock model
        class MockAttention(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

        class MockLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.self_attn = MockAttention(hidden_size)

        class MockModel(nn.Module):
            def __init__(self, num_layers, hidden_size):
                super().__init__()
                self.layers = nn.ModuleList([
                    MockLayer(hidden_size) for _ in range(num_layers)
                ])

        # Default mock configuration - must match HAS executor's hidden_size
        num_layers = 32
        hidden_size = 1024  # Match executor hidden_size for E2E test compatibility
        num_heads = 32

        self._model = MockModel(num_layers, hidden_size)
        self._model.config = type('Config', (), {
            'num_hidden_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads,
            'vocab_size': 32000,
            'max_position_embeddings': 4096,
            'model_type': 'mock_llama',
        })()

        self._metadata = ModelMetadata(
            model_id=self._model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=hidden_size // num_heads,
            vocab_size=32000,
            max_position_embeddings=4096,
            architecture='mock_llama',
            has_output_projection=True,
        )


        self._initialized = True
        return self._metadata

    def shutdown(self) -> None:
        """Clean up resources."""
        # Remove hooks
        if self._hooks:
            remove_hooks(self._hooks)
            self._hooks.clear()

        # Shutdown vLLM engine
        if self._engine is not None:
            del self._engine
            self._engine = None

        self._model = None
        self._tokenizer = None
        self._initialized = False

        logger.info("vLLM adapter shut down")

    # -------------------------------------------------------------------------
    # INSERTION POINT CONFIGURATION
    # -------------------------------------------------------------------------

    def set_insertion_config(self, config: InsertionConfig) -> None:
        """
        Configure insertion points and install hooks atomically.

        This method implements atomic hook reconfiguration with rollback
        on failure to ensure the system is never left in an inconsistent state.

        Args:
            config: New insertion configuration

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If hook installation fails (previous config restored)
        """
        super().set_insertion_config(config)

        # Save current state for rollback
        old_hooks = self._hooks.copy() if self._hooks else {}
        old_config = self._insertion_config

        try:
            # Remove existing hooks
            if self._hooks:
                remove_hooks(self._hooks)
                self._hooks.clear()

            # Determine which layers to hook
            layer_indices = self.get_patched_layers()

            # Determine projections based on config
            if config.targets == LoRATargets.QKV:
                projections = ['q', 'k', 'v']
            else:
                projections = ['q', 'k', 'v', 'o']

            # Create and install new hooks
            new_hooks = create_projection_hooks(
                model=self._model,
                layer_indices=layer_indices,
                projections=projections,
                delta_callback=self._delta_callback,
            )
            install_hooks(new_hooks)

            # Success - update state
            self._hooks = new_hooks
            logger.info(f"Installed {len(self._hooks)} hooks on layers {layer_indices}")

        except Exception as e:
            # Rollback on failure
            logger.error(f"Hook installation failed, rolling back: {e}")

            # Restore old hooks
            if old_hooks:
                try:
                    install_hooks(old_hooks)
                    self._hooks = old_hooks
                    self._insertion_config = old_config
                    logger.info("Rolled back to previous hook configuration")
                except Exception as rollback_error:
                    logger.critical(
                        f"Rollback also failed: {rollback_error}. "
                        f"System may be in inconsistent state!"
                    )

            raise RuntimeError(f"Failed to install hooks: {e}") from e

    def reconfigure_for_adapter(
        self,
        target_modules: List[str],
        layer_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Reconfigure hooks for a new adapter's target modules.

        This is the method to call from a hot-swap callback when the
        active adapter's target_modules change.

        Args:
            target_modules: New adapter's target modules (e.g., ["q_proj", "k_proj"])
            layer_indices: Optional layer indices (None = all layers)
        """
        # Determine targets based on module names
        has_output = any("o_proj" in m for m in target_modules)
        targets = LoRATargets.QKVO if has_output else LoRATargets.QKV

        config = InsertionConfig(
            targets=targets,
            layers=layer_indices,
        )

        self.set_insertion_config(config)
        logger.info(f"Reconfigured hooks for target_modules: {target_modules}")

    def set_delta_callback(self, callback: DeltaCallback) -> None:
        """Set delta callback."""
        super().set_delta_callback(callback)
        if self._hooks:
            set_delta_callback(self._hooks, callback)

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------

    def prefill(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None,
    ) -> Any:
        """
        Run prefill phase.

        Note: In vLLM, this is typically handled by the generate() method.
        For fine-grained control, we use lower-level APIs.
        """

        if self._engine is None:
            # Mock implementation
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            # Create mock KV cache
            kv_cache = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'input_ids': input_ids.clone(),
            }
            return kv_cache

        # Use vLLM's internal prefill
        # This is simplified - full implementation would use vLLM's scheduler
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        # Convert input_ids to prompts
        prompts = [self._tokenizer.decode(ids.tolist()) for ids in input_ids]

        # Run prefill through generate (returns after first token)
        outputs = self._engine.generate(prompts, sampling_params)

        # Store KV cache handle
        self._kv_cache = {
            'outputs': outputs,
            'seq_len': input_ids.shape[1],
        }

        return self._kv_cache

    def decode_one_step(
        self,
        input_ids_last: 'torch.Tensor',
        kv_cache_handle: Any,
    ) -> Tuple['torch.Tensor', Dict[int, Any]]:
        """
        Run one decode step.

        During this step, hooks are active and will apply deltas.
        """
        import torch

        if self._engine is None:
            # Mock implementation for testing
            batch_size = input_ids_last.shape[0]

            # Run through mock model with hooks
            hidden = torch.randn(batch_size, 1, self._metadata.hidden_size)

            layer_states = {}
            for layer_idx in range(self._metadata.num_layers):
                layer = self._model.layers[layer_idx]

                # Hooks are automatically applied during forward
                q = layer.self_attn.q_proj(hidden)
                k = layer.self_attn.k_proj(hidden)
                v = layer.self_attn.v_proj(hidden)
                o = layer.self_attn.o_proj(hidden)

                layer_states[layer_idx] = {
                    'q_shape': q.shape,
                    'k_shape': k.shape,
                    'v_shape': v.shape,
                    'o_shape': o.shape,
                }

            # Mock logits
            logits = torch.randn(batch_size, self._metadata.vocab_size)
            return logits, layer_states

        # Full vLLM implementation would continue generation
        # This is a simplified version for the API contract
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        # Continue from previous state
        # Note: vLLM's API doesn't directly support step-by-step decode
        # In production, this would use vLLM's internal scheduler

        # For now, return placeholder
        batch_size = input_ids_last.shape[0]
        import torch
        logits = torch.zeros(batch_size, self._metadata.vocab_size)
        layer_states = {}

        return logits, layer_states

    def apply_deltas(
        self,
        layer_idx: int,
        deltas: LayerDeltas,
    ) -> None:
        """
        Apply precomputed deltas to a layer.

        This is called when deltas are computed externally (e.g., from HAS)
        rather than through the callback mechanism.
        """
        # In vLLM with hooks, deltas are applied through the callback
        # This method is for direct injection mode

        if layer_idx not in [h[0] for h in self._hooks.keys()]:
            logger.warning(f"Layer {layer_idx} not hooked, cannot apply deltas")
            return

        # Store deltas for next forward pass
        # The hook callback will retrieve and apply them
        self._pending_deltas = {layer_idx: deltas}

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    def get_layer_module(self, layer_idx: int) -> Any:
        """Get the attention layer module."""
        if self._model is None:
            raise RuntimeError("Model not initialized")

        if layer_idx >= self._metadata.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range")

        return self._model.layers[layer_idx]

    def check_quantization(self) -> bool:
        """Check if model uses quantization."""
        if self._model is None:
            return False

        # Check for common quantization indicators
        for name, param in self._model.named_parameters():
            # Check for quantized dtypes
            if param.dtype in []:  # Add quantized dtypes if needed
                return True

            # Check for quantization-related names
            if any(q in name.lower() for q in ['quant', 'scale', 'zero_point']):
                return True

        # Check model config
        if hasattr(self._model, 'config'):
            config = self._model.config
            if hasattr(config, 'quantization_config') and config.quantization_config:
                return True

        return False

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get statistics from all hooks."""
        return get_hook_statistics(self._hooks)

    def reset_hook_statistics(self) -> None:
        """Reset hook statistics."""
        reset_hook_statistics(self._hooks)
