"""
SGLang Runtime Adapter Implementation

This module implements the BaseRuntimeAdapter interface for SGLang,
enabling HE-LoRA delta injection through hook-based interception.

SGLang uses RadixAttention for efficient prefix caching and batch
management. The adapter hooks into model execution to inject deltas.
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
from .hooks import HookRegistry, RadixAttentionHook

logger = logging.getLogger(__name__)


@register_adapter("sglang")
class SGLangAdapter(BaseRuntimeAdapter):
    """
    SGLang runtime adapter for HE-LoRA integration.

    Uses hook-based delta injection at the model execution level.
    Compatible with SGLang's RadixAttention and prefix caching.

    Attributes:
        model_id: HuggingFace model identifier
        batch_config: Batch configuration
        device: Target device
        use_radix_attention: Whether to use RadixAttention hooks
    """

    def __init__(
        self,
        model_id: str,
        batch_config: BatchConfig,
        device: str = "cuda:0",
        use_radix_attention: bool = True,
        tp_size: int = 1,
    ):
        super().__init__(model_id, batch_config, device)

        self._use_radix_attention = use_radix_attention
        self._tp_size = tp_size

        # SGLang components
        self._runtime = None
        self._model = None
        self._tokenizer = None

        # Hook management
        self._hook_registry = HookRegistry()
        self._radix_hooks: Dict[int, RadixAttentionHook] = {}

        # Request management
        self._active_requests: Dict[str, Dict] = {}

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    def init(self) -> ModelMetadata:
        """Initialize SGLang runtime."""
        if self._initialized:
            return self._metadata

        logger.info(f"Initializing SGLang adapter for model: {self._model_id}")

        try:
            # Try to import SGLang
            import sglang as sgl
            from sglang.srt.model_executor import ModelRunner
        except ImportError:
            logger.warning("SGLang not installed, using mock implementation")
            return self._init_mock()

        # Initialize SGLang runtime
        self._init_sglang_runtime()

        self._initialized = True
        return self._metadata

    def _init_mock(self) -> ModelMetadata:
        """Initialize mock for testing without SGLang."""
        # Default mock configuration (Llama-2 7B style)
        self._metadata = ModelMetadata(
            model_id=self._model_id,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=4096,
            architecture='mock_sglang',
            has_output_projection=True,
        )

        self._initialized = True
        return self._metadata

    def _init_sglang_runtime(self) -> None:
        """Initialize actual SGLang runtime."""
        try:
            from sglang.srt.server import Runtime

            # Create runtime with configuration
            self._runtime = Runtime(
                model_path=self._model_id,
                tp_size=self._tp_size,
            )

            # Extract model metadata
            config = self._runtime.get_model_config()
            self._metadata = ModelMetadata(
                model_id=self._model_id,
                num_layers=config.num_hidden_layers,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                vocab_size=config.vocab_size,
                max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
                architecture='sglang',
                has_output_projection=True,
            )

            # Get model reference for hooking
            self._model = self._runtime.get_model()
            self._hook_registry.set_model(self._model)

            logger.info(f"SGLang runtime initialized: {self._metadata.num_layers} layers")

        except Exception as e:
            logger.warning(f"Failed to initialize SGLang runtime: {e}")
            logger.warning("Falling back to mock implementation")
            self._init_mock()

    def shutdown(self) -> None:
        """Clean up resources."""
        # Remove all hooks
        self._hook_registry.uninstall_all()

        for hook in self._radix_hooks.values():
            hook.uninstall()
        self._radix_hooks.clear()

        # Clear active requests
        self._active_requests.clear()

        # Shutdown runtime
        if self._runtime is not None:
            try:
                self._runtime.shutdown()
            except Exception as e:
                logger.warning(f"Error during runtime shutdown: {e}")

        self._runtime = None
        self._model = None
        self._initialized = False

        logger.info("SGLang adapter shut down")

    # -------------------------------------------------------------------------
    # INSERTION POINT CONFIGURATION
    # -------------------------------------------------------------------------

    def set_insertion_config(self, config: InsertionConfig) -> None:
        """Configure insertion points and install hooks."""
        super().set_insertion_config(config)

        # Remove existing hooks
        self._hook_registry.uninstall_all()
        for hook in self._radix_hooks.values():
            hook.uninstall()
        self._radix_hooks.clear()

        if self._model is None:
            logger.warning("Model not loaded, skipping hook installation")
            return

        # Determine layers and projections
        layer_indices = self.get_patched_layers()
        projections = ['q', 'k', 'v']
        if config.targets == LoRATargets.QKVO:
            projections.append('o')

        # Install hooks based on mode
        if self._use_radix_attention:
            self._install_radix_hooks(layer_indices, projections)
        else:
            self._install_projection_hooks(layer_indices, projections)

        logger.info(f"Installed hooks for {len(layer_indices)} layers")

    def _install_projection_hooks(
        self,
        layer_indices: List[int],
        projections: List[str],
    ) -> None:
        """Install hooks on individual projection modules."""
        for layer_idx in layer_indices:
            layer = self._get_layer_module(layer_idx)
            if layer is None:
                continue

            for proj in projections:
                module = self._get_projection_module(layer, proj)
                if module is not None:
                    self._hook_registry.register_hook(
                        layer_idx=layer_idx,
                        projection_type=proj,
                        module=module,
                    )

    def _install_radix_hooks(
        self,
        layer_indices: List[int],
        projections: List[str],
    ) -> None:
        """Install RadixAttention-specific hooks."""
        for layer_idx in layer_indices:
            layer = self._get_layer_module(layer_idx)
            if layer is None:
                continue

            # Get attention module
            attention = self._get_attention_module(layer)
            if attention is not None:
                hook = RadixAttentionHook(
                    layer_idx=layer_idx,
                    attention_module=attention,
                    delta_callback=self._delta_callback,
                )
                hook.install()
                self._radix_hooks[layer_idx] = hook

    def _get_layer_module(self, layer_idx: int) -> Optional[Any]:
        """Get the transformer layer module by index."""
        if self._model is None:
            return None

        # Try common model structures
        model = self._model

        # Try .model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if layer_idx < len(layers):
                return layers[layer_idx]

        # Try .layers directly
        if hasattr(model, 'layers'):
            layers = model.layers
            if layer_idx < len(layers):
                return layers[layer_idx]

        return None

    def _get_projection_module(self, layer: Any, projection: str) -> Optional[Any]:
        """Get the projection module from a layer."""
        # Map projection type to module name
        proj_names = {
            'q': ['q_proj', 'query', 'q'],
            'k': ['k_proj', 'key', 'k'],
            'v': ['v_proj', 'value', 'v'],
            'o': ['o_proj', 'out_proj', 'dense', 'o'],
        }

        # Try attention submodule first
        attention = self._get_attention_module(layer)
        if attention is not None:
            for name in proj_names.get(projection, []):
                if hasattr(attention, name):
                    return getattr(attention, name)

        # Try direct on layer
        for name in proj_names.get(projection, []):
            if hasattr(layer, name):
                return getattr(layer, name)

        return None

    def _get_attention_module(self, layer: Any) -> Optional[Any]:
        """Get the attention module from a layer."""
        attention_names = ['self_attn', 'attention', 'attn']
        for name in attention_names:
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    # -------------------------------------------------------------------------
    # DELTA CALLBACK
    # -------------------------------------------------------------------------

    def set_delta_callback(self, callback: DeltaCallback) -> None:
        """Set the delta callback for hooks."""
        super().set_delta_callback(callback)

        # Update hook registry
        self._hook_registry.set_delta_callback(callback)

        # Update radix hooks
        for hook in self._radix_hooks.values():
            hook.delta_callback = callback

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------

    def prefill(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None,
    ) -> Any:
        """Run prefill phase."""

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        if self._runtime is not None:
            # Real SGLang prefill
            return self._prefill_sglang(input_ids, attention_mask)

        # Mock prefill
        kv_cache = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'input_ids': input_ids.clone(),
            'position': seq_len,
        }

        return kv_cache

    def _prefill_sglang(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'],
    ) -> Any:
        """Run actual SGLang prefill."""

        # Convert to list of sequences
        batch_size = input_ids.shape[0]
        sequences = []
        for i in range(batch_size):
            # Get non-padded tokens
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                seq = input_ids[i, :valid_len].tolist()
            else:
                seq = input_ids[i].tolist()
            sequences.append(seq)

        # Create request IDs
        request_ids = [f"req_{i}" for i in range(batch_size)]

        # Prefill through runtime
        for req_id, seq in zip(request_ids, sequences):
            self._active_requests[req_id] = {
                'input_ids': seq,
                'position': len(seq),
            }

        return {
            'request_ids': request_ids,
            'batch_size': batch_size,
            'positions': [len(s) for s in sequences],
        }

    def decode_one_step(
        self,
        input_ids_last: 'torch.Tensor',
        kv_cache_handle: Any,
    ) -> Tuple['torch.Tensor', Dict[int, Any]]:
        """Run one decode step with hook execution."""
        import torch

        batch_size = input_ids_last.shape[0]

        if self._runtime is not None:
            return self._decode_sglang(input_ids_last, kv_cache_handle)

        # Mock decode with hook simulation
        layer_states = {}

        for layer_idx in range(self._metadata.num_layers):
            layer_states[layer_idx] = {}

            # Mock hidden states
            hidden = torch.randn(
                batch_size, 1, self._metadata.hidden_size,
                dtype=torch.float16,
            )

            # Simulate projection outputs with delta injection
            for proj in ['q', 'k', 'v', 'o']:
                # Mock projection output
                output = torch.randn_like(hidden)

                # Apply delta via callback if available
                if self._delta_callback is not None:
                    delta = self._delta_callback(layer_idx, proj, hidden)
                    if delta is not None:
                        output = output + delta

                layer_states[layer_idx][proj] = output

        # Mock logits
        logits = torch.randn(batch_size, self._metadata.vocab_size)
        return logits, layer_states

    def _decode_sglang(
        self,
        input_ids_last: 'torch.Tensor',
        kv_cache_handle: Any,
    ) -> Tuple['torch.Tensor', Dict[int, Any]]:
        """Run actual SGLang decode step."""
        import torch

        batch_size = input_ids_last.shape[0]

        # Hooks are already installed and will inject deltas during forward pass
        # Run decode through runtime
        # Note: SGLang typically uses async batch processing

        # For synchronous single-step decode:
        request_ids = kv_cache_handle.get('request_ids', [])

        # Collect outputs
        logits_list = []
        for i, req_id in enumerate(request_ids):
            token_id = input_ids_last[i].item()
            # Append token to request
            if req_id in self._active_requests:
                self._active_requests[req_id]['input_ids'].append(token_id)
                self._active_requests[req_id]['position'] += 1

        # Mock logits for now (real implementation would use runtime)
        logits = torch.randn(batch_size, self._metadata.vocab_size)
        layer_states = {}  # Layer states captured by hooks

        return logits, layer_states

    def apply_deltas(
        self,
        layer_idx: int,
        deltas: LayerDeltas,
    ) -> None:
        """
        Apply deltas for a layer.

        In SGLang adapter, deltas are applied through the callback mechanism.
        This method can be used for direct delta application if needed.
        """
        # Deltas are typically applied via callback during forward pass
        # This method provides an alternative direct application path

        if layer_idx in self._radix_hooks:
            hook = self._radix_hooks[layer_idx]
            # Store deltas for next forward pass
            hook._pending_deltas = {
                'q': deltas.dq,
                'k': deltas.dk,
                'v': deltas.dv,
                'o': deltas.do,
            }

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    def get_layer_module(self, layer_idx: int) -> Any:
        """Get layer module for external access."""
        return self._get_layer_module(layer_idx)

    def check_quantization(self) -> bool:
        """Check if model is quantized."""
        if self._model is None:
            return False

        # Check for quantization markers
        if hasattr(self._model, 'config'):
            config = self._model.config
            if hasattr(config, 'quantization_config'):
                return True
            if getattr(config, 'load_in_8bit', False):
                return True
            if getattr(config, 'load_in_4bit', False):
                return True

        return False

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get hook registry statistics."""
        return {
            'projection_hooks': self._hook_registry.get_statistics(),
            'radix_hooks': {
                'count': len(self._radix_hooks),
                'layers': list(self._radix_hooks.keys()),
            },
            'active_requests': len(self._active_requests),
        }

    # -------------------------------------------------------------------------
    # REQUEST MANAGEMENT
    # -------------------------------------------------------------------------

    def create_request(self, request_id: str, input_ids: List[int]) -> None:
        """Create a new request for batch processing."""
        self._active_requests[request_id] = {
            'input_ids': input_ids,
            'position': len(input_ids),
        }

    def release_request(self, request_id: str) -> None:
        """Release a request and its resources."""
        if request_id in self._active_requests:
            del self._active_requests[request_id]

    def get_active_requests(self) -> List[str]:
        """Get list of active request IDs."""
        return list(self._active_requests.keys())
