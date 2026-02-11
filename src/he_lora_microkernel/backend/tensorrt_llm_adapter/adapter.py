"""
TensorRT-LLM Runtime Adapter Implementation

This module implements the BaseRuntimeAdapter interface for TensorRT-LLM,
enabling HE-LoRA delta injection through TensorRT plugins or hybrid mode.

The adapter supports two modes:
  1. Plugin mode: Delta injection within TensorRT execution graph
  2. Hybrid mode: Attention projections in PyTorch, rest in TensorRT
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    import torch

from ..base_adapter import (
    BaseRuntimeAdapter,
    BatchConfig,
    InsertionConfig,
    LayerDeltas,
    LoRATargets,
    ModelMetadata,
    register_adapter,
)
from .engine_builder import EngineConfig, TRTEngineBuilder
from .plugin import HELoRAProjectionPlugin, PluginConfig

logger = logging.getLogger(__name__)


@register_adapter("tensorrt_llm")
class TensorRTLLMAdapter(BaseRuntimeAdapter):
    """
    TensorRT-LLM runtime adapter for HE-LoRA integration.

    Supports delta injection through TensorRT plugins or hybrid mode
    where attention projections run in PyTorch.
    """

    def __init__(
        self,
        model_id: str,
        batch_config: BatchConfig,
        device: str = "cuda:0",
        engine_dir: Optional[str] = None,
        use_hybrid_mode: bool = False,
    ):
        super().__init__(model_id, batch_config, device)

        self._engine_dir = Path(engine_dir) if engine_dir else None
        self._use_hybrid_mode = use_hybrid_mode

        self._engine = None
        self._runtime = None
        self._plugins: Dict[Tuple[int, str], HELoRAProjectionPlugin] = {}
        self._sidecar_config: Optional[Dict] = None

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    def init(self) -> ModelMetadata:
        """Initialize TensorRT-LLM engine."""
        if self._initialized:
            return self._metadata

        logger.info(f"Initializing TensorRT-LLM adapter for model: {self._model_id}")

        try:
            # Try to import TensorRT-LLM
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunner
        except ImportError:
            logger.warning("TensorRT-LLM not installed, using mock implementation")
            return self._init_mock()

        # Load or build engine
        if self._engine_dir and (self._engine_dir / "engine_info.json").exists():
            self._load_existing_engine()
        else:
            self._build_engine()

        # Load sidecar config
        if self._engine_dir:
            config_path = self._engine_dir / "helora_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self._sidecar_config = json.load(f)

        self._initialized = True
        return self._metadata

    def _init_mock(self) -> ModelMetadata:
        """Initialize mock for testing."""
        # Default mock configuration
        self._metadata = ModelMetadata(
            model_id=self._model_id,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=4096,
            architecture='mock_tensorrt_llm',
            has_output_projection=True,
        )

        self._initialized = True
        return self._metadata

    def _load_existing_engine(self) -> None:
        """Load pre-built TensorRT engine."""
        engine_info_path = self._engine_dir / "engine_info.json"

        with open(engine_info_path) as f:
            engine_info = json.load(f)

        self._metadata = ModelMetadata(
            model_id=engine_info.get('model_id', self._model_id),
            num_layers=engine_info.get('num_layers', 32),
            hidden_size=engine_info.get('hidden_size', 4096),
            num_attention_heads=engine_info.get('num_attention_heads', 32),
            head_dim=engine_info.get('hidden_size', 4096) // engine_info.get('num_attention_heads', 32),
            vocab_size=engine_info.get('vocab_size', 32000),
            max_position_embeddings=engine_info.get('max_seq_len', 4096),
            architecture='tensorrt_llm',
            has_output_projection=True,
        )

        logger.info(f"Loaded TensorRT engine from {self._engine_dir}")

    def _build_engine(self) -> None:
        """Build TensorRT engine with HE-LoRA support."""
        if self._engine_dir is None:
            self._engine_dir = Path(f"/tmp/helora_trt_{self._model_id.replace('/', '_')}")

        config = EngineConfig(
            model_id=self._model_id,
            output_dir=str(self._engine_dir),
            max_batch_size=self._batch_config.max_batch_size,
            max_seq_len=self._batch_config.max_context_length,
            enable_plugins=True,
        )

        builder = TRTEngineBuilder(config)
        result = builder.build()

        if not result.success:
            raise RuntimeError(f"Engine build failed: {result.error_message}")

        # Set metadata from config
        self._metadata = ModelMetadata(
            model_id=self._model_id,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            vocab_size=32000,
            max_position_embeddings=config.max_seq_len,
            architecture='tensorrt_llm',
            has_output_projection=True,
        )

        logger.info(f"Built TensorRT engine at {self._engine_dir}")

    def shutdown(self) -> None:
        """Clean up resources."""
        # Detach plugins from shared memory
        for plugin in self._plugins.values():
            plugin.detach_shared_memory()
        self._plugins.clear()

        # Release engine
        self._engine = None
        self._runtime = None
        self._initialized = False

        logger.info("TensorRT-LLM adapter shut down")

    # -------------------------------------------------------------------------
    # INSERTION POINT CONFIGURATION
    # -------------------------------------------------------------------------

    def set_insertion_config(self, config: InsertionConfig) -> None:
        """Configure insertion points and create plugins."""
        super().set_insertion_config(config)

        # Clear existing plugins
        for plugin in self._plugins.values():
            plugin.detach_shared_memory()
        self._plugins.clear()

        # Determine layers and projections
        layer_indices = self.get_patched_layers()
        projections = ['q', 'k', 'v']
        if config.targets == LoRATargets.QKVO:
            projections.append('o')

        # Create plugins
        for layer_idx in layer_indices:
            for proj in projections:
                plugin_config = PluginConfig(
                    layer_idx=layer_idx,
                    projection_type=proj,
                    hidden_size=self._metadata.hidden_size,
                    batch_size=self._batch_config.max_batch_size,
                )
                plugin = HELoRAProjectionPlugin(plugin_config)
                self._plugins[(layer_idx, proj)] = plugin

        logger.info(f"Created {len(self._plugins)} TensorRT plugins")

    def attach_shared_memory(
        self,
        layer_idx: int,
        projection: str,
        shm_name: str,
        shm_offset: int = 0,
    ) -> None:
        """
        Attach a plugin to shared memory for delta data.

        Args:
            layer_idx: Layer index
            projection: Projection type ("q", "k", "v", "o")
            shm_name: Shared memory region name
            shm_offset: Offset within region
        """
        key = (layer_idx, projection)
        if key in self._plugins:
            self._plugins[key].attach_shared_memory(shm_name, shm_offset)
        else:
            logger.warning(f"Plugin not found for layer {layer_idx} {projection}")

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

        # Mock KV cache for testing
        kv_cache = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'input_ids': input_ids.clone(),
        }

        return kv_cache

    def decode_one_step(
        self,
        input_ids_last: 'torch.Tensor',
        kv_cache_handle: Any,
    ) -> Tuple['torch.Tensor', Dict[int, Any]]:
        """Run one decode step with plugin execution."""
        import torch

        batch_size = input_ids_last.shape[0]

        if self._use_hybrid_mode:
            return self._decode_hybrid(input_ids_last, kv_cache_handle)

        # Plugin mode: plugins execute during TensorRT inference
        # For mock, simulate plugin execution
        layer_states = {}

        for layer_idx in range(self._metadata.num_layers):
            layer_states[layer_idx] = {}

            # Mock projection outputs
            for proj in ['q', 'k', 'v', 'o']:
                key = (layer_idx, proj)
                if key in self._plugins:
                    # Plugin would execute during TensorRT inference
                    proj_output = torch.randn(
                        batch_size, 1, self._metadata.hidden_size,
                        dtype=torch.float16,
                    )
                    # Apply plugin (in real TRT, this happens in the engine)
                    plugin = self._plugins[key]
                    output_np = plugin.execute(proj_output.numpy())
                    layer_states[layer_idx][proj] = torch.from_numpy(output_np)

        # Mock logits
        logits = torch.randn(batch_size, self._metadata.vocab_size)
        return logits, layer_states

    def _decode_hybrid(
        self,
        input_ids_last: 'torch.Tensor',
        kv_cache_handle: Any,
    ) -> Tuple['torch.Tensor', Dict[int, Any]]:
        """Hybrid mode: PyTorch attention projections with delta callback."""
        import torch

        batch_size = input_ids_last.shape[0]
        layer_states = {}

        # In hybrid mode, attention projections run in PyTorch
        # allowing direct delta injection via callback
        for layer_idx in range(self._metadata.num_layers):
            layer_states[layer_idx] = {}

            # Mock hidden states
            hidden = torch.randn(
                batch_size, 1, self._metadata.hidden_size,
                dtype=torch.float16,
            )

            for proj in ['q', 'k', 'v', 'o']:
                # Simulate projection
                output = torch.randn_like(hidden)

                # Apply delta if callback set
                if self._delta_callback is not None:
                    delta = self._delta_callback(layer_idx, proj, hidden)
                    if delta is not None:
                        output = output + delta

                layer_states[layer_idx][proj] = output

        logits = torch.randn(batch_size, self._metadata.vocab_size)
        return logits, layer_states

    def apply_deltas(
        self,
        layer_idx: int,
        deltas: LayerDeltas,
    ) -> None:
        """Apply deltas through plugins."""

        delta_map = {
            'q': deltas.dq,
            'k': deltas.dk,
            'v': deltas.dv,
            'o': deltas.do,
        }

        for proj, delta in delta_map.items():
            if delta is not None:
                key = (layer_idx, proj)
                if key in self._plugins:
                    self._plugins[key].set_delta_buffer(delta.numpy())

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    def get_layer_module(self, layer_idx: int) -> Any:
        """Get layer module (returns plugin info in TRT mode)."""
        plugins = {
            proj: self._plugins.get((layer_idx, proj))
            for proj in ['q', 'k', 'v', 'o']
        }
        return {'layer_idx': layer_idx, 'plugins': plugins}

    def check_quantization(self) -> bool:
        """Check for quantization in engine config."""
        if self._sidecar_config:
            dtype = self._sidecar_config.get('dtype', 'float16')
            if dtype in ['int8', 'int4', 'fp8']:
                return True
        return False

    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get statistics from plugins."""
        return {
            'plugin_count': len(self._plugins),
            'layers': list(set(k[0] for k in self._plugins.keys())),
            'projections': list(set(k[1] for k in self._plugins.keys())),
        }
