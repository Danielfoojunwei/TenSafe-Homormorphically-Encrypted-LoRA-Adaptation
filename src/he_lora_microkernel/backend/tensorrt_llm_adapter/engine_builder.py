"""
TensorRT-LLM Engine Builder with HE-LoRA Support

This module provides utilities for building TensorRT-LLM engines
with HE-LoRA delta injection points configured.

The builder:
  1. Takes a model configuration
  2. Generates TensorRT engine with plugin insertion points
  3. Outputs engine file + sidecar config for delta injection
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for TensorRT engine build."""
    model_id: str
    output_dir: str

    # Model configuration
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    max_batch_size: int = 8
    max_seq_len: int = 2048

    # HE-LoRA configuration
    helora_layers: List[int] = field(default_factory=list)  # Empty = all
    helora_targets: str = "qkv"  # "qkv" or "qkvo"

    # Build options
    dtype: str = "float16"
    enable_plugins: bool = True
    use_hybrid_mode: bool = False  # Use PyTorch for attention projections


@dataclass
class EngineBuildResult:
    """Result of engine build."""
    success: bool
    engine_path: Optional[str] = None
    config_path: Optional[str] = None
    error_message: Optional[str] = None

    # Plugin info
    plugin_count: int = 0
    plugin_layers: List[int] = field(default_factory=list)


class TRTEngineBuilder:
    """
    Builder for TensorRT-LLM engines with HE-LoRA support.

    This class generates TensorRT engines that include plugin
    insertion points for delta injection.
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize engine builder.

        Args:
            config: Engine configuration
        """
        self.config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> EngineBuildResult:
        """
        Build TensorRT engine with HE-LoRA plugin support.

        Returns:
            EngineBuildResult with paths and status
        """
        logger.info(f"Building TensorRT engine for {self.config.model_id}")

        # Determine which layers get plugins
        if self.config.helora_layers:
            plugin_layers = self.config.helora_layers
        else:
            plugin_layers = list(range(self.config.num_layers))

        # Determine projections
        projections = ['q', 'k', 'v']
        if self.config.helora_targets == 'qkvo':
            projections.append('o')

        try:
            # Generate sidecar config
            sidecar_config = self._generate_sidecar_config(plugin_layers, projections)
            config_path = self._output_dir / "helora_config.json"
            with open(config_path, 'w') as f:
                json.dump(sidecar_config, f, indent=2)

            # In production, this would call TensorRT-LLM build APIs
            # For now, generate a mock engine info file
            engine_info = self._generate_mock_engine_info(plugin_layers, projections)
            engine_path = self._output_dir / "engine_info.json"
            with open(engine_path, 'w') as f:
                json.dump(engine_info, f, indent=2)

            logger.info(f"Engine build complete: {len(plugin_layers)} layers with plugins")

            return EngineBuildResult(
                success=True,
                engine_path=str(engine_path),
                config_path=str(config_path),
                plugin_count=len(plugin_layers) * len(projections),
                plugin_layers=plugin_layers,
            )

        except Exception as e:
            logger.error(f"Engine build failed: {e}")
            return EngineBuildResult(
                success=False,
                error_message=str(e),
            )

    def _generate_sidecar_config(
        self,
        plugin_layers: List[int],
        projections: List[str],
    ) -> Dict[str, Any]:
        """Generate sidecar configuration for runtime."""
        config = {
            'model_id': self.config.model_id,
            'engine_version': '1.0',
            'helora_enabled': True,
            'helora_config': {
                'targets': self.config.helora_targets,
                'layers': plugin_layers,
                'projections': projections,
            },
            'tensor_layouts': {},
            'shm_config': {
                'region_name_template': 'helora_delta_{layer}_{proj}',
                'dtype': 'float16',
                'alignment': 256,  # CUDA memory alignment
            },
        }

        # Generate tensor layouts for each plugin
        for layer_idx in plugin_layers:
            for proj in projections:
                key = f"layer{layer_idx}_{proj}"
                config['tensor_layouts'][key] = {
                    'layer_idx': layer_idx,
                    'projection': proj,
                    'shape': [
                        self.config.max_batch_size,
                        1,  # seq_len (one token at a time for decode)
                        self.config.hidden_size,
                    ],
                    'dtype': 'float16',
                    'size_bytes': (
                        self.config.max_batch_size *
                        self.config.hidden_size * 2  # float16 = 2 bytes
                    ),
                }

        return config

    def _generate_mock_engine_info(
        self,
        plugin_layers: List[int],
        projections: List[str],
    ) -> Dict[str, Any]:
        """Generate mock engine info for testing."""
        return {
            'format': 'mock_trt_engine',
            'model_id': self.config.model_id,
            'dtype': self.config.dtype,
            'max_batch_size': self.config.max_batch_size,
            'max_seq_len': self.config.max_seq_len,
            'num_layers': self.config.num_layers,
            'hidden_size': self.config.hidden_size,
            'num_attention_heads': self.config.num_attention_heads,
            'plugins': [
                {
                    'name': f'HELoRAProjection_{layer}_{proj}',
                    'layer_idx': layer,
                    'projection': proj,
                }
                for layer in plugin_layers
                for proj in projections
            ],
        }


def build_trt_engine(
    model_id: str,
    output_dir: str,
    helora_layers: Optional[List[int]] = None,
    helora_targets: str = "qkv",
    max_batch_size: int = 8,
    max_seq_len: int = 2048,
) -> EngineBuildResult:
    """
    Convenience function to build a TensorRT engine with HE-LoRA support.

    Args:
        model_id: Model identifier
        output_dir: Directory for output files
        helora_layers: Layers to add plugins to (None = all)
        helora_targets: "qkv" or "qkvo"
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length

    Returns:
        EngineBuildResult
    """
    # Get model config (would query from HuggingFace in production)
    # For now, use defaults for Llama-like models
    config = EngineConfig(
        model_id=model_id,
        output_dir=output_dir,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        helora_layers=helora_layers or [],
        helora_targets=helora_targets,
    )

    builder = TRTEngineBuilder(config)
    return builder.build()
