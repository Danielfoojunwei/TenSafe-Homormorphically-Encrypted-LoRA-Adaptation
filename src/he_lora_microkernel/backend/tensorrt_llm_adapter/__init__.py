"""
TensorRT-LLM Adapter for HE-LoRA Integration

This module provides integration with TensorRT-LLM inference engine,
enabling HE-LoRA delta injection through TensorRT plugins.

Approach:
  1. Build TensorRT engine with custom plugin for attention projections
  2. Plugin receives delta tensors via shared memory
  3. Delta is added to projection output within the engine

For models where plugin injection is complex, a hybrid approach is used:
  - Run most of the model in TensorRT-LLM
  - Execute attention projections in PyTorch with delta injection
  - Transfer data between engines via pinned memory

Usage:
    from he_lora_microkernel.backend.tensorrt_llm_adapter import TensorRTLLMAdapter

    adapter = TensorRTLLMAdapter(model_id, batch_config)
    adapter.init()
    adapter.set_insertion_config(config)
"""

from .adapter import TensorRTLLMAdapter
from .plugin import HELoRAProjectionPlugin
from .engine_builder import TRTEngineBuilder

__all__ = [
    'TensorRTLLMAdapter',
    'HELoRAProjectionPlugin',
    'TRTEngineBuilder',
]
