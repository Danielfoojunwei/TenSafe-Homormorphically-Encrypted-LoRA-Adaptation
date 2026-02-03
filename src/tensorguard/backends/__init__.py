"""TenSafe Backend Integrations.

This module provides integration adapters for high-performance inference engines:
- vLLM: High-throughput LLM serving with PagedAttention
- LoRAX: Multi-adapter serving at scale
- SGLang: Concurrent token generation
- TensorRT-LLM: NVIDIA optimized inference

Each backend preserves TenSafe's privacy guarantees (HE-LoRA, audit trails, TSSP).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vllm import TenSafeVLLMConfig, TenSafeVLLMEngine

__all__ = ["TenSafeVLLMEngine", "TenSafeVLLMConfig"]
