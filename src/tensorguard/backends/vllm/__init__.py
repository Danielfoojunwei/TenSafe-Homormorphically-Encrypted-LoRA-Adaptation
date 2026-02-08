"""TenSafe vLLM Backend Integration.

High-throughput LLM serving with privacy-preserving HE-LoRA injection.

Features:
- PagedAttention for memory-efficient inference
- Continuous batching for maximum throughput
- HE-LoRA injection hooks for encrypted adapter computation
- OpenAI-compatible API
- TSSP package verification
- Audit trail logging

Example:
    ```python
    from tensorguard.backends.vllm import TenSafeVLLMEngine, TenSafeVLLMConfig

    config = TenSafeVLLMConfig(
        model_path="meta-llama/Llama-3-8B",
        tssp_package_path="/path/to/adapter.tssp",
        enable_he_lora=True,
    )

    engine = TenSafeVLLMEngine(config)
    results = await engine.generate(["Hello, world!"])
    ```
"""

from .api import create_openai_router
from .config import TenSafeVLLMConfig
from .engine import TenSafeVLLMEngine
from .hooks import HELoRAHook, HELoRAHookManager

__all__ = [
    "TenSafeVLLMConfig",
    "TenSafeVLLMEngine",
    "HELoRAHook",
    "HELoRAHookManager",
    "create_openai_router",
]
