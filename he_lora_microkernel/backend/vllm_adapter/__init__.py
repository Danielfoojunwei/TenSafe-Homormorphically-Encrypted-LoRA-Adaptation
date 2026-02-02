"""
vLLM Adapter for HE-LoRA Integration

This module provides integration with vLLM inference engine,
enabling HE-LoRA delta injection at attention projection hooks.

Approach:
  1. Wrap attention projection modules (q_proj, k_proj, v_proj, o_proj)
  2. Inject callback to compute and apply HE-LoRA deltas
  3. Support layer-selective patching based on InsertionConfig

Usage:
    from he_lora_microkernel.backend.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(model_id, batch_config)
    adapter.init()
    adapter.set_insertion_config(config)
    adapter.set_delta_callback(callback)

    kv_cache = adapter.prefill(input_ids)
    logits, states = adapter.decode_one_step(last_token, kv_cache)
"""

from .adapter import VLLMAdapter
from .hooks import (
    AttentionProjectionHook,
    create_projection_hooks,
    install_hooks,
    remove_hooks,
)

__all__ = [
    'VLLMAdapter',
    'AttentionProjectionHook',
    'create_projection_hooks',
    'install_hooks',
    'remove_hooks',
]
