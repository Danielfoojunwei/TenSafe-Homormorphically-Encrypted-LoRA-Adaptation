"""
SGLang Runtime Adapter for HE-LoRA Integration

SGLang uses RadixAttention for efficient batch management.
This adapter hooks into the model execution step to inject HE-LoRA deltas.
"""

from .adapter import SGLangAdapter
from .hooks import SGLangModelHook, HookRegistry

__all__ = [
    'SGLangAdapter',
    'SGLangModelHook',
    'HookRegistry',
]
