"""
Gated LoRA Implementation for Hybrid CKKS-TFHE Compiler

Provides reference implementation of:
y = Wx + g(x) * Delta(x)

Where:
- Delta(x) = B(Ax) computed in CKKS with MOAI
- g(x) computed in TFHE via programmable bootstrapping
"""

from .compiler import (
    GatedLoRACompiler,
    GatedLoRAConfig,
    compile_gated_lora,
)
from .executor import (
    ExecutionResult,
    GatedLoRAExecutor,
    execute_gated_lora,
    plaintext_gated_lora,
)

__all__ = [
    "GatedLoRACompiler",
    "GatedLoRAConfig",
    "compile_gated_lora",
    "GatedLoRAExecutor",
    "ExecutionResult",
    "execute_gated_lora",
    "plaintext_gated_lora",
]
