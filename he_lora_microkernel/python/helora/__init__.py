"""
HE-LoRA Microkernel Python Interface

This package provides a high-level Python interface for the HE-LoRA
microkernel system. It enables secure LoRA inference with homomorphic
encryption on every generated token.

Key features:
  - Simple configuration with HELoRAConfig
  - One-line compilation with compile_lora
  - Easy execution with HELoRARunner
  - Built-in telemetry and CI enforcement

Quick start:
    from helora import HELoRAConfig, compile_lora, HELoRARunner

    # Configure
    config = HELoRAConfig(
        hidden_size=4096,
        lora_rank=16,
        batch_size=8,
    )

    # Compile
    result = compile_lora(config, A, B, alpha)

    # Run
    runner = HELoRARunner(config, A, B, alpha)
    for token in generation:
        delta = runner(activations)
"""

# Configuration
from .config import (
    HELoRAConfig,
    PerformanceProfile,
    llama_7b_config,
    llama_13b_config,
    llama_70b_config,
    mistral_7b_config,
)

# Compilation
from .compile import (
    CompilationResult,
    compile_lora,
    compile_and_save,
    validate_compilation,
    recompile_for_batch_size,
    compare_configurations,
)

# Execution
from .run import (
    HELoRARunner,
    create_executor,
    run_inference,
    benchmark_configuration,
)


__all__ = [
    # Configuration
    'HELoRAConfig',
    'PerformanceProfile',
    'llama_7b_config',
    'llama_13b_config',
    'llama_70b_config',
    'mistral_7b_config',
    # Compilation
    'CompilationResult',
    'compile_lora',
    'compile_and_save',
    'validate_compilation',
    'recompile_for_batch_size',
    'compare_configurations',
    # Execution
    'HELoRARunner',
    'create_executor',
    'run_inference',
    'benchmark_configuration',
]

__version__ = '1.0.0'
