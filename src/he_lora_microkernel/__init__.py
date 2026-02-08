"""
HE-LoRA Microkernel

A production-grade system for secure LoRA inference using homomorphic encryption.
Treats LoRA adapters as compilable secure microkernels with MOAI-inspired
rotation minimization.

Key Features:
  - HE-LoRA on EVERY generated token
  - CKKS encryption with GPU acceleration
  - Model-agnostic (configurable hidden_size, rank)
  - Cloud-portable (not hardcoded to specific GPU)
  - Rotation-minimal by design

Quick Start:
    from he_lora_microkernel.compiler import (
        LoRAConfig, LoRATargets, CKKSProfile,
        get_profile, compile_schedule,
    )
    from he_lora_microkernel.runtime import HELoRAExecutor
    from he_lora_microkernel.backend.gpu_ckks_backend import BackendType

    # Configure
    config = LoRAConfig(
        hidden_size=4096,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKV,
        batch_size=8,
        max_context_length=2048,
        ckks_profile=CKKSProfile.FAST,
    )

    # Compile
    ckks_params = get_profile(CKKSProfile.FAST)
    schedule = compile_schedule(config, ckks_params)

    # Execute
    executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
    executor.load_weights(A, B, alpha)

    for token in generation:
        delta = executor.execute_token(activations)

Absolute Prohibitions:
  - NO quantization (QLoRA, INT8/INT4)
  - NO integer HE (TFHE)
  - NO CPU-only HE
  - NO bootstrapping
  - NO token skipping
"""

__version__ = '1.0.0'
__author__ = 'TenSafe Team'

# Version info
VERSION_INFO = {
    'version': __version__,
    'ckks_profiles': ['FAST', 'SAFE'],
    'backends': ['SIMULATION', 'HEONGPU', 'FIDESLIB', 'OPENFHE_GPU'],
}
