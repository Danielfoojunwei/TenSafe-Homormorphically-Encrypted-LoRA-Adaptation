"""
HE-LoRA Microkernel

A production-grade system for secure LoRA inference using homomorphic encryption.
Treats LoRA adapters as compilable secure microkernels with MOAI-inspired
rotation minimization.

Key Features:
  - HE-LoRA on EVERY generated token
  - CKKS encryption with GPU acceleration for linear operations
  - HYBRID CKKS-TFHE mode for non-linear adapters (gated LoRA)
  - Model-agnostic (configurable hidden_size, rank)
  - Cloud-portable (not hardcoded to specific GPU)
  - Rotation-minimal by design (MOAI optimization)

Supported Modes:
  - CKKS-only: Linear LoRA adapters (matmul operations)
  - Hybrid CKKS-TFHE: Gated/non-linear adapters (with discrete gates)

For linear LoRA (CKKS-only):
    from he_lora_microkernel.compiler import (
        LoRAConfig, LoRATargets, CKKSProfile,
        get_profile, compile_schedule,
    )
    from he_lora_microkernel.runtime import HELoRAExecutor
    from he_lora_microkernel.backend.gpu_ckks_backend import BackendType

    config = LoRAConfig(
        hidden_size=4096,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKV,
        batch_size=8,
        max_context_length=2048,
        ckks_profile=CKKSProfile.FAST,
    )
    ckks_params = get_profile(CKKSProfile.FAST)
    schedule = compile_schedule(config, ckks_params)
    executor = HELoRAExecutor(schedule, BackendType.SIMULATION)

For gated LoRA (Hybrid CKKS-TFHE):
    from he_lora_microkernel.hybrid_compiler import (
        GatedLoRAConfig, GatedLoRACompiler, GatedLoRAExecutor,
    )

    config = GatedLoRAConfig(
        hidden_size=4096,
        lora_rank=16,
        gate_type="step",  # or "sign"
    )
    compiler = GatedLoRACompiler(config)
    program, plan = compiler.compile()
    executor = GatedLoRAExecutor(program, plan, config)

Design Principles:
  - NO quantization (QLoRA, INT8/INT4) - preserves accuracy
  - NO token skipping - privacy on every token
  - TFHE used ONLY for discrete gates (scalars/tiny vectors)
  - Bootstrapping ONLY for TFHE LUT evaluation (max 2 per layer)

See docs/architecture/HYBRID_HE.md for full architecture documentation.
"""

__version__ = '1.0.0'
__author__ = 'TenSafe Team'

# Version info
VERSION_INFO = {
    'version': __version__,
    'ckks_profiles': ['FAST', 'SAFE'],
    'backends': ['SIMULATION', 'HEONGPU', 'FIDESLIB', 'OPENFHE_GPU'],
}
