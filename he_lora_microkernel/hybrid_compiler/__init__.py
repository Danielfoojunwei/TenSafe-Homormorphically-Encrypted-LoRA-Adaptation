"""
Hybrid CKKS-MOAI + TFHE HE Compiler for Gated LoRA Inference

This compiler implements a hybrid homomorphic encryption scheme where:
- CKKS + MOAI handles all high-dimensional linear algebra (LoRA matmuls)
- TFHE is used only for discrete non-linear control flow (gating, thresholds)

Architecture:
    CKKS + MOAI (Linear):
        - MatMul, Add, Mul (no activations, no conditionals)
        - Column packing and rotation elimination
        - Optimized for LoRA A/B matrices

    TFHE (Discrete Logic):
        - Threshold, Step, Sign, LUT-based gating
        - Programmable bootstrapping
        - Scalars or very small vectors (<=16 elements)

Supported LoRA Form:
    y = Wx + g(x) * Delta(x)

    Where:
    - Delta(x) = B(Ax) computed entirely in CKKS
    - g(x) is a discrete gate computed in TFHE
    - g(x) in {0, 1} or small integer

Key Constraints:
    - No TFHE on full vectors
    - No per-channel gating
    - No per-token bootstrapping loops
    - Max TFHE bootstraps per layer: <=2
    - Scheme switching only for scalars or tiny vectors

References:
    - MOAI: https://eprint.iacr.org/2025/991
    - TFHE: Fast Fully Homomorphic Encryption over the Torus
"""

from .ir import (
    # Scheme types
    Scheme,
    ValueType,
    # IR nodes
    IRNode,
    IRValue,
    IRProgram,
    # CKKS ops
    CKKSMatMul,
    CKKSAdd,
    CKKSMul,
    CKKSRescale,
    CKKSRotate,
    CKKSPackMOAI,
    # TFHE ops
    TFHELUT,
    TFHECompare,
    TFHEMUX,
    TFHEBootstrap,
    # Bridge ops
    CKKSQuantizeToInt,
    CKKSToTFHE,
    TFHEToCKKS,
    CKKSApplyMask,
    # Validation
    validate_program,
    SchemeViolationError,
)

from .scheduler import (
    HybridScheduler,
    ScheduleConfig,
    ExecutionPlan,
)

from .bridge import (
    CKKSTFHEBridge,
    BridgeConfig,
    QuantizationParams,
)

from .tfhe_lut import (
    LUTLibrary,
    step_lut,
    sign_lut,
    clip_lut,
    argmax_2_lut,
)

from .gated_lora import (
    GatedLoRACompiler,
    GatedLoRAConfig,
    GatedLoRAExecutor,
)

__all__ = [
    # IR
    "Scheme",
    "ValueType",
    "IRNode",
    "IRValue",
    "IRProgram",
    # CKKS ops
    "CKKSMatMul",
    "CKKSAdd",
    "CKKSMul",
    "CKKSRescale",
    "CKKSRotate",
    "CKKSPackMOAI",
    # TFHE ops
    "TFHELUT",
    "TFHECompare",
    "TFHEMUX",
    "TFHEBootstrap",
    # Bridge ops
    "CKKSQuantizeToInt",
    "CKKSToTFHE",
    "TFHEToCKKS",
    "CKKSApplyMask",
    # Validation
    "validate_program",
    "SchemeViolationError",
    # Scheduler
    "HybridScheduler",
    "ScheduleConfig",
    "ExecutionPlan",
    # Bridge
    "CKKSTFHEBridge",
    "BridgeConfig",
    "QuantizationParams",
    # LUT
    "LUTLibrary",
    "step_lut",
    "sign_lut",
    "clip_lut",
    "argmax_2_lut",
    # Gated LoRA
    "GatedLoRACompiler",
    "GatedLoRAConfig",
    "GatedLoRAExecutor",
]
