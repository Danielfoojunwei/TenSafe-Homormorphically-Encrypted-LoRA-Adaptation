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

from .bridge import (
    BridgeConfig,
    CKKSTFHEBridge,
    QuantizationParams,
)
from .gated_lora import (
    GatedLoRACompiler,
    GatedLoRAConfig,
    GatedLoRAExecutor,
)
from .ir import (
    # TFHE ops
    TFHELUT,
    TFHEMUX,
    CKKSAdd,
    CKKSApplyMask,
    # CKKS ops
    CKKSMatMul,
    CKKSMul,
    CKKSPackMOAI,
    # Bridge ops
    CKKSQuantizeToInt,
    CKKSRescale,
    CKKSRotate,
    CKKSToTFHE,
    # IR nodes
    IRNode,
    IRProgram,
    IRValue,
    # Scheme types
    Scheme,
    SchemeViolationError,
    TFHEBootstrap,
    TFHECompare,
    TFHEToCKKS,
    ValueType,
    # Validation
    validate_program,
)
from .scheduler import (
    ExecutionPlan,
    HybridScheduler,
    ScheduleConfig,
)
from .tfhe_lut import (
    LUTLibrary,
    argmax_2_lut,
    clip_lut,
    sign_lut,
    step_lut,
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
