"""
Hybrid Compiler IR - Scheme-Aware Intermediate Representation

This module defines the IR for the hybrid CKKS-TFHE compiler with:
- Explicit scheme domains (CKKS vs TFHE)
- Type-safe value representations
- Mandatory scheme-switching bridge ops
- Static validation rules

Each IR node carries:
- scheme: CKKS | TFHE
- value_type: real_approx | bit | int_k
- shape: scalar | vector[n]
- precision_budget: bits (CKKS only)
- noise_budget: estimate (TFHE only)
- bootstrap_cost: estimate (TFHE only)
"""

from .types import (
    Scheme,
    ValueType,
    Shape,
    IRValue,
    CKKSMetadata,
    TFHEMetadata,
    create_ckks_value,
    create_tfhe_value,
)

from .nodes import (
    IRNode,
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
    # Bridge ops (MANDATORY for scheme switching)
    CKKSQuantizeToInt,
    CKKSToTFHE,
    TFHEToCKKS,
    CKKSApplyMask,
)

from .validation import (
    validate_program,
    validate_node,
    ValidationResult,
    SchemeViolationError,
    TypeMismatchError,
    ShapeError,
    BridgeRequiredError,
)

__all__ = [
    # Types
    "Scheme",
    "ValueType",
    "Shape",
    "IRValue",
    "CKKSMetadata",
    "TFHEMetadata",
    "create_ckks_value",
    "create_tfhe_value",
    # Nodes
    "IRNode",
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
    "validate_node",
    "ValidationResult",
    "SchemeViolationError",
    "TypeMismatchError",
    "ShapeError",
    "BridgeRequiredError",
]
