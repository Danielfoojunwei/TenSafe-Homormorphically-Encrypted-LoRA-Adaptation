"""
N2HE Integration Module for HE-LoRA Microkernel

This module integrates HintSight-Technology's N2HE concepts into the TenSafe
HE-LoRA microkernel, providing:

1. FasterNTT: CPU fallback backend for portable NTT operations
2. LUT System: GPU-accelerated lookup tables for non-linear activations (ReLU, etc.)

Architecture:
    The N2HE integration works alongside the existing GPU MOAI setup:

    Production Path (default):
        GPU CKKS Backend + MOAI Column Packing
        └── For linear LoRA operations (Ct×Pt matmul)

    N2HE Extensions:
        ├── FasterNTT Backend (CPU fallback)
        │   └── For ARM/non-GPU environments
        └── LUT Activation System (GPU)
            └── For non-linear adapters (gated LoRA, etc.)

Usage:
    from he_lora_microkernel.n2he import (
        N2HEAdapterConfig,
        AdapterType,
        NonLinearActivation,
        FasterNTTBackend,
        LUTActivationEngine,
    )

    # Configure non-linear adapter
    config = N2HEAdapterConfig(
        adapter_type=AdapterType.GATED_LORA,
        activation=NonLinearActivation.RELU,
        use_lut=True,
    )

References:
    - N2HE: https://github.com/HintSight-Technology/N2HE
    - MOAI: https://eprint.iacr.org/2025/991
"""

from .adapter_config import (
    AdapterPlacement,
    AdapterType,
    N2HEAdapterConfig,
    NonLinearActivation,
    validate_adapter_config,
)
from .faster_ntt import (
    FasterNTT,
    FasterNTTBackend,
    NTTDirection,
    get_ntt_backend,
)
from .lut_activation import (
    ActivationLUT,
    LUTActivationEngine,
    LUTConfig,
    create_gelu_lut,
    create_relu_lut,
    create_sigmoid_lut,
)
from .n2he_backend import (
    N2HEBackend,
    N2HEBackendType,
    create_n2he_backend,
    is_n2he_available,
)
from .n2he_params import (
    LWEParams,
    N2HEParams,
    N2HEProfile,
    RLWEParams,
    get_n2he_profile,
    select_optimal_n2he_profile,
)

__all__ = [
    # Parameters
    "N2HEParams",
    "N2HEProfile",
    "LWEParams",
    "RLWEParams",
    "get_n2he_profile",
    "select_optimal_n2he_profile",
    # Adapter Configuration
    "AdapterType",
    "NonLinearActivation",
    "N2HEAdapterConfig",
    "AdapterPlacement",
    "validate_adapter_config",
    # FasterNTT
    "FasterNTT",
    "NTTDirection",
    "FasterNTTBackend",
    "get_ntt_backend",
    # LUT Activation
    "LUTActivationEngine",
    "LUTConfig",
    "ActivationLUT",
    "create_relu_lut",
    "create_gelu_lut",
    "create_sigmoid_lut",
    # Backend
    "N2HEBackend",
    "N2HEBackendType",
    "create_n2he_backend",
    "is_n2he_available",
]
