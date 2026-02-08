"""
TenSafe Inference Module - Backward Compatibility Re-exports.

This module re-exports from the unified tensafe.core modules.
All new code should import directly from tensafe.core.inference.

DEPRECATED: This module exists for backward compatibility only.
Use tensafe.core.inference instead.
"""

from tensafe.core.config import InferenceConfig
from tensafe.core.inference import (
    BatchInferenceResult,
    GenerationConfig,
    InferenceMode,
    InferenceResult,
    # Core classes
    TenSafeInference,
    TGSPEnforcementError,
    # Factory function
    create_inference,
)

# Backward compatibility alias
LoRAMode = InferenceMode

__all__ = [
    "TenSafeInference",
    "InferenceMode",
    "LoRAMode",  # Alias for backward compatibility
    "InferenceConfig",
    "InferenceResult",
    "BatchInferenceResult",
    "GenerationConfig",
    "TGSPEnforcementError",
    "create_inference",
]
