"""
Non-Linear Adapter Implementations

Provides adapter implementations for hybrid CKKS-TFHE encryption.
"""

from .gated_lora_adapter import (
    # Configuration
    GatedLoRAAdapterConfig,
    AdapterMetrics,
    AdapterWeights,
    # Protocol
    NonLinearAdapter,
    # Implementation
    HEGatedLoRAAdapter,
    # Factory
    create_gated_lora_adapter,
    plaintext_gated_lora,
)

__all__ = [
    # Configuration
    "GatedLoRAAdapterConfig",
    "AdapterMetrics",
    "AdapterWeights",
    # Protocol
    "NonLinearAdapter",
    # Implementation
    "HEGatedLoRAAdapter",
    # Factory
    "create_gated_lora_adapter",
    "plaintext_gated_lora",
]
