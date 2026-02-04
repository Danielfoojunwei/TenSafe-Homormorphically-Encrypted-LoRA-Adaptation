"""
HE-LoRA Module for TenSafe.

DEPRECATION NOTICE: This module is a compatibility layer that re-exports
from the new he_lora_microkernel package. Direct usage of he_lora_microkernel
is recommended for new code.

Provides homomorphic encryption for LoRA adapter inference with MOAI-style
optimizations:
- Column packing for rotation-free plaintext-ciphertext matmul
- Interleaved batching for efficient multi-sample processing
- Consistent packing strategy to avoid format conversions
- Noise budget tracking and level management

The base model runs in plaintext for low latency. Only the LoRA adapter
delta computation runs under HE:

    y = y_base + decrypt(HE_LoRA_Delta(encrypt(x)))

Where:
    - y_base = W_base @ x  (plaintext, fast)
    - HE_LoRA_Delta = scaling * encrypt(x) @ A^T @ B^T  (encrypted)

References:
    - MOAI: https://eprint.iacr.org/2025/991
"""

import warnings
from enum import Enum

# Import everything from the new microkernel
from he_lora_microkernel.compat import (
    HELoRAAdapter,
    HELoRAConfig,
    HELoRAMetrics,
    HEBackend,
)


class HEBackendNotAvailableError(Exception):
    """Raised when the HE backend is not available or not properly installed."""

    def __init__(self, message=None):
        default_msg = (
            "HE backend is required but not available.\n"
            "Install the he_lora_microkernel package."
        )
        super().__init__(message or default_msg)


def get_backend(params=None):
    """Get a configured HE backend instance."""
    backend = HEBackend(params)
    backend.setup()
    return backend


def verify_backend():
    """Verify the HE backend is properly installed and functional."""
    backend = HEBackend()
    backend.setup()
    return {
        "available": True,
        "backend": "he_lora_microkernel",
        "slot_count": backend.get_slot_count(),
    }


def create_he_lora_adapter(config=None, **kwargs):
    """Create an HE-LoRA adapter instance."""
    if config is None:
        config = HELoRAConfig(**kwargs)
    return HELoRAAdapter(config)


# Packing utilities
class ColumnPackedMatrix:
    """Column-packed matrix for MOAI-style operations."""

    def __init__(self, matrix):
        self.matrix = matrix


class InterleavedBatch:
    """Interleaved batch for efficient multi-sample processing."""

    def __init__(self, data):
        self.data = data


class PackingStrategy(Enum):
    """Packing strategy enumeration."""

    COLUMN = "column"
    ROW = "row"
    DIAGONAL = "diagonal"


def estimate_rotation_count(strategy, shape):
    """Estimate rotation count for a given packing strategy."""
    if strategy == PackingStrategy.COLUMN or strategy == "column":
        return 0  # Column packing has zero rotations
    elif strategy == PackingStrategy.DIAGONAL or strategy == "diagonal":
        return min(shape) if len(shape) >= 2 else 0
    else:
        return shape[0] if len(shape) >= 1 else 0


# Noise tracking
class NoiseBudgetExhaustedError(Exception):
    """Raised when noise budget is exhausted."""

    pass


class NoiseTracker:
    """Track noise budget consumption."""

    def __init__(self, initial_budget=100.0):
        self.budget = initial_budget
        self.initial_budget = initial_budget

    def consume(self, amount):
        """Consume noise budget."""
        self.budget -= amount
        if self.budget <= 0:
            raise NoiseBudgetExhaustedError("Noise budget exhausted")

    def get_remaining(self):
        """Get remaining noise budget."""
        return self.budget

    def reset(self):
        """Reset noise budget to initial value."""
        self.budget = self.initial_budget


# Issue deprecation warning on import
warnings.warn(
    "tensafe.he_lora is deprecated. Use he_lora_microkernel directly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Backend
    "HEBackend",
    "HEBackendNotAvailableError",
    "get_backend",
    "verify_backend",
    # Packing
    "ColumnPackedMatrix",
    "InterleavedBatch",
    "PackingStrategy",
    "estimate_rotation_count",
    # Noise
    "NoiseTracker",
    "NoiseBudgetExhaustedError",
    # Adapter
    "HELoRAAdapter",
    "HELoRAConfig",
    "HELoRAMetrics",
    "create_he_lora_adapter",
]
