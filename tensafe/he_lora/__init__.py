"""
HE-LoRA Module for TenSafe.

DEPRECATION NOTICE: This module is now a compatibility layer that re-exports
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

import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import from the new microkernel first (preferred)
_USE_MICROKERNEL = False
try:
    from he_lora_microkernel.compat import (
        HELoRAAdapter,
        HELoRAConfig,
        HELoRAMetrics,
        HEBackend as _MicrokernelHEBackend,
    )
    _USE_MICROKERNEL = True
    logger.debug("Using he_lora_microkernel as backend")
except ImportError:
    logger.debug("he_lora_microkernel not available, using legacy implementation")

# Import from legacy modules (or provide fallback implementations)
if _USE_MICROKERNEL:
    # Provide compatibility wrappers using the microkernel
    from he_lora_microkernel.compat import HEBackend

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

else:
    # Fall back to legacy implementation
    from .backend import (
        HEBackend,
        HEBackendNotAvailableError,
        get_backend,
        verify_backend,
    )
    from .helora_adapter import (
        HELoRAAdapter,
        HELoRAConfig,
        HELoRAMetrics,
        create_he_lora_adapter,
    )

# Import packing utilities (always from local, with fallback)
try:
    from .packing import (
        ColumnPackedMatrix,
        InterleavedBatch,
        PackingStrategy,
        estimate_rotation_count,
    )
except ImportError:
    # Provide stub implementations if packing module has import issues
    class ColumnPackedMatrix:
        """Column-packed matrix for MOAI-style operations."""
        def __init__(self, matrix):
            self.matrix = matrix

    class InterleavedBatch:
        """Interleaved batch for efficient multi-sample processing."""
        def __init__(self, data):
            self.data = data

    class PackingStrategy:
        """Packing strategy enumeration."""
        COLUMN = "column"
        ROW = "row"
        DIAGONAL = "diagonal"

    def estimate_rotation_count(strategy, shape):
        """Estimate rotation count for a given packing strategy."""
        return 0  # Column packing has zero rotations

# Import noise tracking (always from local, with fallback)
try:
    from .noise_tracker import (
        NoiseTracker,
        NoiseBudgetExhaustedError,
    )
except ImportError:
    # Provide stub implementations
    class NoiseBudgetExhaustedError(Exception):
        """Raised when noise budget is exhausted."""
        pass

    class NoiseTracker:
        """Track noise budget consumption."""
        def __init__(self, initial_budget=100.0):
            self.budget = initial_budget

        def consume(self, amount):
            self.budget -= amount
            if self.budget <= 0:
                raise NoiseBudgetExhaustedError("Noise budget exhausted")

        def get_remaining(self):
            return self.budget

# Issue deprecation warning on import
warnings.warn(
    "tensafe.he_lora is deprecated. Use he_lora_microkernel directly.",
    DeprecationWarning,
    stacklevel=2
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
    # Noise
    "NoiseTracker",
    "NoiseBudgetExhaustedError",
    # Adapter
    "HELoRAAdapter",
    "HELoRAConfig",
    "HELoRAMetrics",
    "create_he_lora_adapter",
]
