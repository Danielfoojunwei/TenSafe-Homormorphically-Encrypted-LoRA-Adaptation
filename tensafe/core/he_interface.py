"""
TenSafe Unified HE Backend Interface.

This module provides a unified interface for homomorphic encryption,
routing ALL operations through the HE-LoRA Microkernel with MOAI optimizations.

Architecture:
    HEBackendInterface (abstract)
    └── UnifiedHEBackend (single production backend)
        └── Uses he_lora_microkernel internally

Key Design Principles:
    1. SINGLE BACKEND: All HE goes through the microkernel
    2. MOAI OPTIMIZATIONS: Rotation-free column packing for CKKS
    3. GPU ACCELERATION: Production mode uses GPU backends
    4. NO MOCKS/SIMULATION: All operations use real cryptographic backends

Usage:
    from tensafe.core.he_interface import get_backend

    # Get the unified backend (production mode)
    backend = get_backend()

    # Use backend
    ct = backend.encrypt(plaintext_vector)
    result = backend.lora_delta(ct, lora_a, lora_b, scaling=0.5)
    plaintext = backend.decrypt(result)

Modes:
    - PRODUCTION: Full HE with GPU acceleration (required)
    - DISABLED: No HE operations (plaintext passthrough)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Backend Types
# ==============================================================================


class HEBackendType(Enum):
    """
    Available HE backend types.

    The unified architecture uses the microkernel for all operations.
    Legacy types map to PRODUCTION.
    """
    PRODUCTION = "production"  # Microkernel with GPU acceleration
    DISABLED = "disabled"  # No HE (plaintext passthrough)

    # Legacy types (deprecated, all map to PRODUCTION)
    N2HE = "n2he"  # DEPRECATED: Maps to PRODUCTION
    HEXL = "hexl"  # DEPRECATED: Maps to PRODUCTION
    CKKS_MOAI = "ckks_moai"  # DEPRECATED: Maps to PRODUCTION
    MICROKERNEL = "microkernel"  # DEPRECATED: Maps to PRODUCTION
    AUTO = "auto"  # DEPRECATED: Maps to PRODUCTION

    @classmethod
    def resolve(cls, backend_type: "HEBackendType") -> "HEBackendType":
        """Resolve legacy backend types to their modern equivalents."""
        legacy_mapping = {
            cls.N2HE: cls.PRODUCTION,
            cls.HEXL: cls.PRODUCTION,
            cls.CKKS_MOAI: cls.PRODUCTION,
            cls.MICROKERNEL: cls.PRODUCTION,
            cls.AUTO: cls.PRODUCTION,
        }
        if backend_type in legacy_mapping:
            resolved = legacy_mapping[backend_type]
            logger.warning(
                f"HEBackendType.{backend_type.value} is deprecated. "
                f"Using HEBackendType.{resolved.value} instead."
            )
            return resolved
        return backend_type


class HEScheme(Enum):
    """HE scheme types (for reference only - microkernel uses CKKS)."""
    CKKS = "ckks"  # ONLY supported scheme for production


# ==============================================================================
# Parameters and Metrics
# ==============================================================================


@dataclass
class HEParams:
    """
    Unified HE parameters for the microkernel backend.

    These parameters configure the CKKS scheme used by the microkernel.
    MOAI optimizations (column packing) are enabled by default.
    """
    # Scheme (only CKKS is supported)
    scheme: HEScheme = HEScheme.CKKS

    # Security level
    security_level: int = 128  # bits

    # CKKS Ring parameters
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40

    # MOAI optimizations (always enabled for production)
    use_column_packing: bool = True
    use_interleaved_batching: bool = True

    # GPU configuration
    gpu_device_id: int = 0
    use_gpu: bool = True

    def to_microkernel_params(self) -> Dict[str, Any]:
        """Convert to microkernel parameter format."""
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "scale_bits": self.scale_bits,
            "security_level": self.security_level,
            "use_column_packing": self.use_column_packing,
            "use_interleaved_batching": self.use_interleaved_batching,
            "gpu_device_id": self.gpu_device_id,
        }


@dataclass
class HEMetrics:
    """Metrics from HE operations."""
    operations_count: int = 0
    rotations_count: int = 0
    multiplications_count: int = 0
    rescale_count: int = 0
    keyswitches_count: int = 0
    encrypt_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    decrypt_time_ms: float = 0.0
    total_time_ms: float = 0.0
    noise_budget_min: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": self.operations_count,
            "rotations": self.rotations_count,
            "multiplications": self.multiplications_count,
            "rescales": self.rescale_count,
            "keyswitches": self.keyswitches_count,
            "encrypt_time_ms": self.encrypt_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "decrypt_time_ms": self.decrypt_time_ms,
            "total_time_ms": self.total_time_ms,
            "noise_budget_min": self.noise_budget_min if self.noise_budget_min != float('inf') else None,
        }


# ==============================================================================
# Backend Interface
# ==============================================================================


class HEBackendInterface(ABC):
    """
    Abstract interface for HE backends.

    In the unified architecture, there is only ONE implementation:
    UnifiedHEBackend, which routes to the microkernel.
    """

    def __init__(self, params: Optional[HEParams] = None):
        self.params = params or HEParams()
        self._metrics = HEMetrics()
        self._is_setup = False

    @property
    @abstractmethod
    def backend_type(self) -> HEBackendType:
        """Get the backend type."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get human-readable backend name."""
        pass

    @property
    def is_production_ready(self) -> bool:
        """Check if backend is suitable for production."""
        return self.backend_type == HEBackendType.PRODUCTION

    @property
    def is_setup(self) -> bool:
        """Check if backend is set up and ready."""
        return self._is_setup

    @abstractmethod
    def setup(self) -> None:
        """Set up the HE context and generate keys."""
        pass

    @abstractmethod
    def encrypt(self, plaintext: np.ndarray) -> Any:
        """Encrypt a plaintext vector."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        """Decrypt a ciphertext."""
        pass

    @abstractmethod
    def add(self, ct1: Any, ct2: Any) -> Any:
        """Add two ciphertexts."""
        pass

    @abstractmethod
    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        """Multiply ciphertext by plaintext."""
        pass

    @abstractmethod
    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        """Encrypted matrix multiplication: ct @ weight^T."""
        pass

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Any:
        """
        Compute LoRA delta: scaling * (ct_x @ A^T @ B^T).

        This uses MOAI column packing for rotation-free computation.
        """
        intermediate = self.matmul(ct_x, lora_a)
        result = self.matmul(intermediate, lora_b)
        if abs(scaling - 1.0) > 1e-6:
            result = self.multiply_plain(result, np.array([scaling]))
        return result

    def get_slot_count(self) -> int:
        """Get number of SIMD slots available."""
        return self.params.poly_modulus_degree // 2

    def get_metrics(self) -> HEMetrics:
        """Get accumulated metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._metrics = HEMetrics()

    def validate_ready(self) -> None:
        """Raise if backend is not ready for operations."""
        if not self._is_setup:
            raise RuntimeError(f"{self.backend_name} backend not set up. Call setup() first.")


# ==============================================================================
# Unified HE Backend (Single Production Implementation)
# ==============================================================================


class UnifiedHEBackend(HEBackendInterface):
    """
    Unified HE backend using the HE-LoRA Microkernel.

    This is the ONLY production backend. It provides:
    - CKKS encryption with MOAI optimizations
    - Rotation-free column packing for LoRA computations
    - GPU acceleration (when available)
    - Comprehensive metrics tracking

    All other backend types (N2HE, HEXL, CKKS_MOAI) are deprecated
    and route through this implementation.
    """

    def __init__(self, params: Optional[HEParams] = None):
        super().__init__(params)
        self._backend = None
        self._packed_weights: Dict[str, Any] = {}

    @property
    def backend_type(self) -> HEBackendType:
        return HEBackendType.PRODUCTION

    @property
    def backend_name(self) -> str:
        return "HE-LoRA Microkernel (Production)"

    @property
    def is_production_ready(self) -> bool:
        return True

    def setup(self) -> None:
        """Set up the HE context using the microkernel."""
        try:
            from he_lora_microkernel.compat import HEBackend as MicrokernelBackend

            # Initialize microkernel backend
            self._backend = MicrokernelBackend(self.params.to_microkernel_params())
            self._backend.setup()

            self._is_setup = True

            logger.info(
                f"Unified HE Backend initialized (PRODUCTION): "
                f"N={self.params.poly_modulus_degree}, "
                f"scale=2^{self.params.scale_bits}, "
                f"MOAI column packing={self.params.use_column_packing}"
            )

        except ImportError as e:
            raise RuntimeError(
                f"HE-LoRA Microkernel not available: {e}\n"
                "For production HE, ensure he_lora_microkernel is installed.\n"
                "Use HEBackendType.DISABLED if HE is not required."
            )

    def encrypt(self, plaintext: np.ndarray) -> Any:
        """Encrypt a plaintext vector using CKKS."""
        self.validate_ready()
        self._metrics.operations_count += 1
        return self._backend.encrypt(plaintext)

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        """Decrypt a ciphertext."""
        self.validate_ready()
        return self._backend.decrypt(ciphertext, output_size)

    def add(self, ct1: Any, ct2: Any) -> Any:
        """Add two ciphertexts."""
        self.validate_ready()
        self._metrics.operations_count += 1

        if hasattr(self._backend, 'add'):
            return self._backend.add(ct1, ct2)

        raise NotImplementedError("Backend does not support add operation")

    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        """Multiply ciphertext by plaintext."""
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1

        if hasattr(self._backend, 'multiply_plain'):
            return self._backend.multiply_plain(ct, plaintext)

        raise NotImplementedError("Backend does not support multiply_plain operation")

    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        """
        Encrypted matrix multiplication using MOAI column packing.

        Uses rotation-free column-packed multiplication when available.
        """
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1

        # Try column-packed matmul (MOAI optimization)
        if self.params.use_column_packing and hasattr(self._backend, 'column_packed_matmul'):
            packed = self._get_or_create_packed(weight)
            return self._backend.column_packed_matmul(ct, packed, rescale=True)

        # Fallback to standard matmul
        if hasattr(self._backend, 'matmul'):
            return self._backend.matmul(ct, weight)

        raise NotImplementedError("Backend does not support matmul operation")

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Any:
        """
        Compute LoRA delta with MOAI optimizations.

        Uses the microkernel's optimized lora_delta if available,
        which applies column packing for rotation-free computation.
        """
        self.validate_ready()

        # Use microkernel's optimized implementation
        if hasattr(self._backend, 'lora_delta'):
            return self._backend.lora_delta(ct_x, lora_a, lora_b, scaling)

        # Fallback to sequential matmuls with column packing
        packed_a = self._get_or_create_packed(lora_a, f"lora_a_{id(lora_a)}")
        packed_b = self._get_or_create_packed(lora_b, f"lora_b_{id(lora_b)}")

        if hasattr(self._backend, 'column_packed_matmul'):
            intermediate = self._backend.column_packed_matmul(ct_x, packed_a)
            result = self._backend.column_packed_matmul(intermediate, packed_b)
            if abs(scaling - 1.0) > 1e-6:
                result = self.multiply_plain(result, np.array([scaling]))
            return result

        # Final fallback
        return super().lora_delta(ct_x, lora_a, lora_b, scaling)

    def _get_or_create_packed(self, matrix: np.ndarray, key: Optional[str] = None) -> Any:
        """Get or create column-packed matrix for MOAI optimization."""
        key = key or f"matrix_{id(matrix)}"

        if key not in self._packed_weights:
            if hasattr(self._backend, 'create_column_packed_matrix'):
                self._packed_weights[key] = self._backend.create_column_packed_matrix(matrix)
            else:
                # Store unpacked if packing not available
                self._packed_weights[key] = matrix

        return self._packed_weights[key]

    def get_slot_count(self) -> int:
        """Get number of SIMD slots."""
        if hasattr(self._backend, 'get_slot_count'):
            return self._backend.get_slot_count()
        return self.params.poly_modulus_degree // 2

    def get_metrics(self) -> HEMetrics:
        """Get accumulated metrics from the microkernel."""
        if hasattr(self._backend, 'get_operation_stats'):
            stats = self._backend.get_operation_stats()
            self._metrics.rotations_count = stats.get("rotations", 0)
            self._metrics.multiplications_count = stats.get("multiplications", self._metrics.multiplications_count)
            self._metrics.rescale_count = stats.get("rescales", 0)
            self._metrics.keyswitches_count = stats.get("keyswitches", 0)
        return self._metrics


# ==============================================================================
# Disabled Backend (No-op)
# ==============================================================================


class DisabledHEBackend(HEBackendInterface):
    """
    No-op HE backend when HE is disabled.

    All operations pass through without encryption.
    """

    @property
    def backend_type(self) -> HEBackendType:
        return HEBackendType.DISABLED

    @property
    def backend_name(self) -> str:
        return "Disabled (No HE)"

    @property
    def is_production_ready(self) -> bool:
        return False  # Not applicable - HE is disabled

    def setup(self) -> None:
        self._is_setup = True
        logger.info("HE backend disabled - all operations will be plaintext")

    def encrypt(self, plaintext: np.ndarray) -> np.ndarray:
        return plaintext.copy()

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        if output_size > 0:
            return np.asarray(ciphertext)[:output_size]
        return np.asarray(ciphertext)

    def add(self, ct1: Any, ct2: Any) -> np.ndarray:
        return np.asarray(ct1) + np.asarray(ct2)

    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> np.ndarray:
        scalar = plaintext.flatten()[0] if plaintext.size == 1 else plaintext
        return np.asarray(ct) * scalar

    def matmul(self, ct: Any, weight: np.ndarray) -> np.ndarray:
        return np.asarray(ct) @ weight.T


# ==============================================================================
# Backend Factory
# ==============================================================================


def get_backend(
    backend_type: Union[HEBackendType, str] = HEBackendType.PRODUCTION,
    params: Optional[HEParams] = None,
    setup: bool = True,
) -> HEBackendInterface:
    """
    Get an HE backend instance.

    In the unified architecture, this returns the single UnifiedHEBackend
    which routes through the microkernel with MOAI optimizations.

    Args:
        backend_type: Type of backend (PRODUCTION or DISABLED)
        params: HE parameters
        setup: Automatically call setup()

    Returns:
        Configured HEBackendInterface

    Example:
        # Production mode (required for HE)
        backend = get_backend(HEBackendType.PRODUCTION)

        # No HE (plaintext passthrough)
        backend = get_backend(HEBackendType.DISABLED)
    """
    # Convert string to enum
    if isinstance(backend_type, str):
        try:
            backend_type = HEBackendType(backend_type.lower())
        except ValueError:
            logger.warning(f"Unknown backend type '{backend_type}', using PRODUCTION")
            backend_type = HEBackendType.PRODUCTION

    # Resolve legacy types
    resolved_type = HEBackendType.resolve(backend_type)

    # Create appropriate backend
    if resolved_type == HEBackendType.DISABLED:
        backend = DisabledHEBackend(params)
    else:  # PRODUCTION
        backend = UnifiedHEBackend(params)

    # Setup if requested
    if setup:
        backend.setup()

    logger.info(f"Created HE backend: {backend.backend_name}")
    return backend


def is_backend_available(backend_type: Union[HEBackendType, str] = HEBackendType.PRODUCTION) -> bool:
    """
    Check if a backend type is available.

    Args:
        backend_type: Backend to check

    Returns:
        True if backend is available
    """
    if isinstance(backend_type, str):
        try:
            backend_type = HEBackendType(backend_type.lower())
        except ValueError:
            return False

    resolved = HEBackendType.resolve(backend_type)

    if resolved == HEBackendType.DISABLED:
        return True
    else:  # PRODUCTION
        try:
            from he_lora_microkernel.compat import HEBackend as _
            return True
        except ImportError:
            return False


def list_available_backends() -> List[str]:
    """
    List all available backend types.

    Returns:
        List of available backend type names
    """
    available = ["disabled"]  # Always available

    if is_backend_available(HEBackendType.PRODUCTION):
        available.append("production")

    return available


# ==============================================================================
# Backward Compatibility Exports
# ==============================================================================

# Legacy backend wrappers (all route to UnifiedHEBackend)
N2HEBackendWrapper = lambda params=None: UnifiedHEBackend(params)
HEXLBackendWrapper = lambda params=None: UnifiedHEBackend(params)
CKKSMOAIBackendWrapper = lambda params=None: UnifiedHEBackend(params)
MicrokernelBackendWrapper = lambda params=None: UnifiedHEBackend(params)


# Export for type checking
__all__ = [
    # Core types
    "HEBackendType",
    "HEScheme",
    "HEParams",
    "HEMetrics",
    # Interfaces
    "HEBackendInterface",
    # Implementations
    "UnifiedHEBackend",
    "DisabledHEBackend",
    # Factory functions
    "get_backend",
    "is_backend_available",
    "list_available_backends",
    # Backward compatibility (deprecated)
    "N2HEBackendWrapper",
    "HEXLBackendWrapper",
    "CKKSMOAIBackendWrapper",
    "MicrokernelBackendWrapper",
]
