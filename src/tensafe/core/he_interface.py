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
    4. NO FRAGMENTATION: Legacy backends (N2HE, HEXL standalone) are deprecated

Usage:
    from tensafe.core.he_interface import get_backend

    # Get the unified backend (production mode)
    backend = get_backend()

    # Use backend
    ct = backend.encrypt(plaintext_vector)
    result = backend.lora_delta(ct, lora_a, lora_b, scaling=0.5)
    plaintext = backend.decrypt(result)

Modes:
    - PRODUCTION: Full HE with GPU acceleration (recommended)
    - SIMULATION: Testing mode (NOT cryptographically secure)
    - DISABLED: No HE operations
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
# Backend Types (Simplified)
# ==============================================================================


class HEBackendType(Enum):
    """
    Available HE backend types.

    The unified architecture uses the microkernel for all operations.
    Legacy types are kept for backward compatibility but map to the same backend.
    """
    PRODUCTION = "production"  # Microkernel with GPU acceleration
    SIMULATION = "simulation"  # Microkernel simulation mode (NOT SECURE)
    DISABLED = "disabled"  # No HE

    # Legacy types (deprecated, mapped to new types)
    TOY = "toy"  # DEPRECATED: Maps to SIMULATION
    N2HE = "n2he"  # DEPRECATED: Maps to PRODUCTION
    HEXL = "hexl"  # DEPRECATED: Maps to PRODUCTION
    CKKS_MOAI = "ckks_moai"  # DEPRECATED: Maps to PRODUCTION
    MICROKERNEL = "microkernel"  # DEPRECATED: Maps to PRODUCTION
    AUTO = "auto"  # DEPRECATED: Maps to PRODUCTION

    @classmethod
    def resolve(cls, backend_type: "HEBackendType") -> "HEBackendType":
        """Resolve legacy backend types to their modern equivalents."""
        legacy_mapping = {
            cls.TOY: cls.SIMULATION,
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

    def __init__(
        self,
        params: Optional[HEParams] = None,
        simulation_mode: bool = False,
    ):
        super().__init__(params)
        self._simulation_mode = simulation_mode
        self._backend = None
        self._packed_weights: Dict[str, Any] = {}

        # Determined backend type
        self._resolved_type = (
            HEBackendType.SIMULATION if simulation_mode else HEBackendType.PRODUCTION
        )

    @property
    def backend_type(self) -> HEBackendType:
        return self._resolved_type

    @property
    def backend_name(self) -> str:
        if self._simulation_mode:
            return "HE-LoRA Microkernel (Simulation)"
        return "HE-LoRA Microkernel (Production)"

    @property
    def is_production_ready(self) -> bool:
        return not self._simulation_mode

    def setup(self) -> None:
        """Set up the HE context using the microkernel."""
        try:
            from he_lora_microkernel.compat import HEBackend as MicrokernelBackend

            # Initialize microkernel backend
            self._backend = MicrokernelBackend(self.params.to_microkernel_params())
            self._backend.setup()

            self._is_setup = True

            mode_str = "SIMULATION" if self._simulation_mode else "PRODUCTION"
            logger.info(
                f"Unified HE Backend initialized ({mode_str}): "
                f"N={self.params.poly_modulus_degree}, "
                f"scale=2^{self.params.scale_bits}, "
                f"MOAI column packing={self.params.use_column_packing}"
            )

        except ImportError as e:
            if self._simulation_mode:
                # Allow pure simulation without microkernel
                logger.warning(
                    f"Microkernel not available ({e}). "
                    "Using pure simulation mode (no real HE)."
                )
                self._backend = None
                self._is_setup = True
            else:
                raise RuntimeError(
                    f"HE-LoRA Microkernel not available: {e}\n"
                    "For production HE, ensure he_lora_microkernel is installed.\n"
                    "For testing, use simulation_mode=True."
                )

    def encrypt(self, plaintext: np.ndarray) -> Any:
        """Encrypt a plaintext vector using CKKS."""
        self.validate_ready()
        self._metrics.operations_count += 1

        if self._backend is not None:
            return self._backend.encrypt(plaintext)

        # Pure simulation fallback
        return _SimulatedCiphertext(plaintext.astype(np.float64).copy())

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        """Decrypt a ciphertext."""
        self.validate_ready()

        # Handle _SimulatedCiphertext (from simulation fallback in add/multiply)
        # before trying the backend, since the backend won't know this type
        if isinstance(ciphertext, _SimulatedCiphertext):
            data = ciphertext.data
            if output_size > 0:
                return np.asarray(data)[:output_size]
            return np.asarray(data)

        if self._backend is not None:
            return self._backend.decrypt(ciphertext, output_size)

        # Pure simulation fallback for dict/other types
        if isinstance(ciphertext, dict):
            data = ciphertext.get("data", ciphertext)
        else:
            data = ciphertext

        if output_size > 0:
            return np.asarray(data)[:output_size]
        return np.asarray(data)

    def add(self, ct1: Any, ct2: Any) -> Any:
        """Add two ciphertexts."""
        self.validate_ready()
        self._metrics.operations_count += 1

        if self._backend is not None and hasattr(self._backend, 'add'):
            return self._backend.add(ct1, ct2)

        # Pure simulation fallback
        d1 = ct1.data if isinstance(ct1, _SimulatedCiphertext) else (ct1.get("data", ct1) if isinstance(ct1, dict) else ct1)
        d2 = ct2.data if isinstance(ct2, _SimulatedCiphertext) else (ct2.get("data", ct2) if isinstance(ct2, dict) else ct2)
        return _SimulatedCiphertext(np.asarray(d1) + np.asarray(d2))

    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        """Multiply ciphertext by plaintext."""
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1

        if self._backend is not None and hasattr(self._backend, 'multiply_plain'):
            return self._backend.multiply_plain(ct, plaintext)

        # Pure simulation fallback
        data = ct.data if isinstance(ct, _SimulatedCiphertext) else (ct.get("data", ct) if isinstance(ct, dict) else ct)
        scalar = plaintext.flatten()[0] if plaintext.size == 1 else plaintext
        return _SimulatedCiphertext(np.asarray(data) * scalar)

    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        """
        Encrypted matrix multiplication using MOAI column packing.

        Uses rotation-free column-packed multiplication when available.
        """
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1

        # Try column-packed matmul (MOAI optimization)
        if (
            self._backend is not None
            and self.params.use_column_packing
            and hasattr(self._backend, 'column_packed_matmul')
        ):
            packed = self._get_or_create_packed(weight)
            return self._backend.column_packed_matmul(ct, packed, rescale=True)

        # Fallback to standard matmul
        if self._backend is not None and hasattr(self._backend, 'matmul'):
            return self._backend.matmul(ct, weight)

        # Pure simulation fallback
        data = ct.data if isinstance(ct, _SimulatedCiphertext) else (ct.get("data", ct) if isinstance(ct, dict) else ct)
        result = np.asarray(data) @ weight.T
        return _SimulatedCiphertext(result)

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
        if self._backend is not None and hasattr(self._backend, 'lora_delta'):
            return self._backend.lora_delta(ct_x, lora_a, lora_b, scaling)

        # Fallback to sequential matmuls with column packing
        packed_a = self._get_or_create_packed(lora_a, f"lora_a_{id(lora_a)}")
        packed_b = self._get_or_create_packed(lora_b, f"lora_b_{id(lora_b)}")

        if (
            self._backend is not None
            and hasattr(self._backend, 'column_packed_matmul')
        ):
            intermediate = self._backend.column_packed_matmul(ct_x, packed_a)
            result = self._backend.column_packed_matmul(intermediate, packed_b)
            if abs(scaling - 1.0) > 1e-6:
                result = self.multiply_plain(result, np.array([scaling]))
            return result

        # Pure simulation
        return super().lora_delta(ct_x, lora_a, lora_b, scaling)

    def _get_or_create_packed(self, matrix: np.ndarray, key: Optional[str] = None) -> Any:
        """Get or create column-packed matrix for MOAI optimization."""
        key = key or f"matrix_{id(matrix)}"

        if key not in self._packed_weights:
            if self._backend is not None and hasattr(self._backend, 'create_column_packed_matrix'):
                self._packed_weights[key] = self._backend.create_column_packed_matrix(matrix)
            else:
                # Store unpacked if packing not available
                self._packed_weights[key] = matrix

        return self._packed_weights[key]

    def get_slot_count(self) -> int:
        """Get number of SIMD slots."""
        if self._backend is not None and hasattr(self._backend, 'get_slot_count'):
            return self._backend.get_slot_count()
        return self.params.poly_modulus_degree // 2

    def get_metrics(self) -> HEMetrics:
        """Get accumulated metrics from the microkernel."""
        if self._backend is not None and hasattr(self._backend, 'get_operation_stats'):
            stats = self._backend.get_operation_stats()
            self._metrics.rotations_count = stats.get("rotations", 0)
            self._metrics.multiplications_count = stats.get("multiplications", self._metrics.multiplications_count)
            self._metrics.rescale_count = stats.get("rescales", 0)
            self._metrics.keyswitches_count = stats.get("keyswitches", 0)
        return self._metrics


@dataclass
class _SimulatedCiphertext:
    """
    Simulated ciphertext for testing (NOT SECURE).

    This is only used when the microkernel is not available
    and simulation mode is enabled.
    """
    data: np.ndarray
    noise_budget: float = 100.0
    level: int = 0

    def to_bytes(self) -> bytes:
        return self.data.tobytes()


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
        return False  # Not applicable

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
        backend_type: Type of backend (PRODUCTION, SIMULATION, or DISABLED)
        params: HE parameters
        setup: Automatically call setup()

    Returns:
        Configured HEBackendInterface

    Example:
        # Production mode (recommended)
        backend = get_backend(HEBackendType.PRODUCTION)

        # Simulation mode (for testing)
        backend = get_backend(HEBackendType.SIMULATION)

        # No HE
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
    elif resolved_type == HEBackendType.SIMULATION:
        backend = UnifiedHEBackend(params, simulation_mode=True)
    else:  # PRODUCTION
        backend = UnifiedHEBackend(params, simulation_mode=False)

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
    elif resolved == HEBackendType.SIMULATION:
        return True  # Always available (can fall back to pure simulation)
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
    available = ["disabled", "simulation"]  # Always available

    if is_backend_available(HEBackendType.PRODUCTION):
        available.append("production")

    return available


# ==============================================================================
# Backward Compatibility Exports
# ==============================================================================

# These are kept for backward compatibility but use the unified backend internally

# Legacy backend wrappers (all route to UnifiedHEBackend)
ToyHEBackend = lambda params=None: UnifiedHEBackend(params, simulation_mode=True)
N2HEBackendWrapper = lambda params=None: UnifiedHEBackend(params, simulation_mode=False)
HEXLBackendWrapper = lambda params=None: UnifiedHEBackend(params, simulation_mode=False)
CKKSMOAIBackendWrapper = lambda params=None: UnifiedHEBackend(params, simulation_mode=False)
MicrokernelBackendWrapper = lambda params=None: UnifiedHEBackend(params, simulation_mode=False)


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
    "ToyHEBackend",
    "N2HEBackendWrapper",
    "HEXLBackendWrapper",
    "CKKSMOAIBackendWrapper",
    "MicrokernelBackendWrapper",
]
