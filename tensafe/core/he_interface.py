"""
TenSafe Unified HE Backend Interface.

This module provides a unified interface for homomorphic encryption backends,
abstracting over the different implementations:
- N2HE (LWE/RLWE, pure Python toy mode + optional native)
- N2HE-HEXL (CKKS with Intel HEXL acceleration, production)

The interface allows:
- Consistent API across backends
- Backend selection at runtime
- Transparent fallback (in development)
- Production-safe defaults

Architecture:
    HEBackendInterface (abstract)
    ├── ToyHEBackend (dev/testing only)
    ├── N2HEBackend (pure Python, optional native)
    └── HEXLBackend (production, CKKS + HEXL)

Usage:
    from tensafe.core.he_interface import (
        get_backend,
        HEBackendInterface,
        HEConfig,
    )

    # Get the best available backend
    backend = get_backend()

    # Or specify explicitly
    backend = get_backend(backend_type="hexl")

    # Use backend
    ct = backend.encrypt(plaintext_vector)
    result = backend.lora_delta(ct, lora_a, lora_b, scaling=0.5)
    plaintext = backend.decrypt(result)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

import numpy as np

from tensafe.core.gates import ProductionGates, GateDeniedError

logger = logging.getLogger(__name__)


class HEBackendType(Enum):
    """Available HE backend types."""
    TOY = "toy"  # Toy/simulation (NOT SECURE)
    N2HE = "n2he"  # N2HE pure Python with optional native
    HEXL = "hexl"  # N2HE-HEXL production backend (Intel HEXL - Intel CPU only)
    CKKS_MOAI = "ckks_moai"  # CKKS MOAI backend (Pyfhel, MOAI-style optimizations)
    AUTO = "auto"  # Auto-select best available


class HEScheme(Enum):
    """HE scheme types."""
    LWE = "lwe"
    RLWE = "rlwe"
    CKKS = "ckks"
    BFV = "bfv"


@dataclass
class HEParams:
    """
    Unified HE parameters for all backends.

    These parameters are normalized across backends.
    """
    # Scheme
    scheme: HEScheme = HEScheme.CKKS

    # Security
    security_level: int = 128  # bits

    # Ring parameters (RLWE/CKKS)
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40

    # LWE parameters
    n: int = 1024  # Lattice dimension
    q: int = 2**32  # Ciphertext modulus
    t: int = 2**16  # Plaintext modulus
    std_dev: float = 3.2

    # MOAI optimizations
    use_column_packing: bool = True
    use_interleaved_batching: bool = True

    def to_n2he_params(self) -> Dict[str, Any]:
        """Convert to N2HE parameter format."""
        return {
            "scheme_type": self.scheme.value,
            "n": self.n,
            "q": self.q,
            "t": self.t,
            "std_dev": self.std_dev,
            "poly_degree": self.poly_modulus_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "security_level": self.security_level,
        }

    def to_hexl_params(self) -> Dict[str, Any]:
        """Convert to HEXL parameter format."""
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "scale_bits": self.scale_bits,
        }


@dataclass
class HEMetrics:
    """Metrics from HE operations."""
    operations_count: int = 0
    rotations_count: int = 0
    multiplications_count: int = 0
    rescale_count: int = 0
    noise_budget_min: float = float('inf')
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": self.operations_count,
            "rotations": self.rotations_count,
            "multiplications": self.multiplications_count,
            "rescales": self.rescale_count,
            "noise_budget_min": self.noise_budget_min if self.noise_budget_min != float('inf') else None,
            "total_time_ms": self.total_time_ms,
        }


class HECiphertext(Protocol):
    """Protocol for ciphertext objects."""

    @property
    def noise_budget(self) -> Optional[float]:
        """Remaining noise budget in bits."""
        ...

    @property
    def level(self) -> int:
        """Modulus chain level."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        ...


class HEBackendInterface(ABC):
    """
    Abstract interface for HE backends.

    All HE backends must implement this interface to ensure
    consistent API across implementations.
    """

    def __init__(self, params: Optional[HEParams] = None):
        """
        Initialize backend with parameters.

        Args:
            params: HE parameters (uses defaults if None)
        """
        self.params = params or HEParams()
        self._metrics = HEMetrics()
        self._is_setup = False
        self._keys_generated = False

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
        return self.backend_type not in (HEBackendType.TOY, HEBackendType.AUTO)

    @property
    def is_setup(self) -> bool:
        """Check if backend is set up and ready."""
        return self._is_setup and self._keys_generated

    @abstractmethod
    def setup(self) -> None:
        """
        Set up the HE context.

        Must be called before any operations.
        Generates keys if not already done.
        """
        pass

    @abstractmethod
    def generate_keys(self, generate_galois: bool = True) -> None:
        """
        Generate key material.

        Args:
            generate_galois: Generate Galois keys for rotations
        """
        pass

    @abstractmethod
    def encrypt(self, plaintext: np.ndarray) -> Any:
        """
        Encrypt a plaintext vector.

        Args:
            plaintext: 1D numpy array

        Returns:
            Ciphertext object
        """
        pass

    @abstractmethod
    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        """
        Decrypt a ciphertext.

        Args:
            ciphertext: Ciphertext object
            output_size: Number of elements to return (0 = all)

        Returns:
            Decrypted plaintext as numpy array
        """
        pass

    @abstractmethod
    def add(self, ct1: Any, ct2: Any) -> Any:
        """
        Add two ciphertexts.

        Args:
            ct1: First ciphertext
            ct2: Second ciphertext

        Returns:
            Sum ciphertext
        """
        pass

    @abstractmethod
    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        """
        Multiply ciphertext by plaintext.

        Args:
            ct: Ciphertext
            plaintext: Plaintext vector/scalar

        Returns:
            Product ciphertext
        """
        pass

    @abstractmethod
    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        """
        Encrypted matrix multiplication: ct @ weight^T.

        Args:
            ct: Encrypted input vector
            weight: Plaintext weight matrix

        Returns:
            Encrypted result
        """
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

        Default implementation uses two matmuls.
        Backends may override for optimization.

        Args:
            ct_x: Encrypted activation
            lora_a: LoRA A matrix [rank, in_features]
            lora_b: LoRA B matrix [out_features, rank]
            scaling: LoRA scaling factor

        Returns:
            Encrypted LoRA delta
        """
        # Step 1: ct_x @ A^T
        intermediate = self.matmul(ct_x, lora_a)

        # Step 2: intermediate @ B^T
        result = self.matmul(intermediate, lora_b)

        # Step 3: Apply scaling
        if abs(scaling - 1.0) > 1e-6:
            result = self.multiply_plain(result, np.array([scaling]))

        return result

    def get_slot_count(self) -> int:
        """Get number of SIMD slots available."""
        # Default for CKKS: poly_degree / 2
        return self.params.poly_modulus_degree // 2

    def get_noise_budget(self, ct: Any) -> Optional[float]:
        """Get remaining noise budget of ciphertext."""
        if hasattr(ct, 'noise_budget'):
            return ct.noise_budget
        return None

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
        if not self._keys_generated:
            raise RuntimeError(f"{self.backend_name} backend has no keys. Call generate_keys() first.")


# ==============================================================================
# Toy Backend (for development only)
# ==============================================================================


class ToyHEBackend(HEBackendInterface):
    """
    Toy HE backend for development and testing.

    WARNING: NOT CRYPTOGRAPHICALLY SECURE!

    This provides API compatibility without real encryption.
    Requires explicit opt-in via TENSAFE_TOY_HE=1.
    """

    def __init__(self, params: Optional[HEParams] = None, _force_enable: bool = False):
        super().__init__(params)

        if not _force_enable:
            # Check gate
            if not ProductionGates.TOY_HE.is_allowed():
                raise GateDeniedError(
                    ProductionGates.TOY_HE,
                    "Toy HE requires TENSAFE_TOY_HE=1 and is not allowed in production"
                )

        logger.warning(
            "*** USING ToyHEBackend - NOT CRYPTOGRAPHICALLY SECURE! ***\n"
            "This is for development/testing only."
        )

    @property
    def backend_type(self) -> HEBackendType:
        return HEBackendType.TOY

    @property
    def backend_name(self) -> str:
        return "Toy HE (NOT SECURE)"

    @property
    def is_production_ready(self) -> bool:
        return False

    def setup(self) -> None:
        self._is_setup = True
        self.generate_keys()

    def generate_keys(self, generate_galois: bool = True) -> None:
        # Toy mode: no real keys
        self._keys_generated = True

    def encrypt(self, plaintext: np.ndarray) -> "_ToyCiphertext":
        self.validate_ready()
        self._metrics.operations_count += 1
        return _ToyCiphertext(plaintext.astype(np.float64).copy())

    def decrypt(self, ciphertext: "_ToyCiphertext", output_size: int = 0) -> np.ndarray:
        self.validate_ready()
        if output_size > 0:
            return ciphertext.data[:output_size]
        return ciphertext.data

    def add(self, ct1: "_ToyCiphertext", ct2: "_ToyCiphertext") -> "_ToyCiphertext":
        self.validate_ready()
        self._metrics.operations_count += 1
        return _ToyCiphertext(ct1.data + ct2.data)

    def multiply_plain(self, ct: "_ToyCiphertext", plaintext: np.ndarray) -> "_ToyCiphertext":
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1
        return _ToyCiphertext(ct.data * plaintext.flatten()[0] if plaintext.size == 1 else ct.data * plaintext)

    def matmul(self, ct: "_ToyCiphertext", weight: np.ndarray) -> "_ToyCiphertext":
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1
        result = ct.data @ weight.T
        return _ToyCiphertext(result)


@dataclass
class _ToyCiphertext:
    """Toy ciphertext that just wraps plaintext data."""
    data: np.ndarray
    noise_budget: float = 100.0
    level: int = 0

    def to_bytes(self) -> bytes:
        return self.data.tobytes()


# ==============================================================================
# N2HE Backend Wrapper
# ==============================================================================


class N2HEBackendWrapper(HEBackendInterface):
    """
    Wrapper for the tensorguard.n2he backend.

    Provides unified interface for the N2HE implementation.
    """

    def __init__(self, params: Optional[HEParams] = None):
        super().__init__(params)
        self._context = None

    @property
    def backend_type(self) -> HEBackendType:
        return HEBackendType.N2HE

    @property
    def backend_name(self) -> str:
        return "N2HE"

    def setup(self) -> None:
        try:
            from tensorguard.n2he.core import N2HEContext, HESchemeParams, ToyN2HEScheme

            # Convert params
            n2he_params = HESchemeParams(
                n=self.params.n,
                q=self.params.q,
                t=self.params.t,
                std_dev=self.params.std_dev,
                poly_degree=self.params.poly_modulus_degree,
                coeff_modulus_bits=self.params.coeff_modulus_bits,
                security_level=self.params.security_level,
            )

            # Try native first, fall back to toy
            try:
                from tensorguard.n2he._native import NativeN2HEScheme
                scheme = NativeN2HEScheme(n2he_params)
                logger.info("Using native N2HE scheme")
            except ImportError:
                if not ProductionGates.TOY_HE.is_allowed():
                    raise RuntimeError(
                        "Native N2HE not available and toy mode not allowed. "
                        "Either install native N2HE or set TENSAFE_TOY_HE=1 for development."
                    )
                scheme = ToyN2HEScheme(n2he_params)

            self._context = N2HEContext(params=n2he_params, scheme=scheme)
            self._is_setup = True
            self.generate_keys()

        except ImportError as e:
            raise RuntimeError(f"N2HE backend not available: {e}")

    def generate_keys(self, generate_galois: bool = True) -> None:
        if self._context is None:
            raise RuntimeError("Context not initialized")
        self._context.generate_keys()
        self._keys_generated = True

    def encrypt(self, plaintext: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        return self._context.encrypt(plaintext)

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        self.validate_ready()
        result = self._context.decrypt(ciphertext)
        if output_size > 0:
            return result[:output_size]
        return result

    def add(self, ct1: Any, ct2: Any) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        return self._context.scheme.add(ct1, ct2)

    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1
        return self._context.scheme.multiply(ct, plaintext)

    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1
        return self._context.scheme.matmul(ct, weight, self._context._ek)

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Any:
        self.validate_ready()
        return self._context.encrypted_lora_delta(ct_x, lora_a, lora_b, scaling)


# ==============================================================================
# HEXL Backend Wrapper
# ==============================================================================


class HEXLBackendWrapper(HEBackendInterface):
    """
    Wrapper for the N2HE-HEXL production backend.

    This is the recommended backend for production use with
    MOAI-style optimizations.
    """

    def __init__(self, params: Optional[HEParams] = None):
        super().__init__(params)
        self._backend = None
        self._packed_weights: Dict[str, Any] = {}

    @property
    def backend_type(self) -> HEBackendType:
        return HEBackendType.HEXL

    @property
    def backend_name(self) -> str:
        return "N2HE-HEXL"

    @property
    def is_production_ready(self) -> bool:
        return True

    def setup(self) -> None:
        try:
            from tensafe.he_lora.backend import HEBackend, HEBackendNotAvailableError

            hexl_params = self.params.to_hexl_params()
            self._backend = HEBackend(hexl_params)
            self._backend.setup()

            self._is_setup = True
            self._keys_generated = True

        except ImportError as e:
            raise RuntimeError(f"HEXL backend not available: {e}")

    def generate_keys(self, generate_galois: bool = True) -> None:
        # Keys are generated in setup() for HEXL
        pass

    def encrypt(self, plaintext: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        return self._backend.encrypt(plaintext)

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        self.validate_ready()
        return self._backend.decrypt(ciphertext, output_size)

    def add(self, ct1: Any, ct2: Any) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1

        # Try backend method first
        if hasattr(self._backend, 'add'):
            return self._backend.add(ct1, ct2)

        # Fallback: use SEAL-style addition if available
        if hasattr(self._backend, 'evaluator') and hasattr(self._backend.evaluator, 'add'):
            result = type(ct1)()
            self._backend.evaluator.add(ct1, ct2, result)
            return result

        # Last resort: decrypt, add, re-encrypt (NOT SECURE - for debugging only)
        logger.warning("Using fallback add (not HE-secure)")
        p1 = self.decrypt(ct1)
        p2 = self.decrypt(ct2)
        return self.encrypt(p1 + p2)

    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1

        # Try backend method first
        if hasattr(self._backend, 'multiply_plain'):
            return self._backend.multiply_plain(ct, plaintext)

        # Fallback: use SEAL-style multiplication if available
        if hasattr(self._backend, 'evaluator') and hasattr(self._backend, 'encoder'):
            pt = self._backend.encoder.encode(plaintext, self._backend.scale)
            result = type(ct)()
            self._backend.evaluator.multiply_plain(ct, pt, result)
            return result

        # Last resort: decrypt, multiply, re-encrypt (NOT SECURE - for debugging only)
        logger.warning("Using fallback multiply_plain (not HE-secure)")
        p = self.decrypt(ct)
        return self.encrypt(p * plaintext.flatten()[0] if plaintext.size == 1 else p * plaintext)

    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1

        # Use column packing if enabled and backend supports it
        if self.params.use_column_packing and hasattr(self._backend, 'column_packed_matmul'):
            packed = self._get_or_create_packed(weight)
            return self._backend.column_packed_matmul(ct, packed, rescale=True)

        # Try direct matmul if available
        if hasattr(self._backend, 'matmul'):
            return self._backend.matmul(ct, weight)

        # Fallback: use baby-step giant-step matrix multiplication
        # This is a simplified implementation using rotations and additions
        if hasattr(self._backend, 'evaluator') and hasattr(self._backend, 'galois_keys'):
            return self._bsgs_matmul(ct, weight)

        # Last resort: decrypt, matmul, re-encrypt (NOT SECURE - for debugging only)
        logger.warning("Using fallback matmul (not HE-secure)")
        p = self.decrypt(ct)
        result = p @ weight.T
        return self.encrypt(result)

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Any:
        self.validate_ready()

        # Use MOAI-style column packing
        packed_a = self._get_or_create_packed(lora_a, f"lora_a_{id(lora_a)}")
        packed_b = self._get_or_create_packed(lora_b, f"lora_b_{id(lora_b)}")

        return self._backend.lora_delta(ct_x, packed_a, packed_b, scaling)

    def _get_or_create_packed(self, matrix: np.ndarray, key: Optional[str] = None) -> Any:
        """Get or create column-packed matrix."""
        key = key or f"matrix_{id(matrix)}"
        if key not in self._packed_weights:
            if hasattr(self._backend, 'create_column_packed_matrix'):
                self._packed_weights[key] = self._backend.create_column_packed_matrix(matrix)
            else:
                # Store unpacked if packing not available
                self._packed_weights[key] = matrix
        return self._packed_weights[key]

    def _bsgs_matmul(self, ct: Any, weight: np.ndarray) -> Any:
        """
        Baby-step giant-step matrix multiplication.

        This is a fallback implementation for when column packing is not available.
        Uses rotation and addition to compute ct @ weight^T.
        """
        n, d = weight.shape
        slot_count = self.get_slot_count()

        # For simplicity, use diagonal method
        # result = sum_i(rotate(ct, i) * weight_diagonal_i)
        evaluator = self._backend.evaluator
        encoder = self._backend.encoder
        galois_keys = self._backend.galois_keys

        result = None

        for i in range(min(d, slot_count)):
            # Create diagonal of weight matrix
            diag = np.zeros(slot_count)
            for j in range(min(n, slot_count)):
                diag[(i + j) % slot_count] = weight[j, i] if i < d else 0.0

            # Encode diagonal
            diag_plain = encoder.encode(diag, self._backend.scale)

            # Rotate ciphertext
            if i == 0:
                rotated = ct
            else:
                rotated = type(ct)()
                evaluator.rotate_vector(ct, i, galois_keys, rotated)

            # Multiply by diagonal
            product = type(ct)()
            evaluator.multiply_plain(rotated, diag_plain, product)

            # Accumulate
            if result is None:
                result = product
            else:
                evaluator.add_inplace(result, product)

        return result

    def get_slot_count(self) -> int:
        return self._backend.get_slot_count()

    def get_metrics(self) -> HEMetrics:
        # Get stats from backend
        if self._backend:
            stats = self._backend.get_operation_stats()
            self._metrics.rotations_count = stats.get("rotations", 0)
            self._metrics.multiplications_count = stats.get("multiplications", 0)
        return self._metrics


# ==============================================================================
# CKKS MOAI Backend Wrapper
# ==============================================================================


class CKKSMOAIBackendWrapper(HEBackendInterface):
    """
    Wrapper for the CKKS MOAI backend.

    This backend implements the MOAI (Modular Optimizing Architecture for Inference)
    approach from Digital Trust Center NTU for homomorphic encryption in neural networks.

    Key features:
    - CKKS encryption scheme for approximate arithmetic on floats
    - Column packing for rotation-free plaintext-ciphertext matrix multiplication
    - Consistent packing strategies across layers (no format conversions)
    - Optimized for LoRA/adapter computations

    Uses Pyfhel (Python wrapper for Microsoft SEAL) for CKKS operations.

    Based on: "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
    Transformer Inference" (eprint.iacr.org/2025/991)
    """

    def __init__(self, params: Optional[HEParams] = None):
        super().__init__(params)
        self._backend = None

    @property
    def backend_type(self) -> HEBackendType:
        return HEBackendType.CKKS_MOAI

    @property
    def backend_name(self) -> str:
        return "CKKS-MOAI"

    @property
    def is_production_ready(self) -> bool:
        return True

    def setup(self) -> None:
        try:
            from crypto_backend.ckks_moai import CKKSMOAIBackend, CKKSParams

            # Convert params to CKKS format
            ckks_params = CKKSParams(
                poly_modulus_degree=self.params.poly_modulus_degree,
                coeff_modulus_bits=self.params.coeff_modulus_bits,
                scale_bits=self.params.scale_bits,
                security_level=self.params.security_level,
                use_column_packing=self.params.use_column_packing,
                use_interleaved_batching=self.params.use_interleaved_batching,
            )

            self._backend = CKKSMOAIBackend(ckks_params)
            self._backend.setup_context()
            self._backend.generate_keys()

            self._is_setup = True
            self._keys_generated = True

            logger.info(
                f"CKKS MOAI backend initialized: "
                f"N={self.params.poly_modulus_degree}, "
                f"scale=2^{self.params.scale_bits}"
            )

        except ImportError as e:
            raise RuntimeError(
                f"CKKS MOAI backend not available: {e}\n"
                "Install with: pip install Pyfhel"
            )

    def generate_keys(self, generate_galois: bool = True) -> None:
        # Keys are generated in setup() for CKKS MOAI
        pass

    def encrypt(self, plaintext: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        return self._backend.encrypt(plaintext)

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        self.validate_ready()
        return self._backend.decrypt(ciphertext, output_size)

    def add(self, ct1: Any, ct2: Any) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        return self._backend.add(ct1, ct2)

    def multiply_plain(self, ct: Any, plaintext: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1
        return self._backend.multiply_plain(ct, plaintext)

    def matmul(self, ct: Any, weight: np.ndarray) -> Any:
        self.validate_ready()
        self._metrics.operations_count += 1
        self._metrics.multiplications_count += 1
        return self._backend.matmul(ct, weight)

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Any:
        self.validate_ready()
        return self._backend.lora_delta(ct_x, lora_a, lora_b, scaling)

    def get_slot_count(self) -> int:
        return self._backend.get_slot_count()

    def get_noise_budget(self, ct: Any) -> Optional[float]:
        return self._backend.get_noise_budget(ct)

    def get_metrics(self) -> HEMetrics:
        if self._backend:
            stats = self._backend.get_operation_stats()
            self._metrics.operations_count = stats.get("operations", 0)
            self._metrics.multiplications_count = stats.get("multiplications", 0)
            self._metrics.rotations_count = stats.get("rotations", 0)
            self._metrics.rescale_count = stats.get("rescales", 0)
        return self._metrics


# ==============================================================================
# Backend Factory
# ==============================================================================


def get_backend(
    backend_type: Union[HEBackendType, str] = HEBackendType.AUTO,
    params: Optional[HEParams] = None,
    setup: bool = True,
) -> HEBackendInterface:
    """
    Get an HE backend instance.

    Args:
        backend_type: Type of backend to create (or AUTO to select best)
        params: HE parameters
        setup: Automatically call setup()

    Returns:
        Configured HEBackendInterface

    Raises:
        RuntimeError: If requested backend is not available
    """
    if isinstance(backend_type, str):
        backend_type = HEBackendType(backend_type.lower())

    backend: HEBackendInterface

    if backend_type == HEBackendType.AUTO:
        backend = _auto_select_backend(params)
    elif backend_type == HEBackendType.TOY:
        backend = ToyHEBackend(params)
    elif backend_type == HEBackendType.N2HE:
        backend = N2HEBackendWrapper(params)
    elif backend_type == HEBackendType.HEXL:
        backend = HEXLBackendWrapper(params)
    elif backend_type == HEBackendType.CKKS_MOAI:
        backend = CKKSMOAIBackendWrapper(params)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    if setup:
        backend.setup()

    logger.info(f"Created HE backend: {backend.backend_name}")
    return backend


def _auto_select_backend(params: Optional[HEParams] = None) -> HEBackendInterface:
    """
    Automatically select the best available backend.

    Priority:
    1. CKKS MOAI (Pyfhel, best for neural network float arithmetic)
    2. HEXL (Intel CPU only, production)
    3. N2HE native
    4. N2HE toy (if allowed)
    """
    # Try CKKS MOAI first (best for neural network float arithmetic)
    try:
        backend = CKKSMOAIBackendWrapper(params)
        # Quick availability check
        from crypto_backend.ckks_moai import CKKSMOAIBackend
        return backend
    except (ImportError, RuntimeError):
        logger.debug("CKKS MOAI backend not available")

    # Try HEXL (Intel CPU only)
    try:
        return HEXLBackendWrapper(params)
    except (ImportError, RuntimeError):
        logger.debug("HEXL backend not available")

    # Try N2HE
    try:
        return N2HEBackendWrapper(params)
    except (ImportError, RuntimeError):
        logger.debug("N2HE backend not available")

    # Fall back to toy if allowed
    if ProductionGates.TOY_HE.is_allowed():
        return ToyHEBackend(params)

    raise RuntimeError(
        "No HE backend available. Options:\n"
        "1. Install Pyfhel for CKKS MOAI: pip install Pyfhel (recommended, best for neural networks)\n"
        "2. Install N2HE-HEXL: ./scripts/build_n2he_hexl.sh (Intel CPU only)\n"
        "3. Install N2HE native library\n"
        "4. Enable toy mode: TENSAFE_TOY_HE=1 (development only)"
    )


def is_backend_available(backend_type: Union[HEBackendType, str]) -> bool:
    """
    Check if a backend type is available.

    Args:
        backend_type: Backend to check

    Returns:
        True if backend is available
    """
    if isinstance(backend_type, str):
        backend_type = HEBackendType(backend_type.lower())

    try:
        if backend_type == HEBackendType.TOY:
            return ProductionGates.TOY_HE.is_allowed()
        elif backend_type == HEBackendType.N2HE:
            from tensorguard.n2he.core import N2HEContext
            return True
        elif backend_type == HEBackendType.HEXL:
            from tensafe.he_lora.backend import HEBackend
            return True
        elif backend_type == HEBackendType.CKKS_MOAI:
            from crypto_backend.ckks_moai import CKKSMOAIBackend
            return True
        elif backend_type == HEBackendType.AUTO:
            return True
    except ImportError:
        pass

    return False


def list_available_backends() -> List[str]:
    """
    List all available backend types.

    Returns:
        List of available backend type names
    """
    available = []
    for bt in HEBackendType:
        if bt != HEBackendType.AUTO and is_backend_available(bt):
            available.append(bt.value)
    return available
