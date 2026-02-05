"""
Hybrid HE Backend for CKKS-TFHE Operations

Provides a unified interface for hybrid homomorphic encryption operations
combining CKKS (linear) and TFHE (non-linear) computations.

Trust Model (v1): Interactive Bridge
- Client holds both CKKS and TFHE secret keys
- Server never sees plaintext during bridge operations
- Bridge requires client round-trip for scheme conversion
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from enum import Enum, auto
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class BridgeMode(Enum):
    """Mode for scheme bridging operations."""
    INTERACTIVE = auto()  # Client-assisted decrypt/re-encrypt
    SIMULATION = auto()   # Plaintext simulation for testing
    # Future: NON_INTERACTIVE for OpenFHE scheme switching


@dataclass
class HybridHEConfig:
    """Configuration for hybrid HE backend."""
    # CKKS configuration
    ckks_profile: str = "FAST"
    ckks_scale_bits: int = 40

    # TFHE configuration
    tfhe_noise_budget: float = 128.0
    tfhe_lut_bits: int = 8

    # Bridge configuration
    bridge_mode: BridgeMode = BridgeMode.SIMULATION
    bridge_service_url: Optional[str] = None
    bridge_timeout_ms: int = 5000
    bridge_retry_count: int = 3

    # Quantization defaults
    quantization_bits: int = 8
    quantization_clip_min: float = -10.0
    quantization_clip_max: float = 10.0
    quantization_symmetric: bool = True

    # Telemetry
    enable_telemetry: bool = True


@dataclass
class HybridOperationStats:
    """Statistics for hybrid HE operations."""
    # Counts
    ckks_ops: int = 0
    tfhe_ops: int = 0
    bridge_to_tfhe_count: int = 0
    bridge_to_ckks_count: int = 0
    bootstrap_count: int = 0

    # Timing (cumulative, milliseconds)
    ckks_time_ms: float = 0.0
    tfhe_time_ms: float = 0.0
    bridge_time_ms: float = 0.0

    # Errors
    quantization_error_total: float = 0.0
    quantization_error_max: float = 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.ckks_ops = 0
        self.tfhe_ops = 0
        self.bridge_to_tfhe_count = 0
        self.bridge_to_ckks_count = 0
        self.bootstrap_count = 0
        self.ckks_time_ms = 0.0
        self.tfhe_time_ms = 0.0
        self.bridge_time_ms = 0.0
        self.quantization_error_total = 0.0
        self.quantization_error_max = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ckks_ops": self.ckks_ops,
            "tfhe_ops": self.tfhe_ops,
            "bridge_to_tfhe_count": self.bridge_to_tfhe_count,
            "bridge_to_ckks_count": self.bridge_to_ckks_count,
            "bootstrap_count": self.bootstrap_count,
            "ckks_time_ms": self.ckks_time_ms,
            "tfhe_time_ms": self.tfhe_time_ms,
            "bridge_time_ms": self.bridge_time_ms,
            "quantization_error_total": self.quantization_error_total,
            "quantization_error_max": self.quantization_error_max,
        }


@dataclass
class CKKSCiphertext:
    """Representation of a CKKS ciphertext."""
    data: Any  # Actual ciphertext data (backend-specific)
    shape: Tuple[int, ...]
    scale: float
    level: int = 0
    is_moai_packed: bool = False

    # For simulation mode
    _plaintext: Optional[np.ndarray] = None


@dataclass
class TFHECiphertext:
    """Representation of a TFHE ciphertext."""
    data: Any  # Actual ciphertext data (backend-specific)
    shape: Tuple[int, ...]
    value_bits: int = 8
    noise_budget: float = 128.0

    # For simulation mode
    _plaintext: Optional[np.ndarray] = None


# =============================================================================
# Backend Interfaces
# =============================================================================

class CKKSBackendInterface(Protocol):
    """Protocol for CKKS backend implementations."""

    def encrypt(self, plaintext: np.ndarray) -> CKKSCiphertext: ...
    def decrypt(self, ciphertext: CKKSCiphertext) -> np.ndarray: ...
    def ct_pt_matvec(self, ct: CKKSCiphertext, matrix: np.ndarray) -> CKKSCiphertext: ...
    def ct_ct_mul(self, ct1: CKKSCiphertext, ct2: CKKSCiphertext) -> CKKSCiphertext: ...
    def ct_ct_add(self, ct1: CKKSCiphertext, ct2: CKKSCiphertext) -> CKKSCiphertext: ...
    def rescale(self, ct: CKKSCiphertext) -> CKKSCiphertext: ...


class TFHEBackendInterface(Protocol):
    """Protocol for TFHE backend implementations."""

    def encrypt(self, plaintext: np.ndarray, bits: int) -> TFHECiphertext: ...
    def decrypt(self, ciphertext: TFHECiphertext) -> np.ndarray: ...
    def apply_lut(self, ct: TFHECiphertext, lut: List[int]) -> TFHECiphertext: ...
    def bootstrap(self, ct: TFHECiphertext) -> TFHECiphertext: ...


class SchemeBridgeServiceClient(Protocol):
    """Protocol for scheme bridge service client."""

    def ckks_to_tfhe(
        self,
        ct_ckks: CKKSCiphertext,
        request_id: str,
        quant_bits: int,
    ) -> TFHECiphertext: ...

    def tfhe_to_ckks(
        self,
        ct_tfhe: TFHECiphertext,
        request_id: str,
    ) -> CKKSCiphertext: ...


# =============================================================================
# Simulation Backends
# =============================================================================

class SimulatedCKKSBackend:
    """Simulated CKKS backend for testing."""

    def __init__(self, scale_bits: int = 40):
        self.scale = 2.0 ** scale_bits
        self._operation_count = 0

    def encrypt(self, plaintext: np.ndarray) -> CKKSCiphertext:
        """Simulate encryption (just wrap plaintext)."""
        return CKKSCiphertext(
            data=None,
            shape=plaintext.shape,
            scale=self.scale,
            _plaintext=plaintext.astype(np.float64),
        )

    def decrypt(self, ciphertext: CKKSCiphertext) -> np.ndarray:
        """Simulate decryption (return stored plaintext)."""
        if ciphertext._plaintext is None:
            raise ValueError("Cannot decrypt: no plaintext in simulation mode")
        return ciphertext._plaintext

    def ct_pt_matvec(
        self,
        ct: CKKSCiphertext,
        matrix: np.ndarray,
    ) -> CKKSCiphertext:
        """Simulate ciphertext-plaintext matrix-vector multiply."""
        self._operation_count += 1
        pt = self.decrypt(ct)
        result = matrix @ pt
        return CKKSCiphertext(
            data=None,
            shape=result.shape,
            scale=self.scale,
            level=ct.level,
            _plaintext=result,
        )

    def ct_ct_mul(
        self,
        ct1: CKKSCiphertext,
        ct2: CKKSCiphertext,
    ) -> CKKSCiphertext:
        """Simulate ciphertext-ciphertext multiply."""
        self._operation_count += 1
        pt1 = self.decrypt(ct1)
        pt2 = self.decrypt(ct2)

        # Handle scalar * vector broadcast
        if pt1.shape != pt2.shape:
            if pt1.size == 1:
                result = float(pt1.flat[0]) * pt2
            elif pt2.size == 1:
                result = pt1 * float(pt2.flat[0])
            else:
                result = pt1 * pt2
        else:
            result = pt1 * pt2

        return CKKSCiphertext(
            data=None,
            shape=result.shape,
            scale=self.scale,
            level=max(ct1.level, ct2.level),
            _plaintext=result,
        )

    def ct_ct_add(
        self,
        ct1: CKKSCiphertext,
        ct2: CKKSCiphertext,
    ) -> CKKSCiphertext:
        """Simulate ciphertext-ciphertext add."""
        self._operation_count += 1
        pt1 = self.decrypt(ct1)
        pt2 = self.decrypt(ct2)
        result = pt1 + pt2
        return CKKSCiphertext(
            data=None,
            shape=result.shape,
            scale=self.scale,
            level=max(ct1.level, ct2.level),
            _plaintext=result,
        )

    def rescale(self, ct: CKKSCiphertext) -> CKKSCiphertext:
        """Simulate rescaling (no-op in simulation)."""
        return CKKSCiphertext(
            data=ct.data,
            shape=ct.shape,
            scale=self.scale,
            level=ct.level + 1,
            _plaintext=ct._plaintext,
        )


class SimulatedTFHEBackend:
    """Simulated TFHE backend for testing."""

    def __init__(self, noise_budget: float = 128.0):
        self.default_noise_budget = noise_budget
        self._operation_count = 0

    def encrypt(self, plaintext: np.ndarray, bits: int = 8) -> TFHECiphertext:
        """Simulate encryption."""
        return TFHECiphertext(
            data=None,
            shape=plaintext.shape,
            value_bits=bits,
            noise_budget=self.default_noise_budget,
            _plaintext=plaintext.astype(np.int32),
        )

    def decrypt(self, ciphertext: TFHECiphertext) -> np.ndarray:
        """Simulate decryption."""
        if ciphertext._plaintext is None:
            raise ValueError("Cannot decrypt: no plaintext in simulation mode")
        return ciphertext._plaintext

    def apply_lut(self, ct: TFHECiphertext, lut: List[int]) -> TFHECiphertext:
        """Simulate LUT application via programmable bootstrapping."""
        self._operation_count += 1
        pt = self.decrypt(ct)
        lut_arr = np.array(lut)

        # Interpret as signed integers for LUT indexing
        lut_size = len(lut)
        offset = lut_size // 2

        # Map signed values to LUT indices
        indices = (pt + offset).astype(int)
        indices = np.clip(indices, 0, lut_size - 1)

        result = lut_arr[indices]

        return TFHECiphertext(
            data=None,
            shape=result.shape,
            value_bits=max(1, int(np.ceil(np.log2(max(lut) + 1)))) if max(lut) > 0 else 1,
            noise_budget=self.default_noise_budget,  # Refreshed by bootstrap
            _plaintext=result,
        )

    def bootstrap(self, ct: TFHECiphertext) -> TFHECiphertext:
        """Simulate bootstrapping (refresh noise)."""
        self._operation_count += 1
        return TFHECiphertext(
            data=ct.data,
            shape=ct.shape,
            value_bits=ct.value_bits,
            noise_budget=self.default_noise_budget,
            _plaintext=ct._plaintext,
        )


class SimulatedBridgeService:
    """Simulated bridge service for testing."""

    def __init__(
        self,
        quant_bits: int = 8,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
    ):
        self.quant_bits = quant_bits
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._stats = {
            "total_error": 0.0,
            "max_error": 0.0,
            "conversions": 0,
        }

    def _quantize(self, value: float) -> int:
        """Quantize a float to integer."""
        clamped = max(self.clip_min, min(self.clip_max, value))
        max_int = (1 << (self.quant_bits - 1)) - 1
        max_abs = max(abs(self.clip_min), abs(self.clip_max))
        scale = max_int / max_abs
        return int(round(clamped * scale))

    def _dequantize(self, value: int) -> float:
        """Dequantize integer to float."""
        max_int = (1 << (self.quant_bits - 1)) - 1
        max_abs = max(abs(self.clip_min), abs(self.clip_max))
        scale = max_int / max_abs
        return value / scale

    def ckks_to_tfhe(
        self,
        ct_ckks: CKKSCiphertext,
        request_id: str,
        quant_bits: int,
    ) -> TFHECiphertext:
        """Simulate CKKS to TFHE conversion."""
        pt = ct_ckks._plaintext
        if pt is None:
            raise ValueError("No plaintext available in simulation mode")

        # Quantize
        quantized = np.vectorize(self._quantize)(pt)

        # Track error
        dequantized = np.vectorize(self._dequantize)(quantized)
        error = np.max(np.abs(pt - dequantized))
        self._stats["total_error"] += float(np.sum(np.abs(pt - dequantized)))
        self._stats["max_error"] = max(self._stats["max_error"], error)
        self._stats["conversions"] += 1

        return TFHECiphertext(
            data=None,
            shape=quantized.shape,
            value_bits=quant_bits,
            noise_budget=128.0,
            _plaintext=quantized,
        )

    def tfhe_to_ckks(
        self,
        ct_tfhe: TFHECiphertext,
        request_id: str,
    ) -> CKKSCiphertext:
        """Simulate TFHE to CKKS conversion."""
        pt = ct_tfhe._plaintext
        if pt is None:
            raise ValueError("No plaintext available in simulation mode")

        # For bit outputs, keep as 0/1 floats
        if ct_tfhe.value_bits == 1:
            result = pt.astype(np.float64)
        else:
            # Dequantize
            result = np.vectorize(self._dequantize)(pt)

        self._stats["conversions"] += 1

        return CKKSCiphertext(
            data=None,
            shape=result.shape,
            scale=2.0 ** 40,
            _plaintext=result,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return dict(self._stats)


# =============================================================================
# Hybrid Backend Implementation
# =============================================================================

class HybridHEBackend:
    """
    Production backend for hybrid CKKS-TFHE operations.

    Responsibilities:
    - CKKS linear operations (delegated to CKKS backend)
    - TFHE LUT evaluation via programmable bootstrapping
    - Scheme bridging (interactive or simulation)
    - Telemetry collection

    Usage:
        backend = HybridHEBackend.create_simulated()  # For testing
        # or
        backend = HybridHEBackend(ckks_backend, tfhe_backend, bridge_service, config)

        # CKKS operations
        ct_result = backend.ckks_matvec(ct_x, matrix)

        # Bridge to TFHE
        ct_tfhe = backend.bridge_ckks_to_tfhe(ct_ckks, "req_123")

        # Apply LUT
        ct_gate = backend.tfhe_lut_apply(ct_tfhe, "step")

        # Bridge back
        ct_ckks_gate = backend.bridge_tfhe_to_ckks(ct_gate, "req_123")
    """

    def __init__(
        self,
        ckks_backend: CKKSBackendInterface,
        tfhe_backend: TFHEBackendInterface,
        bridge_service: SchemeBridgeServiceClient,
        lut_library: Optional[Any] = None,
        config: Optional[HybridHEConfig] = None,
    ):
        self._ckks = ckks_backend
        self._tfhe = tfhe_backend
        self._bridge = bridge_service
        self._config = config or HybridHEConfig()
        self._stats = HybridOperationStats()

        # Initialize LUT library
        if lut_library is None:
            from ..tfhe_lut import LUTLibrary
            self._lut_library = LUTLibrary(self._config.tfhe_lut_bits)
        else:
            self._lut_library = lut_library

    @classmethod
    def create_simulated(
        cls,
        config: Optional[HybridHEConfig] = None,
    ) -> "HybridHEBackend":
        """Create a simulated backend for testing."""
        config = config or HybridHEConfig(bridge_mode=BridgeMode.SIMULATION)

        ckks_backend = SimulatedCKKSBackend(config.ckks_scale_bits)
        tfhe_backend = SimulatedTFHEBackend(config.tfhe_noise_budget)
        bridge_service = SimulatedBridgeService(
            quant_bits=config.quantization_bits,
            clip_min=config.quantization_clip_min,
            clip_max=config.quantization_clip_max,
        )

        return cls(
            ckks_backend=ckks_backend,
            tfhe_backend=tfhe_backend,
            bridge_service=bridge_service,
            config=config,
        )

    # -------------------------------------------------------------------------
    # CKKS Operations
    # -------------------------------------------------------------------------

    def ckks_encrypt(self, plaintext: np.ndarray) -> CKKSCiphertext:
        """Encrypt plaintext to CKKS ciphertext."""
        start = time.perf_counter()
        result = self._ckks.encrypt(plaintext)
        if self._config.enable_telemetry:
            self._stats.ckks_time_ms += (time.perf_counter() - start) * 1000
            self._stats.ckks_ops += 1
        return result

    def ckks_decrypt(self, ciphertext: CKKSCiphertext) -> np.ndarray:
        """Decrypt CKKS ciphertext."""
        start = time.perf_counter()
        result = self._ckks.decrypt(ciphertext)
        if self._config.enable_telemetry:
            self._stats.ckks_time_ms += (time.perf_counter() - start) * 1000
            self._stats.ckks_ops += 1
        return result

    def ckks_matvec(
        self,
        ct_x: CKKSCiphertext,
        pt_matrix: np.ndarray,
    ) -> CKKSCiphertext:
        """
        Compute ciphertext-plaintext matrix-vector product.

        Args:
            ct_x: CKKS ciphertext containing vector
            pt_matrix: Plaintext matrix

        Returns:
            CKKS ciphertext containing result
        """
        start = time.perf_counter()
        result = self._ckks.ct_pt_matvec(ct_x, pt_matrix)
        if self._config.enable_telemetry:
            self._stats.ckks_time_ms += (time.perf_counter() - start) * 1000
            self._stats.ckks_ops += 1
        return result

    def ckks_matmul(
        self,
        ct_x: CKKSCiphertext,
        pt_A: np.ndarray,
        pt_B: np.ndarray,
    ) -> CKKSCiphertext:
        """
        Compute x @ A^T @ B^T (for LoRA: delta = B(Ax)).

        Args:
            ct_x: CKKS ciphertext containing input vector
            pt_A: LoRA A matrix [rank, hidden_size]
            pt_B: LoRA B matrix [hidden_size, rank]

        Returns:
            CKKS ciphertext containing delta
        """
        start = time.perf_counter()

        # u = A @ x (using A.T for right-multiply form)
        ct_u = self._ckks.ct_pt_matvec(ct_x, pt_A)

        # delta = B @ u
        ct_delta = self._ckks.ct_pt_matvec(ct_u, pt_B)

        if self._config.enable_telemetry:
            self._stats.ckks_time_ms += (time.perf_counter() - start) * 1000
            self._stats.ckks_ops += 2

        return ct_delta

    def ckks_scalar_mul(
        self,
        ct_x: CKKSCiphertext,
        ct_scalar: CKKSCiphertext,
    ) -> CKKSCiphertext:
        """
        Multiply ciphertext by scalar ciphertext.

        Args:
            ct_x: CKKS ciphertext (vector)
            ct_scalar: CKKS ciphertext (scalar)

        Returns:
            CKKS ciphertext containing scaled result
        """
        start = time.perf_counter()
        result = self._ckks.ct_ct_mul(ct_x, ct_scalar)
        if self._config.enable_telemetry:
            self._stats.ckks_time_ms += (time.perf_counter() - start) * 1000
            self._stats.ckks_ops += 1
        return result

    def ckks_add(
        self,
        ct_a: CKKSCiphertext,
        ct_b: CKKSCiphertext,
    ) -> CKKSCiphertext:
        """Add two CKKS ciphertexts."""
        start = time.perf_counter()
        result = self._ckks.ct_ct_add(ct_a, ct_b)
        if self._config.enable_telemetry:
            self._stats.ckks_time_ms += (time.perf_counter() - start) * 1000
            self._stats.ckks_ops += 1
        return result

    # -------------------------------------------------------------------------
    # TFHE Operations
    # -------------------------------------------------------------------------

    def tfhe_encrypt(self, plaintext: np.ndarray, bits: int = 8) -> TFHECiphertext:
        """Encrypt plaintext to TFHE ciphertext."""
        start = time.perf_counter()
        result = self._tfhe.encrypt(plaintext, bits)
        if self._config.enable_telemetry:
            self._stats.tfhe_time_ms += (time.perf_counter() - start) * 1000
            self._stats.tfhe_ops += 1
        return result

    def tfhe_decrypt(self, ciphertext: TFHECiphertext) -> np.ndarray:
        """Decrypt TFHE ciphertext."""
        start = time.perf_counter()
        result = self._tfhe.decrypt(ciphertext)
        if self._config.enable_telemetry:
            self._stats.tfhe_time_ms += (time.perf_counter() - start) * 1000
            self._stats.tfhe_ops += 1
        return result

    def tfhe_lut_apply(
        self,
        ct_tfhe: TFHECiphertext,
        lut_id: str,
    ) -> TFHECiphertext:
        """
        Apply LUT via programmable bootstrapping.

        Args:
            ct_tfhe: TFHE ciphertext (scalar or small vector, max 16 elements)
            lut_id: Registered LUT identifier (e.g., "step", "sign")

        Returns:
            TFHE ciphertext with LUT applied and noise refreshed
        """
        start = time.perf_counter()

        # Get LUT entries
        lut = self._lut_library.get(lut_id)
        if lut is None:
            raise ValueError(f"Unknown LUT: {lut_id}")

        result = self._tfhe.apply_lut(ct_tfhe, lut.entries)

        if self._config.enable_telemetry:
            self._stats.tfhe_time_ms += (time.perf_counter() - start) * 1000
            self._stats.tfhe_ops += 1
            self._stats.bootstrap_count += 1

        return result

    def tfhe_lut_apply_custom(
        self,
        ct_tfhe: TFHECiphertext,
        lut_data: List[int],
    ) -> TFHECiphertext:
        """
        Apply custom LUT via programmable bootstrapping.

        Args:
            ct_tfhe: TFHE ciphertext
            lut_data: Custom LUT entries

        Returns:
            TFHE ciphertext with LUT applied
        """
        start = time.perf_counter()

        result = self._tfhe.apply_lut(ct_tfhe, lut_data)

        if self._config.enable_telemetry:
            self._stats.tfhe_time_ms += (time.perf_counter() - start) * 1000
            self._stats.tfhe_ops += 1
            self._stats.bootstrap_count += 1

        return result

    # -------------------------------------------------------------------------
    # Bridge Operations
    # -------------------------------------------------------------------------

    def bridge_ckks_to_tfhe(
        self,
        ct_ckks: CKKSCiphertext,
        request_id: str,
        quantization_bits: Optional[int] = None,
    ) -> TFHECiphertext:
        """
        Convert CKKS ciphertext to TFHE via interactive bridge.

        In production (INTERACTIVE mode):
        - Sends CKKS ciphertext to client
        - Client decrypts, quantizes, re-encrypts with TFHE key
        - Client sends back TFHE ciphertext

        In simulation mode:
        - Directly quantizes plaintext and wraps

        Args:
            ct_ckks: CKKS ciphertext to convert
            request_id: Request ID for session binding
            quantization_bits: Override default quantization bits

        Returns:
            TFHE ciphertext

        Raises:
            HybridExecutionError: If bridge fails
        """
        start = time.perf_counter()
        bits = quantization_bits or self._config.quantization_bits

        try:
            result = self._bridge.ckks_to_tfhe(ct_ckks, request_id, bits)

            if self._config.enable_telemetry:
                self._stats.bridge_time_ms += (time.perf_counter() - start) * 1000
                self._stats.bridge_to_tfhe_count += 1

            return result

        except Exception as e:
            logger.error(f"Bridge CKKS->TFHE failed: {e}")
            raise HybridExecutionError(f"Bridge CKKS->TFHE failed: {e}") from e

    def bridge_tfhe_to_ckks(
        self,
        ct_tfhe: TFHECiphertext,
        request_id: str,
    ) -> CKKSCiphertext:
        """
        Convert TFHE ciphertext back to CKKS via interactive bridge.

        Args:
            ct_tfhe: TFHE ciphertext to convert
            request_id: Request ID for session binding

        Returns:
            CKKS ciphertext

        Raises:
            HybridExecutionError: If bridge fails
        """
        start = time.perf_counter()

        try:
            result = self._bridge.tfhe_to_ckks(ct_tfhe, request_id)

            if self._config.enable_telemetry:
                self._stats.bridge_time_ms += (time.perf_counter() - start) * 1000
                self._stats.bridge_to_ckks_count += 1

            return result

        except Exception as e:
            logger.error(f"Bridge TFHE->CKKS failed: {e}")
            raise HybridExecutionError(f"Bridge TFHE->CKKS failed: {e}") from e

    # -------------------------------------------------------------------------
    # Telemetry
    # -------------------------------------------------------------------------

    def get_operation_stats(self) -> HybridOperationStats:
        """Get accumulated operation statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset operation statistics."""
        self._stats.reset()


# =============================================================================
# Exceptions
# =============================================================================

class HybridExecutionError(Exception):
    """Raised when hybrid HE execution fails."""
    pass


class HybridNotAvailableError(Exception):
    """Raised when hybrid mode requested but not available."""
    pass
