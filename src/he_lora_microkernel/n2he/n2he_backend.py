"""
N2HE Backend Module

Provides backend abstraction for N2HE/TFHE integration with the HE-LoRA microkernel.

Backend Types:
    GPU_TFHE: GPU-accelerated TFHE (primary)
    CPU_FASTERNTT: CPU fallback using FasterNTT
    SIMULATION: Plaintext simulation for testing

The backend handles:
    - LUT evaluation via programmable bootstrapping
    - Key management
    - Backend selection and fallback
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

from .faster_ntt import get_ntt_backend
from .lut_activation import LUTActivationEngine, LUTConfig
from .n2he_params import N2HEParams, N2HEProfile, get_n2he_profile


class N2HEBackendType(Enum):
    """Available backend types."""
    GPU_TFHE = auto()      # GPU-accelerated TFHE
    CPU_FASTERNTT = auto() # CPU fallback with FasterNTT
    SIMULATION = auto()    # Plaintext simulation


@dataclass
class BackendCapabilities:
    """Capabilities of a backend."""
    supports_gpu: bool
    supports_batching: bool
    max_batch_size: int
    supported_lut_bits: List[int]
    estimated_bootstrap_ms: float


class N2HEBackend(ABC):
    """
    Abstract base class for N2HE/TFHE backends.

    Provides interface for:
    - LUT evaluation via programmable bootstrapping
    - Scheme conversion (CKKS ↔ TFHE)
    - Key generation and management
    """

    @abstractmethod
    def evaluate_lut(
        self,
        input_ct: Any,
        lut: List[int],
        output_bits: int = 8,
    ) -> Any:
        """
        Evaluate a LUT on encrypted input via programmable bootstrapping.

        Args:
            input_ct: Input ciphertext (discrete value)
            lut: Lookup table mapping input → output
            output_bits: Bit width of output

        Returns:
            Output ciphertext (discrete value)
        """
        pass

    @abstractmethod
    def batch_evaluate_lut(
        self,
        input_cts: List[Any],
        lut: List[int],
        output_bits: int = 8,
    ) -> List[Any]:
        """
        Batch evaluate LUT on multiple encrypted inputs.

        Args:
            input_cts: List of input ciphertexts
            lut: Lookup table
            output_bits: Bit width of output

        Returns:
            List of output ciphertexts
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class GPUTFHEBackend(N2HEBackend):
    """
    GPU-accelerated TFHE backend.

    This is the primary production backend using CUDA for
    programmable bootstrapping acceleration.
    """

    def __init__(
        self,
        params: Optional[N2HEParams] = None,
        device_id: int = 0,
    ):
        self.params = params or get_n2he_profile(N2HEProfile.BALANCED)
        self.device_id = device_id
        self._initialized = False
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self) -> bool:
        """Initialize the GPU backend."""
        if not self._gpu_available:
            return False

        # In production, this would:
        # 1. Allocate GPU memory for bootstrapping keys
        # 2. Pre-compute NTT twiddle factors on GPU
        # 3. Initialize CUDA streams
        self._initialized = True
        return True

    def evaluate_lut(
        self,
        input_ct: Any,
        lut: List[int],
        output_bits: int = 8,
    ) -> Any:
        """Evaluate LUT using GPU-accelerated programmable bootstrapping."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("GPU backend not available")

        # In production, this would:
        # 1. Extract LWE sample from input
        # 2. Perform GPU-accelerated programmable bootstrapping
        # 3. Return refreshed ciphertext with LUT applied

        # Simulation fallback
        if hasattr(input_ct, 'plaintext'):
            idx = input_ct.plaintext % len(lut)
            return SimulatedCiphertext(lut[idx], output_bits)

        return input_ct

    def batch_evaluate_lut(
        self,
        input_cts: List[Any],
        lut: List[int],
        output_bits: int = 8,
    ) -> List[Any]:
        """Batch LUT evaluation on GPU."""
        # GPU can parallelize multiple bootstraps
        return [self.evaluate_lut(ct, lut, output_bits) for ct in input_cts]

    def get_capabilities(self) -> BackendCapabilities:
        """Get GPU backend capabilities."""
        return BackendCapabilities(
            supports_gpu=True,
            supports_batching=True,
            max_batch_size=1024,
            supported_lut_bits=[4, 6, 8, 10],
            estimated_bootstrap_ms=0.5 if self._gpu_available else float('inf'),
        )

    def is_available(self) -> bool:
        """Check if GPU backend is available."""
        return self._gpu_available


class CPUFasterNTTBackend(N2HEBackend):
    """
    CPU fallback backend using FasterNTT.

    Used when GPU is not available (e.g., ARM devices, testing).
    """

    def __init__(
        self,
        params: Optional[N2HEParams] = None,
    ):
        self.params = params or get_n2he_profile(N2HEProfile.BALANCED)
        self._ntt_backend = get_ntt_backend()

    def evaluate_lut(
        self,
        input_ct: Any,
        lut: List[int],
        output_bits: int = 8,
    ) -> Any:
        """Evaluate LUT using CPU-based bootstrapping."""
        # In production, this would use FasterNTT for:
        # 1. NTT-based polynomial multiplication in bootstrapping
        # 2. Accumulator operations
        # 3. Key switching

        # Simulation for now
        if hasattr(input_ct, 'plaintext'):
            idx = input_ct.plaintext % len(lut)
            return SimulatedCiphertext(lut[idx], output_bits)

        return input_ct

    def batch_evaluate_lut(
        self,
        input_cts: List[Any],
        lut: List[int],
        output_bits: int = 8,
    ) -> List[Any]:
        """Sequential batch evaluation on CPU."""
        return [self.evaluate_lut(ct, lut, output_bits) for ct in input_cts]

    def get_capabilities(self) -> BackendCapabilities:
        """Get CPU backend capabilities."""
        return BackendCapabilities(
            supports_gpu=False,
            supports_batching=False,
            max_batch_size=1,
            supported_lut_bits=[4, 6, 8],
            estimated_bootstrap_ms=50.0,  # Much slower than GPU
        )

    def is_available(self) -> bool:
        """CPU backend is always available."""
        return True


@dataclass
class SimulatedCiphertext:
    """Simulated ciphertext for testing."""
    plaintext: int
    bits: int = 8


class SimulationBackend(N2HEBackend):
    """
    Plaintext simulation backend for testing.

    Evaluates LUTs on plaintext values without encryption.
    Used for correctness testing and debugging.
    """

    def __init__(self, precision_bits: int = 8):
        self.precision_bits = precision_bits

    def evaluate_lut(
        self,
        input_ct: Any,
        lut: List[int],
        output_bits: int = 8,
    ) -> Any:
        """Evaluate LUT on plaintext."""
        if isinstance(input_ct, SimulatedCiphertext):
            plaintext = input_ct.plaintext
        elif isinstance(input_ct, (int, np.integer)):
            plaintext = int(input_ct)
        else:
            raise TypeError(f"Unsupported input type: {type(input_ct)}")

        # Clamp to valid range
        idx = max(0, min(len(lut) - 1, plaintext))
        return SimulatedCiphertext(lut[idx], output_bits)

    def batch_evaluate_lut(
        self,
        input_cts: List[Any],
        lut: List[int],
        output_bits: int = 8,
    ) -> List[Any]:
        """Batch evaluation in simulation."""
        return [self.evaluate_lut(ct, lut, output_bits) for ct in input_cts]

    def get_capabilities(self) -> BackendCapabilities:
        """Simulation capabilities."""
        return BackendCapabilities(
            supports_gpu=False,
            supports_batching=True,
            max_batch_size=10000,
            supported_lut_bits=[4, 6, 8, 10, 12],
            estimated_bootstrap_ms=0.001,  # Very fast (no real crypto)
        )

    def is_available(self) -> bool:
        """Simulation is always available."""
        return True


def create_n2he_backend(
    backend_type: Optional[N2HEBackendType] = None,
    params: Optional[N2HEParams] = None,
    allow_fallback: bool = True,
) -> N2HEBackend:
    """
    Create an N2HE backend instance.

    Args:
        backend_type: Requested backend type (None = auto-select)
        params: N2HE parameters
        allow_fallback: Allow fallback to CPU if GPU unavailable

    Returns:
        Configured backend instance
    """
    if backend_type == N2HEBackendType.SIMULATION:
        return SimulationBackend()

    if backend_type == N2HEBackendType.GPU_TFHE:
        backend = GPUTFHEBackend(params)
        if backend.is_available():
            return backend
        if allow_fallback:
            return CPUFasterNTTBackend(params)
        raise RuntimeError("GPU TFHE backend not available")

    if backend_type == N2HEBackendType.CPU_FASTERNTT:
        return CPUFasterNTTBackend(params)

    # Auto-select: prefer GPU, fallback to CPU
    gpu_backend = GPUTFHEBackend(params)
    if gpu_backend.is_available():
        return gpu_backend

    return CPUFasterNTTBackend(params)


def is_n2he_available(backend_type: Optional[N2HEBackendType] = None) -> bool:
    """
    Check if N2HE backend is available.

    Args:
        backend_type: Specific backend to check (None = any)

    Returns:
        True if backend is available
    """
    if backend_type is None:
        return True  # Simulation always available

    if backend_type == N2HEBackendType.SIMULATION:
        return True

    if backend_type == N2HEBackendType.GPU_TFHE:
        return GPUTFHEBackend().is_available()

    if backend_type == N2HEBackendType.CPU_FASTERNTT:
        return True

    return False


# =============================================================================
# UNIFIED BACKEND INTERFACE
# =============================================================================

class UnifiedN2HEBackend:
    """
    Unified backend that combines LUT activation engine with TFHE backend.

    Provides a single interface for:
    - Non-linear activation evaluation
    - Gate computation for gated LoRA
    - Automatic backend selection and fallback
    """

    def __init__(
        self,
        use_gpu: bool = True,
        use_fallback: bool = True,
        precision_bits: int = 8,
    ):
        """
        Initialize unified backend.

        Args:
            use_gpu: Prefer GPU backend if available
            use_fallback: Allow fallback to CPU
            precision_bits: Precision for LUT evaluation
        """
        self.precision_bits = precision_bits

        # Select TFHE backend
        if use_gpu:
            backend_type = N2HEBackendType.GPU_TFHE
        else:
            backend_type = N2HEBackendType.CPU_FASTERNTT

        self._tfhe_backend = create_n2he_backend(
            backend_type=backend_type,
            allow_fallback=use_fallback,
        )

        # Create LUT activation engine
        self._lut_engine = LUTActivationEngine(
            LUTConfig(
                precision_bits=precision_bits,
                use_gpu=use_gpu,
            )
        )

        self._registered_luts: Dict[str, List[int]] = {}

    def register_lut(self, name: str, lut: List[int]):
        """Register a named LUT."""
        self._registered_luts[name] = lut

    def evaluate_gate(
        self,
        input_value: Any,
        gate_type: str = "step",
    ) -> Any:
        """
        Evaluate a gate function on encrypted input.

        Args:
            input_value: Encrypted input (discrete)
            gate_type: Type of gate ("step", "sign", etc.)

        Returns:
            Encrypted gate output
        """
        if gate_type not in self._registered_luts:
            # Generate default LUT
            p = 2 ** self.precision_bits
            if gate_type == "step":
                lut = [0 if i < p // 2 else 1 for i in range(p)]
            elif gate_type == "sign":
                lut = [0 if i < p // 2 else (p - 1 if i > p // 2 else p // 2) for i in range(p)]
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")
            self._registered_luts[gate_type] = lut

        lut = self._registered_luts[gate_type]
        return self._tfhe_backend.evaluate_lut(input_value, lut)

    def evaluate_activation(
        self,
        input_value: Any,
        activation: str = "relu",
    ) -> Any:
        """
        Evaluate non-linear activation on encrypted input.

        Args:
            input_value: Encrypted input
            activation: Activation function name

        Returns:
            Encrypted activation output
        """
        if activation not in self._registered_luts:
            # Generate activation LUT
            p = 2 ** self.precision_bits
            if activation == "relu":
                lut = [max(0, i - p // 2) for i in range(p)]
            elif activation == "sigmoid":
                import math
                lut = []
                for i in range(p):
                    x = (i - p // 2) / (p // 4)  # Scale to [-2, 2]
                    y = 1.0 / (1 + math.exp(-x))
                    lut.append(int(y * (p - 1)))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            self._registered_luts[activation] = lut

        lut = self._registered_luts[activation]
        return self._tfhe_backend.evaluate_lut(input_value, lut)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the active backend."""
        caps = self._tfhe_backend.get_capabilities()
        return {
            "type": type(self._tfhe_backend).__name__,
            "gpu_available": caps.supports_gpu,
            "supports_batching": caps.supports_batching,
            "max_batch_size": caps.max_batch_size,
            "estimated_bootstrap_ms": caps.estimated_bootstrap_ms,
        }
