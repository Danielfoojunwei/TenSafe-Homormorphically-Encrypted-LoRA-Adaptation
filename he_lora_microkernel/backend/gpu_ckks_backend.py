"""
GPU CKKS Backend Abstraction for HE-LoRA Microkernel

This module provides a backend-agnostic API for GPU-accelerated CKKS operations.
The design supports multiple GPU HE libraries:
  - HEonGPU
  - FIDESlib
  - OpenFHE-GPU fork
  - Custom CUDA implementations

CRITICAL REQUIREMENTS:
  - All backends MUST support GPU-resident ciphertexts and keys
  - Asynchronous execution with CUDA streams
  - Operation counters for rotation/keyswitch/rescale tracking

NO CPU-ONLY FALLBACK - this module REQUIRES GPU acceleration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, Tuple, Dict, Any, Callable
from contextlib import contextmanager
import time
import numpy as np

from ..compiler.ckks_params import CKKSParams, CKKSProfile


# =============================================================================
# OPERATION COUNTERS
# =============================================================================

@dataclass
class OperationCounters:
    """
    Counters for tracking HE operation costs.

    These are critical KPIs for rotation-minimal design.
    CI should fail if these exceed budget.
    """
    rotations: int = 0
    keyswitches: int = 0
    rescales: int = 0
    modswitches: int = 0
    multiplications: int = 0
    additions: int = 0
    encryptions: int = 0
    decryptions: int = 0

    # Timing
    total_time_ms: float = 0.0
    encrypt_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    decrypt_time_ms: float = 0.0

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.rotations = 0
        self.keyswitches = 0
        self.rescales = 0
        self.modswitches = 0
        self.multiplications = 0
        self.additions = 0
        self.encryptions = 0
        self.decryptions = 0
        self.total_time_ms = 0.0
        self.encrypt_time_ms = 0.0
        self.compute_time_ms = 0.0
        self.decrypt_time_ms = 0.0

    def __add__(self, other: 'OperationCounters') -> 'OperationCounters':
        """Combine counters."""
        return OperationCounters(
            rotations=self.rotations + other.rotations,
            keyswitches=self.keyswitches + other.keyswitches,
            rescales=self.rescales + other.rescales,
            modswitches=self.modswitches + other.modswitches,
            multiplications=self.multiplications + other.multiplications,
            additions=self.additions + other.additions,
            encryptions=self.encryptions + other.encryptions,
            decryptions=self.decryptions + other.decryptions,
            total_time_ms=self.total_time_ms + other.total_time_ms,
            encrypt_time_ms=self.encrypt_time_ms + other.encrypt_time_ms,
            compute_time_ms=self.compute_time_ms + other.compute_time_ms,
            decrypt_time_ms=self.decrypt_time_ms + other.decrypt_time_ms,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'rotations': self.rotations,
            'keyswitches': self.keyswitches,
            'rescales': self.rescales,
            'modswitches': self.modswitches,
            'multiplications': self.multiplications,
            'additions': self.additions,
            'encryptions': self.encryptions,
            'decryptions': self.decryptions,
            'total_time_ms': self.total_time_ms,
            'encrypt_time_ms': self.encrypt_time_ms,
            'compute_time_ms': self.compute_time_ms,
            'decrypt_time_ms': self.decrypt_time_ms,
        }


# =============================================================================
# CIPHERTEXT WRAPPER
# =============================================================================

class CiphertextLevel(Enum):
    """Ciphertext level in modulus chain."""
    FRESH = "fresh"  # Just encrypted
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    EXHAUSTED = "exhausted"


@dataclass
class GPUCiphertext:
    """
    GPU-resident ciphertext wrapper.

    This wrapper tracks metadata about the ciphertext's state,
    including its level in the modulus chain and scale.
    The actual encrypted data is held by the backend.
    """
    # Backend-specific handle (opaque to the microkernel)
    handle: Any

    # Metadata
    level: int  # Current level in modulus chain (0 = fresh)
    scale: float  # Current scale
    slot_count: int  # Number of SIMD slots
    is_ntt: bool = True  # Whether in NTT domain

    # Device info
    device_id: int = 0  # GPU device ID
    stream_id: int = 0  # CUDA stream ID

    def copy_metadata(self) -> 'GPUCiphertext':
        """Create copy with same metadata but no handle."""
        return GPUCiphertext(
            handle=None,
            level=self.level,
            scale=self.scale,
            slot_count=self.slot_count,
            is_ntt=self.is_ntt,
            device_id=self.device_id,
            stream_id=self.stream_id,
        )


@dataclass
class PlaintextPacked:
    """
    Pre-packed plaintext for efficient Ct×Pt multiplication.

    Plaintexts are encoded and packed ONCE during compilation,
    then reused for every token's inference.
    """
    # Backend-specific handle
    handle: Any

    # Metadata
    scale: float
    slot_count: int
    is_ntt: bool = True

    # Original values (for debugging/verification)
    original_shape: Tuple[int, ...] = field(default=())


# =============================================================================
# BACKEND INTERFACE
# =============================================================================

class GPUCKKSBackend(ABC):
    """
    Abstract interface for GPU-accelerated CKKS backends.

    This interface defines the contract that ALL GPU HE backends must satisfy.
    Implementations may wrap:
      - HEonGPU
      - FIDESlib
      - OpenFHE-GPU
      - Custom CUDA implementations

    The interface is designed for MOAI-style rotation minimization.
    """

    def __init__(self, params: CKKSParams, device_id: int = 0):
        """
        Initialize backend with CKKS parameters.

        Args:
            params: CKKS encryption parameters
            device_id: GPU device ID (for multi-GPU support)
        """
        self._params = params
        self._device_id = device_id
        self._counters = OperationCounters()
        self._initialized = False

    @property
    def params(self) -> CKKSParams:
        """Get CKKS parameters."""
        return self._params

    @property
    def counters(self) -> OperationCounters:
        """Get operation counters."""
        return self._counters

    def reset_counters(self) -> None:
        """Reset operation counters."""
        self._counters.reset()

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the backend with keys and context.

        This generates:
          - Secret key
          - Public key
          - Relinearization keys
          - Galois keys (for rotations)

        All keys are stored GPU-resident.
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the GPU device.

        Returns:
            Dict with keys: name, compute_capability, memory_gb, etc.
        """
        pass

    # -------------------------------------------------------------------------
    # ENCRYPTION / DECRYPTION
    # -------------------------------------------------------------------------

    @abstractmethod
    def encrypt(self, plaintext: np.ndarray) -> GPUCiphertext:
        """
        Encrypt a plaintext vector.

        Args:
            plaintext: 1D numpy array of FP64 values (length <= slot_count)

        Returns:
            GPU-resident ciphertext

        Note:
            Increments encryptions counter and encrypt_time_ms
        """
        pass

    @abstractmethod
    def decrypt(self, ciphertext: GPUCiphertext) -> np.ndarray:
        """
        Decrypt a ciphertext to plaintext.

        Args:
            ciphertext: GPU-resident ciphertext

        Returns:
            1D numpy array of FP64 values

        Note:
            Increments decryptions counter and decrypt_time_ms
        """
        pass

    @abstractmethod
    def encode_plaintext(self, values: np.ndarray) -> PlaintextPacked:
        """
        Encode values into a plaintext (without encryption).

        This is used for Ct×Pt multiplication where the plaintext
        is a pre-encoded LoRA weight matrix block.

        Args:
            values: 1D numpy array of FP64 values

        Returns:
            Encoded plaintext ready for multiplication
        """
        pass

    # -------------------------------------------------------------------------
    # ARITHMETIC OPERATIONS
    # -------------------------------------------------------------------------

    @abstractmethod
    def add(self, ct1: GPUCiphertext, ct2: GPUCiphertext) -> GPUCiphertext:
        """
        Add two ciphertexts.

        Args:
            ct1: First ciphertext
            ct2: Second ciphertext (same level required)

        Returns:
            ct1 + ct2

        Note:
            Increments additions counter
        """
        pass

    @abstractmethod
    def add_inplace(self, ct1: GPUCiphertext, ct2: GPUCiphertext) -> None:
        """
        Add ct2 to ct1 in-place.

        Args:
            ct1: Target ciphertext (modified in-place)
            ct2: Source ciphertext

        Note:
            Increments additions counter
        """
        pass

    @abstractmethod
    def mul_plain(self, ct: GPUCiphertext, pt: PlaintextPacked) -> GPUCiphertext:
        """
        Multiply ciphertext by plaintext (Ct×Pt).

        This is the core operation for LoRA:
          encrypted_x × plaintext_lora_weight

        Args:
            ct: Encrypted activations
            pt: Pre-encoded LoRA weight block

        Returns:
            ct × pt

        Note:
            Increments multiplications counter
            Does NOT trigger keyswitch (no ciphertext-ciphertext multiply)
        """
        pass

    @abstractmethod
    def mul_plain_inplace(self, ct: GPUCiphertext, pt: PlaintextPacked) -> None:
        """
        Multiply ciphertext by plaintext in-place.

        Args:
            ct: Encrypted activations (modified in-place)
            pt: Pre-encoded LoRA weight block

        Note:
            Increments multiplications counter
        """
        pass

    # -------------------------------------------------------------------------
    # ROTATION OPERATIONS (CRITICAL FOR MOAI)
    # -------------------------------------------------------------------------

    @abstractmethod
    def rotate(self, ct: GPUCiphertext, steps: int) -> GPUCiphertext:
        """
        Rotate ciphertext slots by given steps.

        This is the MOST EXPENSIVE operation in HE.
        MOAI-style packing is designed to MINIMIZE rotations.

        Args:
            ct: Ciphertext to rotate
            steps: Number of steps (positive = left, negative = right)

        Returns:
            Rotated ciphertext

        Note:
            Increments rotations counter AND keyswitches counter
            (rotation requires key switching)
        """
        pass

    @abstractmethod
    def rotate_inplace(self, ct: GPUCiphertext, steps: int) -> None:
        """
        Rotate ciphertext in-place.

        Args:
            ct: Ciphertext to rotate (modified in-place)
            steps: Number of steps

        Note:
            Increments rotations and keyswitches counters
        """
        pass

    # -------------------------------------------------------------------------
    # LEVEL MANAGEMENT
    # -------------------------------------------------------------------------

    @abstractmethod
    def rescale(self, ct: GPUCiphertext) -> GPUCiphertext:
        """
        Rescale ciphertext after multiplication.

        Reduces scale and consumes one level of modulus chain.

        Args:
            ct: Ciphertext to rescale

        Returns:
            Rescaled ciphertext (level + 1)

        Note:
            Increments rescales counter
        """
        pass

    @abstractmethod
    def rescale_inplace(self, ct: GPUCiphertext) -> None:
        """
        Rescale ciphertext in-place.

        Args:
            ct: Ciphertext to rescale (modified in-place)

        Note:
            Increments rescales counter
        """
        pass

    @abstractmethod
    def modswitch(self, ct: GPUCiphertext) -> GPUCiphertext:
        """
        Modulus switch without rescaling.

        Used to align levels for addition.

        Args:
            ct: Ciphertext to modswitch

        Returns:
            Modswitched ciphertext

        Note:
            Increments modswitches counter
        """
        pass

    @abstractmethod
    def modswitch_to_level(self, ct: GPUCiphertext, target_level: int) -> GPUCiphertext:
        """
        Modswitch to a specific level.

        Args:
            ct: Ciphertext to modswitch
            target_level: Target level in modulus chain

        Returns:
            Ciphertext at target level

        Note:
            Increments modswitches counter (possibly multiple times)
        """
        pass

    # -------------------------------------------------------------------------
    # FUSED OPERATIONS (KERNEL FUSION)
    # -------------------------------------------------------------------------

    def mul_plain_rescale(
        self,
        ct: GPUCiphertext,
        pt: PlaintextPacked
    ) -> GPUCiphertext:
        """
        Fused multiply-rescale operation.

        Default implementation calls mul_plain then rescale.
        Backends may override with fused GPU kernel.

        Args:
            ct: Encrypted activations
            pt: Pre-encoded weight

        Returns:
            Rescaled product
        """
        result = self.mul_plain(ct, pt)
        self.rescale_inplace(result)
        return result

    def mul_plain_rescale_add(
        self,
        ct: GPUCiphertext,
        pt: PlaintextPacked,
        accumulator: GPUCiphertext
    ) -> GPUCiphertext:
        """
        Fused multiply-rescale-add operation.

        Default implementation calls individual ops.
        Backends may override with fused GPU kernel.

        Args:
            ct: Encrypted activations
            pt: Pre-encoded weight
            accumulator: Running sum to add to

        Returns:
            accumulator + rescale(ct × pt)
        """
        product = self.mul_plain(ct, pt)
        self.rescale_inplace(product)

        # Align levels if needed
        if product.level != accumulator.level:
            product = self.modswitch_to_level(product, accumulator.level)

        self.add_inplace(accumulator, product)
        return accumulator

    # -------------------------------------------------------------------------
    # STREAM MANAGEMENT (ASYNC EXECUTION)
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_stream(self) -> int:
        """
        Create a new CUDA stream.

        Returns:
            Stream ID for async operations
        """
        pass

    @abstractmethod
    def synchronize_stream(self, stream_id: int) -> None:
        """
        Synchronize a CUDA stream.

        Args:
            stream_id: Stream to synchronize
        """
        pass

    @abstractmethod
    def synchronize_all(self) -> None:
        """Synchronize all CUDA streams."""
        pass

    # -------------------------------------------------------------------------
    # MEMORY MANAGEMENT
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current GPU memory usage.

        Returns:
            Dict with keys: used_bytes, total_bytes, ciphertext_count
        """
        pass

    @abstractmethod
    def free_ciphertext(self, ct: GPUCiphertext) -> None:
        """
        Free a ciphertext from GPU memory.

        Args:
            ct: Ciphertext to free
        """
        pass

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    @contextmanager
    def timed_section(self, section_name: str):
        """
        Context manager for timing a section.

        Args:
            section_name: Name of section ('encrypt', 'compute', 'decrypt')
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if section_name == 'encrypt':
                self._counters.encrypt_time_ms += elapsed_ms
            elif section_name == 'decrypt':
                self._counters.decrypt_time_ms += elapsed_ms
            elif section_name == 'compute':
                self._counters.compute_time_ms += elapsed_ms
            self._counters.total_time_ms += elapsed_ms


# =============================================================================
# BACKEND REGISTRY
# =============================================================================

class BackendType(Enum):
    """Available GPU CKKS backend types."""
    HEONGPU = "heongpu"
    FIDESLIB = "fideslib"
    OPENFHE_GPU = "openfhe_gpu"
    SIMULATION = "simulation"  # For testing without GPU


_BACKEND_REGISTRY: Dict[BackendType, type] = {}


def register_backend(backend_type: BackendType):
    """Decorator to register a backend implementation."""
    def decorator(cls):
        _BACKEND_REGISTRY[backend_type] = cls
        return cls
    return decorator


def get_available_backends() -> List[BackendType]:
    """Get list of available (registered) backends."""
    return list(_BACKEND_REGISTRY.keys())


def create_backend(
    backend_type: BackendType,
    params: CKKSParams,
    device_id: int = 0,
) -> GPUCKKSBackend:
    """
    Create a GPU CKKS backend instance.

    Args:
        backend_type: Which backend to use
        params: CKKS encryption parameters
        device_id: GPU device ID

    Returns:
        Initialized backend instance

    Raises:
        ValueError: If backend type not registered
    """
    if backend_type not in _BACKEND_REGISTRY:
        available = [b.value for b in _BACKEND_REGISTRY.keys()]
        raise ValueError(
            f"Backend '{backend_type.value}' not registered. "
            f"Available: {available}"
        )

    backend_cls = _BACKEND_REGISTRY[backend_type]
    backend = backend_cls(params, device_id)
    backend.initialize()
    return backend


# =============================================================================
# SIMULATION BACKEND (FOR TESTING)
# =============================================================================

@register_backend(BackendType.SIMULATION)
class SimulationBackend(GPUCKKSBackend):
    """
    Simulation backend for testing without real GPU HE.

    This backend simulates CKKS operations using plaintext arithmetic,
    but tracks all operation counts accurately. It's useful for:
      - Unit testing the microkernel
      - Verifying rotation counts
      - Testing on machines without GPU HE libraries

    WARNING: This provides NO SECURITY. Use only for testing.
    """

    def __init__(self, params: CKKSParams, device_id: int = 0):
        super().__init__(params, device_id)
        self._keys_generated = False
        self._ciphertext_counter = 0
        self._plaintexts: Dict[int, np.ndarray] = {}

    def initialize(self) -> None:
        """Initialize (simulated) keys."""
        self._keys_generated = True
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def get_device_info(self) -> Dict[str, Any]:
        return {
            'name': 'Simulation (CPU)',
            'backend': 'simulation',
            'compute_capability': 'N/A',
            'memory_gb': 0,
            'is_real_gpu': False,
        }

    def _new_ct_id(self) -> int:
        self._ciphertext_counter += 1
        return self._ciphertext_counter

    def encrypt(self, plaintext: np.ndarray) -> GPUCiphertext:
        self._counters.encryptions += 1
        ct_id = self._new_ct_id()

        # Store plaintext for simulation
        padded = np.zeros(self._params.slot_count)
        padded[:len(plaintext)] = plaintext
        self._plaintexts[ct_id] = padded

        return GPUCiphertext(
            handle=ct_id,
            level=0,
            scale=self._params.scale,
            slot_count=self._params.slot_count,
            is_ntt=True,
            device_id=self._device_id,
        )

    def decrypt(self, ciphertext: GPUCiphertext) -> np.ndarray:
        self._counters.decryptions += 1
        ct_id = ciphertext.handle
        return self._plaintexts[ct_id].copy()

    def encode_plaintext(self, values: np.ndarray) -> PlaintextPacked:
        padded = np.zeros(self._params.slot_count)
        padded[:len(values)] = values
        return PlaintextPacked(
            handle=padded.copy(),
            scale=self._params.scale,
            slot_count=self._params.slot_count,
            is_ntt=True,
            original_shape=values.shape,
        )

    def add(self, ct1: GPUCiphertext, ct2: GPUCiphertext) -> GPUCiphertext:
        self._counters.additions += 1
        ct_id = self._new_ct_id()
        self._plaintexts[ct_id] = (
            self._plaintexts[ct1.handle] + self._plaintexts[ct2.handle]
        )
        return GPUCiphertext(
            handle=ct_id,
            level=max(ct1.level, ct2.level),
            scale=ct1.scale,
            slot_count=self._params.slot_count,
        )

    def add_inplace(self, ct1: GPUCiphertext, ct2: GPUCiphertext) -> None:
        self._counters.additions += 1
        self._plaintexts[ct1.handle] += self._plaintexts[ct2.handle]

    def mul_plain(self, ct: GPUCiphertext, pt: PlaintextPacked) -> GPUCiphertext:
        self._counters.multiplications += 1
        ct_id = self._new_ct_id()
        self._plaintexts[ct_id] = self._plaintexts[ct.handle] * pt.handle
        return GPUCiphertext(
            handle=ct_id,
            level=ct.level,
            scale=ct.scale * pt.scale,
            slot_count=self._params.slot_count,
        )

    def mul_plain_inplace(self, ct: GPUCiphertext, pt: PlaintextPacked) -> None:
        self._counters.multiplications += 1
        self._plaintexts[ct.handle] *= pt.handle

    def rotate(self, ct: GPUCiphertext, steps: int) -> GPUCiphertext:
        self._counters.rotations += 1
        self._counters.keyswitches += 1  # Rotation requires keyswitch
        ct_id = self._new_ct_id()
        self._plaintexts[ct_id] = np.roll(self._plaintexts[ct.handle], -steps)
        return GPUCiphertext(
            handle=ct_id,
            level=ct.level,
            scale=ct.scale,
            slot_count=self._params.slot_count,
        )

    def rotate_inplace(self, ct: GPUCiphertext, steps: int) -> None:
        self._counters.rotations += 1
        self._counters.keyswitches += 1
        self._plaintexts[ct.handle] = np.roll(self._plaintexts[ct.handle], -steps)

    def rescale(self, ct: GPUCiphertext) -> GPUCiphertext:
        self._counters.rescales += 1
        ct_id = self._new_ct_id()
        self._plaintexts[ct_id] = self._plaintexts[ct.handle].copy()
        return GPUCiphertext(
            handle=ct_id,
            level=ct.level + 1,
            scale=self._params.scale,  # Reset scale after rescale
            slot_count=self._params.slot_count,
        )

    def rescale_inplace(self, ct: GPUCiphertext) -> None:
        self._counters.rescales += 1
        ct.level += 1
        ct.scale = self._params.scale

    def modswitch(self, ct: GPUCiphertext) -> GPUCiphertext:
        self._counters.modswitches += 1
        ct_id = self._new_ct_id()
        self._plaintexts[ct_id] = self._plaintexts[ct.handle].copy()
        return GPUCiphertext(
            handle=ct_id,
            level=ct.level + 1,
            scale=ct.scale,
            slot_count=self._params.slot_count,
        )

    def modswitch_to_level(self, ct: GPUCiphertext, target_level: int) -> GPUCiphertext:
        result = ct
        while result.level < target_level:
            result = self.modswitch(result)
        return result

    def create_stream(self) -> int:
        return 0  # Simulation doesn't use real streams

    def synchronize_stream(self, stream_id: int) -> None:
        pass

    def synchronize_all(self) -> None:
        pass

    def get_memory_usage(self) -> Dict[str, int]:
        ct_bytes = len(self._plaintexts) * self._params.slot_count * 8
        return {
            'used_bytes': ct_bytes,
            'total_bytes': 0,
            'ciphertext_count': len(self._plaintexts),
        }

    def free_ciphertext(self, ct: GPUCiphertext) -> None:
        if ct.handle in self._plaintexts:
            del self._plaintexts[ct.handle]
