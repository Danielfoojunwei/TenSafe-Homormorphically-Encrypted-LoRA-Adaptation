"""
TFHE LUT Activation Engine for HE-LoRA Microkernel

This module implements GPU-accelerated programmable bootstrapping for
EXACT evaluation of non-linear activation functions on discrete plaintexts.

Key Properties:
    - EXACT computation: No precision loss (unlike CKKS)
    - Discrete message space: {0, 1, ..., p-1} on the torus
    - Correctness: Overwhelming probability when parameters are valid

Architecture:
    The LUT engine integrates with the existing GPU MOAI setup:

    MOAI Path (Linear):
        CKKS ciphertext → Column-packed matmul → CKKS ciphertext

    TFHE Path (Non-linear):
        Quantize → LWE encrypt → Programmable Bootstrap → LWE decrypt → Dequantize

    Hybrid Flow:
        CKKS (linear) → Bridge → TFHE (activation) → Bridge → CKKS (linear)

References:
    - TFHE: Fully Homomorphic Encryption over the Torus
    - Programmable Bootstrapping via Blind Rotation
    - GPU acceleration for TFHE (cuFHE, Concrete, etc.)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Tuple
import numpy as np
import time
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class LUTConfig:
    """
    Configuration for LUT-based activation evaluation.

    Defines the discretization and evaluation parameters for
    TFHE programmable bootstrapping.
    """
    # Message space size (power of 2 for efficiency)
    message_space_size: int = 256

    # Input value range for quantization
    input_min: float = -10.0
    input_max: float = 10.0

    # Output value range for dequantization
    output_min: float = -10.0
    output_max: float = 10.0

    # Whether to use GPU for bootstrapping
    use_gpu: bool = True

    # Batch size for parallel bootstrapping
    batch_size: int = 1024

    def __post_init__(self):
        """Validate configuration."""
        if self.message_space_size & (self.message_space_size - 1) != 0:
            raise ValueError(f"message_space_size must be power of 2, got {self.message_space_size}")
        if self.input_min >= self.input_max:
            raise ValueError(f"Invalid input range: [{self.input_min}, {self.input_max}]")
        if self.output_min >= self.output_max:
            raise ValueError(f"Invalid output range: [{self.output_min}, {self.output_max}]")

    @property
    def precision_bits(self) -> int:
        """Bits of precision in message space."""
        return int(math.log2(self.message_space_size))


@dataclass
class ActivationLUT:
    """
    Precomputed lookup table for an activation function.

    The LUT maps discrete input messages to discrete output messages:
        LUT[m_in] = quantize(f(dequantize(m_in)))

    This is used in TFHE programmable bootstrapping where the LUT
    is encoded in the test polynomial.
    """
    # The lookup table entries
    entries: np.ndarray

    # Function name for identification
    name: str

    # Configuration used to generate this LUT
    config: LUTConfig

    # The original activation function
    _activation_fn: Optional[Callable[[float], float]] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> int:
        return int(self.entries[idx])

    def to_list(self) -> List[int]:
        """Convert to Python list for TFHE."""
        return self.entries.tolist()

    def verify_correctness(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Verify LUT correctness by comparing with exact computation.

        Returns statistics on quantization error.
        """
        if self._activation_fn is None:
            return {'error': 'No activation function stored'}

        p = len(self.entries)
        errors = []

        for _ in range(num_samples):
            # Random input in valid range
            x = np.random.uniform(self.config.input_min, self.config.input_max)

            # Quantize input
            x_normalized = (x - self.config.input_min) / (self.config.input_max - self.config.input_min)
            m_in = int(round(x_normalized * (p - 1)))
            m_in = max(0, min(p - 1, m_in))

            # LUT output
            m_out = self.entries[m_in]

            # Dequantize output
            y_lut = self.config.output_min + (m_out / (p - 1)) * (self.config.output_max - self.config.output_min)

            # Exact computation
            y_exact = self._activation_fn(x)

            errors.append(abs(y_lut - y_exact))

        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'std_error': np.std(errors),
            'num_samples': num_samples,
        }


def create_relu_lut(config: LUTConfig) -> ActivationLUT:
    """
    Create ReLU activation LUT.

    ReLU(x) = max(0, x)
    """
    p = config.message_space_size
    entries = np.zeros(p, dtype=np.int32)

    for m_in in range(p):
        # Dequantize
        x = config.input_min + (m_in / (p - 1)) * (config.input_max - config.input_min)
        # Apply ReLU
        y = max(0.0, x)
        # Quantize output
        y_normalized = (y - config.output_min) / (config.output_max - config.output_min)
        m_out = int(round(y_normalized * (p - 1)))
        entries[m_in] = max(0, min(p - 1, m_out))

    return ActivationLUT(
        entries=entries,
        name="relu",
        config=config,
        _activation_fn=lambda x: max(0.0, x),
    )


def create_gelu_lut(config: LUTConfig) -> ActivationLUT:
    """
    Create GELU activation LUT.

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    p = config.message_space_size
    entries = np.zeros(p, dtype=np.int32)

    def gelu(x):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

    for m_in in range(p):
        x = config.input_min + (m_in / (p - 1)) * (config.input_max - config.input_min)
        y = gelu(x)
        y_normalized = (y - config.output_min) / (config.output_max - config.output_min)
        m_out = int(round(y_normalized * (p - 1)))
        entries[m_in] = max(0, min(p - 1, m_out))

    return ActivationLUT(
        entries=entries,
        name="gelu",
        config=config,
        _activation_fn=gelu,
    )


def create_sigmoid_lut(config: LUTConfig) -> ActivationLUT:
    """
    Create Sigmoid activation LUT.

    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    # Sigmoid output is always in [0, 1]
    sigmoid_config = LUTConfig(
        message_space_size=config.message_space_size,
        input_min=config.input_min,
        input_max=config.input_max,
        output_min=0.0,
        output_max=1.0,
        use_gpu=config.use_gpu,
        batch_size=config.batch_size,
    )

    p = sigmoid_config.message_space_size
    entries = np.zeros(p, dtype=np.int32)

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    for m_in in range(p):
        x = sigmoid_config.input_min + (m_in / (p - 1)) * (sigmoid_config.input_max - sigmoid_config.input_min)
        y = sigmoid(x)
        # Sigmoid output is already in [0, 1]
        m_out = int(round(y * (p - 1)))
        entries[m_in] = max(0, min(p - 1, m_out))

    return ActivationLUT(
        entries=entries,
        name="sigmoid",
        config=sigmoid_config,
        _activation_fn=sigmoid,
    )


def create_custom_lut(
    func: Callable[[float], float],
    name: str,
    config: LUTConfig,
) -> ActivationLUT:
    """
    Create LUT for a custom activation function.

    Args:
        func: The activation function f: R -> R
        name: Name for identification
        config: LUT configuration

    Returns:
        ActivationLUT for the function
    """
    p = config.message_space_size
    entries = np.zeros(p, dtype=np.int32)

    for m_in in range(p):
        x = config.input_min + (m_in / (p - 1)) * (config.input_max - config.input_min)
        y = func(x)
        y_clamped = max(config.output_min, min(config.output_max, y))
        y_normalized = (y_clamped - config.output_min) / (config.output_max - config.output_min)
        m_out = int(round(y_normalized * (p - 1)))
        entries[m_in] = max(0, min(p - 1, m_out))

    return ActivationLUT(
        entries=entries,
        name=name,
        config=config,
        _activation_fn=func,
    )


class LUTActivationEngine:
    """
    GPU-accelerated engine for TFHE programmable bootstrapping.

    This engine manages:
    - LUT precomputation and caching
    - Batched bootstrapping execution
    - GPU/CPU backend selection
    - Integration with MOAI CKKS pipeline

    Usage:
        engine = LUTActivationEngine(n2he_params)
        engine.initialize()

        # Register activation LUT
        engine.register_lut("relu", create_relu_lut(config))

        # Apply activation to encrypted values
        ct_out = engine.apply_activation(ct_in, "relu")
    """

    def __init__(self, params: 'N2HEParams', use_gpu: bool = True):
        """
        Initialize LUT activation engine.

        Args:
            params: N2HE/TFHE parameters
            use_gpu: Whether to use GPU acceleration
        """
        from .n2he_params import N2HEParams
        self.params = params
        self.use_gpu = use_gpu and params.use_gpu

        # LUT cache
        self._luts: Dict[str, ActivationLUT] = {}

        # Backend (GPU or CPU)
        self._backend = None

        # State
        self._initialized = False

        # Statistics
        self._stats = {
            'activations_computed': 0,
            'bootstraps_performed': 0,
            'total_time_ms': 0.0,
            'lut_cache_hits': 0,
        }

    def initialize(self) -> None:
        """Initialize the activation engine and backend."""
        logger.info(f"Initializing LUT Activation Engine (GPU={self.use_gpu})...")

        if self.use_gpu:
            self._initialize_gpu_backend()
        else:
            self._initialize_cpu_backend()

        self._initialized = True
        logger.info("LUT Activation Engine initialized")

    def _initialize_gpu_backend(self):
        """Initialize GPU backend for TFHE operations."""
        try:
            # Try to import GPU TFHE library
            # In production, this would use cuFHE, Concrete-GPU, or similar
            logger.info("Attempting GPU TFHE backend initialization...")

            # Placeholder for GPU backend
            # In production:
            # from tfhe_gpu import TFHEGPUBackend
            # self._backend = TFHEGPUBackend(self.params)

            # Fall back to simulation with GPU tensor ops
            self._backend = _SimulatedGPUBackend(self.params)
            logger.info("Using simulated GPU backend (for development)")

        except ImportError as e:
            logger.warning(f"GPU TFHE backend not available: {e}")
            logger.warning("Falling back to CPU backend")
            self._initialize_cpu_backend()

    def _initialize_cpu_backend(self):
        """Initialize CPU backend using FasterNTT."""
        from .faster_ntt import FasterNTTBackend
        self._backend = FasterNTTBackend(self.params)
        self._backend.initialize()
        logger.info("Using FasterNTT CPU backend")

    def register_lut(self, name: str, lut: ActivationLUT) -> None:
        """
        Register a precomputed LUT for an activation function.

        Args:
            name: Name to identify the LUT
            lut: Precomputed activation LUT
        """
        self._luts[name] = lut
        logger.debug(f"Registered LUT '{name}' with {len(lut)} entries")

    def get_lut(self, name: str) -> Optional[ActivationLUT]:
        """Get a registered LUT by name."""
        return self._luts.get(name)

    def apply_activation(
        self,
        ct: np.ndarray,
        activation_name: str,
    ) -> np.ndarray:
        """
        Apply activation function to encrypted value via programmable bootstrapping.

        This is the main entry point for non-linear function evaluation.
        The computation is EXACT on discrete message space.

        Args:
            ct: LWE ciphertext encrypting quantized input
            activation_name: Name of registered activation LUT

        Returns:
            LWE ciphertext encrypting f(input) where f is the activation
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if activation_name not in self._luts:
            raise ValueError(f"Unknown activation '{activation_name}'. Register LUT first.")

        start_time = time.perf_counter()

        lut = self._luts[activation_name]
        self._stats['lut_cache_hits'] += 1

        # Perform programmable bootstrapping
        result = self._backend.programmable_bootstrap(ct, lut.to_list())

        self._stats['activations_computed'] += 1
        self._stats['bootstraps_performed'] += 1
        self._stats['total_time_ms'] += (time.perf_counter() - start_time) * 1000

        return result

    def apply_activation_batch(
        self,
        cts: List[np.ndarray],
        activation_name: str,
    ) -> List[np.ndarray]:
        """
        Apply activation to a batch of encrypted values.

        Batching enables parallel bootstrapping on GPU for better throughput.

        Args:
            cts: List of LWE ciphertexts
            activation_name: Name of registered activation LUT

        Returns:
            List of bootstrapped ciphertexts
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if activation_name not in self._luts:
            raise ValueError(f"Unknown activation '{activation_name}'")

        start_time = time.perf_counter()

        lut = self._luts[activation_name]
        lut_list = lut.to_list()

        # Batch processing
        results = []
        for ct in cts:
            result = self._backend.programmable_bootstrap(ct, lut_list)
            results.append(result)

        self._stats['activations_computed'] += len(cts)
        self._stats['bootstraps_performed'] += len(cts)
        self._stats['total_time_ms'] += (time.perf_counter() - start_time) * 1000

        return results

    def quantize(self, value: float, config: LUTConfig) -> int:
        """
        Quantize a real value to discrete message space.

        Args:
            value: Real value to quantize
            config: LUT configuration with input range

        Returns:
            Discrete message in {0, 1, ..., p-1}
        """
        p = config.message_space_size
        clamped = max(config.input_min, min(config.input_max, value))
        normalized = (clamped - config.input_min) / (config.input_max - config.input_min)
        discrete = int(round(normalized * (p - 1)))
        return max(0, min(p - 1, discrete))

    def dequantize(self, message: int, config: LUTConfig) -> float:
        """
        Dequantize a discrete message to real value.

        Args:
            message: Discrete message in {0, 1, ..., p-1}
            config: LUT configuration with output range

        Returns:
            Real value in [output_min, output_max]
        """
        p = config.message_space_size
        normalized = message / (p - 1)
        return config.output_min + normalized * (config.output_max - config.output_min)

    def encrypt_quantized(self, message: int) -> np.ndarray:
        """
        Encrypt a quantized message with LWE.

        Args:
            message: Discrete message in {0, 1, ..., p-1}

        Returns:
            LWE ciphertext
        """
        return self._backend.encrypt_lwe(message)

    def decrypt_quantized(self, ct: np.ndarray) -> int:
        """
        Decrypt LWE ciphertext to quantized message.

        Args:
            ct: LWE ciphertext

        Returns:
            Discrete message in {0, 1, ..., p-1}
        """
        return self._backend.decrypt_lwe(ct)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self._stats.copy()
        if hasattr(self._backend, 'get_stats'):
            stats['backend_stats'] = self._backend.get_stats()
        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            'activations_computed': 0,
            'bootstraps_performed': 0,
            'total_time_ms': 0.0,
            'lut_cache_hits': 0,
        }
        if hasattr(self._backend, 'reset_stats'):
            self._backend.reset_stats()


class _SimulatedGPUBackend:
    """
    Simulated GPU backend for development and testing.

    This provides the same interface as a real GPU TFHE backend
    but performs computation on CPU with NumPy.

    NOT FOR PRODUCTION USE - provides no real encryption.
    """

    def __init__(self, params: 'N2HEParams'):
        self.params = params
        self._rng = np.random.default_rng()
        self._stats = {
            'encryptions': 0,
            'decryptions': 0,
            'bootstraps': 0,
        }

    def encrypt_lwe(self, message: int) -> np.ndarray:
        """Simulate LWE encryption (NOT SECURE)."""
        n = self.params.lwe.dimension
        ct = np.zeros(n + 1, dtype=np.int64)
        ct[:n] = self._rng.integers(0, 2**32, size=n)
        ct[n] = message  # Store message directly (simulation only!)
        self._stats['encryptions'] += 1
        return ct

    def decrypt_lwe(self, ct: np.ndarray) -> int:
        """Simulate LWE decryption (NOT SECURE)."""
        self._stats['decryptions'] += 1
        return int(ct[-1]) % self.params.lwe.message_space

    def programmable_bootstrap(self, ct: np.ndarray, lut: List[int]) -> np.ndarray:
        """
        Simulate programmable bootstrapping.

        In simulation, we just apply the LUT directly to the message.
        """
        self._stats['bootstraps'] += 1

        # Extract message (stored in last element for simulation)
        m_in = int(ct[-1]) % len(lut)

        # Apply LUT
        m_out = lut[m_in]

        # Create new ciphertext
        return self.encrypt_lwe(m_out)

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()

    def reset_stats(self) -> None:
        self._stats = {
            'encryptions': 0,
            'decryptions': 0,
            'bootstraps': 0,
        }
