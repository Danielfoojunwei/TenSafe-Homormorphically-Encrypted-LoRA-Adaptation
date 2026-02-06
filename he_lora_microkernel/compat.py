"""
HE-LoRA Microkernel Compatibility Layer

Provides backward-compatible interfaces for integration with the existing
TenSafe unified pipeline. This allows the new microkernel to be a drop-in
replacement for the old tensafe.he_lora implementation.

Usage:
    from he_lora_microkernel.compat import HELoRAAdapter, HELoRAConfig

    # Create adapter
    config = HELoRAConfig(rank=16, alpha=32.0)
    adapter = HELoRAAdapter(config)

    # Register LoRA weights and run forward pass
    adapter.register_weights("q_proj", lora_a, lora_b)
    delta = adapter.forward(x_plain, "q_proj")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


@dataclass
class HELoRAConfig:
    """
    Configuration for HE-LoRA adapter.

    Backward-compatible with old tensafe.he_lora.HELoRAConfig.
    """
    # LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # CKKS parameters
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40

    # MOAI optimizations
    use_column_packing: bool = True
    use_interleaved_batching: bool = True

    # Backend selection
    backend_type: str = "SIMULATION"  # SIMULATION, HEONGPU, FIDESLIB, OPENFHE_GPU

    # Performance
    max_batch_size: int = 32

    @property
    def scaling(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank


@dataclass
class HELoRAMetrics:
    """Metrics from HE-LoRA operations."""
    operations_count: int = 0
    rotations_count: int = 0
    keyswitches_count: int = 0
    rescales_count: int = 0
    encrypt_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    decrypt_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": self.operations_count,
            "rotations": self.rotations_count,
            "keyswitches": self.keyswitches_count,
            "rescales": self.rescales_count,
            "encrypt_time_ms": self.encrypt_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "decrypt_time_ms": self.decrypt_time_ms,
            "total_time_ms": self.total_time_ms,
        }


class HELoRAAdapter:
    """
    HE-LoRA Adapter using the new microkernel.

    Backward-compatible interface with the old tensafe.he_lora.HELoRAAdapter.
    Provides encrypted LoRA delta computation using the new MOAI-inspired
    microkernel architecture.

    Usage:
        config = HELoRAConfig(rank=16, alpha=32.0)
        adapter = HELoRAAdapter(config)

        # Register weights for modules
        adapter.register_weights("q_proj", lora_a, lora_b)
        adapter.register_weights("v_proj", lora_a_v, lora_b_v)

        # Compute encrypted delta
        delta = adapter.forward(x_plain, "q_proj")
    """

    def __init__(self, config: Optional[HELoRAConfig] = None):
        """
        Initialize HE-LoRA adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config or HELoRAConfig()

        # Initialize microkernel components
        self._executor = None
        self._backend = None
        self._initialized = False

        # Weight storage
        self._weights: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._packed_weights: Dict[str, Any] = {}

        # Metrics
        self._metrics = HELoRAMetrics()

        # Initialize backend
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the microkernel backend."""
        try:
            from .backend.gpu_ckks_backend import BackendType, get_backend
            from .compiler.ckks_params import CKKSProfile, get_profile

            # Get CKKS parameters
            profile = CKKSProfile.FAST if self.config.backend_type == "SIMULATION" else CKKSProfile.SAFE
            ckks_params = get_profile(profile)

            # Override with config values
            ckks_params.poly_modulus_degree = self.config.poly_modulus_degree
            ckks_params.scale_bits = self.config.scale_bits

            # Get backend
            backend_type = BackendType[self.config.backend_type]
            self._backend = get_backend(backend_type, ckks_params)
            self._backend.initialize()

            self._initialized = True
            logger.info(f"HELoRAAdapter initialized with {self.config.backend_type} backend")

        except ImportError as e:
            logger.warning(f"Microkernel backend not available: {e}")
            logger.warning("Using simulation mode")
            self._initialized = True

    def register_weights(
        self,
        module_name: str,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Register LoRA weights for a module.

        Args:
            module_name: Name of the target module
            lora_a: LoRA A matrix [rank, in_features]
            lora_b: LoRA B matrix [out_features, rank]
            rank: Optional override for LoRA rank
            alpha: Optional override for LoRA alpha
        """
        # Store weights
        self._weights[module_name] = (
            lora_a.astype(np.float64),
            lora_b.astype(np.float64),
        )

        # Update config if overrides provided
        if rank is not None:
            self.config.rank = rank
        if alpha is not None:
            self.config.alpha = alpha

        # Pre-pack weights if backend supports it
        if self._backend is not None and hasattr(self._backend, 'create_column_packed_matrix'):
            self._packed_weights[f"{module_name}_a"] = self._backend.create_column_packed_matrix(lora_a)
            self._packed_weights[f"{module_name}_b"] = self._backend.create_column_packed_matrix(lora_b)

        logger.debug(f"Registered weights for {module_name}: A={lora_a.shape}, B={lora_b.shape}")

    def forward(
        self,
        x_plain: np.ndarray,
        module_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute encrypted LoRA delta.

        Args:
            x_plain: Plaintext activation [batch, hidden_dim] or [hidden_dim]
            module_name: Target module name (uses first registered if not specified)

        Returns:
            Decrypted LoRA delta
        """
        start_time = time.perf_counter()

        # Determine module
        if module_name is None:
            if not self._weights:
                raise ValueError("No weights registered")
            module_name = next(iter(self._weights))

        if module_name not in self._weights:
            raise ValueError(f"Module '{module_name}' not registered")

        lora_a, lora_b = self._weights[module_name]
        scaling = self.config.scaling

        # Flatten input
        original_shape = x_plain.shape
        x_flat = x_plain.flatten().astype(np.float64)

        # Use backend if available
        if self._backend is not None and hasattr(self._backend, 'encrypt'):
            delta = self._forward_encrypted(x_flat, lora_a, lora_b, scaling, module_name)
        else:
            # Fallback to simulation
            delta = self._forward_simulation(x_flat, lora_a, lora_b, scaling)

        # Reshape output
        delta = delta.reshape(original_shape)

        # Update metrics
        self._metrics.operations_count += 1
        self._metrics.total_time_ms += (time.perf_counter() - start_time) * 1000

        return delta

    def _forward_encrypted(
        self,
        x_flat: np.ndarray,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float,
        module_name: str,
    ) -> np.ndarray:
        """Forward pass with real HE encryption."""
        # Encrypt
        t0 = time.perf_counter()
        ct_x = self._backend.encrypt(x_flat)
        self._metrics.encrypt_time_ms += (time.perf_counter() - t0) * 1000

        # Compute: ct_x @ A^T @ B^T
        t1 = time.perf_counter()

        # Try to use packed weights
        packed_a = self._packed_weights.get(f"{module_name}_a")
        packed_b = self._packed_weights.get(f"{module_name}_b")

        if packed_a is not None and packed_b is not None:
            # Use column-packed multiplication (rotation-free)
            intermediate = self._backend.column_packed_matmul(ct_x, packed_a)
            ct_result = self._backend.column_packed_matmul(intermediate, packed_b)
        else:
            # Standard matmul
            intermediate = self._backend.ct_pt_multiply(ct_x, lora_a.T)
            ct_result = self._backend.ct_pt_multiply(intermediate, lora_b.T)

        # Apply scaling
        ct_result = self._backend.multiply_plain(ct_result, np.array([scaling]))

        self._metrics.compute_time_ms += (time.perf_counter() - t1) * 1000

        # Partial decrypt: only the slots carrying real data
        t2 = time.perf_counter()
        output_size = len(x_flat)
        if hasattr(self._backend, 'decrypt_partial'):
            delta = self._backend.decrypt_partial(ct_result, output_size)
        else:
            delta = self._backend.decrypt(ct_result, output_size=output_size)
        self._metrics.decrypt_time_ms += (time.perf_counter() - t2) * 1000

        # Get operation counts from backend
        if hasattr(self._backend, 'get_counters'):
            counters = self._backend.get_counters()
            self._metrics.rotations_count = counters.get('rotations', 0)
            self._metrics.keyswitches_count = counters.get('keyswitches', 0)
            self._metrics.rescales_count = counters.get('rescales', 0)

        return delta

    def _forward_simulation(
        self,
        x_flat: np.ndarray,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float,
    ) -> np.ndarray:
        """Simulated forward pass (no real encryption)."""
        # Simulate timing
        self._metrics.encrypt_time_ms += 0.1
        self._metrics.compute_time_ms += 0.5
        self._metrics.decrypt_time_ms += 0.1

        # Plain computation: delta = scaling * (x @ A^T @ B^T)
        intermediate = x_flat @ lora_a.T
        delta = intermediate @ lora_b.T
        return scaling * delta

    def get_metrics(self) -> HELoRAMetrics:
        """Get accumulated metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._metrics = HELoRAMetrics()

    def get_registered_modules(self) -> List[str]:
        """Get list of registered module names."""
        return list(self._weights.keys())

    def is_initialized(self) -> bool:
        """Check if adapter is initialized."""
        return self._initialized


class HEBackend:
    """
    Backward-compatible HE backend interface.

    Wraps the new microkernel backend to provide the interface
    expected by the old code.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize HE backend.

        Args:
            params: CKKS parameters
        """
        self._params = params or {}
        self._backend = None
        self._is_setup = False

    def setup(self) -> None:
        """Set up the HE context and generate keys."""
        try:
            from .backend.gpu_ckks_backend import BackendType, get_backend
            from .compiler.ckks_params import CKKSProfile, get_profile

            # Use FAST profile for setup
            ckks_params = get_profile(CKKSProfile.FAST)

            # Apply any custom params
            if 'poly_modulus_degree' in self._params:
                ckks_params.poly_modulus_degree = self._params['poly_modulus_degree']
            if 'scale_bits' in self._params:
                ckks_params.scale_bits = self._params['scale_bits']

            # Get simulation backend
            self._backend = get_backend(BackendType.SIMULATION, ckks_params)
            self._backend.initialize()

            self._is_setup = True
            logger.info("HE backend set up successfully")

        except ImportError as e:
            logger.warning(f"Failed to initialize backend: {e}")
            self._is_setup = True  # Allow simulation mode

    def encrypt(self, plaintext: np.ndarray) -> Any:
        """Encrypt a plaintext vector."""
        if self._backend is not None:
            return self._backend.encrypt(plaintext)
        # Simulation: just return the data
        return {"data": plaintext.copy(), "is_encrypted": False}

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        """Decrypt a ciphertext."""
        if self._backend is not None:
            return self._backend.decrypt(ciphertext, output_size)
        # Simulation: return the data
        if isinstance(ciphertext, dict):
            return ciphertext["data"][:output_size] if output_size > 0 else ciphertext["data"]
        return ciphertext

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Any:
        """Compute encrypted LoRA delta."""
        if self._backend is not None and hasattr(self._backend, 'lora_delta'):
            return self._backend.lora_delta(ct_x, lora_a, lora_b, scaling)

        # Simulation
        if isinstance(ct_x, dict):
            x = ct_x["data"]
        else:
            x = ct_x

        intermediate = x @ lora_a.T
        result = intermediate @ lora_b.T
        return {"data": scaling * result, "is_encrypted": False}

    def get_slot_count(self) -> int:
        """Get number of SIMD slots."""
        if self._backend is not None:
            return self._backend.get_slot_count()
        return self._params.get('poly_modulus_degree', 8192) // 2

    def get_operation_stats(self) -> Dict[str, int]:
        """Get operation statistics."""
        if self._backend is not None and hasattr(self._backend, 'get_counters'):
            return self._backend.get_counters()
        return {"rotations": 0, "multiplications": 0, "rescales": 0}

    def create_column_packed_matrix(self, matrix: np.ndarray) -> Any:
        """Create a column-packed matrix for rotation-free multiplication."""
        if self._backend is not None and hasattr(self._backend, 'create_column_packed_matrix'):
            return self._backend.create_column_packed_matrix(matrix)
        return matrix  # Just return as-is in simulation

    def column_packed_matmul(self, ct: Any, packed_matrix: Any, rescale: bool = True) -> Any:
        """Perform column-packed matrix multiplication."""
        if self._backend is not None and hasattr(self._backend, 'column_packed_matmul'):
            return self._backend.column_packed_matmul(ct, packed_matrix, rescale)

        # Simulation
        if isinstance(ct, dict):
            x = ct["data"]
        else:
            x = ct

        if isinstance(packed_matrix, np.ndarray):
            result = x @ packed_matrix.T
        else:
            result = x @ packed_matrix

        return {"data": result, "is_encrypted": False}


# Export for backward compatibility
__all__ = [
    'HELoRAAdapter',
    'HELoRAConfig',
    'HELoRAMetrics',
    'HEBackend',
]
