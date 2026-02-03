"""
HE-Integrated Adapter Forward Pass Module.

This module provides the critical bridge between the adapter types
and the actual HE (Homomorphic Encryption) backends (CKKS, TFHE).

IMPORTANT: Not all adapter types are HE-compatible!

HE-Compatible (CKKS):
    - LoRA: Linear, fits CKKS depth-2 budget
    - rsLoRA: Same as LoRA with different scaling
    - LoRA-FA: Same computation, A is frozen

Partially HE-Compatible (requires TFHE hybrid):
    - VeRA: Element-wise scaling can use TFHE
    - GatedLoRA: Gate requires TFHE bootstrap

NOT HE-Compatible (plaintext only):
    - DoRA: Requires L2 norm and division
    - AdaLoRA: Runtime rank masking is non-linear
    - GLoRA: Weight scaling is non-linear

References:
    - MOAI Paper: Column packing for rotation-free multiplication
    - CKKS: Approximate arithmetic on encrypted data
    - TFHE: Exact discrete computation with bootstrapping

Author: TenSafe Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import time

from .adapter_types import (
    AdapterType,
    AdapterConfig,
    BaseAdapter,
    LoRAAdapter,
    rsLoRAAdapter,
    LoRAFAAdapter,
    DoRAAdapter,
    VeRAAdapter,
)

logger = logging.getLogger(__name__)


class HECompatibility(Enum):
    """HE compatibility level for adapter types."""
    FULL_CKKS = "full_ckks"           # Fully compatible with CKKS
    HYBRID_TFHE = "hybrid_tfhe"       # Requires CKKS + TFHE hybrid
    PLAINTEXT_ONLY = "plaintext_only"  # Cannot run under HE


# Compatibility mapping
ADAPTER_HE_COMPATIBILITY: Dict[AdapterType, HECompatibility] = {
    AdapterType.LORA: HECompatibility.FULL_CKKS,
    AdapterType.RS_LORA: HECompatibility.FULL_CKKS,
    AdapterType.LORA_FA: HECompatibility.FULL_CKKS,
    AdapterType.DORA: HECompatibility.PLAINTEXT_ONLY,  # Needs L2 norm
    AdapterType.VERA: HECompatibility.HYBRID_TFHE,     # Element-wise scaling
    AdapterType.ADALORA: HECompatibility.PLAINTEXT_ONLY,  # Runtime masking
    AdapterType.GATED_LORA: HECompatibility.HYBRID_TFHE,  # Sigmoid gate
    AdapterType.GLORA: HECompatibility.PLAINTEXT_ONLY,  # Weight scaling
}


def get_he_compatibility(adapter_type: AdapterType) -> HECompatibility:
    """Get HE compatibility level for an adapter type."""
    return ADAPTER_HE_COMPATIBILITY.get(adapter_type, HECompatibility.PLAINTEXT_ONLY)


def is_he_compatible(adapter_type: AdapterType) -> bool:
    """Check if adapter type can run under HE."""
    return get_he_compatibility(adapter_type) != HECompatibility.PLAINTEXT_ONLY


@dataclass
class HEForwardConfig:
    """Configuration for HE forward pass."""
    # CKKS parameters
    use_column_packing: bool = True  # MOAI rotation-free multiplication
    rescale_after_multiply: bool = True

    # TFHE hybrid parameters (for gated adapters)
    enable_tfhe_hybrid: bool = False
    max_bootstraps_per_layer: int = 2
    tfhe_message_bits: int = 8

    # Performance
    prefetch_weights: bool = True

    # Validation
    enforce_rotation_budget: bool = True
    max_rotations_per_token: int = 16


@dataclass
class HEForwardMetrics:
    """Metrics from HE forward pass."""
    encrypt_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    decrypt_time_ms: float = 0.0
    total_time_ms: float = 0.0

    rotations: int = 0
    keyswitches: int = 0
    rescales: int = 0
    bootstraps: int = 0  # TFHE only

    def __add__(self, other: 'HEForwardMetrics') -> 'HEForwardMetrics':
        return HEForwardMetrics(
            encrypt_time_ms=self.encrypt_time_ms + other.encrypt_time_ms,
            compute_time_ms=self.compute_time_ms + other.compute_time_ms,
            decrypt_time_ms=self.decrypt_time_ms + other.decrypt_time_ms,
            total_time_ms=self.total_time_ms + other.total_time_ms,
            rotations=self.rotations + other.rotations,
            keyswitches=self.keyswitches + other.keyswitches,
            rescales=self.rescales + other.rescales,
            bootstraps=self.bootstraps + other.bootstraps,
        )


class HEAdapterForward(ABC):
    """
    Abstract base class for HE-integrated adapter forward passes.

    Subclasses implement the actual HE computation using the appropriate
    backend (CKKS, TFHE, or hybrid).
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        config: Optional[HEForwardConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or HEForwardConfig()
        self._metrics = HEForwardMetrics()
        self._backend = None
        self._packed_weights: Dict[str, Any] = {}

    @abstractmethod
    def initialize_backend(self, backend: Any) -> None:
        """Initialize with HE backend."""
        pass

    @abstractmethod
    def forward(
        self,
        x_plain: np.ndarray,
        module_name: str,
    ) -> np.ndarray:
        """
        Compute adapter delta under HE.

        Args:
            x_plain: Plaintext input activation
            module_name: Target module name

        Returns:
            Decrypted delta
        """
        pass

    def get_metrics(self) -> HEForwardMetrics:
        """Get accumulated metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics = HEForwardMetrics()


class CKKSLoRAForward(HEAdapterForward):
    """
    CKKS-based forward pass for linear LoRA adapters.

    Uses MOAI column packing for rotation-free matrix multiplication.

    Supports: LoRA, rsLoRA, LoRA-FA
    """

    def __init__(
        self,
        adapter: Union[LoRAAdapter, rsLoRAAdapter, LoRAFAAdapter],
        config: Optional[HEForwardConfig] = None,
    ):
        # Validate adapter type
        compat = get_he_compatibility(adapter.config.adapter_type)
        if compat != HECompatibility.FULL_CKKS:
            raise ValueError(
                f"CKKSLoRAForward requires CKKS-compatible adapter, "
                f"got {adapter.config.adapter_type} with compatibility {compat}"
            )

        super().__init__(adapter, config)
        self._scaling = adapter.config.scaling

    def initialize_backend(self, backend: Any) -> None:
        """
        Initialize CKKS backend and pre-pack weights.

        Args:
            backend: GPUCKKSBackend instance
        """
        self._backend = backend

        # Pre-pack weights for column-packed multiplication
        if self.config.use_column_packing:
            self._prepack_weights()

    def _prepack_weights(self) -> None:
        """Pre-pack LoRA weights for MOAI column-packed multiplication."""
        params = self.adapter.get_trainable_params()
        frozen = self.adapter.get_frozen_params()

        # Get A matrix (may be frozen for LoRA-FA)
        if "lora_A" in params:
            lora_a = params["lora_A"]
        elif "lora_A" in frozen:
            lora_a = frozen["lora_A"]
        else:
            raise ValueError("No lora_A found in adapter")

        # Get B matrix
        lora_b = params.get("lora_B")
        if lora_b is None:
            raise ValueError("No lora_B found in adapter")

        # Pre-pack if backend supports it
        if hasattr(self._backend, 'create_column_packed_matrix'):
            self._packed_weights["lora_A"] = self._backend.create_column_packed_matrix(
                lora_a.astype(np.float64)
            )
            self._packed_weights["lora_B"] = self._backend.create_column_packed_matrix(
                lora_b.astype(np.float64)
            )
            logger.info("Pre-packed LoRA weights for MOAI column packing")
        else:
            # Store raw weights for standard multiplication
            self._packed_weights["lora_A"] = lora_a.astype(np.float64)
            self._packed_weights["lora_B"] = lora_b.astype(np.float64)

    def forward(
        self,
        x_plain: np.ndarray,
        module_name: str,
    ) -> np.ndarray:
        """
        Compute LoRA delta: Δ = scaling * (x @ A.T) @ B.T

        Under CKKS with MOAI column packing:
        1. Encrypt x
        2. Column-packed matmul: ct_x × A_packed (0 rotations)
        3. Rescale
        4. Column-packed matmul: ct_int × B_packed (0 rotations)
        5. Multiply by scaling
        6. Decrypt
        """
        if self._backend is None:
            raise RuntimeError("Backend not initialized. Call initialize_backend() first.")

        start_time = time.perf_counter()

        # Flatten input
        original_shape = x_plain.shape
        x_flat = x_plain.flatten().astype(np.float64)

        # === ENCRYPT ===
        t0 = time.perf_counter()
        ct_x = self._backend.encrypt(x_flat)
        self._metrics.encrypt_time_ms += (time.perf_counter() - t0) * 1000

        # === COMPUTE ===
        t1 = time.perf_counter()

        lora_a = self._packed_weights["lora_A"]
        lora_b = self._packed_weights["lora_B"]

        if hasattr(self._backend, 'column_packed_matmul'):
            # MOAI rotation-free multiplication
            ct_intermediate = self._backend.column_packed_matmul(
                ct_x, lora_a, rescale=self.config.rescale_after_multiply
            )
            ct_result = self._backend.column_packed_matmul(
                ct_intermediate, lora_b, rescale=self.config.rescale_after_multiply
            )
            # Metrics: 0 rotations!
            self._metrics.rescales += 2
        else:
            # Standard multiplication (with rotations)
            ct_intermediate = self._backend.ct_pt_multiply(ct_x, lora_a.T)
            ct_result = self._backend.ct_pt_multiply(ct_intermediate, lora_b.T)

        # Apply scaling
        scaling_array = np.array([self._scaling], dtype=np.float64)
        ct_result = self._backend.multiply_plain(ct_result, scaling_array)

        self._metrics.compute_time_ms += (time.perf_counter() - t1) * 1000

        # === DECRYPT ===
        t2 = time.perf_counter()
        delta = self._backend.decrypt(ct_result, output_size=len(x_flat))
        self._metrics.decrypt_time_ms += (time.perf_counter() - t2) * 1000

        # Update counters from backend
        if hasattr(self._backend, 'get_counters'):
            counters = self._backend.get_counters()
            self._metrics.rotations = counters.get('rotations', 0)
            self._metrics.keyswitches = counters.get('keyswitches', 0)

        self._metrics.total_time_ms += (time.perf_counter() - start_time) * 1000

        # Reshape to match input
        return delta.reshape(original_shape)


class HybridTFHEGatedForward(HEAdapterForward):
    """
    Hybrid CKKS+TFHE forward pass for gated adapters.

    Uses:
    - CKKS for linear LoRA computation
    - TFHE for gate evaluation (sigmoid via LUT)

    This matches the hybrid_compiler architecture in the codebase.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        config: Optional[HEForwardConfig] = None,
    ):
        config = config or HEForwardConfig(enable_tfhe_hybrid=True)
        super().__init__(adapter, config)

        self._ckks_backend = None
        self._tfhe_backend = None

    def initialize_backend(self, backend: Any) -> None:
        """
        Initialize hybrid backend.

        Args:
            backend: Should have both CKKS and TFHE capabilities
        """
        self._ckks_backend = backend

        # Try to get TFHE backend from hybrid interface
        if hasattr(backend, 'get_tfhe_backend'):
            self._tfhe_backend = backend.get_tfhe_backend()
        else:
            logger.warning(
                "No TFHE backend available. Gate will be computed in plaintext. "
                "This reduces security for gated adapters."
            )

    def forward(
        self,
        x_plain: np.ndarray,
        module_name: str,
    ) -> np.ndarray:
        """
        Compute gated LoRA delta: Δ = gate(x) * scaling * (x @ A.T) @ B.T

        Under hybrid CKKS+TFHE:
        1. CKKS: Encrypt x and compute linear LoRA delta
        2. TFHE: Compute gate = σ(x @ W_g) via LUT bootstrap
        3. TFHE → CKKS: Convert gate back to CKKS
        4. CKKS: Multiply gate * delta
        5. Decrypt
        """
        if self._ckks_backend is None:
            raise RuntimeError("Backend not initialized")

        start_time = time.perf_counter()

        params = self.adapter.get_trainable_params()
        lora_a = params.get("lora_A")
        lora_b = params.get("lora_B")

        if lora_a is None or lora_b is None:
            raise ValueError("Missing LoRA weights")

        original_shape = x_plain.shape
        x_flat = x_plain.flatten().astype(np.float64)

        # === CKKS: Linear LoRA computation ===
        t0 = time.perf_counter()
        ct_x = self._ckks_backend.encrypt(x_flat)
        self._metrics.encrypt_time_ms += (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        ct_intermediate = self._ckks_backend.ct_pt_multiply(ct_x, lora_a.T)
        ct_delta = self._ckks_backend.ct_pt_multiply(ct_intermediate, lora_b.T)

        # === TFHE: Gate computation ===
        if self._tfhe_backend is not None:
            # Get gate projection weights
            gate_proj = params.get("gate_proj")

            if gate_proj is not None:
                # Compute gate input (can be done in CKKS)
                ct_gate_input = self._ckks_backend.ct_pt_multiply(ct_x, gate_proj.T)

                # Convert CKKS → TFHE for sigmoid
                tfhe_gate_input = self._ckks_to_tfhe(ct_gate_input)

                # TFHE sigmoid via LUT (this is the bootstrap)
                tfhe_gate = self._tfhe_backend.apply_lut(tfhe_gate_input, "sigmoid")
                self._metrics.bootstraps += 1

                # Convert TFHE → CKKS
                ct_gate = self._tfhe_to_ckks(tfhe_gate)

                # Multiply gate * delta
                ct_result = self._ckks_backend.ct_ct_multiply(ct_gate, ct_delta)
            else:
                ct_result = ct_delta
        else:
            # Fallback: compute gate in plaintext (reduced security)
            gate_proj = params.get("gate_proj")
            if gate_proj is not None:
                gate_logits = x_flat @ gate_proj.T
                gate = 1.0 / (1.0 + np.exp(-gate_logits))

                # Apply plaintext gate to encrypted delta
                ct_result = self._ckks_backend.multiply_plain(
                    ct_delta, gate.astype(np.float64)
                )
            else:
                ct_result = ct_delta

        # Apply scaling
        scaling = self.adapter.config.scaling
        ct_result = self._ckks_backend.multiply_plain(
            ct_result, np.array([scaling], dtype=np.float64)
        )

        self._metrics.compute_time_ms += (time.perf_counter() - t1) * 1000

        # === DECRYPT ===
        t2 = time.perf_counter()
        delta = self._ckks_backend.decrypt(ct_result, output_size=len(x_flat))
        self._metrics.decrypt_time_ms += (time.perf_counter() - t2) * 1000

        self._metrics.total_time_ms += (time.perf_counter() - start_time) * 1000

        return delta.reshape(original_shape)

    def _ckks_to_tfhe(self, ct_ckks: Any) -> Any:
        """Convert CKKS ciphertext to TFHE."""
        if hasattr(self._ckks_backend, 'ckks_to_tfhe'):
            return self._ckks_backend.ckks_to_tfhe(ct_ckks)
        raise NotImplementedError("CKKS→TFHE conversion not available")

    def _tfhe_to_ckks(self, ct_tfhe: Any) -> Any:
        """Convert TFHE ciphertext to CKKS."""
        if hasattr(self._ckks_backend, 'tfhe_to_ckks'):
            return self._ckks_backend.tfhe_to_ckks(ct_tfhe)
        raise NotImplementedError("TFHE→CKKS conversion not available")


class PlaintextFallbackForward(HEAdapterForward):
    """
    Plaintext fallback for HE-incompatible adapters.

    Used for adapters that cannot run under HE (DoRA, AdaLoRA, GLoRA)
    or when HE is disabled.

    WARNING: This provides no encryption. Use only when:
    - Testing/development
    - HE-incompatible adapter types
    - Explicitly disabled HE mode
    """

    def initialize_backend(self, backend: Any) -> None:
        """No backend needed for plaintext."""
        self._backend = None
        logger.warning(
            f"Using plaintext fallback for {self.adapter.config.adapter_type}. "
            f"NO ENCRYPTION is applied."
        )

    def forward(
        self,
        x_plain: np.ndarray,
        module_name: str,
    ) -> np.ndarray:
        """Compute delta in plaintext using adapter's native forward."""
        start_time = time.perf_counter()

        # Use adapter's own forward method
        delta = self.adapter.forward(x_plain)

        self._metrics.compute_time_ms += (time.perf_counter() - start_time) * 1000
        self._metrics.total_time_ms = self._metrics.compute_time_ms

        return delta


# =============================================================================
# FACTORY
# =============================================================================

def create_he_forward(
    adapter: BaseAdapter,
    config: Optional[HEForwardConfig] = None,
    force_plaintext: bool = False,
) -> HEAdapterForward:
    """
    Create appropriate HE forward pass for an adapter.

    Args:
        adapter: The adapter instance
        config: HE forward configuration
        force_plaintext: If True, always use plaintext fallback

    Returns:
        HEAdapterForward implementation appropriate for the adapter type
    """
    if force_plaintext:
        return PlaintextFallbackForward(adapter, config)

    adapter_type = adapter.config.adapter_type
    compatibility = get_he_compatibility(adapter_type)

    if compatibility == HECompatibility.FULL_CKKS:
        return CKKSLoRAForward(adapter, config)

    elif compatibility == HECompatibility.HYBRID_TFHE:
        config = config or HEForwardConfig()
        if config.enable_tfhe_hybrid:
            return HybridTFHEGatedForward(adapter, config)
        else:
            logger.warning(
                f"Adapter {adapter_type} requires TFHE hybrid but it's disabled. "
                f"Falling back to plaintext."
            )
            return PlaintextFallbackForward(adapter, config)

    else:  # PLAINTEXT_ONLY
        logger.warning(
            f"Adapter {adapter_type} is not HE-compatible. "
            f"Using plaintext computation."
        )
        return PlaintextFallbackForward(adapter, config)


# =============================================================================
# INTEGRATION WITH HOT-SWAP
# =============================================================================

class HEAwareHotSwapForward:
    """
    HE-aware forward computation that integrates with hot-swap.

    This bridges the hot-swap manager with the HE backends,
    ensuring the correct HE forward implementation is used
    based on the active adapter type.
    """

    def __init__(self, backend: Any):
        """
        Initialize with HE backend.

        Args:
            backend: GPUCKKSBackend or hybrid backend
        """
        self._backend = backend
        self._active_forward: Optional[HEAdapterForward] = None
        self._forward_cache: Dict[str, HEAdapterForward] = {}

    def set_active_adapter(
        self,
        adapter_id: str,
        adapter: BaseAdapter,
        config: Optional[HEForwardConfig] = None,
    ) -> None:
        """
        Set the active adapter for HE forward.

        Args:
            adapter_id: Adapter identifier
            adapter: Adapter instance
            config: HE configuration
        """
        # Check cache
        if adapter_id in self._forward_cache:
            self._active_forward = self._forward_cache[adapter_id]
            return

        # Create new HE forward
        he_forward = create_he_forward(adapter, config)
        he_forward.initialize_backend(self._backend)

        # Cache and set active
        self._forward_cache[adapter_id] = he_forward
        self._active_forward = he_forward

        logger.info(
            f"Set active HE forward for {adapter_id}: "
            f"{type(he_forward).__name__}"
        )

    def forward(
        self,
        x_plain: np.ndarray,
        module_name: str,
    ) -> np.ndarray:
        """
        Compute forward pass with active adapter under HE.

        Args:
            x_plain: Plaintext input
            module_name: Target module

        Returns:
            Decrypted delta
        """
        if self._active_forward is None:
            raise RuntimeError("No active adapter set")

        return self._active_forward.forward(x_plain, module_name)

    def get_metrics(self) -> HEForwardMetrics:
        """Get metrics from active forward."""
        if self._active_forward:
            return self._active_forward.get_metrics()
        return HEForwardMetrics()

    def clear_cache(self) -> None:
        """Clear forward cache."""
        self._forward_cache.clear()
        self._active_forward = None


__all__ = [
    "HECompatibility",
    "ADAPTER_HE_COMPATIBILITY",
    "get_he_compatibility",
    "is_he_compatible",
    "HEForwardConfig",
    "HEForwardMetrics",
    "HEAdapterForward",
    "CKKSLoRAForward",
    "HybridTFHEGatedForward",
    "PlaintextFallbackForward",
    "create_he_forward",
    "HEAwareHotSwapForward",
]
