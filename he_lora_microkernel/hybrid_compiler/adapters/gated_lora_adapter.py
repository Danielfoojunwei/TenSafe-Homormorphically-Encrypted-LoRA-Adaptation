"""
HE Gated LoRA Adapter Implementation

Implements the gated LoRA adapter using hybrid CKKS-TFHE encryption:
    y = Wx + g(x) * B(Ax)

Where:
    - Wx is the base model output (computed externally)
    - B(Ax) is the LoRA delta (CKKS matmuls)
    - g(x) = LUT(w_g^T @ x + b_g) is the discrete gate (TFHE)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
import logging
import time
import numpy as np

from ..backend import (
    HybridHEBackend,
    HybridHEConfig,
    CKKSCiphertext,
    TFHECiphertext,
    HybridExecutionError,
)
from ..tfhe_lut import LUTLibrary

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GatedLoRAAdapterConfig:
    """Configuration for gated LoRA adapter."""
    # LoRA parameters
    hidden_size: int
    lora_rank: int
    lora_alpha: float = 32.0

    # Gate parameters
    gate_type: str = "step"  # "step" or "sign"
    gate_lut_id: Optional[str] = None  # Override LUT name

    # Quantization for bridge
    quantization_bits: int = 8
    quantization_clip_min: float = -10.0
    quantization_clip_max: float = 10.0

    # Telemetry
    enable_telemetry: bool = True

    @property
    def scaling_factor(self) -> float:
        """LoRA scaling factor (alpha / rank)."""
        return self.lora_alpha / self.lora_rank


@dataclass
class AdapterMetrics:
    """Metrics from adapter forward pass."""
    # Timing (milliseconds)
    total_time_ms: float = 0.0
    ckks_lora_time_ms: float = 0.0
    ckks_gate_pre_time_ms: float = 0.0
    bridge_to_tfhe_time_ms: float = 0.0
    tfhe_lut_time_ms: float = 0.0
    bridge_to_ckks_time_ms: float = 0.0
    ckks_apply_gate_time_ms: float = 0.0
    ckks_final_add_time_ms: float = 0.0

    # Gate output
    gate_value: float = 0.0

    # Quantization error
    quantization_error: float = 0.0

    # Operation counts
    ckks_ops: int = 0
    tfhe_ops: int = 0
    bootstrap_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_time_ms": self.total_time_ms,
            "ckks_lora_time_ms": self.ckks_lora_time_ms,
            "ckks_gate_pre_time_ms": self.ckks_gate_pre_time_ms,
            "bridge_to_tfhe_time_ms": self.bridge_to_tfhe_time_ms,
            "tfhe_lut_time_ms": self.tfhe_lut_time_ms,
            "bridge_to_ckks_time_ms": self.bridge_to_ckks_time_ms,
            "ckks_apply_gate_time_ms": self.ckks_apply_gate_time_ms,
            "ckks_final_add_time_ms": self.ckks_final_add_time_ms,
            "gate_value": self.gate_value,
            "quantization_error": self.quantization_error,
            "ckks_ops": self.ckks_ops,
            "tfhe_ops": self.tfhe_ops,
            "bootstrap_count": self.bootstrap_count,
        }


@dataclass
class AdapterWeights:
    """Weights for a gated LoRA adapter."""
    lora_A: np.ndarray  # Shape: [rank, hidden_size]
    lora_B: np.ndarray  # Shape: [hidden_size, rank]
    w_gate: np.ndarray  # Shape: [hidden_size]
    b_gate: Optional[np.ndarray] = None  # Shape: [1] or scalar


# =============================================================================
# Adapter Protocol
# =============================================================================

class NonLinearAdapter(Protocol):
    """Protocol for non-linear adapter implementations."""

    @property
    def adapter_type(self) -> str:
        """Adapter type identifier."""
        ...

    @property
    def requires_tfhe(self) -> bool:
        """Whether this adapter requires TFHE operations."""
        ...

    def forward(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
        weights: AdapterWeights,
    ) -> Tuple[np.ndarray, AdapterMetrics]:
        """Compute adapter output (plaintext reference)."""
        ...

    def forward_encrypted(
        self,
        ct_x: CKKSCiphertext,
        ct_base: CKKSCiphertext,
        weights: AdapterWeights,
        request_id: str,
    ) -> Tuple[CKKSCiphertext, AdapterMetrics]:
        """Compute adapter output under encryption."""
        ...


# =============================================================================
# Gated LoRA Adapter Implementation
# =============================================================================

class HEGatedLoRAAdapter:
    """
    Gated LoRA adapter using hybrid CKKS-TFHE encryption.

    Implements: y = Wx + g(x) * scaling * B(Ax)
    Where g(x) = LUT(w_g^T @ x + b_g)

    Execution Phases:
    1. CKKS_LORA_DELTA: Compute delta = B(Ax) using CKKS matmuls
    2. CKKS_GATE_PRE: Compute z = w_g^T @ x + b_g using CKKS
    3. BRIDGE_TO_TFHE: Quantize z and convert to TFHE
    4. TFHE_GATE_EVAL: Apply gate LUT via programmable bootstrap
    5. BRIDGE_TO_CKKS: Convert gate result back to CKKS
    6. CKKS_APPLY_GATE: Compute gated_delta = g * delta
    7. CKKS_FINAL_ADD: y = base + gated_delta

    Usage:
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        # Plaintext forward
        output, metrics = adapter.forward(x, base_output, weights)

        # Encrypted forward
        ct_output, metrics = adapter.forward_encrypted(
            ct_x, ct_base, weights, request_id
        )
    """

    def __init__(
        self,
        config: GatedLoRAAdapterConfig,
        backend: HybridHEBackend,
        lut_library: Optional[LUTLibrary] = None,
    ):
        self._config = config
        self._backend = backend
        self._lut_library = lut_library or LUTLibrary(config.quantization_bits)

        # Determine LUT to use
        self._lut_id = config.gate_lut_id or config.gate_type

        # Verify LUT exists
        if self._lut_library.get(self._lut_id) is None:
            raise ValueError(f"Unknown gate LUT: {self._lut_id}")

    @property
    def adapter_type(self) -> str:
        return "gated_lora"

    @property
    def requires_tfhe(self) -> bool:
        return True

    # -------------------------------------------------------------------------
    # Plaintext Forward (Reference Implementation)
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
        weights: AdapterWeights,
    ) -> Tuple[np.ndarray, AdapterMetrics]:
        """
        Compute gated LoRA output (plaintext reference).

        Args:
            x: Input activation [hidden_size] or [batch, hidden_size]
            base_output: Base model output Wx
            weights: Adapter weights

        Returns:
            Tuple of (output, metrics)
        """
        metrics = AdapterMetrics()
        start_time = time.perf_counter()

        # Flatten for consistency
        x_flat = x.flatten() if x.ndim > 1 else x
        base_flat = base_output.flatten() if base_output.ndim > 1 else base_output

        # Phase 1: LoRA delta
        t0 = time.perf_counter()
        u = weights.lora_A @ x_flat  # [rank]
        delta = weights.lora_B @ u   # [hidden_size]
        metrics.ckks_lora_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 2

        # Phase 2: Gate pre-activation
        t0 = time.perf_counter()
        z = weights.w_gate @ x_flat
        if weights.b_gate is not None:
            b = weights.b_gate.flat[0] if hasattr(weights.b_gate, 'flat') else float(weights.b_gate)
            z = z + b
        metrics.ckks_gate_pre_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 1

        # Phase 3-5: Gate via LUT (simulate quantization and LUT)
        t0 = time.perf_counter()

        # Quantize
        z_q = self._quantize_for_lut(float(z))
        metrics.bridge_to_tfhe_time_ms = (time.perf_counter() - t0) * 1000

        # Compute quantization error
        z_dq = self._dequantize_from_lut(z_q)
        metrics.quantization_error = abs(float(z) - z_dq)

        # Apply LUT
        t0 = time.perf_counter()
        lut = self._lut_library.get(self._lut_id)
        g_encoded = lut.lookup(z_q)
        g = lut.decode_output(g_encoded)
        metrics.tfhe_lut_time_ms = (time.perf_counter() - t0) * 1000
        metrics.tfhe_ops += 1
        metrics.bootstrap_count += 1

        # Convert gate to float
        metrics.bridge_to_ckks_time_ms = 0.0  # Trivial in plaintext
        g_float = float(g)
        metrics.gate_value = g_float

        # Phase 6: Apply gate
        t0 = time.perf_counter()
        gated_delta = g_float * self._config.scaling_factor * delta
        metrics.ckks_apply_gate_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 1

        # Phase 7: Final add
        t0 = time.perf_counter()
        output = base_flat + gated_delta
        metrics.ckks_final_add_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 1

        metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

        # Reshape output to match input
        if x.ndim > 1:
            output = output.reshape(x.shape)

        return output, metrics

    # -------------------------------------------------------------------------
    # Encrypted Forward
    # -------------------------------------------------------------------------

    def forward_encrypted(
        self,
        ct_x: CKKSCiphertext,
        ct_base: CKKSCiphertext,
        weights: AdapterWeights,
        request_id: str,
    ) -> Tuple[CKKSCiphertext, AdapterMetrics]:
        """
        Compute gated LoRA output under hybrid encryption.

        Args:
            ct_x: CKKS ciphertext of input activation
            ct_base: CKKS ciphertext of base model output
            weights: Adapter weights (plaintext, encrypted in TGSP at rest)
            request_id: Request ID for bridge session

        Returns:
            Tuple of (encrypted output, metrics)
        """
        metrics = AdapterMetrics()
        start_time = time.perf_counter()

        # Phase 1: CKKS LoRA delta
        t0 = time.perf_counter()
        ct_delta = self._backend.ckks_matmul(ct_x, weights.lora_A, weights.lora_B)
        metrics.ckks_lora_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 2

        # Phase 2: CKKS gate pre-activation
        t0 = time.perf_counter()
        ct_z = self._backend.ckks_matvec(ct_x, weights.w_gate.reshape(1, -1))
        if weights.b_gate is not None:
            # Add bias (constant add)
            b = float(weights.b_gate.flat[0] if hasattr(weights.b_gate, 'flat') else weights.b_gate)
            # Create bias ciphertext
            bias_pt = np.array([b])
            ct_bias = self._backend.ckks_encrypt(bias_pt)
            ct_z = self._backend.ckks_add(ct_z, ct_bias)
        metrics.ckks_gate_pre_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 2

        # Phase 3: Bridge CKKS -> TFHE
        t0 = time.perf_counter()
        ct_z_tfhe = self._backend.bridge_ckks_to_tfhe(
            ct_z,
            request_id,
            self._config.quantization_bits,
        )
        metrics.bridge_to_tfhe_time_ms = (time.perf_counter() - t0) * 1000

        # Phase 4: TFHE LUT evaluation
        t0 = time.perf_counter()
        ct_g_tfhe = self._backend.tfhe_lut_apply(ct_z_tfhe, self._lut_id)
        metrics.tfhe_lut_time_ms = (time.perf_counter() - t0) * 1000
        metrics.tfhe_ops += 1
        metrics.bootstrap_count += 1

        # Phase 5: Bridge TFHE -> CKKS
        t0 = time.perf_counter()
        ct_g = self._backend.bridge_tfhe_to_ckks(ct_g_tfhe, request_id)
        metrics.bridge_to_ckks_time_ms = (time.perf_counter() - t0) * 1000

        # Record gate value (only available in simulation)
        if ct_g._plaintext is not None:
            lut = self._lut_library.get(self._lut_id)
            encoded = int(ct_g._plaintext.flat[0])
            metrics.gate_value = float(lut.decode_output(encoded))

        # Phase 6: CKKS apply gate (scalar multiply + scale)
        t0 = time.perf_counter()
        # First multiply delta by scaling factor
        scaling_ct = self._backend.ckks_encrypt(
            np.array([self._config.scaling_factor])
        )
        ct_scaled_delta = self._backend.ckks_scalar_mul(ct_delta, scaling_ct)
        # Then multiply by gate
        ct_gated_delta = self._backend.ckks_scalar_mul(ct_scaled_delta, ct_g)
        metrics.ckks_apply_gate_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 2

        # Phase 7: CKKS final add
        t0 = time.perf_counter()
        ct_output = self._backend.ckks_add(ct_base, ct_gated_delta)
        metrics.ckks_final_add_time_ms = (time.perf_counter() - t0) * 1000
        metrics.ckks_ops += 1

        metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

        return ct_output, metrics

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _quantize_for_lut(self, value: float) -> int:
        """Quantize a float value for LUT lookup."""
        cfg = self._config
        clamped = max(cfg.quantization_clip_min, min(cfg.quantization_clip_max, value))
        max_int = (1 << (cfg.quantization_bits - 1)) - 1
        max_abs = max(abs(cfg.quantization_clip_min), abs(cfg.quantization_clip_max))
        scale = max_int / max_abs
        return int(round(clamped * scale))

    def _dequantize_from_lut(self, value: int) -> float:
        """Dequantize an integer from LUT to float."""
        cfg = self._config
        max_int = (1 << (cfg.quantization_bits - 1)) - 1
        max_abs = max(abs(cfg.quantization_clip_min), abs(cfg.quantization_clip_max))
        scale = max_int / max_abs
        return value / scale


# =============================================================================
# Factory Functions
# =============================================================================

def create_gated_lora_adapter(
    hidden_size: int,
    lora_rank: int,
    lora_alpha: float = 32.0,
    gate_type: str = "step",
    backend: Optional[HybridHEBackend] = None,
) -> HEGatedLoRAAdapter:
    """
    Create a gated LoRA adapter with default configuration.

    Args:
        hidden_size: Model hidden size
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling alpha
        gate_type: Gate type ("step" or "sign")
        backend: Optional pre-configured backend

    Returns:
        Configured HEGatedLoRAAdapter
    """
    config = GatedLoRAAdapterConfig(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gate_type=gate_type,
    )

    if backend is None:
        backend = HybridHEBackend.create_simulated()

    return HEGatedLoRAAdapter(config, backend)


def plaintext_gated_lora(
    x: np.ndarray,
    base_output: np.ndarray,
    lora_A: np.ndarray,
    lora_B: np.ndarray,
    w_gate: np.ndarray,
    b_gate: float = 0.0,
    alpha: float = 32.0,
    rank: Optional[int] = None,
    gate_type: str = "step",
    return_metrics: bool = False,
) -> np.ndarray | Tuple[np.ndarray, AdapterMetrics]:
    """
    Plaintext reference implementation of gated LoRA.

    y = Wx + g(x) * (alpha/rank) * B(Ax)
    g(x) = LUT(w_g^T x + b_g)

    Args:
        x: Input activation
        base_output: Base model output Wx
        lora_A: LoRA A matrix [rank, hidden_size]
        lora_B: LoRA B matrix [hidden_size, rank]
        w_gate: Gate weight vector [hidden_size]
        b_gate: Gate bias scalar
        alpha: LoRA alpha
        rank: LoRA rank (inferred from lora_A if not provided)
        gate_type: "step" or "sign"
        return_metrics: If True, return (output, metrics)

    Returns:
        Output array, or (output, metrics) if return_metrics=True
    """
    if rank is None:
        rank = lora_A.shape[0]

    config = GatedLoRAAdapterConfig(
        hidden_size=x.shape[-1],
        lora_rank=rank,
        lora_alpha=alpha,
        gate_type=gate_type,
    )

    backend = HybridHEBackend.create_simulated()
    adapter = HEGatedLoRAAdapter(config, backend)

    weights = AdapterWeights(
        lora_A=lora_A,
        lora_B=lora_B,
        w_gate=w_gate,
        b_gate=np.array([b_gate]) if b_gate is not None else None,
    )

    output, metrics = adapter.forward(x, base_output, weights)

    if return_metrics:
        return output, metrics
    return output
