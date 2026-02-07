"""
CKKS-TFHE Bridge Implementation (DEPRECATED for Server-Side Use)

This module provides utilities for CKKS-TFHE conversion, including:
- Quantization for CKKS->TFHE
- Re-encoding for TFHE->CKKS

==================== SECURITY MODEL ====================
As of the Client-Aided Bridge architecture, the server NEVER performs
scheme switching directly. The `ckks_to_tfhe` and `tfhe_to_ckks` methods
in this module are used for:
1. **Client-Side SDK**: The client decrypts CKKS, quantizes, and re-encrypts.
2. **Unit Testing**: Simulating the client's behavior for integration tests.

For production gated LoRA, the `GateLinkProtocol` (defined in has.proto) is
the secure mechanism. The server sends the encrypted gate signal to the client,
and the client returns the gate bit.
====================================================
"""


from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationParams:
    """Parameters for CKKS to discrete conversion."""
    # Bit width for quantization
    bits: int = 8

    # Clipping range
    clip_min: float = -10.0
    clip_max: float = 10.0

    # Symmetric quantization around zero
    symmetric: bool = True

    @property
    def scale(self) -> float:
        """Quantization scale factor."""
        max_int = (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1
        max_abs = max(abs(self.clip_min), abs(self.clip_max))
        return max_int / max_abs

    @property
    def message_space(self) -> int:
        """Size of discrete message space."""
        return 1 << self.bits

    def quantize(self, value: float) -> int:
        """Quantize a single value."""
        clamped = max(self.clip_min, min(self.clip_max, value))
        scaled = round(clamped * self.scale)
        if self.symmetric:
            max_val = (1 << (self.bits - 1)) - 1
            return max(-max_val, min(max_val, int(scaled)))
        else:
            return max(0, min((1 << self.bits) - 1, int(scaled)))

    def dequantize(self, value: int) -> float:
        """Dequantize a single value."""
        return value / self.scale


@dataclass
class BridgeConfig:
    """Configuration for CKKS-TFHE bridge."""
    # Quantization parameters
    quantization: QuantizationParams = field(default_factory=QuantizationParams)

    # CKKS parameters for re-encoding
    ckks_scale_bits: int = 40

    # TFHE parameters
    tfhe_noise_budget: float = 128.0

    # Verification
    enable_verification: bool = True
    max_quantization_error: float = 0.1


class CKKSTFHEBridge:
    """
    Bridge for CKKS-TFHE conversions.

    This class handles:
    1. CKKS value quantization
    2. Conversion to TFHE ciphertext format
    3. Conversion back to CKKS from TFHE
    4. Correctness verification

    Security: In production, conversions require client interaction.
    This implementation simulates the conversion for testing.
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()

        # Conversion statistics
        self._stats = {
            'ckks_to_tfhe_count': 0,
            'tfhe_to_ckks_count': 0,
            'total_quantization_error': 0.0,
            'max_quantization_error': 0.0,
        }

    def quantize_ckks_value(
        self,
        value: float,
        params: Optional[QuantizationParams] = None,
    ) -> Tuple[int, float]:
        """
        Quantize a CKKS-decrypted value to discrete integer.

        Args:
            value: The real-valued plaintext
            params: Quantization parameters (uses default if None)

        Returns:
            Tuple of (quantized_int, quantization_error)
        """
        params = params or self.config.quantization

        quantized = params.quantize(value)
        reconstructed = params.dequantize(quantized)
        error = abs(value - reconstructed)

        # Update statistics
        self._stats['total_quantization_error'] += error
        self._stats['max_quantization_error'] = max(
            self._stats['max_quantization_error'], error
        )

        return quantized, error

    def quantize_ckks_vector(
        self,
        values: np.ndarray,
        params: Optional[QuantizationParams] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Quantize a vector of CKKS values.

        Args:
            values: Array of real values
            params: Quantization parameters

        Returns:
            Tuple of (quantized_array, max_error)
        """
        params = params or self.config.quantization

        quantized = np.zeros(len(values), dtype=np.int32)
        max_error = 0.0

        for i, v in enumerate(values):
            q, e = self.quantize_ckks_value(v, params)
            quantized[i] = q
            max_error = max(max_error, e)

        return quantized, max_error

    def ckks_to_tfhe(
        self,
        ckks_plaintext: np.ndarray,
        params: Optional[QuantizationParams] = None,
    ) -> Dict[str, Any]:
        """
        Convert CKKS plaintext to TFHE ciphertext representation.

        NOTE: In production, this would be:
        1. Client decrypts CKKS ciphertext
        2. Client quantizes plaintext
        3. Client encrypts with TFHE key

        This simulation returns a dictionary representing the TFHE ciphertext.

        Args:
            ckks_plaintext: Decrypted CKKS values
            params: Quantization parameters

        Returns:
            TFHE ciphertext representation
        """
        params = params or self.config.quantization

        quantized, max_error = self.quantize_ckks_vector(ckks_plaintext, params)

        if self.config.enable_verification:
            if max_error > self.config.max_quantization_error:
                logger.warning(
                    f"High quantization error: {max_error:.4f} > {self.config.max_quantization_error}"
                )

        self._stats['ckks_to_tfhe_count'] += 1

        # Return TFHE ciphertext representation
        return {
            'type': 'tfhe_lwe',
            'values': quantized.tolist(),
            'params': {
                'bits': params.bits,
                'scale': params.scale,
                'symmetric': params.symmetric,
            },
            'noise_budget': self.config.tfhe_noise_budget,
        }

    def tfhe_to_ckks(
        self,
        tfhe_result: Dict[str, Any],
    ) -> np.ndarray:
        """
        Convert TFHE result back to CKKS plaintext representation.

        NOTE: In production, this would be:
        1. Client decrypts TFHE ciphertext
        2. Client dequantizes to real values
        3. Client encrypts with CKKS key

        Args:
            tfhe_result: TFHE ciphertext representation (post-bootstrap)

        Returns:
            Values suitable for CKKS encoding
        """
        values = np.array(tfhe_result['values'], dtype=np.float64)
        params = tfhe_result.get('params', {})

        # For bit output (gate), values are 0 or 1
        if params.get('bits', 8) == 1:
            return values  # Already in [0, 1]

        # For integer output, dequantize
        scale = params.get('scale', 1.0)
        return values / scale

    def apply_lut_simulation(
        self,
        tfhe_input: Dict[str, Any],
        lut: List[int],
    ) -> Dict[str, Any]:
        """
        Simulate TFHE LUT application.

        In real TFHE, this is done via programmable bootstrapping.
        This simulation applies the LUT directly to the plaintext.

        Args:
            tfhe_input: TFHE ciphertext representation
            lut: Lookup table entries

        Returns:
            TFHE ciphertext with LUT applied
        """
        values = np.array(tfhe_input['values'])
        lut_arr = np.array(lut)

        # Handle signed values for LUT indexing
        params = tfhe_input.get('params', {})
        if params.get('symmetric', True):
            # Map signed to unsigned for indexing
            offset = len(lut) // 2
            indices = (values + offset).astype(int)
        else:
            indices = values.astype(int)

        # Clamp indices
        indices = np.clip(indices, 0, len(lut) - 1)

        # Apply LUT
        result_values = lut_arr[indices]

        return {
            'type': 'tfhe_lwe',
            'values': result_values.tolist(),
            'params': {
                'bits': 1 if max(lut) <= 1 else params.get('bits', 8),
                'scale': 1.0,  # LUT output is discrete
                'symmetric': False,
            },
            'noise_budget': self.config.tfhe_noise_budget,  # Refreshed by bootstrap
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            'ckks_to_tfhe_count': 0,
            'tfhe_to_ckks_count': 0,
            'total_quantization_error': 0.0,
            'max_quantization_error': 0.0,
        }
