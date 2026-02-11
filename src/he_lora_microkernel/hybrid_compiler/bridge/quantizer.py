"""
Quantization Utilities for CKKS-TFHE Bridge

Provides various quantization schemes:
- Symmetric quantization (centered at 0)
- Asymmetric quantization (full range)
- Adaptive quantization (based on value distribution)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class QuantizationError(Exception):
    """Raised when quantization fails or exceeds error bounds."""
    pass


@dataclass
class Quantizer:
    """
    Quantizer for CKKS to discrete conversion.

    Supports multiple quantization modes:
    - symmetric: [-max, max] -> [-2^(k-1), 2^(k-1)-1]
    - asymmetric: [min, max] -> [0, 2^k-1]
    - adaptive: Automatically determines range from data
    """
    bits: int = 8
    symmetric: bool = True

    # Static range (used if adaptive=False)
    range_min: float = -10.0
    range_max: float = 10.0

    # Adaptive range
    adaptive: bool = False

    def __post_init__(self):
        if self.bits < 1 or self.bits > 16:
            raise ValueError(f"Bits must be in [1, 16], got {self.bits}")
        self._compute_params()

    def _compute_params(self):
        """Compute scale and zero point."""
        if self.symmetric:
            self.max_int = (1 << (self.bits - 1)) - 1
            max_abs = max(abs(self.range_min), abs(self.range_max))
            self.scale = max_abs / self.max_int if max_abs > 0 else 1.0
            self.zero_point = 0
        else:
            self.max_int = (1 << self.bits) - 1
            range_span = self.range_max - self.range_min
            self.scale = range_span / self.max_int if range_span > 0 else 1.0
            self.zero_point = round(-self.range_min / self.scale)

    def calibrate(self, data: np.ndarray) -> None:
        """
        Calibrate quantizer based on data distribution.

        Args:
            data: Sample data for calibration
        """
        if self.adaptive:
            self.range_min = float(np.min(data))
            self.range_max = float(np.max(data))
            self._compute_params()

    def quantize(self, value: float) -> int:
        """Quantize a single value."""
        if self.symmetric:
            clamped = max(-self.max_int * self.scale, min(self.max_int * self.scale, value))
            return int(round(clamped / self.scale))
        else:
            clamped = max(self.range_min, min(self.range_max, value))
            return int(round(clamped / self.scale + self.zero_point))

    def dequantize(self, value: int) -> float:
        """Dequantize a single value."""
        if self.symmetric:
            return value * self.scale
        else:
            return (value - self.zero_point) * self.scale

    def quantize_array(self, arr: np.ndarray) -> np.ndarray:
        """Quantize an array of values."""
        if self.symmetric:
            clamped = np.clip(arr, -self.max_int * self.scale, self.max_int * self.scale)
            return np.round(clamped / self.scale).astype(np.int32)
        else:
            clamped = np.clip(arr, self.range_min, self.range_max)
            return np.round(clamped / self.scale + self.zero_point).astype(np.int32)

    def dequantize_array(self, arr: np.ndarray) -> np.ndarray:
        """Dequantize an array of values."""
        if self.symmetric:
            return arr.astype(np.float64) * self.scale
        else:
            return (arr.astype(np.float64) - self.zero_point) * self.scale

    def compute_error(self, original: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute quantization error statistics.

        Args:
            original: Original values

        Returns:
            Tuple of (mean_error, max_error, rmse)
        """
        quantized = self.quantize_array(original)
        reconstructed = self.dequantize_array(quantized)
        errors = np.abs(original - reconstructed)

        return float(np.mean(errors)), float(np.max(errors)), float(np.sqrt(np.mean(errors**2)))


def dequantize(value: int, scale: float, zero_point: int = 0) -> float:
    """
    Dequantize a single integer value.

    Args:
        value: Quantized integer
        scale: Quantization scale
        zero_point: Zero point offset

    Returns:
        Dequantized float value
    """
    return (value - zero_point) * scale
