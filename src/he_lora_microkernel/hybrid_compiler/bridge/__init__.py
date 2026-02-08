"""
CKKS-TFHE Bridge Layer

Provides conversion between CKKS and TFHE domains:
- Quantization: CKKS approximate -> discrete integers
- CKKS->TFHE: Re-encryption from CKKS to TFHE
- TFHE->CKKS: Re-encryption from TFHE to CKKS

Security: All conversions maintain encryption.
No plaintext exposure during scheme switching.
"""

from .bridge import (
    CKKSTFHEBridge,
    BridgeConfig,
    QuantizationParams,
)

from .quantizer import (
    Quantizer,
    QuantizationError,
    dequantize,
)

__all__ = [
    "CKKSTFHEBridge",
    "BridgeConfig",
    "QuantizationParams",
    "Quantizer",
    "QuantizationError",
    "dequantize",
]
