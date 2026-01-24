"""
MOAI Configuration Modules
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class PackingStrategy(Enum):
    """Strategy for packing tensor values into FHE ciphertexts."""
    SIMD_DIAGONAL = "simd_diagonal"  # Standard CKKS/BFV rotation-friendly packing
    BLOCK_CIRCULANT = "block_circulant"  # Optimized for matrix-vector multiplication
    COEFF_PACKING = "coeff_packing"  # Higher density but harder ops

@dataclass
class MoaiConfig:
    """
    Configuration for MOAI FHE Inference.
    
    Attributes:
        poly_modulus_degree (int): Ring dimension N (e.g., 8192, 16384).
        coeff_modulus_bit_sizes (List[int]): Bit sizes for modulus chain (q).
        scale (float): Scaling factor for CKKS fixed-point arithmetic (2^40).
        precision_bits (int): Quantization precision for model weights.
        packing_strategy (PackingStrategy): How weights are packed.
        max_batch_size (int): Maximum concurrent inference requests per ciphertext.
    """
    # Cryptographic Parameters (CKKS)
    poly_modulus_degree: int = 8192
    coeff_modulus_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale: float = 2.0**40
    
    # Model Optimization
    precision_bits: int = 8
    packing_strategy: PackingStrategy = PackingStrategy.SIMD_DIAGONAL
    
    # Runtime Constraints
    max_batch_size: int = 1
    security_level: int = 128
    
    def validate(self) -> bool:
        """Validate configuration constraints."""
        if self.poly_modulus_degree not in [4096, 8192, 16384, 32768]:
            return False
        if not self.coeff_modulus_bit_sizes:
            return False
        return True
