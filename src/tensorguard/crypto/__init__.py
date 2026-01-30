"""
TensorGuard Unified Cryptography Layer
Implements Classic, PQC, and Hybrid primitives.
"""

from .kem import (
    decap_hybrid,
    encap_hybrid,
    generate_hybrid_keypair,
)
from .sig import (
    generate_hybrid_sig_keypair,
    sign_hybrid,
    verify_hybrid,
)

__all__ = [
    # Signature
    "generate_hybrid_sig_keypair",
    "sign_hybrid",
    "verify_hybrid",
    # KEM
    "generate_hybrid_keypair",
    "encap_hybrid",
    "decap_hybrid",
]
