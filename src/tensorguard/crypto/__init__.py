"""
TensorGuard Unified Cryptography Layer
Implements Classic, PQC, and Hybrid primitives.
"""

from .sig import (
    generate_hybrid_sig_keypair,
    sign_hybrid,
    verify_hybrid,
)
from .kem import (
    generate_hybrid_keypair,
    encap_hybrid,
    decap_hybrid,
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
