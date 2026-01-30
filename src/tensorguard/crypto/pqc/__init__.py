"""
Post-Quantum Cryptography Module

Provides NIST-standardized post-quantum cryptographic primitives:
- ML-KEM-768 (Kyber-768): Key Encapsulation Mechanism (FIPS 203)
- ML-DSA-65 (Dilithium-3): Digital Signature Algorithm (FIPS 204)

Production Mode:
    Requires liboqs for real cryptographic implementations.
    Install with: pip install tensorguard[pqc]

SECURITY CONFIGURATION:
    TG_PQC_STRICT=true (default in production): Fail if liboqs not available

Usage:
    from tensorguard.crypto.pqc import Kyber768, Dilithium3, is_pqc_production_ready

    # Check if production PQC is available
    if is_pqc_production_ready():
        print("Using production PQC with liboqs")
    else:
        raise RuntimeError("PQC not available")

    # Key encapsulation
    kem = Kyber768()
    pk, sk = kem.keygen()
    shared_secret, ciphertext = kem.encap(pk)
    recovered = kem.decap(sk, ciphertext)

    # Digital signatures
    sig = Dilithium3()
    pk, sk = sig.keygen()
    signature = sig.sign(sk, b"message")
    is_valid = sig.verify(pk, b"message", signature)
"""

import logging
import os

logger = logging.getLogger(__name__)

from .agility import PostQuantumKEM, PostQuantumSig
from .dilithium import Dilithium3
from .dilithium import is_liboqs_available as _dilithium_available
from .kyber import Kyber768
from .kyber import is_liboqs_available as _kyber_available


class PQCSecurityError(RuntimeError):
    """Raised when PQC security requirements are not met."""
    pass


def is_pqc_production_ready() -> bool:
    """
    Check if production-grade PQC is available.

    Returns True if liboqs is installed and both Kyber and Dilithium
    can use the real implementations.
    """
    return _kyber_available() and _dilithium_available()


def enforce_pqc_strict_mode() -> None:
    """
    Enforce PQC strict mode for production environments.

    In production (TG_ENVIRONMENT=production), this will raise an error
    if liboqs is not available, preventing insecure simulator fallback.

    Raises:
        PQCSecurityError: If in production and PQC libraries are not available.
    """
    environment = os.getenv("TG_ENVIRONMENT", "development")
    strict_mode = os.getenv("TG_PQC_STRICT", "").lower()

    # Auto-enable strict mode in production if not explicitly set
    if strict_mode == "":
        strict_mode = "true" if environment == "production" else "false"

    if strict_mode == "true":
        if not is_pqc_production_ready():
            error_msg = (
                "SECURITY ERROR: PQC strict mode enabled but liboqs is not available. "
                "Install liboqs: pip install tensorguard[pqc]."
            )
            logger.critical(error_msg)
            raise PQCSecurityError(error_msg)
        logger.info("PQC strict mode: liboqs verified, using production cryptography")
    elif not is_pqc_production_ready():
        raise PQCSecurityError(
            "PQC libraries are required. Install liboqs: pip install tensorguard[pqc]."
        )


# Enforce on module import in production
_environment = os.getenv("TG_ENVIRONMENT", "development")
if _environment == "production":
    enforce_pqc_strict_mode()


__all__ = [
    "Kyber768",
    "Dilithium3",
    "PostQuantumKEM",
    "PostQuantumSig",
    "is_pqc_production_ready",
    "enforce_pqc_strict_mode",
    "PQCSecurityError",
]
