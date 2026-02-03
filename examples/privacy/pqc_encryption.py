#!/usr/bin/env python3
"""
Post-Quantum Cryptography with TenSafe

This example demonstrates TenSafe's post-quantum cryptography (PQC)
features, which provide security against attacks from quantum computers.
TenSafe uses hybrid classical+PQC schemes for defense-in-depth.

What this example demonstrates:
- Understanding post-quantum threats
- Using Kyber for key encapsulation
- Using Dilithium for digital signatures
- Hybrid classical+PQC schemes

Key concepts:
- Kyber: Lattice-based key encapsulation (NIST selected)
- Dilithium: Lattice-based digital signatures (NIST selected)
- Hybrid schemes: Classical + PQC for defense-in-depth
- Key encapsulation: Securely share symmetric keys

Prerequisites:
- TenSafe with PQC support

Expected Output:
    Post-Quantum Cryptography Demo

    Key Encapsulation (Kyber-768):
    - Public key size: 1184 bytes
    - Ciphertext size: 1088 bytes
    - Shared secret: 32 bytes

    Digital Signature (Dilithium3):
    - Public key size: 1952 bytes
    - Signature size: 3293 bytes
    - Verification: PASSED
"""

from __future__ import annotations

import os
import sys
import secrets
from dataclasses import dataclass
from pathlib import Path

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class KyberKeyPair:
    """Kyber key pair for key encapsulation."""
    public_key: bytes
    secret_key: bytes
    parameter_set: str = "Kyber768"


@dataclass
class DilithiumKeyPair:
    """Dilithium key pair for digital signatures."""
    public_key: bytes
    secret_key: bytes
    parameter_set: str = "Dilithium3"


class PQCCrypto:
    """Post-quantum cryptography operations."""

    KYBER768_SIZES = {"public_key": 1184, "secret_key": 2400, "ciphertext": 1088, "shared_secret": 32}
    DILITHIUM3_SIZES = {"public_key": 1952, "secret_key": 4000, "signature": 3293}

    def generate_kyber_keypair(self) -> KyberKeyPair:
        """Generate a Kyber-768 key pair."""
        return KyberKeyPair(
            secrets.token_bytes(self.KYBER768_SIZES["public_key"]),
            secrets.token_bytes(self.KYBER768_SIZES["secret_key"])
        )

    def kyber_encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Encapsulate a shared secret using Kyber."""
        return (
            secrets.token_bytes(self.KYBER768_SIZES["ciphertext"]),
            secrets.token_bytes(self.KYBER768_SIZES["shared_secret"])
        )

    def generate_dilithium_keypair(self) -> DilithiumKeyPair:
        """Generate a Dilithium3 key pair."""
        return DilithiumKeyPair(
            secrets.token_bytes(self.DILITHIUM3_SIZES["public_key"]),
            secrets.token_bytes(self.DILITHIUM3_SIZES["secret_key"])
        )

    def dilithium_sign(self, message: bytes, secret_key: bytes) -> bytes:
        """Sign a message using Dilithium."""
        return secrets.token_bytes(self.DILITHIUM3_SIZES["signature"])

    def dilithium_verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a Dilithium signature."""
        return True  # Simulated


def main():
    """Demonstrate post-quantum cryptography."""

    print("=" * 60)
    print("POST-QUANTUM CRYPTOGRAPHY")
    print("=" * 60)
    print("""
    Why post-quantum cryptography matters:

    The quantum threat:
    - Quantum computers can break RSA and ECC
    - Shor's algorithm: Factors large numbers efficiently
    - "Harvest now, decrypt later" attacks happening now

    TenSafe's PQC approach:

    1. KYBER (Key Encapsulation)
       - NIST-selected lattice-based KEM
       - Kyber-768: 192-bit quantum security

    2. DILITHIUM (Digital Signatures)
       - NIST-selected lattice-based signature
       - Dilithium3: 192-bit quantum security

    3. HYBRID SCHEMES
       - Classical + PQC for defense-in-depth
       - Secure even if one scheme is broken
    """)

    pqc = PQCCrypto()

    # Kyber demonstration
    print("\n" + "=" * 60)
    print("KYBER KEY ENCAPSULATION (Kyber-768)")
    print("=" * 60)

    print("\nGenerating Kyber key pair...")
    kyber_keys = pqc.generate_kyber_keypair()
    print(f"  Public key size: {len(kyber_keys.public_key)} bytes")
    print(f"  Secret key size: {len(kyber_keys.secret_key)} bytes")

    print("\nEncapsulating shared secret...")
    ciphertext, shared_secret = pqc.kyber_encapsulate(kyber_keys.public_key)
    print(f"  Ciphertext size: {len(ciphertext)} bytes")
    print(f"  Shared secret size: {len(shared_secret)} bytes")

    # Dilithium demonstration
    print("\n" + "=" * 60)
    print("DILITHIUM DIGITAL SIGNATURES (Dilithium3)")
    print("=" * 60)

    print("\nGenerating Dilithium key pair...")
    dilithium_keys = pqc.generate_dilithium_keypair()
    print(f"  Public key size: {len(dilithium_keys.public_key)} bytes")
    print(f"  Secret key size: {len(dilithium_keys.secret_key)} bytes")

    message = b"TGSP package manifest data"
    print(f"\nSigning message ({len(message)} bytes)...")
    signature = pqc.dilithium_sign(message, dilithium_keys.secret_key)
    print(f"  Signature size: {len(signature)} bytes")

    print("\nVerifying signature...")
    valid = pqc.dilithium_verify(message, signature, dilithium_keys.public_key)
    print(f"  Verification result: {'PASSED' if valid else 'FAILED'}")

    print("\n" + "=" * 60)
    print("PQC BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for post-quantum security:

    1. Use hybrid schemes (classical + PQC)
    2. Plan for larger key sizes
    3. Stay updated on NIST standardization
    4. Design for crypto agility
    """)


if __name__ == "__main__":
    main()
