"""
HPKE v0.3 Implementation for TGSP

Hybrid Public Key Encryption (HPKE) utilities for TensorGuard Secure Package v0.3.
Implements RFC 9180 HPKE with X25519-ChaCha20Poly1305-SHA256 suite.

This module provides:
- hpke_seal: Encrypt data to a recipient's X25519 public key
- hpke_open: Decrypt data using recipient's X25519 private key
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# HPKE Suite Constants (RFC 9180)
HPKE_SUITE_ID = b"HPKE_KEM_X25519_AEAD_ChaCha20Poly1305_KDF_SHA256"
KEM_ID = 0x0020  # X25519
KDF_ID = 0x0001  # HKDF-SHA256
AEAD_ID = 0x0003  # ChaCha20Poly1305

NONCE_SIZE = 12
KEY_SIZE = 32
TAG_SIZE = 16


@dataclass
class HPKESealResult:
    """Result of HPKE seal operation."""
    enc: bytes  # Encapsulated key (ephemeral public key)
    ciphertext: bytes  # Encrypted data with authentication tag

    def to_dict(self) -> Dict[str, str]:
        """Serialize to hex-encoded dictionary for transport."""
        return {
            "enc": self.enc.hex(),
            "ciphertext": self.ciphertext.hex()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "HPKESealResult":
        """Deserialize from hex-encoded dictionary."""
        return cls(
            enc=bytes.fromhex(data["enc"]),
            ciphertext=bytes.fromhex(data["ciphertext"])
        )


def _labeled_extract(salt: bytes, label: bytes, ikm: bytes, suite_id: bytes) -> bytes:
    """HPKE LabeledExtract (RFC 9180 Section 4)."""
    labeled_ikm = b"HPKE-v1" + suite_id + label + ikm
    h = hashlib.sha256()
    # HMAC-based extract using HKDF
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt if salt else b"\x00" * 32,
        info=labeled_ikm,
        backend=default_backend()
    )
    return hkdf.derive(ikm)


def _labeled_expand(prk: bytes, label: bytes, info: bytes, length: int, suite_id: bytes) -> bytes:
    """HPKE LabeledExpand (RFC 9180 Section 4)."""
    labeled_info = length.to_bytes(2, 'big') + b"HPKE-v1" + suite_id + label + info
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=prk,
        info=labeled_info,
        backend=default_backend()
    )
    return hkdf.derive(b"\x00" * length)


def _derive_key_and_nonce(shared_secret: bytes, info: bytes = b"") -> Tuple[bytes, bytes]:
    """Derive encryption key and base nonce from shared secret."""
    # Key schedule
    ks_context = b"\x00" + info  # mode_base (0x00) + info

    secret = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=None,
        info=b"secret" + ks_context,
        backend=default_backend()
    ).derive(shared_secret)

    key = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=None,
        info=b"key" + secret,
        backend=default_backend()
    ).derive(secret)

    base_nonce = HKDF(
        algorithm=hashes.SHA256(),
        length=NONCE_SIZE,
        salt=None,
        info=b"base_nonce" + secret,
        backend=default_backend()
    ).derive(secret)

    return key, base_nonce


def hpke_seal(
    plaintext: bytes,
    recipient_public_key: x25519.X25519PublicKey,
    info: bytes = b"",
    aad: bytes = b""
) -> Dict[str, str]:
    """
    HPKE Seal: Encrypt plaintext to recipient's public key.

    Uses X25519 for key encapsulation and ChaCha20-Poly1305 for AEAD.

    Args:
        plaintext: Data to encrypt
        recipient_public_key: X25519 public key of recipient
        info: Optional context info for key derivation
        aad: Additional authenticated data (not encrypted, but authenticated)

    Returns:
        Dictionary with 'enc' (encapsulated key) and 'ciphertext' (encrypted data)
        Both values are hex-encoded strings.
    """
    # Generate ephemeral keypair
    ephemeral_private = x25519.X25519PrivateKey.generate()
    ephemeral_public = ephemeral_private.public_key()

    # Encapsulated key is the ephemeral public key
    enc = ephemeral_public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    # Compute shared secret via ECDH
    shared_secret = ephemeral_private.exchange(recipient_public_key)

    # Derive key and nonce
    key, base_nonce = _derive_key_and_nonce(shared_secret, info)

    # Encrypt with ChaCha20-Poly1305
    chacha = ChaCha20Poly1305(key)
    ciphertext = chacha.encrypt(base_nonce, plaintext, aad)

    result = HPKESealResult(enc=enc, ciphertext=ciphertext)
    return result.to_dict()


def hpke_open(
    sealed: Dict[str, str],
    recipient_private_key: x25519.X25519PrivateKey,
    info: bytes = b"",
    aad: bytes = b""
) -> bytes:
    """
    HPKE Open: Decrypt ciphertext using recipient's private key.

    Args:
        sealed: Dictionary with 'enc' and 'ciphertext' (hex-encoded)
        recipient_private_key: X25519 private key of recipient
        info: Optional context info for key derivation (must match seal)
        aad: Additional authenticated data (must match seal)

    Returns:
        Decrypted plaintext bytes

    Raises:
        ValueError: If decryption fails (invalid ciphertext or wrong key)
    """
    result = HPKESealResult.from_dict(sealed)

    # Reconstruct sender's ephemeral public key
    ephemeral_public = x25519.X25519PublicKey.from_public_bytes(result.enc)

    # Compute shared secret via ECDH
    shared_secret = recipient_private_key.exchange(ephemeral_public)

    # Derive key and nonce
    key, base_nonce = _derive_key_and_nonce(shared_secret, info)

    # Decrypt with ChaCha20-Poly1305
    chacha = ChaCha20Poly1305(key)
    try:
        plaintext = chacha.decrypt(base_nonce, result.ciphertext, aad)
        return plaintext
    except Exception as e:
        raise ValueError(f"HPKE decryption failed: {e}")


def generate_keypair() -> Tuple[x25519.X25519PrivateKey, x25519.X25519PublicKey]:
    """
    Generate a new X25519 keypair for HPKE.

    Returns:
        Tuple of (private_key, public_key)
    """
    private_key = x25519.X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def public_key_to_bytes(public_key: x25519.X25519PublicKey) -> bytes:
    """Serialize X25519 public key to raw bytes."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )


def private_key_to_bytes(private_key: x25519.X25519PrivateKey) -> bytes:
    """Serialize X25519 private key to raw bytes."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
