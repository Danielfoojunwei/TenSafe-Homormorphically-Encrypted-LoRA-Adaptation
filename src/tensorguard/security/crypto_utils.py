"""
Cryptographic Utilities Module.

Provides secure cryptographic primitives:
- Constant-time comparison
- Secure hashing
- Secure token generation
- Key derivation
"""

import hashlib
import hmac
import os
import secrets
from typing import Optional, Union


def constant_time_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """
    Compare two values in constant time.

    This prevents timing attacks by ensuring the comparison takes
    the same amount of time regardless of where the values differ.

    Args:
        a: First value
        b: Second value

    Returns:
        True if values are equal, False otherwise
    """
    # Convert strings to bytes
    if isinstance(a, str):
        a = a.encode("utf-8")
    if isinstance(b, str):
        b = b.encode("utf-8")

    # Use hmac.compare_digest for constant-time comparison
    return hmac.compare_digest(a, b)


def secure_hash(
    data: Union[str, bytes],
    algorithm: str = "sha256",
    salt: Optional[bytes] = None,
) -> str:
    """
    Compute a secure hash of data.

    Args:
        data: Data to hash
        algorithm: Hash algorithm (sha256, sha384, sha512, blake2b)
        salt: Optional salt to prepend

    Returns:
        Hex-encoded hash
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if salt:
        data = salt + data

    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha384":
        return hashlib.sha384(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    elif algorithm == "blake2b":
        return hashlib.blake2b(data).hexdigest()
    elif algorithm == "blake2s":
        return hashlib.blake2s(data).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def generate_secure_token(
    length: int = 32,
    encoding: str = "hex",
) -> str:
    """
    Generate a cryptographically secure token.

    Args:
        length: Token length in bytes (before encoding)
        encoding: Output encoding (hex, base64, urlsafe)

    Returns:
        Encoded token string
    """
    if encoding == "hex":
        return secrets.token_hex(length)
    elif encoding == "base64":
        import base64

        return base64.b64encode(secrets.token_bytes(length)).decode("ascii")
    elif encoding == "urlsafe":
        return secrets.token_urlsafe(length)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def derive_key(
    password: Union[str, bytes],
    salt: bytes,
    length: int = 32,
    iterations: int = 100000,
    algorithm: str = "sha256",
) -> bytes:
    """
    Derive a key from a password using PBKDF2.

    Args:
        password: Password or passphrase
        salt: Random salt (at least 16 bytes recommended)
        length: Derived key length in bytes
        iterations: PBKDF2 iterations (100000+ recommended)
        algorithm: Hash algorithm

    Returns:
        Derived key bytes
    """
    if isinstance(password, str):
        password = password.encode("utf-8")

    return hashlib.pbkdf2_hmac(
        algorithm,
        password,
        salt,
        iterations,
        dklen=length,
    )


def generate_salt(length: int = 16) -> bytes:
    """
    Generate a random salt.

    Args:
        length: Salt length in bytes

    Returns:
        Random salt bytes
    """
    return secrets.token_bytes(length)


def hash_password(
    password: str,
    salt: Optional[bytes] = None,
    iterations: int = 100000,
) -> tuple[str, str]:
    """
    Hash a password using PBKDF2.

    Note: For production use, prefer Argon2id via passlib.
    This function is provided for cases where Argon2 is not available.

    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
        iterations: PBKDF2 iterations

    Returns:
        Tuple of (hash_hex, salt_hex)
    """
    if salt is None:
        salt = generate_salt(16)

    key = derive_key(password, salt, length=32, iterations=iterations)

    return key.hex(), salt.hex()


def verify_password_hash(
    password: str,
    hash_hex: str,
    salt_hex: str,
    iterations: int = 100000,
) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Password to verify
        hash_hex: Stored hash (hex encoded)
        salt_hex: Stored salt (hex encoded)
        iterations: PBKDF2 iterations

    Returns:
        True if password matches
    """
    salt = bytes.fromhex(salt_hex)
    expected_hash = bytes.fromhex(hash_hex)

    computed_hash = derive_key(password, salt, length=32, iterations=iterations)

    return constant_time_compare(computed_hash, expected_hash)


def secure_random_string(
    length: int,
    charset: str = "alphanumeric",
) -> str:
    """
    Generate a cryptographically secure random string.

    Args:
        length: String length
        charset: Character set (alphanumeric, alpha, numeric, hex, base64)

    Returns:
        Random string
    """
    if charset == "alphanumeric":
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    elif charset == "alpha":
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    elif charset == "numeric":
        chars = "0123456789"
    elif charset == "hex":
        chars = "0123456789abcdef"
    elif charset == "base64":
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    else:
        chars = charset  # Use custom charset

    return "".join(secrets.choice(chars) for _ in range(length))


def hmac_sign(
    key: bytes,
    data: Union[str, bytes],
    algorithm: str = "sha256",
) -> str:
    """
    Create an HMAC signature.

    Args:
        key: HMAC key
        data: Data to sign
        algorithm: Hash algorithm

    Returns:
        Hex-encoded HMAC
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if algorithm == "sha256":
        h = hmac.new(key, data, hashlib.sha256)
    elif algorithm == "sha384":
        h = hmac.new(key, data, hashlib.sha384)
    elif algorithm == "sha512":
        h = hmac.new(key, data, hashlib.sha512)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return h.hexdigest()


def hmac_verify(
    key: bytes,
    data: Union[str, bytes],
    signature: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify an HMAC signature.

    Args:
        key: HMAC key
        data: Signed data
        signature: Expected signature (hex)
        algorithm: Hash algorithm

    Returns:
        True if signature is valid
    """
    expected = hmac_sign(key, data, algorithm)
    return constant_time_compare(expected, signature)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    XOR two byte strings.

    Args:
        a: First byte string
        b: Second byte string (must be same length)

    Returns:
        XOR result
    """
    if len(a) != len(b):
        raise ValueError("Byte strings must have the same length")

    return bytes(x ^ y for x, y in zip(a, b))


def pad_pkcs7(data: bytes, block_size: int = 16) -> bytes:
    """
    Apply PKCS#7 padding.

    Args:
        data: Data to pad
        block_size: Block size in bytes

    Returns:
        Padded data
    """
    padding_len = block_size - (len(data) % block_size)
    return data + bytes([padding_len] * padding_len)


def unpad_pkcs7(data: bytes) -> bytes:
    """
    Remove PKCS#7 padding.

    Args:
        data: Padded data

    Returns:
        Unpadded data

    Raises:
        ValueError: If padding is invalid
    """
    if not data:
        raise ValueError("Empty data")

    padding_len = data[-1]

    if padding_len == 0 or padding_len > len(data):
        raise ValueError("Invalid padding")

    # Verify all padding bytes
    for i in range(padding_len):
        if data[-(i + 1)] != padding_len:
            raise ValueError("Invalid padding")

    return data[:-padding_len]


class SecureCompare:
    """
    Context for secure comparisons with timing attack protection.

    Usage:
        with SecureCompare() as sc:
            if sc.equals(user_input, stored_value):
                # Values match
    """

    def __init__(self):
        """Initialize secure compare context."""
        self._comparisons = 0

    def __enter__(self) -> "SecureCompare":
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        pass

    def equals(self, a: Union[str, bytes], b: Union[str, bytes]) -> bool:
        """
        Compare two values securely.

        Args:
            a: First value
            b: Second value

        Returns:
            True if equal
        """
        self._comparisons += 1
        return constant_time_compare(a, b)

    @property
    def comparison_count(self) -> int:
        """Get number of comparisons performed."""
        return self._comparisons
