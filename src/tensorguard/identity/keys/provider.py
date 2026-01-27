"""
TensorGuard Key Provider

Abstractions for key storage and retrieval in identity operations.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class KeyProvider(ABC):
    """Abstract base class for key providers."""

    @abstractmethod
    def get_private_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a private key by ID."""
        pass

    @abstractmethod
    def store_private_key(self, key_id: str, key_data: bytes) -> bool:
        """Store a private key."""
        pass

    @abstractmethod
    def get_certificate(self, cert_id: str) -> Optional[bytes]:
        """Retrieve a certificate by ID."""
        pass

    @abstractmethod
    def store_certificate(self, cert_id: str, cert_data: bytes) -> bool:
        """Store a certificate."""
        pass

    @abstractmethod
    def generate_key_pair(self, key_id: str, algorithm: str = "RSA2048") -> Tuple[bytes, bytes]:
        """Generate a new key pair and store it."""
        pass


class FileKeyProvider(KeyProvider):
    """
    File-based key provider for development and testing.

    WARNING: Not recommended for production use. Use a proper
    key management system (KMS) or HSM in production.
    """

    def __init__(self, base_path: str = "keys/identity"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._keys_dir = self.base_path / "private"
        self._certs_dir = self.base_path / "certs"
        self._keys_dir.mkdir(parents=True, exist_ok=True)
        self._certs_dir.mkdir(parents=True, exist_ok=True)

    def get_private_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a private key by ID."""
        key_path = self._keys_dir / f"{key_id}.key"
        if key_path.exists():
            return key_path.read_bytes()
        return None

    def store_private_key(self, key_id: str, key_data: bytes) -> bool:
        """Store a private key."""
        try:
            key_path = self._keys_dir / f"{key_id}.key"
            key_path.write_bytes(key_data)
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            return True
        except Exception as e:
            logger.error(f"Failed to store private key {key_id}: {e}")
            return False

    def get_certificate(self, cert_id: str) -> Optional[bytes]:
        """Retrieve a certificate by ID."""
        cert_path = self._certs_dir / f"{cert_id}.pem"
        if cert_path.exists():
            return cert_path.read_bytes()
        return None

    def store_certificate(self, cert_id: str, cert_data: bytes) -> bool:
        """Store a certificate."""
        try:
            cert_path = self._certs_dir / f"{cert_id}.pem"
            cert_path.write_bytes(cert_data)
            return True
        except Exception as e:
            logger.error(f"Failed to store certificate {cert_id}: {e}")
            return False

    def generate_key_pair(self, key_id: str, algorithm: str = "RSA2048") -> Tuple[bytes, bytes]:
        """
        Generate a new key pair and store it.

        Returns: (private_key_pem, public_key_pem)
        """
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa, ec
            from cryptography.hazmat.backends import default_backend

            if algorithm.startswith("RSA"):
                key_size = int(algorithm[3:]) if len(algorithm) > 3 else 2048
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )
            elif algorithm.startswith("EC"):
                # Default to P-256
                private_key = ec.generate_private_key(
                    ec.SECP256R1(),
                    backend=default_backend()
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            # Store the private key
            self.store_private_key(key_id, private_pem)

            return private_pem, public_pem

        except ImportError:
            logger.error("cryptography library required for key generation")
            raise
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            raise

    def list_keys(self) -> list:
        """List all stored key IDs."""
        return [p.stem for p in self._keys_dir.glob("*.key")]

    def list_certificates(self) -> list:
        """List all stored certificate IDs."""
        return [p.stem for p in self._certs_dir.glob("*.pem")]

    def delete_key(self, key_id: str) -> bool:
        """Delete a private key."""
        key_path = self._keys_dir / f"{key_id}.key"
        if key_path.exists():
            key_path.unlink()
            return True
        return False

    def delete_certificate(self, cert_id: str) -> bool:
        """Delete a certificate."""
        cert_path = self._certs_dir / f"{cert_id}.pem"
        if cert_path.exists():
            cert_path.unlink()
            return True
        return False
