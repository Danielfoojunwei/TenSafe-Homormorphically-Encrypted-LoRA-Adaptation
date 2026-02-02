"""
KMS Provider Base Interface.

Defines the abstract interface that all KMS providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class KMSError(Exception):
    """Base exception for KMS operations."""

    pass


class KMSKeyNotFoundError(KMSError):
    """Key not found in KMS."""

    pass


class KMSAuthenticationError(KMSError):
    """Authentication to KMS failed."""

    pass


class KMSOperationError(KMSError):
    """KMS operation failed."""

    pass


class KeyType(str, Enum):
    """Types of keys managed by KMS."""

    KEK = "kek"  # Key Encryption Key
    DEK = "dek"  # Data Encryption Key
    SIGNING = "signing"  # Signing key
    SYMMETRIC = "symmetric"  # Generic symmetric key


class KeyAlgorithm(str, Enum):
    """Supported key algorithms."""

    AES_256_GCM = "AES-256-GCM"
    AES_256_CBC = "AES-256-CBC"
    RSA_2048 = "RSA-2048"
    RSA_4096 = "RSA-4096"
    EC_P256 = "EC-P256"
    EC_P384 = "EC-P384"


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""

    key_id: str
    key_type: KeyType
    algorithm: KeyAlgorithm
    created_at: datetime
    rotated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    provider: str = ""
    provider_key_ref: str = ""
    tags: dict = field(default_factory=dict)


class KMSProvider(ABC):
    """
    Abstract base class for KMS providers.

    All KMS integrations (AWS, Vault, GCP, Azure, Local) must implement
    this interface to provide consistent key management operations.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'aws', 'vault', 'gcp', 'azure', 'local')."""
        pass

    @abstractmethod
    def generate_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.DEK,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM,
        tags: Optional[dict] = None,
    ) -> KeyMetadata:
        """
        Generate a new key in the KMS.

        Args:
            key_id: Unique identifier for the key
            key_type: Type of key to generate
            algorithm: Key algorithm
            tags: Optional metadata tags

        Returns:
            KeyMetadata for the generated key

        Raises:
            KMSOperationError: If key generation fails
        """
        pass

    @abstractmethod
    def get_key_material(self, key_id: str) -> bytes:
        """
        Retrieve raw key material (for local encryption operations).

        Note: Some KMS providers (AWS, GCP) don't allow key export.
        In those cases, use encrypt/decrypt directly on the KMS.

        Args:
            key_id: Key identifier

        Returns:
            Raw key bytes

        Raises:
            KMSKeyNotFoundError: If key doesn't exist
            KMSOperationError: If key export is not supported
        """
        pass

    @abstractmethod
    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[dict] = None) -> bytes:
        """
        Encrypt data using a KMS key.

        Args:
            key_id: Key identifier
            plaintext: Data to encrypt
            context: Optional encryption context (for AAD)

        Returns:
            Ciphertext bytes

        Raises:
            KMSKeyNotFoundError: If key doesn't exist
            KMSOperationError: If encryption fails
        """
        pass

    @abstractmethod
    def decrypt(self, key_id: str, ciphertext: bytes, context: Optional[dict] = None) -> bytes:
        """
        Decrypt data using a KMS key.

        Args:
            key_id: Key identifier
            ciphertext: Data to decrypt
            context: Optional encryption context (must match encryption)

        Returns:
            Plaintext bytes

        Raises:
            KMSKeyNotFoundError: If key doesn't exist
            KMSOperationError: If decryption fails
        """
        pass

    @abstractmethod
    def rotate_key(self, key_id: str) -> KeyMetadata:
        """
        Rotate a key to a new version.

        Args:
            key_id: Key identifier

        Returns:
            Updated KeyMetadata

        Raises:
            KMSKeyNotFoundError: If key doesn't exist
            KMSOperationError: If rotation fails
        """
        pass

    @abstractmethod
    def delete_key(self, key_id: str, schedule_days: int = 30) -> None:
        """
        Schedule a key for deletion.

        Args:
            key_id: Key identifier
            schedule_days: Days before permanent deletion (0 for immediate)

        Raises:
            KMSKeyNotFoundError: If key doesn't exist
            KMSOperationError: If deletion fails
        """
        pass

    @abstractmethod
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """
        Get metadata for a key.

        Args:
            key_id: Key identifier

        Returns:
            KeyMetadata

        Raises:
            KMSKeyNotFoundError: If key doesn't exist
        """
        pass

    @abstractmethod
    def list_keys(self, key_type: Optional[KeyType] = None) -> list[KeyMetadata]:
        """
        List all keys, optionally filtered by type.

        Args:
            key_type: Optional filter by key type

        Returns:
            List of KeyMetadata
        """
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """
        Check KMS provider health and connectivity.

        Returns:
            Dict with status, latency, and any warnings
        """
        pass

    def wrap_key(self, kek_id: str, key_to_wrap: bytes) -> bytes:
        """
        Wrap (encrypt) a key using another key.

        Default implementation uses encrypt(). Override for providers
        with native key wrapping support.

        Args:
            kek_id: Key encryption key ID
            key_to_wrap: Raw key bytes to wrap

        Returns:
            Wrapped key bytes
        """
        return self.encrypt(kek_id, key_to_wrap)

    def unwrap_key(self, kek_id: str, wrapped_key: bytes) -> bytes:
        """
        Unwrap (decrypt) a wrapped key.

        Default implementation uses decrypt(). Override for providers
        with native key unwrapping support.

        Args:
            kek_id: Key encryption key ID
            wrapped_key: Wrapped key bytes

        Returns:
            Unwrapped key bytes
        """
        return self.decrypt(kek_id, wrapped_key)
