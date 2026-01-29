"""
TG-Tinker artifact storage interface.

Provides a pluggable storage backend for encrypted artifacts.

Integration Points:
- Uses AES-256-GCM encryption with per-tenant DEKs
- Optional integration with tensorguard.identity.keys providers
- Optional integration with tensorguard.crypto.sig for signed artifacts
"""

import base64
import hashlib
import logging
import os
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .models import TinkerArtifact, generate_artifact_id

if TYPE_CHECKING:
    from tensorguard.identity.keys.provider import KeyProvider

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Write data to storage."""
        pass

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read data from storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data from storage."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists in storage."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "/tmp/tg_tinker_artifacts"):
        """Initialize local storage backend."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get full path for a storage key."""
        # Sanitize key to prevent path traversal
        safe_key = key.replace("..", "").replace("/", "_").replace("\\", "_")
        return self.base_path / safe_key

    def write(self, key: str, data: bytes) -> None:
        """Write data to local filesystem."""
        path = self._get_path(key)
        path.write_bytes(data)

    def read(self, key: str) -> bytes:
        """Read data from local filesystem."""
        path = self._get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        """Delete data from local filesystem."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        """Check if data exists."""
        return self._get_path(key).exists()


class EncryptedArtifactStore:
    """
    Encrypted artifact storage layer.

    Wraps a storage backend with AES-256-GCM encryption.
    Each artifact is encrypted with a unique nonce using the tenant's DEK.
    """

    NONCE_SIZE = 12  # 96 bits for AES-GCM
    TAG_SIZE = 16  # 128 bits authentication tag

    def __init__(
        self,
        backend: StorageBackend,
        key_manager: "KeyManager",
    ):
        """
        Initialize encrypted artifact store.

        Args:
            backend: Storage backend for encrypted data
            key_manager: Key manager for DEK retrieval
        """
        self.backend = backend
        self.key_manager = key_manager

    def save_artifact(
        self,
        data: bytes,
        tenant_id: str,
        training_client_id: str,
        artifact_type: str,
        metadata: Optional[dict] = None,
    ) -> TinkerArtifact:
        """
        Encrypt and save an artifact.

        Args:
            data: Raw artifact bytes
            tenant_id: Tenant ID for key retrieval
            training_client_id: Training client ID
            artifact_type: Type of artifact (checkpoint, weights, etc.)
            metadata: Optional custom metadata

        Returns:
            TinkerArtifact with metadata
        """
        # Generate artifact ID and storage key
        artifact_id = generate_artifact_id()
        storage_key = f"{tenant_id}/{training_client_id}/{artifact_id}"

        # Get tenant DEK
        dek, dek_id = self.key_manager.get_dek(tenant_id)

        # Generate unique nonce
        nonce = secrets.token_bytes(self.NONCE_SIZE)

        # Encrypt data with AAD
        aad = self._build_aad(artifact_id, tenant_id, training_client_id)
        aesgcm = AESGCM(dek)
        ciphertext = aesgcm.encrypt(nonce, data, aad)

        # Compute content hash of plaintext
        content_hash = f"sha256:{hashlib.sha256(data).hexdigest()}"

        # Store encrypted data
        self.backend.write(storage_key, ciphertext)

        # Create artifact record
        artifact = TinkerArtifact(
            id=artifact_id,
            training_client_id=training_client_id,
            tenant_id=tenant_id,
            artifact_type=artifact_type,
            storage_key=storage_key,
            size_bytes=len(data),
            encryption_algorithm="AES-256-GCM",
            encryption_key_id=dek_id,
            encryption_nonce=base64.b64encode(nonce).decode("ascii"),
            content_hash=content_hash,
            metadata_json=metadata or {},
            created_at=datetime.utcnow(),
        )

        return artifact

    def load_artifact(self, artifact: TinkerArtifact) -> bytes:
        """
        Load and decrypt an artifact.

        Args:
            artifact: Artifact metadata

        Returns:
            Decrypted artifact bytes

        Raises:
            ValueError: If decryption fails (tampering detected)
        """
        # Get tenant DEK
        dek, _ = self.key_manager.get_dek(artifact.tenant_id)

        # Read encrypted data
        ciphertext = self.backend.read(artifact.storage_key)

        # Decode nonce
        nonce = base64.b64decode(artifact.encryption_nonce)

        # Build AAD
        aad = self._build_aad(
            artifact.id,
            artifact.tenant_id,
            artifact.training_client_id,
        )

        # Decrypt data
        aesgcm = AESGCM(dek)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
        except Exception as e:
            raise ValueError(f"Artifact decryption failed (tampering?): {e}")

        # Verify content hash
        computed_hash = f"sha256:{hashlib.sha256(plaintext).hexdigest()}"
        if computed_hash != artifact.content_hash:
            raise ValueError(
                f"Content hash mismatch: expected {artifact.content_hash}, "
                f"got {computed_hash}"
            )

        return plaintext

    def delete_artifact(self, artifact: TinkerArtifact) -> None:
        """Delete an artifact from storage."""
        self.backend.delete(artifact.storage_key)

    def _build_aad(
        self,
        artifact_id: str,
        tenant_id: str,
        training_client_id: str,
    ) -> bytes:
        """Build additional authenticated data for encryption."""
        return f"{artifact_id}|{tenant_id}|{training_client_id}".encode("utf-8")


class KeyManager:
    """
    Key management for DEKs.

    In production, this would integrate with a key vault (HashiCorp Vault,
    AWS KMS, etc.). This implementation provides a local fallback.
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        key_store_path: Optional[str] = None,
    ):
        """
        Initialize key manager.

        Args:
            master_key: Master KEK (32 bytes). If None, generates one.
            key_store_path: Path to store wrapped DEKs. If None, uses memory.
        """
        if master_key is None:
            # In production, this should come from a secure vault
            self._master_key = secrets.token_bytes(32)
        else:
            if len(master_key) != 32:
                raise ValueError("Master key must be 32 bytes")
            self._master_key = master_key

        self._key_store_path = key_store_path
        self._dek_cache: dict[str, Tuple[bytes, str]] = {}

        # Load existing keys from store
        if key_store_path:
            self._load_keys()

    def get_dek(self, tenant_id: str) -> Tuple[bytes, str]:
        """
        Get or create DEK for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tuple of (DEK bytes, DEK ID)
        """
        if tenant_id in self._dek_cache:
            return self._dek_cache[tenant_id]

        # Generate new DEK
        dek = secrets.token_bytes(32)
        dek_id = f"dek-{secrets.token_hex(8)}"

        # Cache it
        self._dek_cache[tenant_id] = (dek, dek_id)

        # Persist if we have a store
        if self._key_store_path:
            self._save_keys()

        return dek, dek_id

    def rotate_dek(self, tenant_id: str) -> Tuple[bytes, str]:
        """
        Rotate DEK for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tuple of (new DEK bytes, new DEK ID)
        """
        # Generate new DEK
        dek = secrets.token_bytes(32)
        dek_id = f"dek-{secrets.token_hex(8)}"

        # Update cache
        self._dek_cache[tenant_id] = (dek, dek_id)

        # Persist
        if self._key_store_path:
            self._save_keys()

        return dek, dek_id

    def _wrap_key(self, dek: bytes) -> bytes:
        """Wrap a DEK with the master KEK."""
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._master_key)
        wrapped = aesgcm.encrypt(nonce, dek, None)
        return nonce + wrapped

    def _unwrap_key(self, wrapped: bytes) -> bytes:
        """Unwrap a DEK with the master KEK."""
        nonce = wrapped[:12]
        ciphertext = wrapped[12:]
        aesgcm = AESGCM(self._master_key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def _load_keys(self) -> None:
        """Load wrapped keys from store."""
        if not self._key_store_path:
            return
        path = Path(self._key_store_path)
        if not path.exists():
            return

        import json

        data = json.loads(path.read_text())
        for tenant_id, key_data in data.items():
            wrapped = base64.b64decode(key_data["wrapped"])
            dek = self._unwrap_key(wrapped)
            dek_id = key_data["id"]
            self._dek_cache[tenant_id] = (dek, dek_id)

    def _save_keys(self) -> None:
        """Save wrapped keys to store."""
        if not self._key_store_path:
            return

        import json

        data = {}
        for tenant_id, (dek, dek_id) in self._dek_cache.items():
            wrapped = self._wrap_key(dek)
            data[tenant_id] = {
                "wrapped": base64.b64encode(wrapped).decode("ascii"),
                "id": dek_id,
            }

        path = Path(self._key_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        os.chmod(path, 0o600)


class IdentityKeyManager(KeyManager):
    """
    Key manager that integrates with TensorGuard identity key providers.

    This allows TG-Tinker to use existing key management infrastructure:
    - FileKeyProvider for development
    - PKCS11KeyProvider for HSM integration
    - KMSKeyProvider for cloud KMS integration
    """

    def __init__(
        self,
        key_provider: "KeyProvider",
        master_key_id: str = "tg-tinker-master",
    ):
        """
        Initialize with an identity key provider.

        Args:
            key_provider: TensorGuard identity key provider
            master_key_id: ID for the master key in the provider
        """
        self._key_provider = key_provider
        self._master_key_id = master_key_id
        self._dek_cache: dict[str, Tuple[bytes, str]] = {}

        # Try to load master key from provider
        master_key = key_provider.get_private_key(master_key_id)
        if master_key is None:
            # Generate and store master key
            logger.info(f"Generating new master key: {master_key_id}")
            self._master_key = secrets.token_bytes(32)
            key_provider.store_private_key(master_key_id, self._master_key)
        else:
            # Use existing master key (first 32 bytes if PEM format)
            if master_key.startswith(b"-----"):
                # PEM format - extract raw key material
                self._master_key = hashlib.sha256(master_key).digest()
            elif len(master_key) == 32:
                self._master_key = master_key
            else:
                # Hash to get 32 bytes
                self._master_key = hashlib.sha256(master_key).digest()

    def get_dek(self, tenant_id: str) -> Tuple[bytes, str]:
        """Get or create DEK for a tenant using the key provider."""
        if tenant_id in self._dek_cache:
            return self._dek_cache[tenant_id]

        # Try to load from provider
        dek_key_id = f"tg-tinker-dek-{tenant_id}"
        wrapped_dek = self._key_provider.get_private_key(dek_key_id)

        if wrapped_dek is not None:
            # Unwrap existing DEK
            dek = self._unwrap_key(wrapped_dek)
            dek_id = dek_key_id
        else:
            # Generate new DEK
            dek = secrets.token_bytes(32)
            dek_id = f"dek-{secrets.token_hex(8)}"

            # Wrap and store
            wrapped = self._wrap_key(dek)
            self._key_provider.store_private_key(dek_key_id, wrapped)
            logger.info(f"Generated new DEK for tenant: {tenant_id}")

        self._dek_cache[tenant_id] = (dek, dek_id)
        return dek, dek_id


class SignedArtifactStore(EncryptedArtifactStore):
    """
    Encrypted artifact store with PQC signature support.

    Extends EncryptedArtifactStore to add hybrid signatures using
    Ed25519 + Dilithium3 for post-quantum security.
    """

    def __init__(
        self,
        backend: StorageBackend,
        key_manager: KeyManager,
        signing_key: Optional[dict] = None,
    ):
        """
        Initialize signed artifact store.

        Args:
            backend: Storage backend
            key_manager: Key manager for DEKs
            signing_key: Optional hybrid signing key for artifact signatures
        """
        super().__init__(backend, key_manager)
        self._signing_key = signing_key

    def save_artifact(
        self,
        data: bytes,
        tenant_id: str,
        training_client_id: str,
        artifact_type: str,
        metadata: Optional[dict] = None,
    ) -> TinkerArtifact:
        """
        Encrypt, sign, and save an artifact.

        If a signing key is configured, the artifact will be signed
        with a hybrid Ed25519 + Dilithium3 signature.
        """
        # Call parent to encrypt and save
        artifact = super().save_artifact(
            data, tenant_id, training_client_id, artifact_type, metadata
        )

        # Add signature if key is configured
        if self._signing_key is not None:
            try:
                from tensorguard.crypto.sig import sign_hybrid

                # Sign the content hash
                hash_bytes = artifact.content_hash.encode("utf-8")
                signature = sign_hybrid(self._signing_key, hash_bytes)

                # Add signature to metadata
                if artifact.metadata_json is None:
                    artifact.metadata_json = {}
                artifact.metadata_json["pqc_signature"] = signature
                logger.debug(f"Added PQC signature to artifact {artifact.id}")

            except ImportError:
                logger.warning("tensorguard.crypto.sig not available, skipping signature")
            except Exception as e:
                logger.error(f"Failed to sign artifact: {e}")

        return artifact

    def verify_artifact_signature(
        self,
        artifact: TinkerArtifact,
        public_key: dict,
    ) -> bool:
        """
        Verify the PQC signature on an artifact.

        Args:
            artifact: Artifact to verify
            public_key: Hybrid public key for verification

        Returns:
            True if signature is valid, False otherwise
        """
        if artifact.metadata_json is None:
            return False

        signature = artifact.metadata_json.get("pqc_signature")
        if signature is None:
            return False

        try:
            from tensorguard.crypto.sig import verify_hybrid

            hash_bytes = artifact.content_hash.encode("utf-8")
            return verify_hybrid(public_key, hash_bytes, signature)

        except ImportError:
            logger.warning("tensorguard.crypto.sig not available")
            return False
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


def create_artifact_store(
    use_identity_provider: bool = False,
    use_pqc_signatures: bool = False,
    signing_key: Optional[dict] = None,
    backend_path: str = "/tmp/tg_tinker_artifacts",
    master_key: Optional[bytes] = None,
    key_store_path: Optional[str] = None,
) -> EncryptedArtifactStore:
    """
    Factory function to create an artifact store with appropriate configuration.

    Args:
        use_identity_provider: Use TensorGuard identity key provider
        use_pqc_signatures: Enable PQC artifact signatures
        signing_key: Hybrid signing key for signatures
        backend_path: Path for local storage backend
        master_key: Master KEK (if not using identity provider)
        key_store_path: Path for key store (if not using identity provider)

    Returns:
        Configured EncryptedArtifactStore or SignedArtifactStore
    """
    # Create storage backend
    backend = LocalStorageBackend(backend_path)

    # Create key manager
    if use_identity_provider:
        try:
            from tensorguard.identity.keys.provider import FileKeyProvider

            key_provider = FileKeyProvider("keys/tg_tinker")
            key_manager = IdentityKeyManager(key_provider)
            logger.info("Using identity-based key management")
        except ImportError:
            logger.warning("Identity provider not available, falling back to local keys")
            key_manager = KeyManager(master_key, key_store_path)
    else:
        key_manager = KeyManager(master_key, key_store_path)

    # Create store
    if use_pqc_signatures and signing_key is not None:
        return SignedArtifactStore(backend, key_manager, signing_key)
    else:
        return EncryptedArtifactStore(backend, key_manager)
