"""
Local KMS Provider.

File-based key storage for development and testing.
NOT FOR PRODUCTION USE - keys are stored encrypted but locally.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..provider import (
    KMSProvider,
    KMSError,
    KMSKeyNotFoundError,
    KMSOperationError,
    KeyType,
    KeyAlgorithm,
    KeyMetadata,
)

logger = logging.getLogger(__name__)


class LocalKMSProvider(KMSProvider):
    """
    Local file-based KMS provider.

    This provider stores keys encrypted on the local filesystem.
    Intended ONLY for development and testing - use a real KMS in production.

    Keys are encrypted with a master key derived from an environment variable
    or a local master key file.
    """

    MASTER_KEY_ENV = "TENSAFE_LOCAL_KMS_MASTER_KEY"
    NONCE_SIZE = 12

    def __init__(
        self,
        storage_path: str = "/tmp/tensafe_kms",
        master_key: Optional[bytes] = None,
    ):
        """
        Initialize local KMS provider.

        Args:
            storage_path: Directory for key storage
            master_key: Optional master key (32 bytes). If not provided,
                        reads from environment or generates one.
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._keys_file = self._storage_path / "keys.json.enc"
        self._master_key_file = self._storage_path / ".master_key"

        # Initialize master key
        self._master_key = self._init_master_key(master_key)

        # Load existing keys
        self._keys: dict[str, dict] = self._load_keys()

        logger.warning(
            "LocalKMSProvider initialized - NOT FOR PRODUCTION USE. "
            f"Keys stored at: {self._storage_path}"
        )

    @property
    def provider_name(self) -> str:
        return "local"

    def _init_master_key(self, provided_key: Optional[bytes]) -> bytes:
        """Initialize or load the master key."""
        if provided_key:
            if len(provided_key) != 32:
                raise ValueError("Master key must be 32 bytes")
            return provided_key

        # Try environment variable
        env_key = os.getenv(self.MASTER_KEY_ENV)
        if env_key:
            return base64.b64decode(env_key)

        # Try to load from file
        if self._master_key_file.exists():
            return base64.b64decode(self._master_key_file.read_text().strip())

        # Generate new master key
        master_key = secrets.token_bytes(32)
        self._master_key_file.write_text(base64.b64encode(master_key).decode())
        os.chmod(self._master_key_file, 0o600)
        logger.info("Generated new local KMS master key")
        return master_key

    def _encrypt_storage(self, data: bytes) -> bytes:
        """Encrypt data for storage."""
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        aesgcm = AESGCM(self._master_key)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    def _decrypt_storage(self, data: bytes) -> bytes:
        """Decrypt stored data."""
        nonce = data[: self.NONCE_SIZE]
        ciphertext = data[self.NONCE_SIZE :]
        aesgcm = AESGCM(self._master_key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def _load_keys(self) -> dict:
        """Load keys from encrypted storage."""
        if not self._keys_file.exists():
            return {}

        try:
            encrypted = self._keys_file.read_bytes()
            decrypted = self._decrypt_storage(encrypted)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            return {}

    def _save_keys(self) -> None:
        """Save keys to encrypted storage."""
        data = json.dumps(self._keys, indent=2).encode()
        encrypted = self._encrypt_storage(data)
        self._keys_file.write_bytes(encrypted)
        os.chmod(self._keys_file, 0o600)

    def generate_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.DEK,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM,
        tags: Optional[dict] = None,
    ) -> KeyMetadata:
        """Generate a new key."""
        if key_id in self._keys:
            raise KMSOperationError(f"Key already exists: {key_id}")

        # Generate key material based on algorithm
        if algorithm in (KeyAlgorithm.AES_256_GCM, KeyAlgorithm.AES_256_CBC):
            key_material = secrets.token_bytes(32)
        else:
            raise KMSOperationError(f"Unsupported algorithm: {algorithm}")

        now = datetime.utcnow()
        key_data = {
            "key_id": key_id,
            "key_type": key_type.value,
            "algorithm": algorithm.value,
            "key_material": base64.b64encode(key_material).decode(),
            "created_at": now.isoformat(),
            "rotated_at": None,
            "is_active": True,
            "tags": tags or {},
        }

        self._keys[key_id] = key_data
        self._save_keys()

        logger.info(f"Generated key: {key_id} ({key_type.value})")

        return KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            created_at=now,
            is_active=True,
            provider=self.provider_name,
            provider_key_ref=str(self._keys_file),
            tags=tags or {},
        )

    def get_key_material(self, key_id: str) -> bytes:
        """Get raw key material."""
        if key_id not in self._keys:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")

        key_data = self._keys[key_id]
        if not key_data.get("is_active", True):
            raise KMSOperationError(f"Key is not active: {key_id}")

        return base64.b64decode(key_data["key_material"])

    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[dict] = None) -> bytes:
        """Encrypt data using a key."""
        key_material = self.get_key_material(key_id)

        # Build AAD from context
        aad = None
        if context:
            aad = json.dumps(context, sort_keys=True).encode()

        nonce = secrets.token_bytes(self.NONCE_SIZE)
        aesgcm = AESGCM(key_material)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

        # Return nonce + ciphertext
        return nonce + ciphertext

    def decrypt(self, key_id: str, ciphertext: bytes, context: Optional[dict] = None) -> bytes:
        """Decrypt data using a key."""
        key_material = self.get_key_material(key_id)

        # Build AAD from context
        aad = None
        if context:
            aad = json.dumps(context, sort_keys=True).encode()

        nonce = ciphertext[: self.NONCE_SIZE]
        encrypted = ciphertext[self.NONCE_SIZE :]

        aesgcm = AESGCM(key_material)
        try:
            return aesgcm.decrypt(nonce, encrypted, aad)
        except Exception as e:
            raise KMSOperationError(f"Decryption failed: {e}")

    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key to new material."""
        if key_id not in self._keys:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")

        key_data = self._keys[key_id]
        algorithm = KeyAlgorithm(key_data["algorithm"])

        # Generate new key material
        if algorithm in (KeyAlgorithm.AES_256_GCM, KeyAlgorithm.AES_256_CBC):
            new_material = secrets.token_bytes(32)
        else:
            raise KMSOperationError(f"Unsupported algorithm: {algorithm}")

        # Store old key for potential rollback
        old_material = key_data["key_material"]
        key_data["previous_key_material"] = old_material
        key_data["key_material"] = base64.b64encode(new_material).decode()
        key_data["rotated_at"] = datetime.utcnow().isoformat()

        self._save_keys()
        logger.info(f"Rotated key: {key_id}")

        return self.get_key_metadata(key_id)

    def delete_key(self, key_id: str, schedule_days: int = 30) -> None:
        """Delete a key (immediate in local provider)."""
        if key_id not in self._keys:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")

        if schedule_days == 0:
            # Immediate deletion
            del self._keys[key_id]
        else:
            # Mark for deletion
            self._keys[key_id]["is_active"] = False
            self._keys[key_id]["scheduled_deletion"] = datetime.utcnow().isoformat()

        self._save_keys()
        logger.info(f"Deleted key: {key_id}")

    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata."""
        if key_id not in self._keys:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")

        key_data = self._keys[key_id]
        return KeyMetadata(
            key_id=key_id,
            key_type=KeyType(key_data["key_type"]),
            algorithm=KeyAlgorithm(key_data["algorithm"]),
            created_at=datetime.fromisoformat(key_data["created_at"]),
            rotated_at=(
                datetime.fromisoformat(key_data["rotated_at"]) if key_data.get("rotated_at") else None
            ),
            is_active=key_data.get("is_active", True),
            provider=self.provider_name,
            provider_key_ref=str(self._keys_file),
            tags=key_data.get("tags", {}),
        )

    def list_keys(self, key_type: Optional[KeyType] = None) -> list[KeyMetadata]:
        """List all keys."""
        result = []
        for key_id in self._keys:
            try:
                metadata = self.get_key_metadata(key_id)
                if key_type is None or metadata.key_type == key_type:
                    result.append(metadata)
            except Exception:
                continue
        return result

    def health_check(self) -> dict:
        """Check provider health."""
        start = time.time()
        try:
            # Try to read/write
            test_data = b"health_check_test"
            encrypted = self._encrypt_storage(test_data)
            decrypted = self._decrypt_storage(encrypted)
            success = decrypted == test_data
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
            }

        return {
            "status": "healthy" if success else "unhealthy",
            "latency_ms": (time.time() - start) * 1000,
            "key_count": len(self._keys),
            "storage_path": str(self._storage_path),
            "warning": "Local KMS is not production-grade",
        }
