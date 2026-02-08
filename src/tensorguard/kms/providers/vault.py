"""
HashiCorp Vault KMS Provider.

Production-grade key management using HashiCorp Vault Transit secrets engine.
"""

import base64
import logging
import time
from datetime import datetime
from typing import Optional

from ..provider import (
    KeyAlgorithm,
    KeyMetadata,
    KeyType,
    KMSAuthenticationError,
    KMSError,
    KMSKeyNotFoundError,
    KMSOperationError,
    KMSProvider,
)

logger = logging.getLogger(__name__)


class VaultKMSProvider(KMSProvider):
    """
    HashiCorp Vault KMS Provider.

    Uses Vault's Transit secrets engine for key management and encryption.
    Supports key versioning and rotation.

    Requirements:
    - hvac library installed
    - Vault server accessible
    - Transit secrets engine enabled
    - Valid authentication token or method
    """

    def __init__(
        self,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        mount_point: str = "transit",
        namespace: Optional[str] = None,
        key_prefix: str = "tensafe",
    ):
        """
        Initialize Vault KMS provider.

        Args:
            vault_addr: Vault server address (defaults to VAULT_ADDR env var)
            vault_token: Vault token (defaults to VAULT_TOKEN env var)
            mount_point: Transit secrets engine mount point
            namespace: Optional Vault namespace
            key_prefix: Prefix for key names
        """
        try:
            import hvac
        except ImportError:
            raise KMSError("hvac is required for Vault KMS. Install with: pip install hvac")

        import os

        vault_addr = vault_addr or os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
        vault_token = vault_token or os.getenv("VAULT_TOKEN")

        if not vault_token:
            raise KMSAuthenticationError("Vault token not provided")

        self._mount_point = mount_point
        self._key_prefix = key_prefix

        try:
            self._client = hvac.Client(url=vault_addr, token=vault_token, namespace=namespace)

            if not self._client.is_authenticated():
                raise KMSAuthenticationError("Vault authentication failed")

        except Exception as e:
            if "authentication" in str(e).lower():
                raise KMSAuthenticationError(f"Vault authentication failed: {e}")
            raise KMSOperationError(f"Vault initialization failed: {e}")

        logger.info(f"Vault KMS provider initialized (addr: {vault_addr})")

    @property
    def provider_name(self) -> str:
        return "vault"

    def _get_key_name(self, key_id: str) -> str:
        """Get the full key name in Vault."""
        return f"{self._key_prefix}-{key_id}"

    def _algorithm_to_vault_type(self, algorithm: KeyAlgorithm) -> str:
        """Convert KeyAlgorithm to Vault key type."""
        mapping = {
            KeyAlgorithm.AES_256_GCM: "aes256-gcm96",
            KeyAlgorithm.AES_256_CBC: "aes256-gcm96",  # Vault uses GCM
            KeyAlgorithm.RSA_2048: "rsa-2048",
            KeyAlgorithm.RSA_4096: "rsa-4096",
            KeyAlgorithm.EC_P256: "ecdsa-p256",
            KeyAlgorithm.EC_P384: "ecdsa-p384",
        }
        return mapping.get(algorithm, "aes256-gcm96")

    def generate_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.DEK,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM,
        tags: Optional[dict] = None,
    ) -> KeyMetadata:
        """Generate a new key in Vault Transit."""
        from hvac.exceptions import VaultError

        key_name = self._get_key_name(key_id)
        vault_type = self._algorithm_to_vault_type(algorithm)

        try:
            # Create the key
            self._client.secrets.transit.create_key(
                name=key_name,
                key_type=vault_type,
                exportable=False,  # Security: keys should not be exportable
                allow_plaintext_backup=False,
                mount_point=self._mount_point,
            )

            logger.info(f"Created Vault key: {key_id}")

            return KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                created_at=datetime.utcnow(),
                is_active=True,
                provider=self.provider_name,
                provider_key_ref=f"{self._mount_point}/keys/{key_name}",
                tags=tags or {},
            )

        except VaultError as e:
            raise KMSOperationError(f"Failed to create key: {e}")

    def get_key_material(self, key_id: str) -> bytes:
        """
        Vault Transit does not support key export by default.

        To enable, keys must be created with exportable=True (not recommended).
        Use encrypt/decrypt methods directly instead.
        """
        raise KMSOperationError(
            "Vault Transit keys are not exportable by default. "
            "Use encrypt/decrypt methods directly."
        )

    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[dict] = None) -> bytes:
        """Encrypt data using Vault Transit."""
        from hvac.exceptions import VaultError

        key_name = self._get_key_name(key_id)

        try:
            # Vault expects base64 encoded plaintext
            b64_plaintext = base64.b64encode(plaintext).decode()

            params = {"name": key_name, "plaintext": b64_plaintext, "mount_point": self._mount_point}

            if context:
                # Vault context must be base64 encoded
                import json

                context_bytes = json.dumps(context, sort_keys=True).encode()
                params["context"] = base64.b64encode(context_bytes).decode()

            response = self._client.secrets.transit.encrypt_data(**params)
            ciphertext = response["data"]["ciphertext"]

            # Return the ciphertext as bytes (it's already Vault-formatted)
            return ciphertext.encode()

        except VaultError as e:
            if "no such key" in str(e).lower():
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Encryption failed: {e}")

    def decrypt(self, key_id: str, ciphertext: bytes, context: Optional[dict] = None) -> bytes:
        """Decrypt data using Vault Transit."""
        from hvac.exceptions import VaultError

        key_name = self._get_key_name(key_id)

        try:
            # Ciphertext should be Vault-formatted string
            ciphertext_str = ciphertext.decode() if isinstance(ciphertext, bytes) else ciphertext

            params = {
                "name": key_name,
                "ciphertext": ciphertext_str,
                "mount_point": self._mount_point,
            }

            if context:
                import json

                context_bytes = json.dumps(context, sort_keys=True).encode()
                params["context"] = base64.b64encode(context_bytes).decode()

            response = self._client.secrets.transit.decrypt_data(**params)
            plaintext_b64 = response["data"]["plaintext"]

            return base64.b64decode(plaintext_b64)

        except VaultError as e:
            if "no such key" in str(e).lower():
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Decryption failed: {e}")

    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a Vault Transit key."""
        from hvac.exceptions import VaultError

        key_name = self._get_key_name(key_id)

        try:
            self._client.secrets.transit.rotate_key(name=key_name, mount_point=self._mount_point)
            logger.info(f"Rotated Vault key: {key_id}")
            return self.get_key_metadata(key_id)

        except VaultError as e:
            if "no such key" in str(e).lower():
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to rotate key: {e}")

    def delete_key(self, key_id: str, schedule_days: int = 30) -> None:
        """Delete a Vault Transit key."""
        from hvac.exceptions import VaultError

        key_name = self._get_key_name(key_id)

        try:
            # First, update key config to allow deletion
            self._client.secrets.transit.update_key_configuration(
                name=key_name, deletion_allowed=True, mount_point=self._mount_point
            )

            # Then delete
            self._client.secrets.transit.delete_key(name=key_name, mount_point=self._mount_point)
            logger.info(f"Deleted Vault key: {key_id}")

        except VaultError as e:
            if "no such key" in str(e).lower():
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to delete key: {e}")

    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata from Vault."""
        from hvac.exceptions import VaultError

        key_name = self._get_key_name(key_id)

        try:
            response = self._client.secrets.transit.read_key(
                name=key_name, mount_point=self._mount_point
            )
            data = response["data"]

            return KeyMetadata(
                key_id=key_id,
                key_type=KeyType.DEK,  # Vault doesn't store key type
                algorithm=KeyAlgorithm.AES_256_GCM,
                created_at=datetime.utcnow(),  # Vault doesn't provide creation time easily
                is_active=not data.get("deletion_allowed", False),
                provider=self.provider_name,
                provider_key_ref=f"{self._mount_point}/keys/{key_name}",
                tags={},
            )

        except VaultError as e:
            if "no such key" in str(e).lower() or "404" in str(e):
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to get key metadata: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> list[KeyMetadata]:
        """List keys from Vault Transit."""
        from hvac.exceptions import VaultError

        result = []

        try:
            response = self._client.secrets.transit.list_keys(mount_point=self._mount_point)
            keys = response.get("data", {}).get("keys", [])

            for key_name in keys:
                if key_name.startswith(f"{self._key_prefix}-"):
                    key_id = key_name.replace(f"{self._key_prefix}-", "")
                    try:
                        metadata = self.get_key_metadata(key_id)
                        if key_type is None or metadata.key_type == key_type:
                            result.append(metadata)
                    except KMSKeyNotFoundError:
                        continue

        except VaultError as e:
            raise KMSOperationError(f"Failed to list keys: {e}")

        return result

    def health_check(self) -> dict:
        """Check Vault connectivity."""
        start = time.time()
        try:
            health = self._client.sys.read_health_status(method="GET")
            return {
                "status": "healthy" if health.get("initialized") else "degraded",
                "latency_ms": (time.time() - start) * 1000,
                "provider": "vault",
                "sealed": health.get("sealed", False),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
                "provider": "vault",
            }
