"""
Azure Key Vault KMS Provider.

Production-grade key management using Azure Key Vault.
"""

import base64
import logging
import time
from datetime import datetime
from typing import Optional

from ..provider import (
    KMSProvider,
    KMSError,
    KMSKeyNotFoundError,
    KMSAuthenticationError,
    KMSOperationError,
    KeyType,
    KeyAlgorithm,
    KeyMetadata,
)

logger = logging.getLogger(__name__)


class AzureKMSProvider(KMSProvider):
    """
    Azure Key Vault KMS Provider.

    Uses Azure Key Vault for key management and encryption operations.
    Supports both keys and secrets for different use cases.

    Requirements:
    - azure-keyvault-keys and azure-identity libraries installed
    - Azure credentials configured (DefaultAzureCredential or explicit credentials)
    - Appropriate Key Vault access policies
    """

    def __init__(
        self,
        vault_url: str,
        credential: Optional[object] = None,
        key_prefix: str = "tensafe",
    ):
        """
        Initialize Azure Key Vault provider.

        Args:
            vault_url: Key Vault URL (e.g., 'https://myvault.vault.azure.net/')
            credential: Optional Azure credential object. If None, uses DefaultAzureCredential
            key_prefix: Prefix for key names
        """
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.keys import KeyClient
            from azure.keyvault.keys.crypto import CryptographyClient
        except ImportError:
            raise KMSError(
                "azure-keyvault-keys and azure-identity are required for Azure KMS. "
                "Install with: pip install azure-keyvault-keys azure-identity"
            )

        self._vault_url = vault_url
        self._key_prefix = key_prefix

        if credential is None:
            credential = DefaultAzureCredential()

        self._credential = credential

        try:
            self._key_client = KeyClient(vault_url=vault_url, credential=credential)
            # Validate connectivity
            list(self._key_client.list_properties_of_keys(max_page_size=1))
        except Exception as e:
            if "authentication" in str(e).lower() or "401" in str(e):
                raise KMSAuthenticationError(f"Azure authentication failed: {e}")
            raise KMSOperationError(f"Azure Key Vault initialization failed: {e}")

        logger.info(f"Azure Key Vault provider initialized (vault: {vault_url})")

    @property
    def provider_name(self) -> str:
        return "azure"

    def _get_key_name(self, key_id: str) -> str:
        """Get the full key name in Azure."""
        return f"{self._key_prefix}-{key_id}"

    def _algorithm_to_azure(self, algorithm: KeyAlgorithm) -> tuple[str, Optional[str]]:
        """Convert KeyAlgorithm to Azure key type and curve."""
        from azure.keyvault.keys import KeyType as AzureKeyType

        if algorithm in (KeyAlgorithm.AES_256_GCM, KeyAlgorithm.AES_256_CBC):
            return AzureKeyType.oct, None
        elif algorithm == KeyAlgorithm.RSA_2048:
            return AzureKeyType.rsa, None
        elif algorithm == KeyAlgorithm.RSA_4096:
            return AzureKeyType.rsa, None
        elif algorithm == KeyAlgorithm.EC_P256:
            return AzureKeyType.ec, "P-256"
        elif algorithm == KeyAlgorithm.EC_P384:
            return AzureKeyType.ec, "P-384"
        return AzureKeyType.oct, None

    def _get_crypto_client(self, key_id: str):
        """Get a CryptographyClient for a key."""
        from azure.keyvault.keys.crypto import CryptographyClient

        key_name = self._get_key_name(key_id)
        key = self._key_client.get_key(key_name)
        return CryptographyClient(key, credential=self._credential)

    def generate_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.DEK,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM,
        tags: Optional[dict] = None,
    ) -> KeyMetadata:
        """Generate a new key in Azure Key Vault."""
        from azure.core.exceptions import ResourceExistsError

        key_name = self._get_key_name(key_id)
        azure_key_type, curve = self._algorithm_to_azure(algorithm)

        try:
            # Prepare key parameters
            params = {
                "name": key_name,
                "key_type": azure_key_type,
                "tags": tags or {},
            }

            # Add size for RSA keys
            if algorithm == KeyAlgorithm.RSA_2048:
                params["size"] = 2048
            elif algorithm == KeyAlgorithm.RSA_4096:
                params["size"] = 4096
            elif algorithm in (KeyAlgorithm.AES_256_GCM, KeyAlgorithm.AES_256_CBC):
                params["size"] = 256

            # Add curve for EC keys
            if curve:
                params["curve"] = curve

            # Create the key
            key = self._key_client.create_key(**params)

            logger.info(f"Created Azure Key Vault key: {key_id}")

            return KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                created_at=key.properties.created_on or datetime.utcnow(),
                is_active=key.properties.enabled,
                provider=self.provider_name,
                provider_key_ref=key.id,
                tags=tags or {},
            )

        except ResourceExistsError:
            raise KMSOperationError(f"Key already exists: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to create key: {e}")

    def get_key_material(self, key_id: str) -> bytes:
        """
        Azure Key Vault does not allow symmetric key export by default.
        """
        raise KMSOperationError(
            "Azure Key Vault does not support key export for HSM-protected keys. "
            "Use encrypt/decrypt methods directly, or use software-protected keys "
            "with backup/restore."
        )

    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[dict] = None) -> bytes:
        """Encrypt data using Azure Key Vault."""
        from azure.core.exceptions import ResourceNotFoundError
        from azure.keyvault.keys.crypto import EncryptionAlgorithm

        try:
            crypto_client = self._get_crypto_client(key_id)

            # Azure uses RSA-OAEP or AES-GCM based on key type
            # For symmetric keys, we need to wrap
            result = crypto_client.encrypt(EncryptionAlgorithm.a256_gcm, plaintext)

            # Combine IV/nonce with ciphertext and tag
            encrypted = result.iv + result.ciphertext + result.authentication_tag
            return encrypted

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            # Try RSA-OAEP for asymmetric keys
            try:
                from azure.keyvault.keys.crypto import EncryptionAlgorithm

                crypto_client = self._get_crypto_client(key_id)
                result = crypto_client.encrypt(EncryptionAlgorithm.rsa_oaep_256, plaintext)
                return result.ciphertext
            except Exception:
                raise KMSOperationError(f"Encryption failed: {e}")

    def decrypt(self, key_id: str, ciphertext: bytes, context: Optional[dict] = None) -> bytes:
        """Decrypt data using Azure Key Vault."""
        from azure.core.exceptions import ResourceNotFoundError
        from azure.keyvault.keys.crypto import EncryptionAlgorithm

        try:
            crypto_client = self._get_crypto_client(key_id)

            # For AES-GCM, split IV, ciphertext, and tag
            # IV is 12 bytes, tag is 16 bytes
            iv = ciphertext[:12]
            tag = ciphertext[-16:]
            encrypted = ciphertext[12:-16]

            result = crypto_client.decrypt(
                EncryptionAlgorithm.a256_gcm,
                encrypted,
                iv=iv,
                authentication_tag=tag,
            )
            return result.plaintext

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            # Try RSA-OAEP for asymmetric keys
            try:
                crypto_client = self._get_crypto_client(key_id)
                result = crypto_client.decrypt(EncryptionAlgorithm.rsa_oaep_256, ciphertext)
                return result.plaintext
            except Exception:
                raise KMSOperationError(f"Decryption failed: {e}")

    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key (creates new version in Azure)."""
        from azure.core.exceptions import ResourceNotFoundError

        key_name = self._get_key_name(key_id)

        try:
            # Get current key to determine type
            current_key = self._key_client.get_key(key_name)
            key_type = current_key.key_type

            # Create new version by creating key with same name
            params = {"name": key_name, "key_type": key_type}

            if key_type == "RSA":
                params["size"] = current_key.key.n.bit_length() if current_key.key.n else 2048
            elif key_type == "oct":
                params["size"] = 256

            new_key = self._key_client.create_key(**params)

            logger.info(f"Rotated Azure Key Vault key: {key_id}")

            return KeyMetadata(
                key_id=key_id,
                key_type=KeyType.DEK,
                algorithm=KeyAlgorithm.AES_256_GCM,
                created_at=new_key.properties.created_on or datetime.utcnow(),
                rotated_at=datetime.utcnow(),
                is_active=new_key.properties.enabled,
                provider=self.provider_name,
                provider_key_ref=new_key.id,
                tags={},
            )

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to rotate key: {e}")

    # Default timeout for Azure operations (seconds)
    DEFAULT_OPERATION_TIMEOUT = 60

    def delete_key(self, key_id: str, schedule_days: int = 30, timeout: int = None) -> None:
        """Delete a key from Azure Key Vault.

        Args:
            key_id: Key to delete
            schedule_days: Days before permanent deletion (Azure soft-delete)
            timeout: Operation timeout in seconds (default: 60)
        """
        from azure.core.exceptions import ResourceNotFoundError

        key_name = self._get_key_name(key_id)
        timeout = timeout or self.DEFAULT_OPERATION_TIMEOUT

        try:
            # Azure has soft-delete by default
            poller = self._key_client.begin_delete_key(key_name)
            # Use timeout to prevent indefinite blocking
            poller.wait(timeout=timeout)

            logger.info(f"Deleted Azure Key Vault key: {key_id}")

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            if "timeout" in str(e).lower():
                raise KMSOperationError(f"Delete key operation timed out after {timeout}s: {key_id}")
            raise KMSOperationError(f"Failed to delete key: {e}")

    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata from Azure Key Vault."""
        from azure.core.exceptions import ResourceNotFoundError

        key_name = self._get_key_name(key_id)

        try:
            key = self._key_client.get_key(key_name)

            return KeyMetadata(
                key_id=key_id,
                key_type=KeyType.DEK,
                algorithm=KeyAlgorithm.AES_256_GCM,
                created_at=key.properties.created_on or datetime.utcnow(),
                rotated_at=key.properties.updated_on,
                is_active=key.properties.enabled,
                provider=self.provider_name,
                provider_key_ref=key.id,
                tags=key.properties.tags or {},
            )

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to get key metadata: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> list[KeyMetadata]:
        """List keys from Azure Key Vault."""
        result = []

        try:
            keys = self._key_client.list_properties_of_keys()

            for key_props in keys:
                key_name = key_props.name
                if key_name.startswith(f"{self._key_prefix}-"):
                    key_id = key_name.replace(f"{self._key_prefix}-", "")
                    try:
                        metadata = self.get_key_metadata(key_id)
                        if key_type is None or metadata.key_type == key_type:
                            result.append(metadata)
                    except KMSKeyNotFoundError:
                        continue

        except Exception as e:
            raise KMSOperationError(f"Failed to list keys: {e}")

        return result

    def health_check(self) -> dict:
        """Check Azure Key Vault connectivity."""
        start = time.time()
        try:
            # Try to list keys
            list(self._key_client.list_properties_of_keys(max_page_size=1))
            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000,
                "provider": "azure",
                "vault_url": self._vault_url,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
                "provider": "azure",
            }
