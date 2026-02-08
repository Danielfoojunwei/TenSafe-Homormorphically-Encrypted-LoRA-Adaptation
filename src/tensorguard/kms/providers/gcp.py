"""
Google Cloud KMS Provider.

Production-grade key management using Google Cloud Key Management Service.
"""

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


class GCPKMSProvider(KMSProvider):
    """
    Google Cloud KMS Provider.

    Uses Google Cloud KMS for key management and encryption operations.
    Keys are never exported - all crypto operations happen within KMS.

    Requirements:
    - google-cloud-kms library installed
    - GCP credentials configured (GOOGLE_APPLICATION_CREDENTIALS or default credentials)
    - Appropriate IAM permissions for Cloud KMS
    """

    def __init__(
        self,
        project_id: str,
        location: str = "global",
        key_ring: str = "tensafe",
    ):
        """
        Initialize GCP KMS provider.

        Args:
            project_id: GCP project ID
            location: KMS location (e.g., 'global', 'us-east1')
            key_ring: Key ring name (will be created if doesn't exist)
        """
        try:
            from google.cloud import kms
        except ImportError:
            raise KMSError(
                "google-cloud-kms is required for GCP KMS. "
                "Install with: pip install google-cloud-kms"
            )

        self._project_id = project_id
        self._location = location
        self._key_ring = key_ring

        try:
            self._client = kms.KeyManagementServiceClient()

            # Build the key ring path
            self._key_ring_path = self._client.key_ring_path(project_id, location, key_ring)

            # Ensure key ring exists
            self._ensure_key_ring()

        except Exception as e:
            if "permission" in str(e).lower() or "403" in str(e):
                raise KMSAuthenticationError(f"GCP authentication/authorization failed: {e}")
            raise KMSOperationError(f"GCP KMS initialization failed: {e}")

        logger.info(f"GCP KMS provider initialized (project: {project_id}, location: {location})")

    @property
    def provider_name(self) -> str:
        return "gcp"

    def _ensure_key_ring(self) -> None:
        """Create key ring if it doesn't exist."""
        from google.api_core import exceptions

        try:
            self._client.get_key_ring(request={"name": self._key_ring_path})
        except exceptions.NotFound:
            # Create the key ring
            parent = f"projects/{self._project_id}/locations/{self._location}"
            self._client.create_key_ring(
                request={
                    "parent": parent,
                    "key_ring_id": self._key_ring,
                    "key_ring": {},
                }
            )
            logger.info(f"Created GCP key ring: {self._key_ring}")

    def _get_crypto_key_path(self, key_id: str) -> str:
        """Get the full crypto key path."""
        return self._client.crypto_key_path(
            self._project_id, self._location, self._key_ring, key_id
        )

    def _algorithm_to_gcp(self, algorithm: KeyAlgorithm) -> str:
        """Convert KeyAlgorithm to GCP crypto key version algorithm."""
        from google.cloud import kms

        mapping = {
            KeyAlgorithm.AES_256_GCM: kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION,
            KeyAlgorithm.AES_256_CBC: kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION,
            KeyAlgorithm.RSA_2048: kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.RSA_DECRYPT_OAEP_2048_SHA256,
            KeyAlgorithm.RSA_4096: kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.RSA_DECRYPT_OAEP_4096_SHA256,
            KeyAlgorithm.EC_P256: kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.EC_SIGN_P256_SHA256,
            KeyAlgorithm.EC_P384: kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.EC_SIGN_P384_SHA384,
        }
        return mapping.get(
            algorithm,
            kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION,
        )

    def _key_type_to_purpose(self, key_type: KeyType) -> str:
        """Convert KeyType to GCP crypto key purpose."""
        from google.cloud import kms

        if key_type == KeyType.SIGNING:
            return kms.CryptoKey.CryptoKeyPurpose.ASYMMETRIC_SIGN
        return kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT

    def generate_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.DEK,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM,
        tags: Optional[dict] = None,
    ) -> KeyMetadata:
        """Generate a new key in GCP KMS."""
        from google.api_core import exceptions
        from google.cloud import kms

        try:
            purpose = self._key_type_to_purpose(key_type)

            # Create crypto key
            crypto_key = kms.CryptoKey(
                purpose=purpose,
                version_template=kms.CryptoKeyVersionTemplate(
                    algorithm=self._algorithm_to_gcp(algorithm),
                ),
                labels=tags or {},
            )

            request = kms.CreateCryptoKeyRequest(
                parent=self._key_ring_path,
                crypto_key_id=key_id,
                crypto_key=crypto_key,
            )

            response = self._client.create_crypto_key(request=request)

            logger.info(f"Created GCP KMS key: {key_id}")

            return KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                created_at=datetime.utcnow(),
                is_active=True,
                provider=self.provider_name,
                provider_key_ref=response.name,
                tags=tags or {},
            )

        except exceptions.AlreadyExists:
            raise KMSOperationError(f"Key already exists: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to create key: {e}")

    def get_key_material(self, key_id: str) -> bytes:
        """
        GCP KMS does not allow key export.
        """
        raise KMSOperationError(
            "GCP KMS does not support key export. Use encrypt/decrypt methods directly."
        )

    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[dict] = None) -> bytes:
        """Encrypt data using GCP KMS."""
        from google.api_core import exceptions
        from google.cloud import kms

        try:
            key_path = self._get_crypto_key_path(key_id)

            # Build encryption context as additional authenticated data
            aad = None
            if context:
                import json

                aad = json.dumps(context, sort_keys=True).encode()

            request = kms.EncryptRequest(
                name=key_path,
                plaintext=plaintext,
                additional_authenticated_data=aad,
            )

            response = self._client.encrypt(request=request)
            return response.ciphertext

        except exceptions.NotFound:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Encryption failed: {e}")

    def decrypt(self, key_id: str, ciphertext: bytes, context: Optional[dict] = None) -> bytes:
        """Decrypt data using GCP KMS."""
        from google.api_core import exceptions
        from google.cloud import kms

        try:
            key_path = self._get_crypto_key_path(key_id)

            # Build encryption context
            aad = None
            if context:
                import json

                aad = json.dumps(context, sort_keys=True).encode()

            request = kms.DecryptRequest(
                name=key_path,
                ciphertext=ciphertext,
                additional_authenticated_data=aad,
            )

            response = self._client.decrypt(request=request)
            return response.plaintext

        except exceptions.NotFound:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Decryption failed: {e}")

    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a GCP KMS key (creates new primary version)."""
        from google.api_core import exceptions
        from google.cloud import kms

        try:
            key_path = self._get_crypto_key_path(key_id)

            # Create a new key version
            request = kms.CreateCryptoKeyVersionRequest(
                parent=key_path,
                crypto_key_version=kms.CryptoKeyVersion(),
            )

            self._client.create_crypto_key_version(request=request)

            logger.info(f"Rotated GCP KMS key: {key_id}")
            return self.get_key_metadata(key_id)

        except exceptions.NotFound:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to rotate key: {e}")

    def delete_key(self, key_id: str, schedule_days: int = 30) -> None:
        """Schedule key deletion in GCP KMS (destroys all versions)."""
        from google.api_core import exceptions
        from google.cloud import kms

        try:
            key_path = self._get_crypto_key_path(key_id)

            # List all key versions
            request = kms.ListCryptoKeyVersionsRequest(parent=key_path)
            versions = self._client.list_crypto_key_versions(request=request)

            # Schedule destruction for each version
            for version in versions:
                if version.state != kms.CryptoKeyVersion.CryptoKeyVersionState.DESTROYED:
                    destroy_request = kms.DestroyCryptoKeyVersionRequest(name=version.name)
                    self._client.destroy_crypto_key_version(request=destroy_request)

            logger.info(f"Scheduled deletion for GCP KMS key: {key_id}")

        except exceptions.NotFound:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to delete key: {e}")

    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata from GCP KMS."""
        from google.api_core import exceptions
        from google.cloud import kms

        try:
            key_path = self._get_crypto_key_path(key_id)
            request = kms.GetCryptoKeyRequest(name=key_path)
            response = self._client.get_crypto_key(request=request)

            return KeyMetadata(
                key_id=key_id,
                key_type=KeyType.DEK,
                algorithm=KeyAlgorithm.AES_256_GCM,
                created_at=response.create_time.replace(tzinfo=None),
                is_active=True,
                provider=self.provider_name,
                provider_key_ref=response.name,
                tags=dict(response.labels),
            )

        except exceptions.NotFound:
            raise KMSKeyNotFoundError(f"Key not found: {key_id}")
        except Exception as e:
            raise KMSOperationError(f"Failed to get key metadata: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> list[KeyMetadata]:
        """List keys from GCP KMS."""
        from google.cloud import kms

        result = []

        try:
            request = kms.ListCryptoKeysRequest(parent=self._key_ring_path)
            keys = self._client.list_crypto_keys(request=request)

            for key in keys:
                key_id = key.name.split("/")[-1]
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
        """Check GCP KMS connectivity."""
        start = time.time()
        try:
            # Try to get the key ring
            self._client.get_key_ring(request={"name": self._key_ring_path})
            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000,
                "provider": "gcp",
                "project": self._project_id,
                "location": self._location,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
                "provider": "gcp",
            }
