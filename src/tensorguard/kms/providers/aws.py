"""
AWS KMS Provider.

Production-grade key management using AWS Key Management Service.
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


class AWSKMSProvider(KMSProvider):
    """
    AWS KMS Provider.

    Uses AWS Key Management Service for key generation, encryption, and management.
    Keys are never exported - all crypto operations happen within KMS.

    Requirements:
    - boto3 library installed
    - AWS credentials configured (env vars, IAM role, or credentials file)
    - Appropriate IAM permissions for KMS operations
    """

    def __init__(
        self,
        region: Optional[str] = None,
        key_alias_prefix: str = "alias/tensafe",
        endpoint_url: Optional[str] = None,
    ):
        """
        Initialize AWS KMS provider.

        Args:
            region: AWS region (defaults to AWS_DEFAULT_REGION env var)
            key_alias_prefix: Prefix for key aliases
            endpoint_url: Optional custom endpoint (for LocalStack testing)
        """
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:
            raise KMSError("boto3 is required for AWS KMS. Install with: pip install boto3")

        self._key_alias_prefix = key_alias_prefix

        try:
            self._client = boto3.client(
                "kms",
                region_name=region,
                endpoint_url=endpoint_url,
            )
            # Validate connectivity
            self._client.list_keys(Limit=1)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("AccessDenied", "UnauthorizedAccess"):
                raise KMSAuthenticationError(f"AWS authentication failed: {e}")
            raise KMSOperationError(f"AWS KMS initialization failed: {e}")
        except BotoCoreError as e:
            raise KMSOperationError(f"AWS KMS initialization failed: {e}")

        logger.info(f"AWS KMS provider initialized (region: {region or 'default'})")

    @property
    def provider_name(self) -> str:
        return "aws"

    def _get_key_alias(self, key_id: str) -> str:
        """Get the full alias for a key."""
        return f"{self._key_alias_prefix}/{key_id}"

    def _get_key_arn_or_alias(self, key_id: str) -> str:
        """Get the key ARN or alias for operations."""
        # If it looks like an ARN, use it directly
        if key_id.startswith("arn:aws:kms:"):
            return key_id
        return self._get_key_alias(key_id)

    def _key_type_to_usage(self, key_type: KeyType) -> str:
        """Convert KeyType to AWS key usage."""
        if key_type in (KeyType.KEK, KeyType.DEK, KeyType.SYMMETRIC):
            return "ENCRYPT_DECRYPT"
        elif key_type == KeyType.SIGNING:
            return "SIGN_VERIFY"
        return "ENCRYPT_DECRYPT"

    def _algorithm_to_spec(self, algorithm: KeyAlgorithm) -> str:
        """Convert KeyAlgorithm to AWS key spec."""
        mapping = {
            KeyAlgorithm.AES_256_GCM: "SYMMETRIC_DEFAULT",
            KeyAlgorithm.AES_256_CBC: "SYMMETRIC_DEFAULT",
            KeyAlgorithm.RSA_2048: "RSA_2048",
            KeyAlgorithm.RSA_4096: "RSA_4096",
            KeyAlgorithm.EC_P256: "ECC_NIST_P256",
            KeyAlgorithm.EC_P384: "ECC_NIST_P384",
        }
        return mapping.get(algorithm, "SYMMETRIC_DEFAULT")

    def generate_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.DEK,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM,
        tags: Optional[dict] = None,
    ) -> KeyMetadata:
        """Generate a new key in AWS KMS."""
        from botocore.exceptions import ClientError

        try:
            # Create the key
            key_spec = self._algorithm_to_spec(algorithm)
            key_usage = self._key_type_to_usage(key_type)

            aws_tags = [{"TagKey": "tensafe_key_id", "TagValue": key_id}]
            if tags:
                aws_tags.extend([{"TagKey": k, "TagValue": str(v)} for k, v in tags.items()])

            response = self._client.create_key(
                Description=f"TenSafe {key_type.value} key: {key_id}",
                KeyUsage=key_usage,
                KeySpec=key_spec,
                Origin="AWS_KMS",
                Tags=aws_tags,
            )

            key_arn = response["KeyMetadata"]["Arn"]
            created = response["KeyMetadata"]["CreationDate"]

            # Create alias
            alias = self._get_key_alias(key_id)
            try:
                self._client.create_alias(AliasName=alias, TargetKeyId=key_arn)
            except ClientError as e:
                # Alias might already exist
                if e.response["Error"]["Code"] != "AlreadyExistsException":
                    raise

            logger.info(f"Created AWS KMS key: {key_id} (ARN: {key_arn})")

            return KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                created_at=created if isinstance(created, datetime) else datetime.now(),
                is_active=True,
                provider=self.provider_name,
                provider_key_ref=key_arn,
                tags=tags or {},
            )

        except ClientError as e:
            raise KMSOperationError(f"Failed to create key: {e}")

    def get_key_material(self, key_id: str) -> bytes:
        """
        AWS KMS does not allow key export.

        Raises KMSOperationError indicating keys cannot be exported.
        Use encrypt/decrypt methods directly instead.
        """
        raise KMSOperationError(
            "AWS KMS does not support key export. Use encrypt/decrypt methods directly, "
            "or generate a data key with generate_data_key()."
        )

    def generate_data_key(
        self, key_id: str, key_spec: str = "AES_256", context: Optional[dict] = None
    ) -> tuple[bytes, bytes]:
        """
        Generate a data encryption key.

        Returns both plaintext and encrypted versions of the DEK.
        Use the encrypted version for storage, plaintext for operations.

        Args:
            key_id: Master key ID to use for encryption
            key_spec: Key specification (AES_256 or AES_128)
            context: Optional encryption context

        Returns:
            Tuple of (plaintext_key, encrypted_key)
        """
        from botocore.exceptions import ClientError

        try:
            params = {
                "KeyId": self._get_key_arn_or_alias(key_id),
                "KeySpec": key_spec,
            }
            if context:
                params["EncryptionContext"] = {k: str(v) for k, v in context.items()}

            response = self._client.generate_data_key(**params)

            return response["Plaintext"], response["CiphertextBlob"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to generate data key: {e}")

    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[dict] = None) -> bytes:
        """Encrypt data using AWS KMS."""
        from botocore.exceptions import ClientError

        try:
            params = {
                "KeyId": self._get_key_arn_or_alias(key_id),
                "Plaintext": plaintext,
            }
            if context:
                params["EncryptionContext"] = {k: str(v) for k, v in context.items()}

            response = self._client.encrypt(**params)
            return response["CiphertextBlob"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Encryption failed: {e}")

    def decrypt(self, key_id: str, ciphertext: bytes, context: Optional[dict] = None) -> bytes:
        """Decrypt data using AWS KMS."""
        from botocore.exceptions import ClientError

        try:
            params = {
                "KeyId": self._get_key_arn_or_alias(key_id),
                "CiphertextBlob": ciphertext,
            }
            if context:
                params["EncryptionContext"] = {k: str(v) for k, v in context.items()}

            response = self._client.decrypt(**params)
            return response["Plaintext"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            if e.response["Error"]["Code"] == "InvalidCiphertextException":
                raise KMSOperationError("Decryption failed: invalid ciphertext or context mismatch")
            raise KMSOperationError(f"Decryption failed: {e}")

    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Enable automatic key rotation for an AWS KMS key."""
        from botocore.exceptions import ClientError

        try:
            key_ref = self._get_key_arn_or_alias(key_id)
            self._client.enable_key_rotation(KeyId=key_ref)
            logger.info(f"Enabled key rotation for: {key_id}")
            return self.get_key_metadata(key_id)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to enable rotation: {e}")

    def delete_key(self, key_id: str, schedule_days: int = 30) -> None:
        """Schedule key deletion in AWS KMS."""
        from botocore.exceptions import ClientError

        if schedule_days < 7:
            schedule_days = 7  # AWS minimum is 7 days

        try:
            key_ref = self._get_key_arn_or_alias(key_id)
            self._client.schedule_key_deletion(
                KeyId=key_ref,
                PendingWindowInDays=schedule_days,
            )
            logger.info(f"Scheduled key deletion: {key_id} (in {schedule_days} days)")

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to schedule deletion: {e}")

    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata from AWS KMS."""
        from botocore.exceptions import ClientError

        try:
            key_ref = self._get_key_arn_or_alias(key_id)
            response = self._client.describe_key(KeyId=key_ref)
            metadata = response["KeyMetadata"]

            # Get tags
            try:
                tags_response = self._client.list_resource_tags(KeyId=metadata["Arn"])
                tags = {t["TagKey"]: t["TagValue"] for t in tags_response.get("Tags", [])}
            except ClientError:
                tags = {}

            # Determine key type from tags or usage
            key_type_str = tags.get("tensafe_key_type", "dek")
            try:
                key_type = KeyType(key_type_str)
            except ValueError:
                key_type = KeyType.DEK

            return KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                algorithm=KeyAlgorithm.AES_256_GCM,  # AWS default
                created_at=metadata["CreationDate"],
                is_active=metadata["KeyState"] == "Enabled",
                provider=self.provider_name,
                provider_key_ref=metadata["Arn"],
                tags=tags,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key not found: {key_id}")
            raise KMSOperationError(f"Failed to get key metadata: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> list[KeyMetadata]:
        """List keys from AWS KMS."""
        from botocore.exceptions import ClientError

        result = []
        try:
            paginator = self._client.get_paginator("list_aliases")
            for page in paginator.paginate():
                for alias in page.get("Aliases", []):
                    alias_name = alias.get("AliasName", "")
                    if alias_name.startswith(self._key_alias_prefix):
                        key_id = alias_name.replace(f"{self._key_alias_prefix}/", "")
                        try:
                            metadata = self.get_key_metadata(key_id)
                            if key_type is None or metadata.key_type == key_type:
                                result.append(metadata)
                        except KMSKeyNotFoundError:
                            continue

        except ClientError as e:
            raise KMSOperationError(f"Failed to list keys: {e}")

        return result

    def health_check(self) -> dict:
        """Check AWS KMS connectivity."""
        from botocore.exceptions import ClientError

        start = time.time()
        try:
            self._client.list_keys(Limit=1)
            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000,
                "provider": "aws",
            }
        except ClientError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
                "provider": "aws",
            }
