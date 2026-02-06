"""
KMS Provider Factory.

Creates the appropriate KMS provider based on configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .provider import KMSError, KMSProvider

logger = logging.getLogger(__name__)


@dataclass
class KMSConfig:
    """
    Configuration for KMS provider.

    Set via environment variables or programmatically.
    """

    # Provider type: 'local', 'aws', 'vault', 'gcp', 'azure'
    provider: str = field(default_factory=lambda: os.getenv("TENSAFE_KMS_PROVIDER", "local"))

    # Local provider settings
    local_storage_path: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_LOCAL_PATH", "/tmp/tensafe_kms")
    )

    # AWS KMS settings
    aws_region: Optional[str] = field(default_factory=lambda: os.getenv("AWS_REGION"))
    aws_key_alias_prefix: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_AWS_PREFIX", "alias/tensafe")
    )
    aws_endpoint_url: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_KMS_ENDPOINT_URL")
    )

    # HashiCorp Vault settings
    vault_addr: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_ADDR"))
    vault_token: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_TOKEN"))
    vault_mount_point: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_VAULT_MOUNT", "transit")
    )
    vault_namespace: Optional[str] = field(
        default_factory=lambda: os.getenv("VAULT_NAMESPACE")
    )
    vault_key_prefix: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_VAULT_PREFIX", "tensafe")
    )

    # GCP KMS settings
    gcp_project_id: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    gcp_location: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_GCP_LOCATION", "global")
    )
    gcp_key_ring: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_GCP_KEYRING", "tensafe")
    )

    # Azure Key Vault settings
    azure_vault_url: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_KEYVAULT_URL")
    )
    azure_key_prefix: str = field(
        default_factory=lambda: os.getenv("TENSAFE_KMS_AZURE_PREFIX", "tensafe")
    )

    @classmethod
    def from_env(cls) -> "KMSConfig":
        """Create config from environment variables."""
        return cls()


def create_kms_provider(config: Optional[KMSConfig] = None) -> KMSProvider:
    """
    Create a KMS provider based on configuration.

    Args:
        config: KMS configuration. If None, reads from environment.

    Returns:
        Configured KMS provider

    Raises:
        KMSError: If provider creation fails
    """
    if config is None:
        config = KMSConfig.from_env()

    provider_type = config.provider.lower()

    if provider_type == "local":
        from .providers.local import LocalKMSProvider

        logger.info("Creating local KMS provider (development only)")
        return LocalKMSProvider(storage_path=config.local_storage_path)

    elif provider_type == "aws":
        from .providers.aws import AWSKMSProvider

        logger.info(f"Creating AWS KMS provider (region: {config.aws_region})")
        return AWSKMSProvider(
            region=config.aws_region,
            key_alias_prefix=config.aws_key_alias_prefix,
            endpoint_url=config.aws_endpoint_url,
        )

    elif provider_type == "vault":
        from .providers.vault import VaultKMSProvider

        logger.info(f"Creating HashiCorp Vault KMS provider (addr: {config.vault_addr})")
        return VaultKMSProvider(
            vault_addr=config.vault_addr,
            vault_token=config.vault_token,
            mount_point=config.vault_mount_point,
            namespace=config.vault_namespace,
            key_prefix=config.vault_key_prefix,
        )

    elif provider_type == "gcp":
        from .providers.gcp import GCPKMSProvider

        if not config.gcp_project_id:
            raise KMSError("GCP project ID required. Set GOOGLE_CLOUD_PROJECT env var.")

        logger.info(f"Creating GCP KMS provider (project: {config.gcp_project_id})")
        return GCPKMSProvider(
            project_id=config.gcp_project_id,
            location=config.gcp_location,
            key_ring=config.gcp_key_ring,
        )

    elif provider_type == "azure":
        from .providers.azure import AzureKMSProvider

        if not config.azure_vault_url:
            raise KMSError("Azure Key Vault URL required. Set AZURE_KEYVAULT_URL env var.")

        logger.info(f"Creating Azure Key Vault provider (vault: {config.azure_vault_url})")
        return AzureKMSProvider(
            vault_url=config.azure_vault_url,
            key_prefix=config.azure_key_prefix,
        )

    else:
        raise KMSError(
            f"Unknown KMS provider: {provider_type}. "
            f"Supported: local, aws, vault, gcp, azure"
        )


def get_default_kms_provider() -> KMSProvider:
    """
    Get the default KMS provider based on environment.

    Automatically detects available providers and selects the most appropriate:
    1. If TENSAFE_KMS_PROVIDER is set, use that
    2. If AWS credentials available, use AWS KMS
    3. If VAULT_ADDR is set, use Vault
    4. If GOOGLE_APPLICATION_CREDENTIALS is set, use GCP
    5. If AZURE_KEYVAULT_URL is set, use Azure
    6. Fall back to local provider

    Returns:
        Configured KMS provider
    """
    # Check explicit provider setting
    explicit_provider = os.getenv("TENSAFE_KMS_PROVIDER")
    if explicit_provider:
        return create_kms_provider()

    # Auto-detect
    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ROLE_ARN"):
        try:
            return create_kms_provider(KMSConfig(provider="aws"))
        except Exception as e:
            logger.warning(f"AWS KMS not available: {e}")

    if os.getenv("VAULT_ADDR") and os.getenv("VAULT_TOKEN"):
        try:
            return create_kms_provider(KMSConfig(provider="vault"))
        except Exception as e:
            logger.warning(f"Vault KMS not available: {e}")

    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.getenv("GOOGLE_CLOUD_PROJECT"):
        try:
            return create_kms_provider(KMSConfig(provider="gcp"))
        except Exception as e:
            logger.warning(f"GCP KMS not available: {e}")

    if os.getenv("AZURE_KEYVAULT_URL"):
        try:
            return create_kms_provider(KMSConfig(provider="azure"))
        except Exception as e:
            logger.warning(f"Azure KMS not available: {e}")

    # Fall back to local
    logger.warning("No production KMS configured, falling back to local provider")
    return create_kms_provider(KMSConfig(provider="local"))
