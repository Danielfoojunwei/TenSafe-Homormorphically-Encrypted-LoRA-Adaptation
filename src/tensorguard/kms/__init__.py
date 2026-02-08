"""
TenSafe KMS Plugin System.

Provides pluggable key management integrations for production deployment.

Supported providers:
- AWS KMS
- HashiCorp Vault
- GCP Cloud KMS
- Azure Key Vault
- Local (development/testing only)
"""

from .factory import KMSConfig, create_kms_provider
from .provider import (
    KMSAuthenticationError,
    KMSError,
    KMSKeyNotFoundError,
    KMSOperationError,
    KMSProvider,
)

__all__ = [
    "KMSProvider",
    "KMSError",
    "KMSKeyNotFoundError",
    "KMSAuthenticationError",
    "KMSOperationError",
    "create_kms_provider",
    "KMSConfig",
]
