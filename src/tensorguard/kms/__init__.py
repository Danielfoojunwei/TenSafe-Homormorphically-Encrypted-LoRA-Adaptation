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

from .provider import (
    KMSProvider,
    KMSError,
    KMSKeyNotFoundError,
    KMSAuthenticationError,
    KMSOperationError,
)
from .factory import create_kms_provider, KMSConfig

__all__ = [
    "KMSProvider",
    "KMSError",
    "KMSKeyNotFoundError",
    "KMSAuthenticationError",
    "KMSOperationError",
    "create_kms_provider",
    "KMSConfig",
]
