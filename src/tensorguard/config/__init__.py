"""
TensorGuard Configuration Module.

Provides configuration management for production deployments:
- Production cryptography configuration (HE, PQC, Symmetric)
- Environment-based configuration loading
- Configuration validation

Usage:
    from tensorguard.config import (
        ProductionCryptoConfig,
        get_crypto_config,
        validate_production_crypto,
        CryptoMode,
    )
"""

from .production_crypto import (
    ProductionCryptoConfig,
    HEConfig,
    PQCConfig,
    SymmetricConfig,
    CryptoMode,
    HEBackend,
    PQCBackend,
    SymmetricAlgorithm,
    HashAlgorithm,
    get_crypto_config,
    validate_production_crypto,
    get_recommended_production_config,
    check_production_environment,
)

__all__ = [
    # Configuration classes
    "ProductionCryptoConfig",
    "HEConfig",
    "PQCConfig",
    "SymmetricConfig",
    # Enums
    "CryptoMode",
    "HEBackend",
    "PQCBackend",
    "SymmetricAlgorithm",
    "HashAlgorithm",
    # Functions
    "get_crypto_config",
    "validate_production_crypto",
    "get_recommended_production_config",
    "check_production_environment",
]
