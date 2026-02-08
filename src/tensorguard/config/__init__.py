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
    CryptoMode,
    HashAlgorithm,
    HEBackend,
    HEConfig,
    PQCBackend,
    PQCConfig,
    ProductionCryptoConfig,
    SymmetricAlgorithm,
    SymmetricConfig,
    check_production_environment,
    get_crypto_config,
    get_recommended_production_config,
    validate_production_crypto,
)
from .runtime import (
    ENVIRONMENT,
    Environment,
    is_local_or_dev,
    is_production,
    require_env_var,
    validate_no_demo_mode,
)

__all__ = [
    # Runtime environment
    "ENVIRONMENT",
    "Environment",
    "is_production",
    "is_local_or_dev",
    "require_env_var",
    "validate_no_demo_mode",
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
