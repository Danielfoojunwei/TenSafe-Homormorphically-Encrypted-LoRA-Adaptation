"""
Production Cryptography Configuration.

Provides production-ready configuration for:
- Homomorphic Encryption (HE-LoRA)
- Post-Quantum Cryptography (PQC)
- Key Management
- Algorithm selection

This module ensures that production deployments use
secure, non-simulated cryptographic implementations.

Compliance Requirements:
- SOC 2 CC6.7: Cryptographic controls
- ISO 27001 A.8.24: Use of cryptography
- HIPAA §164.312(a)(2)(iv): Encryption

Usage:
    from tensorguard.config.production_crypto import (
        get_crypto_config,
        validate_production_crypto,
        CryptoMode,
    )
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CryptoMode(str, Enum):
    """Cryptography operation mode."""

    PRODUCTION = "production"  # Full security, requires native libraries
    TESTING = "testing"  # Simulated/toy implementations for testing
    HYBRID = "hybrid"  # Production where available, fallback to testing


class HEBackend(str, Enum):
    """Homomorphic Encryption backend."""

    N2HE_NATIVE = "n2he_native"  # Native N2HE library (production)
    TENSEAL = "tenseal"  # TenSEAL library
    TOY_SIMULATION = "toy_simulation"  # Toy/simulation mode (testing only)


class PQCBackend(str, Enum):
    """Post-Quantum Cryptography backend."""

    LIBOQS = "liboqs"  # Open Quantum Safe (production)
    PQCRYPTO = "pqcrypto"  # PQCrypto library
    SIMULATION = "simulation"  # Simulated implementations (testing only)


class SymmetricAlgorithm(str, Enum):
    """Symmetric encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"  # NIST-approved
    CHACHA20_POLY1305 = "chacha20-poly1305"  # IETF standard


class HashAlgorithm(str, Enum):
    """Hash algorithms."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3-256"
    BLAKE2B = "blake2b"


@dataclass
class HEConfig:
    """Homomorphic Encryption configuration."""

    backend: HEBackend = HEBackend.TOY_SIMULATION
    security_level: int = 128  # Security bits
    poly_modulus_degree: int = 8192  # CKKS parameter
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale: float = 2**40
    enable_batching: bool = True
    enable_rotation: bool = True  # For MOAI optimization

    def validate(self) -> List[str]:
        """Validate HE configuration."""
        errors = []

        if self.security_level < 128:
            errors.append(f"Security level {self.security_level} is below minimum 128 bits")

        if self.poly_modulus_degree not in [4096, 8192, 16384, 32768]:
            errors.append(f"Invalid poly_modulus_degree: {self.poly_modulus_degree}")

        if self.backend == HEBackend.TOY_SIMULATION:
            errors.append("TOY_SIMULATION backend is not secure for production")

        return errors


@dataclass
class PQCConfig:
    """Post-Quantum Cryptography configuration."""

    backend: PQCBackend = PQCBackend.SIMULATION
    signature_algorithm: str = "dilithium3"  # NIST PQC Level 3
    kem_algorithm: str = "kyber768"  # NIST PQC Level 3
    hybrid_mode: bool = True  # Combine with classical crypto
    classical_signature: str = "ed25519"  # Classical fallback
    classical_kem: str = "x25519"  # Classical fallback

    def validate(self) -> List[str]:
        """Validate PQC configuration."""
        errors = []

        if self.backend == PQCBackend.SIMULATION:
            errors.append("SIMULATION backend is not secure for production")

        valid_sig_algos = ["dilithium2", "dilithium3", "dilithium5", "falcon512", "falcon1024"]
        if self.signature_algorithm not in valid_sig_algos:
            errors.append(f"Invalid signature algorithm: {self.signature_algorithm}")

        valid_kem_algos = ["kyber512", "kyber768", "kyber1024"]
        if self.kem_algorithm not in valid_kem_algos:
            errors.append(f"Invalid KEM algorithm: {self.kem_algorithm}")

        return errors


@dataclass
class SymmetricConfig:
    """Symmetric cryptography configuration."""

    algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES_256_GCM
    key_size: int = 256  # bits
    nonce_size: int = 12  # bytes for GCM
    tag_size: int = 16  # bytes
    kdf_algorithm: str = "pbkdf2"
    kdf_iterations: int = 100000
    kdf_hash: HashAlgorithm = HashAlgorithm.SHA256

    def validate(self) -> List[str]:
        """Validate symmetric configuration."""
        errors = []

        if self.key_size < 256:
            errors.append(f"Key size {self.key_size} is below minimum 256 bits")

        if self.kdf_iterations < 100000:
            errors.append(f"KDF iterations {self.kdf_iterations} is below recommended 100000")

        return errors


@dataclass
class ProductionCryptoConfig:
    """Complete production cryptography configuration."""

    mode: CryptoMode = CryptoMode.TESTING
    he: HEConfig = field(default_factory=HEConfig)
    pqc: PQCConfig = field(default_factory=PQCConfig)
    symmetric: SymmetricConfig = field(default_factory=SymmetricConfig)

    # Key management
    key_rotation_days: int = 90
    key_backup_enabled: bool = True
    hsm_enabled: bool = False
    hsm_provider: Optional[str] = None  # "aws_kms", "gcp_kms", "azure_keyvault"

    # Audit
    crypto_audit_enabled: bool = True

    def validate(self) -> Dict[str, List[str]]:
        """Validate complete configuration."""
        errors = {
            "he": self.he.validate(),
            "pqc": self.pqc.validate(),
            "symmetric": self.symmetric.validate(),
            "general": [],
        }

        if self.mode == CryptoMode.PRODUCTION:
            if self.he.backend == HEBackend.TOY_SIMULATION:
                errors["general"].append("Production mode requires non-toy HE backend")
            if self.pqc.backend == PQCBackend.SIMULATION:
                errors["general"].append("Production mode requires non-simulated PQC backend")
            if not self.hsm_enabled:
                errors["general"].append("Production mode recommends HSM for key storage")

        return errors

    def is_production_ready(self) -> bool:
        """Check if configuration is production-ready."""
        errors = self.validate()
        return all(len(e) == 0 for e in errors.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "he": {
                "backend": self.he.backend.value,
                "security_level": self.he.security_level,
            },
            "pqc": {
                "backend": self.pqc.backend.value,
                "signature_algorithm": self.pqc.signature_algorithm,
                "kem_algorithm": self.pqc.kem_algorithm,
                "hybrid_mode": self.pqc.hybrid_mode,
            },
            "symmetric": {
                "algorithm": self.symmetric.algorithm.value,
                "key_size": self.symmetric.key_size,
            },
            "key_rotation_days": self.key_rotation_days,
            "hsm_enabled": self.hsm_enabled,
            "is_production_ready": self.is_production_ready(),
        }


def get_crypto_config() -> ProductionCryptoConfig:
    """
    Get cryptography configuration from environment.

    Returns:
        ProductionCryptoConfig based on environment variables
    """
    mode_str = os.getenv("TG_CRYPTO_MODE", "testing")
    try:
        mode = CryptoMode(mode_str)
    except ValueError:
        mode = CryptoMode.TESTING

    config = ProductionCryptoConfig(mode=mode)

    # Configure HE backend
    he_backend_str = os.getenv("TG_HE_BACKEND", "toy_simulation")
    try:
        config.he.backend = HEBackend(he_backend_str)
    except ValueError:
        pass

    # Configure PQC backend
    pqc_backend_str = os.getenv("TG_PQC_BACKEND", "simulation")
    try:
        config.pqc.backend = PQCBackend(pqc_backend_str)
    except ValueError:
        pass

    # Configure symmetric
    sym_algo_str = os.getenv("TG_SYMMETRIC_ALGORITHM", "aes-256-gcm")
    try:
        config.symmetric.algorithm = SymmetricAlgorithm(sym_algo_str)
    except ValueError:
        pass

    # Key management
    config.key_rotation_days = int(os.getenv("TG_KEY_ROTATION_DAYS", "90"))
    config.hsm_enabled = os.getenv("TG_HSM_ENABLED", "false").lower() == "true"
    config.hsm_provider = os.getenv("TG_HSM_PROVIDER")

    return config


def validate_production_crypto() -> bool:
    """
    Validate cryptography configuration for production.

    Returns:
        True if configuration is valid

    Raises:
        RuntimeError: If configuration is invalid in production mode
    """
    config = get_crypto_config()
    errors = config.validate()

    # Check for any errors
    all_errors = []
    for category, error_list in errors.items():
        for error in error_list:
            all_errors.append(f"{category}: {error}")

    if all_errors:
        error_msg = "Cryptography configuration errors:\n" + "\n".join(f"  - {e}" for e in all_errors)

        if config.mode == CryptoMode.PRODUCTION:
            logger.critical(error_msg)
            raise RuntimeError(f"Production cryptography validation failed: {len(all_errors)} errors")
        else:
            logger.warning(error_msg)

    return len(all_errors) == 0


def get_recommended_production_config() -> ProductionCryptoConfig:
    """
    Get recommended production configuration.

    Returns:
        ProductionCryptoConfig with recommended production settings
    """
    return ProductionCryptoConfig(
        mode=CryptoMode.PRODUCTION,
        he=HEConfig(
            backend=HEBackend.N2HE_NATIVE,
            security_level=128,
            poly_modulus_degree=8192,
        ),
        pqc=PQCConfig(
            backend=PQCBackend.LIBOQS,
            signature_algorithm="dilithium3",
            kem_algorithm="kyber768",
            hybrid_mode=True,
        ),
        symmetric=SymmetricConfig(
            algorithm=SymmetricAlgorithm.AES_256_GCM,
            key_size=256,
            kdf_iterations=100000,
        ),
        key_rotation_days=90,
        hsm_enabled=True,
        hsm_provider="aws_kms",  # Or appropriate provider
        crypto_audit_enabled=True,
    )


# Production environment check
def check_production_environment() -> Dict[str, Any]:
    """
    Check production environment for cryptography requirements.

    Returns:
        Dictionary with check results
    """
    results = {
        "environment": os.getenv("TG_ENVIRONMENT", "development"),
        "checks": {},
    }

    # Check for native N2HE library
    try:
        import n2he  # noqa: F401
        results["checks"]["n2he_native"] = True
    except ImportError:
        results["checks"]["n2he_native"] = False

    # Check for liboqs
    try:
        import oqs  # noqa: F401
        results["checks"]["liboqs"] = True
    except ImportError:
        results["checks"]["liboqs"] = False

    # Check for TenSEAL
    try:
        import tenseal  # noqa: F401
        results["checks"]["tenseal"] = True
    except ImportError:
        results["checks"]["tenseal"] = False

    # Check environment variables
    results["checks"]["secret_key_set"] = bool(os.getenv("TG_SECRET_KEY"))
    results["checks"]["production_mode"] = os.getenv("TG_ENVIRONMENT") == "production"
    results["checks"]["toy_he_disabled"] = os.getenv("TENSAFE_TOY_HE", "1") == "0"

    # Overall readiness
    results["production_ready"] = (
        results["checks"].get("n2he_native", False) or results["checks"].get("tenseal", False)
    ) and results["checks"]["secret_key_set"] and results["checks"]["toy_he_disabled"]

    return results
