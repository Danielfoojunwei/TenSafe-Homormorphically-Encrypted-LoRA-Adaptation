"""
Attestation Provider Factory.

Creates the appropriate attestation provider based on configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .provider import AttestationError, AttestationProvider

logger = logging.getLogger(__name__)


@dataclass
class AttestationConfig:
    """Configuration for attestation provider."""

    # Provider type: 'tpm', 'sgx', 'sev', 'software', 'auto'
    provider: str = field(
        default_factory=lambda: os.getenv("TENSAFE_ATTESTATION_PROVIDER", "auto")
    )

    # TPM settings
    tpm_device_path: Optional[str] = field(
        default_factory=lambda: os.getenv("TENSAFE_TPM_DEVICE")
    )
    tpm_tcti: Optional[str] = field(default_factory=lambda: os.getenv("TPM2TOOLS_TCTI"))

    # Whether to allow software attestation in production
    allow_software_attestation: bool = field(
        default_factory=lambda: os.getenv("TENSAFE_ALLOW_SOFTWARE_ATTESTATION", "false").lower()
        == "true"
    )

    # Environment detection
    is_production: bool = field(
        default_factory=lambda: os.getenv("TG_ENVIRONMENT", "development") == "production"
    )

    @classmethod
    def from_env(cls) -> "AttestationConfig":
        """Create config from environment variables."""
        return cls()


def create_attestation_provider(
    config: Optional[AttestationConfig] = None,
) -> AttestationProvider:
    """
    Create an attestation provider based on configuration.

    Args:
        config: Attestation configuration. If None, reads from environment.

    Returns:
        Configured attestation provider

    Raises:
        AttestationError: If provider creation fails
    """
    if config is None:
        config = AttestationConfig.from_env()

    provider_type = config.provider.lower()

    if provider_type == "auto":
        return _auto_detect_provider(config)

    if provider_type == "tpm":
        from .tpm import TPMAttestationProvider

        provider = TPMAttestationProvider(
            device_path=config.tpm_device_path,
            tcti=config.tpm_tcti,
            use_software_tpm=not config.is_production,
        )

        if not provider.is_available and config.is_production:
            raise AttestationError("TPM not available in production mode")

        return provider

    if provider_type == "software":
        if config.is_production and not config.allow_software_attestation:
            raise AttestationError(
                "Software attestation not allowed in production. "
                "Set TENSAFE_ALLOW_SOFTWARE_ATTESTATION=true to override (NOT RECOMMENDED)"
            )

        from .tpm import TPMAttestationProvider

        logger.warning("Using software attestation - NOT FOR PRODUCTION")
        return TPMAttestationProvider(use_software_tpm=True)

    if provider_type in ("sgx", "tdx"):
        from .tdx import TDXAttestationProvider

        use_sim = not config.is_production
        provider = TDXAttestationProvider(use_simulation=use_sim)

        if not provider.is_available and config.is_production:
            raise AttestationError("Intel TDX not available in production mode")

        return provider

    if provider_type in ("sev", "sev-snp", "snp"):
        from .sev import SEVSNPAttestationProvider

        use_sim = not config.is_production
        provider = SEVSNPAttestationProvider(use_simulation=use_sim)

        if not provider.is_available and config.is_production:
            raise AttestationError("AMD SEV-SNP not available in production mode")

        return provider

    raise AttestationError(
        f"Unknown attestation provider: {provider_type}. "
        f"Supported: auto, tpm, software, tdx, sev-snp"
    )


def _auto_detect_provider(config: AttestationConfig) -> AttestationProvider:
    """Auto-detect the best available attestation provider."""
    # Try TDX first (most common for confidential AI on cloud)
    from .tdx import TDXAttestationProvider
    from .tpm import TPMAttestationProvider

    tdx_provider = TDXAttestationProvider(use_simulation=False)
    if tdx_provider.is_available:
        logger.info("Auto-detected Intel TDX attestation")
        return tdx_provider

    # Try SEV-SNP
    from .sev import SEVSNPAttestationProvider

    sev_provider = SEVSNPAttestationProvider(use_simulation=False)
    if sev_provider.is_available:
        logger.info("Auto-detected AMD SEV-SNP attestation")
        return sev_provider

    # Try TPM
    tpm_provider = TPMAttestationProvider(
        device_path=config.tpm_device_path,
        tcti=config.tpm_tcti,
        use_software_tpm=False,
    )

    if tpm_provider.is_available:
        logger.info("Auto-detected TPM attestation")
        return tpm_provider

    # In production, require hardware attestation
    if config.is_production:
        if not config.allow_software_attestation:
            raise AttestationError(
                "No hardware attestation available in production. "
                "Set TENSAFE_ALLOW_SOFTWARE_ATTESTATION=true to use software fallback "
                "(NOT RECOMMENDED)"
            )

    # Fall back to software
    logger.warning("No hardware attestation available, using software fallback")
    return TPMAttestationProvider(use_software_tpm=True)


def get_default_attestation_provider() -> AttestationProvider:
    """Get the default attestation provider based on environment."""
    return create_attestation_provider()
