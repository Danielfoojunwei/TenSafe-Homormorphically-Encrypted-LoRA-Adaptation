"""
TenSafe Attestation Module.

Provides hardware-based attestation for secure workload verification.

Supported attestation methods:
- TPM 2.0 attestation (hardware-based)
- Intel SGX (future)
- AMD SEV (future)
- Software attestation (development only)
"""

from .factory import AttestationConfig, create_attestation_provider
from .provider import (
    AttestationError,
    AttestationProvider,
    AttestationQuote,
    AttestationResult,
    VerificationPolicy,
)
from .tpm import TPMAttestationProvider

__all__ = [
    "AttestationProvider",
    "AttestationError",
    "AttestationQuote",
    "AttestationResult",
    "VerificationPolicy",
    "TPMAttestationProvider",
    "create_attestation_provider",
    "AttestationConfig",
]
