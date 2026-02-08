"""
TenSafe Attestation Module.

Provides hardware-based attestation for secure workload verification.

Supported attestation methods:
- TPM 2.0 attestation (hardware-based)
- Intel TDX (Trust Domain Extensions) for confidential VMs
- AMD SEV-SNP (Secure Encrypted Virtualization) for confidential VMs
- Software attestation (development only)
"""

from .factory import AttestationConfig, create_attestation_provider
from .provider import (
    AttestationError,
    AttestationProvider,
    AttestationQuote,
    AttestationResult,
    AttestationType,
    QuoteType,
    VerificationPolicy,
)
from .sev import SEVSNPAttestationProvider, SNPVerificationPolicy
from .tdx import TDXAttestationProvider, TDXVerificationPolicy
from .tpm import TPMAttestationProvider

__all__ = [
    "AttestationProvider",
    "AttestationError",
    "AttestationQuote",
    "AttestationResult",
    "AttestationType",
    "QuoteType",
    "VerificationPolicy",
    "TPMAttestationProvider",
    "TDXAttestationProvider",
    "TDXVerificationPolicy",
    "SEVSNPAttestationProvider",
    "SNPVerificationPolicy",
    "create_attestation_provider",
    "AttestationConfig",
]
