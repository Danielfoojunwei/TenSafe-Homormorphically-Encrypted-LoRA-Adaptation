"""
TenSafe Attestation Module.

Provides hardware-based attestation for secure workload verification.

Supported attestation methods:
- TPM 2.0 attestation (hardware-based)
- Intel TDX (Trust Domain Extensions) for confidential VMs
- AMD SEV-SNP (Secure Encrypted Virtualization) for confidential VMs
- Software attestation (development only)
"""

from .provider import (
    AttestationProvider,
    AttestationError,
    AttestationQuote,
    AttestationResult,
    AttestationType,
    QuoteType,
    VerificationPolicy,
)
from .tpm import TPMAttestationProvider
from .tdx import TDXAttestationProvider, TDXVerificationPolicy
from .sev import SEVSNPAttestationProvider, SNPVerificationPolicy
from .factory import create_attestation_provider, AttestationConfig

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
