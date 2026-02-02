"""
Attestation Provider Base Interface.

Defines the abstract interface for hardware attestation providers.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AttestationError(Exception):
    """Base exception for attestation operations."""

    pass


class AttestationVerificationError(AttestationError):
    """Attestation verification failed."""

    pass


class AttestationType(str, Enum):
    """Types of attestation supported."""

    TPM = "tpm"
    SGX = "sgx"
    SEV = "sev"
    SOFTWARE = "software"


class QuoteType(str, Enum):
    """Types of attestation quotes."""

    PLATFORM = "platform"  # Full platform state
    APPLICATION = "application"  # Application-specific
    KEY = "key"  # Key attestation
    BINDING = "binding"  # Data binding


@dataclass
class AttestationQuote:
    """
    Attestation quote from a hardware security module.

    Contains the cryptographic evidence that proves platform state.
    """

    quote_id: str
    quote_type: QuoteType
    attestation_type: AttestationType
    timestamp: datetime

    # Raw quote data
    quote_data: bytes
    signature: bytes

    # Platform measurements
    pcr_values: dict[int, bytes] = field(default_factory=dict)  # PCR index -> value
    nonce: Optional[bytes] = None

    # Metadata
    firmware_version: Optional[str] = None
    attestation_key_id: Optional[str] = None
    extra_data: Optional[bytes] = None


@dataclass
class VerificationPolicy:
    """
    Policy for verifying attestation quotes.

    Defines what platform state is acceptable.
    """

    policy_id: str
    name: str

    # Required PCR values (index -> expected value)
    required_pcrs: dict[int, bytes] = field(default_factory=dict)

    # PCR mask - which PCRs must match (bitmask)
    pcr_mask: int = 0

    # Allowed firmware versions
    allowed_firmware_versions: list[str] = field(default_factory=list)

    # Minimum firmware version
    min_firmware_version: Optional[str] = None

    # Allowed attestation key IDs
    allowed_ak_ids: list[str] = field(default_factory=list)

    # Time validity
    max_quote_age_seconds: int = 300  # 5 minutes default

    # Whether to allow debug/development attestation
    allow_debug: bool = False

    # Custom verification callback
    custom_verifier: Optional[callable] = None


@dataclass
class AttestationResult:
    """
    Result of attestation verification.
    """

    verified: bool
    quote: AttestationQuote
    policy: VerificationPolicy

    # Verification details
    verification_time: datetime = field(default_factory=datetime.utcnow)
    pcr_match: bool = True
    firmware_match: bool = True
    signature_valid: bool = True
    nonce_valid: bool = True
    timestamp_valid: bool = True

    # Failure reasons
    failure_reasons: list[str] = field(default_factory=list)

    # Additional verification data
    platform_info: dict = field(default_factory=dict)


class AttestationProvider(ABC):
    """
    Abstract base class for attestation providers.

    All attestation implementations (TPM, SGX, SEV, etc.) must implement
    this interface for consistent attestation operations.
    """

    @property
    @abstractmethod
    def attestation_type(self) -> AttestationType:
        """Return the attestation type."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the attestation hardware is available."""
        pass

    @abstractmethod
    def generate_quote(
        self,
        nonce: Optional[bytes] = None,
        pcr_selection: Optional[list[int]] = None,
        extra_data: Optional[bytes] = None,
    ) -> AttestationQuote:
        """
        Generate an attestation quote.

        Args:
            nonce: Random nonce for freshness (if None, generates one)
            pcr_selection: Which PCRs to include (if None, uses default set)
            extra_data: Additional data to bind to the quote

        Returns:
            AttestationQuote containing the cryptographic evidence

        Raises:
            AttestationError: If quote generation fails
        """
        pass

    @abstractmethod
    def verify_quote(
        self,
        quote: AttestationQuote,
        policy: VerificationPolicy,
        expected_nonce: Optional[bytes] = None,
    ) -> AttestationResult:
        """
        Verify an attestation quote against a policy.

        Args:
            quote: The attestation quote to verify
            policy: Verification policy defining acceptable state
            expected_nonce: Expected nonce value for freshness check

        Returns:
            AttestationResult with verification outcome

        Raises:
            AttestationVerificationError: If verification fails critically
        """
        pass

    @abstractmethod
    def get_attestation_key(self) -> tuple[bytes, str]:
        """
        Get the attestation key public key and ID.

        Returns:
            Tuple of (public_key_bytes, key_id)
        """
        pass

    @abstractmethod
    def get_endorsement_key_certificate(self) -> Optional[bytes]:
        """
        Get the endorsement key certificate (if available).

        The EK certificate is signed by the TPM manufacturer and
        provides a chain of trust to the hardware.

        Returns:
            Certificate bytes in PEM format, or None if not available
        """
        pass

    def create_binding(self, data: bytes, quote: AttestationQuote) -> bytes:
        """
        Create a cryptographic binding between data and a quote.

        This proves that the data was present when the quote was generated.

        Args:
            data: Data to bind
            quote: Attestation quote

        Returns:
            Binding bytes (typically a hash)
        """
        binding_input = data + quote.quote_data + quote.signature
        return hashlib.sha256(binding_input).digest()

    def verify_binding(
        self,
        data: bytes,
        quote: AttestationQuote,
        expected_binding: bytes,
    ) -> bool:
        """
        Verify a data binding against a quote.

        Args:
            data: Data that was supposedly bound
            quote: Attestation quote
            expected_binding: Expected binding value

        Returns:
            True if binding is valid
        """
        computed = self.create_binding(data, quote)
        return computed == expected_binding

    @abstractmethod
    def extend_pcr(self, pcr_index: int, data: bytes) -> bytes:
        """
        Extend a PCR with new data.

        Args:
            pcr_index: PCR index to extend
            data: Data to extend with

        Returns:
            New PCR value

        Raises:
            AttestationError: If PCR extension fails
        """
        pass

    @abstractmethod
    def seal_data(
        self,
        data: bytes,
        pcr_policy: Optional[dict[int, bytes]] = None,
    ) -> bytes:
        """
        Seal data to the current platform state.

        Sealed data can only be unsealed when PCRs match the policy.

        Args:
            data: Data to seal
            pcr_policy: Optional PCR values that must match for unsealing

        Returns:
            Sealed data blob
        """
        pass

    @abstractmethod
    def unseal_data(self, sealed_blob: bytes) -> bytes:
        """
        Unseal previously sealed data.

        Args:
            sealed_blob: Sealed data blob

        Returns:
            Original data

        Raises:
            AttestationError: If unsealing fails (PCRs don't match)
        """
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """
        Check attestation provider health.

        Returns:
            Dict with status and diagnostics
        """
        pass
