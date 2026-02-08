"""
Unit tests for Intel TDX Attestation Provider.

Tests TDX quote generation, verification, and RTMR operations
in simulation mode.
"""

import hashlib
import os
import sys
from datetime import timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorguard.attestation.provider import (
    AttestationError,
    AttestationType,
    QuoteType,
    VerificationPolicy,
)
from tensorguard.attestation.tdx import (
    RTMR_APPLICATION,
    RTMR_FIRMWARE,
    RTMR_OS_KERNEL,
    RTMR_USER,
    TDXAttestationProvider,
    TDXVerificationPolicy,
)


@pytest.fixture
def tdx_provider():
    """Create a simulated TDX provider."""
    return TDXAttestationProvider(use_simulation=True)


@pytest.fixture
def tdx_policy():
    """Create a basic TDX verification policy."""
    return TDXVerificationPolicy(
        policy_id="test-policy",
        name="Test TDX Policy",
        max_quote_age_seconds=300,
    )


class TestTDXProviderBasics:
    """Test TDX provider basic properties."""

    def test_attestation_type(self, tdx_provider):
        assert tdx_provider.attestation_type == AttestationType.SGX

    def test_is_available_simulation(self, tdx_provider):
        assert tdx_provider.is_available is True

    def test_is_available_no_hardware(self):
        provider = TDXAttestationProvider(
            device_path="/nonexistent/device", use_simulation=False
        )
        assert provider.is_available is False

    def test_health_check(self, tdx_provider):
        health = tdx_provider.health_check()
        assert health["provider"] == "tdx"
        assert health["available"] is True
        assert health["simulation"] is True

    def test_firmware_version_simulation(self, tdx_provider):
        version = tdx_provider._get_firmware_version()
        assert version == "sim-1.5.0"


class TestTDXQuoteGeneration:
    """Test TDX attestation quote generation."""

    def test_generate_quote_basic(self, tdx_provider):
        quote = tdx_provider.generate_quote()
        assert quote is not None
        assert quote.quote_id.startswith("tdx-")
        assert quote.quote_type == QuoteType.PLATFORM
        assert quote.attestation_type == AttestationType.SGX
        assert quote.nonce is not None
        assert len(quote.nonce) == 32

    def test_generate_quote_with_nonce(self, tdx_provider):
        nonce = os.urandom(32)
        quote = tdx_provider.generate_quote(nonce=nonce)
        assert quote.nonce == nonce

    def test_generate_quote_with_extra_data(self, tdx_provider):
        extra = b"test-binding-data"
        quote = tdx_provider.generate_quote(extra_data=extra)
        assert quote.extra_data == extra

    def test_generate_quote_has_rtmrs(self, tdx_provider):
        quote = tdx_provider.generate_quote()
        assert RTMR_FIRMWARE in quote.pcr_values
        assert RTMR_OS_KERNEL in quote.pcr_values
        assert RTMR_APPLICATION in quote.pcr_values
        assert RTMR_USER in quote.pcr_values
        # TDX RTMRs are 48 bytes (SHA-384)
        for val in quote.pcr_values.values():
            assert len(val) == 48

    def test_generate_quote_deterministic_measurements(self, tdx_provider):
        """Simulated measurements should be deterministic for same input."""
        q1 = tdx_provider.generate_quote(nonce=b"\x00" * 32)
        q2 = tdx_provider.generate_quote(nonce=b"\x00" * 32)
        # MRTD should be the same (same simulated TD)
        assert q1.pcr_values[RTMR_FIRMWARE] == q2.pcr_values[RTMR_FIRMWARE]

    def test_generate_quote_has_firmware_version(self, tdx_provider):
        quote = tdx_provider.generate_quote()
        assert quote.firmware_version is not None
        assert quote.firmware_version == "sim-1.5.0"

    def test_generate_quote_has_attestation_key_id(self, tdx_provider):
        quote = tdx_provider.generate_quote()
        assert quote.attestation_key_id is not None
        assert quote.attestation_key_id.startswith("tdx-ecdsa-")

    def test_long_nonce_is_hashed(self, tdx_provider):
        """Nonces longer than 64 bytes should be hashed."""
        long_nonce = os.urandom(128)
        quote = tdx_provider.generate_quote(nonce=long_nonce)
        assert len(quote.nonce) == 32  # SHA-256 output


class TestTDXQuoteVerification:
    """Test TDX attestation quote verification."""

    def test_verify_valid_quote(self, tdx_provider, tdx_policy):
        nonce = os.urandom(32)
        quote = tdx_provider.generate_quote(nonce=nonce)
        result = tdx_provider.verify_quote(
            quote, tdx_policy, expected_nonce=nonce
        )
        assert result.verified is True
        assert result.signature_valid is True
        assert result.nonce_valid is True
        assert result.timestamp_valid is True
        assert len(result.failure_reasons) == 0

    def test_verify_nonce_mismatch(self, tdx_provider, tdx_policy):
        quote = tdx_provider.generate_quote(nonce=b"\x01" * 32)
        result = tdx_provider.verify_quote(
            quote, tdx_policy, expected_nonce=b"\x02" * 32
        )
        assert result.verified is False
        assert result.nonce_valid is False
        assert "Nonce mismatch" in result.failure_reasons

    def test_verify_expired_quote(self, tdx_provider):
        policy = TDXVerificationPolicy(
            policy_id="strict",
            name="Strict Policy",
            max_quote_age_seconds=1,
        )
        quote = tdx_provider.generate_quote()
        # Backdate quote timestamp to ensure expiry
        quote.timestamp = quote.timestamp - timedelta(seconds=5)
        result = tdx_provider.verify_quote(quote, policy)
        assert result.verified is False
        assert result.timestamp_valid is False

    def test_verify_rtmr_mismatch(self, tdx_provider):
        policy = TDXVerificationPolicy(
            policy_id="rtmr-check",
            name="RTMR Check Policy",
            expected_rtmrs={RTMR_FIRMWARE: b"\x00" * 48},
        )
        quote = tdx_provider.generate_quote()
        result = tdx_provider.verify_quote(quote, policy)
        assert result.verified is False
        assert result.pcr_match is False
        assert any("RTMR0 mismatch" in r for r in result.failure_reasons)

    def test_verify_mrtd_match(self, tdx_provider):
        """Verify quote with matching MRTD."""
        expected_mrtd = hashlib.sha384(b"SIM_MRTD_TENSAFE").digest()
        policy = TDXVerificationPolicy(
            policy_id="mrtd-check",
            name="MRTD Check",
            expected_mr_td=expected_mrtd,
        )
        quote = tdx_provider.generate_quote()
        result = tdx_provider.verify_quote(quote, policy)
        assert result.verified is True
        assert result.pcr_match is True

    def test_verify_mrtd_mismatch(self, tdx_provider):
        policy = TDXVerificationPolicy(
            policy_id="mrtd-check",
            name="MRTD Mismatch",
            expected_mr_td=b"\xff" * 48,
        )
        quote = tdx_provider.generate_quote()
        result = tdx_provider.verify_quote(quote, policy)
        assert result.verified is False
        assert "MRTD mismatch" in result.failure_reasons

    def test_verify_platform_info(self, tdx_provider, tdx_policy):
        quote = tdx_provider.generate_quote()
        result = tdx_provider.verify_quote(quote, tdx_policy)
        assert result.platform_info["tee_type"] == "TDX"
        assert result.platform_info["simulation"] is True
        assert "mr_td" in result.platform_info

    def test_verify_with_generic_policy(self, tdx_provider):
        """Verify using a generic VerificationPolicy (not TDX-specific)."""
        policy = VerificationPolicy(
            policy_id="generic",
            name="Generic",
            max_quote_age_seconds=300,
        )
        quote = tdx_provider.generate_quote()
        result = tdx_provider.verify_quote(quote, policy)
        assert result.verified is True


class TestTDXQuoteBody:
    """Test TDX quote body parsing."""

    def test_parse_quote_body(self, tdx_provider):
        quote = tdx_provider.generate_quote()
        body = tdx_provider._parse_quote_body(quote.quote_data)
        assert len(body.mr_td) == 48
        assert len(body.rtmr0) == 48
        assert len(body.rtmr1) == 48
        assert len(body.rtmr2) == 48
        assert len(body.rtmr3) == 48
        assert len(body.report_data) == 64

    def test_quote_body_to_dict(self, tdx_provider):
        quote = tdx_provider.generate_quote()
        body = tdx_provider._parse_quote_body(quote.quote_data)
        d = body.to_dict()
        assert "mr_td" in d
        assert "rtmr0" in d
        assert "report_data" in d


class TestTDXRTMRExtension:
    """Test RTMR extension operations."""

    def test_extend_rtmr2(self, tdx_provider):
        """Application RTMR2 can be extended."""
        result = tdx_provider.extend_pcr(RTMR_APPLICATION, b"test-data")
        assert len(result) == 48

    def test_extend_rtmr3(self, tdx_provider):
        """User RTMR3 can be extended."""
        result = tdx_provider.extend_pcr(RTMR_USER, b"user-measurement")
        assert len(result) == 48

    def test_extend_invalid_rtmr(self, tdx_provider):
        with pytest.raises(AttestationError, match="Invalid RTMR index"):
            tdx_provider.extend_pcr(5, b"data")


class TestTDXSealUnseal:
    """Test data sealing/unsealing."""

    def test_seal_unseal_roundtrip(self, tdx_provider):
        data = b"secret-adapter-key-material"
        sealed = tdx_provider.seal_data(data)
        assert sealed != data
        unsealed = tdx_provider.unseal_data(sealed)
        assert unsealed == data

    def test_seal_with_policy(self, tdx_provider):
        data = b"policy-bound-data"
        policy = {0: b"\x00" * 48}
        sealed = tdx_provider.seal_data(data, pcr_policy=policy)
        assert sealed != data


class TestTDXAttestationKey:
    """Test attestation key access."""

    def test_get_attestation_key_simulation(self, tdx_provider):
        key_bytes, key_id = tdx_provider.get_attestation_key()
        assert len(key_bytes) == 32
        assert key_id == "tdx-sim-ak"

    def test_get_endorsement_key(self, tdx_provider):
        assert tdx_provider.get_endorsement_key_certificate() is None
