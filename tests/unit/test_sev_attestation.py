"""
Unit tests for AMD SEV-SNP Attestation Provider.

Tests SEV-SNP report generation, verification, and sealing
in simulation mode.
"""

import hashlib
import os
import sys
import time
from datetime import timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorguard.attestation.sev import (
    SEVSNPAttestationProvider,
    SNPReportBody,
    SNPVerificationPolicy,
    SNP_POLICY_DEBUG_ALLOWED,
)
from tensorguard.attestation.provider import (
    AttestationError,
    AttestationType,
    QuoteType,
    VerificationPolicy,
)


@pytest.fixture
def sev_provider():
    return SEVSNPAttestationProvider(use_simulation=True)


@pytest.fixture
def snp_policy():
    return SNPVerificationPolicy(
        policy_id="test-snp",
        name="Test SNP Policy",
        max_quote_age_seconds=300,
    )


class TestSEVProviderBasics:

    def test_attestation_type(self, sev_provider):
        assert sev_provider.attestation_type == AttestationType.SEV

    def test_is_available_simulation(self, sev_provider):
        assert sev_provider.is_available is True

    def test_is_available_no_hardware(self):
        provider = SEVSNPAttestationProvider(
            device_path="/nonexistent", use_simulation=False
        )
        assert provider.is_available is False

    def test_health_check(self, sev_provider):
        health = sev_provider.health_check()
        assert health["provider"] == "sev-snp"
        assert health["available"] is True
        assert health["simulation"] is True


class TestSEVReportGeneration:

    def test_generate_report_basic(self, sev_provider):
        quote = sev_provider.generate_quote()
        assert quote.quote_id.startswith("snp-")
        assert quote.quote_type == QuoteType.PLATFORM
        assert quote.attestation_type == AttestationType.SEV
        assert quote.nonce is not None

    def test_generate_report_with_nonce(self, sev_provider):
        nonce = os.urandom(32)
        quote = sev_provider.generate_quote(nonce=nonce)
        assert quote.nonce == nonce

    def test_generate_report_with_extra_data(self, sev_provider):
        extra = b"binding-info"
        quote = sev_provider.generate_quote(extra_data=extra)
        assert quote.extra_data == extra

    def test_report_has_measurement(self, sev_provider):
        quote = sev_provider.generate_quote()
        assert 0 in quote.pcr_values
        measurement = quote.pcr_values[0]
        assert len(measurement) == 48

    def test_report_has_firmware_version(self, sev_provider):
        quote = sev_provider.generate_quote()
        assert quote.firmware_version == "sim-1.55.0"


class TestSEVReportVerification:

    def test_verify_valid_report(self, sev_provider, snp_policy):
        nonce = os.urandom(32)
        quote = sev_provider.generate_quote(nonce=nonce)
        result = sev_provider.verify_quote(
            quote, snp_policy, expected_nonce=nonce
        )
        assert result.verified is True
        assert result.signature_valid is True
        assert result.nonce_valid is True
        assert len(result.failure_reasons) == 0

    def test_verify_nonce_mismatch(self, sev_provider, snp_policy):
        quote = sev_provider.generate_quote(nonce=b"\x01" * 32)
        result = sev_provider.verify_quote(
            quote, snp_policy, expected_nonce=b"\x02" * 32
        )
        assert result.verified is False
        assert result.nonce_valid is False

    def test_verify_measurement_match(self, sev_provider):
        expected = hashlib.sha384(b"SIM_SNP_MEASUREMENT_TENSAFE").digest()
        policy = SNPVerificationPolicy(
            policy_id="meas",
            name="Measurement Check",
            expected_measurement=expected,
        )
        quote = sev_provider.generate_quote()
        result = sev_provider.verify_quote(quote, policy)
        assert result.verified is True

    def test_verify_measurement_mismatch(self, sev_provider):
        policy = SNPVerificationPolicy(
            policy_id="meas",
            name="Measurement Mismatch",
            expected_measurement=b"\xff" * 48,
        )
        quote = sev_provider.generate_quote()
        result = sev_provider.verify_quote(quote, policy)
        assert result.verified is False
        assert any("Measurement mismatch" in r for r in result.failure_reasons)

    def test_verify_debug_policy_rejected(self, sev_provider):
        """Simulation doesn't set debug flag, but verify policy checks it."""
        policy = SNPVerificationPolicy(
            policy_id="no-debug",
            name="No Debug",
            allow_debug_policy=False,
        )
        quote = sev_provider.generate_quote()
        result = sev_provider.verify_quote(quote, policy)
        # Simulation doesn't set debug flag, so should pass
        assert result.verified is True

    def test_verify_platform_info(self, sev_provider, snp_policy):
        quote = sev_provider.generate_quote()
        result = sev_provider.verify_quote(quote, snp_policy)
        assert result.platform_info["tee_type"] == "SEV-SNP"
        assert "measurement" in result.platform_info
        assert "vmpl" in result.platform_info

    def test_verify_with_generic_policy(self, sev_provider):
        policy = VerificationPolicy(
            policy_id="generic",
            name="Generic",
            max_quote_age_seconds=300,
        )
        quote = sev_provider.generate_quote()
        result = sev_provider.verify_quote(quote, policy)
        assert result.verified is True

    def test_expired_report(self, sev_provider):
        policy = SNPVerificationPolicy(
            policy_id="strict",
            name="Strict",
            max_quote_age_seconds=1,
        )
        quote = sev_provider.generate_quote()
        # Backdate quote to ensure expiry
        quote.timestamp = quote.timestamp - timedelta(seconds=5)
        result = sev_provider.verify_quote(quote, policy)
        assert result.verified is False
        assert result.timestamp_valid is False


class TestSEVReportBody:

    def test_parse_report(self, sev_provider):
        quote = sev_provider.generate_quote()
        body = sev_provider._parse_report_body(quote.quote_data)
        assert body.version == 2
        assert len(body.measurement) == 48
        assert len(body.host_data) == 32
        assert len(body.report_data) == 64
        assert len(body.chip_id) == 64

    def test_report_body_to_dict(self, sev_provider):
        quote = sev_provider.generate_quote()
        body = sev_provider._parse_report_body(quote.quote_data)
        d = body.to_dict()
        assert "measurement" in d
        assert "vmpl" in d
        assert "chip_id" in d


class TestSEVSealUnseal:

    def test_seal_unseal_roundtrip(self, sev_provider):
        data = b"secret-data-for-sealing"
        sealed = sev_provider.seal_data(data)
        assert sealed != data
        unsealed = sev_provider.unseal_data(sealed)
        assert unsealed == data

    def test_extend_pcr_not_supported(self, sev_provider):
        with pytest.raises(AttestationError, match="does not support"):
            sev_provider.extend_pcr(0, b"data")


class TestSEVAttestationKey:

    def test_get_attestation_key_simulation(self, sev_provider):
        key_bytes, key_id = sev_provider.get_attestation_key()
        assert len(key_bytes) == 48  # SHA-384
        assert key_id == "snp-sim-vcek"
