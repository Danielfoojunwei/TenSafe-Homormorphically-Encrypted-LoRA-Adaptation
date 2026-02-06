"""
Unit tests for Privacy Receipt Generator.

Tests receipt generation, verification, and hash chain integrity.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorguard.confidential.receipt import (
    PrivacyReceipt,
    PrivacyReceiptGenerator,
    TEEAttestationClaim,
    AdapterProvenanceClaim,
    HEExecutionClaim,
)


@pytest.fixture
def generator():
    return PrivacyReceiptGenerator(
        tee_platform="intel-tdx",
        attestation_quote_hash="abc123",
    )


class TestReceiptGeneration:

    def test_generate_basic_receipt(self, generator):
        receipt = generator.generate(session_id="cs-test123")
        assert receipt.receipt_id.startswith("pr-")
        assert receipt.session_id == "cs-test123"
        assert receipt.audit_hash != ""
        assert receipt.timestamp.endswith("Z")

    def test_receipt_tee_claim(self, generator):
        receipt = generator.generate(session_id="cs-test")
        claim = receipt.tee_attestation
        assert claim.platform == "intel-tdx"
        assert claim.quote_hash == "abc123"
        assert claim.verified is True

    def test_receipt_he_claim(self, generator):
        receipt = generator.generate(
            session_id="cs-test",
            he_mode="HE_ONLY",
            he_backend="CKKS-MOAI",
            adapter_encrypted=True,
            he_metrics={"rotations": 0, "operations": 42, "compute_time_ms": 5.0},
        )
        claim = receipt.he_execution
        assert claim.mode == "HE_ONLY"
        assert claim.backend == "CKKS-MOAI"
        assert claim.adapter_encrypted is True
        assert claim.rotations == 0
        assert claim.operations == 42

    def test_receipt_adapter_claim(self, generator):
        receipt = generator.generate(
            session_id="cs-test",
            tssp_hash="sha256:deadbeef",
            dp_certificate={"epsilon": 8.0, "delta": 1e-5},
            adapter_id="adapter-001",
        )
        claim = receipt.adapter_provenance
        assert claim.tssp_package_hash == "sha256:deadbeef"
        assert claim.signature_algorithm == "ed25519+dilithium3"
        assert claim.dp_certificate["epsilon"] == 8.0
        assert claim.adapter_id == "adapter-001"

    def test_receipt_disabled_he(self, generator):
        receipt = generator.generate(
            session_id="cs-test",
            he_mode="DISABLED",
        )
        assert receipt.he_execution.mode == "DISABLED"
        assert receipt.he_execution.adapter_encrypted is False


class TestReceiptVerification:

    def test_receipt_self_verify(self, generator):
        receipt = generator.generate(session_id="cs-test")
        assert receipt.verify() is True

    def test_receipt_tampered_audit_hash(self, generator):
        receipt = generator.generate(session_id="cs-test")
        receipt.audit_hash = "tampered"
        assert receipt.verify() is False

    def test_receipt_compute_verification_hash(self, generator):
        receipt = generator.generate(session_id="cs-test")
        recomputed = receipt.compute_verification_hash()
        assert recomputed == receipt.audit_hash

    def test_receipt_verify_deterministic(self, generator):
        """Same claims should produce same audit hash."""
        r1 = generator.generate(session_id="cs-test", he_mode="HE_ONLY")
        # Manually recompute
        assert r1.compute_verification_hash() == r1.audit_hash


class TestReceiptHashChain:

    def test_hash_chain(self, generator):
        r1 = generator.generate(session_id="cs-1")
        r2 = generator.generate(session_id="cs-2")
        r3 = generator.generate(session_id="cs-3")

        assert r1.previous_audit_hash is None
        assert r2.previous_audit_hash == r1.audit_hash
        assert r3.previous_audit_hash == r2.audit_hash

    def test_hash_chain_integrity(self, generator):
        receipts = []
        for i in range(5):
            r = generator.generate(session_id=f"cs-{i}")
            receipts.append(r)

        # Verify chain
        for i in range(1, len(receipts)):
            assert receipts[i].previous_audit_hash == receipts[i - 1].audit_hash

        # All should self-verify
        for r in receipts:
            assert r.verify() is True


class TestReceiptSerialization:

    def test_to_dict(self, generator):
        receipt = generator.generate(
            session_id="cs-test",
            he_mode="HE_ONLY",
            he_backend="CKKS-MOAI",
            tssp_hash="sha256:abc",
            latency_ms=42.5,
        )
        d = receipt.to_dict()

        assert d["receipt_id"].startswith("pr-")
        assert d["session_id"] == "cs-test"
        assert d["tee_attestation"]["platform"] == "intel-tdx"
        assert d["he_execution"]["mode"] == "HE_ONLY"
        assert d["adapter_provenance"]["tssp_package_hash"] == "sha256:abc"
        assert d["audit_hash"] != ""
        assert d["total_latency_ms"] == 42.5


class TestTEEAttestationClaim:

    def test_to_dict(self):
        claim = TEEAttestationClaim(
            platform="amd-sev-snp",
            quote_hash="hash123",
            verified=True,
        )
        d = claim.to_dict()
        assert d["platform"] == "amd-sev-snp"
        assert "gpu_attestation" not in d

    def test_to_dict_with_gpu(self):
        claim = TEEAttestationClaim(
            platform="intel-tdx",
            quote_hash="hash",
            gpu_attestation="nras-token-hash",
            verified=True,
        )
        d = claim.to_dict()
        assert d["gpu_attestation"] == "nras-token-hash"


class TestAdapterProvenanceClaim:

    def test_empty_claim(self):
        claim = AdapterProvenanceClaim()
        d = claim.to_dict()
        assert d == {}

    def test_full_claim(self):
        claim = AdapterProvenanceClaim(
            tssp_package_hash="sha256:abc",
            signature_algorithm="ed25519+dilithium3",
            dp_certificate={"epsilon": 8.0, "delta": 1e-5},
            adapter_id="my-adapter",
        )
        d = claim.to_dict()
        assert len(d) == 4
