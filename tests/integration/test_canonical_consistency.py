"""
Integration tests for canonical serialization consistency.

This test module verifies that canonical serialization is consistent across
all TenSafe components that create or verify hashes. This addresses the
identified gap where different modules were using different serialization
methods, leading to hash mismatches.

Tests cover:
1. Canonical serialization consistency between modules
2. Audit hash computation with all security-relevant fields
3. Evidence creation with canonical hashing
4. Cross-module hash verification
"""

import hashlib
from datetime import datetime
from unittest.mock import Mock

import pytest

# Import canonical serialization
from tensorguard.evidence.canonical import canonical_bytes, canonical_json, verify_canonical_hash


class TestCanonicalSerialization:
    """Tests for canonical serialization consistency."""

    def test_canonical_json_determinism(self):
        """Test that canonical_json produces deterministic output."""
        data = {
            "z_field": "last",
            "a_field": "first",
            "nested": {"b": 2, "a": 1},
            "list": [3, 1, 2],
        }

        # Multiple calls should produce identical output
        result1 = canonical_json(data)
        result2 = canonical_json(data)

        assert result1 == result2
        # Keys should be sorted
        assert result1.startswith('{"a_field"')
        # No extra whitespace
        assert " " not in result1.replace('"', "").replace(":", "").replace(",", "")

    def test_canonical_json_format(self):
        """Test that canonical_json uses correct format specifiers."""
        data = {"key": "value", "num": 123}
        result = canonical_json(data)

        # Should use compact separators
        assert '", "' not in result
        assert '": "' not in result
        assert '"key":"value"' in result

    def test_canonical_bytes_consistency(self):
        """Test that canonical_bytes produces consistent output."""
        data = {"test": "data", "number": 42}

        bytes1 = canonical_bytes(data)
        bytes2 = canonical_bytes(data)

        assert bytes1 == bytes2
        # Hash should be deterministic
        hash1 = hashlib.sha256(bytes1).hexdigest()
        hash2 = hashlib.sha256(bytes2).hexdigest()
        assert hash1 == hash2

    def test_verify_canonical_hash(self):
        """Test hash verification works correctly."""
        data = {"test": "data"}
        expected_hash = hashlib.sha256(canonical_bytes(data)).hexdigest()

        assert verify_canonical_hash(data, expected_hash) is True
        assert verify_canonical_hash(data, "wrong_hash") is False

    def test_canonical_json_ensure_ascii(self):
        """Test that canonical_json ensures ASCII output."""
        data = {"unicode": "\u00e9\u00e0\u00fc"}  # éàü
        result = canonical_json(data)

        # All characters should be ASCII
        assert result.isascii()
        # Unicode should be escaped
        assert "\\u" in result


class TestAuditHashConsistency:
    """Tests for audit hash computation consistency."""

    def test_audit_hash_includes_all_fields(self):
        """Test that audit hash includes all security-relevant fields."""
        from tensorguard.platform.tg_tinker_api.audit import AuditLogger

        logger = AuditLogger()

        # Create entry with all fields
        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-123",
            operation="forward_backward",
            request_hash="sha256:abc123",
            request_size_bytes=1024,
            artifact_ids_produced=["art-1"],
            artifact_ids_consumed=["art-0"],
            success=True,
            dp_metrics={"epsilon": 1.0, "delta": 1e-5},
        )

        # Verify hash was computed
        assert entry.record_hash.startswith("sha256:")

        # Verify chain integrity
        assert logger.verify_chain() is True

    def test_audit_hash_detects_field_modification(self):
        """Test that modifying fields breaks hash verification."""
        from tensorguard.platform.tg_tinker_api.audit import AuditLogger

        logger = AuditLogger()

        # Create entry
        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-123",
            operation="forward_backward",
            request_hash="sha256:abc123",
            request_size_bytes=1024,
            artifact_ids_consumed=["art-consumed"],
            dp_metrics={"epsilon": 1.0},
        )

        # Tamper with artifact_ids_consumed (now included in hash)
        original_consumed = entry.artifact_ids_consumed
        entry.artifact_ids_consumed = ["tampered"]

        # Chain should now be invalid
        assert logger.verify_chain() is False

        # Restore and verify chain is valid again
        entry.artifact_ids_consumed = original_consumed

    def test_audit_hash_uses_canonical_json(self):
        """Test that audit hash uses canonical JSON format."""
        from tensorguard.platform.tg_tinker_api.audit import AuditLogger

        logger = AuditLogger()

        # Create entry
        entry = logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-123",
            operation="test",
            request_hash="sha256:test",
            request_size_bytes=100,
        )

        # Recompute hash manually using canonical_json
        expected_data = {
            "entry_id": entry.id,
            "tenant_id": entry.tenant_id,
            "training_client_id": entry.training_client_id,
            "operation": entry.operation,
            "request_hash": entry.request_hash,
            "request_size_bytes": entry.request_size_bytes,
            "artifact_ids_produced": sorted(entry.artifact_ids_produced),
            "artifact_ids_consumed": sorted(entry.artifact_ids_consumed),
            "started_at": entry.started_at.isoformat(),
            "completed_at": entry.completed_at.isoformat() if entry.completed_at else None,
            "success": entry.success,
            "prev_hash": entry.prev_hash,
        }

        canonical_str = canonical_json(expected_data)
        expected_hash = f"sha256:{hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()}"

        assert entry.record_hash == expected_hash


class TestEvidenceCanonicalConsistency:
    """Tests for evidence creation with canonical hashing."""

    def test_evidence_includes_canonical_hash(self):
        """Test that evidence includes a canonical hash."""
        from unittest.mock import Mock

        from tensorguard.platform.tg_tinker_api.tgsp_bridge import TinkerTGSPBridge

        # Create mock artifact store and audit logger
        artifact_store = Mock()
        audit_logger = Mock()
        audit_logger.get_logs.return_value = []

        bridge = TinkerTGSPBridge(artifact_store, audit_logger)

        # Create mock artifact
        artifact = Mock()
        artifact.training_client_id = "tc-123"
        artifact.id = "art-456"
        artifact.artifact_type = "checkpoint"
        artifact.content_hash = "sha256:content"
        artifact.created_at = datetime.utcnow()
        artifact.tenant_id = "tenant-1"
        artifact.encryption_algorithm = "AES-256-GCM"
        artifact.metadata_json = None

        # Create evidence
        evidence = bridge._create_evidence(artifact)

        # Evidence should include schema version
        assert evidence["schema_version"] == "2.0"

        # Evidence should include canonical hash
        assert "evidence_hash" in evidence
        assert evidence["evidence_hash"].startswith("sha256:")

    def test_evidence_dp_validation(self):
        """Test that evidence validates DP parameters."""
        from tensorguard.platform.tg_tinker_api.tgsp_bridge import TinkerTGSPBridge

        artifact_store = Mock()
        audit_logger = Mock()
        audit_logger.get_logs.return_value = []

        bridge = TinkerTGSPBridge(artifact_store, audit_logger)

        artifact = Mock()
        artifact.training_client_id = "tc-123"
        artifact.id = "art-456"
        artifact.artifact_type = "checkpoint"
        artifact.content_hash = "sha256:content"
        artifact.created_at = datetime.utcnow()
        artifact.tenant_id = "tenant-1"
        artifact.encryption_algorithm = "AES-256-GCM"
        artifact.metadata_json = None

        # Invalid epsilon should raise
        with pytest.raises(ValueError, match="Invalid epsilon"):
            bridge._create_evidence(artifact, dp_certificate={"total_epsilon": -1.0})

        # Invalid delta should raise
        with pytest.raises(ValueError, match="Invalid delta"):
            bridge._create_evidence(artifact, dp_certificate={"total_delta": 2.0})

        # Valid values should work
        evidence = bridge._create_evidence(
            artifact,
            dp_certificate={"total_epsilon": 8.0, "total_delta": 1e-5}
        )
        assert evidence["privacy"]["epsilon"] == 8.0

    def test_evidence_audit_chain_canonical_hash(self):
        """Test that evidence includes canonical audit chain hash."""
        from tensorguard.platform.tg_tinker_api.tgsp_bridge import TinkerTGSPBridge

        artifact_store = Mock()
        audit_logger = Mock()

        # Create mock audit logs
        log1 = Mock()
        log1.sequence = 1
        log1.record_hash = "sha256:hash1"
        log1.operation = "forward_backward"

        log2 = Mock()
        log2.sequence = 2
        log2.record_hash = "sha256:hash2"
        log2.operation = "optim_step"

        audit_logger.get_logs.return_value = [log1, log2]

        bridge = TinkerTGSPBridge(artifact_store, audit_logger)

        artifact = Mock()
        artifact.training_client_id = "tc-123"
        artifact.id = "art-456"
        artifact.artifact_type = "checkpoint"
        artifact.content_hash = "sha256:content"
        artifact.created_at = datetime.utcnow()
        artifact.tenant_id = "tenant-1"
        artifact.encryption_algorithm = "AES-256-GCM"
        artifact.metadata_json = None

        evidence = bridge._create_evidence(artifact)

        # Should have chain canonical hash
        assert "chain_canonical_hash" in evidence["audit_chain"]
        assert evidence["audit_chain"]["chain_canonical_hash"] != "none"


class TestCrossModuleHashVerification:
    """Tests for hash verification across modules."""

    def test_audit_to_evidence_hash_consistency(self):
        """Test that audit hashes can be verified in evidence context."""
        from tensorguard.platform.tg_tinker_api.audit import AuditLogger
        from tensorguard.platform.tg_tinker_api.tgsp_bridge import TinkerTGSPBridge

        # Create audit logs
        audit_logger = AuditLogger()
        entry = audit_logger.log_operation(
            tenant_id="tenant-1",
            training_client_id="tc-123",
            operation="forward_backward",
            request_hash="sha256:request",
            request_size_bytes=1024,
        )

        # Create evidence bridge
        artifact_store = Mock()
        bridge = TinkerTGSPBridge(artifact_store, audit_logger)

        artifact = Mock()
        artifact.training_client_id = "tc-123"
        artifact.id = "art-456"
        artifact.artifact_type = "checkpoint"
        artifact.content_hash = "sha256:content"
        artifact.created_at = datetime.utcnow()
        artifact.tenant_id = "tenant-1"
        artifact.encryption_algorithm = "AES-256-GCM"
        artifact.metadata_json = None

        evidence = bridge._create_evidence(artifact)

        # Audit chain should reference the entry
        assert evidence["audit_chain"]["first_entry_hash"] == entry.record_hash
        assert evidence["audit_chain"]["last_entry_hash"] == entry.record_hash
        assert evidence["audit_chain"]["total_entries"] == 1

    def test_evidence_store_canonical_compatibility(self):
        """Test that evidence store uses same canonical format."""
        from tensorguard.evidence.store import EvidenceStore

        store = EvidenceStore()

        # Add a record
        record = store.add_record(
            record_type="test",
            data={"key": "value", "nested": {"a": 1, "b": 2}},
        )

        # Record hash should use canonical bytes
        expected_hash = hashlib.sha256(
            canonical_bytes({
                "record_id": record.record_id,
                "record_type": record.record_type,
                "timestamp": record.timestamp,
                "data_hash": record.data_hash,
                "prev_hash": record.prev_hash,
            })
        ).hexdigest()

        assert record.compute_record_hash() == expected_hash

        # Chain verification should work
        assert store.verify_chain_integrity() is True


class TestHELoRAIntegrationGaps:
    """Tests for HE-LoRA integration gaps."""

    def test_noise_budget_validation(self):
        """Test that noise budget is validated during computation."""
        from unittest.mock import Mock

        import numpy as np

        from tensorguard.n2he.adapter import (
            AdapterEncryptionConfig,
            AdapterMode,
            EncryptedActivation,
            EncryptedLoRARuntime,
        )

        config = AdapterEncryptionConfig(
            mode=AdapterMode.ENCRYPTED,
            noise_budget_threshold=10.0,
        )

        runtime = EncryptedLoRARuntime(config=config)

        # Register an adapter
        runtime.register_adapter(
            adapter_id="test-adapter",
            module_name="test_layer",
            lora_a=np.random.randn(16, 64).astype(np.float32),
            lora_b=np.random.randn(64, 16).astype(np.float32),
        )

        # Create mock encrypted activation with low noise budget
        mock_ciphertext = Mock()
        mock_ciphertext.noise_budget = 5.0  # Below threshold

        encrypted_activation = EncryptedActivation(
            ciphertext=mock_ciphertext,
            batch_size=1,
            seq_len=8,
            hidden_dim=64,
            key_bundle_id="test-bundle",
        )

        # Mock the context
        runtime._context = Mock()

        # Should raise due to low noise budget
        with pytest.raises(ValueError, match="Noise budget"):
            runtime.compute_delta(encrypted_activation, "test-adapter")

    def test_adapter_manifest_claims(self):
        """Test that manifest claims match actual adapter capabilities."""
        import numpy as np

        from tensorguard.n2he.adapter import (
            AdapterEncryptionConfig,
            AdapterMode,
            EncryptedLoRARuntime,
            HESchemeParams,
        )

        he_params = HESchemeParams.default_lora_params()

        config = AdapterEncryptionConfig(
            mode=AdapterMode.ENCRYPTED,
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "v_proj"],
            he_params=he_params,
            key_bundle_id="test-bundle",
        )

        runtime = EncryptedLoRARuntime(config=config)

        # Register adapters
        for module in config.target_modules:
            runtime.register_adapter(
                adapter_id=f"adapter-{module}",
                module_name=f"model.{module}",
                lora_a=np.random.randn(16, 64).astype(np.float32),
                lora_b=np.random.randn(64, 16).astype(np.float32),
            )

        # Get manifest claims
        claims = runtime.get_manifest_claims()

        # Claims should match config
        assert claims["he_scheme_config"]["rank"] == config.rank
        assert claims["he_scheme_config"]["alpha"] == config.alpha
        assert claims["adapters_registered"] == len(config.target_modules)
        assert set(claims["adapter_ids"]) == {f"adapter-{m}" for m in config.target_modules}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
