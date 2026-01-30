"""
Tests for N2HE private inference mode.

Tests encrypted batch processing, private forward pass, and sample generation.
"""

import numpy as np
import pytest

from tensorguard.n2he.inference import (
    PrivateInferenceConfig,
    PrivateInferenceProfile,
    EncryptedBatch,
    EncryptedOutput,
    create_private_inference_mode,
)


class TestPrivateInferenceConfig:
    """Tests for private inference configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = PrivateInferenceConfig()
        assert config.profile == PrivateInferenceProfile.ENCRYPTED_INPUT
        assert config.max_seq_len == 128
        assert config.max_layers_encrypted == 4

    def test_serialization(self):
        """Test config serialization."""
        config = PrivateInferenceConfig(
            profile=PrivateInferenceProfile.FULL_ENCRYPTED,
            max_layers_encrypted=8,
        )
        data = config.to_dict()

        assert data["profile"] == "full_encrypted"
        assert data["max_layers_encrypted"] == 8


class TestPrivateInferenceMode:
    """Tests for private inference mode."""

    @pytest.fixture
    def mode_and_bundle(self):
        """Create a private inference mode with bundle."""
        return create_private_inference_mode(
            profile="encrypted_input",
            hidden_dim=256,
            max_layers_encrypted=2,
            tenant_id="test-tenant",
        )

    def test_create_mode(self, mode_and_bundle):
        """Test mode creation."""
        mode, bundle = mode_and_bundle
        assert mode is not None
        assert bundle is not None
        assert mode.config.key_bundle_id == bundle.bundle_id

    def test_encrypt_batch(self, mode_and_bundle):
        """Test batch encryption."""
        mode, bundle = mode_and_bundle

        token_ids = [[1, 2, 3, 4, 5], [6, 7, 8]]
        encrypted = mode.encrypt_batch(token_ids)

        assert isinstance(encrypted, EncryptedBatch)
        assert encrypted.batch_size == 2
        assert encrypted.seq_len == 5  # max length
        assert len(encrypted.encrypted_embeddings) == 5

    def test_private_forward(self, mode_and_bundle):
        """Test private forward pass."""
        mode, _ = mode_and_bundle

        token_ids = [[1, 2, 3, 4, 5]]
        encrypted_batch = mode.encrypt_batch(token_ids)
        output = mode.private_forward(encrypted_batch)

        assert isinstance(output, EncryptedOutput)
        assert output.layers_processed > 0
        assert output.computation_time_ms > 0

    def test_decrypt_output(self, mode_and_bundle):
        """Test output decryption."""
        mode, _ = mode_and_bundle

        token_ids = [[1, 2, 3]]
        encrypted_batch = mode.encrypt_batch(token_ids)
        encrypted_output = mode.private_forward(encrypted_batch)

        decrypted = mode.decrypt_output(encrypted_output)
        assert decrypted is not None

    def test_private_sample(self, mode_and_bundle):
        """Test private sample generation."""
        mode, _ = mode_and_bundle

        prompts = ["Hello world", "How are you"]
        results = mode.private_sample(prompts)

        assert len(results) == 2
        for result in results:
            assert result.privacy_preserved
            assert result.encrypted_output is not None

    def test_metrics(self, mode_and_bundle):
        """Test inference mode metrics."""
        mode, _ = mode_and_bundle

        # Run some operations
        token_ids = [[1, 2, 3]]
        encrypted = mode.encrypt_batch(token_ids)
        mode.private_forward(encrypted)

        metrics = mode.get_metrics()
        assert metrics["samples_processed"] > 0
        assert metrics["total_encrypted_ops"] > 0

    def test_audit_record(self, mode_and_bundle):
        """Test audit record generation."""
        mode, _ = mode_and_bundle

        token_ids = [[1, 2, 3]]
        encrypted_batch = mode.encrypt_batch(token_ids)
        encrypted_output = mode.private_forward(encrypted_batch)

        record = mode.get_audit_record(
            encrypted_batch=encrypted_batch,
            encrypted_output=encrypted_output,
        )

        assert record["operation"] == "private_inference"
        assert "input_metadata" in record
        assert "output_metadata" in record


class TestEncryptedBatch:
    """Tests for encrypted batch."""

    def test_metadata(self):
        """Test batch metadata."""
        from tensorguard.n2he.core import create_context

        ctx = create_context()
        ctx.generate_keys()

        ct = ctx.encrypt(np.array([1], dtype=np.int64))

        batch = EncryptedBatch(
            batch_id="test-batch",
            encrypted_embeddings=[ct, ct, ct],
            batch_size=1,
            hidden_dim=64,
            key_bundle_id="test-bundle",
        )

        metadata = batch.get_metadata()
        assert metadata["batch_id"] == "test-batch"
        assert metadata["seq_len"] == 3


class TestEncryptedOutput:
    """Tests for encrypted output."""

    def test_metadata(self):
        """Test output metadata."""
        from tensorguard.n2he.core import create_context

        ctx = create_context()
        ctx.generate_keys()

        ct = ctx.encrypt(np.array([1], dtype=np.int64))

        output = EncryptedOutput(
            output_id="test-output",
            encrypted_logits=ct,
            key_bundle_id="test-bundle",
            layers_processed=4,
            operations_performed=100,
        )

        metadata = output.get_metadata()
        assert metadata["output_id"] == "test-output"
        assert metadata["layers_processed"] == 4
        assert metadata["operations_performed"] == 100
