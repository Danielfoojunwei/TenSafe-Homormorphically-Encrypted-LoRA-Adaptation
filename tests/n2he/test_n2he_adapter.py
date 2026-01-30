"""
Tests for N2HE encrypted LoRA adapter runtime.

Tests the encrypted adapter computation, key management integration,
and audit record generation.
"""

import numpy as np
import pytest

from tensorguard.n2he.adapter import (
    AdapterEncryptionConfig,
    AdapterMode,
    EncryptedLoRAAdapter,
    EncryptedLoRARuntime,
    create_encrypted_runtime,
)
from tensorguard.n2he.keys import HEKeyManager


class TestAdapterEncryptionConfig:
    """Tests for adapter encryption configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = AdapterEncryptionConfig()
        assert config.mode == AdapterMode.ENCRYPTED
        assert config.rank == 16
        assert config.alpha == 32.0
        assert len(config.target_modules) == 4

    def test_scaling_calculation(self):
        """Test LoRA scaling factor calculation."""
        config = AdapterEncryptionConfig(rank=8, alpha=16.0)
        assert config.get_scaling() == 2.0

    def test_serialization(self):
        """Test config serialization/deserialization."""
        config = AdapterEncryptionConfig(rank=32, alpha=64.0)
        data = config.to_dict()

        restored = AdapterEncryptionConfig.from_dict(data)
        assert restored.rank == config.rank
        assert restored.alpha == config.alpha
        assert restored.mode == config.mode


class TestEncryptedLoRAAdapter:
    """Tests for encrypted LoRA adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a test adapter."""
        rank = 8
        hidden_dim = 64
        lora_a = np.random.randn(rank, hidden_dim).astype(np.float32) * 0.02
        lora_b = np.random.randn(hidden_dim, rank).astype(np.float32) * 0.02

        return EncryptedLoRAAdapter(
            adapter_id="test-adapter",
            module_name="model.layers.0.self_attn.q_proj",
            lora_a=lora_a,
            lora_b=lora_b,
            rank=rank,
            alpha=16.0,
            scaling=2.0,
        )

    def test_adapter_creation(self, adapter):
        """Test adapter creation."""
        assert adapter.adapter_id == "test-adapter"
        assert adapter.rank == 8
        assert adapter.in_features == 64
        assert adapter.out_features == 64

    def test_content_hash(self, adapter):
        """Test content hash generation."""
        assert adapter.content_hash is not None
        assert adapter.content_hash.startswith("sha256:")

    def test_forward_plaintext(self, adapter):
        """Test plaintext forward pass."""
        batch = np.random.randn(2, 10, 64).astype(np.float32)
        delta = adapter.forward_plaintext(batch)

        assert delta.shape == (2, 10, 64)

    def test_forward_plaintext_scaling(self, adapter):
        """Test that scaling is applied correctly."""
        x = np.ones((1, 1, 64), dtype=np.float32)
        delta_scaled = adapter.forward_plaintext(x)

        # Compute without scaling for comparison
        intermediate = np.matmul(x, adapter.lora_a.T)
        delta_unscaled = np.matmul(intermediate, adapter.lora_b.T)

        np.testing.assert_allclose(
            delta_scaled,
            adapter.scaling * delta_unscaled,
            rtol=1e-5,
        )


class TestEncryptedLoRARuntime:
    """Tests for encrypted LoRA runtime."""

    @pytest.fixture
    def runtime_and_bundle(self):
        """Create a runtime with key bundle."""
        return create_encrypted_runtime(
            rank=8,
            alpha=16.0,
            tenant_id="test-tenant",
        )

    def test_create_runtime(self, runtime_and_bundle):
        """Test runtime creation."""
        runtime, bundle = runtime_and_bundle
        assert runtime is not None
        assert bundle is not None
        assert runtime.config.key_bundle_id == bundle.bundle_id

    def test_register_adapter(self, runtime_and_bundle):
        """Test adapter registration."""
        runtime, _ = runtime_and_bundle

        lora_a = np.random.randn(8, 64).astype(np.float32)
        lora_b = np.random.randn(64, 8).astype(np.float32)

        adapter = runtime.register_adapter(
            adapter_id="q_proj",
            module_name="model.layers.0.self_attn.q_proj",
            lora_a=lora_a,
            lora_b=lora_b,
        )

        assert adapter is not None
        assert "q_proj" in runtime._adapters

    def test_encrypt_activation(self, runtime_and_bundle):
        """Test activation encryption."""
        runtime, bundle = runtime_and_bundle

        activation = np.random.randn(1, 10, 64).astype(np.float32)
        encrypted = runtime.encrypt_activation(activation)

        assert encrypted is not None
        assert encrypted.batch_size == 1
        assert encrypted.key_bundle_id == bundle.bundle_id

    def test_compute_delta(self, runtime_and_bundle):
        """Test encrypted delta computation."""
        runtime, bundle = runtime_and_bundle

        # Register adapter
        lora_a = np.random.randn(8, 64).astype(np.float32)
        lora_b = np.random.randn(64, 8).astype(np.float32)
        runtime.register_adapter(
            adapter_id="test",
            module_name="test.module",
            lora_a=lora_a,
            lora_b=lora_b,
        )

        # Encrypt and compute
        activation = np.random.randn(1, 10, 64).astype(np.float32)
        encrypted = runtime.encrypt_activation(activation)
        delta = runtime.compute_delta(encrypted, "test")

        assert delta is not None
        assert delta.adapter_id == "test"
        assert delta.computation_time_ms > 0

    def test_decrypt_delta(self, runtime_and_bundle):
        """Test delta decryption."""
        runtime, bundle = runtime_and_bundle

        # Register adapter and compute delta
        lora_a = np.random.randn(8, 64).astype(np.float32)
        lora_b = np.random.randn(64, 8).astype(np.float32)
        runtime.register_adapter(
            adapter_id="test",
            module_name="test.module",
            lora_a=lora_a,
            lora_b=lora_b,
        )

        activation = np.random.randn(1, 10, 64).astype(np.float32)
        encrypted = runtime.encrypt_activation(activation)
        delta = runtime.compute_delta(encrypted, "test")

        # Decrypt
        decrypted = runtime.decrypt_delta(delta)
        assert decrypted is not None

    def test_forward_multiple_adapters(self, runtime_and_bundle):
        """Test forward pass with multiple adapters."""
        runtime, _ = runtime_and_bundle

        # Register multiple adapters
        for name in ["q_proj", "k_proj", "v_proj"]:
            lora_a = np.random.randn(8, 64).astype(np.float32)
            lora_b = np.random.randn(64, 8).astype(np.float32)
            runtime.register_adapter(
                adapter_id=name,
                module_name=f"test.{name}",
                lora_a=lora_a,
                lora_b=lora_b,
            )

        activation = np.random.randn(1, 10, 64).astype(np.float32)
        encrypted = runtime.encrypt_activation(activation)
        deltas = runtime.forward(encrypted)

        assert len(deltas) == 3

    def test_metrics(self, runtime_and_bundle):
        """Test runtime metrics."""
        runtime, _ = runtime_and_bundle

        # Register and compute
        lora_a = np.random.randn(8, 64).astype(np.float32)
        lora_b = np.random.randn(64, 8).astype(np.float32)
        runtime.register_adapter(
            adapter_id="test",
            module_name="test.module",
            lora_a=lora_a,
            lora_b=lora_b,
        )

        activation = np.random.randn(1, 10, 64).astype(np.float32)
        encrypted = runtime.encrypt_activation(activation)
        runtime.compute_delta(encrypted, "test")

        metrics = runtime.get_metrics()
        assert metrics["operations_count"] > 0
        assert metrics["adapters_registered"] == 1

    def test_audit_record(self, runtime_and_bundle):
        """Test audit record generation."""
        runtime, _ = runtime_and_bundle

        activation = np.random.randn(1, 10, 64).astype(np.float32)
        encrypted = runtime.encrypt_activation(activation)

        record = runtime.get_audit_record(
            operation="encrypt",
            encrypted_activation=encrypted,
        )

        assert record["operation"] == "encrypt"
        assert "input_metadata" in record
        assert record["input_metadata"]["batch_size"] == 1


class TestHEKeyManagerIntegration:
    """Tests for HE key manager integration with adapter."""

    def test_key_bundle_generation(self):
        """Test key bundle generation for adapter."""
        key_manager = HEKeyManager()
        bundle = key_manager.generate_key_bundle(tenant_id="test")

        assert bundle is not None
        assert bundle.public_key is not None
        assert bundle.evaluation_key is not None
        assert bundle.secret_key is not None

    def test_key_bundle_in_runtime(self):
        """Test key bundle usage in runtime."""
        key_manager = HEKeyManager()
        bundle = key_manager.generate_key_bundle(tenant_id="test")

        config = AdapterEncryptionConfig(
            key_bundle_id=bundle.bundle_id,
        )

        runtime = EncryptedLoRARuntime(
            config=config,
            key_manager=key_manager,
        )

        assert runtime._context is not None

    def test_manifest_claims(self):
        """Test manifest claims generation from bundle."""
        key_manager = HEKeyManager()
        bundle = key_manager.generate_key_bundle(tenant_id="test")

        claims = bundle.to_manifest_claims()

        assert claims["mode"] == "n2he"
        assert claims["provider"] == "n2he"
        assert "scheme_params_hash" in claims
        assert "public_key_fingerprint" in claims
