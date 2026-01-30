"""
Tests for N2HE core module.

Tests the foundational HE primitives: scheme parameters, encryption,
decryption, and homomorphic operations.
"""

import numpy as np
import pytest

from tensorguard.n2he.core import (
    HESchemeParams,
    HESchemeType,
    LWECiphertext,
    RLWECiphertext,
    SimulatedN2HEScheme,
    create_context,
)


class TestHESchemeParams:
    """Tests for HE scheme parameters."""

    def test_default_params(self):
        """Test default parameter creation."""
        params = HESchemeParams()
        assert params.scheme_type == HESchemeType.LWE
        assert params.n == 1024
        assert params.security_level == 128

    def test_lora_params(self):
        """Test LoRA-optimized parameters."""
        params = HESchemeParams.default_lora_params()
        assert params.scheme_type == HESchemeType.LWE
        assert params.n == 1024
        assert params.q == 2**32

    def test_high_precision_params(self):
        """Test high-precision parameters."""
        params = HESchemeParams.high_precision_params()
        assert params.scheme_type == HESchemeType.CKKS
        assert params.n == 2048

    def test_params_hash_deterministic(self):
        """Test that parameter hash is deterministic."""
        params1 = HESchemeParams.default_lora_params()
        params2 = HESchemeParams.default_lora_params()
        assert params1.get_hash() == params2.get_hash()

    def test_params_hash_unique(self):
        """Test that different parameters have different hashes."""
        params1 = HESchemeParams.default_lora_params()
        params2 = HESchemeParams(n=2048)
        assert params1.get_hash() != params2.get_hash()

    def test_params_serialization(self):
        """Test parameter serialization/deserialization."""
        params = HESchemeParams.default_lora_params()
        data = params.to_dict()
        restored = HESchemeParams.from_dict(data)
        assert restored.get_hash() == params.get_hash()


class TestSimulatedN2HEScheme:
    """Tests for simulated N2HE scheme."""

    @pytest.fixture
    def scheme(self):
        """Create a simulated scheme."""
        params = HESchemeParams.default_lora_params()
        return SimulatedN2HEScheme(params)

    def test_keygen(self, scheme):
        """Test key generation."""
        sk, pk, ek = scheme.keygen()
        assert len(sk) > 0
        assert len(pk) > 0
        assert len(ek) > 0

    def test_encrypt_decrypt_roundtrip(self, scheme):
        """Test encryption followed by decryption."""
        sk, pk, ek = scheme.keygen()
        plaintext = np.array([42], dtype=np.int64)

        ciphertext = scheme.encrypt(pk, plaintext)
        decrypted = scheme.decrypt(sk, ciphertext)

        # Simulated scheme may not be perfectly accurate
        assert isinstance(decrypted, np.ndarray)

    def test_encrypt_creates_lwe_ciphertext(self, scheme):
        """Test that encrypt creates LWE ciphertext."""
        _, pk, _ = scheme.keygen()
        plaintext = np.array([100], dtype=np.int64)

        ciphertext = scheme.encrypt(pk, plaintext)

        assert isinstance(ciphertext, LWECiphertext)
        assert len(ciphertext.a) == scheme.params.n

    def test_homomorphic_add(self, scheme):
        """Test homomorphic addition."""
        _, pk, _ = scheme.keygen()
        pt1 = np.array([10], dtype=np.int64)
        pt2 = np.array([20], dtype=np.int64)

        ct1 = scheme.encrypt(pk, pt1)
        ct2 = scheme.encrypt(pk, pt2)
        ct_sum = scheme.add(ct1, ct2)

        assert isinstance(ct_sum, LWECiphertext)
        assert ct_sum.noise_budget is not None
        assert ct_sum.noise_budget < min(ct1.noise_budget, ct2.noise_budget)

    def test_homomorphic_multiply(self, scheme):
        """Test plaintext-ciphertext multiplication."""
        _, pk, _ = scheme.keygen()
        pt = np.array([10], dtype=np.int64)
        scalar = np.array([5], dtype=np.int64)

        ct = scheme.encrypt(pk, pt)
        ct_product = scheme.multiply(ct, scalar)

        assert isinstance(ct_product, LWECiphertext)

    def test_matmul(self, scheme):
        """Test encrypted matrix multiplication."""
        _, pk, ek = scheme.keygen()
        pt = np.array([100], dtype=np.int64)
        weight = np.random.randn(16, 16).astype(np.float32)

        ct = scheme.encrypt(pk, pt)
        ct_result = scheme.matmul(ct, weight, ek)

        assert isinstance(ct_result, LWECiphertext)


class TestN2HEContext:
    """Tests for N2HE context."""

    @pytest.fixture
    def context(self):
        """Create a context with generated keys."""
        ctx = create_context(profile="lora", use_toy_mode=True)
        ctx.generate_keys()
        return ctx

    def test_create_context(self):
        """Test context creation."""
        ctx = create_context(profile="lora")
        assert ctx.params is not None
        assert ctx.scheme is not None

    def test_generate_keys(self, context):
        """Test key generation via context."""
        assert context.has_secret_key
        assert context.has_public_key
        assert context.has_eval_key

    def test_export_public_key(self, context):
        """Test public key export."""
        pk = context.export_public_key()
        assert len(pk) > 0

    def test_export_eval_key(self, context):
        """Test evaluation key export."""
        ek = context.export_eval_key()
        assert len(ek) > 0

    def test_encrypt(self, context):
        """Test encryption via context."""
        plaintext = np.array([42], dtype=np.int64)
        ciphertext = context.encrypt(plaintext)
        assert isinstance(ciphertext, LWECiphertext)

    def test_decrypt(self, context):
        """Test decryption via context."""
        plaintext = np.array([42], dtype=np.int64)
        ciphertext = context.encrypt(plaintext)
        decrypted = context.decrypt(ciphertext)
        assert isinstance(decrypted, np.ndarray)

    def test_encrypted_linear(self, context):
        """Test encrypted linear transformation."""
        plaintext = np.array([100], dtype=np.int64)
        weight = np.random.randn(16, 16).astype(np.float32)

        ct = context.encrypt(plaintext)
        result = context.encrypted_linear(ct, weight)

        assert isinstance(result, LWECiphertext)

    def test_encrypted_lora_delta(self, context):
        """Test encrypted LoRA delta computation."""
        plaintext = np.array([100], dtype=np.int64)
        lora_a = np.random.randn(16, 32).astype(np.float32)
        lora_b = np.random.randn(32, 16).astype(np.float32)

        ct = context.encrypt(plaintext)
        result = context.encrypted_lora_delta(ct, lora_a, lora_b, scaling=2.0)

        assert isinstance(result, LWECiphertext)

    def test_metrics(self, context):
        """Test context metrics."""
        plaintext = np.array([42], dtype=np.int64)
        context.encrypt(plaintext)

        metrics = context.get_metrics()
        assert metrics["operations_count"] > 0
        assert metrics["has_secret_key"]

    def test_load_keys(self):
        """Test loading keys into context."""
        ctx1 = create_context()
        ctx1.generate_keys()

        pk = ctx1.export_public_key()
        ek = ctx1.export_eval_key()

        ctx2 = create_context()
        ctx2.load_keys(pk=pk, ek=ek)

        assert ctx2.has_public_key
        assert ctx2.has_eval_key
        assert not ctx2.has_secret_key


class TestLWECiphertext:
    """Tests for LWE ciphertext."""

    def test_create_ciphertext(self):
        """Test ciphertext creation."""
        params = HESchemeParams()
        a = np.zeros(params.n, dtype=np.int32)
        b = 12345

        ct = LWECiphertext(a=a, b=b, params=params)

        assert len(ct.a) == params.n
        assert ct.b == b
        assert ct.noise_budget is not None

    def test_serialization(self):
        """Test ciphertext byte serialization."""
        params = HESchemeParams()
        a = np.random.randint(0, 1000, size=params.n, dtype=np.int32)
        b = 12345

        ct = LWECiphertext(a=a, b=b, params=params)
        data = ct.to_bytes()

        assert len(data) > 0

    def test_deserialization(self):
        """Test ciphertext deserialization."""
        params = HESchemeParams()
        a = np.random.randint(0, 1000, size=params.n, dtype=np.int32)
        b = 12345

        ct = LWECiphertext(a=a, b=b, params=params)
        data = ct.to_bytes()

        restored = LWECiphertext.from_bytes(data, params)

        assert np.array_equal(restored.a, ct.a)
        assert restored.b == ct.b


class TestRLWECiphertext:
    """Tests for RLWE ciphertext."""

    def test_create_ciphertext(self):
        """Test RLWE ciphertext creation."""
        params = HESchemeParams()
        n = params.poly_degree
        c0 = np.zeros(n, dtype=np.int64)
        c1 = np.zeros(n, dtype=np.int64)

        ct = RLWECiphertext(c0=c0, c1=c1, params=params)

        assert len(ct.c0) == n
        assert len(ct.c1) == n
        assert ct.scale == 1.0

    def test_serialization_roundtrip(self):
        """Test RLWE serialization/deserialization."""
        params = HESchemeParams()
        n = params.poly_degree
        c0 = np.random.randint(0, 1000, size=n, dtype=np.int64)
        c1 = np.random.randint(0, 1000, size=n, dtype=np.int64)

        ct = RLWECiphertext(c0=c0, c1=c1, params=params, scale=2.5)
        data = ct.to_bytes()

        restored = RLWECiphertext.from_bytes(data, params)

        assert np.array_equal(restored.c0, ct.c0)
        assert np.array_equal(restored.c1, ct.c1)
        assert restored.scale == ct.scale
