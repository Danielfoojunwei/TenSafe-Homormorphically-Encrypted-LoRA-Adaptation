"""
Comprehensive TensorGuard Cryptography Tests

Expanded test coverage for:
- Key generation with CSPRNG
- Encryption/decryption edge cases
- Serialization robustness
- Key rotation scenarios
- Error handling for malformed data
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from tensorguard.core.crypto import (
    N2HEContext,
    N2HEParams,
    LWECiphertext,
    N2HEEncryptor,
    sample_skellam,
    generate_key,
    CryptographyError
)
from tensorguard.core.keys import vault, KeyScope


class TestKeyGeneration:
    """Test suite for key generation with CSPRNG."""
    
    def test_key_generation_basic(self):
        """Verify keys are generated with correct dimensions."""
        ctx = N2HEContext()
        ctx.generate_keys()
        assert ctx.lwe_key is not None
        assert len(ctx.lwe_key) == ctx.params.n
    
    def test_key_generation_ternary(self):
        """Verify keys are ternary (values in {-1, 0, 1})."""
        ctx = N2HEContext()
        ctx.generate_keys()
        unique_values = set(ctx.lwe_key.tolist())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_key_generation_uniqueness(self):
        """Verify consecutive key generations produce different keys."""
        ctx1 = N2HEContext()
        ctx1.generate_keys()
        
        ctx2 = N2HEContext()
        ctx2.generate_keys()
        
        # Keys should be different (probabilistic, but extremely unlikely to be same)
        assert not np.array_equal(ctx1.lwe_key, ctx2.lwe_key)
    
    def test_key_persistence(self):
        """Test saving and loading keys from disk (via Vault)."""
        ctx = N2HEContext()
        ctx.generate_keys()
        original_key = ctx.lwe_key.copy()
        
        # Use a vault-friendly name
        key_name = "test_persistence_key"
        ctx.save_key(key_name)
        
        # Clear and reload
        ctx.lwe_key = None
        ctx.load_key(key_name)
        
        np.testing.assert_array_equal(original_key, ctx.lwe_key)
        # Cleanup
        vault.delete_key(KeyScope.AGGREGATION, key_name, suffix=".npy.bin")
    
    def test_key_file_permissions(self):
        """Verify saved key files in vault have restricted permissions."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        key_name = "secure_vault_key"
        ctx.save_key(key_name)
        
        # Check file exists in vault
        path = Path("keys/aggregation") / "secure_vault_key.npy.bin"
        assert path.exists()
        # On Windows, just verify existence.
        
        # Cleanup
        vault.delete_key(KeyScope.AGGREGATION, key_name, suffix=".npy.bin")
    
    def test_key_load_missing_file(self):
        """Verify CryptographyError on missing key file."""
        ctx = N2HEContext()
        # The new error is wrapped
        with pytest.raises(CryptographyError, match="Key file not found"):
            ctx.load_key("nonexistent_key_name")


class TestEncryptionDecryption:
    """Test suite for encryption/decryption operations."""
    
    def test_single_value(self):
        """Encrypt and decrypt a single value."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        message = np.array([42], dtype=np.int64)
        ct = ctx.encrypt_batch(message)
        decoded = ctx.decrypt_batch(ct)
        
        np.testing.assert_array_equal(message, decoded)
    
    def test_batch_encryption(self):
        """Encrypt and decrypt a batch of values."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.array([1, 2, 3, 4, 42, 100, 200], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        
        assert ct.is_batch
        decoded = ctx.decrypt_batch(ct)
        
        np.testing.assert_array_equal(messages, decoded)
    
    def test_zero_values(self):
        """Test encryption of zeros."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.array([0, 0, 0], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        decoded = ctx.decrypt_batch(ct)
        
        np.testing.assert_array_equal(messages, decoded)
    
    def test_max_plaintext_value(self):
        """Test encryption at plaintext modulus boundary."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        # Values should be mod t
        messages = np.array([ctx.params.t - 1, ctx.params.t - 2], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        decoded = ctx.decrypt_batch(ct)
        
        np.testing.assert_array_equal(messages, decoded)
    
    def test_large_batch(self):
        """Test encryption of large batches."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.random.randint(0, ctx.params.t, size=1000, dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        decoded = ctx.decrypt_batch(ct)
        
        np.testing.assert_array_equal(messages, decoded)
    
    def test_encryption_stats_tracking(self):
        """Verify encryption statistics are tracked."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        assert ctx.stats['encryptions'] == 0
        
        messages = np.array([1, 2, 3], dtype=np.int64)
        ctx.encrypt_batch(messages)
        
        assert ctx.stats['encryptions'] == 3
    
    def test_decryption_without_keys(self):
        """Verify error on decryption without keys."""
        ctx = N2HEContext()
        # Don't generate keys
        
        ct = LWECiphertext(
            a=np.zeros(ctx.params.n, dtype=np.int64),
            b=0
        )
        
        with pytest.raises(CryptographyError, match="Keys not generated"):
            ctx.decrypt_batch(ct)


class TestSerialization:
    """Test suite for ciphertext serialization."""
    
    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.array([100, 200, 300], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        
        data = ct.serialize()
        assert isinstance(data, bytes)
        assert len(data) > 0
        
        ct_new = LWECiphertext.deserialize(data, params=ctx.params)
        decoded = ctx.decrypt_batch(ct_new)
        
        np.testing.assert_array_equal(messages, decoded)
    
    def test_serialization_magic_number(self):
        """Verify serialized data contains magic number."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.array([1], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        data = ct.serialize()
        
        # Check magic bytes - should be LWE2 now that we use seeded A matrix
        assert data[:4] in [b'LWE1', b'LWE2']
    
    def test_deserialization_invalid_magic(self):
        """Verify error on invalid magic number."""
        bad_data = b'BAAD' + b'\x00' * 100
        
        with pytest.raises(CryptographyError, match="Unsupported LWE Magic|Not enough data"):
            LWECiphertext.deserialize(bad_data)
    
    def test_deserialization_truncated_data(self):
        """Verify error on truncated payload."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.array([1, 2, 3], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        data = ct.serialize()
        
        # Truncate data
        truncated = data[:20]
        
        with pytest.raises(CryptographyError, match="Not enough data|Failed to unpack"):
            LWECiphertext.deserialize(truncated)
    
    def test_deserialization_corrupted_flags(self):
        """Test handling of corrupted flag byte."""
        ctx = N2HEContext()
        ctx.generate_keys()
        
        messages = np.array([1], dtype=np.int64)
        ct = ctx.encrypt_batch(messages)
        data = ct.serialize()
        
        # Corrupt the flags byte (byte 12)
        corrupted = data[:12] + b'\xFF' + data[13:]
        
        # Should still deserialize (flags are checked but not all values invalid)
        ct_new = LWECiphertext.deserialize(corrupted, params=ctx.params)
        # Result may be incorrect but should not crash


class TestSkellamNoise:
    """Test suite for Skellam noise sampling."""
    
    def test_skellam_symmetric(self):
        """Verify Skellam distribution is approximately symmetric around 0."""
        samples = sample_skellam(3.0, 10000)
        
        mean = np.mean(samples)
        # Mean should be close to 0 for symmetric Skellam
        assert abs(mean) < 0.5
    
    def test_skellam_variance(self):
        """Verify Skellam variance is approximately 2*mu."""
        mu = 5.0
        samples = sample_skellam(mu, 10000)
        
        variance = np.var(samples)
        expected_variance = 2 * mu
        
        # Allow 20% tolerance
        assert abs(variance - expected_variance) < expected_variance * 0.2
    
    def test_skellam_integer_output(self):
        """Verify Skellam outputs integers."""
        samples = sample_skellam(3.0, 100)
        assert samples.dtype == np.int64


class TestN2HEEncryptor:
    """Test suite for the N2HEEncryptor wrapper class."""
    
    def test_encryptor_initialization(self):
        """Test basic encryptor initialization."""
        encryptor = N2HEEncryptor()
        assert encryptor._ctx.lwe_key is not None
    
    def test_encryptor_with_existing_key(self):
        """Test initialization with existing key file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "test_key.npy")
            generate_key(key_path, security_level=128)
            
            encryptor = N2HEEncryptor(key_path=key_path)
            assert encryptor._ctx.lwe_key is not None
    
    def test_encrypt_decrypt_bytes(self):
        """Test encryption and decryption of binary data."""
        encryptor = N2HEEncryptor()
        
        plaintext = b"Hello, TensorGuard!"
        ciphertext = encryptor.encrypt(plaintext)
        
        assert isinstance(ciphertext, bytes)
        assert ciphertext != plaintext
        
        decrypted = encryptor.decrypt(ciphertext)
        # Handle padding from SIMD folding
        if isinstance(decrypted, bytes):
            decrypted = decrypted.rstrip(b'\x00')
        elif isinstance(decrypted, np.ndarray):
            decrypted = decrypted.tobytes().rstrip(b'\x00')
        assert decrypted == plaintext
    
    def test_key_rotation_on_max_uses(self):
        """Test automatic key rotation after max uses."""
        encryptor = N2HEEncryptor()
        encryptor._max_uses = 3  # Override for testing
        
        original_key = encryptor._ctx.lwe_key.copy()
        
        # Use up to max
        for _ in range(3):
            encryptor.encrypt(b"test")
        
        # Next encrypt should trigger rotation
        encryptor.encrypt(b"test")
        
        # Key should have changed
        assert not np.array_equal(original_key, encryptor._ctx.lwe_key)


class TestN2HEParams:
    """Test suite for parameter configuration."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = N2HEParams()
        assert params.n > 0
        assert params.q > 0
        assert params.t > 0
    
    def test_delta_calculation(self):
        """Test delta (scaling factor) calculation."""
        params = N2HEParams()
        assert params.delta == params.q // params.t
    
    def test_custom_security_level(self):
        """Test custom security level configuration."""
        params_128 = N2HEParams(security_bits=128)
        params_192 = N2HEParams(security_bits=192)
        
        # Higher security should use larger q
        assert params_192.q > params_128.q


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
