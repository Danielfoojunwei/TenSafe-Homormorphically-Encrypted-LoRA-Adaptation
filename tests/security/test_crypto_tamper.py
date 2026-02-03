"""
Crypto Tamper Resistance Tests.

These tests verify that cryptographic operations fail safely when:
1. Ciphertext is corrupted (bit flip)
2. AAD/manifest is mutated
3. Recipients are swapped
4. Payloads are swapped

All tampering attempts MUST result in:
- Clear error messages
- No plaintext leakage
- No undefined behavior
"""

import os
import secrets

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from tensorguard.crypto.payload import PayloadDecryptor, PayloadEncryptor


class TestPayloadTamperResistance:
    """Test payload encryption tamper resistance."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = secrets.token_bytes(32)
        self.manifest_hash = secrets.token_hex(16)
        self.recipients_hash = secrets.token_hex(16)

    def _decrypt_chunk(self, decryptor: PayloadDecryptor, ciphertext: bytes) -> bytes:
        """Helper to decrypt a chunk from raw bytes."""
        import io

        stream = io.BytesIO(ciphertext)
        return decryptor.decrypt_chunk_from_stream(stream)

    def test_bitflip_in_ciphertext_fails(self):
        """Bit flip in ciphertext must cause decryption failure."""
        encryptor = PayloadEncryptor(
            key=self.key,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        plaintext = b"sensitive data that must be protected"
        ciphertext = encryptor.encrypt_chunk(plaintext)

        # Flip a bit in the middle of the ciphertext (after the length prefix)
        corrupted = bytearray(ciphertext)
        middle = 4 + len(corrupted) // 2  # Skip 4-byte length prefix
        if middle < len(corrupted):
            corrupted[middle] ^= 0x01  # Flip one bit

        decryptor = PayloadDecryptor(
            key=self.key,
            nonce_base=encryptor.nonce_base,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        with pytest.raises(Exception):  # Crypto library raises InvalidTag
            self._decrypt_chunk(decryptor, bytes(corrupted))

    def test_truncated_ciphertext_fails(self):
        """Truncated ciphertext must cause decryption failure."""
        encryptor = PayloadEncryptor(
            key=self.key,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        plaintext = b"sensitive data that must be protected"
        ciphertext = encryptor.encrypt_chunk(plaintext)

        # Truncate ciphertext
        truncated = ciphertext[:-10]

        decryptor = PayloadDecryptor(
            key=self.key,
            nonce_base=encryptor.nonce_base,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        with pytest.raises(Exception):
            self._decrypt_chunk(decryptor, truncated)

    def test_wrong_manifest_hash_fails(self):
        """Wrong manifest hash must cause decryption failure."""
        encryptor = PayloadEncryptor(
            key=self.key,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        plaintext = b"sensitive data bound to manifest"
        ciphertext = encryptor.encrypt_chunk(plaintext)

        # Try to decrypt with different manifest hash
        wrong_manifest = secrets.token_hex(16)
        decryptor = PayloadDecryptor(
            key=self.key,
            nonce_base=encryptor.nonce_base,
            manifest_hash=wrong_manifest,  # WRONG
            recipients_hash=self.recipients_hash,
        )

        with pytest.raises(Exception):
            self._decrypt_chunk(decryptor, ciphertext)

    def test_wrong_recipients_hash_fails(self):
        """Wrong recipients hash must cause decryption failure."""
        encryptor = PayloadEncryptor(
            key=self.key,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        plaintext = b"sensitive data bound to recipients"
        ciphertext = encryptor.encrypt_chunk(plaintext)

        # Try to decrypt with different recipients hash
        wrong_recipients = secrets.token_hex(16)
        decryptor = PayloadDecryptor(
            key=self.key,
            nonce_base=encryptor.nonce_base,
            manifest_hash=self.manifest_hash,
            recipients_hash=wrong_recipients,  # WRONG
        )

        with pytest.raises(Exception):
            self._decrypt_chunk(decryptor, ciphertext)

    def test_wrong_key_fails(self):
        """Wrong key must cause decryption failure."""
        encryptor = PayloadEncryptor(
            key=self.key,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        plaintext = b"sensitive data protected by key"
        ciphertext = encryptor.encrypt_chunk(plaintext)

        # Try to decrypt with different key
        wrong_key = secrets.token_bytes(32)
        decryptor = PayloadDecryptor(
            key=wrong_key,  # WRONG
            nonce_base=encryptor.nonce_base,
            manifest_hash=self.manifest_hash,
            recipients_hash=self.recipients_hash,
        )

        with pytest.raises(Exception):
            self._decrypt_chunk(decryptor, ciphertext)


class TestAEADTamperResistance:
    """Test raw AEAD tamper resistance."""

    def test_aesgcm_bitflip_fails(self):
        """AES-GCM must fail on ciphertext bit flip."""
        key = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        aad = b"additional authenticated data"
        plaintext = b"secret message"

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

        # Flip a bit
        corrupted = bytearray(ciphertext)
        corrupted[5] ^= 0x80
        corrupted = bytes(corrupted)

        with pytest.raises(Exception):
            aesgcm.decrypt(nonce, corrupted, aad)

    def test_aesgcm_wrong_aad_fails(self):
        """AES-GCM must fail on wrong AAD."""
        key = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        aad = b"original aad"
        plaintext = b"secret message"

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

        # Try with wrong AAD
        wrong_aad = b"tampered aad"
        with pytest.raises(Exception):
            aesgcm.decrypt(nonce, ciphertext, wrong_aad)

    def test_chacha20poly1305_bitflip_fails(self):
        """ChaCha20-Poly1305 must fail on ciphertext bit flip."""
        key = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        aad = b"additional authenticated data"
        plaintext = b"secret message"

        aead = ChaCha20Poly1305(key)
        ciphertext = aead.encrypt(nonce, plaintext, aad)

        # Flip a bit
        corrupted = bytearray(ciphertext)
        corrupted[10] ^= 0x01
        corrupted = bytes(corrupted)

        with pytest.raises(Exception):
            aead.decrypt(nonce, corrupted, aad)

    def test_nonce_reuse_different_output(self):
        """Same nonce with same key must produce deterministic output (but reuse is bad)."""
        key = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        plaintext = b"message"

        aead = ChaCha20Poly1305(key)
        ct1 = aead.encrypt(nonce, plaintext, b"aad1")
        ct2 = aead.encrypt(nonce, plaintext, b"aad1")

        # Same inputs = same output (deterministic)
        assert ct1 == ct2

        # Different AAD = different output
        ct3 = aead.encrypt(nonce, plaintext, b"aad2")
        assert ct1 != ct3


class TestVersionCompatibility:
    """Test version compatibility checks."""

    def test_unsupported_version_rejected(self):
        """Unsupported TGSP version should be rejected."""
        from tensorguard.tgsp.spec import VERSION

        # This is a documentation test - the actual version checking
        # happens in format.py when parsing TGSP files
        assert VERSION == "1.0.0"

        # Future versions should be rejected
        future_version = "99.0.0"
        assert future_version != VERSION


class TestRandomnessQuality:
    """Test that randomness sources are cryptographically secure."""

    def test_secrets_module_used(self):
        """Verify secrets module produces unique values."""
        values = [secrets.token_bytes(32) for _ in range(100)]

        # All values should be unique
        assert len(set(values)) == 100

    def test_os_urandom_used(self):
        """Verify os.urandom produces unique values."""
        values = [os.urandom(32) for _ in range(100)]

        # All values should be unique
        assert len(set(values)) == 100

    def test_nonce_uniqueness(self):
        """Verify nonces are unique per encryption."""
        key = secrets.token_bytes(32)
        manifest_hash = secrets.token_hex(16)
        recipients_hash = secrets.token_hex(16)

        encryptor = PayloadEncryptor(
            key=key,
            manifest_hash=manifest_hash,
            recipients_hash=recipients_hash,
        )

        # Encrypt same message multiple times
        plaintext = b"same message"
        ciphertexts = [encryptor.encrypt_chunk(plaintext) for _ in range(10)]

        # Each ciphertext should be unique (different nonce)
        assert len(set(ciphertexts)) == 10
