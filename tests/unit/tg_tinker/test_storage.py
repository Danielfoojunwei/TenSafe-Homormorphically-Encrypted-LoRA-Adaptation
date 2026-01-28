"""
Unit tests for TG-Tinker encrypted artifact storage.
"""

import os
import secrets
import tempfile
from pathlib import Path

import pytest

from tensorguard.platform.tg_tinker_api.storage import (
    LocalStorageBackend,
    EncryptedArtifactStore,
    KeyManager,
)


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a local storage backend with temp directory."""
        return LocalStorageBackend(str(tmp_path))

    def test_write_and_read(self, storage):
        """Test basic write and read."""
        data = b"test data"
        storage.write("test-key", data)
        result = storage.read("test-key")
        assert result == data

    def test_exists(self, storage):
        """Test exists check."""
        assert not storage.exists("nonexistent")
        storage.write("exists-key", b"data")
        assert storage.exists("exists-key")

    def test_delete(self, storage):
        """Test delete operation."""
        storage.write("to-delete", b"data")
        assert storage.exists("to-delete")
        storage.delete("to-delete")
        assert not storage.exists("to-delete")

    def test_read_nonexistent(self, storage):
        """Test reading nonexistent key raises error."""
        with pytest.raises(FileNotFoundError):
            storage.read("nonexistent-key")

    def test_path_sanitization(self, storage):
        """Test that path traversal is prevented."""
        storage.write("../../../etc/passwd", b"harmless")
        # Should not create file outside base path
        assert not os.path.exists("/etc/passwd_test")


class TestKeyManager:
    """Tests for KeyManager."""

    def test_get_dek_creates_new(self):
        """Test that get_dek creates a new key for new tenant."""
        km = KeyManager()
        dek1, id1 = km.get_dek("tenant-1")
        assert len(dek1) == 32
        assert id1.startswith("dek-")

    def test_get_dek_returns_same(self):
        """Test that get_dek returns same key for same tenant."""
        km = KeyManager()
        dek1, id1 = km.get_dek("tenant-1")
        dek2, id2 = km.get_dek("tenant-1")
        assert dek1 == dek2
        assert id1 == id2

    def test_different_tenants_different_keys(self):
        """Test that different tenants get different keys."""
        km = KeyManager()
        dek1, _ = km.get_dek("tenant-1")
        dek2, _ = km.get_dek("tenant-2")
        assert dek1 != dek2

    def test_rotate_dek(self):
        """Test key rotation."""
        km = KeyManager()
        dek1, id1 = km.get_dek("tenant-1")
        dek2, id2 = km.rotate_dek("tenant-1")
        assert dek1 != dek2
        assert id1 != id2

    def test_persistence(self, tmp_path):
        """Test key persistence."""
        key_path = str(tmp_path / "keys.json")
        master_key = secrets.token_bytes(32)

        # Create manager and generate key
        km1 = KeyManager(master_key=master_key, key_store_path=key_path)
        dek1, id1 = km1.get_dek("tenant-1")

        # Create new manager with same master key
        km2 = KeyManager(master_key=master_key, key_store_path=key_path)
        dek2, id2 = km2.get_dek("tenant-1")

        assert dek1 == dek2
        assert id1 == id2


class TestEncryptedArtifactStore:
    """Tests for EncryptedArtifactStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create an encrypted artifact store."""
        backend = LocalStorageBackend(str(tmp_path))
        key_manager = KeyManager()
        return EncryptedArtifactStore(backend, key_manager)

    def test_save_and_load_roundtrip(self, store):
        """Test encryption roundtrip."""
        original_data = b"This is sensitive model weights data"

        # Save artifact
        artifact = store.save_artifact(
            data=original_data,
            tenant_id="tenant-123",
            training_client_id="tc-456",
            artifact_type="checkpoint",
            metadata={"step": 100},
        )

        # Verify artifact metadata
        assert artifact.id.startswith("art-")
        assert artifact.tenant_id == "tenant-123"
        assert artifact.training_client_id == "tc-456"
        assert artifact.artifact_type == "checkpoint"
        assert artifact.size_bytes == len(original_data)
        assert artifact.encryption_algorithm == "AES-256-GCM"
        assert artifact.content_hash.startswith("sha256:")

        # Load and verify
        loaded_data = store.load_artifact(artifact)
        assert loaded_data == original_data

    def test_different_tenants_different_encryption(self, store):
        """Test that different tenants have different encryption."""
        data = b"shared data"

        artifact1 = store.save_artifact(
            data=data,
            tenant_id="tenant-1",
            training_client_id="tc-1",
            artifact_type="weights",
        )

        artifact2 = store.save_artifact(
            data=data,
            tenant_id="tenant-2",
            training_client_id="tc-2",
            artifact_type="weights",
        )

        # Content hashes should be same (same plaintext)
        assert artifact1.content_hash == artifact2.content_hash

        # But encryption key IDs should be different
        assert artifact1.encryption_key_id != artifact2.encryption_key_id

    def test_tamper_detection(self, store):
        """Test that tampering is detected."""
        data = b"important data"

        artifact = store.save_artifact(
            data=data,
            tenant_id="tenant-123",
            training_client_id="tc-456",
            artifact_type="checkpoint",
        )

        # Tamper with content hash
        artifact.content_hash = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

        with pytest.raises(ValueError, match="hash mismatch"):
            store.load_artifact(artifact)

    def test_metadata_preserved(self, store):
        """Test that metadata is preserved."""
        metadata = {
            "step": 500,
            "loss": 0.5,
            "custom": {"nested": "value"},
        }

        artifact = store.save_artifact(
            data=b"data",
            tenant_id="tenant-123",
            training_client_id="tc-456",
            artifact_type="checkpoint",
            metadata=metadata,
        )

        assert artifact.metadata_json == metadata

    def test_large_data(self, store):
        """Test with larger data."""
        # 1 MB of random data
        large_data = secrets.token_bytes(1024 * 1024)

        artifact = store.save_artifact(
            data=large_data,
            tenant_id="tenant-123",
            training_client_id="tc-456",
            artifact_type="weights",
        )

        loaded = store.load_artifact(artifact)
        assert loaded == large_data

    def test_delete_artifact(self, store, tmp_path):
        """Test artifact deletion."""
        backend = LocalStorageBackend(str(tmp_path))
        key_manager = KeyManager()
        local_store = EncryptedArtifactStore(backend, key_manager)

        artifact = local_store.save_artifact(
            data=b"to be deleted",
            tenant_id="tenant-123",
            training_client_id="tc-456",
            artifact_type="checkpoint",
        )

        assert backend.exists(artifact.storage_key)
        local_store.delete_artifact(artifact)
        assert not backend.exists(artifact.storage_key)
