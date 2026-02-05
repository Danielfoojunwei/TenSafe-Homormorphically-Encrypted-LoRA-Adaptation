"""
Artifact Store Tests

Verifies that the artifact store can:
1. Save artifacts with hierarchical keys
2. Load artifacts after "restart" (new instance)
3. Properly encrypt and decrypt data
4. Fail-closed in production mode without proper key configuration
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest


class TestLocalStorageBackend:
    """Test LocalStorageBackend hierarchical key support."""

    def test_write_and_read_simple_key(self):
        """Test writing and reading with a simple key."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            backend.write("simple-key", b"test data")
            assert backend.read("simple-key") == b"test data"

    def test_write_and_read_hierarchical_key(self):
        """Test writing and reading with a hierarchical key (with /)."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            backend.write("tenant/client/artifact", b"hierarchical data")
            assert backend.read("tenant/client/artifact") == b"hierarchical data"

    def test_hierarchical_key_creates_directories(self):
        """Test that hierarchical keys create necessary subdirectories."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            backend.write("tenant123/client456/artifact789", b"data")

            # Verify directory structure was created
            assert Path(tmpdir, "tenant123").is_dir()
            assert Path(tmpdir, "tenant123", "client456").is_dir()
            assert Path(tmpdir, "tenant123", "client456", "artifact789").is_file()

    def test_exists_with_hierarchical_key(self):
        """Test exists() with hierarchical keys."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            assert not backend.exists("tenant/client/artifact")
            backend.write("tenant/client/artifact", b"data")
            assert backend.exists("tenant/client/artifact")

    def test_delete_with_hierarchical_key(self):
        """Test delete() with hierarchical keys."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            backend.write("tenant/client/artifact", b"data")
            assert backend.exists("tenant/client/artifact")
            backend.delete("tenant/client/artifact")
            assert not backend.exists("tenant/client/artifact")

    def test_rejects_path_traversal(self):
        """Test that path traversal attempts are rejected."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)

            # '..' is caught by segment validation
            with pytest.raises(ValueError, match="Invalid segment|Path traversal"):
                backend.write("../escape", b"evil")

            with pytest.raises(ValueError, match="Invalid segment|Path traversal"):
                backend.write("tenant/../escape", b"evil")

            with pytest.raises(ValueError, match="Invalid segment|Path traversal"):
                backend.write("tenant/../../escape", b"evil")

    def test_rejects_empty_key(self):
        """Test that empty keys are rejected."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)

            with pytest.raises(ValueError, match="cannot be empty"):
                backend.write("", b"data")

    def test_rejects_empty_segment(self):
        """Test that keys with empty segments (double slash) are rejected."""
        from tensorguard.platform.tg_tinker_api.storage import LocalStorageBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)

            with pytest.raises(ValueError, match="Empty segment"):
                backend.write("tenant//artifact", b"data")


class TestEncryptedArtifactStore:
    """Test EncryptedArtifactStore functionality."""

    def test_save_and_load_artifact(self):
        """Test saving and loading an artifact."""
        from tensorguard.platform.tg_tinker_api.storage import (
            EncryptedArtifactStore,
            LocalStorageBackend,
            KeyManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            key_manager = KeyManager()
            store = EncryptedArtifactStore(backend, key_manager)

            # Save artifact
            data = b"This is test artifact data!"
            artifact = store.save_artifact(
                data=data,
                tenant_id="tenant123",
                training_client_id="client456",
                artifact_type="checkpoint",
                metadata={"step": 100},
            )

            # Verify artifact metadata
            assert artifact.tenant_id == "tenant123"
            assert artifact.training_client_id == "client456"
            assert artifact.artifact_type == "checkpoint"
            assert artifact.size_bytes == len(data)
            assert artifact.encryption_algorithm == "AES-256-GCM"
            assert artifact.content_hash.startswith("sha256:")

            # Load artifact
            loaded_data = store.load_artifact(artifact)
            assert loaded_data == data

    def test_artifact_survives_restart(self):
        """Test that artifacts can be loaded after 'restart' (new store instance)."""
        from tensorguard.platform.tg_tinker_api.storage import (
            EncryptedArtifactStore,
            LocalStorageBackend,
            KeyManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance - save artifact
            master_key = os.urandom(32)  # Fixed key for this test
            key_store_path = os.path.join(tmpdir, "keys.json")

            backend1 = LocalStorageBackend(os.path.join(tmpdir, "artifacts"))
            key_manager1 = KeyManager(master_key=master_key, key_store_path=key_store_path)
            store1 = EncryptedArtifactStore(backend1, key_manager1)

            data = b"Persistent artifact data"
            artifact = store1.save_artifact(
                data=data,
                tenant_id="tenant-restart",
                training_client_id="client-restart",
                artifact_type="checkpoint",
            )

            # Simulate restart - create new instances
            backend2 = LocalStorageBackend(os.path.join(tmpdir, "artifacts"))
            key_manager2 = KeyManager(master_key=master_key, key_store_path=key_store_path)
            store2 = EncryptedArtifactStore(backend2, key_manager2)

            # Load artifact with new instance
            loaded_data = store2.load_artifact(artifact)
            assert loaded_data == data

    def test_tampered_ciphertext_fails(self):
        """Test that tampered ciphertext is detected."""
        from tensorguard.platform.tg_tinker_api.storage import (
            EncryptedArtifactStore,
            LocalStorageBackend,
            KeyManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(tmpdir)
            key_manager = KeyManager()
            store = EncryptedArtifactStore(backend, key_manager)

            # Save artifact
            artifact = store.save_artifact(
                data=b"Original data",
                tenant_id="tenant",
                training_client_id="client",
                artifact_type="checkpoint",
            )

            # Tamper with stored ciphertext
            ciphertext = backend.read(artifact.storage_key)
            tampered = ciphertext[:-1] + bytes([ciphertext[-1] ^ 0xFF])
            backend.write(artifact.storage_key, tampered)

            # Load should fail
            with pytest.raises(ValueError, match="tampering|decryption"):
                store.load_artifact(artifact)


class TestKeyManagerProductionMode:
    """Test KeyManager fail-closed behavior in production."""

    def test_fails_in_production_without_key(self):
        """Test that KeyManager fails in production mode without explicit key."""
        from tensorguard.platform.tg_tinker_api.storage import KeyManager

        # Set production environment
        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "production"

            with pytest.raises(RuntimeError, match="production mode"):
                KeyManager()

        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]

    def test_works_in_production_with_key(self):
        """Test that KeyManager works in production with explicit key."""
        from tensorguard.platform.tg_tinker_api.storage import KeyManager

        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "production"
            master_key = os.urandom(32)

            km = KeyManager(master_key=master_key)
            assert km._master_key == master_key

        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]

    def test_generates_key_in_development(self):
        """Test that KeyManager generates key in development mode."""
        from tensorguard.platform.tg_tinker_api.storage import KeyManager

        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "development"

            km = KeyManager()
            assert km._master_key is not None
            assert len(km._master_key) == 32

        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
