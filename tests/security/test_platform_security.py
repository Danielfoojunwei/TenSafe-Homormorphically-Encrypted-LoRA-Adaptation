import unittest
import os
import shutil
import pytest
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool


class TestPlatformSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test database and lazy import to avoid database initialization at import time."""
        # Import models to register them with SQLModel
        from tensorguard.platform.models import core, identity_models, enablement_models, evidence_models

        # Create in-memory database with proper tables
        cls._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        SQLModel.metadata.create_all(cls._engine)

        # Now import the app and override the session
        from fastapi.testclient import TestClient
        from tensorguard.platform.main import app
        from tensorguard.platform.database import get_session

        # Override the session with our test database
        def get_session_override():
            with Session(cls._engine) as session:
                yield session

        app.dependency_overrides[get_session] = get_session_override

        cls._app = app
        cls._TestClient = TestClient

    @classmethod
    def tearDownClass(cls):
        # Clean up dependency overrides
        cls._app.dependency_overrides.clear()

    def setUp(self):
        self.client = self._TestClient(self._app)

    def test_upload_path_traversal_sanitization(self):
        """Test that uploaded filenames with ../ are sanitized."""
        import secrets
        filename = "../../../evil_script.py"
        # Create a dummy TGSP file
        from unittest.mock import patch

        mock_manifest = {
            "package_id": f"evil-{secrets.token_hex(4)}",
            "author_id": "malicious",
            "created_at": 1735732800.0,
            "policy_id": "none",
            "policy_version": 1,
            "content_index": [],
            "compat_base_model_id": []
        }

        with patch("tensorguard.platform.api.community_tgsp.read_tgsp_header") as mock_read:
            mock_read.return_value = {
                "version": "0.2",
                "header": {"hashes": {"manifest": "hash123"}},
                "manifest": mock_manifest,
                "recipients": []
            }

            response = self.client.post(
                "/api/community/tgsp/upload",
                files={"file": (filename, b"fake_tgsp_binary", "application/octet-stream")}
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify the saved filename is just 'evil_script.py'
        self.assertEqual(data["filename"], "evil_script.py")

        # Verify it's in the designated storage dir, not root
        storage_path = data["storage_path"]
        self.assertIn("storage", storage_path)
        self.assertNotIn("..", storage_path)


if __name__ == "__main__":
    unittest.main()
