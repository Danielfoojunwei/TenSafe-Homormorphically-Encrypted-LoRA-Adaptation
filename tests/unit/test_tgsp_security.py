"""
Security tests for TGSP (TensorGuard Secure Package) extraction.

These tests verify that TGSP extraction is safe against:
- Path traversal attacks (../../ escapes)
- Absolute path extraction
- Symlink attacks
- Device file attacks
"""

import os
import tarfile
import tempfile
import pytest
from pathlib import Path

from tensorguard.tgsp.cli import safe_extract_tar, TarExtractionError


class TestSafeExtractTar:
    """Tests for safe_extract_tar function."""

    def test_normal_extraction_succeeds(self, tmp_path):
        """Test that normal files extract correctly."""
        # Create a legitimate tar
        tar_path = tmp_path / "test.tar"
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        (content_dir / "file.txt").write_text("hello")
        (content_dir / "subdir").mkdir()
        (content_dir / "subdir" / "nested.txt").write_text("world")

        with tarfile.open(tar_path, "w") as tar:
            tar.add(content_dir / "file.txt", arcname="file.txt")
            tar.add(content_dir / "subdir", arcname="subdir")
            tar.add(content_dir / "subdir" / "nested.txt", arcname="subdir/nested.txt")

        # Extract to a new directory
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            extracted = safe_extract_tar(tar, str(extract_dir))

        # At least the expected files should be extracted
        assert len(extracted) >= 3
        assert (extract_dir / "file.txt").read_text() == "hello"
        assert (extract_dir / "subdir" / "nested.txt").read_text() == "world"

    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal attacks are blocked."""
        tar_path = tmp_path / "malicious.tar"

        # Create tar with path traversal
        with tarfile.open(tar_path, "w") as tar:
            # Create a TarInfo with malicious path
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 5
            tar.addfile(info, fileobj=__import__('io').BytesIO(b"evil!"))

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            with pytest.raises(TarExtractionError) as exc_info:
                safe_extract_tar(tar, str(extract_dir))

            assert "Path traversal detected" in str(exc_info.value)
            assert "../../../etc/passwd" in str(exc_info.value)

    def test_absolute_path_blocked(self, tmp_path):
        """Test that absolute paths are blocked."""
        tar_path = tmp_path / "malicious.tar"

        with tarfile.open(tar_path, "w") as tar:
            info = tarfile.TarInfo(name="/etc/passwd")
            info.size = 5
            tar.addfile(info, fileobj=__import__('io').BytesIO(b"evil!"))

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            with pytest.raises(TarExtractionError) as exc_info:
                safe_extract_tar(tar, str(extract_dir))

            # Should be blocked either as absolute path or path traversal
            error_msg = str(exc_info.value).lower()
            assert "absolute" in error_msg or "traversal" in error_msg

    def test_symlink_blocked_by_default(self, tmp_path):
        """Test that symlinks are blocked by default."""
        tar_path = tmp_path / "symlink.tar"

        with tarfile.open(tar_path, "w") as tar:
            info = tarfile.TarInfo(name="evil_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            with pytest.raises(TarExtractionError) as exc_info:
                safe_extract_tar(tar, str(extract_dir), allow_symlinks=False)

            assert "Symlink" in str(exc_info.value) or "symlink" in str(exc_info.value)

    def test_symlink_escape_blocked(self, tmp_path):
        """Test that symlinks pointing outside dest are blocked even when allowed."""
        tar_path = tmp_path / "symlink_escape.tar"

        with tarfile.open(tar_path, "w") as tar:
            info = tarfile.TarInfo(name="escape_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../../etc/passwd"
            tar.addfile(info)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            with pytest.raises(TarExtractionError) as exc_info:
                safe_extract_tar(tar, str(extract_dir), allow_symlinks=True)

            assert "escapes" in str(exc_info.value).lower() or "traversal" in str(exc_info.value).lower()

    def test_hardlink_blocked_by_default(self, tmp_path):
        """Test that hardlinks are blocked by default."""
        tar_path = tmp_path / "hardlink.tar"

        with tarfile.open(tar_path, "w") as tar:
            # First add a normal file
            info1 = tarfile.TarInfo(name="original.txt")
            info1.size = 5
            tar.addfile(info1, fileobj=__import__('io').BytesIO(b"data!"))

            # Then add a hardlink
            info2 = tarfile.TarInfo(name="hardlink.txt")
            info2.type = tarfile.LNKTYPE
            info2.linkname = "original.txt"
            tar.addfile(info2)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            with pytest.raises(TarExtractionError) as exc_info:
                safe_extract_tar(tar, str(extract_dir), allow_symlinks=False)

            assert "hardlink" in str(exc_info.value).lower()

    def test_device_file_blocked(self, tmp_path):
        """Test that device files are blocked."""
        tar_path = tmp_path / "device.tar"

        with tarfile.open(tar_path, "w") as tar:
            info = tarfile.TarInfo(name="evil_device")
            info.type = tarfile.CHRTYPE  # Character device
            info.devmajor = 1
            info.devminor = 3
            tar.addfile(info)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r") as tar:
            with pytest.raises(TarExtractionError) as exc_info:
                safe_extract_tar(tar, str(extract_dir))

            assert "Device file" in str(exc_info.value)

    def test_dot_dot_in_filename_blocked(self, tmp_path):
        """Test various path traversal patterns."""
        test_cases = [
            "foo/../../../bar",
            "foo/bar/../../baz/../../../etc/passwd",
            "./../../outside",
        ]

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        for malicious_path in test_cases:
            tar_path = tmp_path / f"test_{malicious_path.replace('/', '_')}.tar"

            with tarfile.open(tar_path, "w") as tar:
                info = tarfile.TarInfo(name=malicious_path)
                info.size = 4
                tar.addfile(info, fileobj=__import__('io').BytesIO(b"bad!"))

            with tarfile.open(tar_path, "r") as tar:
                # This should either raise TarExtractionError or extract safely within dest
                try:
                    safe_extract_tar(tar, str(extract_dir))
                    # If it didn't raise, verify nothing was extracted outside
                    for root, dirs, files in os.walk(tmp_path):
                        for f in files:
                            full_path = Path(root) / f
                            if full_path.suffix == ".tar":
                                continue
                            # Verify file is within extract_dir
                            try:
                                full_path.resolve().relative_to(extract_dir.resolve())
                            except ValueError:
                                pytest.fail(f"File extracted outside dest: {full_path}")
                except TarExtractionError:
                    pass  # Expected for malicious paths


class TestTGSPIntegrity:
    """Tests for TGSP cryptographic integrity."""

    def test_manifest_modification_detected(self):
        """Test that modifying the manifest is detected."""
        # This would require a full TGSP package to test properly
        # Placeholder for integration test
        pass

    def test_recipient_list_modification_detected(self):
        """Test that modifying the recipient list is detected."""
        # This would require a full TGSP package to test properly
        # Placeholder for integration test
        pass
