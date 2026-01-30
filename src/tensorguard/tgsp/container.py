"""
TGSP Container - Deterministic ZIP-based packaging

SECURITY NOTE: This module implements deterministic packaging to ensure
byte-for-byte reproducible artifacts for audit and verification.
"""

import os
import zipfile
from typing import Dict, List

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB safety limit

# Fixed timestamp for deterministic builds (2020-01-01 00:00:00)
DETERMINISTIC_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


class TGSPContainer:
    """
    Deterministic ZIP container for TGSP packages.

    All files are written with normalized metadata to ensure reproducibility:
    - Fixed timestamps (2020-01-01 00:00:00)
    - No compression (for predictable output)
    - Sorted file ordering
    """

    def __init__(self, path: str, mode: str = 'r'):
        self.path = path
        self.mode = mode
        # Use ZIP_STORED (no compression) for determinism
        compression = zipfile.ZIP_STORED if mode == 'w' else zipfile.ZIP_DEFLATED
        self.zip = zipfile.ZipFile(path, mode=mode, compression=compression)
        self._pending_files: List[tuple] = []  # For deterministic ordering

    def write_file(self, arcname: str, data: bytes):
        """
        Queue a file for writing with normalized metadata.

        Files are written in sorted order when close() is called.
        """
        if len(data) > MAX_FILE_SIZE:
            raise ValueError(f"File {arcname} exceeds safety limit of {MAX_FILE_SIZE} bytes")

        if self.mode == 'w':
            self._pending_files.append((arcname, data))
        else:
            raise ValueError("Cannot write to container opened in read mode")

    def _write_deterministic(self, arcname: str, data: bytes):
        """Write file with deterministic metadata."""
        info = zipfile.ZipInfo(arcname, date_time=DETERMINISTIC_TIMESTAMP)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16  # Unix permissions
        self.zip.writestr(info, data)

    def read_file(self, arcname: str) -> bytes:
        info = self.zip.getinfo(arcname)
        if info.file_size > MAX_FILE_SIZE:
            raise ValueError(f"File {arcname} exceeds safety limit of {MAX_FILE_SIZE} bytes")
        return self.zip.read(arcname)

    def list_files(self) -> List[str]:
        return sorted(self.zip.namelist())

    def get_inventory_hashes(self) -> Dict[str, str]:
        import hashlib
        hashes = {}
        for name in sorted(self.zip.namelist()):
            info = self.zip.getinfo(name)
            if name.endswith("/") or name.startswith("META/"):
                continue

            if info.file_size > MAX_FILE_SIZE:
                raise ValueError(f"File {name} exceeds safety limit of {MAX_FILE_SIZE} bytes")

            h = hashlib.sha256()
            with self.zip.open(name) as f:
                total_read = 0
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    if total_read > MAX_FILE_SIZE:
                        raise ValueError(f"File {name} stream exceeded limit during hash")
                    h.update(chunk)
            hashes[name] = h.hexdigest()
        return hashes

    def close(self):
        """Close the container, writing pending files in sorted order."""
        if self.mode == 'w' and self._pending_files:
            # Sort files by name for deterministic ordering
            for arcname, data in sorted(self._pending_files, key=lambda x: x[0]):
                self._write_deterministic(arcname, data)
            self._pending_files.clear()
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_safely(zf: zipfile.ZipFile, name: str, out_dir: str):
    """Secure extraction helper with path traversal and size protection."""
    info = zf.getinfo(name)
    if info.file_size > MAX_FILE_SIZE:
        raise ValueError(f"Security: {name} is too large ({info.file_size} bytes)")

    # Normalize paths for strict comparison
    abs_out_dir = os.path.abspath(out_dir)
    target_path = os.path.normpath(os.path.join(abs_out_dir, name))

    if not target_path.startswith(abs_out_dir + os.sep) and target_path != abs_out_dir:
        raise ValueError(f"Zip-Slip attempt detected: {name}")

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with zf.open(name) as source, open(target_path, 'wb') as target:
        import shutil
        shutil.copyfileobj(source, target)
