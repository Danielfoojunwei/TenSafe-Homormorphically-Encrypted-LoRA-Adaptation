"""
Deterministic TAR creation for TGSP packages.

SECURITY NOTE: This module creates reproducible tarballs by normalizing:
- File ordering (sorted alphabetically)
- Timestamps (fixed to epoch 0)
- Ownership (uid/gid = 0)
- Permissions (644 for files, 755 for directories/executables)
"""

import io
import os
import tarfile


def deterministic_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """
    Normalize tar info for determinism.

    All metadata that could vary between builds is normalized:
    - uid/gid set to 0
    - uname/gname cleared
    - mtime set to 0 (Unix epoch)
    - permissions normalized
    """
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0

    # Normalize permissions
    if tarinfo.isdir():
        tarinfo.mode = 0o755
    else:
        # Preserve executable bit if set, otherwise use 644
        if tarinfo.mode & 0o100:
            tarinfo.mode = 0o755
        else:
            tarinfo.mode = 0o644

    return tarinfo


def create_deterministic_tar(source_dir: str, output_path: str = None) -> bytes:
    """
    Create a deterministic tarball of a directory.

    Args:
        source_dir: Directory to archive
        output_path: If provided, writes to this file path.
                    If None, returns bytes.

    Returns:
        Tarball bytes if output_path is None, otherwise empty bytes.

    The tarball is deterministic because:
    - Files are added in sorted order
    - All metadata is normalized via deterministic_filter
    - USTAR format is used for compatibility
    """
    # Collect all files first for sorting
    files_to_add = []
    for root, dirs, files in os.walk(source_dir):
        # Sort directories in-place for consistent recursion order
        dirs.sort()

        for filename in sorted(files):
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, source_dir)
            # Normalize path separators to forward slash
            arcname = rel_path.replace(os.sep, "/")
            files_to_add.append((full_path, arcname))

    # Sort by archive name for deterministic ordering
    files_to_add.sort(key=lambda x: x[1])

    # Create tarball
    if output_path is None:
        # Write to BytesIO
        bio = io.BytesIO()
        with tarfile.open(fileobj=bio, mode="w:gz", format=tarfile.USTAR_FORMAT) as tar:
            for full_path, arcname in files_to_add:
                tar.add(full_path, arcname=arcname, recursive=False, filter=deterministic_filter)
        return bio.getvalue()
    else:
        # Write to file
        with tarfile.open(name=output_path, mode="w:gz", format=tarfile.USTAR_FORMAT) as tar:
            for full_path, arcname in files_to_add:
                tar.add(full_path, arcname=arcname, recursive=False, filter=deterministic_filter)
        return b""


def verify_determinism(source_dir: str, iterations: int = 3) -> bool:
    """
    Verify that create_deterministic_tar produces identical output.

    Args:
        source_dir: Directory to test
        iterations: Number of times to create tarball

    Returns:
        True if all iterations produce identical bytes
    """
    import hashlib

    hashes = []
    for _ in range(iterations):
        tar_bytes = create_deterministic_tar(source_dir)
        h = hashlib.sha256(tar_bytes).hexdigest()
        hashes.append(h)

    return len(set(hashes)) == 1
