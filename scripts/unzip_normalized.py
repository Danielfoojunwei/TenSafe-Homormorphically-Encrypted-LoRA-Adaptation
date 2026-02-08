#!/usr/bin/env python3
"""
Normalize ZIP extraction paths from Windows to POSIX.

TenSafe_v4.1.0_Final.zip may contain Windows-style backslash paths
(e.g., "src\\tensorguard\\__init__.py") that fail to extract as
proper directory trees on Linux/macOS.

This script reads the zip, replaces every "\\" with "/", and writes
files to the output directory with POSIX paths.

Usage:
    python scripts/unzip_normalized.py --zip TenSafe_v4.1.0_Final.zip --out ./tensafe_src
    python scripts/unzip_normalized.py --zip TenSafe_v4.1.0_Final.zip --out ./tensafe_src --force
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath


def normalize_and_extract(zip_path: str, out_dir: str, force: bool = False) -> int:
    """Extract a zip with POSIX-normalized paths.

    Returns:
        Number of files extracted.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)

    if not zip_path.exists():
        print(f"ERROR: {zip_path} does not exist", file=sys.stderr)
        return -1

    if out_dir.exists() and not force:
        print(
            f"ERROR: {out_dir} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return -1

    count = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # Normalize: replace Windows backslashes with POSIX forward slashes
            normalized = info.filename.replace("\\", "/")

            # Skip directory entries
            if normalized.endswith("/"):
                (out_dir / normalized).mkdir(parents=True, exist_ok=True)
                continue

            # Security: reject absolute paths and path traversal
            posix = PurePosixPath(normalized)
            if posix.is_absolute() or ".." in posix.parts:
                print(f"  SKIP (unsafe path): {info.filename}", file=sys.stderr)
                skipped += 1
                continue

            dest = out_dir / normalized
            dest.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            count += 1

    print(f"Extracted {count} files to {out_dir}")
    if skipped:
        print(f"Skipped {skipped} unsafe paths")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract a ZIP with POSIX-normalized paths"
    )
    parser.add_argument("--zip", required=True, help="Path to the ZIP file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output directory"
    )
    args = parser.parse_args()

    result = normalize_and_extract(args.zip, args.out, args.force)
    sys.exit(0 if result >= 0 else 1)


if __name__ == "__main__":
    main()
