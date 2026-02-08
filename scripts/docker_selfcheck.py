#!/usr/bin/env python3
"""
Docker container startup self-check.

Validates that the TenSafe installation is functional:
1. Core packages importable (tensorguard, tensafe, tg_tinker)
2. HE backend available (tenseal OR crypto_backend.ckks_moai OR explicitly disabled)
3. Version consistency across packages

Exit codes:
  0  All checks passed (or HE explicitly disabled via TENSAFE_SKIP_HE_CHECK=1)
  1  One or more checks failed
"""

import os
import sys


def check_core_imports():
    """Verify core packages are importable."""
    errors = []
    for pkg in ("tensorguard", "tensafe", "tg_tinker"):
        try:
            __import__(pkg)
        except ImportError as e:
            errors.append(f"  FAIL: import {pkg} -> {e}")
    return errors


def check_version_consistency():
    """Verify all packages report the same version."""
    import tensorguard
    import tensafe
    import tg_tinker

    versions = {
        "tensorguard": tensorguard.__version__,
        "tensafe": tensafe.__version__,
        "tg_tinker": tg_tinker.__version__,
    }

    unique = set(versions.values())
    if len(unique) > 1:
        parts = ", ".join(f"{k}={v}" for k, v in versions.items())
        return [f"  FAIL: version mismatch: {parts}"]
    return []


def check_he_backend():
    """Check that at least one HE backend is available."""
    skip = os.getenv("TENSAFE_SKIP_HE_CHECK", "0") in ("1", "true", "yes")
    if skip:
        print("  SKIP: HE check disabled via TENSAFE_SKIP_HE_CHECK=1")
        return []

    backends_tried = []

    # Try tenseal
    try:
        import tenseal  # noqa: F401
        print(f"  OK: tenseal {tenseal.__version__} available")
        return []
    except ImportError:
        backends_tried.append("tenseal")

    # Try crypto_backend.ckks_moai
    try:
        import crypto_backend.ckks_moai  # noqa: F401
        print("  OK: crypto_backend.ckks_moai available")
        return []
    except ImportError:
        backends_tried.append("crypto_backend.ckks_moai")

    # Try native n2he
    try:
        import n2he  # noqa: F401
        print("  OK: n2he native available")
        return []
    except ImportError:
        backends_tried.append("n2he")

    return [
        f"  FAIL: No HE backend found (tried: {', '.join(backends_tried)}). "
        "Set TENSAFE_SKIP_HE_CHECK=1 to explicitly run without HE."
    ]


def main():
    print("=" * 60)
    print("TenSafe Docker Self-Check")
    print("=" * 60)

    all_errors = []

    print("\n[1/3] Core imports...")
    errors = check_core_imports()
    all_errors.extend(errors)
    if not errors:
        print("  OK: all core packages importable")

    print("\n[2/3] Version consistency...")
    errors = check_version_consistency()
    all_errors.extend(errors)
    if not errors:
        import tensorguard
        print(f"  OK: all packages at v{tensorguard.__version__}")

    print("\n[3/3] HE backend availability...")
    errors = check_he_backend()
    all_errors.extend(errors)

    print()
    if all_errors:
        print("SELF-CHECK FAILED:")
        for e in all_errors:
            print(e)
        sys.exit(1)
    else:
        print("SELF-CHECK PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
