"""
Guard tests to ensure no toy/simulated HE implementations in production code.

These tests MUST fail if any of the following are detected:
1. Ciphertext implemented as torch.Tensor or np.ndarray
2. encrypt() returning tensor/ndarray
3. Conditional bypass like `if not he_enabled: return plaintext`
4. Any stub that "pretends" to encrypt

This is a critical security test - DO NOT SKIP.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Tuple

import pytest

# Directories to scan for toy HE patterns
HE_CRITICAL_DIRS = [
    "src/tensorguard/n2he",
    "tensafe/he_lora",
    "crypto_backend",
]

# Patterns that indicate toy/mock HE (FORBIDDEN)
FORBIDDEN_PATTERNS = [
    # Ciphertext as tensor/array
    (r"class\s+Ciphertext.*:\s*\n.*torch\.Tensor", "Ciphertext inherits/wraps torch.Tensor"),
    (r"class\s+Ciphertext.*:\s*\n.*np\.ndarray", "Ciphertext inherits/wraps np.ndarray"),
    (r"self\.data\s*=\s*.*\.astype\(", "Ciphertext stores numpy array directly"),

    # Plaintext bypass patterns
    (r"if\s+not\s+he_enabled.*:\s*\n\s*return\s+plaintext", "Plaintext fallback when HE disabled"),
    (r"if\s+not\s+self\._he_backend.*:\s*\n\s*return", "Backend bypass returning plaintext"),
    (r"if.*DEBUG.*:\s*\n\s*return.*plain", "Debug plaintext bypass"),
    (r"if.*TOY.*:\s*\n\s*return", "Toy mode bypass"),

    # Fake encryption patterns
    (r"def\s+encrypt.*:\s*\n\s*return\s+.*\.copy\(\)", "Encrypt just copies data"),
    (r"def\s+encrypt.*:\s*\n\s*return\s+np\.", "Encrypt returns numpy array"),
    (r"def\s+encrypt.*:\s*\n\s*return\s+torch\.", "Encrypt returns torch tensor"),

    # Simulation class names in production code (outside tests/)
    (r"class\s+ToyN2HEScheme", "ToyN2HEScheme in production code"),
    (r"class\s+SimulatedN2HEScheme", "SimulatedN2HEScheme in production code"),
    (r"class\s+Mock.*Scheme", "Mock HE scheme in production code"),
    (r"class\s+Fake.*Ciphertext", "Fake ciphertext in production code"),
]

# Allowed exceptions (file patterns where toy code is acceptable)
ALLOWED_EXCEPTIONS = [
    r"tests/.*",           # Test files can have mocks
    r".*_test\.py$",       # Test files
    r".*test_.*\.py$",     # Test files
    r"conftest\.py$",      # Pytest fixtures
    r".*mock.*\.py$",      # Explicit mock files (only for tests)
]


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def should_skip_file(filepath: Path) -> bool:
    """Check if file should be skipped (e.g., test files)."""
    rel_path = str(filepath.relative_to(get_project_root()))
    for pattern in ALLOWED_EXCEPTIONS:
        if re.match(pattern, rel_path):
            return True
    return False


def scan_file_for_patterns(filepath: Path) -> List[Tuple[str, int, str]]:
    """
    Scan a file for forbidden toy HE patterns.

    Returns:
        List of (pattern_desc, line_number, matched_text) tuples
    """
    violations = []

    try:
        content = filepath.read_text()
    except Exception:
        return violations

    for pattern, description in FORBIDDEN_PATTERNS:
        for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
            # Find line number
            line_num = content[:match.start()].count('\n') + 1
            matched_text = match.group(0)[:100]  # Truncate for readability
            violations.append((description, line_num, matched_text))

    return violations


def scan_for_toy_he_imports(filepath: Path) -> List[Tuple[str, int]]:
    """
    Scan for imports of toy/simulated HE modules.

    Returns:
        List of (import_statement, line_number) tuples
    """
    violations = []

    try:
        content = filepath.read_text()
        tree = ast.parse(content)
    except Exception:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if 'toy' in alias.name.lower() or 'simulated' in alias.name.lower():
                    violations.append((f"import {alias.name}", node.lineno))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if 'toy' in module.lower() or 'simulated' in module.lower():
                violations.append((f"from {module} import ...", node.lineno))

            for alias in node.names:
                if 'toy' in alias.name.lower() or 'simulated' in alias.name.lower():
                    violations.append((f"from {module} import {alias.name}", node.lineno))

    return violations


class TestNoToyHE:
    """Test suite to detect toy HE implementations."""

    def test_no_forbidden_patterns_in_he_modules(self):
        """Scan HE-critical modules for forbidden toy patterns."""
        root = get_project_root()
        all_violations = []

        for dir_path in HE_CRITICAL_DIRS:
            target_dir = root / dir_path
            if not target_dir.exists():
                continue

            for filepath in target_dir.rglob("*.py"):
                if should_skip_file(filepath):
                    continue

                violations = scan_file_for_patterns(filepath)
                if violations:
                    rel_path = filepath.relative_to(root)
                    for desc, line_num, text in violations:
                        all_violations.append(f"{rel_path}:{line_num}: {desc}\n  {text}")

        if all_violations:
            pytest.fail(
                f"Found {len(all_violations)} toy HE pattern(s) in production code:\n\n"
                + "\n\n".join(all_violations)
            )

    def test_no_toy_imports_in_he_modules(self):
        """Ensure no imports of toy/simulated HE modules in production code."""
        root = get_project_root()
        all_violations = []

        for dir_path in HE_CRITICAL_DIRS:
            target_dir = root / dir_path
            if not target_dir.exists():
                continue

            for filepath in target_dir.rglob("*.py"):
                if should_skip_file(filepath):
                    continue

                violations = scan_for_toy_he_imports(filepath)
                if violations:
                    rel_path = filepath.relative_to(root)
                    for import_stmt, line_num in violations:
                        all_violations.append(f"{rel_path}:{line_num}: {import_stmt}")

        if all_violations:
            pytest.fail(
                f"Found {len(all_violations)} toy HE import(s) in production code:\n\n"
                + "\n".join(all_violations)
            )

    def test_he_backend_available(self):
        """Verify that the real HE backend is available."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            assert backend.is_available(), "N2HE-HEXL backend not available"

            # Verify it reports real CKKS parameters
            params = backend.get_context_params()
            assert params.get("ring_degree") is not None, "Missing ring_degree"
            assert params.get("coeff_modulus_chain_length") is not None, "Missing coeff_modulus"
            assert params.get("has_galois_keys") is True, "Missing Galois keys"

        except ImportError as e:
            pytest.fail(
                f"N2HE-HEXL backend not installed: {e}\n"
                "Run: ./scripts/build_n2he_hexl.sh"
            )

    def test_ciphertext_is_not_tensor(self):
        """Verify Ciphertext class is not a thin wrapper around tensors."""
        try:
            import numpy as np
            from crypto_backend.n2he_hexl import CKKSCiphertext

            # Check that ciphertext doesn't expose raw tensor data
            assert not hasattr(CKKSCiphertext, 'data'), "Ciphertext exposes raw data attribute"
            assert not hasattr(CKKSCiphertext, 'tensor'), "Ciphertext exposes tensor attribute"

        except ImportError:
            pytest.fail("N2HE-HEXL backend not available for verification")

    def test_encrypt_returns_real_ciphertext(self):
        """Verify encrypt() returns real ciphertext, not tensor."""
        try:
            import numpy as np
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            backend.setup_context()
            backend.generate_keys()

            # Encrypt some test data
            plaintext = np.array([1.0, 2.0, 3.0, 4.0])
            ciphertext = backend.encrypt(plaintext)

            # Verify it's not just a tensor
            assert not isinstance(ciphertext, np.ndarray), "Encrypt returned numpy array"

            try:
                import torch
                assert not isinstance(ciphertext, torch.Tensor), "Encrypt returned torch tensor"
            except ImportError:
                pass  # torch not installed, skip this check

            # Verify ciphertext has expected HE metadata
            assert hasattr(ciphertext, 'level') or hasattr(ciphertext, 'get_level'), (
                "Ciphertext missing level information"
            )
            assert hasattr(ciphertext, 'scale') or hasattr(ciphertext, 'get_scale'), (
                "Ciphertext missing scale information"
            )

        except ImportError:
            pytest.fail("N2HE-HEXL backend not available for verification")


class TestNoPlaintextBypass:
    """Tests to ensure no plaintext bypass paths exist."""

    def test_encrypt_fails_without_keys(self):
        """Encrypt should fail without proper key setup, not fall back to plaintext."""
        try:
            import numpy as np
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            # Don't call setup_context() or generate_keys()

            plaintext = np.array([1.0, 2.0, 3.0])

            with pytest.raises(Exception):  # Should raise, not return plaintext
                backend.encrypt(plaintext)

        except ImportError:
            pytest.fail("N2HE-HEXL backend not available")

    def test_no_environment_variable_bypass(self):
        """Ensure no environment variables can bypass HE."""

        # Save original env
        original_env = os.environ.copy()

        try:
            # Set various "bypass" environment variables
            bypass_vars = [
                "TENSAFE_TOY_HE",
                "DISABLE_HE",
                "DEBUG_HE",
                "HE_PLAINTEXT_MODE",
                "NO_ENCRYPTION",
            ]

            for var in bypass_vars:
                os.environ[var] = "1"

            # Try to import and use the real backend
            # It should NOT fall back to toy mode
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()

            # Even with bypass vars set, should still be real backend
            assert backend.is_available(), "Backend bypassed with env var"
            assert backend.get_backend_name() == "N2HE-HEXL", "Wrong backend loaded"

        except ImportError:
            pytest.fail("N2HE-HEXL backend not available")
        finally:
            # Restore original env
            os.environ.clear()
            os.environ.update(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
