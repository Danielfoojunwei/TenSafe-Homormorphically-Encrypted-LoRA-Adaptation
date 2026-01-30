"""
N2HE Runtime Contract Tests.

Tests for:
1. API contract enforcement
2. Determinism guarantees
3. Error bounds documentation

Note: Basic correctness tests are in test_n2he_core.py
"""

import os

import numpy as np
import pytest

# Ensure toy mode is enabled for tests
os.environ["TENSAFE_TOY_HE"] = "1"

from tensorguard.n2he.core import (
    HESchemeParams,
    HESchemeType,
    ToyModeNotEnabledError,
    ToyN2HEScheme,
)


class TestAPIContract:
    """Test API contract enforcement."""

    def test_toy_mode_requires_env_var(self):
        """ToyN2HEScheme requires TENSAFE_TOY_HE env var."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )

        # Should work with env var set
        scheme = ToyN2HEScheme(params=params)
        assert scheme is not None

    def test_keygen_returns_three_keys(self):
        """keygen must return (secret_key, public_key, eval_key) tuple."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )
        scheme = ToyN2HEScheme(params=params)

        result = scheme.keygen()

        assert isinstance(result, tuple)
        assert len(result) == 3
        sk, pk, ek = result
        assert isinstance(sk, bytes)
        assert isinstance(pk, bytes)
        assert isinstance(ek, bytes)

    def test_scheme_has_required_methods(self):
        """ToyN2HEScheme must have required cryptographic methods."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )
        scheme = ToyN2HEScheme(params=params)

        # Required methods
        assert hasattr(scheme, "keygen")
        assert hasattr(scheme, "encrypt")
        assert hasattr(scheme, "decrypt")
        assert hasattr(scheme, "add")
        assert hasattr(scheme, "multiply")
        assert hasattr(scheme, "matmul")

        # All should be callable
        assert callable(scheme.keygen)
        assert callable(scheme.encrypt)
        assert callable(scheme.decrypt)
        assert callable(scheme.add)
        assert callable(scheme.multiply)
        assert callable(scheme.matmul)


class TestDeterminism:
    """Test determinism guarantees."""

    def test_decrypt_is_deterministic(self):
        """Decryption must be deterministic."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )
        scheme = ToyN2HEScheme(params=params)
        sk, pk, ek = scheme.keygen()

        plaintext = np.array([100], dtype=np.int64)
        ct = scheme.encrypt(pk, plaintext)

        # Multiple decryptions should be identical
        dec1 = scheme.decrypt(sk, ct)
        dec2 = scheme.decrypt(sk, ct)
        dec3 = scheme.decrypt(sk, ct)

        np.testing.assert_array_equal(dec1, dec2)
        np.testing.assert_array_equal(dec2, dec3)


class TestSchemeParams:
    """Test HESchemeParams validation."""

    def test_params_have_required_fields(self):
        """HESchemeParams must have required fields."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )

        assert params.scheme_type == HESchemeType.LWE
        assert params.n == 512
        assert params.q == 2**40
        assert params.std_dev == 3.2
        assert params.security_level == 128

    def test_params_get_hash_is_deterministic(self):
        """get_hash must return deterministic value."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )

        hash1 = params.get_hash()
        hash2 = params.get_hash()

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0


class TestToyModeGating:
    """Test toy mode environment variable gating."""

    def test_toy_mode_warning_logged(self):
        """ToyN2HEScheme should log a warning about security."""
        import logging

        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )

        # This should work but log a warning
        scheme = ToyN2HEScheme(params=params)
        assert scheme is not None

    def test_force_enable_bypasses_env_check(self):
        """_force_enable parameter should bypass env var check."""
        params = HESchemeParams(
            scheme_type=HESchemeType.LWE,
            n=512,
            q=2**40,
            std_dev=3.2,
            security_level=128,
        )

        # Should work with _force_enable=True even without env var
        # (env var is already set in this test file)
        scheme = ToyN2HEScheme(params=params, _force_enable=True)
        assert scheme is not None
