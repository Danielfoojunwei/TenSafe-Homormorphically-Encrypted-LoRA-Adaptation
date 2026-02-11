"""
Test that the MOAIContext alias is properly exported from crypto_backend.ckks_moai.

execution_policy.py imports MOAIContext; the actual class is CKKSMOAIBackend.
The alias must always be available so the import path never breaks.

These tests require numpy (a core dependency) to be installed, since
crypto_backend.ckks_moai imports numpy at module level.
"""

import pytest

try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


@needs_numpy
def test_moai_context_importable():
    """MOAIContext symbol must be importable from crypto_backend.ckks_moai."""
    from crypto_backend.ckks_moai import MOAIContext
    assert MOAIContext is not None


@needs_numpy
def test_moai_context_is_ckks_backend():
    """MOAIContext should be the same class as CKKSMOAIBackend."""
    from crypto_backend.ckks_moai import CKKSMOAIBackend, MOAIContext
    assert MOAIContext is CKKSMOAIBackend


@needs_numpy
def test_moai_context_in_all():
    """MOAIContext must be listed in __all__."""
    import crypto_backend.ckks_moai as mod
    assert "MOAIContext" in mod.__all__
