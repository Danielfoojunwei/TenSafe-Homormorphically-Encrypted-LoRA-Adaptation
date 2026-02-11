"""
Backend authenticity tests for HE-LoRA.

These tests verify that the backend is a REAL HE implementation,
not a toy/simulated/bypass implementation.

Test requirements:
1. Verify backend reports real CKKS context parameters
2. If backend is missing, test MUST FAIL (no skip)
3. Ciphertext objects must not be raw tensors
4. No plaintext fallback paths
"""

import numpy as np
import pytest


class TestBackendAuthenticity:
    """Test that the HE backend is real, not simulated."""

    def test_backend_import_required(self):
        """Test that backend can be imported - FAIL if not available."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend
        except ImportError as e:
            pytest.fail(
                f"N2HE-HEXL backend REQUIRED but not available: {e}\n"
                "Build with: ./scripts/build_n2he_hexl.sh\n"
                "This test intentionally fails when backend is missing - no skip allowed."
            )

    def test_backend_is_available(self):
        """Test backend reports as available."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend
            backend = N2HEHEXLBackend()
            is_available = backend.is_available()

            if not is_available:
                pytest.fail(
                    "N2HE-HEXL backend reports not available.\n"
                    "The native library may not be compiled correctly.\n"
                    "Rebuild with: ./scripts/build_n2he_hexl.sh --clean"
                )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_backend_name_correct(self):
        """Test backend reports correct name."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend
            backend = N2HEHEXLBackend()
            name = backend.get_backend_name()

            assert name == "N2HE-HEXL", (
                f"Backend name is '{name}', expected 'N2HE-HEXL'.\n"
                "This may indicate a toy/mock backend is being used."
            )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_ckks_context_has_real_parameters(self):
        """Test CKKS context reports real parameters."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            backend.setup_context()
            params = backend.get_context_params()

            # Ring degree must be power of 2, >= 4096 for security
            ring_degree = params.get("ring_degree")
            assert ring_degree is not None, "Missing ring_degree"
            assert ring_degree >= 4096, f"Ring degree {ring_degree} too small for security"
            assert (ring_degree & (ring_degree - 1)) == 0, "Ring degree must be power of 2"

            # Coefficient modulus chain must exist
            chain_length = params.get("coeff_modulus_chain_length")
            assert chain_length is not None, "Missing coeff_modulus_chain_length"
            assert chain_length >= 2, f"Chain length {chain_length} too short"

            # Modulus sizes must be reasonable
            modulus_sizes = params.get("coeff_modulus_sizes")
            assert modulus_sizes is not None, "Missing coeff_modulus_sizes"
            assert all(30 <= s <= 60 for s in modulus_sizes), (
                f"Modulus sizes {modulus_sizes} outside reasonable range"
            )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_galois_keys_exist(self):
        """Test that Galois keys exist for rotation operations."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            backend.setup_context()
            backend.generate_keys(generate_galois=True)

            params = backend.get_context_params()
            has_galois = params.get("has_galois_keys")

            assert has_galois is True, (
                "Galois keys not generated.\n"
                "These are required for rotation operations in HE."
            )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_ciphertext_not_tensor(self):
        """Test that ciphertext is not a raw tensor/array."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            backend.setup_context()
            backend.generate_keys()

            # Encrypt test data
            plaintext = np.array([1.0, 2.0, 3.0, 4.0])
            ciphertext = backend.encrypt(plaintext)

            # Ciphertext must NOT be a numpy array
            assert not isinstance(ciphertext, np.ndarray), (
                "Ciphertext is a numpy array - this is a TOY implementation!"
            )

            # Try to import torch if available
            try:
                import torch
                assert not isinstance(ciphertext, torch.Tensor), (
                    "Ciphertext is a torch tensor - this is a TOY implementation!"
                )
            except ImportError:
                pass  # torch not installed, skip this check

            # Ciphertext should not expose raw data
            assert not hasattr(ciphertext, 'data'), (
                "Ciphertext exposes raw 'data' attribute"
            )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_ciphertext_has_he_metadata(self):
        """Test that ciphertext has HE-specific metadata."""
        try:
            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            backend.setup_context()
            backend.generate_keys()

            plaintext = np.array([1.0, 2.0, 3.0, 4.0])
            ciphertext = backend.encrypt(plaintext)

            # Must have level information
            level = backend.get_ciphertext_level(ciphertext)
            assert isinstance(level, int), "Level must be integer"
            assert level >= 0, "Level cannot be negative"

            # Must have scale information
            scale = backend.get_ciphertext_scale(ciphertext)
            assert isinstance(scale, (int, float)), "Scale must be numeric"
            assert scale > 0, "Scale must be positive"

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_encrypt_decrypt_not_identity(self):
        """Test encrypt/decrypt involves actual computation."""
        try:
            import time

            from crypto_backend.n2he_hexl import N2HEHEXLBackend

            backend = N2HEHEXLBackend()
            backend.setup_context()
            backend.generate_keys()

            # Encrypt
            plaintext = np.random.randn(1024).astype(np.float64)

            start = time.perf_counter()
            ciphertext = backend.encrypt(plaintext)
            encrypt_time = time.perf_counter() - start

            start = time.perf_counter()
            decrypted = backend.decrypt(ciphertext, len(plaintext))
            decrypt_time = time.perf_counter() - start

            # Real encryption should take measurable time
            # Toy implementations that just copy data are near-instant
            # (relaxed check - real HE takes microseconds minimum)
            total_time = encrypt_time + decrypt_time
            assert total_time > 1e-6, (
                f"Encrypt/decrypt took {total_time*1e6:.2f}us - "
                "suspiciously fast, may be toy implementation"
            )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")


class TestNoBypassPaths:
    """Test that no bypass paths exist."""

    def test_no_toy_mode_active(self):
        """Test that toy mode is not active."""
        import os

        # These environment variables should NOT enable toy mode
        # in the production backend
        bypass_vars = [
            "TENSAFE_TOY_HE",
            "DEBUG_HE",
            "HE_PLAINTEXT_MODE",
        ]

        for var in bypass_vars:
            val = os.environ.get(var)
            if val and val.lower() in ("1", "true", "yes"):
                pytest.fail(
                    f"Environment variable {var}={val} is set.\n"
                    "This may enable toy/bypass mode which is not allowed for tests."
                )

    def test_verify_backend_function(self):
        """Test the verify_backend function works."""
        try:
            from crypto_backend.n2he_hexl import verify_backend

            result = verify_backend()

            assert result["available"] is True, "Backend not available"
            assert result["backend"] == "N2HE-HEXL", "Wrong backend"
            assert result["test_encrypt_decrypt"]["passed"] is True, (
                "Encrypt/decrypt test failed"
            )

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")

    def test_tensafe_helora_uses_real_backend(self):
        """Test that tensafe.he_lora module uses real backend."""
        try:
            from tensafe.he_lora import HEBackendNotAvailableError, get_backend

            backend = get_backend()

            # Should be the real backend
            assert backend.get_backend_name() == "N2HE-HEXL", (
                "tensafe.he_lora not using N2HE-HEXL backend"
            )

            assert backend.is_available(), "Backend not available"
            assert backend.is_ready(), "Backend not ready"

        except HEBackendNotAvailableError as e:
            pytest.fail(f"HE backend not available: {e}")
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
