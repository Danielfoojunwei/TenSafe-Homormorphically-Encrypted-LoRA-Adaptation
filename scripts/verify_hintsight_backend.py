#!/usr/bin/env python3
"""
Verify HintSight N2HE backend installation and functionality.

This script checks that:
1. The native module is installed and loadable
2. Key generation works
3. Encrypt/decrypt roundtrip is accurate
4. LoRA delta computation works correctly
5. Basic performance metrics

Usage:
    python scripts/verify_hintsight_backend.py
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---")


def verify_import() -> bool:
    """Verify the native module can be imported."""
    print_section("Checking Import")
    try:
        from crypto_backend.n2he_hintsight import (
            N2HEHintSightBackend,
            N2HEParams,
            verify_backend,
        )
        print("[OK] HintSight N2HE module imported successfully")

        # Check version
        from crypto_backend.n2he_hintsight import _NATIVE_MODULE, _NATIVE_AVAILABLE
        if _NATIVE_AVAILABLE and _NATIVE_MODULE:
            print(f"[OK] Native module version: {_NATIVE_MODULE.__version__}")
            print(f"[OK] Backend name: {_NATIVE_MODULE.BACKEND_NAME}")
        else:
            print("[WARN] Native module not available (will use fallback)")
            return False

        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        print("\nTo fix this, run:")
        print("  ./scripts/build_n2he_hintsight.sh")
        return False


def verify_context_setup() -> bool:
    """Verify context setup and key generation."""
    print_section("Checking Context Setup")
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend, N2HEParams

        # Use default params
        params = N2HEParams.default_lora_params()
        print(f"[OK] Parameters: n={params.n}, q=2^{int(np.log2(params.q))}, t=2^{int(np.log2(params.t))}")

        # Create backend
        backend = N2HEHintSightBackend(params)
        print("[OK] Backend created")

        # Setup context
        start = time.time()
        backend.setup_context()
        setup_time = (time.time() - start) * 1000
        print(f"[OK] Context setup completed ({setup_time:.1f} ms)")

        # Generate keys
        start = time.time()
        sk, pk, ek = backend.generate_keys()
        keygen_time = (time.time() - start) * 1000
        print(f"[OK] Key generation completed ({keygen_time:.1f} ms)")
        print(f"     Secret key size: {len(sk):,} bytes")
        print(f"     Public key size: {len(pk):,} bytes")
        print(f"     Eval key size: {len(ek):,} bytes")

        return True
    except Exception as e:
        print(f"[FAIL] Context setup failed: {e}")
        traceback.print_exc()
        return False


def verify_encrypt_decrypt() -> bool:
    """Verify encrypt/decrypt roundtrip."""
    print_section("Checking Encrypt/Decrypt")
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend

        backend = N2HEHintSightBackend()
        backend.setup_context()
        backend.generate_keys()

        # Test data
        test_vectors = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.5, -0.5, 0.25, -0.25]),
            np.random.randn(16).astype(np.float64),
        ]

        all_passed = True
        for i, data in enumerate(test_vectors):
            # Encrypt
            start = time.time()
            ct = backend.encrypt(data)
            encrypt_time = (time.time() - start) * 1000

            # Decrypt
            start = time.time()
            decrypted = backend.decrypt(ct, len(data))
            decrypt_time = (time.time() - start) * 1000

            # Compute error
            error = np.max(np.abs(data - decrypted))

            # Check accuracy (LWE has higher error than CKKS)
            passed = error < 0.1  # Tolerance for LWE

            status = "[OK]" if passed else "[FAIL]"
            print(f"{status} Test {i+1}: max_error={error:.6f}, encrypt={encrypt_time:.1f}ms, decrypt={decrypt_time:.1f}ms")

            if not passed:
                print(f"     Input:  {data[:4]}...")
                print(f"     Output: {decrypted[:4]}...")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"[FAIL] Encrypt/decrypt failed: {e}")
        traceback.print_exc()
        return False


def verify_lora_delta() -> bool:
    """Verify LoRA delta computation."""
    print_section("Checking LoRA Delta Computation")
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend

        backend = N2HEHintSightBackend()
        backend.setup_context()
        backend.generate_keys()

        # Test parameters
        in_dim = 4
        rank = 8
        out_dim = 4
        scaling = 0.5

        # Create test data
        x = np.array([1.0, 2.0, 3.0, 4.0])
        lora_a = np.random.randn(rank, in_dim).astype(np.float64) * 0.1
        lora_b = np.random.randn(out_dim, rank).astype(np.float64) * 0.1

        # Compute expected (plaintext)
        expected = scaling * (x @ lora_a.T @ lora_b.T)

        # Compute encrypted
        start = time.time()
        ct_x = backend.encrypt(x)
        encrypt_time = (time.time() - start) * 1000

        start = time.time()
        ct_delta = backend.lora_delta(ct_x, lora_a, lora_b, scaling)
        compute_time = (time.time() - start) * 1000

        start = time.time()
        decrypted = backend.decrypt(ct_delta, out_dim)
        decrypt_time = (time.time() - start) * 1000

        # Compute error
        error = np.max(np.abs(expected - decrypted))

        # Check accuracy (LWE has higher error)
        passed = error < 0.5  # Higher tolerance for chained operations

        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} LoRA delta: max_error={error:.6f}")
        print(f"     in_dim={in_dim}, rank={rank}, out_dim={out_dim}, scaling={scaling}")
        print(f"     encrypt={encrypt_time:.1f}ms, compute={compute_time:.1f}ms, decrypt={decrypt_time:.1f}ms")
        print(f"     Expected: {expected}")
        print(f"     Got:      {decrypted}")

        return passed
    except Exception as e:
        print(f"[FAIL] LoRA delta failed: {e}")
        traceback.print_exc()
        return False


def verify_homomorphic_ops() -> bool:
    """Verify homomorphic operations."""
    print_section("Checking Homomorphic Operations")
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend

        backend = N2HEHintSightBackend()
        backend.setup_context()
        backend.generate_keys()

        # Test data
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([0.5, 1.0, 1.5, 2.0])
        scalar = 2.0

        # Encrypt
        ct_a = backend.encrypt(a)
        ct_b = backend.encrypt(b)

        # Test addition
        ct_sum = backend.add(ct_a, ct_b)
        decrypted_sum = backend.decrypt(ct_sum, len(a))
        expected_sum = a + b
        sum_error = np.max(np.abs(expected_sum - decrypted_sum))
        sum_passed = sum_error < 0.1

        status = "[OK]" if sum_passed else "[FAIL]"
        print(f"{status} Addition: max_error={sum_error:.6f}")

        # Test scalar multiplication
        ct_scaled = backend.multiply_plain(ct_a, np.array([scalar]))
        decrypted_scaled = backend.decrypt(ct_scaled, len(a))
        expected_scaled = a * scalar
        scale_error = np.max(np.abs(expected_scaled - decrypted_scaled))
        scale_passed = scale_error < 0.2  # Higher tolerance after multiplication

        status = "[OK]" if scale_passed else "[FAIL]"
        print(f"{status} Scalar multiply: max_error={scale_error:.6f}")

        return sum_passed and scale_passed
    except Exception as e:
        print(f"[FAIL] Homomorphic ops failed: {e}")
        traceback.print_exc()
        return False


def verify_matmul() -> bool:
    """Verify matrix multiplication."""
    print_section("Checking Matrix Multiplication")
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend

        backend = N2HEHintSightBackend()
        backend.setup_context()
        backend.generate_keys()

        # Test data
        x = np.array([1.0, 2.0, 3.0, 4.0])
        W = np.random.randn(4, 4).astype(np.float64) * 0.1

        # Expected
        expected = x @ W.T

        # Encrypted
        start = time.time()
        ct_x = backend.encrypt(x)
        ct_result = backend.matmul(ct_x, W)
        decrypted = backend.decrypt(ct_result, 4)
        total_time = (time.time() - start) * 1000

        # Error
        error = np.max(np.abs(expected - decrypted))
        passed = error < 0.5  # Higher tolerance for matmul

        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} Matrix multiply (4x4): max_error={error:.6f}, time={total_time:.1f}ms")
        print(f"     Expected: {expected}")
        print(f"     Got:      {decrypted}")

        return passed
    except Exception as e:
        print(f"[FAIL] Matrix multiply failed: {e}")
        traceback.print_exc()
        return False


def run_quick_benchmark() -> None:
    """Run a quick benchmark."""
    print_section("Quick Benchmark")
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend

        backend = N2HEHintSightBackend()
        backend.setup_context()
        backend.generate_keys()

        # Benchmark parameters
        iterations = 10
        vector_size = 64

        data = np.random.randn(vector_size).astype(np.float64)

        # Encrypt benchmark
        start = time.time()
        for _ in range(iterations):
            ct = backend.encrypt(data)
        encrypt_time = (time.time() - start) / iterations * 1000

        # Decrypt benchmark
        start = time.time()
        for _ in range(iterations):
            _ = backend.decrypt(ct, vector_size)
        decrypt_time = (time.time() - start) / iterations * 1000

        # LoRA delta benchmark
        lora_a = np.random.randn(16, vector_size).astype(np.float64) * 0.1
        lora_b = np.random.randn(vector_size, 16).astype(np.float64) * 0.1

        ct = backend.encrypt(data)
        start = time.time()
        for _ in range(iterations):
            ct_delta = backend.lora_delta(ct, lora_a, lora_b, 0.5)
        lora_time = (time.time() - start) / iterations * 1000

        print(f"Vector size: {vector_size}")
        print(f"Iterations: {iterations}")
        print(f"Avg encrypt time: {encrypt_time:.2f} ms")
        print(f"Avg decrypt time: {decrypt_time:.2f} ms")
        print(f"Avg LoRA delta time: {lora_time:.2f} ms (rank=16)")

        # Get stats
        stats = backend.get_operation_stats()
        print(f"Total operations: {stats.get('operations', 0)}")
        print(f"Total multiplications: {stats.get('multiplications', 0)}")
        print(f"Total additions: {stats.get('additions', 0)}")

    except Exception as e:
        print(f"[WARN] Benchmark failed: {e}")


def main() -> int:
    """Run all verification checks."""
    print_header("HintSight N2HE Backend Verification")

    results = []

    # Run checks
    results.append(("Import", verify_import()))

    if results[-1][1]:  # Only continue if import succeeded
        results.append(("Context Setup", verify_context_setup()))
        results.append(("Encrypt/Decrypt", verify_encrypt_decrypt()))
        results.append(("Homomorphic Ops", verify_homomorphic_ops()))
        results.append(("Matrix Multiply", verify_matmul()))
        results.append(("LoRA Delta", verify_lora_delta()))

        # Run benchmark
        run_quick_benchmark()

    # Summary
    print_header("Summary")
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("SUCCESS: HintSight N2HE backend is properly installed and functional!")
        return 0
    else:
        print("FAILURE: Some checks failed. See details above.")
        print("\nTo fix, try rebuilding:")
        print("  ./scripts/build_n2he_hintsight.sh --clean")
        return 1


if __name__ == "__main__":
    sys.exit(main())
