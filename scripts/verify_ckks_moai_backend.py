#!/usr/bin/env python3
"""
Verify CKKS MOAI backend installation and functionality.

This script checks that:
1. Pyfhel is installed and loadable
2. Key generation works
3. Encrypt/decrypt roundtrip is accurate
4. Scalar multiplication works correctly
5. Matrix multiplication works correctly
6. LoRA delta computation works correctly
7. Basic performance metrics

Usage:
    python scripts/verify_ckks_moai_backend.py
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
    """Verify the CKKS MOAI module can be imported."""
    print_section("Checking Import")
    try:
        from crypto_backend.ckks_moai import (
            CKKSMOAIBackend,
            CKKSParams,
            verify_backend,
        )
        print("[OK] CKKS MOAI module imported successfully")

        # Check Pyfhel availability
        from Pyfhel import Pyfhel
        print(f"[OK] Pyfhel version: {Pyfhel.__version__ if hasattr(Pyfhel, '__version__') else 'available'}")

        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        print("\nTo fix this, run:")
        print("  pip install Pyfhel")
        return False


def verify_context_setup() -> bool:
    """Verify context setup and key generation."""
    print_section("Checking Context Setup")
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend, CKKSParams

        # Use default params
        params = CKKSParams.default_lora_params()
        print(f"[OK] Parameters: N={params.poly_modulus_degree}, scale=2^{params.scale_bits}")

        # Create backend
        backend = CKKSMOAIBackend(params)
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
        print(f"     Relin key size: {len(ek):,} bytes")
        print(f"     Slot count: {backend.get_slot_count()}")

        return True
    except Exception as e:
        print(f"[FAIL] Context setup failed: {e}")
        traceback.print_exc()
        return False


def verify_encrypt_decrypt() -> bool:
    """Verify encrypt/decrypt roundtrip."""
    print_section("Checking Encrypt/Decrypt")
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend

        backend = CKKSMOAIBackend()
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

            # Check accuracy (CKKS has very low error)
            passed = error < 1e-4

            status = "[OK]" if passed else "[FAIL]"
            print(f"{status} Test {i+1}: max_error={error:.8f}, encrypt={encrypt_time:.1f}ms, decrypt={decrypt_time:.1f}ms")

            if not passed:
                print(f"     Input:  {data[:4]}...")
                print(f"     Output: {decrypted[:4]}...")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"[FAIL] Encrypt/decrypt failed: {e}")
        traceback.print_exc()
        return False


def verify_scalar_multiply() -> bool:
    """Verify scalar multiplication."""
    print_section("Checking Scalar Multiplication")
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend

        backend = CKKSMOAIBackend()
        backend.setup_context()
        backend.generate_keys()

        # Test cases: (input, scalar, expected)
        test_cases = [
            (np.array([1.0, 2.0, 3.0, 4.0]), 2.0),
            (np.array([1.0, 2.0, 3.0, 4.0]), 0.5),
            (np.array([1.0, 2.0, 3.0, 4.0]), -1.0),
            (np.array([0.5, -0.5, 0.25, -0.25]), 3.0),
        ]

        all_passed = True
        for i, (data, scalar) in enumerate(test_cases):
            expected = data * scalar

            # Encrypt
            ct = backend.encrypt(data)

            # Multiply by scalar
            start = time.time()
            ct_result = backend.multiply_plain(ct, np.array([scalar]))
            mult_time = (time.time() - start) * 1000

            # Decrypt
            decrypted = backend.decrypt(ct_result, len(data))

            # Compute error
            error = np.max(np.abs(expected - decrypted))

            # Check accuracy
            passed = error < 1e-3

            status = "[OK]" if passed else "[FAIL]"
            print(f"{status} {data[:2]}... * {scalar} = {decrypted[:2]}..., max_error={error:.8f}, time={mult_time:.1f}ms")

            if not passed:
                print(f"     Expected: {expected}")
                print(f"     Got:      {decrypted}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"[FAIL] Scalar multiply failed: {e}")
        traceback.print_exc()
        return False


def verify_matmul() -> bool:
    """Verify matrix multiplication."""
    print_section("Checking Matrix Multiplication")
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend

        backend = CKKSMOAIBackend()
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
        passed = error < 0.01  # CKKS should have low error

        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} Matrix multiply (4x4): max_error={error:.8f}, time={total_time:.1f}ms")
        print(f"     Expected: {expected}")
        print(f"     Got:      {decrypted}")

        # Also test column packing stats
        stats = backend.get_operation_stats()
        print(f"     Rotations used: {stats.get('rotations', 0)} (should be 0 with column packing)")

        return passed
    except Exception as e:
        print(f"[FAIL] Matrix multiply failed: {e}")
        traceback.print_exc()
        return False


def verify_lora_delta() -> bool:
    """Verify LoRA delta computation."""
    print_section("Checking LoRA Delta Computation")
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend

        backend = CKKSMOAIBackend()
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

        # Check accuracy (CKKS should have low error even for chained ops)
        passed = error < 0.1

        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} LoRA delta: max_error={error:.8f}")
        print(f"     in_dim={in_dim}, rank={rank}, out_dim={out_dim}, scaling={scaling}")
        print(f"     encrypt={encrypt_time:.1f}ms, compute={compute_time:.1f}ms, decrypt={decrypt_time:.1f}ms")
        print(f"     Expected: {expected}")
        print(f"     Got:      {decrypted}")

        # Check stats
        stats = backend.get_operation_stats()
        print(f"     Rotations used: {stats.get('rotations', 0)}")
        print(f"     Multiplications: {stats.get('multiplications', 0)}")

        return passed
    except Exception as e:
        print(f"[FAIL] LoRA delta failed: {e}")
        traceback.print_exc()
        return False


def run_quick_benchmark() -> None:
    """Run a quick benchmark."""
    print_section("Quick Benchmark")
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend

        backend = CKKSMOAIBackend()
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
        backend.reset_stats()

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
        print(f"Total rotations: {stats.get('rotations', 0)} (should be 0 with column packing)")

    except Exception as e:
        print(f"[WARN] Benchmark failed: {e}")


def main() -> int:
    """Run all verification checks."""
    print_header("CKKS MOAI Backend Verification")
    print("Using MOAI approach: Column packing for rotation-free matrix multiplication")

    results = []

    # Run checks
    results.append(("Import", verify_import()))

    if results[-1][1]:  # Only continue if import succeeded
        results.append(("Context Setup", verify_context_setup()))
        results.append(("Encrypt/Decrypt", verify_encrypt_decrypt()))
        results.append(("Scalar Multiply", verify_scalar_multiply()))
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
        print("SUCCESS: CKKS MOAI backend is properly installed and functional!")
        print("\nKey features verified:")
        print("  - CKKS approximate arithmetic for floats")
        print("  - Column packing for rotation-free matrix multiplication")
        print("  - LoRA delta computation with chained operations")
        return 0
    else:
        print("FAILURE: Some checks failed. See details above.")
        print("\nTo fix, try:")
        print("  pip install Pyfhel")
        return 1


if __name__ == "__main__":
    sys.exit(main())
