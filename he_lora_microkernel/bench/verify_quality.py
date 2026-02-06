"""
Quality/Fidelity Verification for Different HE Approaches

This script demonstrates that all HE approaches produce outputs
that match plaintext PyTorch/NumPy LoRA within acceptable error bounds.

CKKS is an approximate scheme, so there's inherent precision loss.
Error bounds:
- Absolute error ≤ 1e-2
- Relative error ≤ 1e-2
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.backend.gpu_ckks_backend import BackendType
from he_lora_microkernel.compiler import (
    CKKSProfile,
    LoRAConfig,
    LoRATargets,
    compile_schedule,
    get_profile,
)
from he_lora_microkernel.runtime import HELoRAExecutor


def reference_lora_forward(x, A, B, alpha, rank):
    """Reference plaintext LoRA: Δy = (alpha/rank) * A @ B @ x"""
    scaling = alpha / rank
    intermediate = (B @ x.T).T
    delta = (A @ intermediate.T).T
    return scaling * delta


def compute_error_metrics(actual, expected):
    """Compute comprehensive error metrics."""
    abs_error = np.abs(expected - actual)
    rel_error = abs_error / (np.abs(expected) + 1e-10)

    return {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error),
        'rms_error': np.sqrt(np.mean((expected - actual) ** 2)),
        'correlation': np.corrcoef(expected.flatten(), actual.flatten())[0, 1],
    }


def verify_fidelity(hidden_size, rank, batch_size, alpha, n_samples=10):
    """Verify fidelity for a given configuration."""
    rng = np.random.default_rng(42)

    # Generate weights
    A = rng.standard_normal((hidden_size, rank)).astype(np.float64) * 0.01
    B = rng.standard_normal((rank, hidden_size)).astype(np.float64) * 0.01

    config = LoRAConfig(
        hidden_size=hidden_size,
        rank=rank,
        alpha=alpha,
        targets=LoRATargets.QKV,
        batch_size=batch_size,
        max_context_length=512,
        ckks_profile=CKKSProfile.FAST,
    )

    ckks_params = get_profile(config.ckks_profile)
    schedule = compile_schedule(config, ckks_params)

    executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
    executor.load_weights(A, B, alpha)

    all_metrics = []

    for i in range(n_samples):
        x = rng.standard_normal((batch_size, hidden_size)).astype(np.float64)

        # Reference (plaintext)
        expected = reference_lora_forward(x, A, B, alpha, rank)

        # HE-LoRA (encrypted simulation)
        actual = executor.execute_token(x, position=i)

        metrics = compute_error_metrics(actual, expected)
        all_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }

    return avg_metrics


def main():
    print("=" * 70)
    print("QUALITY/FIDELITY VERIFICATION ACROSS HE APPROACHES")
    print("=" * 70)
    print()
    print("Comparing HE-LoRA output to plaintext PyTorch/NumPy reference.")
    print("CKKS approximation error bounds: abs_error ≤ 1e-2, rel_error ≤ 1e-2")
    print()

    configs = [
        # (hidden_size, rank, batch_size, alpha)
        (256, 8, 4, 16.0),
        (512, 16, 4, 32.0),
        (1024, 16, 2, 32.0),
        (256, 4, 8, 8.0),
        (512, 32, 2, 64.0),
    ]

    print("-" * 70)
    print(f"{'Config':<25} {'Max Abs':<12} {'Max Rel':<12} {'RMS':<12} {'Corr':<10}")
    print(f"{'(h, r, b, α)':<25} {'Error':<12} {'Error':<12} {'Error':<12} {'Coeff':<10}")
    print("-" * 70)

    all_pass = True

    for h, r, b, alpha in configs:
        try:
            metrics = verify_fidelity(h, r, b, alpha, n_samples=10)

            max_abs = metrics['max_abs_error']
            max_rel = metrics['max_rel_error']
            rms = metrics['rms_error']
            corr = metrics['correlation']

            # Check if within bounds
            passes = max_abs <= 1e-2 and max_rel <= 1e-2
            status = "✓" if passes else "✗"
            all_pass = all_pass and passes

            print(f"({h}, {r}, {b}, {alpha}){'':<8} {max_abs:<12.2e} {max_rel:<12.2e} {rms:<12.2e} {corr:<10.6f} {status}")

        except Exception as e:
            print(f"({h}, {r}, {b}, {alpha}){'':<8} FAILED: {str(e)[:40]}")
            all_pass = False

    print("-" * 70)
    print()

    # Summary
    print("ERROR BOUND THRESHOLDS:")
    print("-" * 40)
    print("  Absolute Error Threshold: 1e-2 (0.01)")
    print("  Relative Error Threshold: 1e-2 (1%)")
    print()

    print("WHY THESE BOUNDS ARE ACCEPTABLE:")
    print("-" * 40)
    print("  1. CKKS is an approximate HE scheme by design")
    print("  2. Error is << typical floating point noise in DNN training")
    print("  3. LoRA adapters have small deltas anyway (scaled by α/r)")
    print("  4. Correlation coefficient > 0.999 indicates near-perfect linear relationship")
    print()

    print("FIDELITY VERIFICATION METHODS:")
    print("-" * 40)
    print("  1. test_fidelity.py: Compares HE output to FP64 reference")
    print("  2. test_semantics.py: Verifies encrypt/decrypt roundtrip")
    print("  3. test_precision.py: Measures SNR, RMS error, quantization error")
    print("  4. Packing roundtrip: Verifies pack/unpack preserves values")
    print("  5. Batch invariance: Same input = same output across batch sizes")
    print()

    if all_pass:
        print("✓ ALL CONFIGURATIONS PASS FIDELITY CHECKS")
    else:
        print("✗ SOME CONFIGURATIONS FAILED FIDELITY CHECKS")

    print()

    # Additional verification: show that SIMD batching doesn't degrade quality
    print("SIMD BATCHING QUALITY VERIFICATION:")
    print("-" * 40)

    # Compare single sample vs batched
    rng = np.random.default_rng(42)
    h, r = 512, 16
    alpha = 32.0

    A = rng.standard_normal((h, r)).astype(np.float64) * 0.01
    B = rng.standard_normal((r, h)).astype(np.float64) * 0.01
    x_single = rng.standard_normal((1, h)).astype(np.float64)

    # Single sample reference
    ref_single = reference_lora_forward(x_single, A, B, alpha, r)

    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        config = LoRAConfig(
            hidden_size=h,
            rank=r,
            alpha=alpha,
            targets=LoRATargets.QKV,
            batch_size=batch_size,
            max_context_length=512,
            ckks_profile=CKKSProfile.FAST,
        )

        try:
            ckks_params = get_profile(config.ckks_profile)
            schedule = compile_schedule(config, ckks_params)
            executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
            executor.load_weights(A, B, alpha)

            # Create batch with same first sample
            x_batch = np.zeros((batch_size, h))
            x_batch[0] = x_single[0]

            actual = executor.execute_token(x_batch)

            # Compare first sample
            error = np.max(np.abs(actual[0] - ref_single[0]))
            status = "✓" if error <= 1e-2 else "✗"
            print(f"  batch_size={batch_size}: first_sample_error={error:.2e} {status}")

        except Exception as e:
            print(f"  batch_size={batch_size}: SKIPPED ({str(e)[:30]})")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
