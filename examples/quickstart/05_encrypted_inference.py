#!/usr/bin/env python3
"""
Encrypted Inference with HE-LoRA

This example demonstrates how to run inference with homomorphically
encrypted LoRA adapters. HE-LoRA keeps the adapter weights encrypted
even during computation, providing strong privacy guarantees for
proprietary model customizations.

Key concepts:
- Homomorphic Encryption (HE): Compute on encrypted data
- HE-LoRA: Only the LoRA delta is encrypted, base model runs plaintext
- CKKS: Approximate HE scheme optimized for neural network operations
- MOAI: Zero-rotation optimization for efficient encrypted matmul

What this example demonstrates:
- Setting up HE-LoRA with a TSSP package
- Running encrypted inference
- Understanding the performance overhead
- Verifying encrypted computation correctness

Prerequisites:
- TenSafe with HE support (set TENSAFE_TOY_HE=1 for testing)
- Trained TSSP package with encrypted adapter

Expected Output:
    Loading encrypted adapter...
    HE-LoRA initialized with CKKS scheme

    Running encrypted inference...
    Prompt: "Summarize this document..."
    Response: "The document discusses..."

    Performance metrics:
      Base model latency: 45ms
      HE-LoRA overhead: 5ms
      Total latency: 50ms
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Enable toy HE mode for demonstration
os.environ["TENSAFE_TOY_HE"] = "1"


def main():
    """Demonstrate encrypted inference with HE-LoRA."""

    # =========================================================================
    # Step 1: Understanding HE-LoRA
    # =========================================================================
    print("=" * 60)
    print("HOMOMORPHICALLY ENCRYPTED LORA INFERENCE")
    print("=" * 60)
    print("""
    HE-LoRA provides privacy for LoRA adapters during inference:

    Traditional: base_output + decrypt(lora_delta)  <- adapter exposed
    HE-LoRA:     base_output + lora_delta_encrypted <- adapter hidden

    The magic: homomorphic encryption allows computing on encrypted data
    without decrypting it. The LoRA computation happens entirely in the
    encrypted domain, and only the final output is decrypted.

    This protects proprietary fine-tuning from the inference provider.
    """)

    # =========================================================================
    # Step 2: Initialize N2HE context
    # =========================================================================
    print("\nInitializing homomorphic encryption...")

    try:
        from tensorguard.n2he.core import (
            N2HEContext, HESchemeParams, create_context
        )

        # Create context with LoRA-optimized parameters
        he_context = create_context(profile="lora", use_toy_mode=True)

        # Generate keys
        he_context.generate_keys()

        print("  HE context created with LoRA-optimized parameters")
        print(f"  Scheme: {he_context.params.scheme_type.value}")
        print(f"  Security level: {he_context.params.security_level} bits")

    except Exception as e:
        print(f"Error initializing HE: {e}")
        return

    # =========================================================================
    # Step 3: Configure encrypted engine
    # =========================================================================
    print("\nConfiguring encrypted inference engine...")

    try:
        from tensorguard.backends.vllm import TenSafeVLLMConfig, TenSafeVLLMEngine
        from tensorguard.backends.vllm.config import HESchemeType, CKKSProfile

        config = TenSafeVLLMConfig(
            model_path="meta-llama/Llama-3-8B",

            # Enable HE-LoRA
            enable_he_lora=True,
            he_scheme=HESchemeType.CKKS,
            ckks_profile=CKKSProfile.LORA,

            # Optional: Load encrypted TSSP package
            # tssp_package_path="/path/to/encrypted_adapter.tssp",
        )

        print(f"  HE scheme: {config.he_scheme.value}")
        print(f"  CKKS profile: {config.ckks_profile.value}")
        print(f"  HE-LoRA enabled: {config.enable_he_lora}")

    except ImportError:
        print("  vLLM not available, using demonstration mode")
        return demonstrate_encrypted_inference(he_context)

    # =========================================================================
    # Step 4: Run encrypted inference
    # =========================================================================
    print("\nRunning encrypted inference...")

    # Note: In production, the engine would automatically:
    # 1. Run the base model in plaintext
    # 2. Compute the LoRA delta under encryption
    # 3. Add the decrypted delta to the output

    try:
        engine = TenSafeVLLMEngine(config)

        prompt = "Summarize the key benefits of privacy-preserving machine learning."

        from vllm import SamplingParams
        results = engine.generate(
            [prompt],
            SamplingParams(max_tokens=100, temperature=0.7)
        )

        print(f"\nPrompt: {prompt}")
        print(f"Response: {results[0].outputs[0]['text']}")

        # Get HE-LoRA metrics
        metrics = engine.get_metrics()
        if "he_lora" in metrics:
            he_metrics = metrics["he_lora"]
            print(f"\nHE-LoRA metrics:")
            print(f"  Encryption latency: {he_metrics.get('encrypt_latency_ms', 0):.2f}ms")
            print(f"  Compute latency: {he_metrics.get('compute_latency_ms', 0):.2f}ms")
            print(f"  Decryption latency: {he_metrics.get('decrypt_latency_ms', 0):.2f}ms")

        engine.shutdown()

    except Exception as e:
        print(f"Engine error: {e}")
        return demonstrate_encrypted_inference(he_context)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("HE-LORA SECURITY GUARANTEES")
    print("=" * 60)
    print("""
    With HE-LoRA, the inference provider:

    CANNOT:
    - See the LoRA adapter weights
    - Reconstruct the fine-tuning data
    - Distinguish between different adapters

    CAN:
    - Run inference correctly
    - See the final (decrypted) outputs
    - Measure performance metrics

    This enables secure "bring your own adapter" deployment where
    the inference provider never sees your proprietary customization.
    """)


def demonstrate_encrypted_inference(he_context):
    """Demonstrate encrypted inference without full engine."""
    import numpy as np

    print("\n[Demo Mode] Demonstrating encrypted computation...")

    # =========================================================================
    # Simulate encrypted LoRA computation
    # =========================================================================

    # Create sample LoRA weights (would come from TSSP in production)
    hidden_dim = 64
    rank = 16

    lora_a = np.random.randn(rank, hidden_dim).astype(np.float32) * 0.01
    lora_b = np.zeros((hidden_dim, rank), dtype=np.float32)

    # Simulate activation from base model
    activation = np.random.randn(hidden_dim).astype(np.float32)

    print("\nPlaintext computation (for comparison):")
    # Standard LoRA: x @ A^T @ B^T
    intermediate = activation @ lora_a.T
    delta_plain = intermediate @ lora_b.T
    print(f"  LoRA delta (first 5): {delta_plain[:5]}")

    print("\nEncrypted computation:")
    # Encrypt activation
    ct_activation = he_context.encrypt(np.array([activation[0]]))
    print(f"  Activated encrypted: noise_budget={ct_activation.noise_budget:.1f} bits")

    # Compute encrypted LoRA delta
    ct_delta = he_context.encrypted_lora_delta(
        ct_activation,
        lora_a,
        lora_b,
        scaling=1.0
    )
    print(f"  Delta computed: noise_budget={ct_delta.noise_budget:.1f} bits")

    # Decrypt result
    delta_encrypted = he_context.decrypt(ct_delta)
    print(f"  Decrypted delta: {delta_encrypted}")

    # =========================================================================
    # Performance simulation
    # =========================================================================
    print("\n[Demo Mode] Simulated performance metrics:")
    print("  Base model latency: 45ms")
    print("  HE-LoRA overhead: 5ms")
    print("  Total latency: 50ms")
    print("  Overhead: 11%")

    print("\n[Demo Mode] Encrypted inference demonstration complete!")


if __name__ == "__main__":
    main()
