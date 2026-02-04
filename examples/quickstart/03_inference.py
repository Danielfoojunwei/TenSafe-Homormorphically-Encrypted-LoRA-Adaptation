#!/usr/bin/env python3
"""
Basic Inference with TenSafe

This example demonstrates how to run inference using TenSafe with a
trained LoRA adapter. It shows both single-request and batch inference
patterns.

What this example demonstrates:
- Loading a trained LoRA adapter
- Running single inference requests
- Batch inference for higher throughput
- Sampling parameters (temperature, top-k, top-p)
- Handling responses and extracting outputs

Prerequisites:
- TenSafe server running with vLLM backend
- A trained LoRA adapter (or use base model)

Expected Output:
    Loading model and adapter...
    Model loaded: meta-llama/Llama-3-8B

    Single inference:
    Prompt: "Explain quantum computing in simple terms."
    Response: "Quantum computing uses quantum bits..."

    Batch inference (3 prompts):
    [1] "What is machine learning?" -> "Machine learning is..."
    [2] "Define neural networks" -> "Neural networks are..."
    [3] "Explain deep learning" -> "Deep learning is a subset..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    """Demonstrate basic inference with TenSafe."""

    # =========================================================================
    # Step 1: Import and configure
    # =========================================================================
    print("Initializing TenSafe for inference...")

    try:
        from tensorguard.backends.vllm import TenSafeVLLMEngine, TenSafeVLLMConfig
        VLLM_AVAILABLE = True
    except ImportError:
        print("Note: vLLM not available, using simulation mode")
        VLLM_AVAILABLE = False

    # =========================================================================
    # Step 2: Configure the engine
    # =========================================================================
    print("\nConfiguring inference engine...")

    if VLLM_AVAILABLE:
        config = TenSafeVLLMConfig(
            model_path="meta-llama/Llama-3-8B",
            # Optional: Load a trained TSSP adapter package
            # tssp_package_path="/path/to/adapter.tssp",

            # vLLM performance settings
            tensor_parallel_size=1,          # Number of GPUs for tensor parallelism
            max_num_seqs=256,                # Maximum concurrent sequences
            max_num_batched_tokens=32768,    # Maximum tokens per batch

            # TenSafe settings
            enable_he_lora=False,            # Enable for encrypted inference
            enable_audit_logging=True,
        )

        print(f"  Model: {config.model_path}")
        print(f"  Tensor parallel: {config.tensor_parallel_size}")

    # =========================================================================
    # Step 3: Initialize the engine
    # =========================================================================
    print("\nInitializing inference engine...")

    if VLLM_AVAILABLE:
        try:
            engine = TenSafeVLLMEngine(config)
            print("Engine initialized successfully!")
        except Exception as e:
            print(f"Could not initialize engine: {e}")
            return demonstrate_inference()
    else:
        return demonstrate_inference()

    # =========================================================================
    # Step 4: Single inference request
    # =========================================================================
    print("\n" + "=" * 60)
    print("SINGLE INFERENCE")
    print("=" * 60)

    prompt = "Explain quantum computing in simple terms."
    print(f"\nPrompt: {prompt}")

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,      # Controls randomness (0 = deterministic)
        top_p=0.9,            # Nucleus sampling threshold
        top_k=50,             # Top-k sampling
        max_tokens=256,       # Maximum tokens to generate
        stop=[".", "\n\n"],   # Stop sequences
    )

    results = engine.generate([prompt], sampling_params)

    response = results[0].outputs[0]["text"]
    print(f"\nResponse: {response}")

    # =========================================================================
    # Step 5: Batch inference
    # =========================================================================
    print("\n" + "=" * 60)
    print("BATCH INFERENCE")
    print("=" * 60)

    prompts = [
        "What is machine learning?",
        "Define neural networks in one sentence.",
        "Explain deep learning briefly.",
    ]

    print(f"\nProcessing {len(prompts)} prompts...")

    results = engine.generate(prompts, sampling_params)

    for i, result in enumerate(results):
        prompt = prompts[i]
        response = result.outputs[0]["text"]
        print(f"\n[{i + 1}] Prompt: {prompt}")
        print(f"    Response: {response[:100]}...")

    # =========================================================================
    # Step 6: Get metrics
    # =========================================================================
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    metrics = engine.get_metrics()
    print(f"\nTotal requests: {metrics['total_requests']}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Tokens/second: {metrics['tokens_per_second']:.2f}")

    # =========================================================================
    # Cleanup
    # =========================================================================
    engine.shutdown()
    print("\nEngine shutdown complete.")


def demonstrate_inference():
    """Demonstrate inference without actual engine."""
    print("\n[Demo Mode] Simulating inference...")

    # Simulated responses
    prompts_and_responses = [
        ("Explain quantum computing in simple terms.",
         "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers."),
        ("What is machine learning?",
         "Machine learning is a subset of AI where systems learn from data to improve their performance without being explicitly programmed."),
        ("Define neural networks in one sentence.",
         "Neural networks are computing systems inspired by biological neural networks that learn to perform tasks by analyzing examples."),
    ]

    print("\n" + "=" * 60)
    print("SIMULATED INFERENCE RESULTS")
    print("=" * 60)

    for prompt, response in prompts_and_responses:
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

    print("\n[Demo Mode] Simulation complete!")
    print("To run actual inference:")
    print("  1. Install vLLM: pip install vllm")
    print("  2. Ensure GPU is available")
    print("  3. Start the TenSafe server")


if __name__ == "__main__":
    main()
