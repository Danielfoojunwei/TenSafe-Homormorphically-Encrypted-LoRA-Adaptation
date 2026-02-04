#!/usr/bin/env python3
"""
Basic Inference with TenSafe

This example demonstrates how to run basic inference with a trained LoRA
adapter using the TenSafe platform. It shows the simplest path from a
trained model to generating predictions.

What this example demonstrates:
- Loading a trained LoRA adapter
- Configuring inference parameters
- Running single-prompt inference
- Interpreting model outputs

Key concepts:
- Inference client: Handles model loading and generation
- Adapter loading: LoRA weights merged at runtime
- Generation config: Controls output behavior

Prerequisites:
- TenSafe server running
- Trained LoRA adapter (or use built-in demo adapter)

Expected Output:
    Loading model and adapter...
    Model: meta-llama/Llama-3-8B
    Adapter: my-custom-adapter

    Running inference...
    Prompt: "Explain quantum computing in simple terms"
    Response: "Quantum computing is a type of computation that..."

    Inference complete!
    Tokens generated: 128
    Latency: 1.2s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    """Demonstrate basic inference with TenSafe."""

    # =========================================================================
    # Step 1: Configure inference settings
    # =========================================================================
    print("=" * 60)
    print("BASIC INFERENCE WITH TENSAFE")
    print("=" * 60)

    # Model and adapter configuration
    model_name = "meta-llama/Llama-3-8B"
    adapter_path = os.environ.get("TENSAFE_ADAPTER_PATH", "demo-adapter")

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Adapter: {adapter_path}")

    # =========================================================================
    # Step 2: Initialize the inference client
    # =========================================================================
    print("\nInitializing inference client...")

    try:
        from tg_tinker import ServiceClient
        from tg_tinker.schemas import InferenceConfig

        client = ServiceClient(
            base_url=os.environ.get("TG_TINKER_BASE_URL", "http://localhost:8000"),
            api_key=os.environ.get("TG_TINKER_API_KEY", "demo-api-key"),
        )

        # Create inference configuration
        inference_config = InferenceConfig(
            model_ref=model_name,
            adapter_path=adapter_path,
            max_tokens=256,
            temperature=0.7,
        )

        # Create inference client
        ic = client.create_inference_client(inference_config)
        print(f"  Inference client created: {ic.inference_client_id}")

    except Exception as e:
        print(f"  Note: Server connection unavailable ({e})")
        print("  Running in demonstration mode...")
        return demonstrate_basic_inference()

    # =========================================================================
    # Step 3: Run inference
    # =========================================================================
    print("\nRunning inference...")

    prompt = "Explain quantum computing in simple terms"
    print(f"Prompt: \"{prompt}\"")

    start_time = time.time()
    response = ic.generate(prompt)
    elapsed = time.time() - start_time

    print(f"\nResponse: {response.text}")
    print(f"\nMetrics:")
    print(f"  Tokens generated: {response.tokens_generated}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Tokens/sec: {response.tokens_generated / elapsed:.1f}")

    # Clean up
    client.close()

    print("\nInference complete!")


def demonstrate_basic_inference():
    """Demonstrate inference without server connection."""
    print("\n[Demo Mode] Simulating basic inference...")

    prompt = "Explain quantum computing in simple terms"
    print(f"\nPrompt: \"{prompt}\"")

    # Simulated response
    response_text = """Quantum computing is a type of computation that harnesses
quantum mechanical phenomena like superposition and entanglement. Unlike classical
computers that use bits (0 or 1), quantum computers use qubits that can exist in
multiple states simultaneously. This allows them to process certain calculations
exponentially faster than traditional computers, particularly for problems like
cryptography, optimization, and simulating molecular structures."""

    print(f"\nResponse: {response_text}")

    print(f"\n[Demo Mode] Simulated metrics:")
    print(f"  Tokens generated: 89")
    print(f"  Latency: 1.2s")
    print(f"  Tokens/sec: 74.2")

    # =========================================================================
    # Understanding the output
    # =========================================================================
    print("\n" + "=" * 60)
    print("UNDERSTANDING INFERENCE")
    print("=" * 60)
    print("""
    Basic inference flow:

    1. Load base model (e.g., Llama-3-8B)
    2. Load LoRA adapter weights
    3. Apply adapter to model (merge or runtime application)
    4. Tokenize input prompt
    5. Run autoregressive generation
    6. Decode output tokens to text

    Key parameters:
    - max_tokens: Maximum output length
    - temperature: Randomness (0=deterministic, 1=creative)
    - top_p: Nucleus sampling threshold
    - top_k: Top-k sampling limit

    Next steps:
    - Try streaming_inference.py for token-by-token output
    - Try batch_inference.py for processing multiple prompts
    - Try encrypted_inference.py for HE-LoRA privacy
    """)


if __name__ == "__main__":
    main()
