"""
Multi-Adapter Inference Example

Demonstrates hot-swapping between multiple LoRA adapters for different tasks.

Requirements:
- TenSafe account and API key
- Multiple trained adapters

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python multi_adapter.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Load multiple adapters
    adapters = {
        "coding": client.tgsp.load_adapter("path/to/coding-adapter.tgsp"),
        "writing": client.tgsp.load_adapter("path/to/writing-adapter.tgsp"),
        "translation": client.tgsp.load_adapter("path/to/translation-adapter.tgsp"),
    }

    print("Loaded adapters:", list(adapters.keys()))
    print()

    tc = client.create_training_client(model_ref="meta-llama/Llama-3-8B")

    # Test each adapter with appropriate prompts
    test_prompts = {
        "coding": "Write a Python function to calculate fibonacci numbers:",
        "writing": "Write a creative story opening about a mysterious forest:",
        "translation": "Translate to French: The quick brown fox jumps over the lazy dog.",
    }

    for adapter_name, prompt in test_prompts.items():
        print(f"=== Using {adapter_name} adapter ===")
        print(f"Prompt: {prompt}\n")

        # Activate adapter (hot-swap)
        client.tgsp.activate_adapter(adapters[adapter_name].adapter_id)

        # Generate
        result = tc.sample(prompts=[prompt], max_tokens=100, temperature=0.7)
        print(f"Response: {result.samples[0].completion}\n")

    # Show adapter usage stats
    print("=== Adapter Usage Stats ===")
    for name, adapter in adapters.items():
        info = client.tgsp.get_adapter_info(adapter.adapter_id)
        print(f"{name}: {info.forward_count} inferences")


if __name__ == "__main__":
    main()
