"""
Basic Inference Example

Demonstrates simple text generation using a TenSafe training client.

Requirements:
- TenSafe account and API key
- A trained model or training client

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python basic_inference.py
"""

import os
from tensafe import TenSafeClient


def main():
    # Initialize client
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
        base_url=os.environ.get("TENSAFE_BASE_URL", "https://api.tensafe.io"),
    )

    # Create a training client (or use existing)
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={
            "rank": 16,
            "alpha": 32.0,
            "target_modules": ["q_proj", "v_proj"],
        },
    )

    print(f"Training client created: {tc.id}")

    # Generate text
    prompts = [
        "The future of AI is",
        "Privacy-preserving machine learning enables",
        "Homomorphic encryption allows",
    ]

    # Run inference
    results = tc.sample(
        prompts=prompts,
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    # Print results
    for sample in results.samples:
        print(f"\nPrompt: {sample.prompt}")
        print(f"Completion: {sample.completion}")
        print(f"Tokens: {sample.tokens_generated}")
        print("-" * 50)


if __name__ == "__main__":
    main()
