"""
Batch Inference Example

Demonstrates processing multiple prompts efficiently in a single batch.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python batch_inference.py
"""

import os
import time
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
    )

    # Large batch of prompts
    prompts = [
        "Translate to French: Hello, how are you?",
        "Translate to French: The weather is beautiful today.",
        "Translate to French: I love programming.",
        "Translate to French: Machine learning is fascinating.",
        "Translate to French: Privacy is important.",
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "Summarize: Artificial intelligence is transforming industries.",
        "Summarize: Data privacy regulations are becoming stricter.",
        "Question: What is 2 + 2?",
        "Question: What is the capital of France?",
    ]

    print(f"Processing {len(prompts)} prompts in batch...")
    print("=" * 50)

    start_time = time.time()

    # Batch inference
    results = tc.sample(
        prompts=prompts,
        max_tokens=50,
        temperature=0.3,  # Lower temperature for more deterministic output
    )

    elapsed = time.time() - start_time

    # Print results
    for i, sample in enumerate(results.samples):
        print(f"\n[{i+1}] {sample.prompt}")
        print(f"    → {sample.completion.strip()}")

    print("\n" + "=" * 50)
    print(f"Processed {len(prompts)} prompts in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(prompts)*1000:.1f}ms per prompt")


if __name__ == "__main__":
    main()
