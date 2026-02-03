"""
Streaming Inference Example

Demonstrates token-by-token streaming generation for real-time output.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python streaming_inference.py
"""

import os
import sys
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Create or get training client
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
    )

    prompt = "Write a short story about a robot learning to paint:"

    print(f"Prompt: {prompt}\n")
    print("=" * 50)
    print("Streaming response:\n")

    # Stream tokens
    for token in tc.stream(
        prompt=prompt,
        max_tokens=200,
        temperature=0.8,
    ):
        # Print each token as it arrives
        sys.stdout.write(token.text)
        sys.stdout.flush()

    print("\n")
    print("=" * 50)
    print("Streaming complete!")


if __name__ == "__main__":
    main()
