"""
vLLM Serving Example

Demonstrates high-throughput serving with vLLM backend.

Requirements:
- TenSafe account and API key
- vLLM backend enabled

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python vllm_serving.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Create inference client with vLLM backend
    inference = client.create_inference_client(
        model_ref="meta-llama/Llama-3-8B",
        backend="vllm",
        vllm_config={
            "tensor_parallel_size": 1,
            "max_num_seqs": 256,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
        },
    )

    print("vLLM Backend Configuration:")
    print(f"  Max sequences: 256")
    print(f"  Max model length: 4096")
    print(f"  GPU memory utilization: 90%")
    print()

    # High-throughput inference
    import time
    prompts = ["Hello, how are you?"] * 100

    print(f"Running {len(prompts)} inference requests...")
    start = time.time()

    results = inference.batch_generate(
        prompts=prompts,
        max_tokens=50,
        temperature=0.7,
    )

    elapsed = time.time() - start
    tokens_generated = sum(r.tokens_generated for r in results)

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {len(prompts)/elapsed:.1f} requests/sec")
    print(f"  Tokens: {tokens_generated} ({tokens_generated/elapsed:.0f} tok/sec)")


if __name__ == "__main__":
    main()
