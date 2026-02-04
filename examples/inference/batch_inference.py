#!/usr/bin/env python3
"""
Batch Inference with TenSafe

This example demonstrates efficient batch processing of multiple prompts,
maximizing GPU utilization and throughput for offline inference workloads.
Batch inference is ideal for processing large datasets without real-time
latency requirements.

What this example demonstrates:
- Configuring batch inference parameters
- Processing multiple prompts efficiently
- Optimizing batch sizes for throughput
- Handling batch results and errors

Key concepts:
- Batching: Process multiple inputs simultaneously
- Dynamic batching: Automatic batch formation
- Continuous batching: Efficient GPU scheduling
- Throughput vs latency tradeoff

Prerequisites:
- TenSafe server running
- Trained LoRA adapter

Expected Output:
    Configuring batch inference...
    Batch size: 8
    Total prompts: 20

    Processing batches...
    Batch 1/3: 8 prompts, 1.2s
    Batch 2/3: 8 prompts, 1.1s
    Batch 3/3: 4 prompts, 0.6s

    Batch inference complete!
    Total prompts: 20
    Total time: 3.2s
    Throughput: 6.25 prompts/sec
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class BatchConfig:
    """Configuration for batch inference."""
    batch_size: int = 8
    max_tokens: int = 256
    temperature: float = 0.7
    num_workers: int = 4
    timeout_per_prompt: float = 30.0


@dataclass
class BatchResult:
    """Result from batch inference."""
    prompt_id: int
    prompt: str
    response: str
    tokens_generated: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BatchInferenceProcessor:
    """Process multiple prompts in batches."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.results: List[BatchResult] = []

    def process_batch(
        self,
        prompts: List[str],
        inference_client=None
    ) -> List[BatchResult]:
        """Process a batch of prompts."""
        results = []

        if inference_client is not None:
            # Use actual inference client
            batch_response = inference_client.batch_generate(
                prompts,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            for i, (prompt, response) in enumerate(zip(prompts, batch_response)):
                results.append(BatchResult(
                    prompt_id=i,
                    prompt=prompt,
                    response=response.text,
                    tokens_generated=response.tokens_generated,
                    latency_ms=response.latency_ms,
                    success=True,
                ))
        else:
            # Simulation mode
            for i, prompt in enumerate(prompts):
                time.sleep(0.1)  # Simulate processing
                results.append(BatchResult(
                    prompt_id=i,
                    prompt=prompt,
                    response=f"[Simulated response for: {prompt[:30]}...]",
                    tokens_generated=50 + i * 5,
                    latency_ms=100 + i * 10,
                    success=True,
                ))

        return results

    def process_all(
        self,
        prompts: List[str],
        inference_client=None,
        progress_callback=None,
    ) -> List[BatchResult]:
        """Process all prompts in batches."""
        all_results = []
        total_batches = (len(prompts) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]

            batch_start = time.time()
            batch_results = self.process_batch(batch_prompts, inference_client)
            batch_time = time.time() - batch_start

            # Update prompt IDs to be global
            for i, result in enumerate(batch_results):
                result.prompt_id = start_idx + i

            all_results.extend(batch_results)

            if progress_callback:
                progress_callback(batch_idx + 1, total_batches, len(batch_prompts), batch_time)

        self.results = all_results
        return all_results


def main():
    """Demonstrate batch inference with TenSafe."""

    # =========================================================================
    # Step 1: Understanding batch inference
    # =========================================================================
    print("=" * 60)
    print("BATCH INFERENCE WITH TENSAFE")
    print("=" * 60)
    print("""
    Batch inference processes multiple prompts efficiently:

    Single inference (sequential):
      Prompt 1 -> [GPU] -> Response 1    (2s)
      Prompt 2 -> [GPU] -> Response 2    (2s)
      Prompt 3 -> [GPU] -> Response 3    (2s)
      Total: 6s for 3 prompts

    Batch inference (parallel):
      [Prompt 1]
      [Prompt 2] -> [GPU] -> [Response 1, 2, 3]  (2.5s)
      [Prompt 3]
      Total: 2.5s for 3 prompts

    Benefits:
    - Higher GPU utilization
    - Better throughput
    - Lower cost per inference
    - Efficient for offline processing
    """)

    # =========================================================================
    # Step 2: Prepare prompts
    # =========================================================================
    print("\nPreparing prompts...")

    # Sample prompts for batch processing
    prompts = [
        "Summarize the benefits of machine learning.",
        "Explain how neural networks work.",
        "What is transfer learning?",
        "Describe the attention mechanism.",
        "How does gradient descent optimize models?",
        "What are the applications of NLP?",
        "Explain reinforcement learning.",
        "What is a transformer architecture?",
        "How do convolutional networks work?",
        "What is the vanishing gradient problem?",
        "Explain batch normalization.",
        "What is dropout regularization?",
        "How does backpropagation work?",
        "What are embedding vectors?",
        "Explain the softmax function.",
        "What is cross-entropy loss?",
        "How do LSTMs handle sequences?",
        "What is fine-tuning a model?",
        "Explain data augmentation.",
        "What are hyperparameters?",
    ]

    print(f"  Total prompts: {len(prompts)}")

    # =========================================================================
    # Step 3: Configure batch processing
    # =========================================================================
    print("\nConfiguring batch inference...")

    config = BatchConfig(
        batch_size=8,
        max_tokens=128,
        temperature=0.7,
        num_workers=4,
    )

    print(f"  Batch size: {config.batch_size}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Workers: {config.num_workers}")

    # =========================================================================
    # Step 4: Process batches
    # =========================================================================
    print("\nProcessing batches...")
    print("-" * 50)

    def progress_callback(current: int, total: int, batch_size: int, batch_time: float):
        print(f"  Batch {current}/{total}: {batch_size} prompts, {batch_time:.2f}s")

    processor = BatchInferenceProcessor(config)
    start_time = time.time()

    try:
        from tg_tinker import ServiceClient
        from tg_tinker.schemas import InferenceConfig

        client = ServiceClient(
            base_url=os.environ.get("TG_TINKER_BASE_URL", "http://localhost:8000"),
            api_key=os.environ.get("TG_TINKER_API_KEY", "demo-api-key"),
        )

        ic = client.create_inference_client(InferenceConfig(
            model_ref="meta-llama/Llama-3-8B",
            batch_mode=True,
        ))

        results = processor.process_all(prompts, ic, progress_callback)
        client.close()

    except Exception as e:
        print(f"  Note: Server unavailable ({e})")
        print("  Running in demonstration mode...")
        results = processor.process_all(prompts, None, progress_callback)

    total_time = time.time() - start_time
    print("-" * 50)

    # =========================================================================
    # Step 5: Analyze results
    # =========================================================================
    print("\nBatch inference complete!")

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_tokens = sum(r.tokens_generated for r in successful)

    print(f"\nResults:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total tokens generated: {total_tokens}")

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {len(prompts) / total_time:.2f} prompts/sec")
    print(f"  Token throughput: {total_tokens / total_time:.0f} tokens/sec")

    # Show sample results
    print("\nSample results:")
    for result in results[:3]:
        print(f"\n  Prompt: {result.prompt[:50]}...")
        print(f"  Response: {result.response[:80]}...")
        print(f"  Tokens: {result.tokens_generated}, Latency: {result.latency_ms:.0f}ms")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BATCH INFERENCE OPTIMIZATION TIPS")
    print("=" * 60)
    print("""
    Tips for optimal batch inference:

    1. Batch size selection
       - Larger batches = better GPU utilization
       - Limited by GPU memory
       - Typical range: 8-64 prompts

    2. Prompt length considerations
       - Pad shorter prompts for efficiency
       - Group similar lengths together
       - Consider dynamic batching

    3. Memory management
       - Monitor GPU memory usage
       - Use gradient checkpointing if needed
       - Consider model parallelism for large models

    4. Error handling
       - Implement retry logic for failures
       - Log and track failed prompts
       - Use timeouts to prevent hanging
    """)


if __name__ == "__main__":
    main()
