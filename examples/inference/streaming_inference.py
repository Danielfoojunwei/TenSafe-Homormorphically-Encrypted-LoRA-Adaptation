#!/usr/bin/env python3
"""
Streaming Inference with TenSafe

This example demonstrates streaming token generation, where tokens are
returned incrementally as they are generated rather than waiting for
the complete response. This provides better user experience for
interactive applications.

What this example demonstrates:
- Setting up streaming generation
- Processing tokens as they arrive
- Handling stream events and metadata
- Building responsive chat interfaces

Key concepts:
- Streaming: Tokens returned incrementally
- Server-sent events: HTTP streaming protocol
- Token callback: Function called per token
- Partial responses: Accumulating text chunks

Prerequisites:
- TenSafe server running
- Trained LoRA adapter

Expected Output:
    Initializing streaming client...

    Streaming response:
    Quantum [+23ms]
    computing [+45ms]
    is [+67ms]
    ...

    Stream complete!
    Total tokens: 89
    Total time: 1.8s
    Time to first token: 120ms
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Generator, Optional

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class StreamingMetrics:
    """Track metrics during streaming generation."""

    def __init__(self):
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.token_count = 0
        self.token_times: list[float] = []

    def record_token(self):
        """Record a token arrival."""
        now = time.time()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
        self.token_count += 1

    @property
    def time_to_first_token(self) -> float:
        """Time from start to first token in ms."""
        if self.first_token_time is None:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000

    @property
    def total_time(self) -> float:
        """Total generation time in seconds."""
        if not self.token_times:
            return 0.0
        return self.token_times[-1] - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Average tokens per second."""
        if self.total_time == 0:
            return 0.0
        return self.token_count / self.total_time


def main():
    """Demonstrate streaming inference with TenSafe."""

    # =========================================================================
    # Step 1: Understanding streaming generation
    # =========================================================================
    print("=" * 60)
    print("STREAMING INFERENCE WITH TENSAFE")
    print("=" * 60)
    print("""
    Streaming inference returns tokens as they're generated:

    Traditional inference:
      [Wait 2s for complete response...]
      "Complete response appears at once"

    Streaming inference:
      "Tokens" [50ms]
      "appear" [100ms]
      "one" [150ms]
      "by" [200ms]
      "one" [250ms]

    Benefits:
    - Better perceived latency
    - Responsive UI feedback
    - Early termination possible
    - Progress indication
    """)

    # =========================================================================
    # Step 2: Initialize streaming client
    # =========================================================================
    print("\nInitializing streaming client...")

    try:
        from tg_tinker import ServiceClient
        from tg_tinker.schemas import InferenceConfig, StreamingConfig

        client = ServiceClient(
            base_url=os.environ.get("TG_TINKER_BASE_URL", "http://localhost:8000"),
            api_key=os.environ.get("TG_TINKER_API_KEY", "demo-api-key"),
        )

        inference_config = InferenceConfig(
            model_ref="meta-llama/Llama-3-8B",
            max_tokens=256,
            temperature=0.7,
            streaming=StreamingConfig(
                enabled=True,
                chunk_size=1,  # One token at a time
                include_usage=True,
            ),
        )

        ic = client.create_inference_client(inference_config)
        run_streaming_inference(ic)
        client.close()

    except Exception as e:
        print(f"  Note: Server unavailable ({e})")
        print("  Running demonstration mode...")
        demonstrate_streaming_inference()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("STREAMING BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for streaming inference:

    1. Handle backpressure
       - Process tokens as fast as they arrive
       - Buffer if processing is slow

    2. Implement timeout handling
       - Set reasonable stream timeout
       - Handle partial responses gracefully

    3. Enable early termination
       - Allow users to cancel generation
       - Implement stop sequences

    4. Track metrics
       - Time to first token (TTFT)
       - Inter-token latency
       - Total generation time

    5. Error handling
       - Handle stream disconnections
       - Implement retry logic
       - Save partial responses
    """)


def run_streaming_inference(ic):
    """Run streaming inference with the client."""
    print("\nStreaming response:")
    print("-" * 40)

    prompt = "Write a haiku about machine learning"
    metrics = StreamingMetrics()

    full_response = ""
    for token in ic.stream(prompt):
        metrics.record_token()
        full_response += token.text
        elapsed_ms = (time.time() - metrics.start_time) * 1000
        print(f"{token.text}", end="", flush=True)

    print("\n" + "-" * 40)
    print(f"\nStream complete!")
    print(f"  Total tokens: {metrics.token_count}")
    print(f"  Total time: {metrics.total_time:.2f}s")
    print(f"  Time to first token: {metrics.time_to_first_token:.0f}ms")
    print(f"  Tokens/sec: {metrics.tokens_per_second:.1f}")


def demonstrate_streaming_inference():
    """Demonstrate streaming without server connection."""
    print("\n[Demo Mode] Simulating streaming inference...")
    print("\nPrompt: \"Write a haiku about machine learning\"")
    print("\nStreaming response:")
    print("-" * 40)

    # Simulated streaming tokens
    tokens = [
        "Data", " flows", " like", " water", "\n",
        "Neural", " paths", " light", " up", " at", " dawn", "\n",
        "Wisdom", " emerges"
    ]

    metrics = StreamingMetrics()
    full_response = ""

    for token in tokens:
        metrics.record_token()
        full_response += token
        elapsed_ms = (time.time() - metrics.start_time) * 1000

        # Simulate generation delay
        time.sleep(0.05)  # 50ms per token

        print(f"{token}", end="", flush=True)

    print("\n" + "-" * 40)

    print(f"\n[Demo Mode] Stream complete!")
    print(f"  Total tokens: {metrics.token_count}")
    print(f"  Total time: {metrics.total_time:.2f}s")
    print(f"  Time to first token: {metrics.time_to_first_token:.0f}ms")
    print(f"  Tokens/sec: {metrics.tokens_per_second:.1f}")


if __name__ == "__main__":
    main()
