#!/usr/bin/env python3
"""
Multi-Adapter Inference with TenSafe

This example demonstrates how to use multiple LoRA adapters simultaneously
or switch between them efficiently. This enables use cases like multi-task
models, A/B testing, and personalized inference.

What this example demonstrates:
- Loading multiple LoRA adapters
- Switching adapters without reloading base model
- Combining adapters with weighted merging
- Per-request adapter selection

Key concepts:
- Adapter pool: Pre-loaded adapters for fast switching
- Hot-swapping: Change adapter without restart
- Adapter merging: Combine multiple adapters
- Routing: Select adapter based on request

Prerequisites:
- TenSafe server running
- Multiple trained LoRA adapters

Expected Output:
    Loading adapters...
    Adapter pool: 3 adapters loaded

    Testing adapter switching...
    Adapter: summarization -> Summary response
    Adapter: translation   -> Translated response
    Adapter: code-gen      -> Code response

    Adapter switch latency: 0.5ms (vs 2.5s for full reload)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class AdapterConfig:
    """Configuration for a single LoRA adapter."""
    name: str
    path: str
    rank: int = 16
    alpha: int = 32
    description: str = ""


@dataclass
class MultiAdapterConfig:
    """Configuration for multi-adapter inference."""
    base_model: str = "meta-llama/Llama-3-8B"
    adapters: List[AdapterConfig] = field(default_factory=list)
    default_adapter: Optional[str] = None
    enable_merging: bool = True


class AdapterPool:
    """Manage multiple LoRA adapters efficiently."""

    def __init__(self, config: MultiAdapterConfig):
        self.config = config
        self.adapters: Dict[str, Any] = {}
        self.current_adapter: Optional[str] = None
        self._load_metrics: Dict[str, float] = {}

    def load_adapter(self, adapter_config: AdapterConfig) -> float:
        """Load an adapter into the pool. Returns load time in ms."""
        start = time.time()

        # Simulate adapter loading (in production, would load actual weights)
        self.adapters[adapter_config.name] = {
            "config": adapter_config,
            "weights": f"<weights for {adapter_config.name}>",
            "loaded_at": time.time(),
        }

        load_time = (time.time() - start) * 1000
        self._load_metrics[adapter_config.name] = load_time
        return load_time

    def load_all(self) -> Dict[str, float]:
        """Load all configured adapters."""
        load_times = {}
        for adapter_config in self.config.adapters:
            load_times[adapter_config.name] = self.load_adapter(adapter_config)
        return load_times

    def switch_adapter(self, adapter_name: str) -> float:
        """Switch to a different adapter. Returns switch time in ms."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not loaded")

        start = time.time()

        # Hot-swap is very fast since base model stays loaded
        self.current_adapter = adapter_name

        switch_time = (time.time() - start) * 1000
        return switch_time

    def get_merged_weights(
        self,
        adapter_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Merge multiple adapters with given weights."""
        if not self.config.enable_merging:
            raise ValueError("Adapter merging is not enabled")

        # Validate weights sum to 1
        total = sum(adapter_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1, got {total}")

        # Validate all adapters are loaded
        for name in adapter_weights:
            if name not in self.adapters:
                raise ValueError(f"Adapter '{name}' not loaded")

        # Create merged configuration
        merged = {
            "type": "merged",
            "components": adapter_weights,
            "created_at": time.time(),
        }

        return merged

    def list_adapters(self) -> List[str]:
        """List all loaded adapters."""
        return list(self.adapters.keys())


def main():
    """Demonstrate multi-adapter inference."""

    # =========================================================================
    # Step 1: Understanding multi-adapter inference
    # =========================================================================
    print("=" * 60)
    print("MULTI-ADAPTER INFERENCE")
    print("=" * 60)
    print("""
    Multi-adapter inference enables flexible model customization:

    Traditional approach (slow):
      Request for Task A -> Load Model + Adapter A -> Generate
      Request for Task B -> Reload Model + Adapter B -> Generate
      Switch time: 2-10 seconds

    Multi-adapter approach (fast):
      Pre-load: Model + [Adapter A, Adapter B, Adapter C]
      Request for Task A -> Select Adapter A -> Generate
      Request for Task B -> Select Adapter B -> Generate
      Switch time: <1 millisecond

    Use cases:
    - Multi-task serving (summarization, translation, etc.)
    - A/B testing different adapters
    - Personalized models per user/tenant
    - Progressive rollout of adapter updates
    """)

    # =========================================================================
    # Step 2: Configure adapters
    # =========================================================================
    print("\nConfiguring adapters...")

    config = MultiAdapterConfig(
        base_model="meta-llama/Llama-3-8B",
        adapters=[
            AdapterConfig(
                name="summarization",
                path="/adapters/summarize-v1",
                rank=16,
                alpha=32,
                description="Text summarization adapter"
            ),
            AdapterConfig(
                name="translation",
                path="/adapters/translate-en-fr",
                rank=32,
                alpha=64,
                description="English to French translation"
            ),
            AdapterConfig(
                name="code-gen",
                path="/adapters/code-python",
                rank=64,
                alpha=128,
                description="Python code generation"
            ),
        ],
        default_adapter="summarization",
        enable_merging=True,
    )

    for adapter in config.adapters:
        print(f"  {adapter.name}: {adapter.description}")

    # =========================================================================
    # Step 3: Load adapter pool
    # =========================================================================
    print("\nLoading adapter pool...")

    pool = AdapterPool(config)
    load_times = pool.load_all()

    print(f"  Adapters loaded: {len(pool.list_adapters())}")
    for name, load_time in load_times.items():
        print(f"    {name}: {load_time:.1f}ms")

    # =========================================================================
    # Step 4: Test adapter switching
    # =========================================================================
    print("\nTesting adapter switching...")
    print("-" * 50)

    test_prompts = {
        "summarization": "Summarize: Machine learning is a subset of AI...",
        "translation": "Translate to French: Hello, how are you today?",
        "code-gen": "Write a Python function to calculate factorial.",
    }

    simulated_responses = {
        "summarization": "ML is a subset of AI that enables learning from data.",
        "translation": "Bonjour, comment allez-vous aujourd'hui?",
        "code-gen": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    }

    switch_times = []
    for adapter_name, prompt in test_prompts.items():
        switch_time = pool.switch_adapter(adapter_name)
        switch_times.append(switch_time)

        print(f"\nAdapter: {adapter_name}")
        print(f"  Switch time: {switch_time:.3f}ms")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Response: {simulated_responses[adapter_name][:60]}...")

    avg_switch = sum(switch_times) / len(switch_times)
    print("-" * 50)
    print(f"\nAverage switch time: {avg_switch:.3f}ms")
    print("(Compare to ~2.5s for full model reload)")

    # =========================================================================
    # Step 5: Test adapter merging
    # =========================================================================
    print("\nTesting adapter merging...")
    print("-" * 50)

    merge_weights = {
        "summarization": 0.7,
        "translation": 0.3,
    }

    merged = pool.get_merged_weights(merge_weights)
    print(f"\nMerged adapter configuration:")
    for name, weight in merge_weights.items():
        print(f"  {name}: {weight * 100:.0f}%")

    print("\nMerged inference (simulated):")
    print("  Prompt: Summarize and translate: Machine learning enables...")
    print("  Response: L'apprentissage automatique permet aux ordinateurs...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("MULTI-ADAPTER BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for multi-adapter deployment:

    1. Adapter pool sizing
       - Pre-load frequently used adapters
       - Use LRU eviction for large adapter sets
       - Monitor memory usage per adapter

    2. Routing strategies
       - Header-based: X-Adapter-Name header
       - Path-based: /v1/adapters/{name}/generate
       - Content-based: Automatic task detection

    3. Merging considerations
       - Only merge compatible adapters (same base model)
       - Validate merged output quality
       - Consider task conflicts

    4. Production deployment
       - Health checks per adapter
       - Metrics per adapter (latency, usage)
       - Graceful adapter updates (blue-green)
    """)


if __name__ == "__main__":
    main()
