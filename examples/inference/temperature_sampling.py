#!/usr/bin/env python3
"""
Temperature and Sampling Parameters with TenSafe

This example demonstrates how different sampling parameters affect model
outputs. Understanding these parameters is essential for controlling
the creativity, diversity, and quality of generated text.

What this example demonstrates:
- Temperature effects on output randomness
- Top-p (nucleus) sampling
- Top-k sampling
- Repetition penalty and frequency penalty
- Combining parameters effectively

Key concepts:
- Temperature: Controls probability distribution sharpness
- Top-p: Cumulative probability threshold
- Top-k: Number of top tokens to consider
- Penalties: Reduce repetition in outputs

Prerequisites:
- TenSafe server running
- Trained LoRA adapter

Expected Output:
    Temperature comparison:
    T=0.0: "The capital of France is Paris."
    T=0.7: "The capital of France is Paris, known for..."
    T=1.5: "The capital of France is the magnificent Paris..."

    Top-p comparison:
    p=0.1: More focused, predictable output
    p=0.9: More diverse, creative output
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class SamplingParams:
    """Parameters controlling text generation sampling."""
    # Temperature: Higher = more random (0.0 to 2.0)
    temperature: float = 1.0

    # Top-p (nucleus sampling): Cumulative probability cutoff (0.0 to 1.0)
    top_p: float = 1.0

    # Top-k: Number of highest probability tokens to consider
    top_k: int = 50

    # Repetition penalty: Penalize repeated tokens (1.0 = no penalty)
    repetition_penalty: float = 1.0

    # Frequency penalty: Penalize based on frequency in output (0.0 to 2.0)
    frequency_penalty: float = 0.0

    # Presence penalty: Penalize any token that appeared (0.0 to 2.0)
    presence_penalty: float = 0.0

    # Max tokens to generate
    max_tokens: int = 100

    # Stop sequences
    stop: Optional[List[str]] = None


def simulate_generation(prompt: str, params: SamplingParams) -> str:
    """Simulate text generation with given parameters."""
    # This simulates how different parameters affect outputs
    # In production, this would call the actual model

    base_responses = {
        "creative_writing": {
            "cold": "The sun set over the mountains.",
            "warm": "The sun painted the mountains in hues of gold and crimson as it descended.",
            "hot": "Like a celestial artist bidding farewell, the sun splashed its palette across the ancient peaks, each ray a brushstroke of amber and rose.",
        },
        "factual": {
            "cold": "The capital of France is Paris.",
            "warm": "The capital of France is Paris, a city known for its culture and history.",
            "hot": "The magnificent capital of France is none other than Paris, the City of Light!",
        },
        "code": {
            "cold": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "warm": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "hot": "def calculate_factorial(number):\n    # Recursive factorial\n    if number <= 1:\n        return 1\n    result = number * calculate_factorial(number - 1)\n    return result",
        }
    }

    # Determine response type based on prompt
    if "story" in prompt.lower() or "write" in prompt.lower():
        category = "creative_writing"
    elif "code" in prompt.lower() or "function" in prompt.lower():
        category = "code"
    else:
        category = "factual"

    # Select response based on temperature
    if params.temperature < 0.3:
        return base_responses[category]["cold"]
    elif params.temperature < 1.0:
        return base_responses[category]["warm"]
    else:
        return base_responses[category]["hot"]


def main():
    """Demonstrate temperature and sampling parameters."""

    # =========================================================================
    # Step 1: Understanding sampling parameters
    # =========================================================================
    print("=" * 60)
    print("TEMPERATURE AND SAMPLING PARAMETERS")
    print("=" * 60)
    print("""
    Sampling parameters control how the model selects tokens:

    TEMPERATURE (0.0 - 2.0)
    Controls randomness of the probability distribution.
    - 0.0: Deterministic (always pick highest probability)
    - 0.7: Balanced creativity (recommended for most uses)
    - 1.0: Sample from unmodified distribution
    - >1.0: More random, potentially incoherent

    TOP-P (Nucleus Sampling, 0.0 - 1.0)
    Consider tokens until cumulative probability reaches p.
    - 0.1: Very focused (top ~10% probability mass)
    - 0.9: Diverse (top ~90% probability mass)
    - 1.0: Consider all tokens

    TOP-K
    Only consider the k highest probability tokens.
    - 1: Always pick the most likely token
    - 50: Consider top 50 tokens
    - 0: Disable top-k filtering

    These parameters interact! Common combinations:
    - Factual/Code: temperature=0.0-0.3, top_p=0.1
    - Balanced: temperature=0.7, top_p=0.9
    - Creative: temperature=1.0-1.2, top_p=0.95
    """)

    # =========================================================================
    # Step 2: Temperature comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEMPERATURE COMPARISON")
    print("=" * 60)

    prompt = "Tell me about the capital of France"
    print(f"\nPrompt: \"{prompt}\"")
    print("-" * 50)

    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]

    for temp in temperatures:
        params = SamplingParams(temperature=temp, top_p=1.0)
        response = simulate_generation(prompt, params)
        temp_label = {0.0: "deterministic", 0.3: "focused", 0.7: "balanced",
                      1.0: "creative", 1.5: "very random"}
        print(f"\nT={temp} ({temp_label.get(temp, '')}):")
        print(f"  {response}")

    # =========================================================================
    # Step 3: Top-p comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP-P (NUCLEUS SAMPLING) COMPARISON")
    print("=" * 60)

    prompt = "Write the beginning of a story"
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Temperature: 0.8 (fixed)")
    print("-" * 50)

    top_p_values = [0.1, 0.5, 0.9, 1.0]

    simulated_top_p_responses = {
        0.1: "Once upon a time, there was a young girl named Alice.",
        0.5: "Once upon a time, in a distant kingdom, there lived a curious young woman.",
        0.9: "In the mystical realm of Eldoria, where dragons soared and magic flowed like rivers, a young adventurer named Kira began her journey.",
        1.0: "Beneath the shimmering auroras of the crystal moon, where time itself danced in spirals of starlight, an unlikely hero emerged from the shadows of forgotten dreams.",
    }

    for top_p in top_p_values:
        params = SamplingParams(temperature=0.8, top_p=top_p)
        response = simulated_top_p_responses[top_p]
        print(f"\ntop_p={top_p}:")
        print(f"  {response}")

    # =========================================================================
    # Step 4: Top-k comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP-K COMPARISON")
    print("=" * 60)

    print(f"\nTop-k limits the token selection pool:")
    print("-" * 50)

    top_k_examples = {
        1: "Most probable token only (greedy decoding)",
        10: "Top 10 tokens - focused but some variety",
        50: "Top 50 tokens - balanced (common default)",
        100: "Top 100 tokens - more diverse outputs",
    }

    for k, description in top_k_examples.items():
        print(f"  k={k:3d}: {description}")

    # =========================================================================
    # Step 5: Repetition penalties
    # =========================================================================
    print("\n" + "=" * 60)
    print("REPETITION PENALTIES")
    print("=" * 60)
    print("""
    Penalties reduce repetitive outputs:

    REPETITION PENALTY (default: 1.0)
    Multiplier applied to repeated token probabilities.
    - 1.0: No penalty
    - 1.1-1.3: Mild penalty (recommended)
    - >1.5: Strong penalty (may hurt coherence)

    FREQUENCY PENALTY (default: 0.0)
    Subtracts value based on token frequency in output.
    - 0.0: No penalty
    - 0.5-1.0: Moderate (encourages variety)
    - >1.5: Strong (may cause topic drift)

    PRESENCE PENALTY (default: 0.0)
    Subtracts value if token appeared at all.
    - 0.0: No penalty
    - 0.5-1.0: Encourages new topics
    - >1.5: Strong (may hurt coherence)
    """)

    # Example with repetition
    print("Example - Without penalty:")
    print("  'The cat sat on the mat. The cat was a happy cat. The cat...'")
    print("\nExample - With repetition_penalty=1.2:")
    print("  'The cat sat on the mat. She was happy and content. Her soft fur...'")

    # =========================================================================
    # Step 6: Recommended configurations
    # =========================================================================
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 60)

    configs = {
        "Code Generation": SamplingParams(
            temperature=0.2,
            top_p=0.1,
            top_k=10,
            repetition_penalty=1.0,
            max_tokens=500,
        ),
        "Factual Q&A": SamplingParams(
            temperature=0.3,
            top_p=0.5,
            top_k=40,
            repetition_penalty=1.0,
            max_tokens=200,
        ),
        "Creative Writing": SamplingParams(
            temperature=0.9,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.1,
            frequency_penalty=0.5,
            max_tokens=1000,
        ),
        "Chat/Conversation": SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            presence_penalty=0.3,
            max_tokens=500,
        ),
        "Summarization": SamplingParams(
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.2,
            max_tokens=300,
        ),
    }

    for task, params in configs.items():
        print(f"\n{task}:")
        print(f"  temperature={params.temperature}, top_p={params.top_p}, "
              f"top_k={params.top_k}")
        print(f"  repetition_penalty={params.repetition_penalty}, "
              f"frequency_penalty={params.frequency_penalty}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key takeaways:

    1. Start with recommended configs, then tune
    2. Lower temperature for factual/code tasks
    3. Higher temperature for creative tasks
    4. Use repetition penalties for longer outputs
    5. top_p and temperature interact - adjust together
    6. Test with representative prompts before deployment
    7. Monitor output quality in production
    """)


if __name__ == "__main__":
    main()
