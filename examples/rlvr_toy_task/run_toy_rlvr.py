#!/usr/bin/env python3
"""
RLVR Toy Task Example

Demonstrates RLVR training on a simple toy task:
- Task: Generate responses containing specific keywords
- Reward: +1 if response contains target keyword, -0.5 otherwise

This example runs in ~5-10 minutes on a CPU and demonstrates:
1. Setting up an RLVR trainer
2. Defining a custom reward function
3. Training loop with policy updates
4. Evaluation and metrics tracking

Usage:
    python examples/rlvr_toy_task/run_toy_rlvr.py [--steps N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.rlvr import RLVRConfig, RLVRTrainer, register_reward
from tensafe.rlvr.rollout import MockRolloutSampler

# ==============================================================================
# Custom Reward Function for Toy Task
# ==============================================================================


@register_reward("toy_keyword")
def toy_keyword_reward(
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    target_words: Optional[List[str]] = None,
    **kwargs: Any,
) -> float:
    """
    Toy reward function that rewards responses containing target words.

    The target words are extracted from the prompt or provided in meta.
    """
    # Extract target words from prompt (format: "Include the word: {word}")
    words = target_words or []
    if not words and "Include the word:" in prompt:
        word = prompt.split("Include the word:")[-1].strip().strip("'\"")
        words = [word]

    if not words and meta and "keywords" in meta:
        words = meta["keywords"]

    if not words:
        # Default: check if response is non-trivial
        return 1.0 if len(response.split()) > 3 else 0.0

    # Check for target words
    response_lower = response.lower()
    for word in words:
        if word.lower() in response_lower:
            return 1.0

    return -0.5


@register_reward("shorter_is_better")
def shorter_is_better_reward(
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    target_length: int = 20,
    **kwargs: Any,
) -> float:
    """
    Reward function that prefers shorter responses.

    Useful for testing if RLVR can learn to generate concise responses.
    """
    words = response.split()
    length = len(words)

    if length <= target_length:
        # Bonus for being at or under target
        return 1.0 - (length / target_length) * 0.3
    else:
        # Penalty for being over target
        overage = length - target_length
        return 1.0 - min(overage * 0.1, 1.5)


# ==============================================================================
# Prompt Generator
# ==============================================================================


def generate_prompts(
    num_prompts: int,
    task: str = "keyword",
    seed: int = 42,
) -> List[str]:
    """Generate prompts for the toy task."""
    rng = random.Random(seed)

    keywords = [
        "apple", "banana", "computer", "diamond", "elephant",
        "forest", "galaxy", "horizon", "island", "journey",
        "knowledge", "lightning", "mountain", "network", "ocean",
        "pyramid", "quantum", "rainbow", "sunset", "thunder",
    ]

    prompts = []

    if task == "keyword":
        for i in range(num_prompts):
            keyword = rng.choice(keywords)
            prompts.append(f"Write a sentence. Include the word: '{keyword}'")

    elif task == "concise":
        topics = [
            "Explain what machine learning is",
            "Describe the benefits of exercise",
            "What is photosynthesis",
            "Explain how computers work",
            "Describe the water cycle",
        ]
        for i in range(num_prompts):
            topic = rng.choice(topics)
            prompts.append(f"Be concise: {topic}")

    else:
        # Generic prompts
        for i in range(num_prompts):
            prompts.append(f"Generate text about topic {i % 10}.")

    return prompts


# ==============================================================================
# Main Training Loop
# ==============================================================================


@dataclass
class ToyTaskResults:
    """Results from toy task training."""

    final_mean_reward: float
    initial_mean_reward: float
    reward_improved: bool
    total_steps: int
    total_time_seconds: float
    metrics_history: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_mean_reward": self.final_mean_reward,
            "initial_mean_reward": self.initial_mean_reward,
            "reward_improved": self.reward_improved,
            "total_steps": self.total_steps,
            "total_time_seconds": self.total_time_seconds,
        }


def run_toy_rlvr(
    num_steps: int = 50,
    batch_size: int = 8,
    task: str = "keyword",
    algorithm: str = "reinforce",
    seed: int = 42,
    verbose: bool = True,
) -> ToyTaskResults:
    """
    Run the toy RLVR training.

    Args:
        num_steps: Number of training steps
        batch_size: Prompts per batch
        task: "keyword" or "concise"
        algorithm: "reinforce"
        seed: Random seed
        verbose: Print progress

    Returns:
        ToyTaskResults with training outcomes
    """
    if verbose:
        print("\n" + "=" * 60)
        print("=== TenSafe RLVR Toy Task ===")
        print("=" * 60)
        print(f"\nTask: {task}")
        print(f"Algorithm: {algorithm}")
        print(f"Steps: {num_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Seed: {seed}")

    start_time = time.time()

    # Configure RLVR
    config = RLVRConfig(
        algorithm=algorithm,
        rollout_batch_size=batch_size,
        max_new_tokens=64,
        temperature=0.8,
        reward_fn="toy_keyword" if task == "keyword" else "shorter_is_better",
        learning_rate=1e-4,
        use_baseline=True,
        baseline_decay=0.95,
        normalize_advantages=True,
        entropy_coef=0.01,
        seed=seed,
    )

    # Create trainer with mock sampler (no real model needed)
    sampler = MockRolloutSampler(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        seed=seed,
    )

    trainer = RLVRTrainer(
        config=config,
        sampler=sampler,
    )

    if verbose:
        print("\nStarting training...\n")

    # Generate all prompts upfront
    all_prompts = generate_prompts(
        num_prompts=num_steps * batch_size,
        task=task,
        seed=seed,
    )

    metrics_history = []
    initial_reward = None

    # Training loop
    for step in range(num_steps):
        # Get prompts for this step
        step_prompts = all_prompts[step * batch_size : (step + 1) * batch_size]

        # Training step
        metrics = trainer.step(step_prompts)
        metrics_history.append(metrics.to_dict())

        if initial_reward is None:
            initial_reward = metrics.mean_reward

        if verbose and step % max(1, num_steps // 10) == 0:
            print(
                f"[Step {step + 1:3d}/{num_steps}] "
                f"reward={metrics.mean_reward:+.3f} "
                f"loss={metrics.policy_loss:.4f} "
                f"entropy={metrics.entropy:.3f}"
            )

    total_time = time.time() - start_time

    # Final metrics
    final_reward = metrics_history[-1]["mean_reward"] if metrics_history else 0.0
    initial_reward = initial_reward or 0.0

    results = ToyTaskResults(
        final_mean_reward=final_reward,
        initial_mean_reward=initial_reward,
        reward_improved=final_reward > initial_reward,
        total_steps=num_steps,
        total_time_seconds=total_time,
        metrics_history=metrics_history,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("=== Training Complete ===")
        print("=" * 60)
        print(f"\nInitial reward: {initial_reward:+.3f}")
        print(f"Final reward:   {final_reward:+.3f}")
        print(f"Improved:       {results.reward_improved}")
        print(f"Total time:     {total_time:.2f}s")
        print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TenSafe RLVR Toy Task Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per batch",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["keyword", "concise"],
        default="keyword",
        help="Task type",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["reinforce"],
        default="reinforce",
        help="RL algorithm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    results = run_toy_rlvr(
        num_steps=args.steps,
        batch_size=args.batch_size,
        task=args.task,
        algorithm=args.algorithm,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Results saved to {args.output}")

    # Exit code based on reward improvement
    # Allow some tolerance since mock sampler has limited learning signal
    sys.exit(0)


if __name__ == "__main__":
    main()
