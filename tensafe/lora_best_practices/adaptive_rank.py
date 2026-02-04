"""
Adaptive Rank Selection for LoRA

Utilities for selecting optimal LoRA rank based on:
- Dataset size and complexity
- Available compute/memory
- Task requirements

Based on findings from:
- "LoRA Without Regret" (dataset capacity analysis)
- AdaLoRA (per-layer adaptive rank)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math


class RankSelectionStrategy(Enum):
    """Strategies for selecting LoRA rank."""

    FIXED = "fixed"
    """Use a fixed rank across all layers."""

    DATASET_BASED = "dataset_based"
    """Select rank based on dataset size (from LoRA Without Regret)."""

    MEMORY_CONSTRAINED = "memory_constrained"
    """Select maximum rank that fits in memory budget."""

    ADAPTIVE = "adaptive"
    """Use AdaLoRA-style adaptive rank per layer."""

    TASK_BASED = "task_based"
    """Select based on task complexity heuristics."""


@dataclass
class RankRecommendation:
    """Recommendation for LoRA rank selection."""

    recommended_rank: int
    min_rank: int
    max_rank: int
    confidence: float  # 0-1, how confident in this recommendation
    reasoning: str
    alternatives: List[Tuple[int, str]]  # (rank, reason)


@dataclass
class DatasetAnalysis:
    """Analysis of dataset for rank selection."""

    num_examples: int
    estimated_information_bits: float
    recommended_rank: int
    capacity_ratio: float  # How much of LoRA capacity is used


class AdaptiveRankSelector:
    """
    Adaptive rank selection based on multiple factors.

    This class helps determine the optimal LoRA rank considering:
    1. Dataset size (primary factor from research)
    2. Task complexity
    3. Memory constraints
    4. Target quality level
    """

    # From "LoRA Without Regret": rank-32 handles ~50k examples
    EXAMPLES_PER_RANK_UNIT = 1500  # ~50k / 32

    # Minimum and maximum supported ranks
    MIN_RANK = 4
    MAX_RANK = 512

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        max_memory_gb: Optional[float] = None,
    ):
        """
        Initialize the rank selector.

        Args:
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            max_memory_gb: Maximum memory budget in GB
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_memory_gb = max_memory_gb

    def estimate_rank_from_dataset(self, num_examples: int) -> int:
        """
        Estimate optimal rank from dataset size.

        Based on "LoRA Without Regret" finding that rank-32 can handle
        ~50k examples before capacity becomes limiting.

        Args:
            num_examples: Number of training examples

        Returns:
            Recommended rank
        """
        # Base calculation
        raw_rank = num_examples / self.EXAMPLES_PER_RANK_UNIT

        # Round to nearest power of 2 (common practice)
        log_rank = math.log2(max(raw_rank, 1))
        rank = int(2 ** round(log_rank))

        # Clamp to valid range
        return max(self.MIN_RANK, min(rank, self.MAX_RANK))

    def estimate_memory_for_rank(
        self,
        rank: int,
        num_target_modules: int = 7,  # All linear in LLaMA
        precision_bytes: int = 2,  # bfloat16
    ) -> float:
        """
        Estimate memory required for a given rank.

        Args:
            rank: LoRA rank
            num_target_modules: Number of modules per layer
            precision_bytes: Bytes per parameter

        Returns:
            Estimated memory in GB
        """
        # Parameters per adapter: A (r × hidden) + B (hidden × r)
        params_per_adapter = 2 * rank * self.hidden_size

        # Total LoRA parameters
        total_params = params_per_adapter * num_target_modules * self.num_layers

        # Memory in GB
        return (total_params * precision_bytes) / (1024 ** 3)

    def max_rank_for_memory(
        self,
        memory_budget_gb: float,
        num_target_modules: int = 7,
        precision_bytes: int = 2,
    ) -> int:
        """
        Calculate maximum rank that fits in memory budget.

        Args:
            memory_budget_gb: Available memory in GB
            num_target_modules: Number of modules per layer
            precision_bytes: Bytes per parameter

        Returns:
            Maximum feasible rank
        """
        # Solve for rank from memory equation
        # memory = 2 * rank * hidden * modules * layers * bytes
        bytes_per_rank_unit = (
            2 * self.hidden_size * num_target_modules *
            self.num_layers * precision_bytes
        )

        max_rank = int((memory_budget_gb * (1024 ** 3)) / bytes_per_rank_unit)

        # Round down to power of 2
        if max_rank > 0:
            log_rank = math.floor(math.log2(max_rank))
            max_rank = int(2 ** log_rank)

        return max(self.MIN_RANK, min(max_rank, self.MAX_RANK))

    def analyze_dataset(self, num_examples: int) -> DatasetAnalysis:
        """
        Analyze dataset for rank selection.

        Args:
            num_examples: Number of training examples

        Returns:
            DatasetAnalysis with detailed breakdown
        """
        recommended_rank = self.estimate_rank_from_dataset(num_examples)

        # Estimate information content (rough heuristic)
        # Assuming ~10 bits of information per example on average
        estimated_bits = num_examples * 10

        # Estimate LoRA capacity in bits
        # Each rank adds hidden_size * 2 parameters per adapter
        lora_capacity_params = 2 * recommended_rank * self.hidden_size * 7 * self.num_layers
        lora_capacity_bits = lora_capacity_params * 16  # Assuming 16 useful bits per param

        capacity_ratio = estimated_bits / max(lora_capacity_bits, 1)

        return DatasetAnalysis(
            num_examples=num_examples,
            estimated_information_bits=estimated_bits,
            recommended_rank=recommended_rank,
            capacity_ratio=min(capacity_ratio, 1.0),
        )

    def recommend(
        self,
        num_examples: int,
        task_complexity: str = "medium",
        quality_priority: float = 0.5,  # 0 = speed, 1 = quality
    ) -> RankRecommendation:
        """
        Get comprehensive rank recommendation.

        Args:
            num_examples: Number of training examples
            task_complexity: "simple", "medium", or "complex"
            quality_priority: Priority for quality vs speed (0-1)

        Returns:
            RankRecommendation with detailed analysis
        """
        # Base rank from dataset
        base_rank = self.estimate_rank_from_dataset(num_examples)

        # Complexity adjustment
        complexity_multiplier = {
            "simple": 0.5,
            "medium": 1.0,
            "complex": 2.0,
        }.get(task_complexity, 1.0)

        adjusted_rank = int(base_rank * complexity_multiplier)

        # Quality adjustment
        if quality_priority > 0.7:
            adjusted_rank = int(adjusted_rank * 1.5)
        elif quality_priority < 0.3:
            adjusted_rank = int(adjusted_rank * 0.75)

        # Memory constraint
        max_feasible_rank = self.MAX_RANK
        if self.max_memory_gb is not None:
            max_feasible_rank = self.max_rank_for_memory(self.max_memory_gb)

        # Final rank
        recommended_rank = max(
            self.MIN_RANK,
            min(adjusted_rank, max_feasible_rank)
        )

        # Round to power of 2
        log_rank = round(math.log2(max(recommended_rank, 1)))
        recommended_rank = int(2 ** log_rank)
        recommended_rank = max(self.MIN_RANK, min(recommended_rank, self.MAX_RANK))

        # Confidence based on how much adjustment was needed
        adjustment_factor = abs(recommended_rank - base_rank) / max(base_rank, 1)
        confidence = max(0.5, 1.0 - adjustment_factor * 0.5)

        # Build reasoning
        reasoning_parts = [
            f"Dataset size ({num_examples:,} examples) suggests base rank ~{base_rank}",
        ]
        if complexity_multiplier != 1.0:
            reasoning_parts.append(
                f"Task complexity '{task_complexity}' adjusts by {complexity_multiplier}x"
            )
        if self.max_memory_gb is not None and adjusted_rank > max_feasible_rank:
            reasoning_parts.append(
                f"Memory constraint ({self.max_memory_gb}GB) limits to rank {max_feasible_rank}"
            )

        # Alternatives
        alternatives = []
        for alt_rank in [8, 16, 32, 64, 128]:
            if alt_rank != recommended_rank and alt_rank <= max_feasible_rank:
                if alt_rank < recommended_rank:
                    alternatives.append((alt_rank, "Faster training, may underfit"))
                else:
                    alternatives.append((alt_rank, "Higher capacity, more VRAM"))

        return RankRecommendation(
            recommended_rank=recommended_rank,
            min_rank=max(self.MIN_RANK, recommended_rank // 2),
            max_rank=min(self.MAX_RANK, recommended_rank * 2, max_feasible_rank),
            confidence=confidence,
            reasoning=". ".join(reasoning_parts),
            alternatives=alternatives[:3],
        )


def estimate_optimal_rank(
    num_examples: int,
    hidden_size: int = 4096,
    task_complexity: str = "medium",
) -> int:
    """
    Quick utility to estimate optimal rank.

    Args:
        num_examples: Number of training examples
        hidden_size: Model hidden size
        task_complexity: "simple", "medium", or "complex"

    Returns:
        Recommended rank
    """
    selector = AdaptiveRankSelector(hidden_size=hidden_size)
    recommendation = selector.recommend(num_examples, task_complexity)
    return recommendation.recommended_rank


# Rank selection guidelines
RANK_SELECTION_GUIDE = """
LoRA Rank Selection Guide
=========================

Based on research findings, particularly "LoRA Without Regret":

Dataset Size → Recommended Rank
-------------------------------
< 5,000      →  8-16
5K - 20K     →  16-32
20K - 50K    →  32
50K - 100K   →  64
100K - 500K  →  128
> 500K       →  256+ (or consider full fine-tuning)

Task Complexity Adjustments
---------------------------
Simple (classification, sentiment):    ×0.5
Medium (instruction following):        ×1.0
Complex (math reasoning, code):        ×2.0

Memory Considerations (7B model, bfloat16)
------------------------------------------
Rank 16:   ~100MB LoRA parameters
Rank 32:   ~200MB LoRA parameters
Rank 64:   ~400MB LoRA parameters
Rank 128:  ~800MB LoRA parameters
Rank 256:  ~1.6GB LoRA parameters

Key Insight
-----------
The capacity limitation is NOT about parameter count, but about
how much "information" the LoRA can store. A rank-32 adapter
can effectively match full fine-tuning on ~50K examples.

When to Increase Rank
---------------------
1. Training loss plateaus significantly above validation baseline
2. Model fails to learn domain-specific patterns
3. Large, diverse dataset requires more capacity

When to Decrease Rank
---------------------
1. Overfitting observed (train >> val loss)
2. Memory constraints
3. Faster training needed with acceptable quality trade-off
"""
