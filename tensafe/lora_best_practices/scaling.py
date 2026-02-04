"""
LoRA Scaling Methods

Implements different scaling strategies for LoRA adapters:
- Standard scaling: α/r
- rsLoRA scaling: α/√r (rank-stabilized)

Reference:
- "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"
  (Kalajdzievski, 2023) https://arxiv.org/abs/2312.03732
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import math


class LoRAScaling(ABC):
    """Abstract base class for LoRA scaling strategies."""

    @abstractmethod
    def compute_scaling(self, alpha: float, rank: int) -> float:
        """Compute the scaling factor."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this scaling method."""
        pass


class StandardLoRAScaling(LoRAScaling):
    """
    Standard LoRA scaling: α/r

    This is the original scaling from the LoRA paper (Hu et al., 2021).
    Works well for low ranks (r <= 32) but can cause gradient instability
    at higher ranks.

    The scaling factor decreases linearly with rank, which means
    higher ranks contribute less per-parameter.
    """

    def compute_scaling(self, alpha: float, rank: int) -> float:
        """Compute standard scaling factor."""
        if rank <= 0:
            raise ValueError(f"Rank must be positive, got {rank}")
        return alpha / rank

    def get_name(self) -> str:
        return "standard"


class RSLoRAScaling(LoRAScaling):
    """
    Rank-Stabilized LoRA (rsLoRA) scaling: α/√r

    From "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"
    (Kalajdzievski, 2023).

    Key benefits:
    1. Stable learning at very high ranks (tested up to 2048)
    2. Better compute/performance trade-off
    3. Allows increasing rank for better performance without
       changing other hyperparameters

    The scaling factor decreases with the square root of rank,
    providing more stable gradients at higher ranks.

    Recommended when:
    - rank > 32
    - You want to experiment with different ranks
    - Training shows instability with standard scaling
    """

    def compute_scaling(self, alpha: float, rank: int) -> float:
        """Compute rsLoRA scaling factor."""
        if rank <= 0:
            raise ValueError(f"Rank must be positive, got {rank}")
        return alpha / math.sqrt(rank)

    def get_name(self) -> str:
        return "rslora"


class UnitScaling(LoRAScaling):
    """
    Unit scaling: 1.0

    No scaling applied. Useful when you want to control
    the contribution of LoRA updates directly through
    the learning rate or when merging adapters.
    """

    def compute_scaling(self, alpha: float, rank: int) -> float:
        """Return unit scaling."""
        return 1.0

    def get_name(self) -> str:
        return "unit"


@dataclass
class ScalingAnalysis:
    """Analysis of scaling behavior across different ranks."""

    method: str
    alpha: float
    scaling_factors: dict  # rank -> scaling_factor
    gradient_magnitude_ratio: dict  # rank -> relative gradient magnitude

    def get_recommendation(self) -> str:
        """Get recommendation based on analysis."""
        if self.method == "standard":
            max_stable_rank = 32
            for rank, factor in sorted(self.scaling_factors.items()):
                if factor < 0.1:  # Scaling too aggressive
                    max_stable_rank = rank - 1
                    break
            return f"Standard scaling stable up to rank ~{max_stable_rank}"
        else:
            return "rsLoRA provides stable scaling at all tested ranks"


def compute_lora_scaling(
    alpha: float,
    rank: int,
    use_rslora: bool = True,
) -> float:
    """
    Compute LoRA scaling factor.

    Args:
        alpha: LoRA alpha parameter
        rank: LoRA rank
        use_rslora: Whether to use rsLoRA scaling (recommended for rank > 32)

    Returns:
        Scaling factor to apply to LoRA output
    """
    if use_rslora:
        return RSLoRAScaling().compute_scaling(alpha, rank)
    else:
        return StandardLoRAScaling().compute_scaling(alpha, rank)


def analyze_scaling_stability(
    alpha: float,
    ranks: Optional[list] = None,
) -> ScalingAnalysis:
    """
    Analyze scaling stability across different ranks.

    This helps understand how scaling behaves and when to switch
    from standard to rsLoRA scaling.

    Args:
        alpha: LoRA alpha parameter
        ranks: List of ranks to analyze (default: powers of 2 from 1 to 512)

    Returns:
        ScalingAnalysis with detailed breakdown
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    standard = StandardLoRAScaling()
    rslora = RSLoRAScaling()

    standard_factors = {r: standard.compute_scaling(alpha, r) for r in ranks}
    rslora_factors = {r: rslora.compute_scaling(alpha, r) for r in ranks}

    # Compute relative gradient magnitudes (normalized to rank=8)
    base_rank = 8
    standard_grad_ratios = {
        r: standard_factors[r] / standard_factors[base_rank] for r in ranks
    }

    return ScalingAnalysis(
        method="comparison",
        alpha=alpha,
        scaling_factors={
            "standard": standard_factors,
            "rslora": rslora_factors,
        },
        gradient_magnitude_ratio=standard_grad_ratios,
    )


def get_optimal_alpha(
    rank: int,
    target_scaling: float = 1.0,
    use_rslora: bool = True,
) -> float:
    """
    Compute optimal alpha for a target scaling factor.

    Args:
        rank: LoRA rank
        target_scaling: Desired scaling factor
        use_rslora: Whether using rsLoRA scaling

    Returns:
        Alpha value that achieves the target scaling
    """
    if use_rslora:
        # scaling = alpha / sqrt(rank) -> alpha = scaling * sqrt(rank)
        return target_scaling * math.sqrt(rank)
    else:
        # scaling = alpha / rank -> alpha = scaling * rank
        return target_scaling * rank


def recommend_scaling_method(rank: int) -> str:
    """
    Recommend scaling method based on rank.

    Args:
        rank: LoRA rank

    Returns:
        Recommended scaling method name
    """
    if rank <= 32:
        return "standard"  # Either works well at low ranks
    elif rank <= 64:
        return "rslora"  # rsLoRA recommended
    else:
        return "rslora"  # rsLoRA strongly recommended


# Comparison table for documentation
SCALING_COMPARISON = """
Scaling Method Comparison
=========================

| Rank |  Standard (α/r)  |  rsLoRA (α/√r)  | Recommendation |
|------|------------------|-----------------|----------------|
|   8  |     α/8 = 0.125α |  α/2.83 = 0.35α | Either works   |
|  16  |    α/16 = 0.062α |     α/4 = 0.25α | Either works   |
|  32  |    α/32 = 0.031α |  α/5.66 = 0.18α | rsLoRA better  |
|  64  |    α/64 = 0.016α |     α/8 = 0.12α | Use rsLoRA     |
| 128  |   α/128 = 0.008α | α/11.3 = 0.088α | Use rsLoRA     |
| 256  |   α/256 = 0.004α |    α/16 = 0.06α | Use rsLoRA     |

Note: With standard scaling, the contribution per parameter decreases
rapidly with rank, leading to slower learning. rsLoRA maintains more
consistent learning dynamics across ranks.
"""
