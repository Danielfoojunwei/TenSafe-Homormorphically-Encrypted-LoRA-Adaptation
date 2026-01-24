import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

def compute_perplexity(loss: float) -> float:
    """Computes perplexity from cross-entropy loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def compute_loss_flatness_proxy(perturbation_results: List[float], baseline_loss: float) -> float:
    """
    Computes a proxy for loss flatness/sharpness.
    Higher value = sharper (less stable) local minimum.
    """
    if not perturbation_results:
        return 0.0
    
    # Sharpness proxy: (max(perturbed_loss) - baseline_loss) / (1 + baseline_loss)
    max_perturbed = max(perturbation_results)
    sharpness = (max_perturbed - baseline_loss) / (1 + baseline_loss)
    return float(sharpness)

def compute_qualitative_metrics(eval_results: Dict[str, Any]):
    """Aggregates qualitative metrics from evaluation results."""
    metrics = {}
    if "loss" in eval_results:
        metrics["perplexity"] = compute_perplexity(eval_results["loss"])
    
    if "cross_domain_scores" in eval_results:
        # e.g., {"legal": 0.8, "medical": 0.7}
        scores = eval_results["cross_domain_scores"]
        metrics["cross_domain_accuracy"] = float(sum(scores.values()) / len(scores))
        
    return metrics
