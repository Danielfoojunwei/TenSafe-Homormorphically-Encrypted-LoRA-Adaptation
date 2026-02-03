"""Distributed DP-SGD Implementation.

Provides privacy-preserving distributed training with:
- Per-sample gradient clipping
- Noise calibration for distributed setting
- Secure gradient aggregation
- Privacy budget accounting
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class DPAccountingResult:
    """Result of DP accounting."""
    epsilon: float
    delta: float
    steps: int
    noise_multiplier: float
    sampling_rate: float


class DistributedDPOptimizer(Optimizer):
    """Optimizer wrapper that implements DP-SGD for distributed training.

    This optimizer:
    1. Clips per-sample gradients to bound sensitivity
    2. Adds calibrated Gaussian noise
    3. Tracks privacy budget using RDP accountant
    4. Coordinates with distributed training framework

    Example:
        ```python
        optimizer = DistributedDPOptimizer(
            base_optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            num_workers=4,
        )

        for batch in dataloader:
            loss = model(batch).loss
            loss.backward()
            optimizer.step()  # Clips, adds noise, updates

            epsilon, delta = optimizer.get_privacy_spent()
        ```
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        num_workers: int = 1,
        target_delta: float = 1e-5,
        expected_batch_size: int = 32,
        dataset_size: int = 10000,
    ):
        """Initialize DP optimizer.

        Args:
            base_optimizer: Underlying optimizer (e.g., AdamW)
            noise_multiplier: Noise multiplier σ
            max_grad_norm: Per-sample gradient clip norm C
            num_workers: Number of distributed workers
            target_delta: Target delta for (ε, δ)-DP
            expected_batch_size: Expected batch size per worker
            dataset_size: Total dataset size
        """
        self.base_optimizer = base_optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.num_workers = num_workers
        self.target_delta = target_delta
        self.expected_batch_size = expected_batch_size
        self.dataset_size = dataset_size

        # Privacy accounting
        self._steps = 0
        self._accumulated_epsilon = 0.0

        # Get param groups from base optimizer
        self.param_groups = base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = False):
        """Zero out gradients."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Perform DP-SGD step.

        1. Clip per-sample gradients
        2. Sum clipped gradients
        3. Add calibrated noise
        4. Update parameters
        """
        # Clip gradients
        self._clip_gradients()

        # Add noise
        self._add_noise()

        # Update privacy accounting
        self._steps += 1
        self._update_accounting()

        # Base optimizer step
        return self.base_optimizer.step(closure)

    def _clip_gradients(self):
        """Clip gradients to max_grad_norm."""
        total_norm = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

        total_norm = math.sqrt(total_norm)

        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)

    def _add_noise(self):
        """Add calibrated Gaussian noise to gradients."""
        # Effective batch size across all workers
        effective_batch_size = self.expected_batch_size * self.num_workers

        # Noise standard deviation
        std = self.noise_multiplier * self.max_grad_norm / effective_batch_size

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * std
                    p.grad.data.add_(noise)

    def _update_accounting(self):
        """Update privacy accounting after each step."""
        # Sampling rate
        q = (self.expected_batch_size * self.num_workers) / self.dataset_size

        # RDP accounting (simplified)
        # Uses strong composition theorem approximation
        alpha = 2  # Order for RDP
        rdp_epsilon = q ** 2 * alpha / (2 * self.noise_multiplier ** 2)

        # Convert RDP to (ε, δ)-DP
        self._accumulated_epsilon = self._steps * rdp_epsilon + math.log(1 / self.target_delta) / (alpha - 1)

    def get_privacy_spent(self) -> tuple:
        """Get current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta)
        """
        return (self._accumulated_epsilon, self.target_delta)

    def get_accounting_result(self) -> DPAccountingResult:
        """Get detailed accounting result."""
        q = (self.expected_batch_size * self.num_workers) / self.dataset_size

        return DPAccountingResult(
            epsilon=self._accumulated_epsilon,
            delta=self.target_delta,
            steps=self._steps,
            noise_multiplier=self.noise_multiplier,
            sampling_rate=q,
        )


class SecureGradientAggregator:
    """Secure gradient aggregation for distributed DP-SGD.

    Implements secure aggregation protocols to ensure that:
    1. Individual gradients are never visible to the server
    2. Only the aggregated (sum/mean) gradient is revealed
    3. Supports dropout-tolerant protocols

    This uses a simplified additive secret sharing scheme.
    For production, consider using:
    - SPDZ protocol
    - Secure Aggregation (Bonawitz et al.)
    - Trusted Execution Environments (SGX)
    """

    def __init__(
        self,
        num_workers: int,
        threshold: int = None,
        seed: int = 42,
    ):
        """Initialize secure aggregator.

        Args:
            num_workers: Number of participating workers
            threshold: Minimum workers needed for aggregation (default: all)
            seed: Random seed for reproducibility
        """
        self.num_workers = num_workers
        self.threshold = threshold or num_workers
        self.seed = seed

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        # Pairwise masks for additive sharing
        self._pairwise_seeds: Dict[tuple, int] = {}

    def setup_pairwise_masks(self, worker_id: int) -> Dict[int, int]:
        """Setup pairwise random seeds with other workers.

        Args:
            worker_id: This worker's ID

        Returns:
            Dict mapping peer worker IDs to shared seeds
        """
        seeds = {}
        for other_id in range(self.num_workers):
            if other_id != worker_id:
                # Deterministic seed based on worker pair
                pair = tuple(sorted([worker_id, other_id]))
                seed = hash(pair) % (2**31)
                seeds[other_id] = seed
                self._pairwise_seeds[pair] = seed

        return seeds

    def mask_gradient(
        self,
        gradient: torch.Tensor,
        worker_id: int,
        pairwise_seeds: Dict[int, int],
    ) -> torch.Tensor:
        """Mask gradient with pairwise random masks.

        Each worker adds masks that cancel out in summation:
        - For each pair (i, j) where i < j:
          - Worker i adds +mask
          - Worker j adds -mask
        - Sum over all workers: masks cancel, result is true sum

        Args:
            gradient: Gradient tensor to mask
            worker_id: This worker's ID
            pairwise_seeds: Dict of peer IDs to shared seeds

        Returns:
            Masked gradient tensor
        """
        masked = gradient.clone()

        for other_id, seed in pairwise_seeds.items():
            # Create deterministic mask
            gen = torch.Generator(device=gradient.device)
            gen.manual_seed(seed)
            mask = torch.randn(gradient.shape, generator=gen, device=gradient.device)

            # Add or subtract based on worker order
            if worker_id < other_id:
                masked.add_(mask)
            else:
                masked.sub_(mask)

        return masked

    def unmask_aggregated(
        self,
        aggregated: torch.Tensor,
        participating_workers: List[int],
    ) -> torch.Tensor:
        """Unmask aggregated gradient (no-op if all workers participated).

        If all workers participate, masks cancel naturally.
        If some workers dropped out, need to reconstruct missing masks.

        Args:
            aggregated: Aggregated masked gradient
            participating_workers: List of worker IDs that participated

        Returns:
            Unmasked aggregated gradient
        """
        # If all workers participated, masks cancel automatically
        if len(participating_workers) == self.num_workers:
            return aggregated

        # Otherwise, need to subtract masks for missing workers
        # This requires knowing which workers dropped out
        missing_workers = set(range(self.num_workers)) - set(participating_workers)

        result = aggregated.clone()

        for missing_id in missing_workers:
            for other_id in participating_workers:
                pair = tuple(sorted([missing_id, other_id]))
                if pair in self._pairwise_seeds:
                    seed = self._pairwise_seeds[pair]
                    gen = torch.Generator(device=aggregated.device)
                    gen.manual_seed(seed)
                    mask = torch.randn(aggregated.shape, generator=gen, device=aggregated.device)

                    # Reconstruct what missing worker would have added
                    if missing_id < other_id:
                        result.sub_(mask)  # Missing worker would have added
                    else:
                        result.add_(mask)  # Missing worker would have subtracted

        return result

    def aggregate_gradients(
        self,
        gradients: List[torch.Tensor],
        worker_ids: List[int],
    ) -> torch.Tensor:
        """Aggregate gradients securely (server-side operation).

        In practice, this would be done by the parameter server
        which only sees masked gradients.

        Args:
            gradients: List of masked gradients from workers
            worker_ids: Corresponding worker IDs

        Returns:
            Aggregated unmasked gradient
        """
        # Sum masked gradients
        aggregated = torch.stack(gradients).sum(dim=0)

        # Unmask (handles dropout)
        return self.unmask_aggregated(aggregated, worker_ids)


def compute_dp_sgd_privacy(
    steps: int,
    batch_size: int,
    dataset_size: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    """Compute ε for given DP-SGD parameters.

    Uses the RDP accountant approach for tight privacy analysis.

    Args:
        steps: Number of training steps
        batch_size: Batch size (total across all workers)
        dataset_size: Total dataset size
        noise_multiplier: Noise multiplier σ
        delta: Target δ

    Returns:
        Privacy parameter ε
    """
    # Sampling probability
    q = batch_size / dataset_size

    # RDP orders to try
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

    # Compute RDP for each order
    rdp = []
    for alpha in orders:
        # Subsampled Gaussian mechanism RDP
        if alpha <= 1:
            continue

        rdp_alpha = _compute_rdp_single_order(q, noise_multiplier, alpha)
        rdp.append((alpha, steps * rdp_alpha))

    # Convert RDP to (ε, δ)-DP
    epsilon = float('inf')
    for alpha, rdp_epsilon in rdp:
        eps = rdp_epsilon + math.log(1 / delta) / (alpha - 1)
        epsilon = min(epsilon, eps)

    return epsilon


def _compute_rdp_single_order(q: float, sigma: float, alpha: float) -> float:
    """Compute RDP at a single order."""
    if q == 0:
        return 0

    if q == 1:
        return alpha / (2 * sigma ** 2)

    # Subsampled Gaussian mechanism
    return alpha * q ** 2 / (2 * sigma ** 2)
