#!/usr/bin/env python3
"""
Privacy Budget Tracking with TenSafe

This example demonstrates how to monitor and manage differential privacy
budget consumption during training. Proper budget tracking is essential
to ensure privacy guarantees are maintained throughout the training process.

What this example demonstrates:
- Setting up privacy budget alerts
- Monitoring epsilon consumption over time
- Implementing early stopping based on budget
- Visualizing privacy budget usage

Key concepts:
- Privacy accounting: Track cumulative privacy loss
- Budget allocation: Divide epsilon across training phases
- Composition: How privacy degrades with multiple queries
- Renyi DP: Tighter bounds for privacy accounting

Prerequisites:
- TenSafe server running

Expected Output:
    Privacy Budget Dashboard
    ========================
    Total budget: 8.0 epsilon
    Warning threshold: 6.4 epsilon (80%)

    Step  20: [====            ] 10% (0.80 / 8.00)
    Step  40: [========        ] 20% (1.60 / 8.00)
    ...
    Step 160: [==========WARNING] 80% (6.40 / 8.00)
    ...
    Step 200: [================STOP] 100% Budget exhausted
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class BudgetAlert:
    """Configuration for budget alerts."""
    threshold_pct: float  # Percentage of budget
    message: str
    triggered: bool = False


@dataclass
class BudgetTrackerConfig:
    """Configuration for privacy budget tracking."""
    total_epsilon: float = 8.0
    total_delta: float = 1e-5
    warning_threshold_pct: float = 0.8
    critical_threshold_pct: float = 0.95
    log_interval: int = 20


class PrivacyBudgetTracker:
    """Track and manage differential privacy budget."""

    def __init__(self, config: BudgetTrackerConfig):
        self.config = config
        self.epsilon_spent = 0.0
        self.steps = 0
        self.history: List[tuple[int, float]] = []
        self.alerts: List[BudgetAlert] = [
            BudgetAlert(0.5, "50% of privacy budget consumed"),
            BudgetAlert(0.8, "WARNING: 80% of privacy budget consumed"),
            BudgetAlert(0.95, "CRITICAL: 95% of privacy budget consumed"),
            BudgetAlert(1.0, "STOP: Privacy budget exhausted"),
        ]
        self._callbacks: List[Callable] = []

    def record_step(self, epsilon_delta: float) -> bool:
        """
        Record a training step's privacy cost.

        Returns True if training can continue, False if budget exhausted.
        """
        self.epsilon_spent += epsilon_delta
        self.steps += 1
        self.history.append((self.steps, self.epsilon_spent))

        # Check alerts
        usage_pct = self.epsilon_spent / self.config.total_epsilon
        for alert in self.alerts:
            if not alert.triggered and usage_pct >= alert.threshold_pct:
                alert.triggered = True
                self._trigger_alert(alert)

        return self.epsilon_spent < self.config.total_epsilon

    def _trigger_alert(self, alert: BudgetAlert):
        """Trigger an alert callback."""
        for callback in self._callbacks:
            callback(alert)

    def add_alert_callback(self, callback: Callable):
        """Add a callback to be called when alerts trigger."""
        self._callbacks.append(callback)

    @property
    def remaining_epsilon(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.config.total_epsilon - self.epsilon_spent)

    @property
    def usage_percentage(self) -> float:
        """Get percentage of budget used."""
        return (self.epsilon_spent / self.config.total_epsilon) * 100

    def get_progress_bar(self, width: int = 20) -> str:
        """Generate a text progress bar for budget usage."""
        pct = min(1.0, self.epsilon_spent / self.config.total_epsilon)
        filled = int(width * pct)
        empty = width - filled

        bar = "=" * filled + " " * empty

        # Add status indicator
        if pct >= 1.0:
            status = "STOP"
        elif pct >= 0.95:
            status = "CRITICAL"
        elif pct >= 0.8:
            status = "WARNING"
        else:
            status = ""

        return f"[{bar}] {pct*100:.0f}%{' ' + status if status else ''}"

    def get_summary(self) -> dict:
        """Get summary of budget usage."""
        return {
            "total_epsilon": self.config.total_epsilon,
            "epsilon_spent": self.epsilon_spent,
            "epsilon_remaining": self.remaining_epsilon,
            "usage_percentage": self.usage_percentage,
            "steps": self.steps,
            "epsilon_per_step": self.epsilon_spent / max(1, self.steps),
        }


def main():
    """Demonstrate privacy budget tracking."""

    # =========================================================================
    # Step 1: Understanding privacy budget
    # =========================================================================
    print("=" * 60)
    print("PRIVACY BUDGET TRACKING")
    print("=" * 60)
    print("""
    Privacy budget management is crucial for DP training:

    Key concepts:

    1. Composition Theorem
       If you run k DP algorithms with privacy (eps_1, delta_1), ...,
       (eps_k, delta_k), the total privacy is at most:
       (sum(eps_i), sum(delta_i))-DP (basic composition)

    2. Why budget matters
       - Each training step consumes some privacy budget
       - Budget is finite - once exhausted, no more training
       - Must balance utility vs. privacy preservation

    3. Budget allocation strategies
       - Fixed allocation: Same epsilon per step
       - Adaptive: More budget for important phases
       - Epoch-based: Allocate by epoch

    4. Monitoring best practices
       - Set warning thresholds (e.g., 80%)
       - Implement early stopping
       - Log budget usage for auditing
    """)

    # =========================================================================
    # Step 2: Configure budget tracker
    # =========================================================================
    print("\nConfiguring privacy budget tracker...")

    config = BudgetTrackerConfig(
        total_epsilon=8.0,
        total_delta=1e-5,
        warning_threshold_pct=0.8,
        critical_threshold_pct=0.95,
        log_interval=20,
    )

    tracker = PrivacyBudgetTracker(config)

    print(f"  Total budget: {config.total_epsilon} epsilon")
    print(f"  Warning at: {config.warning_threshold_pct * 100:.0f}%")
    print(f"  Critical at: {config.critical_threshold_pct * 100:.0f}%")

    # Add alert callback
    def alert_handler(alert: BudgetAlert):
        print(f"\n  ** ALERT: {alert.message} **\n")

    tracker.add_alert_callback(alert_handler)

    # =========================================================================
    # Step 3: Simulate training with budget tracking
    # =========================================================================
    print("\n" + "=" * 60)
    print("PRIVACY BUDGET DASHBOARD")
    print("=" * 60)

    # Simulate epsilon consumption per step (varies slightly)
    import random
    random.seed(42)

    max_steps = 250
    base_epsilon_per_step = 0.04  # ~200 steps to exhaust budget

    print(f"\nSimulating training (max {max_steps} steps)...")
    print("-" * 50)

    for step in range(1, max_steps + 1):
        # Add some variance to epsilon per step
        epsilon_delta = base_epsilon_per_step * (0.9 + random.random() * 0.2)

        # Record step and check if we can continue
        can_continue = tracker.record_step(epsilon_delta)

        # Log progress periodically
        if step % config.log_interval == 0:
            bar = tracker.get_progress_bar(20)
            spent = tracker.epsilon_spent
            total = config.total_epsilon
            print(f"Step {step:3d}: {bar} ({spent:.2f} / {total:.2f})")

        # Stop if budget exhausted
        if not can_continue:
            print(f"\nTraining stopped: Budget exhausted at step {step}")
            break

    print("-" * 50)

    # =========================================================================
    # Step 4: Budget summary
    # =========================================================================
    summary = tracker.get_summary()

    print("\n" + "=" * 60)
    print("BUDGET SUMMARY")
    print("=" * 60)
    print(f"""
    Total budget:       {summary['total_epsilon']:.2f} epsilon
    Epsilon spent:      {summary['epsilon_spent']:.2f} epsilon
    Epsilon remaining:  {summary['epsilon_remaining']:.2f} epsilon
    Usage:              {summary['usage_percentage']:.1f}%

    Training statistics:
    Total steps:        {summary['steps']}
    Avg epsilon/step:   {summary['epsilon_per_step']:.4f}

    Final guarantee:    ({summary['epsilon_spent']:.2f}, {config.total_delta:.0e})-DP
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("BUDGET MANAGEMENT TIPS")
    print("=" * 60)
    print("""
    Best practices for privacy budget management:

    1. Pre-compute expected budget consumption
       - Estimate steps needed for convergence
       - Calculate epsilon per step
       - Ensure budget is sufficient

    2. Implement monitoring and alerts
       - Set warning thresholds (80% recommended)
       - Log budget usage for audit trails
       - Send alerts to training operators

    3. Use checkpointing wisely
       - Save model at key budget milestones
       - Compare utility at different privacy levels
       - Select best privacy-utility tradeoff

    4. Consider privacy amplification
       - Subsampling amplifies privacy
       - Shuffling can provide amplification
       - Use these to stretch your budget
    """)


if __name__ == "__main__":
    main()
