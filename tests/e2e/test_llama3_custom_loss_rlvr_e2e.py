"""
E2E Test: Llama3 Custom Loss Functions and RLVR Mode

This test validates the new Pluggable Loss Functions and RLVR features:
1. Custom loss function registration and resolution
2. SFT training with pluggable losses
3. RLVR training with REINFORCE algorithm
4. RLVR training with PPO algorithm
5. Custom reward function registration
6. Performance metrics collection

Metrics collected:
- Loss computation overhead
- RLVR rollout generation time
- Policy gradient computation time
- Reward computation time
- Algorithm update time (REINFORCE, PPO)
"""

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class CustomLossMetrics:
    """Metrics collected during custom loss testing."""

    loss_computation_times: List[float] = field(default_factory=list)
    loss_values: List[float] = field(default_factory=list)
    custom_metric_values: Dict[str, List[float]] = field(default_factory=dict)
    steps_completed: int = 0

    def percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def to_report(self) -> Dict[str, Any]:
        return {
            "loss_computation_ms": {
                "p50": self.percentile(self.loss_computation_times, 50) * 1000,
                "p95": self.percentile(self.loss_computation_times, 95) * 1000,
                "mean": statistics.mean(self.loss_computation_times) * 1000 if self.loss_computation_times else 0,
            },
            "average_loss": statistics.mean(self.loss_values) if self.loss_values else 0.0,
            "steps_completed": self.steps_completed,
            "custom_metrics": {
                k: statistics.mean(v) if v else 0.0 for k, v in self.custom_metric_values.items()
            },
        }


@dataclass
class RLVRMetrics:
    """Metrics collected during RLVR testing."""

    rollout_times: List[float] = field(default_factory=list)
    reward_computation_times: List[float] = field(default_factory=list)
    policy_update_times: List[float] = field(default_factory=list)
    reward_values: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    entropy_values: List[float] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)
    epochs_completed: int = 0
    total_training_time: float = 0.0

    def percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def to_report(self) -> Dict[str, Any]:
        return {
            "rollout_ms": {
                "p50": self.percentile(self.rollout_times, 50) * 1000,
                "p95": self.percentile(self.rollout_times, 95) * 1000,
                "mean": statistics.mean(self.rollout_times) * 1000 if self.rollout_times else 0,
            },
            "reward_computation_ms": {
                "p50": self.percentile(self.reward_computation_times, 50) * 1000,
                "p95": self.percentile(self.reward_computation_times, 95) * 1000,
                "mean": statistics.mean(self.reward_computation_times) * 1000 if self.reward_computation_times else 0,
            },
            "policy_update_ms": {
                "p50": self.percentile(self.policy_update_times, 50) * 1000,
                "p95": self.percentile(self.policy_update_times, 95) * 1000,
                "mean": statistics.mean(self.policy_update_times) * 1000 if self.policy_update_times else 0,
            },
            "mean_reward": statistics.mean(self.reward_values) if self.reward_values else 0.0,
            "final_reward": self.reward_values[-1] if self.reward_values else 0.0,
            "mean_policy_loss": statistics.mean(self.policy_losses) if self.policy_losses else 0.0,
            "mean_entropy": statistics.mean(self.entropy_values) if self.entropy_values else 0.0,
            "epochs_completed": self.epochs_completed,
            "total_training_time_seconds": self.total_training_time,
        }


class TestPluggableLossE2E:
    """E2E tests for Pluggable Loss Functions."""

    @pytest.mark.e2e
    def test_builtin_loss_registry(self):
        """Test that all built-in losses are registered and resolvable."""
        from tensafe.training.losses import resolve_loss, get_registered_losses

        builtin_losses = ["token_ce", "margin_ranking", "contrastive", "mse"]

        for loss_name in builtin_losses:
            loss_fn = resolve_loss(loss_name)
            assert loss_fn is not None, f"Failed to resolve {loss_name}"
            assert callable(loss_fn), f"{loss_name} is not callable"

        registered = get_registered_losses()
        for loss_name in builtin_losses:
            assert loss_name in registered, f"{loss_name} not in registry"

    @pytest.mark.e2e
    def test_custom_loss_registration(self):
        """Test custom loss function registration and usage."""
        from tensafe.training.losses import register_loss, resolve_loss

        @register_loss("test_focal_loss")
        def focal_loss(outputs, batch, gamma=2.0, **kwargs):
            """Custom focal loss for testing."""
            # Mock loss computation
            loss_value = 0.5 * gamma
            return {
                "loss": loss_value,
                "metrics": {"gamma": gamma, "computed": True},
            }

        # Resolve and use
        loss_fn = resolve_loss("test_focal_loss", gamma=2.5)
        assert loss_fn is not None

        # Mock outputs and batch
        mock_outputs = type("Outputs", (), {"logits": [[0.1, 0.2, 0.7]]})()
        mock_batch = {"labels": [2]}

        result = loss_fn(mock_outputs, mock_batch)
        assert "loss" in result
        assert "metrics" in result
        assert result["metrics"]["gamma"] == 2.5
        assert result["metrics"]["computed"] is True

    @pytest.mark.e2e
    def test_loss_computation_performance(self):
        """Benchmark loss computation overhead."""
        from tensafe.training.losses import resolve_loss
        from tensafe.training.losses.builtin import MockLossFunctions

        metrics = CustomLossMetrics()

        # Test built-in loss
        loss_fn = MockLossFunctions.token_cross_entropy

        for i in range(100):
            # Mock data
            mock_outputs = type("Outputs", (), {"logits": [[0.1] * 1000]})()
            mock_batch = {"labels": [i % 1000], "input_ids": [[i % 1000] * 100]}

            start = time.perf_counter()
            result = loss_fn(mock_outputs, mock_batch)
            elapsed = time.perf_counter() - start

            metrics.loss_computation_times.append(elapsed)
            metrics.loss_values.append(result["loss"])
            metrics.steps_completed += 1

        report = metrics.to_report()

        print("\n" + "=" * 60)
        print("LOSS COMPUTATION PERFORMANCE")
        print("=" * 60)
        print(f"  Steps: {report['steps_completed']}")
        print(f"  Loss computation (p50): {report['loss_computation_ms']['p50']:.3f}ms")
        print(f"  Loss computation (p95): {report['loss_computation_ms']['p95']:.3f}ms")
        print(f"  Average loss: {report['average_loss']:.4f}")

        # Assert reasonable performance
        assert report['loss_computation_ms']['p95'] < 10.0, "Loss computation too slow"
        assert report['steps_completed'] == 100

        return report

    @pytest.mark.e2e
    def test_sft_with_custom_loss(self):
        """Test SFT training with a custom loss function."""
        from tensafe.training.losses import register_loss, resolve_loss
        from tensafe.training.losses.builtin import MockLossFunctions

        metrics = CustomLossMetrics()
        metrics.custom_metric_values["label_smoothing_effect"] = []

        @register_loss("test_smoothed_ce")
        def smoothed_ce(outputs, batch, smoothing=0.1, **kwargs):
            """Cross-entropy with label smoothing."""
            base_result = MockLossFunctions.token_cross_entropy(outputs, batch)
            # Simulate smoothing effect
            smoothed_loss = base_result["loss"] * (1 - smoothing)
            return {
                "loss": smoothed_loss,
                "metrics": {
                    "smoothing": smoothing,
                    "original_loss": base_result["loss"],
                },
            }

        loss_fn = resolve_loss("test_smoothed_ce", smoothing=0.1)

        # Simulate SFT training loop
        for step in range(50):
            mock_outputs = type("Outputs", (), {"logits": [[0.1] * 100]})()
            mock_batch = {"labels": [step % 100], "input_ids": [[step % 100] * 50]}

            start = time.perf_counter()
            result = loss_fn(mock_outputs, mock_batch)
            elapsed = time.perf_counter() - start

            metrics.loss_computation_times.append(elapsed)
            metrics.loss_values.append(result["loss"])
            metrics.custom_metric_values["label_smoothing_effect"].append(
                result["metrics"]["original_loss"] - result["loss"]
            )
            metrics.steps_completed += 1

        report = metrics.to_report()

        print("\n" + "=" * 60)
        print("SFT WITH CUSTOM LOSS")
        print("=" * 60)
        print(f"  Steps: {report['steps_completed']}")
        print(f"  Average loss: {report['average_loss']:.4f}")
        print(f"  Label smoothing effect: {report['custom_metrics']['label_smoothing_effect']:.4f}")

        assert report['steps_completed'] == 50
        assert report['custom_metrics']['label_smoothing_effect'] > 0


class TestRLVRE2E:
    """E2E tests for RLVR (Reinforcement Learning with Verifiable Rewards)."""

    @pytest.mark.e2e
    def test_reward_registry(self):
        """Test reward function registration and resolution."""
        from tensafe.rlvr import resolve_reward, register_reward, get_registered_rewards

        # Test built-in rewards
        builtin_rewards = ["keyword_contains", "length_penalty", "format_compliance"]

        for reward_name in builtin_rewards:
            reward_fn = resolve_reward(reward_name)
            assert reward_fn is not None, f"Failed to resolve {reward_name}"
            assert callable(reward_fn), f"{reward_name} is not callable"

        # Test custom reward registration
        @register_reward("test_custom_reward")
        def custom_reward(prompt: str, response: str, meta=None) -> float:
            return 0.5 if "test" in response.lower() else 0.0

        reward_fn = resolve_reward("test_custom_reward")
        assert reward_fn("prompt", "This is a test response") == 0.5
        assert reward_fn("prompt", "No keyword here") == 0.0

    @pytest.mark.e2e
    def test_reinforce_algorithm(self):
        """Test REINFORCE algorithm training loop."""
        from tensafe.rlvr import (
            MockRolloutSampler,
            REINFORCE,
            REINFORCEConfig,
            resolve_reward,
        )

        metrics = RLVRMetrics()

        # Setup
        sampler = MockRolloutSampler(max_new_tokens=32)
        reward_fn = resolve_reward("keyword_contains", keywords=["solution", "answer"])
        algo = REINFORCE(REINFORCEConfig(
            learning_rate=1e-4,
            use_baseline=True,
            normalize_advantages=True,
            entropy_coef=0.01,
        ))

        prompts = [
            "What is 2 + 2?",
            "Solve x + 3 = 7",
            "Calculate 5 * 6",
        ]

        # Create a mock training client
        class MockTrainingClient:
            def __init__(self):
                self.step = 0

            def forward_backward(self, batch):
                self.step += 1
                return type("Result", (), {"loss": 0.5, "grad_norm": 1.0})()

            def optim_step(self):
                return type("Result", (), {"step": self.step})()

        tc = MockTrainingClient()

        # Training loop
        start_time = time.perf_counter()

        for epoch in range(20):
            # Generate rollouts
            rollout_start = time.perf_counter()
            batch = sampler.generate_trajectories(prompts)
            metrics.rollout_times.append(time.perf_counter() - rollout_start)

            # Compute rewards
            reward_start = time.perf_counter()
            for traj in batch:
                traj.reward = reward_fn(traj.prompt, traj.response)
            metrics.reward_computation_times.append(time.perf_counter() - reward_start)

            # Policy update
            update_start = time.perf_counter()
            result = algo.update(batch, tc)
            metrics.policy_update_times.append(time.perf_counter() - update_start)

            # Collect metrics
            metrics.reward_values.append(batch.mean_reward)
            metrics.policy_losses.append(result.policy_loss)
            metrics.entropy_values.append(result.entropy)
            metrics.epochs_completed += 1

        metrics.total_training_time = time.perf_counter() - start_time

        report = metrics.to_report()

        print("\n" + "=" * 60)
        print("REINFORCE ALGORITHM TEST")
        print("=" * 60)
        print(f"  Epochs: {report['epochs_completed']}")
        print(f"  Total time: {report['total_training_time_seconds']:.2f}s")
        print(f"  Rollout time (p50): {report['rollout_ms']['p50']:.3f}ms")
        print(f"  Reward computation (p50): {report['reward_computation_ms']['p50']:.3f}ms")
        print(f"  Policy update (p50): {report['policy_update_ms']['p50']:.3f}ms")
        print(f"  Mean reward: {report['mean_reward']:.4f}")
        print(f"  Final reward: {report['final_reward']:.4f}")
        print(f"  Mean policy loss: {report['mean_policy_loss']:.4f}")

        assert report['epochs_completed'] == 20
        assert report['total_training_time_seconds'] < 30, "Training took too long"

        return report

    @pytest.mark.e2e
    def test_ppo_algorithm(self):
        """Test PPO algorithm training loop."""
        from tensafe.rlvr import (
            MockRolloutSampler,
            PPO,
            PPOConfig,
            resolve_reward,
        )

        metrics = RLVRMetrics()

        # Setup
        sampler = MockRolloutSampler(max_new_tokens=32)
        reward_fn = resolve_reward("length_penalty", target_length=50, penalty_scale=0.01)
        algo = PPO(PPOConfig(
            learning_rate=1e-4,
            clip_range=0.2,
            ppo_epochs=4,
            target_kl=0.01,
            entropy_coef=0.01,
        ))

        prompts = [
            "Explain machine learning.",
            "What is neural network?",
            "How does backpropagation work?",
        ]

        # Mock training client
        class MockTrainingClient:
            def __init__(self):
                self.step = 0

            def forward_backward(self, batch):
                self.step += 1
                return type("Result", (), {"loss": 0.5, "grad_norm": 1.0})()

            def optim_step(self):
                return type("Result", (), {"step": self.step})()

        tc = MockTrainingClient()

        # Training loop
        start_time = time.perf_counter()

        for epoch in range(15):
            # Generate rollouts
            rollout_start = time.perf_counter()
            batch = sampler.generate_trajectories(prompts)
            metrics.rollout_times.append(time.perf_counter() - rollout_start)

            # Compute rewards
            reward_start = time.perf_counter()
            for traj in batch:
                traj.reward = reward_fn(traj.prompt, traj.response)
            metrics.reward_computation_times.append(time.perf_counter() - reward_start)

            # PPO update (multiple epochs)
            update_start = time.perf_counter()
            result = algo.update(batch, tc)
            metrics.policy_update_times.append(time.perf_counter() - update_start)

            # Collect metrics
            metrics.reward_values.append(batch.mean_reward)
            metrics.policy_losses.append(result.policy_loss)
            metrics.entropy_values.append(result.entropy)
            if hasattr(result, 'kl_divergence'):
                metrics.kl_divergences.append(result.kl_divergence)
            metrics.epochs_completed += 1

        metrics.total_training_time = time.perf_counter() - start_time

        report = metrics.to_report()

        print("\n" + "=" * 60)
        print("PPO ALGORITHM TEST")
        print("=" * 60)
        print(f"  Epochs: {report['epochs_completed']}")
        print(f"  Total time: {report['total_training_time_seconds']:.2f}s")
        print(f"  Rollout time (p50): {report['rollout_ms']['p50']:.3f}ms")
        print(f"  Policy update (p50): {report['policy_update_ms']['p50']:.3f}ms")
        print(f"  Mean reward: {report['mean_reward']:.4f}")
        print(f"  Mean entropy: {report['mean_entropy']:.4f}")

        assert report['epochs_completed'] == 15
        assert report['total_training_time_seconds'] < 30

        return report

    @pytest.mark.e2e
    def test_custom_reward_function(self):
        """Test training with custom verifiable reward."""
        from tensafe.rlvr import (
            MockRolloutSampler,
            REINFORCE,
            REINFORCEConfig,
            register_reward,
            resolve_reward,
        )

        @register_reward("math_check")
        def math_check_reward(prompt: str, response: str, meta=None) -> float:
            """Reward for correct math answers."""
            import re

            # Extract expected answer from meta
            expected = meta.get("answer") if meta else None
            if expected is None:
                return 0.0

            # Find number in response
            numbers = re.findall(r"\d+", response)
            if not numbers:
                return 0.0

            # Check if any number matches expected
            for num in numbers:
                if int(num) == expected:
                    return 1.0

            return 0.0

        metrics = RLVRMetrics()
        sampler = MockRolloutSampler(max_new_tokens=20)
        reward_fn = resolve_reward("math_check")
        algo = REINFORCE(REINFORCEConfig(use_baseline=True))

        # Problems with verifiable answers
        problems = [
            {"prompt": "What is 2+2?", "answer": 4},
            {"prompt": "What is 3*3?", "answer": 9},
            {"prompt": "What is 10-5?", "answer": 5},
        ]

        class MockTrainingClient:
            step = 0
            def forward_backward(self, batch):
                self.step += 1
                return type("Result", (), {"loss": 0.5})()
            def optim_step(self):
                return type("Result", (), {"step": self.step})()

        tc = MockTrainingClient()

        for epoch in range(10):
            prompts = [p["prompt"] for p in problems]
            batch = sampler.generate_trajectories(prompts)

            # Compute verifiable rewards
            for i, traj in enumerate(batch):
                traj.reward = reward_fn(traj.prompt, traj.response, {"answer": problems[i]["answer"]})

            result = algo.update(batch, tc)
            metrics.reward_values.append(batch.mean_reward)
            metrics.epochs_completed += 1

        print("\n" + "=" * 60)
        print("CUSTOM VERIFIABLE REWARD TEST")
        print("=" * 60)
        print(f"  Epochs: {metrics.epochs_completed}")
        print(f"  Mean reward: {statistics.mean(metrics.reward_values):.4f}")

        assert metrics.epochs_completed == 10


class TestCombinedE2E:
    """Combined E2E tests for Loss + RLVR features."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_feature_benchmark(self):
        """
        Full benchmark of all new features.

        Tests:
        1. Custom loss with SFT
        2. REINFORCE training
        3. PPO training
        4. Performance comparison
        """
        from tensafe.training.losses import register_loss, resolve_loss
        from tensafe.training.losses.builtin import MockLossFunctions
        from tensafe.rlvr import (
            MockRolloutSampler,
            REINFORCE,
            REINFORCEConfig,
            PPO,
            PPOConfig,
            resolve_reward,
        )

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_type": "Full Feature Benchmark",
        }

        # For CI, use shorter iterations
        num_iterations = 10 if os.getenv("FULL_E2E") != "1" else 50

        # ===================================================================
        # Phase 1: Custom Loss SFT
        # ===================================================================
        print("\n" + "=" * 60)
        print("PHASE 1: Custom Loss SFT Benchmark")
        print("=" * 60)

        loss_metrics = CustomLossMetrics()

        @register_loss("benchmark_loss")
        def benchmark_loss(outputs, batch, weight=1.0, **kwargs):
            base = MockLossFunctions.token_cross_entropy(outputs, batch)
            return {
                "loss": base["loss"] * weight,
                "metrics": {"weight": weight},
            }

        loss_fn = resolve_loss("benchmark_loss", weight=1.0)

        for i in range(num_iterations):
            mock_outputs = type("Outputs", (), {"logits": [[0.1] * 100]})()
            mock_batch = {"labels": [i % 100], "input_ids": [[i % 100] * 50]}

            start = time.perf_counter()
            result = loss_fn(mock_outputs, mock_batch)
            elapsed = time.perf_counter() - start

            loss_metrics.loss_computation_times.append(elapsed)
            loss_metrics.loss_values.append(result["loss"])
            loss_metrics.steps_completed += 1

        results["custom_loss_sft"] = loss_metrics.to_report()
        print(f"  Steps: {results['custom_loss_sft']['steps_completed']}")
        print(f"  Loss computation (p50): {results['custom_loss_sft']['loss_computation_ms']['p50']:.3f}ms")

        # ===================================================================
        # Phase 2: REINFORCE Training
        # ===================================================================
        print("\n" + "=" * 60)
        print("PHASE 2: REINFORCE Training Benchmark")
        print("=" * 60)

        reinforce_metrics = RLVRMetrics()
        sampler = MockRolloutSampler(max_new_tokens=32)
        reward_fn = resolve_reward("keyword_contains", keywords=["result"])
        algo = REINFORCE(REINFORCEConfig(use_baseline=True, entropy_coef=0.01))

        prompts = ["Solve the problem", "Find the answer", "Calculate the result"]

        class MockTC:
            step = 0
            def forward_backward(self, batch):
                self.step += 1
                return type("R", (), {"loss": 0.5})()
            def optim_step(self):
                return type("R", (), {"step": self.step})()

        tc = MockTC()
        start_time = time.perf_counter()

        for epoch in range(num_iterations):
            rollout_start = time.perf_counter()
            batch = sampler.generate_trajectories(prompts)
            reinforce_metrics.rollout_times.append(time.perf_counter() - rollout_start)

            for traj in batch:
                traj.reward = reward_fn(traj.prompt, traj.response)

            update_start = time.perf_counter()
            result = algo.update(batch, tc)
            reinforce_metrics.policy_update_times.append(time.perf_counter() - update_start)

            reinforce_metrics.reward_values.append(batch.mean_reward)
            reinforce_metrics.policy_losses.append(result.policy_loss)
            reinforce_metrics.epochs_completed += 1

        reinforce_metrics.total_training_time = time.perf_counter() - start_time
        results["reinforce"] = reinforce_metrics.to_report()

        print(f"  Epochs: {results['reinforce']['epochs_completed']}")
        print(f"  Total time: {results['reinforce']['total_training_time_seconds']:.2f}s")
        print(f"  Mean reward: {results['reinforce']['mean_reward']:.4f}")

        # ===================================================================
        # Phase 3: PPO Training
        # ===================================================================
        print("\n" + "=" * 60)
        print("PHASE 3: PPO Training Benchmark")
        print("=" * 60)

        ppo_metrics = RLVRMetrics()
        ppo_algo = PPO(PPOConfig(clip_range=0.2, ppo_epochs=4, entropy_coef=0.01))

        tc2 = MockTC()
        start_time = time.perf_counter()

        for epoch in range(num_iterations):
            rollout_start = time.perf_counter()
            batch = sampler.generate_trajectories(prompts)
            ppo_metrics.rollout_times.append(time.perf_counter() - rollout_start)

            for traj in batch:
                traj.reward = reward_fn(traj.prompt, traj.response)

            update_start = time.perf_counter()
            result = ppo_algo.update(batch, tc2)
            ppo_metrics.policy_update_times.append(time.perf_counter() - update_start)

            ppo_metrics.reward_values.append(batch.mean_reward)
            ppo_metrics.policy_losses.append(result.policy_loss)
            ppo_metrics.epochs_completed += 1

        ppo_metrics.total_training_time = time.perf_counter() - start_time
        results["ppo"] = ppo_metrics.to_report()

        print(f"  Epochs: {results['ppo']['epochs_completed']}")
        print(f"  Total time: {results['ppo']['total_training_time_seconds']:.2f}s")
        print(f"  Mean reward: {results['ppo']['mean_reward']:.4f}")

        # ===================================================================
        # Generate Report
        # ===================================================================
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Save report
        reports_dir = Path(__file__).parent.parent.parent / "reports" / "e2e"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"custom_loss_rlvr_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n  Report saved to: {report_path}")

        print("\n  Custom Loss SFT:")
        print(f"    Loss computation (p50): {results['custom_loss_sft']['loss_computation_ms']['p50']:.3f}ms")

        print("\n  REINFORCE:")
        print(f"    Policy update (p50): {results['reinforce']['policy_update_ms']['p50']:.3f}ms")
        print(f"    Mean reward: {results['reinforce']['mean_reward']:.4f}")

        print("\n  PPO:")
        print(f"    Policy update (p50): {results['ppo']['policy_update_ms']['p50']:.3f}ms")
        print(f"    Mean reward: {results['ppo']['mean_reward']:.4f}")

        # Assertions
        assert results['custom_loss_sft']['steps_completed'] == num_iterations
        assert results['reinforce']['epochs_completed'] == num_iterations
        assert results['ppo']['epochs_completed'] == num_iterations

        return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
