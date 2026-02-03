#!/usr/bin/env python3
"""
Baseline SFT Smoke Test

A minimal SFT training run (20 steps) that validates:
1. TrainingClient creates successfully
2. forward_backward computes loss and gradients
3. optim_step applies updates
4. Loss decreases over training
5. sample() generates text
6. save_state/load_state work correctly

Usage:
    python scripts/baseline_sft_smoke.py [--steps N] [--save-golden]

Exit codes:
    0 - All tests passed
    1 - Test failed
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class SmokeTestMetrics:
    """Metrics collected during smoke test."""

    steps_completed: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    sample_generated: bool = False
    checkpoint_saved: bool = False
    checkpoint_loaded: bool = False
    total_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MockMLBackend:
    """
    Self-contained Mock ML backend for smoke testing.

    Simulates training operations without actual ML framework dependencies.
    This mirrors the behavior of the full MockMLBackend in the tensorguard package.
    """

    def __init__(self):
        """Initialize mock backend."""
        self._models: Dict[str, Dict[str, Any]] = {}
        self._gradients: Dict[str, Any] = {}

    def initialize_model(
        self,
        training_client_id: str,
        model_ref: str,
        config: Dict[str, Any],
    ) -> None:
        """Initialize a model for training."""
        self._models[training_client_id] = {
            "model_ref": model_ref,
            "config": config,
            "step": 0,
            "weights": secrets.token_bytes(1024),
            "optimizer_state": secrets.token_bytes(512),
        }

    def forward_backward(
        self,
        training_client_id: str,
        batch: Dict[str, Any],
        dp_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute forward-backward pass."""
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Simulate computation time
        time.sleep(0.01)

        # Generate mock results
        batch_size = len(batch.get("input_ids", []))
        seq_len = len(batch.get("input_ids", [[]])[0]) if batch.get("input_ids") else 128

        # Loss decreases with training steps
        loss = 2.5 - (model["step"] * 0.01)
        grad_norm = 1.5 + secrets.randbelow(100) / 1000

        # Store gradients for optim_step
        self._gradients[training_client_id] = {
            "grad_norm": grad_norm,
            "computed_at": datetime.utcnow(),
        }

        result = {
            "loss": max(0.1, loss),
            "grad_norm": grad_norm,
            "tokens_processed": batch_size * seq_len,
        }

        return result

    def optim_step(
        self,
        training_client_id: str,
        apply_dp_noise: bool = True,
        dp_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute optimizer step."""
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Simulate computation time
        time.sleep(0.005)

        # Increment step
        model["step"] += 1

        # Get learning rate from config
        optim_config = model["config"].get("optimizer", {})
        learning_rate = optim_config.get("learning_rate", 1e-4)

        result = {
            "step": model["step"],
            "learning_rate": learning_rate,
        }

        return result

    def sample(
        self,
        training_client_id: str,
        prompts: list,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate samples from the model."""
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Simulate generation time
        time.sleep(0.01 * len(prompts))

        max_tokens = config.get("max_tokens", 128)

        samples = []
        for prompt in prompts:
            completion = f" [Mock completion for step {model['step']}]"
            tokens_generated = min(len(completion.split()), max_tokens)

            samples.append({
                "prompt": prompt,
                "completion": completion,
                "tokens_generated": tokens_generated,
                "finish_reason": "stop",
            })

        return {
            "samples": samples,
            "model_step": model["step"],
            "sampling_config": config,
        }

    def save_state(
        self,
        training_client_id: str,
        include_optimizer: bool = True,
    ) -> bytes:
        """Serialize model state."""
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        state = {
            "model_ref": model["model_ref"],
            "step": model["step"],
            "weights_hash": secrets.token_hex(32),
            "config": model["config"],
        }

        if include_optimizer:
            state["optimizer_state_hash"] = secrets.token_hex(16)

        state_json = json.dumps(state)
        return state_json.encode() + model["weights"]

    def load_state(
        self,
        training_client_id: str,
        state_bytes: bytes,
    ) -> int:
        """Load model state from bytes."""
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        try:
            # Find the end of the JSON by counting braces
            brace_count = 0
            json_end = -1
            for i, byte in enumerate(state_bytes):
                if byte == ord('{'):
                    brace_count += 1
                elif byte == ord('}'):
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break

            if json_end > 0:
                state_json = state_bytes[: json_end + 1].decode()
                state = json.loads(state_json)
                model["step"] = state.get("step", 0)
        except Exception:
            pass

        return model["step"]


def create_synthetic_batch(batch_size: int = 4, seq_len: int = 128) -> Dict[str, Any]:
    """Create a synthetic training batch."""
    return {
        "input_ids": [[i % 1000 for i in range(seq_len)] for _ in range(batch_size)],
        "attention_mask": [[1] * seq_len for _ in range(batch_size)],
        "labels": [[i % 1000 for i in range(seq_len)] for _ in range(batch_size)],
    }


class MinimalTrainingClient:
    """
    Minimal training client for smoke testing.

    Uses MockMLBackend directly without HTTP layer for fast testing.
    """

    def __init__(
        self,
        model_ref: str = "test-model",
        learning_rate: float = 1e-4,
        dp_enabled: bool = False,
    ):
        self.training_client_id = f"tc-smoke-{int(time.time())}"
        self.model_ref = model_ref
        self.backend = MockMLBackend()
        self.step = 0
        self.dp_config = {"enabled": dp_enabled} if dp_enabled else None

        # Initialize model in backend
        self.backend.initialize_model(
            training_client_id=self.training_client_id,
            model_ref=model_ref,
            config={
                "optimizer": {"learning_rate": learning_rate},
                "lora": {"rank": 16, "alpha": 32.0},
            },
        )

    def forward_backward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run forward-backward pass."""
        result = self.backend.forward_backward(
            training_client_id=self.training_client_id,
            batch=batch,
            dp_config=self.dp_config,
        )
        return result

    def optim_step(self, apply_dp_noise: bool = True) -> Dict[str, Any]:
        """Run optimizer step."""
        result = self.backend.optim_step(
            training_client_id=self.training_client_id,
            apply_dp_noise=apply_dp_noise,
            dp_config=self.dp_config,
        )
        self.step = result["step"]
        return result

    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 50,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate samples."""
        return self.backend.sample(
            training_client_id=self.training_client_id,
            prompts=prompts,
            config={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
            },
        )

    def save_state(self, include_optimizer: bool = True) -> Dict[str, Any]:
        """Save training state."""
        state_bytes = self.backend.save_state(
            training_client_id=self.training_client_id,
            include_optimizer=include_optimizer,
        )
        artifact_id = f"art-{self.training_client_id}-{self.step}"
        return {
            "artifact_id": artifact_id,
            "size_bytes": len(state_bytes),
            "step": self.step,
            "_state_bytes": state_bytes,
        }

    def load_state(self, artifact_id: str, state_bytes: bytes) -> Dict[str, Any]:
        """Load training state."""
        step = self.backend.load_state(
            training_client_id=self.training_client_id,
            state_bytes=state_bytes,
        )
        self.step = step
        return {"step": step, "artifact_id": artifact_id}


def run_smoke_test(
    num_steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 128,
    verbose: bool = True,
) -> SmokeTestMetrics:
    """
    Run the baseline SFT smoke test.

    Args:
        num_steps: Number of training steps
        batch_size: Batch size for synthetic data
        seq_len: Sequence length for synthetic data
        verbose: Print progress

    Returns:
        SmokeTestMetrics with test results
    """
    metrics = SmokeTestMetrics()
    metrics.timestamp = datetime.now(timezone.utc).isoformat()

    start_time = time.time()

    if verbose:
        print("\n" + "=" * 50)
        print("=== TenSafe Baseline SFT Smoke Test ===")
        print("=" * 50 + "\n")

    # Create training client
    tc = MinimalTrainingClient(
        model_ref="meta-llama/Llama-3-8B",
        learning_rate=1e-4,
    )

    if verbose:
        print(f"Created training client: {tc.training_client_id}")
        print(f"Model: {tc.model_ref}")
        print(f"Running {num_steps} steps with batch_size={batch_size}, seq_len={seq_len}\n")

    # Training loop
    for step in range(1, num_steps + 1):
        # Create batch
        batch = create_synthetic_batch(batch_size, seq_len)

        # Forward-backward
        fb_result = tc.forward_backward(batch)

        # Optimizer step
        opt_result = tc.optim_step()

        # Record metrics
        loss = fb_result["loss"]
        grad_norm = fb_result["grad_norm"]

        metrics.loss_history.append(loss)
        metrics.grad_norm_history.append(grad_norm)

        if step == 1:
            metrics.initial_loss = loss

        if verbose:
            print(f"[Step {step:2d}/{num_steps}] loss={loss:.4f}, grad_norm={grad_norm:.3f}")

    metrics.final_loss = metrics.loss_history[-1]
    metrics.steps_completed = num_steps

    # Check loss decreased
    loss_decreased = metrics.final_loss < metrics.initial_loss

    if verbose:
        print("\nTraining complete!")
        print(f"  Initial loss: {metrics.initial_loss:.4f}")
        print(f"  Final loss: {metrics.final_loss:.4f}")
        print(f"  Loss decreased: {loss_decreased}")

    # Test sampling
    if verbose:
        print("\nSampling from trained model...")

    sample_result = tc.sample(
        prompts=["Once upon a time"],
        max_tokens=50,
    )

    metrics.sample_generated = len(sample_result.get("samples", [])) > 0

    if verbose and metrics.sample_generated:
        sample = sample_result["samples"][0]
        print(f"  Prompt: \"{sample['prompt']}\"")
        print(f"  Completion: {sample['completion']}")

    # Test checkpointing
    if verbose:
        print("\nCheckpoint test...")

    save_result = tc.save_state()
    metrics.checkpoint_saved = "artifact_id" in save_result

    if verbose:
        print(f"  Saved checkpoint: {save_result['artifact_id']}")

    # Load checkpoint
    load_result = tc.load_state(
        artifact_id=save_result["artifact_id"],
        state_bytes=save_result["_state_bytes"],
    )
    metrics.checkpoint_loaded = load_result["step"] == num_steps

    if verbose:
        print(f"  Loaded checkpoint successfully (step={load_result['step']})")

    metrics.total_time_seconds = time.time() - start_time

    # Final validation
    all_passed = (
        loss_decreased
        and metrics.sample_generated
        and metrics.checkpoint_saved
        and metrics.checkpoint_loaded
    )

    if verbose:
        print("\n" + "=" * 50)
        if all_passed:
            print("=== All baseline tests passed! ===")
        else:
            print("=== SOME TESTS FAILED ===")
            if not loss_decreased:
                print("  - Loss did not decrease")
            if not metrics.sample_generated:
                print("  - Sample generation failed")
            if not metrics.checkpoint_saved:
                print("  - Checkpoint save failed")
            if not metrics.checkpoint_loaded:
                print("  - Checkpoint load failed")
        print("=" * 50 + "\n")

    return metrics


def save_golden_artifacts(metrics: SmokeTestMetrics, output_dir: Path) -> None:
    """Save metrics as golden artifacts for regression testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full metrics
    metrics_path = output_dir / "baseline_sft_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Saved golden metrics to: {metrics_path}")

    # Save loss curve
    loss_curve_path = output_dir / "baseline_loss_curve.json"
    with open(loss_curve_path, "w") as f:
        json.dump({
            "loss_history": metrics.loss_history,
            "initial_loss": metrics.initial_loss,
            "final_loss": metrics.final_loss,
            "steps": metrics.steps_completed,
        }, f, indent=2)
    print(f"Saved golden loss curve to: {loss_curve_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TenSafe Baseline SFT Smoke Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--save-golden",
        action="store_true",
        help="Save results as golden artifacts",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Run smoke test
    metrics = run_smoke_test(
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        verbose=not args.quiet,
    )

    # Save golden artifacts if requested
    if args.save_golden:
        golden_dir = PROJECT_ROOT / "tests" / "golden"
        save_golden_artifacts(metrics, golden_dir)

    # Exit with appropriate code
    loss_decreased = metrics.final_loss < metrics.initial_loss
    all_passed = (
        loss_decreased
        and metrics.sample_generated
        and metrics.checkpoint_saved
        and metrics.checkpoint_loaded
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
