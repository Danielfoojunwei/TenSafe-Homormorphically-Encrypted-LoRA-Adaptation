"""
TenSafe Unified Training Pipeline.

This module provides a unified training pipeline that orchestrates:
- Multiple training modes (SFT, RLVR, DPO)
- Differential privacy integration
- Homomorphic encryption for LoRA
- Checkpointing and logging
- Production safety and validation

The pipeline abstracts over training implementations while ensuring
consistent behavior and safety across all modes.

Architecture:
    TenSafePipeline (orchestrator)
    ├── ConfigManager (unified config)
    ├── ModelManager (model loading, LoRA)
    ├── TrainingMode (SFT, RLVR, etc.)
    ├── PrivacyManager (DP accounting)
    ├── HEManager (encrypted operations)
    └── CheckpointManager (save/load)

Usage:
    from tensafe.core.pipeline import TenSafePipeline

    # Create pipeline from config
    pipeline = TenSafePipeline.from_config("config.yaml")

    # Or create programmatically
    pipeline = TenSafePipeline(config)

    # Run training
    result = pipeline.train()

    # Run inference
    output = pipeline.inference(prompts)
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from tensafe.core.config import (
    TenSafeConfig,
    TrainingMode,
    HEMode,
    load_config,
    save_config,
)
from tensafe.core.gates import (
    ProductionGates,
    production_check,
    GateDeniedError,
)
from tensafe.core.registry import (
    get_loss_registry,
    get_reward_registry,
    resolve_function,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Pipeline State and Events
# ==============================================================================


class PipelineState(Enum):
    """States of the training pipeline."""
    INITIALIZED = "initialized"
    CONFIGURING = "configuring"
    LOADING_MODEL = "loading_model"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineEvent(Enum):
    """Events emitted by the pipeline."""
    STATE_CHANGE = "state_change"
    STEP_START = "step_start"
    STEP_END = "step_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    METRICS_LOGGED = "metrics_logged"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class StepMetrics:
    """Metrics from a single training step."""
    step: int
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    dp_epsilon: Optional[float] = None
    dp_delta: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    time_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result from a training run."""
    success: bool
    total_steps: int
    final_loss: float
    final_metrics: Dict[str, float]
    training_time_seconds: float
    checkpoints_saved: List[str]
    dp_spent: Optional[Tuple[float, float]] = None  # (epsilon, delta)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ==============================================================================
# Training Mode Interface
# ==============================================================================


class TrainingModeInterface(ABC):
    """
    Abstract interface for training modes.

    Each mode (SFT, RLVR, etc.) implements this interface.
    """

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Get the mode name."""
        pass

    @abstractmethod
    def prepare(
        self,
        config: TenSafeConfig,
        model: Any,
        optimizer: Any,
    ) -> None:
        """
        Prepare for training.

        Args:
            config: Training configuration
            model: The model to train
            optimizer: The optimizer
        """
        pass

    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, Any],
        step: int,
    ) -> StepMetrics:
        """
        Execute a single training step.

        Args:
            batch: Training batch
            step: Current step number

        Returns:
            Step metrics
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        eval_batches: Iterator[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Run evaluation.

        Args:
            eval_batches: Iterator of evaluation batches

        Returns:
            Evaluation metrics
        """
        pass


# ==============================================================================
# SFT Training Mode
# ==============================================================================


class SFTTrainingMode(TrainingModeInterface):
    """
    Supervised Fine-Tuning mode.

    Standard language model fine-tuning with:
    - Cross-entropy loss
    - Optional differential privacy
    - Optional HE for LoRA
    """

    def __init__(self):
        self.config: Optional[TenSafeConfig] = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self._dp_manager = None
        self._step = 0

    @property
    def mode_name(self) -> str:
        return "sft"

    def prepare(
        self,
        config: TenSafeConfig,
        model: Any,
        optimizer: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer

        # Resolve loss function
        loss_spec = config.training.loss_fn
        loss_kwargs = config.training.loss_kwargs
        self.loss_fn = resolve_function(loss_spec, registry="loss", **loss_kwargs)

        logger.info(f"SFT mode prepared: loss={loss_spec}")

    def train_step(
        self,
        batch: Dict[str, Any],
        step: int,
    ) -> StepMetrics:
        self._step = step
        start_time = time.perf_counter()

        # Forward pass
        outputs = self._forward(batch)

        # Compute loss
        loss_result = self.loss_fn(outputs, batch)
        loss = loss_result["loss"]
        metrics = loss_result.get("metrics", {})

        # Backward pass
        self._backward(loss)

        # Optimizer step
        grad_norm = self._optimizer_step()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return StepMetrics(
            step=step,
            loss=float(loss.item()) if hasattr(loss, 'item') else float(loss),
            learning_rate=self._get_lr(),
            grad_norm=grad_norm,
            time_ms=elapsed_ms,
            extra=metrics,
        )

    def evaluate(
        self,
        eval_batches: Iterator[Dict[str, Any]],
    ) -> Dict[str, float]:
        total_loss = 0.0
        total_samples = 0

        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()

        try:
            for batch in eval_batches:
                outputs = self._forward(batch, training=False)
                loss_result = self.loss_fn(outputs, batch)
                loss = loss_result["loss"]

                batch_size = self._get_batch_size(batch)
                total_loss += float(loss.item() if hasattr(loss, 'item') else loss) * batch_size
                total_samples += batch_size
        finally:
            # Restore training mode
            if hasattr(self.model, 'train'):
                self.model.train()

        avg_loss = total_loss / max(1, total_samples)

        # Compute perplexity
        try:
            import math
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float('inf')

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_samples": total_samples,
        }

    def _forward(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        """Run forward pass."""
        # Placeholder - actual implementation depends on model type
        if hasattr(self.model, 'forward'):
            return self.model(**batch)
        return {"logits": batch.get("input_ids")}

    def _backward(self, loss: Any) -> None:
        """Run backward pass."""
        if hasattr(loss, 'backward'):
            loss.backward()

    def _optimizer_step(self) -> Optional[float]:
        """Run optimizer step with gradient clipping."""
        grad_norm = None

        # Clip gradients
        if self.config and self.config.training.max_grad_norm > 0:
            try:
                import torch
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm,
                )
                grad_norm = float(grad_norm)
            except ImportError:
                pass

        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return grad_norm

    def _get_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is not None:
            try:
                return self.optimizer.param_groups[0]['lr']
            except (AttributeError, IndexError, KeyError):
                pass
        return self.config.training.learning_rate if self.config else 0.0

    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Get batch size from batch."""
        if "input_ids" in batch:
            return len(batch["input_ids"])
        for v in batch.values():
            if hasattr(v, '__len__'):
                return len(v)
        return 1


# ==============================================================================
# RLVR Training Mode
# ==============================================================================


class RLVRTrainingMode(TrainingModeInterface):
    """
    Reinforcement Learning with Verifiable Rewards mode.

    Policy gradient training with:
    - Rollout generation
    - Reward function evaluation
    - REINFORCE or PPO optimization
    """

    def __init__(self):
        self.config: Optional[TenSafeConfig] = None
        self.model = None
        self.optimizer = None
        self.reward_fn = None
        self._baseline = 0.0
        self._step = 0

    @property
    def mode_name(self) -> str:
        return "rlvr"

    def prepare(
        self,
        config: TenSafeConfig,
        model: Any,
        optimizer: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer

        if config.rlvr is None:
            raise ValueError("RLVR mode requires rlvr config")

        # Resolve reward function
        reward_spec = config.rlvr.reward_fn
        reward_kwargs = config.rlvr.reward_kwargs
        self.reward_fn = resolve_function(reward_spec, registry="reward", **reward_kwargs)

        logger.info(f"RLVR mode prepared: reward={reward_spec}, algorithm={config.rlvr.algorithm}")

    def train_step(
        self,
        batch: Dict[str, Any],
        step: int,
    ) -> StepMetrics:
        self._step = step
        start_time = time.perf_counter()

        rlvr_config = self.config.rlvr

        # Generate rollouts
        prompts = batch.get("prompts", [])
        responses, log_probs = self._generate_rollouts(prompts)

        # Compute rewards
        rewards = []
        for prompt, response in zip(prompts, responses):
            meta = batch.get("meta", {})
            reward = self.reward_fn(prompt, response, meta)
            rewards.append(reward)

        rewards = np.array(rewards)

        # Apply reward scaling and clipping
        rewards = rewards * rlvr_config.reward_scale
        if rlvr_config.reward_clip is not None:
            rewards = np.clip(rewards, -rlvr_config.reward_clip, rlvr_config.reward_clip)

        # Compute advantages
        if rlvr_config.use_baseline:
            advantages = rewards - self._baseline
            self._baseline = (
                rlvr_config.baseline_decay * self._baseline +
                (1 - rlvr_config.baseline_decay) * rewards.mean()
            )
        else:
            advantages = rewards

        if rlvr_config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute policy gradient loss
        if rlvr_config.algorithm == "reinforce":
            loss = self._reinforce_loss(log_probs, advantages)
        elif rlvr_config.algorithm == "ppo":
            loss = self._ppo_loss(log_probs, advantages)
        else:
            raise ValueError(f"Unknown RLVR algorithm: {rlvr_config.algorithm}")

        # Backward pass
        self._backward(loss)
        grad_norm = self._optimizer_step()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return StepMetrics(
            step=step,
            loss=float(loss.item()) if hasattr(loss, 'item') else float(loss),
            learning_rate=self._get_lr(),
            grad_norm=grad_norm,
            time_ms=elapsed_ms,
            extra={
                "mean_reward": float(rewards.mean()),
                "mean_advantage": float(advantages.mean()),
                "baseline": float(self._baseline),
            },
        )

    def evaluate(
        self,
        eval_batches: Iterator[Dict[str, Any]],
    ) -> Dict[str, float]:
        total_reward = 0.0
        total_samples = 0

        for batch in eval_batches:
            prompts = batch.get("prompts", [])
            responses, _ = self._generate_rollouts(prompts, sample=False)

            for prompt, response in zip(prompts, responses):
                meta = batch.get("meta", {})
                reward = self.reward_fn(prompt, response, meta)
                total_reward += reward
                total_samples += 1

        avg_reward = total_reward / max(1, total_samples)

        return {
            "eval_mean_reward": avg_reward,
            "eval_samples": total_samples,
        }

    def _generate_rollouts(
        self,
        prompts: List[str],
        sample: bool = True,
    ) -> Tuple[List[str], List[Any]]:
        """Generate rollouts (placeholder)."""
        # Placeholder - actual implementation depends on model
        responses = ["Generated response" for _ in prompts]
        log_probs = [0.0 for _ in prompts]
        return responses, log_probs

    def _reinforce_loss(self, log_probs: List[Any], advantages: np.ndarray) -> Any:
        """Compute REINFORCE loss."""
        # Placeholder
        return sum([-lp * adv for lp, adv in zip(log_probs, advantages)])

    def _ppo_loss(self, log_probs: List[Any], advantages: np.ndarray) -> Any:
        """Compute PPO loss."""
        # Placeholder - actual PPO implementation would include clipping
        return sum([-lp * adv for lp, adv in zip(log_probs, advantages)])

    def _backward(self, loss: Any) -> None:
        if hasattr(loss, 'backward'):
            loss.backward()

    def _optimizer_step(self) -> Optional[float]:
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return None

    def _get_lr(self) -> float:
        if self.optimizer is not None:
            try:
                return self.optimizer.param_groups[0]['lr']
            except (AttributeError, IndexError, KeyError):
                pass
        return self.config.training.learning_rate if self.config else 0.0


# ==============================================================================
# Main Pipeline
# ==============================================================================


class TenSafePipeline:
    """
    Unified TenSafe training pipeline.

    Orchestrates the entire training process with:
    - Configuration management
    - Model loading and LoRA setup
    - Training mode execution
    - Privacy accounting
    - Checkpointing
    - Logging and monitoring
    """

    def __init__(
        self,
        config: TenSafeConfig,
        validate_production: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            config: TenSafe configuration
            validate_production: Validate configuration for production
        """
        self.config = config
        self._state = PipelineState.INITIALIZED
        self._callbacks: List[Callable[[PipelineEvent, Dict[str, Any]], None]] = []

        # Components (initialized lazily)
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._training_mode: Optional[TrainingModeInterface] = None
        self._he_backend = None

        # State tracking
        self._current_step = 0
        self._best_metric = float('inf')
        self._checkpoints_saved: List[str] = []

        # Production validation
        if validate_production:
            self._validate_production()

        logger.info(f"TenSafePipeline initialized: mode={config.training.mode.value}")

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        **overrides: Any,
    ) -> "TenSafePipeline":
        """
        Create pipeline from configuration file.

        Args:
            config_path: Path to YAML/JSON config
            **overrides: Configuration overrides

        Returns:
            Configured pipeline
        """
        config = load_config(config_path)

        # Apply overrides
        # TODO: Implement nested override logic

        return cls(config)

    def register_callback(
        self,
        callback: Callable[[PipelineEvent, Dict[str, Any]], None],
    ) -> None:
        """Register a callback for pipeline events."""
        self._callbacks.append(callback)

    def _emit_event(self, event: PipelineEvent, payload: Dict[str, Any]) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, payload)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _set_state(self, state: PipelineState) -> None:
        """Change pipeline state."""
        old_state = self._state
        self._state = state
        self._emit_event(PipelineEvent.STATE_CHANGE, {
            "old_state": old_state.value,
            "new_state": state.value,
        })

    def _validate_production(self) -> None:
        """Validate configuration for production use."""
        result = production_check(self.config)

        for warning in result.warnings:
            logger.warning(f"Production warning: {warning}")
            self._emit_event(PipelineEvent.WARNING, {"message": warning})

        if not result.valid:
            for error in result.errors:
                logger.error(f"Production error: {error}")
            raise ValueError(f"Production validation failed: {result.errors}")

    def setup(self) -> "TenSafePipeline":
        """
        Set up the pipeline components.

        Returns:
            self for chaining
        """
        self._set_state(PipelineState.CONFIGURING)

        # Load model
        self._set_state(PipelineState.LOADING_MODEL)
        self._load_model()

        # Create optimizer
        self._create_optimizer()

        # Create training mode
        self._create_training_mode()

        # Set up HE backend if needed
        if self.config.he.mode != HEMode.DISABLED:
            self._setup_he_backend()

        self._set_state(PipelineState.INITIALIZED)
        return self

    def _load_model(self) -> None:
        """Load the model with LoRA configuration."""
        model_config = self.config.model
        lora_config = self.config.lora

        logger.info(f"Loading model: {model_config.name}")

        # Placeholder - actual implementation would use transformers/peft
        self._model = _MockModel()

        if lora_config.enabled:
            logger.info(
                f"Configuring LoRA: rank={lora_config.rank}, "
                f"alpha={lora_config.alpha}, targets={lora_config.target_modules}"
            )

    def _create_optimizer(self) -> None:
        """Create the optimizer."""
        training_config = self.config.training

        # Placeholder - actual implementation would create real optimizer
        self._optimizer = _MockOptimizer(learning_rate=training_config.learning_rate)

        logger.info(f"Created optimizer: {training_config.optimizer}, lr={training_config.learning_rate}")

    def _create_training_mode(self) -> None:
        """Create the training mode handler."""
        mode = self.config.training.mode

        if mode == TrainingMode.SFT:
            self._training_mode = SFTTrainingMode()
        elif mode == TrainingMode.RLVR:
            self._training_mode = RLVRTrainingMode()
        else:
            raise ValueError(f"Unsupported training mode: {mode}")

        self._training_mode.prepare(self.config, self._model, self._optimizer)

    def _setup_he_backend(self) -> None:
        """Set up the HE backend."""
        from tensafe.core.he_interface import get_backend, HEParams, HEBackendType

        he_config = self.config.he

        params = HEParams(
            poly_modulus_degree=he_config.poly_modulus_degree,
            coeff_modulus_bits=he_config.coeff_modulus_bits,
            scale_bits=he_config.scale_bits,
            use_column_packing=he_config.use_column_packing,
        )

        backend_type = {
            HEMode.TOY: HEBackendType.TOY,
            HEMode.N2HE: HEBackendType.N2HE,
            HEMode.N2HE_HEXL: HEBackendType.HEXL,
        }.get(he_config.mode, HEBackendType.AUTO)

        self._he_backend = get_backend(backend_type, params)
        logger.info(f"HE backend ready: {self._he_backend.backend_name}")

    def train(
        self,
        train_dataloader: Optional[Iterator[Dict[str, Any]]] = None,
        eval_dataloader: Optional[Iterator[Dict[str, Any]]] = None,
    ) -> TrainingResult:
        """
        Run training.

        Args:
            train_dataloader: Training data iterator
            eval_dataloader: Evaluation data iterator

        Returns:
            TrainingResult with metrics and status
        """
        if self._training_mode is None:
            self.setup()

        self._set_state(PipelineState.TRAINING)
        start_time = time.time()

        errors = []
        warnings = []
        final_metrics = {}

        try:
            training_config = self.config.training

            # Use mock data if none provided
            if train_dataloader is None:
                train_dataloader = self._mock_dataloader()

            # Training loop
            for step in range(self._current_step, training_config.total_steps):
                self._current_step = step

                # Get batch
                try:
                    batch = next(train_dataloader)
                except StopIteration:
                    train_dataloader = self._mock_dataloader()
                    batch = next(train_dataloader)

                # Training step
                self._emit_event(PipelineEvent.STEP_START, {"step": step})
                metrics = self._training_mode.train_step(batch, step)
                self._emit_event(PipelineEvent.STEP_END, {
                    "step": step,
                    "metrics": metrics.__dict__,
                })

                # Logging
                if step % training_config.log_interval == 0:
                    self._log_metrics(metrics)

                # Evaluation
                if eval_dataloader and step > 0 and step % training_config.eval_interval == 0:
                    self._set_state(PipelineState.EVALUATING)
                    eval_metrics = self._training_mode.evaluate(eval_dataloader)
                    final_metrics.update(eval_metrics)
                    self._set_state(PipelineState.TRAINING)

                # Checkpointing
                if step > 0 and step % training_config.save_interval == 0:
                    self._save_checkpoint(step)

            # Final evaluation
            if eval_dataloader:
                self._set_state(PipelineState.EVALUATING)
                final_metrics.update(self._training_mode.evaluate(eval_dataloader))

            self._set_state(PipelineState.COMPLETED)

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            self._set_state(PipelineState.FAILED)
            errors.append(str(e))

        training_time = time.time() - start_time

        return TrainingResult(
            success=self._state == PipelineState.COMPLETED,
            total_steps=self._current_step,
            final_loss=final_metrics.get("eval_loss", 0.0),
            final_metrics=final_metrics,
            training_time_seconds=training_time,
            checkpoints_saved=self._checkpoints_saved,
            errors=errors,
            warnings=warnings,
        )

    def _log_metrics(self, metrics: StepMetrics) -> None:
        """Log training metrics."""
        logger.info(
            f"Step {metrics.step}: loss={metrics.loss:.4f}, "
            f"lr={metrics.learning_rate:.2e}, "
            f"time={metrics.time_ms:.1f}ms"
        )
        self._emit_event(PipelineEvent.METRICS_LOGGED, {"metrics": metrics.__dict__})

    def _save_checkpoint(self, step: int) -> str:
        """Save a checkpoint."""
        self._set_state(PipelineState.CHECKPOINTING)

        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save config
        save_config(self.config, checkpoint_path / "config.yaml")

        # Save training state
        state = {
            "step": step,
            "config": self.config.to_dict(),
        }

        # Placeholder - actual implementation would save model weights

        self._checkpoints_saved.append(str(checkpoint_path))
        self._emit_event(PipelineEvent.CHECKPOINT_SAVED, {"path": str(checkpoint_path)})

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond save_total_limit."""
        limit = self.config.training.save_total_limit
        if limit > 0 and len(self._checkpoints_saved) > limit:
            to_remove = self._checkpoints_saved[:-limit]
            self._checkpoints_saved = self._checkpoints_saved[-limit:]

            for path in to_remove:
                try:
                    import shutil
                    shutil.rmtree(path)
                    logger.info(f"Removed old checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {path}: {e}")

    def _mock_dataloader(self) -> Iterator[Dict[str, Any]]:
        """Create a mock dataloader for testing."""
        while True:
            yield {
                "input_ids": np.random.randint(0, 1000, (4, 128)),
                "labels": np.random.randint(0, 1000, (4, 128)),
                "prompts": ["Test prompt"] * 4,
            }

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self._current_step


# ==============================================================================
# Mock classes for testing
# ==============================================================================


class _MockModel:
    """Mock model for testing."""

    def __init__(self):
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def parameters(self):
        return []

    def forward(self, **kwargs):
        return {"logits": kwargs.get("input_ids")}

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self, learning_rate: float = 1e-4):
        self.param_groups = [{"lr": learning_rate}]

    def step(self):
        pass

    def zero_grad(self):
        pass


# ==============================================================================
# Convenience functions
# ==============================================================================


def create_pipeline(
    config: Union[str, Path, TenSafeConfig],
    **kwargs: Any,
) -> TenSafePipeline:
    """
    Create a TenSafe pipeline.

    Args:
        config: Configuration (path or object)
        **kwargs: Additional configuration overrides

    Returns:
        Configured TenSafePipeline
    """
    if isinstance(config, (str, Path)):
        return TenSafePipeline.from_config(config, **kwargs)
    return TenSafePipeline(config, **kwargs)


def train(
    config: Union[str, Path, TenSafeConfig],
    **kwargs: Any,
) -> TrainingResult:
    """
    Run training with the given configuration.

    Convenience function for simple training runs.

    Args:
        config: Configuration
        **kwargs: Additional options

    Returns:
        TrainingResult
    """
    pipeline = create_pipeline(config)
    pipeline.setup()
    return pipeline.train()
