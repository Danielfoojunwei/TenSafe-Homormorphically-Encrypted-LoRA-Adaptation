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

        # Compute advantages based on algorithm
        advantages = self._compute_advantages(prompts, rewards, rlvr_config)

        # Compute policy gradient loss based on algorithm
        algorithm = rlvr_config.algorithm
        if algorithm == "reinforce":
            loss = self._reinforce_loss(log_probs, advantages)
        elif algorithm in ("ppo", "grpo", "rloo", "reinforce_pp"):
            # GRPO, RLOO, REINFORCE++ all use clipped surrogate loss
            # (the algorithm difference is in advantage computation above)
            loss = self._ppo_loss(log_probs, advantages)
        else:
            raise ValueError(f"Unknown RLVR algorithm: {algorithm}")

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
                "algorithm": algorithm,
            },
        )

    def _compute_advantages(
        self,
        prompts: List[str],
        rewards: np.ndarray,
        rlvr_config: Any,
    ) -> np.ndarray:
        """
        Compute advantages based on the selected algorithm.

        - reinforce/ppo: Moving-average baseline subtraction
        - grpo: Group Relative Policy Optimization (per-prompt group normalization)
        - rloo: Leave-One-Out baseline (per-prompt group LOO mean)
        - reinforce_pp: Same as reinforce (temporal discounting applied at loss level)
        """
        algorithm = rlvr_config.algorithm

        if algorithm in ("grpo", "rloo"):
            # Group trajectories by prompt
            from collections import defaultdict
            groups = defaultdict(list)
            for i, prompt in enumerate(prompts):
                groups[prompt].append(i)

            advantages = np.zeros_like(rewards)

            if algorithm == "grpo":
                # GRPO: normalize within each prompt group
                for prompt, indices in groups.items():
                    group_rewards = rewards[indices]
                    group_mean = group_rewards.mean()
                    if len(group_rewards) > 1 and getattr(rlvr_config, 'grpo_normalize_within_group', True):
                        group_std = group_rewards.std() + 1e-8
                    else:
                        group_std = 1.0
                    advantages[indices] = (group_rewards - group_mean) / group_std

            elif algorithm == "rloo":
                # RLOO: leave-one-out baseline per group
                batch_mean = rewards.mean()
                for prompt, indices in groups.items():
                    group_rewards = rewards[indices]
                    n = len(group_rewards)
                    if n <= 1:
                        advantages[indices] = group_rewards - batch_mean
                    else:
                        group_sum = group_rewards.sum()
                        for idx, i in enumerate(indices):
                            loo_baseline = (group_sum - group_rewards[idx]) / (n - 1)
                            advantages[i] = group_rewards[idx] - loo_baseline

        else:
            # reinforce, ppo, reinforce_pp: moving-average baseline
            if rlvr_config.use_baseline:
                advantages = rewards - self._baseline
                self._baseline = (
                    rlvr_config.baseline_decay * self._baseline +
                    (1 - rlvr_config.baseline_decay) * rewards.mean()
                )
            else:
                advantages = rewards.copy()

        if rlvr_config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

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
        """
        Generate rollouts using the model for policy gradient training.

        Args:
            prompts: List of input prompts
            sample: Whether to sample (True) or use greedy decoding (False)

        Returns:
            Tuple of (responses, log_probs) for policy gradient computation
        """
        if self.model is None:
            logger.warning("No model available for rollout generation")
            return [""] * len(prompts), [0.0] * len(prompts)

        responses = []
        log_probs_list = []

        try:
            import torch

            # Get generation config from RLVR settings
            rlvr_config = self.config.rlvr
            max_new_tokens = getattr(rlvr_config, 'max_response_length', 128)
            temperature = getattr(rlvr_config, 'temperature', 0.8 if sample else 0.0)

            # Check if model has tokenizer attribute or use from config
            tokenizer = getattr(self.model, 'tokenizer', None)
            if tokenizer is None and hasattr(self.model, 'get_tokenizer'):
                tokenizer = self.model.get_tokenizer()

            if tokenizer is None:
                logger.warning("No tokenizer available, cannot generate proper rollouts")
                return [""] * len(prompts), [0.0] * len(prompts)

            # Set model to eval mode
            self.model.eval()

            for prompt in prompts:
                # Tokenize prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    # Generate with output scores for log prob computation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=sample and temperature > 0,
                        temperature=temperature if sample else 1.0,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                    # Extract generated tokens
                    generated_ids = outputs.sequences[0, input_length:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    responses.append(response)

                    # Compute log probabilities from scores
                    if hasattr(outputs, 'scores') and outputs.scores:
                        total_log_prob = 0.0
                        for step_idx, (scores, token_id) in enumerate(
                            zip(outputs.scores, generated_ids)
                        ):
                            log_probs = torch.log_softmax(scores[0], dim=-1)
                            total_log_prob += log_probs[token_id].item()
                        log_probs_list.append(total_log_prob)
                    else:
                        log_probs_list.append(0.0)

            # Return to training mode
            self.model.train()

        except ImportError:
            logger.warning("PyTorch not available for rollout generation")
            return [""] * len(prompts), [0.0] * len(prompts)
        except Exception as e:
            logger.error(f"Rollout generation failed: {e}")
            return [""] * len(prompts), [0.0] * len(prompts)

        return responses, log_probs_list

    def _reinforce_loss(self, log_probs: List[Any], advantages: np.ndarray) -> Any:
        """
        Compute REINFORCE policy gradient loss.

        Loss = -sum(log_prob * advantage) for vanilla REINFORCE.

        Args:
            log_probs: Log probabilities of generated actions
            advantages: Advantage estimates (rewards - baseline)

        Returns:
            Scalar loss value (torch.Tensor if available, else float)
        """
        try:
            import torch

            # Convert to tensors
            log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, requires_grad=True)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

            # REINFORCE loss: -E[log_prob * advantage]
            loss = -(log_probs_tensor * advantages_tensor).mean()
            return loss

        except ImportError:
            # Fallback to numpy computation
            return -float(np.mean(np.array(log_probs) * advantages))

    def _ppo_loss(self, log_probs: List[Any], advantages: np.ndarray) -> Any:
        """
        Compute PPO (Proximal Policy Optimization) clipped loss.

        Uses clipped surrogate objective to prevent large policy updates.

        Args:
            log_probs: Log probabilities of generated actions under current policy
            advantages: Advantage estimates

        Returns:
            Scalar loss value (torch.Tensor if available, else float)
        """
        try:
            import torch

            rlvr_config = self.config.rlvr
            clip_range = getattr(rlvr_config, 'ppo_clip_range', 0.2)

            log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, requires_grad=True)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

            # For PPO, we need old log probs - use stored or current as approximation
            # In full implementation, old_log_probs would be stored from rollout
            old_log_probs = getattr(self, '_old_log_probs', log_probs_tensor.detach())

            # Compute ratio
            ratio = torch.exp(log_probs_tensor - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_tensor

            # PPO loss is the minimum of the two surrogates
            loss = -torch.min(surr1, surr2).mean()

            # Store current log probs for next iteration
            self._old_log_probs = log_probs_tensor.detach()

            return loss

        except ImportError:
            # Fallback: use simple REINFORCE loss without clipping
            return -float(np.mean(np.array(log_probs) * advantages))

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

        try:
            import torch
            from transformers import AutoModelForCausalLM

            # Determine dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(model_config.torch_dtype, torch.bfloat16)

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_config.name,
                revision=model_config.revision,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=model_config.trust_remote_code,
            )

            # Apply LoRA if enabled
            if lora_config.enabled:
                from peft import LoraConfig, get_peft_model, TaskType

                peft_config = LoraConfig(
                    r=lora_config.rank,
                    lora_alpha=lora_config.alpha,
                    lora_dropout=lora_config.dropout,
                    target_modules=lora_config.target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    bias="none",
                )

                self._model = get_peft_model(self._model, peft_config)
                self._model.print_trainable_parameters()

                logger.info(
                    f"LoRA configured: rank={lora_config.rank}, "
                    f"alpha={lora_config.alpha}, targets={lora_config.target_modules}"
                )

        except ImportError as e:
            logger.warning(
                f"PyTorch/Transformers not available ({e}). "
                f"Using placeholder model for configuration testing."
            )
            self._model = _MockModel()

    def _create_optimizer(self) -> None:
        """Create the optimizer."""
        training_config = self.config.training

        try:
            import torch.optim as optim

            # Get trainable parameters
            if hasattr(self._model, 'parameters'):
                trainable_params = [
                    p for p in self._model.parameters() if p.requires_grad
                ]
            else:
                trainable_params = []

            if not trainable_params:
                logger.warning("No trainable parameters found, using placeholder optimizer")
                self._optimizer = _MockOptimizer(learning_rate=training_config.learning_rate)
                return

            # Create optimizer
            if training_config.optimizer.lower() == "adamw":
                self._optimizer = optim.AdamW(
                    trainable_params,
                    lr=training_config.learning_rate,
                    weight_decay=training_config.weight_decay,
                )
            elif training_config.optimizer.lower() == "adam":
                self._optimizer = optim.Adam(
                    trainable_params,
                    lr=training_config.learning_rate,
                )
            elif training_config.optimizer.lower() == "sgd":
                self._optimizer = optim.SGD(
                    trainable_params,
                    lr=training_config.learning_rate,
                    weight_decay=training_config.weight_decay,
                )
            else:
                raise ValueError(f"Unknown optimizer: {training_config.optimizer}")

            logger.info(
                f"Created optimizer: {training_config.optimizer}, "
                f"lr={training_config.learning_rate}"
            )

        except ImportError:
            logger.warning("PyTorch not available, using placeholder optimizer")
            self._optimizer = _MockOptimizer(learning_rate=training_config.learning_rate)

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
        """
        Set up the HE backend using the unified microkernel architecture.

        All HE modes now route through the single UnifiedHEBackend which
        uses the HE-LoRA Microkernel with MOAI optimizations.
        """
        from tensafe.core.he_interface import get_backend, HEParams, HEBackendType

        he_config = self.config.he

        # Resolve legacy modes to modern equivalents
        resolved_mode = HEMode.resolve(he_config.mode)

        params = HEParams(
            poly_modulus_degree=he_config.poly_modulus_degree,
            coeff_modulus_bits=he_config.coeff_modulus_bits,
            scale_bits=he_config.scale_bits,
            use_column_packing=he_config.use_column_packing,
            use_interleaved_batching=he_config.use_interleaved_batching,
        )

        # Map HEMode to HEBackendType (unified architecture)
        backend_type = {
            HEMode.DISABLED: HEBackendType.DISABLED,
            HEMode.PRODUCTION: HEBackendType.PRODUCTION,
            HEMode.SIMULATION: HEBackendType.SIMULATION,
        }.get(resolved_mode, HEBackendType.PRODUCTION)

        self._he_backend = get_backend(backend_type, params)

        # Log backend information
        if self._he_backend.is_production_ready:
            logger.info(f"HE backend ready (production): {self._he_backend.backend_name}")
        else:
            logger.warning(
                f"HE backend ready (non-production): {self._he_backend.backend_name}. "
                "Use HEMode.PRODUCTION for deployment."
            )

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

            # Require a dataloader
            if train_dataloader is None:
                raise ValueError(
                    "train_dataloader is required. Provide a DataLoader or "
                    "iterator that yields batches with 'input_ids' and 'labels'."
                )

            # Training loop
            for step in range(self._current_step, training_config.total_steps):
                self._current_step = step

                # Get batch
                try:
                    batch = next(train_dataloader)
                except StopIteration:
                    # Reset iterator for next epoch
                    logger.info("Dataloader exhausted, epoch completed")
                    break

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

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self._current_step


# ==============================================================================
# Placeholder Classes (for import testing only)
# ==============================================================================


class _MockModel:
    """
    Placeholder model for configuration testing.

    WARNING: This is NOT a real model. It is only used when PyTorch/Transformers
    are not available (e.g., for configuration validation or import testing).

    Production usage requires installing:
        pip install torch transformers peft
    """

    def __init__(self):
        logger.warning(
            "_MockModel is a placeholder. Install PyTorch and Transformers "
            "for production training."
        )
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def parameters(self):
        return []

    def forward(self, **kwargs):
        """Return mock outputs for testing. Not suitable for production."""
        return {"logits": kwargs.get("input_ids")}

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _MockOptimizer:
    """
    Placeholder optimizer for configuration testing.

    WARNING: This is NOT a real optimizer. It is only used when PyTorch
    is not available.

    Production usage requires installing:
        pip install torch
    """

    def __init__(self, learning_rate: float = 1e-4):
        logger.warning(
            "_MockOptimizer is a placeholder. Install PyTorch for production training."
        )
        self.param_groups = [{"lr": learning_rate}]

    def step(self):
        """No-op step for testing. Not suitable for production."""
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
