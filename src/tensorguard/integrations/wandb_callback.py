"""TenSafe Weights & Biases Integration.

Provides experiment tracking with privacy-aware logging:
- Tracks training metrics and loss curves
- Logs privacy budget consumption (epsilon, delta)
- Records HE-LoRA performance metrics
- Supports artifact logging (metadata only for privacy)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import time
import os

logger = logging.getLogger(__name__)

# Conditional W&B import
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("wandb not installed. Install with: pip install wandb")


@dataclass
class TenSafeWandbConfig:
    """Configuration for W&B integration."""
    project: str = "tensafe"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = None
    notes: Optional[str] = None

    # Privacy settings
    log_privacy_metrics: bool = True
    log_he_metrics: bool = True

    # Safety settings
    log_model_weights: bool = False  # NEVER log actual weights
    log_gradients: bool = False  # NEVER log actual gradients
    redact_sensitive_config: bool = True

    # Logging frequency
    log_frequency: int = 10  # Log every N steps

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["tensafe", "privacy-preserving"]


class TenSafeWandbCallback:
    """Weights & Biases callback with privacy tracking.

    This callback integrates with W&B for experiment tracking while
    ensuring that no sensitive data (model weights, gradients) is logged.

    Example:
        ```python
        callback = TenSafeWandbCallback(
            config=TenSafeWandbConfig(project="my-project"),
            dp_config=DPConfig(target_epsilon=8.0),
        )

        callback.on_train_begin(model_config={"model": "llama-3-8b"})

        for step in range(num_steps):
            loss = train_step()
            callback.on_step_end(step=step, loss=loss, epsilon=current_epsilon)

        callback.on_train_end(final_epsilon=final_epsilon)
        ```
    """

    def __init__(
        self,
        config: Optional[TenSafeWandbConfig] = None,
        dp_config: Optional[Any] = None,
    ):
        """Initialize W&B callback.

        Args:
            config: W&B configuration
            dp_config: Differential privacy configuration
        """
        self.config = config or TenSafeWandbConfig()
        self.dp_config = dp_config
        self.run = None
        self._step = 0
        self._start_time = None

    def on_train_begin(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize W&B run at training start.

        Args:
            model_config: Model configuration (safe to log)
            training_config: Training configuration (will be redacted)
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, skipping initialization")
            return

        # Build config dict
        run_config = {
            "framework": "tensafe",
            "framework_version": "3.0.0",
            "privacy_enabled": self.dp_config is not None,
        }

        # Add DP config (safe metadata only)
        if self.dp_config:
            run_config.update({
                "dp_enabled": getattr(self.dp_config, 'enabled', True),
                "dp_noise_multiplier": getattr(self.dp_config, 'noise_multiplier', None),
                "dp_max_grad_norm": getattr(self.dp_config, 'max_grad_norm', None),
                "dp_target_epsilon": getattr(self.dp_config, 'target_epsilon', None),
                "dp_target_delta": getattr(self.dp_config, 'target_delta', None),
            })

        # Add model config (redact sensitive fields)
        if model_config:
            safe_model_config = self._redact_config(model_config)
            run_config.update(safe_model_config)

        # Add training config
        if training_config:
            safe_training_config = self._redact_config(training_config)
            run_config.update(safe_training_config)

        # Initialize run
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.name,
            tags=self.config.tags,
            notes=self.config.notes,
            config=run_config,
        )

        self._start_time = time.time()
        logger.info(f"W&B run initialized: {self.run.url}")

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        he_lora_latency_ms: Optional[float] = None,
        **kwargs,
    ):
        """Log metrics at step end.

        Args:
            step: Current training step
            loss: Training loss
            learning_rate: Current learning rate
            epsilon: Current privacy epsilon
            delta: Current privacy delta
            he_lora_latency_ms: HE-LoRA latency in milliseconds
            **kwargs: Additional metrics to log
        """
        if not WANDB_AVAILABLE or self.run is None:
            return

        self._step = step

        # Only log at specified frequency
        if step % self.config.log_frequency != 0:
            return

        metrics = {
            "train/step": step,
            "train/loss": loss,
        }

        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        # Privacy metrics
        if self.config.log_privacy_metrics:
            if epsilon is not None:
                metrics["privacy/epsilon"] = epsilon
            if delta is not None:
                metrics["privacy/delta"] = delta

            # Privacy budget utilization (if target epsilon set)
            if epsilon is not None and self.dp_config:
                target = getattr(self.dp_config, 'target_epsilon', None)
                if target:
                    metrics["privacy/budget_utilization"] = epsilon / target

        # HE-LoRA metrics
        if self.config.log_he_metrics and he_lora_latency_ms is not None:
            metrics["he_lora/latency_ms"] = he_lora_latency_ms

        # Additional safe metrics
        for key, value in kwargs.items():
            if self._is_safe_to_log(key, value):
                metrics[f"custom/{key}"] = value

        # Compute throughput
        if self._start_time:
            elapsed = time.time() - self._start_time
            metrics["train/steps_per_second"] = step / elapsed if elapsed > 0 else 0

        wandb.log(metrics, step=step)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        eval_loss: Optional[float] = None,
        **kwargs,
    ):
        """Log metrics at epoch end.

        Args:
            epoch: Current epoch
            train_loss: Average training loss for epoch
            eval_loss: Evaluation loss (if available)
            **kwargs: Additional metrics
        """
        if not WANDB_AVAILABLE or self.run is None:
            return

        metrics = {
            "epoch/epoch": epoch,
            "epoch/train_loss": train_loss,
        }

        if eval_loss is not None:
            metrics["epoch/eval_loss"] = eval_loss

        for key, value in kwargs.items():
            if self._is_safe_to_log(key, value):
                metrics[f"epoch/{key}"] = value

        wandb.log(metrics)

    def on_train_end(
        self,
        final_epsilon: Optional[float] = None,
        final_delta: Optional[float] = None,
        final_loss: Optional[float] = None,
        **kwargs,
    ):
        """Finalize W&B run at training end.

        Args:
            final_epsilon: Final privacy epsilon
            final_delta: Final privacy delta
            final_loss: Final training loss
            **kwargs: Additional summary metrics
        """
        if not WANDB_AVAILABLE or self.run is None:
            return

        # Log final summary
        if final_epsilon is not None:
            wandb.summary["final_privacy_epsilon"] = final_epsilon
        if final_delta is not None:
            wandb.summary["final_privacy_delta"] = final_delta
        if final_loss is not None:
            wandb.summary["final_train_loss"] = final_loss

        # Training duration
        if self._start_time:
            duration = time.time() - self._start_time
            wandb.summary["training_duration_seconds"] = duration

        for key, value in kwargs.items():
            if self._is_safe_to_log(key, value):
                wandb.summary[key] = value

        wandb.finish()
        logger.info("W&B run finished")

    def log_model_metadata(
        self,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log model metadata as artifact (NOT weights).

        Args:
            model_path: Path to model directory
            metadata: Additional metadata
        """
        if not WANDB_AVAILABLE or self.run is None:
            return

        if self.config.log_model_weights:
            logger.warning(
                "log_model_weights is True but TenSafe will NOT log actual weights "
                "for privacy. Only metadata will be logged."
            )

        artifact = wandb.Artifact(
            name="model-metadata",
            type="model-metadata",
            metadata=metadata or {},
        )

        # Only add config files, NOT weight files
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            artifact.add_file(config_file)

        wandb.log_artifact(artifact)

    def _redact_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from config."""
        if not self.config.redact_sensitive_config:
            return config

        sensitive_patterns = [
            "password", "token", "key", "secret", "credential",
            "api_key", "private", "auth"
        ]

        redacted = {}
        for key, value in config.items():
            key_lower = key.lower()
            is_sensitive = any(pattern in key_lower for pattern in sensitive_patterns)

            if is_sensitive:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_config(value)
            else:
                redacted[key] = value

        return redacted

    def _is_safe_to_log(self, key: str, value: Any) -> bool:
        """Check if a metric is safe to log."""
        # Never log tensors (could be weights/gradients)
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return False
        except ImportError:
            pass

        try:
            import numpy as np
            if isinstance(value, np.ndarray) and value.size > 100:
                return False
        except ImportError:
            pass

        # Check key patterns
        unsafe_patterns = ["weight", "gradient", "param", "embedding"]
        key_lower = key.lower()

        return not any(pattern in key_lower for pattern in unsafe_patterns)
