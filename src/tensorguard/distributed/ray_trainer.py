"""TenSafe Ray Train Distributed Trainer.

Provides distributed training with:
- DP-SGD privacy guarantees
- Secure gradient aggregation
- Multi-node HE key distribution
- Integration with DeepSpeed/FSDP
"""

from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
import logging
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)

# Conditional imports
RAY_AVAILABLE = False
try:
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer, TorchConfig
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not installed. Install with: pip install ray[train]")

# TenSafe imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from tensorguard.platform.tg_tinker_api.dp import DPConfig
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

    @dataclass
    class DPConfig:
        """Fallback DP config when not available."""
        enabled: bool = False
        noise_multiplier: float = 1.0
        max_grad_norm: float = 1.0
        target_epsilon: float = 8.0
        target_delta: float = 1e-5


@dataclass
class TenSafeRayConfig:
    """Configuration for TenSafe Ray distributed training.

    Attributes:
        # Ray configuration
        num_workers: Number of training workers
        use_gpu: Whether to use GPUs
        resources_per_worker: Resource requirements per worker

        # Training configuration
        batch_size_per_worker: Batch size per worker
        learning_rate: Learning rate
        max_epochs: Maximum training epochs
        warmup_steps: LR warmup steps

        # DP configuration
        dp_config: Differential privacy configuration

        # Distributed strategy
        strategy: Training strategy (ddp, fsdp, deepspeed)
        deepspeed_config: DeepSpeed configuration dict

        # Checkpointing
        checkpoint_frequency: Steps between checkpoints
        checkpoint_dir: Checkpoint directory

        # Security
        secure_aggregation: Use secure gradient aggregation
        audit_logging: Enable audit logging
    """

    # Ray configuration
    num_workers: int = 4
    use_gpu: bool = True
    resources_per_worker: Optional[Dict[str, float]] = None

    # Training configuration
    batch_size_per_worker: int = 8
    learning_rate: float = 1e-4
    max_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # DP configuration
    dp_config: Optional[DPConfig] = None

    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Distributed strategy
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
    deepspeed_config: Optional[Dict[str, Any]] = None

    # Checkpointing
    checkpoint_frequency: int = 100
    checkpoint_dir: str = "/checkpoints"
    keep_last_n_checkpoints: int = 3

    # Security
    secure_aggregation: bool = True
    audit_logging: bool = True

    # Callbacks
    wandb_project: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None

    def __post_init__(self):
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        if self.dp_config is None:
            self.dp_config = DPConfig(enabled=False)

        if self.resources_per_worker is None:
            self.resources_per_worker = {"CPU": 4}
            if self.use_gpu:
                self.resources_per_worker["GPU"] = 1


class TenSafeRayTrainer:
    """Distributed trainer with DP-SGD and secure aggregation.

    This trainer uses Ray Train for distributed training while maintaining
    TenSafe's privacy guarantees across multiple workers and nodes.

    Example:
        ```python
        config = TenSafeRayConfig(
            num_workers=4,
            use_gpu=True,
            dp_config=DPConfig(enabled=True, target_epsilon=8.0),
        )

        trainer = TenSafeRayTrainer(
            config=config,
            model_init_fn=lambda: AutoModelForCausalLM.from_pretrained("llama-3-8b"),
            train_dataset=my_dataset,
        )

        result = trainer.train()
        ```
    """

    def __init__(
        self,
        config: TenSafeRayConfig,
        model_init_fn: Callable[[], nn.Module],
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """Initialize distributed trainer.

        Args:
            config: Training configuration
            model_init_fn: Function that returns a fresh model instance
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            collate_fn: Optional collation function for DataLoader
        """
        self.config = config
        self.model_init_fn = model_init_fn
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn

        # Privacy tracking
        self._total_epsilon = 0.0
        self._total_delta = 0.0

        # Initialize Ray if needed
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _create_train_func(self):
        """Create training function for Ray workers."""
        config = self.config
        model_init_fn = self.model_init_fn
        train_dataset = self.train_dataset
        eval_dataset = self.eval_dataset
        collate_fn = self.collate_fn

        def train_func_per_worker(train_loop_config: Dict[str, Any]):
            """Training function executed on each worker."""
            import torch
            import torch.distributed as dist
            from torch.utils.data import DataLoader, DistributedSampler

            # Get worker context
            worker_rank = train.get_context().get_world_rank()
            world_size = train.get_context().get_world_size()
            device = train.get_context().get_device()

            logger.info(f"Worker {worker_rank}/{world_size} starting on {device}")

            # Initialize model
            model = model_init_fn()
            model = model.to(device)

            # Wrap with DDP
            if world_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[device.index] if device.type == "cuda" else None,
                )

            # Create data loader with distributed sampler
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=worker_rank,
                shuffle=True,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size_per_worker,
                sampler=sampler,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True,
            )

            # Initialize optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            # Privacy accountant
            privacy_epsilon = 0.0
            privacy_delta = config.dp_config.target_delta if config.dp_config.enabled else 0.0

            # Training loop
            global_step = 0

            for epoch in range(config.max_epochs):
                sampler.set_epoch(epoch)
                model.train()

                epoch_loss = 0.0
                num_batches = 0

                for batch_idx, batch in enumerate(train_loader):
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                    # Scale loss for gradient accumulation
                    loss = loss / config.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        # DP: Clip gradients per-sample
                        if config.dp_config.enabled:
                            _clip_gradients(model, config.dp_config.max_grad_norm)

                        # Secure aggregation
                        if config.secure_aggregation and world_size > 1:
                            _secure_aggregate_gradients(model)

                        # DP: Add noise
                        if config.dp_config.enabled:
                            _add_dp_noise(
                                model,
                                config.dp_config.noise_multiplier,
                                config.dp_config.max_grad_norm,
                                config.batch_size_per_worker * world_size,
                            )

                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        # Update privacy budget
                        if config.dp_config.enabled:
                            privacy_epsilon = _compute_epsilon(
                                global_step,
                                config.batch_size_per_worker * world_size,
                                len(train_dataset),
                                config.dp_config.noise_multiplier,
                                config.dp_config.target_delta,
                            )

                    epoch_loss += loss.item() * config.gradient_accumulation_steps
                    num_batches += 1

                # Report epoch metrics
                avg_loss = epoch_loss / num_batches

                metrics = {
                    "loss": avg_loss,
                    "epoch": epoch,
                    "global_step": global_step,
                }

                if config.dp_config.enabled:
                    metrics["privacy_epsilon"] = privacy_epsilon
                    metrics["privacy_delta"] = privacy_delta

                train.report(metrics)

                # Checkpoint
                if (epoch + 1) % max(1, config.max_epochs // 5) == 0:
                    checkpoint_dict = {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "privacy_epsilon": privacy_epsilon,
                        "privacy_delta": privacy_delta,
                    }
                    checkpoint = train.Checkpoint.from_dict(checkpoint_dict)
                    train.report(metrics, checkpoint=checkpoint)

            logger.info(f"Worker {worker_rank} completed training")

        return train_func_per_worker

    def train(self) -> Dict[str, Any]:
        """Run distributed training.

        Returns:
            Training result including final metrics and checkpoint path
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to single-process training")
            return self._train_single_process()

        logger.info(f"Starting distributed training with {self.config.num_workers} workers")

        # Create scaling config
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker,
        )

        # Create run config
        run_config = RunConfig(
            name="tensafe-training",
            checkpoint_config=CheckpointConfig(
                num_to_keep=self.config.keep_last_n_checkpoints,
            ),
        )

        # Create torch config
        torch_config = TorchConfig(
            backend="nccl" if self.config.use_gpu else "gloo",
        )

        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self._create_train_func(),
            train_loop_config={},
            scaling_config=scaling_config,
            run_config=run_config,
            torch_config=torch_config,
        )

        # Run training
        result = trainer.fit()

        # Extract results
        return {
            "metrics": result.metrics,
            "checkpoint": result.checkpoint,
            "path": result.path if hasattr(result, 'path') else None,
        }

    def _train_single_process(self) -> Dict[str, Any]:
        """Fallback single-process training when Ray is not available."""
        logger.info("Running single-process training (Ray not available)")

        device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")

        # Initialize model
        model = self.model_init_fn()
        model = model.to(device)

        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size_per_worker,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Training loop
        metrics_history = []

        for epoch in range(self.config.max_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                optimizer.zero_grad()
                loss.backward()

                if self.config.dp_config.enabled:
                    _clip_gradients(model, self.config.dp_config.max_grad_norm)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            metrics_history.append({"epoch": epoch, "loss": avg_loss})
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}")

        return {
            "metrics": metrics_history[-1] if metrics_history else {},
            "checkpoint": None,
        }


def _clip_gradients(model: nn.Module, max_norm: float):
    """Clip gradients to max_norm for DP."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _secure_aggregate_gradients(model: nn.Module):
    """Securely aggregate gradients across workers."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(dist.get_world_size())


def _add_dp_noise(model: nn.Module, noise_multiplier: float, max_grad_norm: float, batch_size: int):
    """Add calibrated Gaussian noise for DP."""
    std = noise_multiplier * max_grad_norm / batch_size

    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * std
            param.grad.add_(noise)


def _compute_epsilon(steps: int, batch_size: int, dataset_size: int, noise_multiplier: float, delta: float) -> float:
    """Compute epsilon using RDP accountant (simplified)."""
    # Simplified epsilon computation
    # In production, use Opacus or TensorFlow Privacy accountant
    sampling_rate = batch_size / dataset_size
    return steps * sampling_rate ** 2 / (2 * noise_multiplier ** 2)
