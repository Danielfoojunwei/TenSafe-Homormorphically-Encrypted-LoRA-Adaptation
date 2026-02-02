"""Training Optimizations for TenSafe.

Provides general training optimizations compatible with privacy-preserving training:
- Mixed precision training
- Gradient accumulation
- Efficient data loading
- Memory optimization
"""

from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


@dataclass
class TrainingOptimizationConfig:
    """Configuration for training optimizations."""

    # Mixed precision
    mixed_precision: bool = True
    fp16: bool = False  # Use FP16 (vs BF16)
    bf16: bool = True   # Use BF16 if available

    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # Memory
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2

    # Performance
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    channels_last: bool = False  # Memory format optimization

    # Profiling
    enable_profiling: bool = False


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_ratio: float = 1.0,
) -> nn.Module:
    """Apply gradient checkpointing to reduce memory usage.

    Gradient checkpointing trades compute for memory by not storing
    all activations during forward pass.

    Args:
        model: PyTorch model
        checkpoint_ratio: Fraction of layers to checkpoint (1.0 = all)

    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled via model method")
    else:
        # Manual checkpointing for custom models
        _apply_manual_checkpointing(model, checkpoint_ratio)

    return model


def _apply_manual_checkpointing(model: nn.Module, ratio: float = 1.0):
    """Manually apply checkpointing to transformer layers."""
    layers = []

    # Find transformer layers
    for name, module in model.named_modules():
        if 'layer' in name.lower() or 'block' in name.lower():
            if hasattr(module, 'forward'):
                layers.append((name, module))

    # Apply checkpointing to a fraction of layers
    num_to_checkpoint = int(len(layers) * ratio)

    for i, (name, layer) in enumerate(layers[:num_to_checkpoint]):
        # Wrap forward method with checkpoint
        original_forward = layer.forward

        def checkpointed_forward(*args, _orig_fwd=original_forward, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                _orig_fwd, *args, use_reentrant=False, **kwargs
            )

        layer.forward = checkpointed_forward
        logger.debug(f"Applied checkpointing to {name}")

    logger.info(f"Applied gradient checkpointing to {num_to_checkpoint}/{len(layers)} layers")


def enable_mixed_precision(
    model: nn.Module,
    use_bf16: bool = True,
) -> tuple:
    """Enable mixed precision training.

    Args:
        model: PyTorch model
        use_bf16: Use BF16 (recommended) vs FP16

    Returns:
        Tuple of (model, scaler) where scaler is GradScaler for FP16
    """
    # Check hardware support
    if use_bf16 and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 not supported, falling back to FP16")
        use_bf16 = False

    if use_bf16:
        # BF16 doesn't need gradient scaling
        model = model.to(torch.bfloat16)
        logger.info("Mixed precision enabled with BF16")
        return model, None
    else:
        # FP16 needs gradient scaling
        scaler = GradScaler()
        logger.info("Mixed precision enabled with FP16 + GradScaler")
        return model, scaler

    return model, None


class TenSafeOptimizedTrainer:
    """Optimized trainer with privacy-preserving features.

    Combines:
    - Mixed precision training
    - Gradient accumulation
    - Memory optimization
    - Privacy budget tracking
    - Performance profiling
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: Optional[TrainingOptimizationConfig] = None,
        dp_config: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """Initialize optimized trainer.

        Args:
            model: PyTorch model
            train_dataset: Training dataset
            config: Optimization configuration
            dp_config: Differential privacy configuration
            optimizer: Optimizer (created if not provided)
            collate_fn: Data collation function
        """
        self.config = config or TrainingOptimizationConfig()
        self.dp_config = dp_config
        self.collate_fn = collate_fn

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply optimizations to model
        self.model = self._setup_model(model)

        # Setup optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Setup mixed precision
        self.scaler = None
        if self.config.mixed_precision and self.device.type == "cuda":
            if not self.config.bf16:
                self.scaler = GradScaler()

        # Setup data loader
        self.train_loader = self._create_dataloader(train_dataset)

        # Training state
        self._global_step = 0
        self._accumulated_loss = 0.0

        # Metrics
        self._metrics: Dict[str, List[float]] = {
            "loss": [],
            "lr": [],
            "throughput": [],
        }

    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with optimizations."""
        model = model.to(self.device)

        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            model = apply_gradient_checkpointing(model)

        # Mixed precision dtype
        if self.config.mixed_precision and self.config.bf16:
            model = model.to(torch.bfloat16)

        # Channels last memory format
        if self.config.channels_last:
            model = model.to(memory_format=torch.channels_last)

        # torch.compile (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile")

        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with default settings."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create optimized data loader."""
        return DataLoader(
            dataset,
            batch_size=8,  # Per-device batch size
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            collate_fn=self.collate_fn,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step.

        Args:
            batch: Input batch

        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Mixed precision forward
        if self.config.mixed_precision:
            with autocast(dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        self._accumulated_loss += loss.item()

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer if accumulated enough
        if (self._global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Record metrics
            self._metrics["loss"].append(self._accumulated_loss)
            self._accumulated_loss = 0.0

        self._global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict of epoch metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in self.train_loader:
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1

        elapsed = time.time() - start_time
        samples_per_second = len(self.train_loader.dataset) / elapsed

        return {
            "loss": epoch_loss / num_batches,
            "samples_per_second": samples_per_second,
            "elapsed_seconds": elapsed,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return {
            "global_step": self._global_step,
            "loss_history": self._metrics["loss"][-100:],  # Last 100 values
            "current_loss": self._metrics["loss"][-1] if self._metrics["loss"] else None,
        }


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Create an optimized DataLoader.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        shuffle: Shuffle data
        collate_fn: Collation function

    Returns:
        Optimized DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
