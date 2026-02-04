"""
Chat Supervised Learning Recipe.

Fine-tune language models on conversational datasets using supervised learning.
This is the simplest form of instruction tuning, training the model to predict
assistant responses given conversation history.

Example usage:
    from tensafe.cookbook.recipes import ChatSLConfig, run_chat_sl

    config = ChatSLConfig(
        model_name="meta-llama/Llama-3.1-8B",
        dataset="HuggingFaceH4/no_robots",
        batch_size=128,
        learning_rate=2e-4,
    )
    await run_chat_sl(config, client)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, Union

from ..hyperparam_utils import LoRAConfig, calculate_warmup_steps, get_lora_lr
from ..model_info import get_recommended_renderer_name
from ..renderers import get_renderer

logger = logging.getLogger(__name__)


class TrainingClient(Protocol):
    """Protocol for training clients."""

    def forward_backward(self, batch: Dict[str, Any]) -> Any:
        ...

    def optim_step(self) -> Any:
        ...

    def save_state(self, **kwargs) -> Any:
        ...


@dataclass
class ChatSLConfig:
    """Configuration for chat supervised learning."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: Optional[str] = None  # Auto-detect if None

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.0
    lora_targets: Optional[List[str]] = None

    # Dataset settings
    dataset: str = "HuggingFaceH4/no_robots"
    dataset_split: str = "train"
    max_seq_length: int = 4096
    shuffle: bool = True
    seed: int = 42

    # Training settings
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    lr_scheduler: str = "linear"  # "linear", "cosine", "constant"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training duration
    num_epochs: int = 1
    max_steps: Optional[int] = None  # If set, overrides epochs

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    checkpoint_dir: str = "/tmp/tensafe-chat-sl"

    # Logging
    log_steps: int = 10
    log_path: Optional[str] = None

    # Privacy (optional)
    use_dp: bool = False
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5

    def __post_init__(self):
        if self.renderer_name is None:
            self.renderer_name = get_recommended_renderer_name(self.model_name)

    @property
    def lora_config(self) -> LoRAConfig:
        """Get LoRA configuration."""
        return LoRAConfig(
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_modules=self.lora_targets,
        )

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class ChatSLMetrics:
    """Metrics for chat SL training."""

    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    samples_seen: int = 0
    tokens_seen: int = 0
    grad_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "tokens_per_second": self.tokens_per_second,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
            "grad_norm": self.grad_norm,
        }


class ConversationDataset:
    """
    Dataset wrapper for conversational data.

    Handles loading and preprocessing of conversation datasets
    into training examples.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        renderer_name: str = "llama3",
        model_name: str = "meta-llama/Llama-3.1-8B",
        max_seq_length: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_name: HuggingFace dataset name or path
            split: Dataset split
            renderer_name: Renderer for tokenization
            model_name: Model name for tokenizer
            max_seq_length: Maximum sequence length
            shuffle: Whether to shuffle data
            seed: Random seed
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.seed = seed

        # Load tokenizer and renderer
        from ..tokenizer_utils import get_tokenizer

        self.tokenizer = get_tokenizer(model_name)
        self.renderer = get_renderer(renderer_name, self.tokenizer)

        # Will be loaded lazily
        self._data: Optional[List[Dict[str, Any]]] = None
        self._index = 0

    def _load_data(self):
        """Load and preprocess the dataset."""
        try:
            from datasets import load_dataset

            dataset = load_dataset(self.dataset_name, split=self.split)

            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)

            self._data = list(dataset)
        except ImportError:
            logger.warning("datasets library not available, using mock data")
            self._data = self._generate_mock_data()

    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock conversation data for testing."""
        return [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer to question {i}"},
                ]
            }
            for i in range(100)
        ]

    def __len__(self) -> int:
        if self._data is None:
            self._load_data()
        return len(self._data)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._data is None:
            self._load_data()
        self._index = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._index >= len(self._data):
            raise StopIteration

        item = self._data[self._index]
        self._index += 1
        return self._process_item(item)

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single conversation into training tokens."""
        messages = item.get("messages", item.get("conversation", []))

        # Convert to Messages if needed
        from ..renderers.base import Message

        msg_list = [
            Message.from_dict(m) if isinstance(m, dict) else m for m in messages
        ]

        # Build supervised example
        input_ids, weights = self.renderer.build_supervised_example(msg_list)

        # Truncate if needed
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            weights = weights[: self.max_seq_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Create labels (shift input_ids and mask non-trainable)
        labels = []
        for i, (token, weight) in enumerate(zip(input_ids, weights)):
            if weight > 0 and i > 0:
                labels.append(input_ids[i])
            else:
                labels.append(-100)  # Ignore in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_batch(self, batch_size: int) -> Dict[str, List[List[int]]]:
        """Get a batch of examples."""
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for _ in range(batch_size):
            try:
                item = next(self)
                batch["input_ids"].append(item["input_ids"])
                batch["attention_mask"].append(item["attention_mask"])
                batch["labels"].append(item["labels"])
            except StopIteration:
                break

        return batch


class ChatSLTrainer:
    """
    Trainer for chat supervised learning.

    Manages the training loop, checkpointing, and metrics logging.
    """

    def __init__(
        self,
        config: ChatSLConfig,
        client: TrainingClient,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            client: TrainingClient for API calls
        """
        self.config = config
        self.client = client
        self.metrics = ChatSLMetrics()

        # Initialize dataset
        self.dataset = ConversationDataset(
            dataset_name=config.dataset,
            split=config.dataset_split,
            renderer_name=config.renderer_name,
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            shuffle=config.shuffle,
            seed=config.seed,
        )

        # Learning rate schedule
        self._current_lr = config.learning_rate

    def get_learning_rate(self, step: int, total_steps: int) -> float:
        """Calculate learning rate for current step."""
        warmup_steps = calculate_warmup_steps(
            total_steps, self.config.warmup_ratio
        )

        if step < warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / warmup_steps)

        if self.config.lr_scheduler == "linear":
            # Linear decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return self.config.learning_rate * (1 - progress)

        elif self.config.lr_scheduler == "cosine":
            # Cosine decay
            import math

            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return self.config.learning_rate * (1 + math.cos(math.pi * progress)) / 2

        else:
            # Constant
            return self.config.learning_rate

    async def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute a single training step.

        Args:
            batch: Training batch

        Returns:
            Loss value
        """
        # Forward-backward pass
        fb_future = self.client.forward_backward(batch)
        fb_result = fb_future.result()

        # Optimizer step
        opt_future = self.client.optim_step()
        opt_result = opt_future.result()

        # Update metrics
        self.metrics.step = opt_result.step
        self.metrics.loss = fb_result.loss

        return fb_result.loss

    async def train_epoch(self, epoch: int, total_steps: int) -> None:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            total_steps: Total training steps
        """
        logger.info(f"Starting epoch {epoch + 1}")

        # Reset dataset iterator
        iter(self.dataset)

        accumulated_loss = 0.0
        accumulated_steps = 0

        step = self.metrics.step

        while True:
            batch = self.dataset.get_batch(
                self.config.batch_size // self.config.gradient_accumulation_steps
            )

            if not batch["input_ids"]:
                break  # End of epoch

            # Train step
            loss = await self.train_step(batch)
            accumulated_loss += loss
            accumulated_steps += 1

            # Update learning rate
            self._current_lr = self.get_learning_rate(step, total_steps)
            self.metrics.learning_rate = self._current_lr

            # Log metrics
            if step % self.config.log_steps == 0:
                avg_loss = accumulated_loss / max(accumulated_steps, 1)
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | LR: {self._current_lr:.2e}"
                )
                accumulated_loss = 0.0
                accumulated_steps = 0

            # Save checkpoint
            if step % self.config.save_steps == 0:
                await self.save_checkpoint(step)

            step += 1

            if self.config.max_steps and step >= self.config.max_steps:
                break

    async def save_checkpoint(self, step: int) -> None:
        """Save a training checkpoint."""
        logger.info(f"Saving checkpoint at step {step}")

        result = self.client.save_state(
            metadata={
                "step": step,
                "loss": self.metrics.loss,
                "config": {
                    "model_name": self.config.model_name,
                    "lora_rank": self.config.lora_rank,
                    "batch_size": self.config.batch_size,
                },
            }
        )

        logger.info(f"Checkpoint saved: {result.artifact_id}")

    async def train(self) -> ChatSLMetrics:
        """
        Run the full training loop.

        Returns:
            Final training metrics
        """
        # Calculate total steps
        steps_per_epoch = len(self.dataset) // self.config.batch_size
        total_steps = (
            self.config.max_steps
            if self.config.max_steps
            else steps_per_epoch * self.config.num_epochs
        )

        logger.info(f"Starting training for {total_steps} steps")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Dataset: {self.config.dataset}")
        logger.info(f"  Batch size: {self.config.effective_batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")

        for epoch in range(self.config.num_epochs):
            await self.train_epoch(epoch, total_steps)

            if self.config.max_steps and self.metrics.step >= self.config.max_steps:
                break

        # Final checkpoint
        await self.save_checkpoint(self.metrics.step)

        logger.info("Training complete!")
        return self.metrics


async def run_chat_sl(
    config: ChatSLConfig,
    client: TrainingClient,
) -> ChatSLMetrics:
    """
    Run chat supervised learning training.

    Args:
        config: Training configuration
        client: TrainingClient instance

    Returns:
        Final training metrics
    """
    trainer = ChatSLTrainer(config, client)
    return await trainer.train()
