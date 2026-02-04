"""
Prompt Distillation Recipe.

Train models to internalize complex instructions into their weights,
reducing inference-time prompt length while maintaining quality.

This technique is useful for:
- Reducing token costs at inference
- Improving latency by shortening prompts
- Creating specialized models from general ones

Example usage:
    from tensafe.cookbook.recipes import DistillationConfig, run_distillation

    config = DistillationConfig(
        model_name="meta-llama/Llama-3.1-8B",
        long_prompt="You are a helpful assistant that...",  # Full system prompt
        short_prompt="Be helpful.",  # Shortened version
    )
    await run_distillation(config, client)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..hyperparam_utils import LoRAConfig
from ..model_info import get_recommended_renderer_name

logger = logging.getLogger(__name__)


class TrainingClient(Protocol):
    """Protocol for training clients."""

    def forward_backward(self, batch: Dict[str, Any]) -> Any:
        ...

    def optim_step(self) -> Any:
        ...

    def save_state(self, **kwargs) -> Any:
        ...


class SamplingClient(Protocol):
    """Protocol for sampling clients."""

    def sample(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Any:
        ...


@dataclass
class DistillationConfig:
    """Configuration for prompt distillation."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: Optional[str] = None

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: float = 64.0

    # Prompts
    long_prompt: str = ""  # Full detailed system prompt
    short_prompt: str = ""  # Shortened version to use at inference

    # Dataset settings
    dataset: str = "HuggingFaceH4/no_robots"  # Queries to train on
    dataset_split: str = "train"
    max_seq_length: int = 4096

    # Training settings
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: Optional[int] = None

    # Distillation settings
    temperature: float = 1.0  # Temperature for generating teacher outputs
    use_kl_loss: bool = True  # Use KL divergence loss
    kl_weight: float = 0.5  # Weight for KL loss vs CE loss

    # Checkpointing
    checkpoint_dir: str = "/tmp/tensafe-distillation"
    save_steps: int = 500

    # Logging
    log_steps: int = 10

    def __post_init__(self):
        if self.renderer_name is None:
            self.renderer_name = get_recommended_renderer_name(self.model_name)

    @property
    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


@dataclass
class DistillationExample:
    """A distillation training example."""

    query: str  # User query
    teacher_response: str  # Response with long prompt
    short_prompt: str  # Shortened prompt for training


class DistillationDataset:
    """
    Dataset for prompt distillation.

    Generates teacher outputs using the long prompt, then trains
    the model to produce the same outputs with the short prompt.
    """

    def __init__(
        self,
        long_prompt: str,
        short_prompt: str,
        dataset_name: str = "HuggingFaceH4/no_robots",
        split: str = "train",
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            long_prompt: Full detailed system prompt
            short_prompt: Shortened prompt
            dataset_name: Source of user queries
            split: Dataset split
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.long_prompt = long_prompt
        self.short_prompt = short_prompt
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle = shuffle
        self.seed = seed

        self._queries: Optional[List[str]] = None
        self._examples: Optional[List[DistillationExample]] = None
        self._index = 0

    def _load_queries(self) -> None:
        """Load queries from dataset."""
        try:
            from datasets import load_dataset

            dataset = load_dataset(self.dataset_name, split=self.split)

            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)

            self._queries = []
            for item in dataset:
                messages = item.get("messages", item.get("conversation", []))
                # Extract user queries
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        self._queries.append(msg.get("content", ""))

        except ImportError:
            logger.warning("datasets library not available, using mock data")
            self._queries = [f"Question {i}" for i in range(100)]

    def generate_teacher_outputs(
        self,
        client: SamplingClient,
        batch_size: int = 32,
        max_tokens: int = 512,
    ) -> None:
        """
        Generate teacher outputs using the long prompt.

        Args:
            client: Sampling client
            batch_size: Batch size for generation
            max_tokens: Max tokens per response
        """
        if self._queries is None:
            self._load_queries()

        self._examples = []

        # Process in batches
        for i in range(0, len(self._queries), batch_size):
            batch_queries = self._queries[i : i + batch_size]

            # Build prompts with long system prompt
            prompts = [
                f"System: {self.long_prompt}\n\nUser: {q}\n\nAssistant:"
                for q in batch_queries
            ]

            # Generate teacher outputs
            result = client.sample(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
            )

            # Extract responses
            if hasattr(result, "samples"):
                responses = [s.completion for s in result.samples]
            else:
                responses = [
                    s.get("completion", "") for s in result.get("samples", [])
                ]

            # Create examples
            for query, response in zip(batch_queries, responses):
                self._examples.append(
                    DistillationExample(
                        query=query,
                        teacher_response=response,
                        short_prompt=self.short_prompt,
                    )
                )

            logger.info(
                f"Generated {len(self._examples)} / {len(self._queries)} examples"
            )

    def __len__(self) -> int:
        if self._examples is None:
            if self._queries is None:
                self._load_queries()
            return len(self._queries)
        return len(self._examples)

    def get_batch(self, batch_size: int) -> List[DistillationExample]:
        """Get a batch of examples."""
        if self._examples is None:
            raise RuntimeError(
                "Must call generate_teacher_outputs before getting batches"
            )

        batch = []
        for _ in range(batch_size):
            if self._index >= len(self._examples):
                self._index = 0

            batch.append(self._examples[self._index])
            self._index += 1

        return batch


class DistillationTrainer:
    """
    Trainer for prompt distillation.

    Trains the model to produce teacher-quality outputs with
    a shorter prompt.
    """

    def __init__(
        self,
        config: DistillationConfig,
        training_client: TrainingClient,
        sampling_client: SamplingClient,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            training_client: Client for training
            sampling_client: Client for generating teacher outputs
        """
        self.config = config
        self.training_client = training_client
        self.sampling_client = sampling_client

        # Initialize dataset
        self.dataset = DistillationDataset(
            long_prompt=config.long_prompt,
            short_prompt=config.short_prompt,
            dataset_name=config.dataset,
            split=config.dataset_split,
        )

    def build_training_example(
        self, example: DistillationExample
    ) -> Dict[str, Any]:
        """
        Build training tokens from a distillation example.

        The student is trained to produce the teacher's response
        given only the short prompt.

        Args:
            example: Distillation example

        Returns:
            Training batch item
        """
        from ..tokenizer_utils import get_tokenizer
        from ..renderers import get_renderer
        from ..renderers.base import Message

        tokenizer = get_tokenizer(self.config.model_name)
        renderer = get_renderer(self.config.renderer_name, tokenizer)

        # Build conversation with SHORT prompt
        messages = [
            Message(role="system", content=example.short_prompt),
            Message(role="user", content=example.query),
            Message(role="assistant", content=example.teacher_response),
        ]

        # Build supervised example
        input_ids, weights = renderer.build_supervised_example(messages)

        # Truncate if needed
        max_len = self.config.max_seq_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            weights = weights[:max_len]

        attention_mask = [1] * len(input_ids)

        # Build labels
        labels = []
        for i, (token, weight) in enumerate(zip(input_ids, weights)):
            if weight > 0 and i > 0:
                labels.append(input_ids[i])
            else:
                labels.append(-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    async def train(self) -> Dict[str, Any]:
        """
        Run distillation training.

        Returns:
            Training metrics
        """
        logger.info("Starting prompt distillation")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Long prompt length: {len(self.config.long_prompt)} chars")
        logger.info(f"  Short prompt length: {len(self.config.short_prompt)} chars")
        logger.info(
            f"  Compression ratio: {len(self.config.long_prompt) / max(1, len(self.config.short_prompt)):.1f}x"
        )

        # Generate teacher outputs
        logger.info("Generating teacher outputs...")
        self.dataset.generate_teacher_outputs(
            self.sampling_client,
            batch_size=self.config.batch_size,
        )

        # Train
        total_loss = 0.0
        num_steps = self.config.max_steps or (
            len(self.dataset) * self.config.num_epochs // self.config.batch_size
        )

        logger.info(f"Training for {num_steps} steps...")

        for step in range(num_steps):
            # Get batch
            examples = self.dataset.get_batch(self.config.batch_size)

            # Build training batch
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for example in examples:
                item = self.build_training_example(example)
                batch["input_ids"].append(item["input_ids"])
                batch["attention_mask"].append(item["attention_mask"])
                batch["labels"].append(item["labels"])

            # Train step
            fb_future = self.training_client.forward_backward(batch)
            fb_result = fb_future.result()

            opt_future = self.training_client.optim_step()
            opt_future.result()

            total_loss += fb_result.loss

            # Log
            if step % self.config.log_steps == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"Step {step} | Loss: {avg_loss:.4f}")

            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.training_client.save_state(metadata={"step": step})

        final_loss = total_loss / num_steps
        logger.info(f"Distillation complete. Final loss: {final_loss:.4f}")

        return {
            "final_loss": final_loss,
            "num_steps": num_steps,
            "num_examples": len(self.dataset),
            "compression_ratio": len(self.config.long_prompt)
            / max(1, len(self.config.short_prompt)),
        }


async def run_distillation(
    config: DistillationConfig,
    training_client: TrainingClient,
    sampling_client: SamplingClient,
) -> Dict[str, Any]:
    """
    Run prompt distillation.

    Args:
        config: Training configuration
        training_client: Client for training
        sampling_client: Client for teacher generation

    Returns:
        Training metrics
    """
    trainer = DistillationTrainer(config, training_client, sampling_client)
    return await trainer.train()
