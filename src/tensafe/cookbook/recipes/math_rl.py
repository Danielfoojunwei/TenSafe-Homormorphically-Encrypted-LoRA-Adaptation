"""
Math Reasoning RL Recipe.

Train language models to solve mathematical problems using reinforcement
learning with verifiable rewards. Uses GRPO-style reward centering for
stable training.

The reward is computed by extracting the boxed answer from the model's
response and comparing it against the ground truth.

Example usage:
    from tensafe.cookbook.recipes import MathRLConfig, run_math_rl

    config = MathRLConfig(
        model_name="meta-llama/Llama-3.1-8B",
        dataset="gsm8k",
        batch_size=128,
        group_size=16,
        learning_rate=4e-5,
    )
    await run_math_rl(config, client)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..completers import TokensWithLogprobs
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
class MathRLConfig:
    """Configuration for math reasoning RL."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: Optional[str] = None

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: float = 64.0

    # Dataset settings
    dataset: str = "gsm8k"  # "gsm8k", "math", or custom path
    dataset_split: str = "train"

    # RL settings
    batch_size: int = 128  # Number of problems per batch
    group_size: int = 16  # Number of responses per problem
    max_response_tokens: int = 256
    temperature: float = 0.7

    # Training settings
    learning_rate: float = 4e-5
    weight_decay: float = 0.01
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1

    # Reward settings
    reward_correct: float = 1.0
    reward_incorrect: float = 0.0
    normalize_advantages: bool = True

    # Checkpointing
    save_steps: int = 100
    checkpoint_dir: str = "/tmp/tensafe-math-rl"

    # Logging
    log_steps: int = 10
    eval_steps: int = 100

    def __post_init__(self):
        if self.renderer_name is None:
            self.renderer_name = get_recommended_renderer_name(self.model_name)

    @property
    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


@dataclass
class MathProblem:
    """A math problem with question and answer."""

    question: str
    answer: str
    answer_value: Optional[float] = None  # Numeric answer if applicable

    @classmethod
    def from_gsm8k(cls, item: Dict[str, Any]) -> "MathProblem":
        """Create from GSM8K format."""
        question = item["question"]
        answer = item["answer"]

        # Extract numeric answer from GSM8K format (#### followed by number)
        match = re.search(r"####\s*([\d,.-]+)", answer)
        value = None
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        return cls(question=question, answer=answer, answer_value=value)


@dataclass
class RolloutResult:
    """Result from a single rollout (response generation)."""

    problem: MathProblem
    response: str
    tokens: List[int]
    logprobs: List[float]
    reward: float = 0.0
    advantage: float = 0.0
    extracted_answer: Optional[str] = None
    is_correct: bool = False


@dataclass
class GroupRollout:
    """A group of rollouts for the same problem."""

    problem: MathProblem
    rollouts: List[RolloutResult]

    @property
    def mean_reward(self) -> float:
        """Mean reward across rollouts."""
        if not self.rollouts:
            return 0.0
        return sum(r.reward for r in self.rollouts) / len(self.rollouts)

    @property
    def all_same_reward(self) -> bool:
        """Check if all rollouts have the same reward."""
        if not self.rollouts:
            return True
        first_reward = self.rollouts[0].reward
        return all(r.reward == first_reward for r in self.rollouts)

    def compute_advantages(self) -> None:
        """Compute advantages using reward centering."""
        mean = self.mean_reward
        for rollout in self.rollouts:
            rollout.advantage = rollout.reward - mean


class MathRewardFunction:
    """
    Reward function for math problems.

    Extracts boxed answers and compares against ground truth.
    """

    # Pattern to extract boxed answers: \boxed{...} or \\boxed{...}
    BOXED_PATTERN = re.compile(r"\\?\\boxed\{([^}]+)\}")

    # Pattern for GSM8K-style answers: #### followed by number
    GSM8K_PATTERN = re.compile(r"####\s*([\d,.-]+)")

    # Pattern for simple numeric answers at end
    NUMERIC_PATTERN = re.compile(r"(?:(?:answer|result|=)\s*[:=]?\s*)?([\d,.-]+)\s*$", re.IGNORECASE)

    def __init__(
        self,
        reward_correct: float = 1.0,
        reward_incorrect: float = 0.0,
        tolerance: float = 1e-6,
    ):
        """
        Initialize reward function.

        Args:
            reward_correct: Reward for correct answer
            reward_incorrect: Reward for incorrect answer
            tolerance: Numeric comparison tolerance
        """
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.tolerance = tolerance

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the answer from a response."""
        # Try boxed format first
        match = self.BOXED_PATTERN.search(response)
        if match:
            return match.group(1).strip()

        # Try GSM8K format
        match = self.GSM8K_PATTERN.search(response)
        if match:
            return match.group(1).strip()

        # Try numeric at end
        match = self.NUMERIC_PATTERN.search(response)
        if match:
            return match.group(1).strip()

        return None

    def normalize_answer(self, answer: str) -> Optional[float]:
        """Normalize answer to a float for comparison."""
        if answer is None:
            return None

        # Remove commas and extra whitespace
        cleaned = answer.replace(",", "").strip()

        try:
            return float(cleaned)
        except ValueError:
            return None

    def compute_reward(
        self,
        response: str,
        ground_truth: str,
        ground_truth_value: Optional[float] = None,
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Compute reward for a response.

        Args:
            response: Model response
            ground_truth: Ground truth answer string
            ground_truth_value: Pre-computed numeric ground truth

        Returns:
            Tuple of (reward, is_correct, extracted_answer)
        """
        extracted = self.extract_answer(response)

        if extracted is None:
            return self.reward_incorrect, False, None

        # Try numeric comparison
        pred_value = self.normalize_answer(extracted)

        if ground_truth_value is not None and pred_value is not None:
            is_correct = abs(pred_value - ground_truth_value) < self.tolerance
        elif pred_value is not None:
            gt_value = self.normalize_answer(ground_truth)
            if gt_value is not None:
                is_correct = abs(pred_value - gt_value) < self.tolerance
            else:
                # Fall back to string comparison
                is_correct = extracted.lower() == ground_truth.lower()
        else:
            # String comparison
            is_correct = extracted.lower() == ground_truth.lower()

        reward = self.reward_correct if is_correct else self.reward_incorrect
        return reward, is_correct, extracted


class MathDatasetBuilder:
    """
    Dataset builder for math problems.

    Loads GSM8K or other math datasets and prepares prompts.
    """

    SYSTEM_PROMPT = (
        "Solve the following math problem step by step. "
        "Put your final answer in a box like this: \\boxed{answer}"
    )

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        split: str = "train",
        renderer_name: str = "llama3",
        model_name: str = "meta-llama/Llama-3.1-8B",
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize dataset builder."""
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle = shuffle
        self.seed = seed

        # Load tokenizer and renderer
        from ..tokenizer_utils import get_tokenizer
        from ..renderers import get_renderer

        self.tokenizer = get_tokenizer(model_name)
        self.renderer = get_renderer(renderer_name, self.tokenizer)

        self._data: Optional[List[MathProblem]] = None
        self._index = 0

    def _load_data(self) -> None:
        """Load the dataset."""
        try:
            from datasets import load_dataset

            if self.dataset_name == "gsm8k":
                dataset = load_dataset("gsm8k", "main", split=self.split)
            else:
                dataset = load_dataset(self.dataset_name, split=self.split)

            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)

            self._data = [MathProblem.from_gsm8k(item) for item in dataset]

        except ImportError:
            logger.warning("datasets library not available, using mock data")
            self._data = self._generate_mock_data()

    def _generate_mock_data(self) -> List[MathProblem]:
        """Generate mock math problems for testing."""
        problems = []
        for i in range(100):
            a, b = i + 1, i + 2
            problems.append(
                MathProblem(
                    question=f"What is {a} + {b}?",
                    answer=f"Let me calculate: {a} + {b} = {a + b}. #### {a + b}",
                    answer_value=float(a + b),
                )
            )
        return problems

    def __len__(self) -> int:
        if self._data is None:
            self._load_data()
        return len(self._data)

    def get_batch(self, batch_size: int) -> List[MathProblem]:
        """Get a batch of problems."""
        if self._data is None:
            self._load_data()

        batch = []
        for _ in range(batch_size):
            if self._index >= len(self._data):
                self._index = 0  # Wrap around

            batch.append(self._data[self._index])
            self._index += 1

        return batch

    def build_prompt(self, problem: MathProblem) -> str:
        """Build prompt for a problem."""
        from ..renderers.base import Message

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=problem.question),
        ]

        tokens = self.renderer.build_generation_prompt(messages)
        return self.renderer.decode(tokens, skip_special_tokens=False)


class MathRLTrainer:
    """
    Trainer for math reasoning RL.

    Implements GRPO-style training with reward centering.
    """

    def __init__(
        self,
        config: MathRLConfig,
        training_client: TrainingClient,
        sampling_client: SamplingClient,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            training_client: Client for training operations
            sampling_client: Client for sampling responses
        """
        self.config = config
        self.training_client = training_client
        self.sampling_client = sampling_client

        # Initialize components
        self.dataset = MathDatasetBuilder(
            dataset_name=config.dataset,
            split=config.dataset_split,
            renderer_name=config.renderer_name,
            model_name=config.model_name,
        )

        self.reward_fn = MathRewardFunction(
            reward_correct=config.reward_correct,
            reward_incorrect=config.reward_incorrect,
        )

        # Metrics
        self.step = 0
        self.total_correct = 0
        self.total_samples = 0

    async def generate_rollouts(
        self,
        problems: List[MathProblem],
    ) -> List[GroupRollout]:
        """
        Generate rollouts for a batch of problems.

        Args:
            problems: List of math problems

        Returns:
            List of GroupRollouts, one per problem
        """
        # Build prompts
        prompts = [self.dataset.build_prompt(p) for p in problems]

        # Generate multiple responses per problem
        all_prompts = prompts * self.config.group_size

        # Sample from model
        result = self.sampling_client.sample(
            prompts=all_prompts,
            max_tokens=self.config.max_response_tokens,
            temperature=self.config.temperature,
            top_p=0.95,
            top_k=50,
        )

        # Extract responses
        if hasattr(result, "samples"):
            responses = [s.completion for s in result.samples]
        else:
            responses = [s.get("completion", "") for s in result.get("samples", [])]

        # Group by problem
        group_rollouts = []
        num_problems = len(problems)

        for i, problem in enumerate(problems):
            rollouts = []

            for g in range(self.config.group_size):
                idx = i + g * num_problems
                response = responses[idx] if idx < len(responses) else ""

                # Compute reward
                reward, is_correct, extracted = self.reward_fn.compute_reward(
                    response,
                    problem.answer,
                    problem.answer_value,
                )

                rollout = RolloutResult(
                    problem=problem,
                    response=response,
                    tokens=[],  # Would be filled with actual tokens
                    logprobs=[],  # Would be filled with actual logprobs
                    reward=reward,
                    extracted_answer=extracted,
                    is_correct=is_correct,
                )
                rollouts.append(rollout)

                # Update metrics
                self.total_samples += 1
                if is_correct:
                    self.total_correct += 1

            group = GroupRollout(problem=problem, rollouts=rollouts)
            group.compute_advantages()
            group_rollouts.append(group)

        return group_rollouts

    def build_training_batch(
        self,
        rollouts: List[GroupRollout],
    ) -> Dict[str, Any]:
        """
        Build training batch from rollouts.

        Filters out rollouts where all advantages are zero (all same reward).

        Args:
            rollouts: List of GroupRollouts

        Returns:
            Training batch dictionary
        """
        input_ids = []
        attention_mask = []
        advantages = []

        for group in rollouts:
            # Skip if all same reward (no learning signal)
            if group.all_same_reward:
                continue

            for rollout in group.rollouts:
                if rollout.advantage == 0:
                    continue

                # Build full sequence (prompt + response)
                # In real implementation, would use actual tokens
                tokens = list(range(100))  # Placeholder
                mask = [1] * len(tokens)

                input_ids.append(tokens)
                attention_mask.append(mask)
                advantages.append(rollout.advantage)

        if not input_ids:
            return {}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "advantages": advantages,
        }

    async def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute a single training step."""
        if not batch:
            return 0.0

        # Forward-backward with advantages as weights
        fb_future = self.training_client.forward_backward(batch)
        fb_result = fb_future.result()

        # Optimizer step
        opt_future = self.training_client.optim_step()
        opt_future.result()

        return fb_result.loss

    async def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Final training metrics
        """
        logger.info(f"Starting math RL training for {self.config.max_steps} steps")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Dataset: {self.config.dataset}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Group size: {self.config.group_size}")

        for step in range(self.config.max_steps):
            self.step = step

            # Get batch of problems
            problems = self.dataset.get_batch(self.config.batch_size)

            # Generate rollouts
            rollouts = await self.generate_rollouts(problems)

            # Build training batch
            batch = self.build_training_batch(rollouts)

            # Train step
            loss = await self.train_step(batch)

            # Log metrics
            if step % self.config.log_steps == 0:
                accuracy = (
                    self.total_correct / self.total_samples
                    if self.total_samples > 0
                    else 0.0
                )
                logger.info(
                    f"Step {step} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}"
                )

            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.training_client.save_state(
                    metadata={"step": step, "accuracy": accuracy}
                )

        # Final metrics
        final_accuracy = (
            self.total_correct / self.total_samples if self.total_samples > 0 else 0.0
        )

        return {
            "final_step": self.step,
            "accuracy": final_accuracy,
            "total_samples": self.total_samples,
        }


async def run_math_rl(
    config: MathRLConfig,
    training_client: TrainingClient,
    sampling_client: SamplingClient,
) -> Dict[str, Any]:
    """
    Run math reasoning RL training.

    Args:
        config: Training configuration
        training_client: Client for training operations
        sampling_client: Client for sampling

    Returns:
        Final training metrics
    """
    trainer = MathRLTrainer(config, training_client, sampling_client)
    return await trainer.train()
