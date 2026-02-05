"""
Preference Learning (RLHF) Recipe.

Implements a three-stage RLHF pipeline:
1. Supervised Fine-Tuning (SFT) - Train on high-quality demonstrations
2. Reward Model Training - Train a reward model on preference data
3. RL Optimization - Optimize policy with PPO or DPO

Example usage:
    from tensafe.cookbook.recipes import PreferenceConfig, run_preference_learning

    config = PreferenceConfig(
        model_name="meta-llama/Llama-3.1-8B",
        sft_dataset="HuggingFaceH4/no_robots",
        preference_dataset="Anthropic/hh-rlhf",
    )
    await run_preference_learning(config, client)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

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


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning stage."""

    dataset: str = "HuggingFaceH4/no_robots"
    dataset_split: str = "train"
    batch_size: int = 128
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: Optional[int] = None


@dataclass
class RewardModelConfig:
    """Configuration for reward model training stage."""

    dataset: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train"
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: Optional[int] = None

    # Reward model architecture
    use_separate_model: bool = False  # Use same base model with different head
    hidden_dim: int = 4096


@dataclass
class PPOConfig:
    """Configuration for PPO optimization stage."""

    batch_size: int = 64
    mini_batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 4
    max_steps: int = 1000

    # PPO hyperparameters
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # KL penalty
    kl_penalty: str = "kl"  # "kl", "abs", "mse"
    kl_target: float = 6.0
    kl_coef: float = 0.1


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""

    dataset: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train"
    batch_size: int = 64
    learning_rate: float = 5e-5
    num_epochs: int = 1
    max_steps: Optional[int] = None

    # DPO hyperparameters
    beta: float = 0.1  # Temperature for DPO loss
    reference_free: bool = False  # If True, don't use reference model


@dataclass
class PreferenceConfig:
    """Full configuration for preference learning pipeline."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: Optional[str] = None

    # LoRA settings (shared across stages)
    lora_rank: int = 32
    lora_alpha: float = 64.0

    # Stage configurations
    sft_config: SFTConfig = field(default_factory=SFTConfig)
    reward_config: RewardModelConfig = field(default_factory=RewardModelConfig)
    ppo_config: Optional[PPOConfig] = None
    dpo_config: Optional[DPOConfig] = None

    # Training mode
    use_dpo: bool = False  # If True, use DPO instead of PPO
    skip_sft: bool = False  # Skip SFT stage (use pretrained model)
    skip_reward: bool = False  # Skip reward model (for DPO)

    # Checkpointing
    checkpoint_dir: str = "/tmp/tensafe-preference"
    save_steps: int = 500

    # Logging
    log_steps: int = 10

    def __post_init__(self):
        if self.renderer_name is None:
            self.renderer_name = get_recommended_renderer_name(self.model_name)

        # Set up default configs
        if self.use_dpo and self.dpo_config is None:
            self.dpo_config = DPOConfig()
        elif not self.use_dpo and self.ppo_config is None:
            self.ppo_config = PPOConfig()

    @property
    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


@dataclass
class PreferencePair:
    """A preference pair with chosen and rejected responses."""

    prompt: str
    chosen: str
    rejected: str
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None


class PreferenceDataset:
    """
    Dataset for preference learning.

    Handles loading preference pairs from various formats.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize dataset."""
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self._data: Optional[List[PreferencePair]] = None
        self._index = 0

    def _load_data(self) -> None:
        """Load the dataset."""
        try:
            from datasets import load_dataset

            dataset = load_dataset(self.dataset_name, split=self.split)

            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)

            self._data = []
            for item in dataset:
                pair = self._parse_item(item)
                if pair:
                    self._data.append(pair)

        except ImportError:
            logger.warning("datasets library not available, using mock data")
            self._data = self._generate_mock_data()

    def _parse_item(self, item: Dict[str, Any]) -> Optional[PreferencePair]:
        """Parse an item from the dataset."""
        # Handle Anthropic HH-RLHF format
        if "chosen" in item and "rejected" in item:
            return PreferencePair(
                prompt=item.get("prompt", ""),
                chosen=item["chosen"],
                rejected=item["rejected"],
            )

        # Handle OpenAI comparison format
        if "prompt" in item and "response_a" in item and "response_b" in item:
            if item.get("preference") == "a":
                return PreferencePair(
                    prompt=item["prompt"],
                    chosen=item["response_a"],
                    rejected=item["response_b"],
                )
            else:
                return PreferencePair(
                    prompt=item["prompt"],
                    chosen=item["response_b"],
                    rejected=item["response_a"],
                )

        return None

    def _generate_mock_data(self) -> List[PreferencePair]:
        """Generate mock preference data."""
        return [
            PreferencePair(
                prompt=f"Question {i}",
                chosen=f"This is a helpful response to question {i}.",
                rejected=f"This response to question {i} is not as helpful.",
            )
            for i in range(100)
        ]

    def __len__(self) -> int:
        if self._data is None:
            self._load_data()
        return len(self._data)

    def get_batch(self, batch_size: int) -> List[PreferencePair]:
        """Get a batch of preference pairs."""
        if self._data is None:
            self._load_data()

        batch = []
        for _ in range(batch_size):
            if self._index >= len(self._data):
                self._index = 0

            batch.append(self._data[self._index])
            self._index += 1

        return batch


class RewardModel:
    """
    Reward model for scoring responses.

    Can be trained on preference data or used to score new responses.
    """

    def __init__(self, hidden_dim: int = 4096):
        """Initialize reward model."""
        self.hidden_dim = hidden_dim
        # In real implementation, would initialize reward head

    def compute_reward(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """
        Compute reward for a response.

        Args:
            prompt: Input prompt
            response: Model response

        Returns:
            Scalar reward value
        """
        # Mock implementation
        # Real implementation would forward through model
        return 0.5

    def compute_pairwise_loss(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Tuple[float, float, float]:
        """
        Compute pairwise ranking loss.

        Args:
            prompt: Input prompt
            chosen: Preferred response
            rejected: Non-preferred response

        Returns:
            Tuple of (loss, chosen_reward, rejected_reward)
        """
        chosen_reward = self.compute_reward(prompt, chosen)
        rejected_reward = self.compute_reward(prompt, rejected)

        # Bradley-Terry loss
        import math

        margin = chosen_reward - rejected_reward
        loss = -math.log(1.0 / (1.0 + math.exp(-margin)))

        return loss, chosen_reward, rejected_reward


class DPOLoss:
    """
    Direct Preference Optimization loss.

    Implements the DPO loss function that directly optimizes the policy
    on preference data without needing a separate reward model.
    """

    def __init__(self, beta: float = 0.1, reference_free: bool = False):
        """
        Initialize DPO loss.

        Args:
            beta: Temperature parameter
            reference_free: If True, skip reference model computation
        """
        self.beta = beta
        self.reference_free = reference_free

    def compute_loss(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        ref_chosen_logps: Optional[float] = None,
        ref_rejected_logps: Optional[float] = None,
    ) -> float:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log prob of chosen under policy
            policy_rejected_logps: Log prob of rejected under policy
            ref_chosen_logps: Log prob of chosen under reference
            ref_rejected_logps: Log prob of rejected under reference

        Returns:
            DPO loss value
        """
        import math

        if self.reference_free:
            # Reference-free DPO
            logits = self.beta * (policy_chosen_logps - policy_rejected_logps)
        else:
            # Standard DPO
            chosen_ratio = policy_chosen_logps - (ref_chosen_logps or 0)
            rejected_ratio = policy_rejected_logps - (ref_rejected_logps or 0)
            logits = self.beta * (chosen_ratio - rejected_ratio)

        # Log-sigmoid loss
        loss = -math.log(1.0 / (1.0 + math.exp(-logits)))
        return loss


class PreferenceTrainer:
    """
    Trainer for the preference learning pipeline.

    Orchestrates all three stages of RLHF training.
    """

    def __init__(
        self,
        config: PreferenceConfig,
        client: TrainingClient,
    ):
        """
        Initialize trainer.

        Args:
            config: Full preference learning config
            client: Training client
        """
        self.config = config
        self.client = client

        # Initialize components based on config
        self.reward_model = RewardModel(
            hidden_dim=config.reward_config.hidden_dim
        )

        if config.use_dpo:
            self.dpo_loss = DPOLoss(
                beta=config.dpo_config.beta,
                reference_free=config.dpo_config.reference_free,
            )
        else:
            self.dpo_loss = None

    async def run_sft_stage(self) -> Dict[str, Any]:
        """
        Run supervised fine-tuning stage.

        Returns:
            SFT training metrics
        """
        if self.config.skip_sft:
            logger.info("Skipping SFT stage")
            return {"skipped": True}

        logger.info("Starting SFT stage")
        sft_config = self.config.sft_config

        # Import and run chat SL trainer
        from .chat_sl import ChatSLConfig, ChatSLTrainer

        chat_config = ChatSLConfig(
            model_name=self.config.model_name,
            renderer_name=self.config.renderer_name,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            dataset=sft_config.dataset,
            dataset_split=sft_config.dataset_split,
            batch_size=sft_config.batch_size,
            learning_rate=sft_config.learning_rate,
            num_epochs=sft_config.num_epochs,
            max_steps=sft_config.max_steps,
        )

        trainer = ChatSLTrainer(chat_config, self.client)
        metrics = await trainer.train()

        logger.info(f"SFT stage complete. Final loss: {metrics.loss:.4f}")
        return metrics.to_dict()

    async def run_reward_model_stage(self) -> Dict[str, Any]:
        """
        Run reward model training stage.

        Returns:
            Reward model training metrics
        """
        if self.config.skip_reward:
            logger.info("Skipping reward model stage")
            return {"skipped": True}

        logger.info("Starting reward model training stage")
        rm_config = self.config.reward_config

        # Load preference dataset
        dataset = PreferenceDataset(
            dataset_name=rm_config.dataset,
            split=rm_config.dataset_split,
        )

        total_loss = 0.0
        num_steps = rm_config.max_steps or (len(dataset) // rm_config.batch_size)

        for step in range(num_steps):
            batch = dataset.get_batch(rm_config.batch_size)

            # Compute losses for batch
            batch_loss = 0.0
            for pair in batch:
                loss, _, _ = self.reward_model.compute_pairwise_loss(
                    pair.prompt, pair.chosen, pair.rejected
                )
                batch_loss += loss

            batch_loss /= len(batch)
            total_loss += batch_loss

            if step % self.config.log_steps == 0:
                logger.info(f"RM Step {step} | Loss: {batch_loss:.4f}")

        avg_loss = total_loss / num_steps
        logger.info(f"Reward model training complete. Avg loss: {avg_loss:.4f}")

        return {"avg_loss": avg_loss, "num_steps": num_steps}

    async def run_ppo_stage(self) -> Dict[str, Any]:
        """
        Run PPO optimization stage.

        Returns:
            PPO training metrics
        """
        logger.info("Starting PPO optimization stage")
        ppo_config = self.config.ppo_config

        # In a full implementation, would:
        # 1. Generate responses from policy
        # 2. Score with reward model
        # 3. Compute PPO loss and update

        total_reward = 0.0
        num_steps = ppo_config.max_steps

        for step in range(num_steps):
            # Placeholder for PPO step
            reward = 0.5  # Would come from actual rollouts

            total_reward += reward

            if step % self.config.log_steps == 0:
                avg_reward = total_reward / (step + 1)
                logger.info(f"PPO Step {step} | Avg Reward: {avg_reward:.4f}")

        final_reward = total_reward / num_steps
        logger.info(f"PPO optimization complete. Final avg reward: {final_reward:.4f}")

        return {"avg_reward": final_reward, "num_steps": num_steps}

    async def run_dpo_stage(self) -> Dict[str, Any]:
        """
        Run DPO optimization stage.

        Returns:
            DPO training metrics
        """
        logger.info("Starting DPO optimization stage")
        dpo_config = self.config.dpo_config

        # Load preference dataset
        dataset = PreferenceDataset(
            dataset_name=dpo_config.dataset,
            split=dpo_config.dataset_split,
        )

        total_loss = 0.0
        num_steps = dpo_config.max_steps or (len(dataset) // dpo_config.batch_size)

        for step in range(num_steps):
            batch = dataset.get_batch(dpo_config.batch_size)

            # Compute DPO losses
            batch_loss = 0.0
            for pair in batch:
                # In real implementation, would compute log probs
                policy_chosen_logps = -0.5
                policy_rejected_logps = -0.7

                loss = self.dpo_loss.compute_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                )
                batch_loss += loss

            batch_loss /= len(batch)
            total_loss += batch_loss

            if step % self.config.log_steps == 0:
                logger.info(f"DPO Step {step} | Loss: {batch_loss:.4f}")

        avg_loss = total_loss / num_steps
        logger.info(f"DPO optimization complete. Avg loss: {avg_loss:.4f}")

        return {"avg_loss": avg_loss, "num_steps": num_steps}

    async def train(self) -> Dict[str, Any]:
        """
        Run the full preference learning pipeline.

        Returns:
            Combined metrics from all stages
        """
        results = {}

        # Stage 1: SFT
        results["sft"] = await self.run_sft_stage()

        # Stage 2: Reward Model (skip for DPO)
        if not self.config.use_dpo:
            results["reward_model"] = await self.run_reward_model_stage()

        # Stage 3: Policy Optimization
        if self.config.use_dpo:
            results["dpo"] = await self.run_dpo_stage()
        else:
            results["ppo"] = await self.run_ppo_stage()

        logger.info("Preference learning pipeline complete!")
        return results


async def run_preference_learning(
    config: PreferenceConfig,
    client: TrainingClient,
) -> Dict[str, Any]:
    """
    Run the preference learning pipeline.

    Args:
        config: Preference learning configuration
        client: Training client

    Returns:
        Combined metrics from all stages
    """
    trainer = PreferenceTrainer(config, client)
    return await trainer.train()
