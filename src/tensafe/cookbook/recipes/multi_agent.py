"""
Multi-Agent Training Recipe.

Train language models through competitive or cooperative multi-agent
interactions. Supports self-play, arena-style evaluation, and
collaborative training scenarios.

Example usage:
    from tensafe.cookbook.recipes import MultiAgentConfig, run_multi_agent

    config = MultiAgentConfig(
        model_name="meta-llama/Llama-3.1-8B",
        num_agents=2,
        game_type="debate",
    )
    await run_multi_agent(config, client)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

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


class GameType(str, Enum):
    """Types of multi-agent games."""

    DEBATE = "debate"  # Two agents debate a topic
    NEGOTIATION = "negotiation"  # Agents negotiate an outcome
    QUIZ = "quiz"  # One agent asks, other answers
    ROLEPLAY = "roleplay"  # Collaborative storytelling
    ADVERSARIAL = "adversarial"  # One tries to trick the other
    COOPERATIVE = "cooperative"  # Work together on a task


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    name: str = "Agent"
    role: str = ""  # Role description
    system_prompt: str = ""
    temperature: float = 0.7
    is_trainable: bool = True  # Whether to update this agent's weights


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent training."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: Optional[str] = None

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: float = 64.0

    # Agent settings
    num_agents: int = 2
    agent_configs: List[AgentConfig] = field(default_factory=list)

    # Game settings
    game_type: GameType = GameType.DEBATE
    max_turns: int = 6  # Maximum turns per game
    max_tokens_per_turn: int = 256

    # Training settings
    batch_size: int = 32  # Number of games per batch
    learning_rate: float = 2e-5
    max_steps: int = 1000

    # Reward settings
    use_judge: bool = True  # Use a judge to score outcomes
    judge_prompt: str = ""  # Custom judge prompt
    win_reward: float = 1.0
    lose_reward: float = -1.0
    draw_reward: float = 0.0

    # Self-play settings
    self_play: bool = True  # Both agents are the same model
    elo_tracking: bool = True  # Track ELO ratings

    # Checkpointing
    checkpoint_dir: str = "/tmp/tensafe-multi-agent"
    save_steps: int = 100

    # Logging
    log_steps: int = 10

    def __post_init__(self):
        if self.renderer_name is None:
            self.renderer_name = get_recommended_renderer_name(self.model_name)

        # Create default agent configs if not provided
        if not self.agent_configs:
            self.agent_configs = [
                AgentConfig(name=f"Agent_{i}", is_trainable=(i == 0))
                for i in range(self.num_agents)
            ]

    @property
    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


@dataclass
class GameTurn:
    """A single turn in a game."""

    agent_idx: int
    agent_name: str
    message: str
    timestamp: int = 0


@dataclass
class GameResult:
    """Result of a game."""

    turns: List[GameTurn]
    winner_idx: Optional[int] = None  # None for draw
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_draw(self) -> bool:
        return self.winner_idx is None


class GameEnvironment:
    """
    Environment for multi-agent games.

    Manages turn taking, game state, and outcome determination.
    """

    # Default prompts for different game types
    GAME_PROMPTS = {
        GameType.DEBATE: {
            "system": "You are participating in a debate. Argue your position clearly and persuasively.",
            "topics": [
                "AI will be net positive for humanity",
                "Remote work is better than office work",
                "Electric cars are the future",
            ],
        },
        GameType.NEGOTIATION: {
            "system": "You are negotiating a deal. Try to get the best outcome for yourself while reaching an agreement.",
            "scenarios": [
                "Splitting a prize",
                "Setting a price",
                "Dividing responsibilities",
            ],
        },
        GameType.QUIZ: {
            "system": "You are playing a trivia game.",
            "categories": ["Science", "History", "Geography"],
        },
        GameType.ADVERSARIAL: {
            "system": "One agent tries to get the other to agree with false information.",
            "challenges": ["Convince them of a fake fact", "Get them to contradict themselves"],
        },
        GameType.COOPERATIVE: {
            "system": "Work together to solve a problem or complete a task.",
            "tasks": ["Write a story together", "Solve a puzzle", "Plan an event"],
        },
    }

    def __init__(
        self,
        game_type: GameType,
        agent_configs: List[AgentConfig],
        max_turns: int = 6,
    ):
        """
        Initialize game environment.

        Args:
            game_type: Type of game to play
            agent_configs: Configuration for each agent
            max_turns: Maximum turns per game
        """
        self.game_type = game_type
        self.agent_configs = agent_configs
        self.max_turns = max_turns

        self.turns: List[GameTurn] = []
        self.current_turn = 0
        self.topic: Optional[str] = None

    def reset(self, topic: Optional[str] = None) -> str:
        """
        Reset the environment for a new game.

        Args:
            topic: Optional topic/scenario for the game

        Returns:
            Initial prompt for the first agent
        """
        self.turns = []
        self.current_turn = 0

        # Select topic
        if topic:
            self.topic = topic
        else:
            prompts = self.GAME_PROMPTS.get(self.game_type, {})
            topics = prompts.get("topics", prompts.get("scenarios", prompts.get("tasks", ["General topic"])))
            self.topic = random.choice(topics)

        return self._get_initial_prompt()

    def _get_initial_prompt(self) -> str:
        """Get the initial prompt for the game."""
        prompts = self.GAME_PROMPTS.get(self.game_type, {})
        system = prompts.get("system", "You are playing a game.")

        if self.game_type == GameType.DEBATE:
            return f"{system}\n\nTopic: {self.topic}\n\nYou are arguing FOR this position."
        elif self.game_type == GameType.NEGOTIATION:
            return f"{system}\n\nScenario: {self.topic}"
        elif self.game_type == GameType.COOPERATIVE:
            return f"{system}\n\nTask: {self.topic}"
        else:
            return f"{system}\n\n{self.topic}"

    def get_current_agent_idx(self) -> int:
        """Get the index of the agent whose turn it is."""
        return self.current_turn % len(self.agent_configs)

    def step(self, message: str) -> Tuple[str, bool]:
        """
        Process a turn in the game.

        Args:
            message: Message from current agent

        Returns:
            Tuple of (next_prompt, is_done)
        """
        agent_idx = self.get_current_agent_idx()
        agent_name = self.agent_configs[agent_idx].name

        # Record turn
        turn = GameTurn(
            agent_idx=agent_idx,
            agent_name=agent_name,
            message=message,
            timestamp=self.current_turn,
        )
        self.turns.append(turn)
        self.current_turn += 1

        # Check if game is done
        is_done = self.current_turn >= self.max_turns

        if is_done:
            return "", True

        # Build prompt for next agent
        next_prompt = self._build_context_prompt()
        return next_prompt, False

    def _build_context_prompt(self) -> str:
        """Build prompt with conversation history for next agent."""
        history = "\n".join(
            f"{t.agent_name}: {t.message}" for t in self.turns
        )
        return f"Previous conversation:\n{history}\n\nYour response:"

    def get_result(self) -> GameResult:
        """Get the game result."""
        return GameResult(
            turns=self.turns,
            metadata={"topic": self.topic, "game_type": self.game_type.value},
        )


class GameJudge:
    """
    Judges game outcomes using an LLM.

    Determines winners and assigns scores based on game performance.
    """

    DEFAULT_JUDGE_PROMPT = """You are a fair judge evaluating a {game_type} between agents.

Review the conversation and determine:
1. Which agent performed better (or if it's a draw)
2. A score from 0-10 for each agent

Conversation:
{conversation}

Respond in this format:
Winner: [Agent_0 / Agent_1 / Draw]
Agent_0 Score: [0-10]
Agent_1 Score: [0-10]
Reasoning: [Brief explanation]"""

    def __init__(
        self,
        client: SamplingClient,
        custom_prompt: Optional[str] = None,
    ):
        """
        Initialize judge.

        Args:
            client: Sampling client for LLM judge
            custom_prompt: Custom judge prompt template
        """
        self.client = client
        self.prompt_template = custom_prompt or self.DEFAULT_JUDGE_PROMPT

    def judge(self, game: GameResult) -> Tuple[Optional[int], List[float]]:
        """
        Judge a game and determine winner.

        Args:
            game: Game result to judge

        Returns:
            Tuple of (winner_idx, scores)
        """
        # Build conversation string
        conversation = "\n".join(
            f"{t.agent_name}: {t.message}" for t in game.turns
        )

        # Build judge prompt
        prompt = self.prompt_template.format(
            game_type=game.metadata.get("game_type", "game"),
            conversation=conversation,
        )

        # Get judge's verdict
        result = self.client.sample(
            prompts=[prompt],
            max_tokens=256,
            temperature=0.3,
        )

        # Parse result
        if hasattr(result, "samples"):
            response = result.samples[0].completion
        else:
            response = result.get("samples", [{}])[0].get("completion", "")

        return self._parse_verdict(response)

    def _parse_verdict(self, response: str) -> Tuple[Optional[int], List[float]]:
        """Parse judge's verdict from response."""
        import re

        # Extract winner
        winner_match = re.search(r"Winner:\s*(Agent_\d|Draw)", response, re.IGNORECASE)
        if winner_match:
            winner_str = winner_match.group(1)
            if winner_str.lower() == "draw":
                winner_idx = None
            else:
                winner_idx = int(winner_str.split("_")[1])
        else:
            winner_idx = None

        # Extract scores
        scores = [5.0, 5.0]  # Default scores
        for i in range(2):
            score_match = re.search(
                rf"Agent_{i}\s*Score:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE
            )
            if score_match:
                scores[i] = float(score_match.group(1))

        return winner_idx, scores


class MultiAgentTrainer:
    """
    Trainer for multi-agent scenarios.

    Manages game playing, outcome evaluation, and policy updates.
    """

    def __init__(
        self,
        config: MultiAgentConfig,
        training_client: TrainingClient,
        sampling_client: SamplingClient,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            training_client: Client for training
            sampling_client: Client for sampling responses
        """
        self.config = config
        self.training_client = training_client
        self.sampling_client = sampling_client

        # Initialize components
        self.environment = GameEnvironment(
            game_type=config.game_type,
            agent_configs=config.agent_configs,
            max_turns=config.max_turns,
        )

        if config.use_judge:
            self.judge = GameJudge(
                client=sampling_client,
                custom_prompt=config.judge_prompt or None,
            )
        else:
            self.judge = None

        # ELO tracking
        self.elo_ratings = {i: 1500.0 for i in range(config.num_agents)}

    def play_game(self) -> GameResult:
        """
        Play a single game between agents.

        Returns:
            Game result
        """
        # Reset environment
        initial_prompt = self.environment.reset()

        # Play turns
        prompt = initial_prompt
        while True:
            agent_idx = self.environment.get_current_agent_idx()
            agent_config = self.config.agent_configs[agent_idx]

            # Build full prompt
            full_prompt = f"{agent_config.system_prompt}\n\n{prompt}"

            # Get agent response
            result = self.sampling_client.sample(
                prompts=[full_prompt],
                max_tokens=self.config.max_tokens_per_turn,
                temperature=agent_config.temperature,
            )

            if hasattr(result, "samples"):
                response = result.samples[0].completion
            else:
                response = result.get("samples", [{}])[0].get("completion", "")

            # Step environment
            next_prompt, is_done = self.environment.step(response)

            if is_done:
                break

            prompt = next_prompt

        # Get result
        game = self.environment.get_result()

        # Judge if enabled
        if self.judge:
            winner_idx, scores = self.judge.judge(game)
            game.winner_idx = winner_idx
            game.scores = scores

            # Update ELO
            if self.config.elo_tracking and winner_idx is not None:
                self._update_elo(winner_idx)

        return game

    def _update_elo(self, winner_idx: int) -> None:
        """Update ELO ratings based on game outcome."""
        K = 32  # ELO K-factor

        for i in range(self.config.num_agents):
            if i == winner_idx:
                continue

            # Calculate expected scores
            r1 = self.elo_ratings[winner_idx]
            r2 = self.elo_ratings[i]

            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

            # Update ratings
            self.elo_ratings[winner_idx] += K * (1 - e1)
            self.elo_ratings[i] += K * (0 - e2)

    def compute_rewards(
        self, games: List[GameResult]
    ) -> List[Tuple[int, float]]:
        """
        Compute rewards for training from game results.

        Args:
            games: List of game results

        Returns:
            List of (agent_idx, reward) tuples
        """
        rewards = []

        for game in games:
            if game.winner_idx is not None:
                # Winner gets positive reward
                rewards.append((game.winner_idx, self.config.win_reward))
                # Loser gets negative reward
                loser_idx = 1 - game.winner_idx
                rewards.append((loser_idx, self.config.lose_reward))
            else:
                # Draw
                for i in range(self.config.num_agents):
                    rewards.append((i, self.config.draw_reward))

        return rewards

    async def train(self) -> Dict[str, Any]:
        """
        Run multi-agent training.

        Returns:
            Training metrics
        """
        logger.info("Starting multi-agent training")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Game type: {self.config.game_type.value}")
        logger.info(f"  Agents: {[a.name for a in self.config.agent_configs]}")

        total_games = 0
        wins = {i: 0 for i in range(self.config.num_agents)}
        draws = 0

        for step in range(self.config.max_steps):
            # Play batch of games
            games = [self.play_game() for _ in range(self.config.batch_size)]
            total_games += len(games)

            # Track stats
            for game in games:
                if game.winner_idx is not None:
                    wins[game.winner_idx] += 1
                else:
                    draws += 1

            # Compute rewards and update
            rewards = self.compute_rewards(games)

            # In full implementation, would build training batches from
            # game trajectories and update trainable agents

            # Log
            if step % self.config.log_steps == 0:
                win_rates = {
                    i: wins[i] / max(1, total_games)
                    for i in range(self.config.num_agents)
                }
                logger.info(
                    f"Step {step} | Games: {total_games} | "
                    f"Win rates: {win_rates} | ELO: {self.elo_ratings}"
                )

            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.training_client.save_state(
                    metadata={
                        "step": step,
                        "elo_ratings": self.elo_ratings,
                        "total_games": total_games,
                    }
                )

        logger.info("Multi-agent training complete!")
        return {
            "total_games": total_games,
            "wins": wins,
            "draws": draws,
            "final_elo": self.elo_ratings,
        }


async def run_multi_agent(
    config: MultiAgentConfig,
    training_client: TrainingClient,
    sampling_client: SamplingClient,
) -> Dict[str, Any]:
    """
    Run multi-agent training.

    Args:
        config: Training configuration
        training_client: Client for training
        sampling_client: Client for agent responses

    Returns:
        Training metrics
    """
    trainer = MultiAgentTrainer(config, training_client, sampling_client)
    return await trainer.train()
