"""
RLVR Environment Protocol

Provides a lightweight Gymnasium-compatible environment abstraction for
RLVR training. Supports both single-turn (existing RewardFn) and
multi-turn (agent-style) interaction patterns.

Key features:
- Protocol-based interface (no heavy base class)
- Auto-wrapper for existing RewardFn callables
- Multi-turn state management with turn counting
- Environment registry for pluggable task types
- Composable reward shaping via environment wrappers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from .reward import RewardFn

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Observation returned by an environment."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result from an environment step."""

    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    """
    Protocol for RLVR environments.

    Follows the Gymnasium pattern (reset/step/close) adapted for
    text-based LLM interaction.
    """

    def reset(self, prompt: str, **kwargs: Any) -> Observation:
        """
        Reset the environment with a new prompt.

        Args:
            prompt: The input prompt/task
            **kwargs: Additional environment parameters

        Returns:
            Initial observation
        """
        ...

    def step(self, action: str) -> StepResult:
        """
        Take an action (model response) in the environment.

        Args:
            action: The model's response/action text

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...


# ==============================================================================
# Environment Registry
# ==============================================================================


_ENV_REGISTRY: Dict[str, Callable[..., Environment]] = {}


def register_env(
    name: str,
    factory: Optional[Callable[..., Environment]] = None,
    *,
    overwrite: bool = False,
) -> Any:
    """Register an environment factory. Can be used as a decorator."""

    def decorator(f: Callable[..., Environment]) -> Callable[..., Environment]:
        if name in _ENV_REGISTRY and not overwrite:
            raise ValueError(f"Environment '{name}' already registered")
        _ENV_REGISTRY[name] = f
        logger.debug(f"Registered environment: {name}")
        return f

    if factory is not None:
        return decorator(factory)
    return decorator


def make_env(name: str, **kwargs: Any) -> Environment:
    """Create an environment by name from the registry."""
    if name not in _ENV_REGISTRY:
        available = list(_ENV_REGISTRY.keys())
        raise ValueError(f"Unknown environment '{name}'. Available: {available}")
    return _ENV_REGISTRY[name](**kwargs)


def list_envs() -> List[str]:
    """List all registered environments."""
    return list(_ENV_REGISTRY.keys())


# ==============================================================================
# RewardFn Auto-Wrapper (single-turn environment)
# ==============================================================================


class SingleTurnEnv:
    """
    Wraps an existing RewardFn as a single-turn Environment.

    reset() returns the prompt as the observation.
    step() calls the reward function and returns done=True.

    This provides backward compatibility: existing reward functions
    work seamlessly with the new environment protocol.
    """

    def __init__(
        self,
        reward_fn: RewardFn,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.reward_fn = reward_fn
        self.default_meta = meta or {}
        self._prompt: Optional[str] = None
        self._done: bool = True

    def reset(self, prompt: str, **kwargs: Any) -> Observation:
        self._prompt = prompt
        self._done = False
        meta = {**self.default_meta, **kwargs}
        return Observation(text=prompt, metadata=meta)

    def step(self, action: str) -> StepResult:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() first.")
        if self._prompt is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        meta = {**self.default_meta}
        reward = self.reward_fn(
            prompt=self._prompt,
            response=action,
            meta=meta,
        )
        self._done = True

        return StepResult(
            observation=Observation(text="", metadata={}),
            reward=reward,
            done=True,
            info={"prompt": self._prompt, "response": action},
        )

    def close(self) -> None:
        pass


def wrap_reward_fn(
    reward_fn: RewardFn,
    meta: Optional[Dict[str, Any]] = None,
) -> SingleTurnEnv:
    """
    Wrap a RewardFn callable as a single-turn Environment.

    Args:
        reward_fn: The reward function to wrap
        meta: Optional default metadata to pass to the reward function

    Returns:
        A SingleTurnEnv wrapping the reward function
    """
    return SingleTurnEnv(reward_fn=reward_fn, meta=meta)


# ==============================================================================
# Multi-Turn Environment Base Class
# ==============================================================================


class MultiTurnEnv:
    """
    Base class for multi-turn environments.

    Provides turn counting, conversation history tracking, and
    configurable termination conditions.
    """

    def __init__(
        self,
        max_turns: int = 10,
        reward_on_done_only: bool = True,
    ):
        self.max_turns = max_turns
        self.reward_on_done_only = reward_on_done_only

        self._prompt: Optional[str] = None
        self._history: List[Dict[str, str]] = []
        self._turn: int = 0
        self._done: bool = True
        self._total_reward: float = 0.0

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history.copy()

    @property
    def is_done(self) -> bool:
        return self._done

    def reset(self, prompt: str, **kwargs: Any) -> Observation:
        self._prompt = prompt
        self._history = [{"role": "user", "content": prompt}]
        self._turn = 0
        self._done = False
        self._total_reward = 0.0
        return self._get_observation()

    def step(self, action: str) -> StepResult:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() first.")

        self._turn += 1
        self._history.append({"role": "assistant", "content": action})

        # Compute step reward
        reward = self._compute_reward(action)
        self._total_reward += reward

        # Check termination
        done = self._is_terminal(action) or self._turn >= self.max_turns
        truncated = self._turn >= self.max_turns and not self._is_terminal(action)
        self._done = done

        # Generate next observation (if not done)
        if not done:
            next_obs = self._get_next_observation(action)
            self._history.append({"role": "user", "content": next_obs.text})
        else:
            next_obs = Observation(text="", metadata={"total_reward": self._total_reward})

        # Decide which reward to return
        if self.reward_on_done_only:
            step_reward = self._total_reward if done else 0.0
        else:
            step_reward = reward

        return StepResult(
            observation=next_obs,
            reward=step_reward,
            done=done,
            truncated=truncated,
            info={
                "turn": self._turn,
                "total_reward": self._total_reward,
                "history_length": len(self._history),
            },
        )

    def close(self) -> None:
        pass

    def _compute_reward(self, action: str) -> float:
        """Override to implement step-wise reward computation."""
        return 0.0

    def _is_terminal(self, action: str) -> bool:
        """Override to implement custom termination conditions."""
        return False

    def _get_observation(self) -> Observation:
        """Get current observation from state."""
        return Observation(
            text=self._prompt or "",
            metadata={"turn": self._turn},
        )

    def _get_next_observation(self, action: str) -> Observation:
        """Generate the next observation after an action. Override for multi-turn."""
        return Observation(
            text=f"Continue from turn {self._turn}.",
            metadata={"turn": self._turn},
        )


# ==============================================================================
# Environment Wrappers
# ==============================================================================


class RewardShapingWrapper:
    """
    Wraps an environment with additional reward shaping.

    Applies a transformation to the base environment's rewards
    (e.g., scaling, clipping, bonus terms).
    """

    def __init__(
        self,
        env: Environment,
        reward_scale: float = 1.0,
        reward_clip: Optional[float] = None,
        reward_bonus_fn: Optional[Callable[[str, str, float], float]] = None,
    ):
        self.env = env
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.reward_bonus_fn = reward_bonus_fn

    def reset(self, prompt: str, **kwargs: Any) -> Observation:
        return self.env.reset(prompt, **kwargs)

    def step(self, action: str) -> StepResult:
        result = self.env.step(action)

        # Apply shaping
        reward = result.reward * self.reward_scale

        if self.reward_bonus_fn is not None:
            prompt = result.info.get("prompt", "")
            reward += self.reward_bonus_fn(prompt, action, reward)

        if self.reward_clip is not None:
            reward = max(-self.reward_clip, min(self.reward_clip, reward))

        return StepResult(
            observation=result.observation,
            reward=reward,
            done=result.done,
            truncated=result.truncated,
            info=result.info,
        )

    def close(self) -> None:
        self.env.close()


class TurnLimitWrapper:
    """Wraps an environment with an additional turn limit."""

    def __init__(self, env: Environment, max_turns: int):
        self.env = env
        self.max_turns = max_turns
        self._turn = 0

    def reset(self, prompt: str, **kwargs: Any) -> Observation:
        self._turn = 0
        return self.env.reset(prompt, **kwargs)

    def step(self, action: str) -> StepResult:
        self._turn += 1
        result = self.env.step(action)

        if self._turn >= self.max_turns and not result.done:
            return StepResult(
                observation=result.observation,
                reward=result.reward,
                done=True,
                truncated=True,
                info={**result.info, "truncated_by_wrapper": True},
            )

        return result

    def close(self) -> None:
        self.env.close()


# ==============================================================================
# Built-in Environment: Batch Environment Runner
# ==============================================================================


class BatchEnvRunner:
    """
    Runs multiple environments in parallel (single-turn).

    Manages a batch of environments, applying prompts and collecting
    rewards in a vectorized fashion. For single-turn environments,
    this is equivalent to batch reward computation.
    """

    def __init__(self, env_factory: Callable[[], Environment]):
        self.env_factory = env_factory

    def run_batch(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[StepResult]:
        """
        Run a batch of single-turn episodes.

        Args:
            prompts: Batch of prompts
            responses: Batch of model responses

        Returns:
            List of StepResults
        """
        results = []
        for prompt, response in zip(prompts, responses):
            env = self.env_factory()
            env.reset(prompt)
            result = env.step(response)
            env.close()
            results.append(result)
        return results

    def run_multi_turn(
        self,
        prompts: List[str],
        generate_fn: Callable[[str], str],
        max_turns: int = 10,
    ) -> List[List[StepResult]]:
        """
        Run multi-turn episodes with a generation function.

        Args:
            prompts: Batch of initial prompts
            generate_fn: Function that generates responses from observations
            max_turns: Maximum number of turns

        Returns:
            List of episode trajectories (list of StepResults per episode)
        """
        all_trajectories = []

        for prompt in prompts:
            env = self.env_factory()
            obs = env.reset(prompt)
            trajectory = []

            for _ in range(max_turns):
                action = generate_fn(obs.text)
                result = env.step(action)
                trajectory.append(result)
                if result.done:
                    break
                obs = result.observation

            env.close()
            all_trajectories.append(trajectory)

        return all_trajectories
