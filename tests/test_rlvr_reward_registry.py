"""
Tests for RLVR reward registry and resolution.

Verifies that reward functions can be:
- Resolved from callable
- Resolved from registry name
- Resolved from dotted path
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add tensafe to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.rlvr.reward import (
    get_registered_rewards,
    register_reward,
    resolve_reward,
)


class TestRewardFnProtocol:
    """Tests for RewardFn protocol compliance."""

    def test_callable_matches_protocol(self):
        """Test that a simple callable matches the protocol."""

        def my_reward(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 1.0

        # Should be usable as RewardFn
        result = my_reward("test prompt", "test response")
        assert result == 1.0

    def test_class_based_reward(self):
        """Test class-based reward function."""

        class MyReward:
            def __init__(self, scale: float = 1.0):
                self.scale = scale

            def __call__(
                self,
                prompt: str,
                response: str,
                meta: Optional[Dict[str, Any]] = None,
            ) -> float:
                return self.scale * len(response)

        reward_fn = MyReward(scale=0.1)
        result = reward_fn("prompt", "hello")
        assert result == 0.5  # 5 chars * 0.1


class TestResolveRewardFromCallable:
    """Tests for resolving reward from callable."""

    def test_resolve_simple_callable(self):
        """Test resolving from a simple function."""

        def constant_reward(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 1.0

        resolved = resolve_reward(constant_reward)
        assert resolved("prompt", "response") == 1.0

    def test_resolve_callable_with_kwargs(self):
        """Test resolving callable with default kwargs."""

        def scaled_reward(
            prompt: str,
            response: str,
            meta: Optional[Dict[str, Any]] = None,
            scale: float = 1.0,
        ) -> float:
            return scale * len(response)

        resolved = resolve_reward(scaled_reward, scale=2.0)
        result = resolved("prompt", "hello")  # 5 chars
        assert result == 10.0  # 5 * 2.0

    def test_resolve_lambda(self):
        """Test resolving from lambda function."""
        resolved = resolve_reward(lambda p, r, m=None: 0.5)
        assert resolved("p", "r") == 0.5


class TestResolveRewardFromRegistry:
    """Tests for resolving reward from registry name."""

    def test_resolve_keyword_contains(self):
        """Test resolving keyword_contains reward."""
        resolved = resolve_reward("keyword_contains", keywords=["magic"])

        # Response with keyword
        result = resolved("prompt", "This has the magic word")
        assert result > 0

        # Response without keyword
        result = resolved("prompt", "This has no special word")
        assert result < 0

    def test_resolve_length_penalty(self):
        """Test resolving length_penalty reward."""
        resolved = resolve_reward(
            "length_penalty", target_length=5, penalty_scale=0.1
        )

        # Short response (2 words) - below target
        result = resolved("prompt", "hello world")
        assert result > 0  # Close to target gets positive reward

        # Long response (10 words) - above target
        result = resolved("prompt", "one two three four five six seven eight nine ten")
        assert result < 1.0  # Penalized for being too long

    def test_resolve_format_compliance_json(self):
        """Test resolving format_compliance reward for JSON."""
        resolved = resolve_reward("format_compliance", required_format="json")

        # Response with JSON structure
        result = resolved("prompt", '{"answer": 42}')
        assert result == 1.0

        # Response without JSON structure
        result = resolved("prompt", "no json here")
        assert result == -0.5

    def test_resolve_format_compliance_numbered_list(self):
        """Test resolving format_compliance reward for numbered list."""
        resolved = resolve_reward("format_compliance", required_format="numbered_list")

        # Response with numbered list
        result = resolved("prompt", "1. First item\n2. Second item\n3. Third item")
        assert result == 1.0

        # Response without numbered list
        result = resolved("prompt", "Just plain text")
        assert result == 0.0

    def test_unknown_registry_name_raises(self):
        """Test that unknown registry name raises error."""
        with pytest.raises(ValueError, match="Cannot resolve reward"):
            resolve_reward("nonexistent_reward")


class TestResolveRewardFromDottedPath:
    """Tests for resolving reward from dotted path."""

    def test_resolve_from_dotted_path(self):
        """Test resolving from dotted module path."""
        # This should resolve the keyword_contains_reward function from tensafe.rlvr.reward
        resolved = resolve_reward(
            "tensafe.rlvr.reward:keyword_contains_reward", keywords=["test"]
        )

        result = resolved("prompt", "this is a test")
        assert result > 0

    def test_resolve_invalid_dotted_path_raises(self):
        """Test that invalid module path raises error."""
        with pytest.raises(ValueError, match="Cannot resolve reward"):
            resolve_reward("nonexistent.module:function")

    def test_resolve_invalid_attribute_raises(self):
        """Test that invalid attribute raises error."""
        with pytest.raises(ValueError, match="Cannot resolve reward"):
            resolve_reward("tensafe.rlvr.reward:nonexistent_function")


class TestRegisterReward:
    """Tests for reward registration."""

    def test_register_custom_reward(self):
        """Test registering a custom reward function."""

        def my_custom_reward(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 42.0

        register_reward("test_custom_42", my_custom_reward, overwrite=True)

        # Should now be resolvable
        resolved = resolve_reward("test_custom_42")
        assert resolved("p", "r") == 42.0

    def test_register_overwrites_with_flag(self):
        """Test that registering overwrites existing entry with flag."""

        def reward_v1(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 1.0

        def reward_v2(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 2.0

        register_reward("test_overwrite", reward_v1, overwrite=True)
        assert resolve_reward("test_overwrite")("p", "r") == 1.0

        register_reward("test_overwrite", reward_v2, overwrite=True)
        assert resolve_reward("test_overwrite")("p", "r") == 2.0

    def test_register_without_overwrite_raises(self):
        """Test that registering without overwrite flag raises."""

        def reward_fn(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 1.0

        # First registration should work
        register_reward("test_no_overwrite", reward_fn, overwrite=True)

        # Second registration without overwrite should raise
        with pytest.raises(ValueError, match="already registered"):
            register_reward("test_no_overwrite", reward_fn, overwrite=False)

    def test_get_registered_rewards(self):
        """Test listing registered rewards."""
        rewards = get_registered_rewards()

        # Should have built-in rewards
        assert "keyword_contains" in rewards
        assert "length_penalty" in rewards
        assert "format_compliance" in rewards
        assert "composite" in rewards


class TestCompositeReward:
    """Tests for composite reward functions."""

    def test_resolve_composite_reward(self):
        """Test resolving and using composite reward."""
        resolved = resolve_reward(
            "composite",
            rewards=[
                {"name": "keyword_contains", "weight": 0.5, "kwargs": {"keywords": ["magic"]}},
                {"name": "length_penalty", "weight": 0.5, "kwargs": {"target_length": 5}},
            ],
        )

        # Has "magic" keyword, length near target
        result = resolved("prompt", "magic word")
        # Should be positive (has keyword, reasonable length)
        assert result > 0

    def test_composite_empty_specs(self):
        """Test composite with empty specs returns zero."""
        resolved = resolve_reward("composite", rewards=[])
        result = resolved("prompt", "response")
        assert result == 0.0

    def test_composite_weighted_average(self):
        """Test that composite computes weighted average correctly."""
        # Create rewards with known outputs
        def reward_a(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return 1.0

        def reward_b(
            prompt: str, response: str, meta: Optional[Dict[str, Any]] = None
        ) -> float:
            return -1.0

        register_reward("test_reward_a", reward_a, overwrite=True)
        register_reward("test_reward_b", reward_b, overwrite=True)

        # Equal weights should give average of 0
        resolved = resolve_reward(
            "composite",
            rewards=[
                {"name": "test_reward_a", "weight": 1.0},
                {"name": "test_reward_b", "weight": 1.0},
            ],
        )
        result = resolved("p", "r")
        assert abs(result - 0.0) < 1e-6


class TestRewardFunctionBehavior:
    """Tests for reward function behavior."""

    def test_keyword_multiple_keywords(self):
        """Test keyword_contains with multiple keywords."""
        resolved = resolve_reward("keyword_contains", keywords=["foo", "bar", "baz"])

        # Has one keyword - returns positive reward
        result = resolved("prompt", "contains foo somewhere")
        assert result > 0

        # Has no keywords - returns negative reward
        result = resolved("prompt", "nothing special")
        assert result < 0

    def test_keyword_case_insensitive(self):
        """Test keyword matching is case insensitive."""
        resolved = resolve_reward("keyword_contains", keywords=["MAGIC"])

        result = resolved("prompt", "This has magic in lowercase")
        assert result > 0

    def test_metadata_keywords(self):
        """Test that keywords can come from metadata."""
        resolved = resolve_reward("keyword_contains")

        meta = {"keywords": ["secret"]}
        result = resolved("prompt", "the secret word", meta)
        assert result > 0


class TestRewardEdgeCases:
    """Tests for edge cases in reward computation."""

    def test_empty_response(self):
        """Test reward with empty response."""
        resolved = resolve_reward("length_penalty", target_length=10)
        result = resolved("prompt", "")
        # Empty is 0 words, far from target 10
        assert result < 1.0

    def test_empty_keywords(self):
        """Test keyword_contains with no keywords."""
        resolved = resolve_reward("keyword_contains", keywords=[])

        # With no keywords, checks for non-empty response
        result = resolved("prompt", "some response")
        assert result > 0

        result = resolved("prompt", "   ")  # Whitespace only
        assert result < 0

    def test_format_compliance_no_format(self):
        """Test format_compliance with no format specified."""
        resolved = resolve_reward("format_compliance")
        result = resolved("prompt", "any response")
        assert result == 0.0  # No format requirement


class TestRewardModelWrapper:
    """Tests for RewardModelWrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        from tensafe.rlvr.reward import RewardModelWrapper

        wrapper = RewardModelWrapper(
            model_path="/path/to/model",
            device="cpu",
            batch_size=4,
        )

        assert wrapper.model_path == "/path/to/model"
        assert wrapper.device == "cpu"
        assert wrapper.batch_size == 4

    def test_wrapper_call(self):
        """Test wrapper can compute reward."""
        from tensafe.rlvr.reward import RewardModelWrapper

        wrapper = RewardModelWrapper(model_path="/mock/path")

        # Should return a float reward
        result = wrapper("prompt", "response")
        assert isinstance(result, float)

    def test_wrapper_batch(self):
        """Test wrapper batch reward computation."""
        from tensafe.rlvr.reward import RewardModelWrapper

        wrapper = RewardModelWrapper(model_path="/mock/path")

        prompts = ["p1", "p2", "p3"]
        responses = ["r1", "r2", "r3"]
        results = wrapper.batch_reward(prompts, responses)

        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
