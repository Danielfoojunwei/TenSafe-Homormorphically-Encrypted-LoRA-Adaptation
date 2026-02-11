"""Tests for cookbook completers."""

from typing import Any, List
from unittest.mock import MagicMock

import pytest


class MockSamplingClient:
    """Mock sampling client for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Test response"]
        self._call_count = 0

    def sample(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Any:
        """Mock sample method."""
        samples = []
        for i, prompt in enumerate(prompts):
            response = self.responses[i % len(self.responses)]
            samples.append(MagicMock(
                prompt=prompt,
                completion=response,
            ))
        self._call_count += 1
        return MagicMock(samples=samples)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 32000
        self.unk_token_id = 0
        self._counter = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = list(range(self._counter, self._counter + len(text.split())))
        self._counter += len(tokens)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"decoded_{len(tokens)}"


class TestTokensWithLogprobs:
    """Tests for TokensWithLogprobs dataclass."""

    def test_creation(self):
        """Test TokensWithLogprobs creation."""
        from tensafe.cookbook.completers import TokensWithLogprobs

        twl = TokensWithLogprobs(
            tokens=[1, 2, 3],
            logprobs=[-0.5, -0.3, -0.4],
            cumulative_logprob=-1.2,
        )

        assert twl.tokens == [1, 2, 3]
        assert len(twl) == 3
        assert twl.cumulative_logprob == -1.2

    def test_mean_logprob(self):
        """Test mean log probability calculation."""
        from tensafe.cookbook.completers import TokensWithLogprobs

        twl = TokensWithLogprobs(
            tokens=[1, 2, 3],
            logprobs=[-0.6, -0.3, -0.3],
        )

        assert twl.mean_logprob == pytest.approx(-0.4)

    def test_mean_logprob_empty(self):
        """Test mean log probability for empty sequence."""
        from tensafe.cookbook.completers import TokensWithLogprobs

        twl = TokensWithLogprobs(tokens=[], logprobs=[])
        assert twl.mean_logprob == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from tensafe.cookbook.completers import TokensWithLogprobs

        twl = TokensWithLogprobs(
            tokens=[1, 2],
            logprobs=[-0.5, -0.5],
            cumulative_logprob=-1.0,
        )

        d = twl.to_dict()
        assert d["tokens"] == [1, 2]
        assert d["logprobs"] == [-0.5, -0.5]
        assert d["cumulative_logprob"] == -1.0


class TestMockTokenCompleter:
    """Tests for MockTokenCompleter."""

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test mock token completion."""
        from tensafe.cookbook.completers import MockTokenCompleter

        tokenizer = MockTokenizer()
        completer = MockTokenCompleter(tokenizer, response_length=10)

        result = await completer.complete([1, 2, 3])

        assert len(result.tokens) == 10
        assert len(result.logprobs) == 10

    @pytest.mark.asyncio
    async def test_complete_batch(self):
        """Test mock batch completion."""
        from tensafe.cookbook.completers import MockTokenCompleter

        tokenizer = MockTokenizer()
        completer = MockTokenCompleter(tokenizer, response_length=5)

        results = await completer.complete_batch(
            [[1, 2], [3, 4], [5, 6]]
        )

        assert len(results) == 3
        for result in results:
            assert len(result.tokens) == 5

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self):
        """Test that mock completer is deterministic with same seed."""
        from tensafe.cookbook.completers import MockTokenCompleter

        tokenizer = MockTokenizer()
        completer1 = MockTokenCompleter(tokenizer, seed=42)
        completer2 = MockTokenCompleter(tokenizer, seed=42)

        result1 = await completer1.complete([1, 2, 3])
        result2 = await completer2.complete([1, 2, 3])

        assert result1.tokens == result2.tokens


class TestMockMessageCompleter:
    """Tests for MockMessageCompleter."""

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test mock message completion."""
        from tensafe.cookbook.completers import MockMessageCompleter
        from tensafe.cookbook.renderers.base import Message

        completer = MockMessageCompleter()

        messages = [
            Message(role="user", content="Hello!"),
        ]

        result = await completer.complete(messages)

        assert result.role == "assistant"
        assert isinstance(result.content, str)

    @pytest.mark.asyncio
    async def test_complete_with_custom_responses(self):
        """Test mock completer with custom responses."""
        from tensafe.cookbook.completers import MockMessageCompleter

        custom_responses = ["Response 1", "Response 2"]
        completer = MockMessageCompleter(responses=custom_responses)

        result1 = await completer.complete([{"role": "user", "content": "Hi"}])
        result2 = await completer.complete([{"role": "user", "content": "Hello"}])

        assert result1.content == "Response 1"
        assert result2.content == "Response 2"

    @pytest.mark.asyncio
    async def test_complete_batch(self):
        """Test mock batch message completion."""
        from tensafe.cookbook.completers import MockMessageCompleter

        completer = MockMessageCompleter()

        messages_batch = [
            [{"role": "user", "content": "Q1"}],
            [{"role": "user", "content": "Q2"}],
        ]

        results = await completer.complete_batch(messages_batch)

        assert len(results) == 2
        for result in results:
            assert result.role == "assistant"


class TestTinkerTokenCompleter:
    """Tests for TinkerTokenCompleter."""

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test token completion with mock client."""
        from tensafe.cookbook.completers import TinkerTokenCompleter

        client = MockSamplingClient(["Generated text"])
        tokenizer = MockTokenizer()

        completer = TinkerTokenCompleter(client, tokenizer)

        result = await completer.complete([1, 2, 3], max_tokens=10)

        assert len(result.tokens) > 0
        assert client._call_count == 1

    @pytest.mark.asyncio
    async def test_complete_batch(self):
        """Test batch token completion."""
        from tensafe.cookbook.completers import TinkerTokenCompleter

        client = MockSamplingClient(["Response 1", "Response 2"])
        tokenizer = MockTokenizer()

        completer = TinkerTokenCompleter(client, tokenizer)

        results = await completer.complete_batch(
            [[1, 2], [3, 4]],
            max_tokens=10,
        )

        assert len(results) == 2


class TestTinkerMessageCompleter:
    """Tests for TinkerMessageCompleter."""

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test message completion with mock client."""
        from tensafe.cookbook.completers import TinkerMessageCompleter
        from tensafe.cookbook.renderers.role_colon import RoleColonRenderer

        client = MockSamplingClient(["Hello! How can I help?"])
        tokenizer = MockTokenizer()
        renderer = RoleColonRenderer(tokenizer)

        completer = TinkerMessageCompleter(client, renderer)

        result = await completer.complete([{"role": "user", "content": "Hi!"}])

        assert result.role == "assistant"
        assert client._call_count == 1
