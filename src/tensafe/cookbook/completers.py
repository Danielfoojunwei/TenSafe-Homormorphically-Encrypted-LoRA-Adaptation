"""
Completers for sampling from language models at different abstraction levels.

TokenCompleter operates on tokens for RL algorithms, while MessageCompleter
works with messages and requires a renderer.

These provide clean abstractions for generating text completions during
training and evaluation, with support for async operations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from .renderers.base import Message, Renderer, StopCondition


@dataclass
class TokensWithLogprobs:
    """
    Token sequence with associated log probabilities.

    Used for RL algorithms that need token-level probability information.
    """

    tokens: List[int]
    logprobs: List[float] = field(default_factory=list)
    cumulative_logprob: float = 0.0

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def mean_logprob(self) -> float:
        """Mean log probability per token."""
        if not self.logprobs:
            return 0.0
        return sum(self.logprobs) / len(self.logprobs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens": self.tokens,
            "logprobs": self.logprobs,
            "cumulative_logprob": self.cumulative_logprob,
        }


class SamplingClient(Protocol):
    """Protocol for clients that can generate samples."""

    def sample(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: Optional[List[str]] = None,
    ) -> Any:
        """Generate samples from prompts."""
        ...


class AsyncSamplingClient(Protocol):
    """Protocol for async sampling clients."""

    async def sample_async(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: Optional[List[str]] = None,
    ) -> Any:
        """Generate samples from prompts asynchronously."""
        ...


class TokenCompleter(ABC):
    """
    Abstract base class for token-level sampling.

    TokenCompleters operate directly on token IDs, making them
    suitable for RL algorithms that need fine-grained control.
    """

    @abstractmethod
    async def complete(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 128,
        temperature: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> TokensWithLogprobs:
        """
        Complete a token sequence.

        Args:
            prompt_tokens: Input token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Token IDs that stop generation

        Returns:
            TokensWithLogprobs with generated tokens and logprobs
        """
        pass

    @abstractmethod
    async def complete_batch(
        self,
        prompt_tokens_batch: List[List[int]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[TokensWithLogprobs]:
        """
        Complete a batch of token sequences.

        Args:
            prompt_tokens_batch: List of input token ID lists
            max_tokens: Maximum tokens to generate per sequence
            temperature: Sampling temperature
            stop_tokens: Token IDs that stop generation

        Returns:
            List of TokensWithLogprobs for each input
        """
        pass


class MessageCompleter(ABC):
    """
    Abstract base class for message-level completion.

    MessageCompleters work with structured Message objects and
    use a Renderer to handle formatting.
    """

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Union[Message, Dict[str, Any]]],
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Message:
        """
        Complete a conversation with an assistant response.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated assistant Message
        """
        pass

    @abstractmethod
    async def complete_batch(
        self,
        messages_batch: List[Sequence[Union[Message, Dict[str, Any]]]],
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> List[Message]:
        """
        Complete a batch of conversations.

        Args:
            messages_batch: List of conversation histories
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            List of generated assistant Messages
        """
        pass


class TinkerTokenCompleter(TokenCompleter):
    """
    Token completer using the TenSafe/Tinker sampling API.

    Wraps a SamplingClient to provide token-level completions
    with log probability tracking.
    """

    def __init__(
        self,
        client: SamplingClient,
        tokenizer: Any,
        default_max_tokens: int = 128,
        default_temperature: float = 1.0,
    ):
        """
        Initialize the token completer.

        Args:
            client: SamplingClient for API calls
            tokenizer: Tokenizer for encoding/decoding
            default_max_tokens: Default maximum tokens
            default_temperature: Default temperature
        """
        self.client = client
        self.tokenizer = tokenizer
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

    async def complete(
        self,
        prompt_tokens: List[int],
        max_tokens: int = None,
        temperature: float = None,
        stop_tokens: Optional[List[int]] = None,
    ) -> TokensWithLogprobs:
        """Complete a token sequence."""
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Decode prompt to text for the API
        prompt_text = self.tokenizer.decode(prompt_tokens)

        # Convert stop tokens to strings if provided
        stop_sequences = None
        if stop_tokens:
            stop_sequences = [self.tokenizer.decode([t]) for t in stop_tokens]

        # Call the sampling API
        result = self.client.sample(
            prompts=[prompt_text],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            stop_sequences=stop_sequences,
        )

        # Extract result
        if hasattr(result, "samples"):
            sample = result.samples[0]
            completion = sample.completion if hasattr(sample, "completion") else sample.get("completion", "")
        else:
            completion = result.get("samples", [{}])[0].get("completion", "")

        # Encode completion to tokens
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

        # Create mock logprobs (real implementation would get from model)
        logprobs = [-0.5] * len(completion_tokens)

        return TokensWithLogprobs(
            tokens=completion_tokens,
            logprobs=logprobs,
            cumulative_logprob=sum(logprobs),
        )

    async def complete_batch(
        self,
        prompt_tokens_batch: List[List[int]],
        max_tokens: int = None,
        temperature: float = None,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[TokensWithLogprobs]:
        """Complete a batch of token sequences."""
        # Run completions concurrently
        tasks = [
            self.complete(tokens, max_tokens, temperature, stop_tokens)
            for tokens in prompt_tokens_batch
        ]
        return await asyncio.gather(*tasks)


class TinkerMessageCompleter(MessageCompleter):
    """
    Message completer using the TenSafe/Tinker API with a renderer.

    Handles message formatting and parsing through the renderer,
    providing a clean interface for chat completions.
    """

    def __init__(
        self,
        client: SamplingClient,
        renderer: Renderer,
        default_max_tokens: int = 128,
        default_temperature: float = 1.0,
    ):
        """
        Initialize the message completer.

        Args:
            client: SamplingClient for API calls
            renderer: Renderer for message formatting
            default_max_tokens: Default maximum tokens
            default_temperature: Default temperature
        """
        self.client = client
        self.renderer = renderer
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

    async def complete(
        self,
        messages: Sequence[Union[Message, Dict[str, Any]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> Message:
        """Complete a conversation with an assistant response."""
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Build generation prompt
        prompt_tokens = self.renderer.build_generation_prompt(messages)
        prompt_text = self.renderer.decode(prompt_tokens, skip_special_tokens=False)

        # Get stop sequences
        stop_sequences = self._get_stop_strings()

        # Call the sampling API
        result = self.client.sample(
            prompts=[prompt_text],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            stop_sequences=stop_sequences,
        )

        # Extract completion
        if hasattr(result, "samples"):
            sample = result.samples[0]
            completion = sample.completion if hasattr(sample, "completion") else sample.get("completion", "")
        else:
            completion = result.get("samples", [{}])[0].get("completion", "")

        # Encode completion tokens
        completion_tokens = self.renderer.encode(completion)

        # Parse response through renderer
        return self.renderer.parse_response(completion_tokens)

    async def complete_batch(
        self,
        messages_batch: List[Sequence[Union[Message, Dict[str, Any]]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[Message]:
        """Complete a batch of conversations."""
        # Run completions concurrently
        tasks = [
            self.complete(messages, max_tokens, temperature)
            for messages in messages_batch
        ]
        return await asyncio.gather(*tasks)

    def _get_stop_strings(self) -> List[str]:
        """Get stop sequences as strings."""
        stops = self.renderer.get_stop_sequences()
        result = []
        for stop in stops:
            if isinstance(stop, str):
                result.append(stop)
            elif isinstance(stop, int):
                result.append(self.renderer.decode([stop], skip_special_tokens=False))
            elif isinstance(stop, list):
                if stop and isinstance(stop[0], int):
                    result.append(self.renderer.decode(stop, skip_special_tokens=False))
                else:
                    result.extend(stop)
        return result


class MockTokenCompleter(TokenCompleter):
    """
    Mock token completer for testing.

    Generates deterministic mock completions without calling any API.
    """

    def __init__(
        self,
        tokenizer: Any,
        response_length: int = 20,
        seed: int = 42,
    ):
        """
        Initialize mock completer.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            response_length: Fixed response length
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.response_length = response_length
        self.seed = seed
        self._counter = 0

    async def complete(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 128,
        temperature: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> TokensWithLogprobs:
        """Generate mock completion."""
        self._counter += 1

        # Generate deterministic mock tokens
        vocab_size = getattr(self.tokenizer, "vocab_size", 32000)
        length = min(self.response_length, max_tokens)

        tokens = [
            ((self.seed + self._counter + i) * 1103515245 + 12345) % vocab_size
            for i in range(length)
        ]

        # Mock logprobs
        logprobs = [-0.3 - (i * 0.01) for i in range(length)]

        return TokensWithLogprobs(
            tokens=tokens,
            logprobs=logprobs,
            cumulative_logprob=sum(logprobs),
        )

    async def complete_batch(
        self,
        prompt_tokens_batch: List[List[int]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[TokensWithLogprobs]:
        """Generate mock completions for batch."""
        return [
            await self.complete(tokens, max_tokens, temperature, stop_tokens)
            for tokens in prompt_tokens_batch
        ]


class MockMessageCompleter(MessageCompleter):
    """
    Mock message completer for testing.

    Generates deterministic mock responses without calling any API.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
    ):
        """
        Initialize mock completer.

        Args:
            responses: Optional list of canned responses to cycle through
        """
        self.responses = responses or [
            "I understand. How can I help you?",
            "That's a great question. Let me explain...",
            "Here's what I think about that.",
        ]
        self._index = 0

    async def complete(
        self,
        messages: Sequence[Union[Message, Dict[str, Any]]],
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Message:
        """Generate mock response."""
        response = self.responses[self._index % len(self.responses)]
        self._index += 1

        return Message(role="assistant", content=response)

    async def complete_batch(
        self,
        messages_batch: List[Sequence[Union[Message, Dict[str, Any]]]],
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> List[Message]:
        """Generate mock responses for batch."""
        return [
            await self.complete(messages, max_tokens, temperature)
            for messages in messages_batch
        ]
