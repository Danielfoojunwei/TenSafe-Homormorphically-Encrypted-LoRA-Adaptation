"""
Base renderer class for converting between tokens and structured chat messages.

The Renderer provides the foundation for model-specific token formatting,
handling the conversion between high-level message abstractions and
the token sequences expected by different language models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeAlias, Union

# Type alias for tokenizer (avoid hard dependency on transformers)
Tokenizer: TypeAlias = Any

# Stop conditions can be token IDs or string sequences
StopCondition: TypeAlias = Union[int, str, List[int], List[str]]


class SupervisedWeightMode(str, Enum):
    """Modes for weighting tokens in supervised learning."""

    # Only weight the last assistant message
    LAST_ASSISTANT = "last_assistant"
    # Weight all assistant messages
    ALL_ASSISTANT = "all_assistant"
    # Weight all messages (including user/system)
    ALL_MESSAGES = "all_messages"
    # Custom per-token weights
    CUSTOM = "custom"


@dataclass
class Message:
    """
    A single message in a conversation.

    Attributes:
        role: The role of the message author (system, user, assistant, tool)
        content: The text content of the message
        name: Optional name for the speaker
        tool_calls: Optional tool calls made by the assistant
        tool_call_id: Optional ID linking to a tool call (for tool responses)
        metadata: Optional additional metadata
    """

    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary representation."""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MessageChunk:
    """
    A chunk of tokens with associated metadata.

    Used for decomposing messages into header/output/stop_overlap components
    for fine-grained loss masking control.
    """

    tokens: List[int]
    is_trainable: bool = True  # Whether to compute loss on these tokens
    is_header: bool = False  # Is this a message header (role markers, etc.)
    is_stop: bool = False  # Is this the stop/end-of-turn tokens


@dataclass
class RenderedMessage:
    """
    A message rendered into token chunks.

    Attributes:
        chunks: List of MessageChunks that make up the message
        original_message: The original Message object
    """

    chunks: List[MessageChunk]
    original_message: Optional[Message] = None

    @property
    def all_tokens(self) -> List[int]:
        """Get all tokens from all chunks."""
        tokens: List[int] = []
        for chunk in self.chunks:
            tokens.extend(chunk.tokens)
        return tokens

    @property
    def trainable_tokens(self) -> List[int]:
        """Get only trainable tokens."""
        tokens: List[int] = []
        for chunk in self.chunks:
            if chunk.is_trainable:
                tokens.extend(chunk.tokens)
        return tokens

    @property
    def token_weights(self) -> List[float]:
        """Get weight for each token (1.0 if trainable, 0.0 otherwise)."""
        weights: List[float] = []
        for chunk in self.chunks:
            weight = 1.0 if chunk.is_trainable else 0.0
            weights.extend([weight] * len(chunk.tokens))
        return weights


@dataclass
class RenderContext:
    """
    Context for rendering messages.

    Provides information about the rendering context, such as
    position in the conversation and whether this is for generation
    or supervised training.
    """

    # Position in conversation (0-indexed)
    message_index: int = 0

    # Total number of messages in conversation
    total_messages: int = 1

    # Is this the first message?
    is_first: bool = True

    # Is this the last message?
    is_last: bool = True

    # Is this for generation (vs supervised training)?
    for_generation: bool = False

    # Include stop tokens?
    include_stop: bool = True


class Renderer(ABC):
    """
    Abstract base class for rendering message lists into training and sampling prompts.

    Renderers handle the conversion between high-level message abstractions
    and the token sequences expected by different language models.

    Key responsibilities:
    - Format messages with model-specific special tokens
    - Build prompts for generation (sampling)
    - Build supervised examples with loss masking
    - Parse generated tokens back into messages

    Attributes:
        tokenizer: The tokenizer for the target model
    """

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the renderer.

        Args:
            tokenizer: Tokenizer for encoding/decoding text
        """
        self.tokenizer = tokenizer

    @property
    def has_extension_property(self) -> bool:
        """
        Whether successive assistant turns produce token sequences where each
        is a prefix of the next.

        When True, this enables KV-cache reuse and efficient compute scaling
        for streaming generation.
        """
        return True

    @property
    def _bos_tokens(self) -> List[int]:
        """Beginning-of-sequence tokens (default: empty list)."""
        return []

    @abstractmethod
    def get_stop_sequences(self) -> List[StopCondition]:
        """
        Get stop tokens/sequences for sampling.

        Returns:
            List of stop conditions (token IDs or strings)
        """
        pass

    @abstractmethod
    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """
        Render a single message into token chunks.

        This decomposes the message into header/output/stop_overlap components
        for fine-grained loss masking control.

        Args:
            message: The message to render
            ctx: Rendering context

        Returns:
            RenderedMessage with token chunks
        """
        pass

    @abstractmethod
    def parse_response(
        self,
        response_tokens: List[int],
    ) -> Message:
        """
        Parse generated tokens back into a Message.

        Args:
            response_tokens: Token IDs from generation

        Returns:
            Parsed Message object
        """
        pass

    def build_generation_prompt(
        self,
        messages: Sequence[Union[Message, Dict[str, Any]]],
    ) -> List[int]:
        """
        Build token sequence for generation/sampling.

        Creates the prompt tokens that will be used to condition generation.
        This includes all messages plus any necessary prompt tokens to
        trigger assistant response generation.

        Args:
            messages: Conversation history

        Returns:
            List of token IDs for the prompt
        """
        # Convert dicts to Messages
        msg_list = [
            Message.from_dict(m) if isinstance(m, dict) else m for m in messages
        ]

        tokens: List[int] = list(self._bos_tokens)

        for i, msg in enumerate(msg_list):
            ctx = RenderContext(
                message_index=i,
                total_messages=len(msg_list),
                is_first=(i == 0),
                is_last=(i == len(msg_list) - 1),
                for_generation=True,
                include_stop=(i < len(msg_list) - 1),  # No stop for last
            )
            rendered = self.render_message(msg, ctx)
            tokens.extend(rendered.all_tokens)

        # Add generation trigger for assistant
        tokens.extend(self._get_generation_trigger_tokens())

        return tokens

    def build_supervised_example(
        self,
        messages: Sequence[Union[Message, Dict[str, Any]]],
        weight_mode: SupervisedWeightMode = SupervisedWeightMode.LAST_ASSISTANT,
        custom_weights: Optional[List[float]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Build token sequence and weights for supervised training.

        Creates training tokens and per-token weights for loss masking.

        Args:
            messages: Conversation with assistant responses to train on
            weight_mode: How to weight tokens for loss computation
            custom_weights: Per-message weights (for CUSTOM mode)

        Returns:
            Tuple of (token_ids, token_weights)
        """
        # Convert dicts to Messages
        msg_list = [
            Message.from_dict(m) if isinstance(m, dict) else m for m in messages
        ]

        tokens: List[int] = list(self._bos_tokens)
        weights: List[float] = [0.0] * len(self._bos_tokens)

        # Find last assistant message index
        last_assistant_idx = -1
        for i, msg in enumerate(msg_list):
            if msg.role == "assistant":
                last_assistant_idx = i

        for i, msg in enumerate(msg_list):
            ctx = RenderContext(
                message_index=i,
                total_messages=len(msg_list),
                is_first=(i == 0),
                is_last=(i == len(msg_list) - 1),
                for_generation=False,
                include_stop=True,
            )
            rendered = self.render_message(msg, ctx)

            # Determine weight for this message based on mode
            if weight_mode == SupervisedWeightMode.LAST_ASSISTANT:
                should_weight = msg.role == "assistant" and i == last_assistant_idx
            elif weight_mode == SupervisedWeightMode.ALL_ASSISTANT:
                should_weight = msg.role == "assistant"
            elif weight_mode == SupervisedWeightMode.ALL_MESSAGES:
                should_weight = True
            elif weight_mode == SupervisedWeightMode.CUSTOM:
                if custom_weights and i < len(custom_weights):
                    msg_weight = custom_weights[i]
                else:
                    msg_weight = 0.0
                # Apply custom weight to trainable chunks
                for chunk in rendered.chunks:
                    tokens.extend(chunk.tokens)
                    chunk_weight = msg_weight if chunk.is_trainable else 0.0
                    weights.extend([chunk_weight] * len(chunk.tokens))
                continue
            else:
                should_weight = False

            # Apply weights
            for chunk in rendered.chunks:
                tokens.extend(chunk.tokens)
                if should_weight and chunk.is_trainable:
                    weights.extend([1.0] * len(chunk.tokens))
                else:
                    weights.extend([0.0] * len(chunk.tokens))

        return tokens, weights

    def _get_generation_trigger_tokens(self) -> List[int]:
        """
        Get tokens to append after messages to trigger assistant generation.

        Override in subclasses for model-specific triggers.

        Returns:
            Token IDs to trigger generation
        """
        return []

    def to_openai_message(self, message: Message) -> Dict[str, Any]:
        """
        Convert Message to OpenAI Chat API format.

        Args:
            message: Internal Message format

        Returns:
            Dictionary in OpenAI Chat format
        """
        result: Dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.name:
            result["name"] = message.name
        if message.tool_calls:
            result["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        return result

    def from_openai_message(self, data: Dict[str, Any]) -> Message:
        """
        Convert OpenAI Chat format to internal Message.

        Args:
            data: Dictionary in OpenAI Chat format

        Returns:
            Internal Message object
        """
        return Message.from_dict(data)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
