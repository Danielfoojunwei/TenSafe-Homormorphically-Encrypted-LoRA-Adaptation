"""
DeepSeek V3 renderer for DeepSeek's model family.

This renderer handles the token formatting for DeepSeek V3 models,
supporting both regular chat and thinking mode with chain-of-thought.

Special Tokens:
- <|begin_of_sentence|>: Beginning of sequence
- <|User|>: User message marker
- <|Assistant|>: Assistant message marker
- <|end_of_sentence|>: End of turn marker
- <|begin_of_thought|>/<|end_of_thought|>: Thinking mode markers
"""

from __future__ import annotations

from typing import List, Optional

from .base import (
    Message,
    MessageChunk,
    RenderedMessage,
    Renderer,
    RenderContext,
    StopCondition,
    Tokenizer,
)


class DeepSeekV3Renderer(Renderer):
    """
    Renderer for DeepSeek V3 models.

    Supports both standard chat and thinking mode (chain-of-thought).

    Example format (standard):
        <|begin_of_sentence|><|User|>Hello!<|end_of_sentence|>
        <|Assistant|>Hi there!<|end_of_sentence|>

    Example format (thinking mode):
        <|begin_of_sentence|><|User|>Solve this problem<|end_of_sentence|>
        <|Assistant|><|begin_of_thought|>
        Let me think...
        <|end_of_thought|>
        The answer is 42.<|end_of_sentence|>
    """

    # Special token strings
    BOS = "<|begin_of_sentence|>"
    EOS = "<|end_of_sentence|>"
    USER = "<|User|>"
    ASSISTANT = "<|Assistant|>"
    BEGIN_THOUGHT = "<|begin_of_thought|>"
    END_THOUGHT = "<|end_of_thought|>"

    def __init__(
        self,
        tokenizer: Tokenizer,
        thinking_mode: bool = False,
    ):
        """
        Initialize the DeepSeek V3 renderer.

        Args:
            tokenizer: Tokenizer for DeepSeek model
            thinking_mode: Whether to use thinking mode with CoT
        """
        super().__init__(tokenizer)
        self.thinking_mode = thinking_mode

        # Cache special token IDs
        self._bos_id = self._get_token_id(self.BOS)
        self._eos_id = self._get_token_id(self.EOS)
        self._user_id = self._get_token_id(self.USER)
        self._assistant_id = self._get_token_id(self.ASSISTANT)
        self._begin_thought_id = self._get_token_id(self.BEGIN_THOUGHT)
        self._end_thought_id = self._get_token_id(self.END_THOUGHT)

    def _get_token_id(self, token: str) -> int:
        """Get the token ID for a special token string."""
        if hasattr(self.tokenizer, "vocab"):
            vocab = self.tokenizer.vocab
            if token in vocab:
                return vocab[token]

        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                return token_id

        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]

        return -1

    @property
    def _bos_tokens(self) -> List[int]:
        """Beginning of sequence token."""
        if self._bos_id >= 0:
            return [self._bos_id]
        return []

    def get_stop_sequences(self) -> List[StopCondition]:
        """
        Get stop sequences for DeepSeek V3.

        Returns:
            List containing the EOS token and potentially thinking markers
        """
        stops: List[StopCondition] = []

        if self._eos_id >= 0:
            stops.append(self._eos_id)
        else:
            stops.append(self.EOS)

        return stops

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """
        Render a message in DeepSeek V3 format.

        Args:
            message: The message to render
            ctx: Rendering context

        Returns:
            RenderedMessage with appropriate chunks
        """
        chunks: List[MessageChunk] = []

        # Get role marker
        if message.role == "user":
            role_token = self.USER
        elif message.role == "assistant":
            role_token = self.ASSISTANT
        elif message.role == "system":
            # System messages are formatted as user messages with special prefix
            role_token = self.USER
        else:
            role_token = f"<|{message.role.title()}|>"

        # Role marker token
        role_tokens = self.encode(role_token)
        chunks.append(
            MessageChunk(
                tokens=role_tokens,
                is_trainable=False,
                is_header=True,
            )
        )

        # Handle system message formatting
        content = message.content
        if message.role == "system":
            content = f"[System] {content}"

        # Check for thinking content in assistant messages
        thinking = message.metadata.get("thinking", None)

        if message.role == "assistant" and thinking and self.thinking_mode:
            # Add thinking section
            begin_thought = self.encode(f"\n{self.BEGIN_THOUGHT}\n")
            chunks.append(
                MessageChunk(
                    tokens=begin_thought,
                    is_trainable=False,
                    is_header=True,
                )
            )

            thinking_tokens = self.encode(thinking)
            chunks.append(
                MessageChunk(
                    tokens=thinking_tokens,
                    is_trainable=True,  # Can optionally train on thinking
                )
            )

            end_thought = self.encode(f"\n{self.END_THOUGHT}\n")
            chunks.append(
                MessageChunk(
                    tokens=end_thought,
                    is_trainable=False,
                    is_header=True,
                )
            )

        # Main content tokens
        content_tokens = self.encode(content)
        chunks.append(
            MessageChunk(
                tokens=content_tokens,
                is_trainable=True,
            )
        )

        # Add EOS if requested
        if ctx.include_stop:
            eos_tokens = [self._eos_id] if self._eos_id >= 0 else self.encode(self.EOS)
            chunks.append(
                MessageChunk(
                    tokens=eos_tokens,
                    is_trainable=False,
                    is_stop=True,
                )
            )

        return RenderedMessage(chunks=chunks, original_message=message)

    def parse_response(
        self,
        response_tokens: List[int],
    ) -> Message:
        """
        Parse generated tokens into a Message.

        Handles thinking mode by extracting thinking content to metadata.

        Args:
            response_tokens: Token IDs from generation

        Returns:
            Parsed assistant Message with optional thinking metadata
        """
        # Remove trailing EOS if present
        tokens = list(response_tokens)
        while tokens and tokens[-1] == self._eos_id:
            tokens.pop()

        # Decode content
        text = self.decode(tokens, skip_special_tokens=False)

        # Extract thinking if present
        thinking: Optional[str] = None
        content = text

        if self.BEGIN_THOUGHT in text and self.END_THOUGHT in text:
            start_idx = text.find(self.BEGIN_THOUGHT)
            end_idx = text.find(self.END_THOUGHT)

            if start_idx < end_idx:
                thinking = text[start_idx + len(self.BEGIN_THOUGHT):end_idx].strip()
                content = text[end_idx + len(self.END_THOUGHT):].strip()

        # Clean up content
        content = content.strip()

        # Build message
        msg = Message(role="assistant", content=content)
        if thinking:
            msg.metadata["thinking"] = thinking

        return msg

    def _get_generation_trigger_tokens(self) -> List[int]:
        """Get tokens to trigger assistant generation."""
        trigger_text = self.ASSISTANT
        return self.encode(trigger_text)
