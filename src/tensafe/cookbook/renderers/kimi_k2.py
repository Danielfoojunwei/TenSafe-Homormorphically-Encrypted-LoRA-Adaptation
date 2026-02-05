"""
Kimi K2 renderer for Moonshot's Kimi K2 Thinking model.

This renderer handles the token formatting for Kimi K2 models,
with special support for the thinking/reasoning mode.
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


class KimiK2Renderer(Renderer):
    """
    Renderer for Moonshot's Kimi K2 Thinking model.

    Uses ChatML-style format with thinking mode support.

    Example format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What is 2+2?<|im_end|>
        <|im_start|>assistant
        <think>
        Let me calculate...
        </think>
        4<|im_end|>
    """

    # Special token strings
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINK_START = "<think>"
    THINK_END = "</think>"

    def __init__(
        self,
        tokenizer: Tokenizer,
        thinking_mode: bool = True,
    ):
        """
        Initialize the Kimi K2 renderer.

        Args:
            tokenizer: Tokenizer for Kimi K2 model
            thinking_mode: Whether to use thinking mode
        """
        super().__init__(tokenizer)
        self.thinking_mode = thinking_mode

        # Cache special token IDs
        self._im_start_id = self._get_token_id(self.IM_START)
        self._im_end_id = self._get_token_id(self.IM_END)

    def _get_token_id(self, token: str) -> int:
        """Get the token ID for a special token string."""
        if hasattr(self.tokenizer, "vocab"):
            vocab = self.tokenizer.vocab
            if token in vocab:
                return vocab[token]

        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if hasattr(self.tokenizer, "unk_token_id"):
                if token_id != self.tokenizer.unk_token_id:
                    return token_id
            else:
                return token_id

        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]

        return -1

    def get_stop_sequences(self) -> List[StopCondition]:
        """
        Get stop sequences for Kimi K2.

        Returns:
            List containing the im_end token as stop condition
        """
        if self._im_end_id >= 0:
            return [self._im_end_id]
        return [self.IM_END]

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """
        Render a message in Kimi K2 format.

        Format: <|im_start|>{role}
        [<think>thinking</think>]
        {content}<|im_end|>

        Args:
            message: The message to render
            ctx: Rendering context

        Returns:
            RenderedMessage with header and content chunks
        """
        chunks: List[MessageChunk] = []

        # Build header: <|im_start|>{role}\n
        header_text = f"{self.IM_START}{message.role}\n"
        header_tokens = self.encode(header_text)

        chunks.append(
            MessageChunk(
                tokens=header_tokens,
                is_trainable=False,
                is_header=True,
            )
        )

        # Check for thinking content in assistant messages
        thinking = message.metadata.get("thinking", None)

        if message.role == "assistant" and thinking and self.thinking_mode:
            # Add thinking section
            think_start = self.encode(f"{self.THINK_START}\n")
            chunks.append(
                MessageChunk(
                    tokens=think_start,
                    is_trainable=False,
                    is_header=True,
                )
            )

            thinking_tokens = self.encode(thinking)
            chunks.append(
                MessageChunk(
                    tokens=thinking_tokens,
                    is_trainable=True,  # Train on thinking
                )
            )

            think_end = self.encode(f"\n{self.THINK_END}\n")
            chunks.append(
                MessageChunk(
                    tokens=think_end,
                    is_trainable=False,
                    is_header=True,
                )
            )

        # Content tokens
        content_tokens = self.encode(message.content)
        chunks.append(
            MessageChunk(
                tokens=content_tokens,
                is_trainable=True,
            )
        )

        # Add im_end if requested
        if ctx.include_stop:
            end_text = f"{self.IM_END}\n"
            end_tokens = self.encode(end_text)
            chunks.append(
                MessageChunk(
                    tokens=end_tokens,
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

        Extracts thinking content if present.

        Args:
            response_tokens: Token IDs from generation

        Returns:
            Parsed assistant Message with optional thinking metadata
        """
        # Remove trailing im_end if present
        tokens = list(response_tokens)
        while tokens and tokens[-1] == self._im_end_id:
            tokens.pop()

        # Decode content
        text = self.decode(tokens, skip_special_tokens=False)

        # Extract thinking if present
        thinking: Optional[str] = None
        content = text

        if self.THINK_START in text and self.THINK_END in text:
            start_idx = text.find(self.THINK_START)
            end_idx = text.find(self.THINK_END)

            if start_idx < end_idx:
                thinking = text[start_idx + len(self.THINK_START):end_idx].strip()
                content = text[end_idx + len(self.THINK_END):].strip()

        # Clean up content
        content = content.strip()

        # Build message
        msg = Message(role="assistant", content=content)
        if thinking:
            msg.metadata["thinking"] = thinking

        return msg

    def _get_generation_trigger_tokens(self) -> List[int]:
        """Get tokens to trigger assistant generation."""
        trigger_text = f"{self.IM_START}assistant\n"
        return self.encode(trigger_text)
