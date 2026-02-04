"""
Qwen 3 renderer for Alibaba's Qwen model family.

This renderer handles the token formatting for Qwen 3 models,
supporting both base and instruct variants.

Format uses ChatML-style special tokens:
- <|im_start|>: Start of message
- <|im_end|>: End of message
"""

from __future__ import annotations

from typing import List

from .base import (
    Message,
    MessageChunk,
    RenderedMessage,
    Renderer,
    RenderContext,
    StopCondition,
    Tokenizer,
)


class Qwen3Renderer(Renderer):
    """
    Renderer for Qwen 3 models using ChatML format.

    Example format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
    """

    # Special token strings (ChatML format)
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the Qwen 3 renderer.

        Args:
            tokenizer: Tokenizer for Qwen 3 model
        """
        super().__init__(tokenizer)

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
        Get stop sequences for Qwen 3.

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
        Render a message in Qwen 3 ChatML format.

        Format: <|im_start|>{role}
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
            # Format: <|im_end|>\n
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

        Removes any trailing im_end tokens and decodes the content.

        Args:
            response_tokens: Token IDs from generation

        Returns:
            Parsed assistant Message
        """
        # Remove trailing im_end if present
        tokens = list(response_tokens)
        while tokens and tokens[-1] == self._im_end_id:
            tokens.pop()

        # Decode content
        content = self.decode(tokens, skip_special_tokens=True).strip()

        return Message(role="assistant", content=content)

    def _get_generation_trigger_tokens(self) -> List[int]:
        """
        Get tokens to trigger assistant generation.

        Returns the header tokens for an assistant message:
        <|im_start|>assistant\n
        """
        trigger_text = f"{self.IM_START}assistant\n"
        return self.encode(trigger_text)
