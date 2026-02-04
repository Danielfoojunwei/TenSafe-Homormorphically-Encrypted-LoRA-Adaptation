"""
Llama 3 renderer for Meta's Llama 3.x model family.

This renderer handles the token formatting for Llama 3 Instruct models,
using the special tokens and formatting conventions specific to this family.

Special Tokens:
- <|begin_of_text|>: Beginning of sequence
- <|start_header_id|>...<|end_header_id|>: Message header markers
- <|eot_id|>: End of turn marker

Note: Tool calling is NOT supported for Llama 3 in this renderer because
the format lacks delimiters to reliably distinguish tool calls from
standard JSON responses.
"""

from __future__ import annotations

from typing import Any, List

from .base import (
    Message,
    MessageChunk,
    RenderedMessage,
    Renderer,
    RenderContext,
    StopCondition,
    Tokenizer,
)


class Llama3Renderer(Renderer):
    """
    Renderer for Meta's Llama 3.x Instruct models.

    Implements the Llama 3 chat format with special tokens for
    message delimitation and role identification.

    Example format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Hi there!<|eot_id|>
    """

    # Special token strings
    BOS = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the Llama 3 renderer.

        Args:
            tokenizer: Tokenizer for Llama 3 model
        """
        super().__init__(tokenizer)

        # Cache special token IDs
        self._bos_id = self._get_token_id(self.BOS)
        self._start_header_id = self._get_token_id(self.START_HEADER)
        self._end_header_id = self._get_token_id(self.END_HEADER)
        self._eot_id = self._get_token_id(self.EOT)

    def _get_token_id(self, token: str) -> int:
        """Get the token ID for a special token string."""
        # Try to get from vocab directly
        if hasattr(self.tokenizer, "vocab"):
            vocab = self.tokenizer.vocab
            if token in vocab:
                return vocab[token]

        # Try convert_tokens_to_ids
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                return token_id

        # Fall back to encoding
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]

        # Return a placeholder if not found
        return -1

    @property
    def _bos_tokens(self) -> List[int]:
        """Beginning of sequence token."""
        if self._bos_id >= 0:
            return [self._bos_id]
        return []

    def get_stop_sequences(self) -> List[StopCondition]:
        """
        Get stop sequences for Llama 3.

        Returns:
            List containing the EOT token as stop condition
        """
        if self._eot_id >= 0:
            return [self._eot_id]
        return [self.EOT]

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """
        Render a message in Llama 3 format.

        Format: <|start_header_id|>{role}<|end_header_id|>

        {content}<|eot_id|>

        Args:
            message: The message to render
            ctx: Rendering context

        Returns:
            RenderedMessage with header and content chunks
        """
        chunks: List[MessageChunk] = []

        # Build header: <|start_header_id|>{role}<|end_header_id|>\n\n
        header_text = f"{self.START_HEADER}{message.role}{self.END_HEADER}\n\n"
        header_tokens = self.encode(header_text)

        chunks.append(
            MessageChunk(
                tokens=header_tokens,
                is_trainable=False,  # Don't train on headers
                is_header=True,
            )
        )

        # Content tokens
        content_tokens = self.encode(message.content)
        chunks.append(
            MessageChunk(
                tokens=content_tokens,
                is_trainable=True,
                is_header=False,
            )
        )

        # Add EOT if requested
        if ctx.include_stop:
            eot_tokens = [self._eot_id] if self._eot_id >= 0 else self.encode(self.EOT)
            chunks.append(
                MessageChunk(
                    tokens=eot_tokens,
                    is_trainable=False,  # Don't train on stop tokens
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

        Removes any trailing EOT tokens and decodes the content.

        Args:
            response_tokens: Token IDs from generation

        Returns:
            Parsed assistant Message
        """
        # Remove trailing EOT if present
        tokens = list(response_tokens)
        while tokens and tokens[-1] == self._eot_id:
            tokens.pop()

        # Decode content
        content = self.decode(tokens, skip_special_tokens=True).strip()

        return Message(role="assistant", content=content)

    def _get_generation_trigger_tokens(self) -> List[int]:
        """
        Get tokens to trigger assistant generation.

        Returns the header tokens for an assistant message:
        <|start_header_id|>assistant<|end_header_id|>

        """
        trigger_text = f"{self.START_HEADER}assistant{self.END_HEADER}\n\n"
        return self.encode(trigger_text)


class Llama3VisionRenderer(Llama3Renderer):
    """
    Renderer for Llama 3 Vision models.

    Extends the base Llama 3 renderer with support for image tokens.
    """

    # Image token placeholder
    IMAGE_TOKEN = "<|image|>"

    def __init__(self, tokenizer: Tokenizer):
        """Initialize the Llama 3 Vision renderer."""
        super().__init__(tokenizer)
        self._image_token_id = self._get_token_id(self.IMAGE_TOKEN)

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """
        Render a message, handling image content.

        If the message contains image metadata, inserts image tokens
        at the appropriate location.

        Args:
            message: The message to render (may contain image metadata)
            ctx: Rendering context

        Returns:
            RenderedMessage with image tokens if applicable
        """
        # Check for image content in metadata
        has_image = message.metadata.get("has_image", False)
        num_images = message.metadata.get("num_images", 0)

        if not has_image or num_images == 0:
            # No images, use standard rendering
            return super().render_message(message, ctx)

        chunks: List[MessageChunk] = []

        # Build header
        header_text = f"{self.START_HEADER}{message.role}{self.END_HEADER}\n\n"
        header_tokens = self.encode(header_text)
        chunks.append(
            MessageChunk(
                tokens=header_tokens,
                is_trainable=False,
                is_header=True,
            )
        )

        # Add image tokens at the start of content
        if self._image_token_id >= 0:
            image_tokens = [self._image_token_id] * num_images
        else:
            image_tokens = self.encode(self.IMAGE_TOKEN * num_images)

        chunks.append(
            MessageChunk(
                tokens=image_tokens,
                is_trainable=False,  # Don't train on image placeholders
                is_header=False,
            )
        )

        # Content tokens
        content_tokens = self.encode(message.content)
        chunks.append(
            MessageChunk(
                tokens=content_tokens,
                is_trainable=True,
                is_header=False,
            )
        )

        # Add EOT if requested
        if ctx.include_stop:
            eot_tokens = [self._eot_id] if self._eot_id >= 0 else self.encode(self.EOT)
            chunks.append(
                MessageChunk(
                    tokens=eot_tokens,
                    is_trainable=False,
                    is_stop=True,
                )
            )

        return RenderedMessage(chunks=chunks, original_message=message)
