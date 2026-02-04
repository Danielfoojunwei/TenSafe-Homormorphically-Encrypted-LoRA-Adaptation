"""
Generic role:colon renderer for simple chat formatting.

This renderer uses a straightforward format:
    Role: Content

This can be used as a fallback for models that don't have
specialized chat templates.
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


class RoleColonRenderer(Renderer):
    """
    Generic renderer using role:content format.

    Example format:
        System: You are a helpful assistant.

        User: Hello!

        Assistant: Hi there!
    """

    # Role names (capitalized)
    ROLE_MAP = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }

    # Separator between messages
    MESSAGE_SEP = "\n\n"

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop_strings: List[str] = None,
    ):
        """
        Initialize the role:colon renderer.

        Args:
            tokenizer: Tokenizer for the model
            stop_strings: Custom stop strings (default: role prefixes)
        """
        super().__init__(tokenizer)
        self.stop_strings = stop_strings or ["User:", "System:", "\n\nUser", "\n\nSystem"]

    def get_stop_sequences(self) -> List[StopCondition]:
        """
        Get stop sequences for role:colon format.

        Returns:
            List of stop strings
        """
        return self.stop_strings

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """
        Render a message in role:content format.

        Format: {Role}: {content}

        Args:
            message: The message to render
            ctx: Rendering context

        Returns:
            RenderedMessage with header and content chunks
        """
        chunks: List[MessageChunk] = []

        # Get display role
        role_display = self.ROLE_MAP.get(message.role, message.role.title())

        # Build header: {Role}:
        header_text = f"{role_display}: "
        if not ctx.is_first:
            header_text = self.MESSAGE_SEP + header_text

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

        return RenderedMessage(chunks=chunks, original_message=message)

    def parse_response(
        self,
        response_tokens: List[int],
    ) -> Message:
        """
        Parse generated tokens into a Message.

        Args:
            response_tokens: Token IDs from generation

        Returns:
            Parsed assistant Message
        """
        # Decode content
        content = self.decode(response_tokens, skip_special_tokens=True)

        # Clean up any stop strings that might be at the end
        for stop in self.stop_strings:
            if content.endswith(stop):
                content = content[:-len(stop)]

        content = content.strip()

        return Message(role="assistant", content=content)

    def _get_generation_trigger_tokens(self) -> List[int]:
        """Get tokens to trigger assistant generation."""
        trigger_text = f"{self.MESSAGE_SEP}Assistant: "
        return self.encode(trigger_text)
