"""Tests for cookbook renderers."""

from typing import List

import pytest


class MockTokenizer:
    """Mock tokenizer for testing without transformers dependency."""

    def __init__(self):
        self.vocab = {
            "<|begin_of_text|>": 0,
            "<|start_header_id|>": 1,
            "<|end_header_id|>": 2,
            "<|eot_id|>": 3,
            "<|im_start|>": 4,
            "<|im_end|>": 5,
        }
        self.unk_token_id = 100
        self._counter = 10

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Mock encode - returns list of token IDs."""
        # Check for special tokens
        for token, tid in self.vocab.items():
            if text == token:
                return [tid]
        # Otherwise return mock tokens based on length
        return list(range(self._counter, self._counter + len(text.split())))

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Mock decode - returns placeholder text."""
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.vocab.values()]
        return f"decoded_{len(tokens)}_tokens"

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert token string to ID."""
        return self.vocab.get(token, self.unk_token_id)


class TestRendererBase:
    """Tests for base Renderer class."""

    def test_message_creation(self):
        """Test Message dataclass creation."""
        from tensafe.cookbook.renderers.base import Message

        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.name is None
        assert msg.tool_calls is None

    def test_message_to_dict(self):
        """Test Message to dict conversion."""
        from tensafe.cookbook.renderers.base import Message

        msg = Message(role="assistant", content="Hi there!", name="Bot")
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "Hi there!"
        assert d["name"] == "Bot"

    def test_message_from_dict(self):
        """Test Message from dict creation."""
        from tensafe.cookbook.renderers.base import Message

        d = {"role": "user", "content": "Test message"}
        msg = Message.from_dict(d)

        assert msg.role == "user"
        assert msg.content == "Test message"

    def test_message_chunk_creation(self):
        """Test MessageChunk dataclass."""
        from tensafe.cookbook.renderers.base import MessageChunk

        chunk = MessageChunk(
            tokens=[1, 2, 3],
            is_trainable=True,
            is_header=False,
        )
        assert chunk.tokens == [1, 2, 3]
        assert chunk.is_trainable is True
        assert chunk.is_header is False

    def test_rendered_message_all_tokens(self):
        """Test RenderedMessage.all_tokens property."""
        from tensafe.cookbook.renderers.base import MessageChunk, RenderedMessage

        rm = RenderedMessage(
            chunks=[
                MessageChunk(tokens=[1, 2], is_trainable=False),
                MessageChunk(tokens=[3, 4, 5], is_trainable=True),
            ]
        )

        assert rm.all_tokens == [1, 2, 3, 4, 5]

    def test_rendered_message_token_weights(self):
        """Test RenderedMessage.token_weights property."""
        from tensafe.cookbook.renderers.base import MessageChunk, RenderedMessage

        rm = RenderedMessage(
            chunks=[
                MessageChunk(tokens=[1, 2], is_trainable=False),
                MessageChunk(tokens=[3, 4], is_trainable=True),
            ]
        )

        weights = rm.token_weights
        assert weights == [0.0, 0.0, 1.0, 1.0]


class TestLlama3Renderer:
    """Tests for Llama3Renderer."""

    def test_init(self):
        """Test Llama3Renderer initialization."""
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = Llama3Renderer(tokenizer)

        assert renderer.tokenizer is tokenizer
        assert renderer._bos_id == 0
        assert renderer._eot_id == 3

    def test_get_stop_sequences(self):
        """Test stop sequence retrieval."""
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = Llama3Renderer(tokenizer)

        stops = renderer.get_stop_sequences()
        assert len(stops) >= 1
        # Should include EOT token
        assert 3 in stops or "<|eot_id|>" in stops

    def test_render_message(self):
        """Test message rendering."""
        from tensafe.cookbook.renderers.base import Message, RenderContext
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = Llama3Renderer(tokenizer)

        msg = Message(role="user", content="Hello!")
        ctx = RenderContext(include_stop=True)

        rendered = renderer.render_message(msg, ctx)

        assert len(rendered.chunks) >= 2  # Header + content + possibly stop
        assert rendered.chunks[0].is_header is True
        assert rendered.chunks[0].is_trainable is False

    def test_build_generation_prompt(self):
        """Test generation prompt building."""
        from tensafe.cookbook.renderers.base import Message
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = Llama3Renderer(tokenizer)

        messages = [
            Message(role="user", content="Hi!"),
        ]

        tokens = renderer.build_generation_prompt(messages)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_build_supervised_example(self):
        """Test supervised example building."""
        from tensafe.cookbook.renderers.base import Message, SupervisedWeightMode
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = Llama3Renderer(tokenizer)

        messages = [
            Message(role="user", content="Hi!"),
            Message(role="assistant", content="Hello!"),
        ]

        tokens, weights = renderer.build_supervised_example(
            messages,
            weight_mode=SupervisedWeightMode.LAST_ASSISTANT,
        )

        assert len(tokens) == len(weights)
        assert len(tokens) > 0
        # Should have some non-zero weights for assistant message
        assert sum(weights) > 0

    def test_parse_response(self):
        """Test response parsing."""
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = Llama3Renderer(tokenizer)

        # Parse some mock tokens
        tokens = [10, 11, 12, 3]  # Content + EOT
        msg = renderer.parse_response(tokens)

        assert msg.role == "assistant"
        assert isinstance(msg.content, str)


class TestQwen3Renderer:
    """Tests for Qwen3Renderer."""

    def test_init(self):
        """Test Qwen3Renderer initialization."""
        from tensafe.cookbook.renderers.qwen3 import Qwen3Renderer

        tokenizer = MockTokenizer()
        renderer = Qwen3Renderer(tokenizer)

        assert renderer.tokenizer is tokenizer

    def test_get_stop_sequences(self):
        """Test stop sequence retrieval."""
        from tensafe.cookbook.renderers.qwen3 import Qwen3Renderer

        tokenizer = MockTokenizer()
        renderer = Qwen3Renderer(tokenizer)

        stops = renderer.get_stop_sequences()
        assert len(stops) >= 1


class TestDeepSeekV3Renderer:
    """Tests for DeepSeekV3Renderer."""

    def test_init(self):
        """Test DeepSeekV3Renderer initialization."""
        from tensafe.cookbook.renderers.deepseek_v3 import DeepSeekV3Renderer

        tokenizer = MockTokenizer()
        renderer = DeepSeekV3Renderer(tokenizer)

        assert renderer.tokenizer is tokenizer
        assert renderer.thinking_mode is False

    def test_thinking_mode(self):
        """Test thinking mode configuration."""
        from tensafe.cookbook.renderers.deepseek_v3 import DeepSeekV3Renderer

        tokenizer = MockTokenizer()
        renderer = DeepSeekV3Renderer(tokenizer, thinking_mode=True)

        assert renderer.thinking_mode is True


class TestRoleColonRenderer:
    """Tests for RoleColonRenderer."""

    def test_init(self):
        """Test RoleColonRenderer initialization."""
        from tensafe.cookbook.renderers.role_colon import RoleColonRenderer

        tokenizer = MockTokenizer()
        renderer = RoleColonRenderer(tokenizer)

        assert renderer.tokenizer is tokenizer
        assert len(renderer.stop_strings) > 0

    def test_get_stop_sequences(self):
        """Test stop sequence retrieval."""
        from tensafe.cookbook.renderers.role_colon import RoleColonRenderer

        tokenizer = MockTokenizer()
        renderer = RoleColonRenderer(tokenizer)

        stops = renderer.get_stop_sequences()
        assert "User:" in stops or len(stops) > 0


class TestGetRenderer:
    """Tests for get_renderer factory function."""

    def test_get_llama3_renderer(self):
        """Test getting Llama3 renderer."""
        from tensafe.cookbook.renderers import get_renderer
        from tensafe.cookbook.renderers.llama3 import Llama3Renderer

        tokenizer = MockTokenizer()
        renderer = get_renderer("llama3", tokenizer)

        assert isinstance(renderer, Llama3Renderer)

    def test_get_qwen_renderer(self):
        """Test getting Qwen renderer."""
        from tensafe.cookbook.renderers import get_renderer
        from tensafe.cookbook.renderers.qwen3 import Qwen3Renderer

        tokenizer = MockTokenizer()
        renderer = get_renderer("qwen3", tokenizer)

        assert isinstance(renderer, Qwen3Renderer)

    def test_get_unknown_renderer_raises(self):
        """Test that unknown renderer raises ValueError."""
        from tensafe.cookbook.renderers import get_renderer

        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match="Unknown renderer"):
            get_renderer("nonexistent", tokenizer)
