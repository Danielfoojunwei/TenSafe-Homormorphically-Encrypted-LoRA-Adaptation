"""Tests for cookbook model info utilities."""



class TestModelAttributes:
    """Tests for ModelAttributes dataclass."""

    def test_creation(self):
        """Test ModelAttributes creation."""
        from tensafe.cookbook.model_info import ModelAttributes

        attrs = ModelAttributes(
            organization="meta-llama",
            version="Llama-3.1",
            size="8B",
            is_instruct=True,
        )

        assert attrs.organization == "meta-llama"
        assert attrs.version == "Llama-3.1"
        assert attrs.size == "8B"
        assert attrs.is_instruct is True
        assert attrs.is_vision is False

    def test_full_name(self):
        """Test full name generation."""
        from tensafe.cookbook.model_info import ModelAttributes

        attrs = ModelAttributes(
            organization="meta-llama",
            version="Llama-3.1",
            size="8B",
            is_instruct=True,
        )

        assert "meta-llama" in attrs.full_name
        assert "8B" in attrs.full_name


class TestGetModelAttributes:
    """Tests for get_model_attributes function."""

    def test_known_llama_model(self):
        """Test getting attributes for Llama model."""
        from tensafe.cookbook.model_info import get_model_attributes

        attrs = get_model_attributes("meta-llama/Llama-3.1-8B")

        assert attrs is not None
        assert attrs.organization == "meta-llama"
        assert attrs.size == "8B"
        assert attrs.context_length == 131072

    def test_known_llama_instruct_model(self):
        """Test getting attributes for Llama Instruct model."""
        from tensafe.cookbook.model_info import get_model_attributes

        attrs = get_model_attributes("meta-llama/Llama-3.1-8B-Instruct")

        assert attrs is not None
        assert attrs.is_instruct is True

    def test_known_qwen_model(self):
        """Test getting attributes for Qwen model."""
        from tensafe.cookbook.model_info import get_model_attributes

        attrs = get_model_attributes("Qwen/Qwen3-8B")

        assert attrs is not None
        assert attrs.organization == "Qwen"

    def test_unknown_model_returns_none(self):
        """Test that unknown model returns None."""
        from tensafe.cookbook.model_info import get_model_attributes

        attrs = get_model_attributes("unknown/model-123")
        assert attrs is None


class TestGetRecommendedRendererName:
    """Tests for renderer recommendation functions."""

    def test_llama_renderer(self):
        """Test renderer recommendation for Llama."""
        from tensafe.cookbook.model_info import get_recommended_renderer_name

        name = get_recommended_renderer_name("meta-llama/Llama-3.1-8B")
        assert name == "llama3"

    def test_qwen_renderer(self):
        """Test renderer recommendation for Qwen."""
        from tensafe.cookbook.model_info import get_recommended_renderer_name

        name = get_recommended_renderer_name("Qwen/Qwen3-8B")
        assert name == "qwen3"

    def test_deepseek_renderer(self):
        """Test renderer recommendation for DeepSeek."""
        from tensafe.cookbook.model_info import get_recommended_renderer_name

        name = get_recommended_renderer_name("deepseek-ai/DeepSeek-V3")
        assert name == "deepseek-v3"

    def test_unknown_model_fallback(self):
        """Test fallback for unknown model."""
        from tensafe.cookbook.model_info import get_recommended_renderer_name

        name = get_recommended_renderer_name("unknown/model")
        assert name == "role-colon"

    def test_get_recommended_renderer_names_list(self):
        """Test getting list of recommended renderers."""
        from tensafe.cookbook.model_info import get_recommended_renderer_names

        names = get_recommended_renderer_names("meta-llama/Llama-3.1-8B")

        assert isinstance(names, list)
        assert len(names) > 0
        assert "llama3" in names


class TestModelUtilities:
    """Tests for model utility functions."""

    def test_is_chat_model(self):
        """Test chat model detection."""
        from tensafe.cookbook.model_info import is_chat_model

        assert is_chat_model("meta-llama/Llama-3.1-8B-Instruct") is True
        assert is_chat_model("meta-llama/Llama-3.1-8B") is False

    def test_is_vision_model(self):
        """Test vision model detection."""
        from tensafe.cookbook.model_info import is_vision_model

        assert is_vision_model("meta-llama/Llama-3.2-11B-Vision") is True
        assert is_vision_model("meta-llama/Llama-3.1-8B") is False

    def test_is_thinking_model(self):
        """Test thinking model detection."""
        from tensafe.cookbook.model_info import is_thinking_model

        assert is_thinking_model("moonshotai/Kimi-K2-Thinking") is True
        assert is_thinking_model("meta-llama/Llama-3.1-8B") is False

    def test_get_context_length(self):
        """Test context length retrieval."""
        from tensafe.cookbook.model_info import get_context_length

        length = get_context_length("meta-llama/Llama-3.1-8B")
        assert length == 131072

        # Unknown model should return default
        length_unknown = get_context_length("unknown/model")
        assert length_unknown > 0
