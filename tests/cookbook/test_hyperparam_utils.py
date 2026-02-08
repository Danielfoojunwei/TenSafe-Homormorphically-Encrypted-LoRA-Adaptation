"""Tests for cookbook hyperparameter utilities."""



class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from tensafe.cookbook.hyperparam_utils import LoRAConfig

        config = LoRAConfig()

        assert config.rank == 32
        assert config.alpha == 64.0
        assert config.dropout == 0.0
        assert config.bias == "none"

    def test_custom_values(self):
        """Test custom configuration values."""
        from tensafe.cookbook.hyperparam_utils import LoRAConfig

        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_scaling_property(self):
        """Test scaling factor computation."""
        from tensafe.cookbook.hyperparam_utils import LoRAConfig

        config = LoRAConfig(rank=32, alpha=64.0)
        assert config.scaling == 2.0

        config2 = LoRAConfig(rank=16, alpha=32.0)
        assert config2.scaling == 2.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from tensafe.cookbook.hyperparam_utils import LoRAConfig

        config = LoRAConfig(rank=8, alpha=16.0)
        d = config.to_dict()

        assert d["rank"] == 8
        assert d["alpha"] == 16.0
        assert "target_modules" in d


class TestHyperparamFunctions:
    """Tests for hyperparameter utility functions."""

    def test_get_lora_lr_over_full_finetune_lr(self):
        """Test LR scaling factor."""
        from tensafe.cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr

        factor = get_lora_lr_over_full_finetune_lr()
        assert factor == 10.0

    def test_get_hidden_size_known_model(self):
        """Test hidden size for known model."""
        from tensafe.cookbook.hyperparam_utils import get_hidden_size

        size = get_hidden_size("meta-llama/Llama-3.1-8B")
        assert size == 4096

    def test_get_hidden_size_unknown_model(self):
        """Test hidden size fallback for unknown model."""
        from tensafe.cookbook.hyperparam_utils import get_hidden_size

        # Should return default or try to load config
        size = get_hidden_size("unknown/model")
        assert isinstance(size, int)
        assert size > 0

    def test_get_lora_param_count(self):
        """Test LoRA parameter counting."""
        from tensafe.cookbook.hyperparam_utils import get_lora_param_count

        count = get_lora_param_count(
            model_name="meta-llama/Llama-3.1-8B",
            rank=32,
            num_layers=32,
        )

        assert isinstance(count, int)
        assert count > 0

    def test_get_lora_param_count_detailed(self):
        """Test detailed LoRA parameter breakdown."""
        from tensafe.cookbook.hyperparam_utils import get_lora_param_count

        breakdown = get_lora_param_count(
            model_name="meta-llama/Llama-3.1-8B",
            rank=32,
            num_layers=32,
            detailed=True,
        )

        assert isinstance(breakdown, dict)
        assert "total" in breakdown

    def test_get_lora_lr(self):
        """Test LoRA learning rate computation."""
        from tensafe.cookbook.hyperparam_utils import get_lora_lr

        lr = get_lora_lr("meta-llama/Llama-3.1-8B")
        assert isinstance(lr, float)
        assert lr > 0

        # Larger model should have smaller LR
        lr_70b = get_lora_lr("meta-llama/Llama-3.1-70B")
        assert lr_70b < lr

    def test_get_lora_lr_multiplier(self):
        """Test LR multiplier for different models."""
        from tensafe.cookbook.hyperparam_utils import get_lora_lr_multiplier

        mult_llama = get_lora_lr_multiplier("meta-llama/Llama-3.1-8B")
        mult_deepseek = get_lora_lr_multiplier("deepseek-ai/DeepSeek-V3")

        assert mult_llama == 1.0
        assert mult_deepseek == 0.8

    def test_calculate_warmup_steps(self):
        """Test warmup steps calculation."""
        from tensafe.cookbook.hyperparam_utils import calculate_warmup_steps

        warmup = calculate_warmup_steps(total_steps=10000, warmup_ratio=0.03)
        assert warmup == 300  # 3% of 10000

        # Test min/max bounds
        warmup_small = calculate_warmup_steps(total_steps=100, warmup_ratio=0.03)
        assert warmup_small >= 100  # min_warmup

        warmup_large = calculate_warmup_steps(
            total_steps=100000, warmup_ratio=0.1, max_warmup=1000
        )
        assert warmup_large <= 1000  # max_warmup

    def test_get_recommended_batch_size(self):
        """Test batch size recommendations."""
        from tensafe.cookbook.hyperparam_utils import get_recommended_batch_size

        micro, accum = get_recommended_batch_size("meta-llama/Llama-3.1-8B")
        assert micro > 0
        assert accum > 0

        # 70B should have smaller micro batch
        micro_70b, accum_70b = get_recommended_batch_size("meta-llama/Llama-3.1-70B")
        assert micro_70b <= micro
        assert accum_70b >= accum
