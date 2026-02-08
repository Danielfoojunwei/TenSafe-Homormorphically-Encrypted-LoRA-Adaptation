"""
Unified ML Backend Interface and Implementations.

This module provides a unified interface for machine learning operations
that abstracts over different ML frameworks (PyTorch, JAX, etc.).

Production systems should use TorchMLBackend for real training.
MockMLBackend is available ONLY for testing (not in production paths).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class MLBackendConfig:
    """Configuration for ML backends."""

    # Model configuration
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    model_revision: Optional[str] = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False

    # LoRA configuration
    lora_enabled: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Optimizer configuration
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Training configuration
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "lora_enabled": self.lora_enabled,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_target_modules": self.lora_target_modules,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
        }


@dataclass
class DPConfig:
    """Differential privacy configuration."""

    enabled: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_epsilon: Optional[float] = 8.0
    target_delta: Optional[float] = 1e-5


@dataclass
class ForwardBackwardResult:
    """Result from forward-backward pass."""

    loss: float
    grad_norm: float
    tokens_processed: int
    time_ms: float

    # DP metrics
    dp_metrics: Optional[Dict[str, Any]] = None

    # Additional metrics
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimStepResult:
    """Result from optimizer step."""

    step: int
    learning_rate: float
    time_ms: float

    # DP metrics
    dp_metrics: Optional[Dict[str, Any]] = None


@dataclass
class SampleResult:
    """Result from sampling/generation."""

    samples: List[Dict[str, Any]]
    model_step: int
    time_ms: float


@dataclass
class StateInfo:
    """Information about saved state."""

    step: int
    model_hash: str
    size_bytes: int
    includes_optimizer: bool


# ==============================================================================
# ML Backend Interface
# ==============================================================================


class MLBackendInterface(ABC):
    """
    Abstract interface for ML backends.

    This interface defines all operations needed for training and inference.
    Implementations should handle:
    - Model initialization (with LoRA)
    - Forward-backward passes
    - Optimizer steps
    - Sampling/generation
    - State serialization
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get human-readable backend name."""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        pass

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Get current training step."""
        pass

    @abstractmethod
    def initialize(self, config: MLBackendConfig) -> None:
        """
        Initialize the model and optimizer.

        Args:
            config: Backend configuration
        """
        pass

    @abstractmethod
    def forward_backward(
        self,
        batch: Dict[str, Any],
        dp_config: Optional[DPConfig] = None,
    ) -> ForwardBackwardResult:
        """
        Execute forward-backward pass.

        Args:
            batch: Training batch with input_ids, labels, attention_mask
            dp_config: Optional DP configuration

        Returns:
            ForwardBackwardResult with loss and gradient information
        """
        pass

    @abstractmethod
    def optim_step(
        self,
        apply_dp_noise: bool = True,
        dp_config: Optional[DPConfig] = None,
    ) -> OptimStepResult:
        """
        Execute optimizer step.

        Args:
            apply_dp_noise: Whether to apply DP noise to gradients
            dp_config: Optional DP configuration

        Returns:
            OptimStepResult with step information
        """
        pass

    @abstractmethod
    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
    ) -> SampleResult:
        """
        Generate samples from the model.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences that stop generation

        Returns:
            SampleResult with generated samples
        """
        pass

    @abstractmethod
    def save_state(self, include_optimizer: bool = True) -> bytes:
        """
        Serialize model state to bytes.

        Args:
            include_optimizer: Whether to include optimizer state

        Returns:
            Serialized state bytes
        """
        pass

    @abstractmethod
    def load_state(self, state_bytes: bytes) -> int:
        """
        Load model state from bytes.

        Args:
            state_bytes: Serialized state

        Returns:
            Loaded step number
        """
        pass

    def get_lora_weights(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get LoRA weights for HE inference.

        Returns:
            Dict of module_name -> (lora_a, lora_b)
        """
        return {}


# ==============================================================================
# PyTorch ML Backend
# ==============================================================================


class TorchMLBackend(MLBackendInterface):
    """
    Production ML backend using PyTorch and HuggingFace Transformers.

    This is the recommended backend for production training with:
    - PEFT/LoRA integration
    - Gradient checkpointing
    - Mixed precision training
    - DP-SGD support
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._scheduler = None
        self._config: Optional[MLBackendConfig] = None
        self._step = 0
        self._gradients = None
        self._device = None

    @property
    def backend_name(self) -> str:
        return "PyTorch/Transformers"

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    @property
    def current_step(self) -> int:
        return self._step

    def initialize(self, config: MLBackendConfig) -> None:
        """Initialize model with LoRA and optimizer."""
        import torch

        self._config = config
        start_time = time.perf_counter()

        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(f"Using device: {self._device}")

        # Load model and tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                revision=config.model_revision,
                trust_remote_code=config.trust_remote_code,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                revision=config.model_revision,
                torch_dtype=torch_dtype,
                device_map=config.device_map if config.device_map != "auto" else None,
                trust_remote_code=config.trust_remote_code,
            )

            # Move to device if not using device_map
            if config.device_map != "auto":
                self._model = self._model.to(self._device)

            # Apply LoRA if enabled
            if config.lora_enabled:
                self._apply_lora(config)

            # Enable gradient checkpointing
            if config.gradient_checkpointing:
                self._model.gradient_checkpointing_enable()

            # Create optimizer
            self._create_optimizer(config)

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Model initialized: {config.model_name_or_path}, "
                f"LoRA={config.lora_enabled}, "
                f"time={elapsed:.1f}ms"
            )

        except ImportError as e:
            raise RuntimeError(
                f"PyTorch/Transformers not available: {e}. "
                "Install with: pip install torch transformers peft"
            )

    def _apply_lora(self, config: MLBackendConfig) -> None:
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )

            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()

        except ImportError as e:
            raise RuntimeError(
                f"PEFT not available for LoRA: {e}. "
                "Install with: pip install peft"
            )

    def _create_optimizer(self, config: MLBackendConfig) -> None:
        """Create optimizer for trainable parameters."""
        import torch.optim as optim

        trainable_params = [p for p in self._model.parameters() if p.requires_grad]

        if config.optimizer.lower() == "adamw":
            self._optimizer = optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )
        elif config.optimizer.lower() == "adam":
            self._optimizer = optim.Adam(
                trainable_params,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps,
            )
        elif config.optimizer.lower() == "sgd":
            self._optimizer = optim.SGD(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def forward_backward(
        self,
        batch: Dict[str, Any],
        dp_config: Optional[DPConfig] = None,
    ) -> ForwardBackwardResult:
        """Execute forward-backward pass."""

        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        # Prepare batch
        input_ids = self._to_tensor(batch.get("input_ids"))
        labels = self._to_tensor(batch.get("labels", batch.get("input_ids")))
        attention_mask = self._to_tensor(batch.get("attention_mask"))

        # Forward pass
        self._model.train()
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Compute gradient norm
        grad_norm = self._compute_grad_norm()

        # Apply gradient clipping for DP
        dp_metrics = None
        if dp_config and dp_config.enabled:
            clipped_norm, was_clipped = self._clip_gradients(dp_config.max_grad_norm)
            dp_metrics = {
                "noise_applied": False,  # Noise applied in optim_step
                "epsilon_spent": 0.0,
                "grad_norm_before_clip": grad_norm,
                "grad_norm_after_clip": clipped_norm,
                "num_clipped": 1 if was_clipped else 0,
            }
            grad_norm = clipped_norm

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        tokens_processed = input_ids.numel()

        return ForwardBackwardResult(
            loss=loss.item(),
            grad_norm=grad_norm,
            tokens_processed=tokens_processed,
            time_ms=elapsed_ms,
            dp_metrics=dp_metrics,
        )

    def _to_tensor(self, data: Any):
        """Convert data to tensor on correct device."""
        import torch

        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self._device)
        if isinstance(data, (list, tuple)):
            return torch.tensor(data).to(self._device)
        return data

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm."""

        total_norm = 0.0
        for p in self._model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _clip_gradients(self, max_grad_norm: float) -> Tuple[float, bool]:
        """Clip gradients and return clipped norm."""
        import torch

        total_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            max_grad_norm,
        )
        return float(total_norm), total_norm > max_grad_norm

    def optim_step(
        self,
        apply_dp_noise: bool = True,
        dp_config: Optional[DPConfig] = None,
    ) -> OptimStepResult:
        """Execute optimizer step."""

        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        start_time = time.perf_counter()

        # Apply DP noise if configured
        dp_metrics = None
        if dp_config and dp_config.enabled and apply_dp_noise:
            epsilon_spent = self._apply_dp_noise(dp_config)
            dp_metrics = {
                "noise_applied": True,
                "epsilon_spent": epsilon_spent,
                "delta": dp_config.target_delta,
            }

        # Optimizer step
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._step += 1

        # Get learning rate
        lr = self._optimizer.param_groups[0]["lr"]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return OptimStepResult(
            step=self._step,
            learning_rate=lr,
            time_ms=elapsed_ms,
            dp_metrics=dp_metrics,
        )

    def _apply_dp_noise(self, dp_config: DPConfig) -> float:
        """Apply DP noise to gradients and return epsilon spent."""
        import math

        import torch

        noise_scale = dp_config.noise_multiplier * dp_config.max_grad_norm

        for p in self._model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=p.grad.shape,
                    device=p.grad.device,
                    dtype=p.grad.dtype,
                )
                p.grad.add_(noise)

        # Simple epsilon calculation (actual should use RDP)
        delta = dp_config.target_delta or 1e-5
        epsilon = math.sqrt(2 * math.log(1.25 / delta)) / dp_config.noise_multiplier

        return epsilon

    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
    ) -> SampleResult:
        """Generate samples from the model."""
        import torch

        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        start_time = time.perf_counter()
        self._model.eval()

        samples = []
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._device)

                # Generate
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

                # Decode
                completion = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                samples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "tokens_generated": len(outputs[0]) - inputs["input_ids"].shape[1],
                    "finish_reason": "stop",
                })

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SampleResult(
            samples=samples,
            model_step=self._step,
            time_ms=elapsed_ms,
        )

    def save_state(self, include_optimizer: bool = True) -> bytes:
        """Serialize model state."""
        import io

        import torch

        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        state = {
            "step": self._step,
            "model_state_dict": self._model.state_dict(),
            "config": self._config.to_dict() if self._config else {},
        }

        if include_optimizer and self._optimizer is not None:
            state["optimizer_state_dict"] = self._optimizer.state_dict()

        buffer = io.BytesIO()
        torch.save(state, buffer)
        return buffer.getvalue()

    def load_state(self, state_bytes: bytes) -> int:
        """Load model state from bytes."""
        import io

        import torch

        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        buffer = io.BytesIO(state_bytes)
        state = torch.load(buffer, map_location=self._device, weights_only=False)

        self._model.load_state_dict(state["model_state_dict"])

        if "optimizer_state_dict" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer_state_dict"])

        self._step = state.get("step", 0)

        return self._step

    def get_lora_weights(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extract LoRA weights for HE inference."""
        if not self.is_initialized or not self._config.lora_enabled:
            return {}

        weights = {}
        state_dict = self._model.state_dict()

        for module in self._config.lora_target_modules:
            lora_a_key = None
            lora_b_key = None

            # Find matching keys
            for key in state_dict.keys():
                if module in key:
                    if "lora_A" in key or "lora_a" in key:
                        lora_a_key = key
                    elif "lora_B" in key or "lora_b" in key:
                        lora_b_key = key

            if lora_a_key and lora_b_key:
                lora_a = state_dict[lora_a_key].cpu().numpy()
                lora_b = state_dict[lora_b_key].cpu().numpy()
                weights[module] = (lora_a, lora_b)

        return weights


# ==============================================================================
# Backend Factory
# ==============================================================================


_BACKENDS: Dict[str, Type[MLBackendInterface]] = {
    "torch": TorchMLBackend,
    "pytorch": TorchMLBackend,
}


def register_ml_backend(name: str, backend_class: Type[MLBackendInterface]) -> None:
    """Register a new ML backend."""
    _BACKENDS[name.lower()] = backend_class


def get_ml_backend(
    backend_type: str = "torch",
    config: Optional[MLBackendConfig] = None,
    initialize: bool = True,
) -> MLBackendInterface:
    """
    Get an ML backend instance.

    Args:
        backend_type: Type of backend ("torch", "pytorch")
        config: Optional configuration
        initialize: Whether to initialize the backend

    Returns:
        MLBackendInterface instance
    """
    backend_class = _BACKENDS.get(backend_type.lower())

    if backend_class is None:
        available = list(_BACKENDS.keys())
        raise ValueError(
            f"Unknown ML backend: {backend_type}. Available: {available}"
        )

    backend = backend_class()

    if initialize and config is not None:
        backend.initialize(config)

    return backend


def list_available_ml_backends() -> List[str]:
    """List available ML backend types."""
    return list(_BACKENDS.keys())


def is_ml_backend_available(backend_type: str) -> bool:
    """Check if an ML backend is available."""
    if backend_type.lower() in ("torch", "pytorch"):
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    return False
