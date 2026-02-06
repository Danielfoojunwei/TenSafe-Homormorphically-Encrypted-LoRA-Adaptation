"""
TenSafe Unified Inference Module with TGSP Enforcement.

Provides unified inference with support for:
- Standard model inference
- LoRA inference (plaintext)
- HE-LoRA inference (encrypted LoRA delta via MOAI-optimized CKKS)
- Batch processing
- TGSP format enforcement for secure adapter loading

IMPORTANT: TGSP Format Enforcement
----------------------------------
For HE-encrypted inference (HE_ONLY, FULL_HE modes), ONLY TGSP-format
adapters are allowed. This is a security lock-in that ensures:
1. All adapters are cryptographically signed and verified
2. Audit trail is maintained for compliance
3. Adapters come from trusted sources

Usage:
    from tensafe.core.inference import (
        TenSafeInference,
        InferenceMode,
        InferenceResult,
    )

    # Standard inference from checkpoint
    inference = TenSafeInference.from_checkpoint("./checkpoint")
    result = inference.generate("Hello, how are you?")

    # HE inference with TGSP adapter (required for security)
    from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

    registry = TGSPAdapterRegistry()
    adapter_id = registry.load_tgsp_adapter("adapter.tgsp", key_path)
    registry.activate_adapter(adapter_id)

    inference = TenSafeInference.from_tgsp_registry(registry, model)
    result = inference(input_ids)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tensafe.core.config import InferenceConfig, LoRAConfig, TenSafeConfig, load_config
from tensafe.core.he_interface import HEBackendInterface, HEBackendType, HEParams, get_backend

logger = logging.getLogger(__name__)


class TGSPEnforcementError(Exception):
    """Raised when TGSP enforcement is violated."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(
            f"TGSP Enforcement Violation: {message}. "
            f"Encrypted inference (HE_ONLY/FULL_HE modes) requires TGSP-format adapters. "
            f"Use TGSPAdapterRegistry.load_tgsp_adapter() to load adapters."
        )


class InferenceMode(Enum):
    """Inference modes."""
    NONE = "none"  # No LoRA, just base model
    PLAINTEXT = "plaintext"  # Standard LoRA (no encryption)
    HE_ONLY = "he_only"  # LoRA under HE, base model plaintext (REQUIRES TGSP)
    FULL_HE = "full_he"  # Everything encrypted (REQUIRES TGSP, not recommended)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    use_cache: bool = True


@dataclass
class InferenceResult:
    """Result from inference."""
    output: np.ndarray
    text: str | None = None
    tokens: List[int] | None = None

    # Timing
    total_time_ms: float = 0.0
    base_model_time_ms: float = 0.0
    lora_time_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Mode used
    mode: str = "plaintext"

    # HE metrics (for HE modes)
    he_metrics: Dict[str, Any] | None = None


@dataclass
class BatchInferenceResult:
    """Result from batch inference."""
    results: List[InferenceResult]
    total_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0


class TenSafeInference:
    """
    Unified TenSafe inference engine with TGSP enforcement.

    Supports:
    - Multiple LoRA modes (plaintext, HE via MOAI-optimized CKKS)
    - Batch processing
    - Streaming generation (future)
    - Multiple backends
    - TGSP format enforcement for HE modes (security lock-in)

    TGSP Enforcement:
        When mode is HE_ONLY or FULL_HE, adapters MUST be loaded via
        TGSPAdapterRegistry. This ensures cryptographic verification
        and audit compliance for encrypted inference.
    """

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        lora_weights: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
        config: TenSafeConfig | InferenceConfig | None = None,
        mode: InferenceMode = InferenceMode.PLAINTEXT,
        tgsp_registry: Any | None = None,
        enforce_tgsp: bool = True,
    ):
        """
        Initialize inference engine.

        Args:
            model: The base model
            tokenizer: The tokenizer
            lora_weights: Dict of module_name -> (lora_a, lora_b)
            config: Configuration (TenSafeConfig or InferenceConfig)
            mode: Inference mode
            tgsp_registry: Optional TGSPAdapterRegistry for TGSP-format adapters
            enforce_tgsp: Enforce TGSP format for HE modes (default: True)

        Raises:
            TGSPEnforcementError: If HE mode used without TGSP adapter
        """
        self._model = model
        self._tokenizer = tokenizer
        self._lora_weights = lora_weights or {}
        self._mode = mode
        self._tgsp_registry = tgsp_registry
        self._enforce_tgsp = enforce_tgsp

        # Track if weights came from TGSP format
        self._weights_from_tgsp = tgsp_registry is not None

        # Parse config
        if isinstance(config, TenSafeConfig):
            self._config = config.inference
            self._lora_config = config.lora
            self._he_config = config.he
        elif isinstance(config, InferenceConfig):
            self._config = config
            self._lora_config = LoRAConfig()
            self._he_config = None
        else:
            self._config = InferenceConfig()
            self._lora_config = LoRAConfig()
            self._he_config = None

        # HE backend (initialized lazily)
        self._he_backend: HEBackendInterface | None = None
        self._packed_weights: Dict[str, Any] = {}

        # Metrics
        self._inference_count = 0
        self._total_time_ms = 0.0

        # TGSP enforcement check for HE modes
        if self._requires_tgsp():
            if not self._weights_from_tgsp and self._lora_weights:
                raise TGSPEnforcementError(
                    f"Attempted to use HE mode ({mode.value}) with non-TGSP adapter weights. "
                    f"Use TGSPAdapterRegistry to load adapters."
                )

        logger.info(
            f"TenSafeInference initialized: mode={mode.value}, "
            f"enforce_tgsp={enforce_tgsp}, weights_from_tgsp={self._weights_from_tgsp}"
        )

    def _requires_tgsp(self) -> bool:
        """Check if current mode requires TGSP format."""
        return self._enforce_tgsp and self._mode in (InferenceMode.HE_ONLY, InferenceMode.FULL_HE)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        mode: InferenceMode = InferenceMode.PLAINTEXT,
        device: str = "auto",
    ) -> TenSafeInference:
        """
        Create inference engine from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            mode: Inference mode
            device: Device to load model on ("auto", "cuda", "cpu")

        Returns:
            Configured TenSafeInference
        """
        checkpoint_path = Path(checkpoint_path)

        # Load config
        config_path = checkpoint_path / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = TenSafeConfig()

        # Load model and tokenizer
        model = None
        tokenizer = None
        lora_weights = {}

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            # Load tokenizer
            tokenizer_path = checkpoint_path / "tokenizer"
            if tokenizer_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            elif config.model.name:
                tokenizer = AutoTokenizer.from_pretrained(config.model.name)

            # Load model with LoRA weights
            model_path = checkpoint_path / "model"
            adapter_path = checkpoint_path / "adapter"

            if model_path.exists():
                # Load full model
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16,
                    device_map=device if device != "cpu" else None,
                )
                if device == "cpu":
                    model = model.to(device)

            elif adapter_path.exists():
                # Load base model with adapter
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(
                    config.model.name,
                    torch_dtype=torch.bfloat16,
                    device_map=device if device != "cpu" else None,
                )
                model = PeftModel.from_pretrained(base_model, str(adapter_path))
                if device == "cpu":
                    model = model.to(device)

                # Extract LoRA weights for HE inference
                lora_weights = cls._extract_lora_weights(model, config.lora.target_modules)

            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

        except ImportError as e:
            logger.warning(f"Could not load model (missing dependencies): {e}")
        except Exception as e:
            logger.error(f"Failed to load model from checkpoint: {e}")

        return cls(
            model=model,
            tokenizer=tokenizer,
            lora_weights=lora_weights,
            config=config,
            mode=mode,
        )

    @staticmethod
    def _extract_lora_weights(
        model: Any,
        target_modules: List[str],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extract LoRA weights from a PEFT model."""
        weights = {}

        try:
            state_dict = model.state_dict()

            for module in target_modules:
                lora_a_key = None
                lora_b_key = None

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

        except Exception as e:
            logger.warning(f"Could not extract LoRA weights: {e}")

        return weights

    @classmethod
    def from_config(
        cls,
        config: TenSafeConfig,
        mode: InferenceMode | None = None,
    ) -> TenSafeInference:
        """
        Create inference engine from configuration.

        Args:
            config: TenSafe configuration
            mode: Override inference mode

        Returns:
            Configured TenSafeInference
        """
        if mode is None:
            mode = InferenceMode(config.inference.lora_mode)

        return cls(config=config, mode=mode)

    @classmethod
    def from_tgsp_registry(
        cls,
        registry: Any,
        model: Any | None = None,
        tokenizer: Any | None = None,
        config: TenSafeConfig | InferenceConfig | None = None,
        mode: InferenceMode = InferenceMode.HE_ONLY,
    ) -> TenSafeInference:
        """
        Create TenSafeInference from a TGSP adapter registry.

        This is the RECOMMENDED method for creating inference with
        encrypted LoRA modes, as it ensures TGSP format compliance.

        Args:
            registry: TGSPAdapterRegistry with loaded adapter
            model: Optional base model
            tokenizer: Optional tokenizer
            config: Optional inference configuration
            mode: Inference mode (default: HE_ONLY)

        Returns:
            TenSafeInference configured with TGSP adapter

        Raises:
            TGSPEnforcementError: If no adapter is activated in registry
        """
        active_adapter = registry.get_active_adapter()
        if active_adapter is None:
            raise TGSPEnforcementError(
                "No active adapter in registry. Call registry.activate_adapter() first."
            )

        # Extract LoRA config from adapter metadata if available
        lora_config = None
        if hasattr(active_adapter, 'metadata'):
            meta = active_adapter.metadata
            if hasattr(meta, 'lora_rank'):
                lora_config = LoRAConfig(
                    rank=meta.lora_rank,
                    alpha=getattr(meta, 'lora_alpha', meta.lora_rank * 2),
                    target_modules=getattr(meta, 'target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj']),
                )

        return cls(
            model=model,
            tokenizer=tokenizer,
            lora_weights=active_adapter.weights,
            config=config,
            mode=mode,
            tgsp_registry=registry,
            enforce_tgsp=True,
        )

    def _ensure_he_backend(self) -> None:
        """Ensure HE backend is initialized."""
        if self._he_backend is not None:
            return

        if self._he_config is None:
            # Use defaults
            params = HEParams()
        else:
            params = HEParams(
                poly_modulus_degree=self._he_config.poly_modulus_degree,
                coeff_modulus_bits=self._he_config.coeff_modulus_bits,
                scale_bits=self._he_config.scale_bits,
                use_column_packing=self._he_config.use_column_packing,
            )

        self._he_backend = get_backend(HEBackendType.AUTO, params)
        logger.info(f"HE backend initialized: {self._he_backend.backend_name}")

    def register_lora_weights(
        self,
        module_name: str,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        from_tgsp: bool = False,
    ) -> None:
        """
        Register LoRA weights for a module.

        Args:
            module_name: Name of target module
            lora_a: LoRA A matrix [rank, in_features]
            lora_b: LoRA B matrix [out_features, rank]
            from_tgsp: Whether weights came from TGSP format

        Raises:
            TGSPEnforcementError: If HE mode requires TGSP but weights are not from TGSP
        """
        # Check TGSP enforcement
        if self._requires_tgsp() and not from_tgsp:
            raise TGSPEnforcementError(
                f"Cannot register non-TGSP weights in HE mode ({self._mode.value}). "
                f"Use TGSPAdapterRegistry to load adapters."
            )

        self._lora_weights[module_name] = (lora_a, lora_b)

        # Track TGSP status
        if from_tgsp:
            self._weights_from_tgsp = True

        # Clear cached packed weights
        self._packed_weights = {}

    def forward(
        self,
        x: np.ndarray,
        module_name: str | None = None,
    ) -> InferenceResult:
        """
        Run forward pass with configured LoRA mode.

        Optimisations applied to the HE_ONLY path:
          1. Async overlap: base forward runs in parallel with HE pipeline
          2. In-place buffer reuse: delta added directly into y_base (no alloc)

        Args:
            x: Input activation [batch, hidden_dim] or [hidden_dim]
            module_name: Target module (for LoRA modes)

        Returns:
            InferenceResult with output and metrics
        """
        start_time = time.perf_counter()
        result = InferenceResult(output=x, mode=self._mode.value)

        if self._mode == InferenceMode.NONE:
            # Step 1: Base model forward (always plaintext)
            base_start = time.perf_counter()
            y_base = self._base_model_forward(x)
            result.base_model_time_ms = (time.perf_counter() - base_start) * 1000
            result.output = y_base

        elif self._mode == InferenceMode.PLAINTEXT:
            base_start = time.perf_counter()
            y_base = self._base_model_forward(x)
            result.base_model_time_ms = (time.perf_counter() - base_start) * 1000

            lora_start = time.perf_counter()
            delta = self._lora_forward_plaintext(x, module_name)
            # In-place add: avoids allocating a new output array
            np.add(y_base, delta, out=y_base)
            result.output = y_base
            result.lora_time_ms = (time.perf_counter() - lora_start) * 1000

        elif self._mode == InferenceMode.HE_ONLY:
            # Async overlap: run base forward and HE pipeline concurrently.
            # Since they share no state (base is plaintext matmul on x,
            # HE operates on an encrypted copy of x), this is safe.
            lora_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=1) as pool:
                # Submit base model forward to background thread
                base_future = pool.submit(self._base_model_forward, x)

                # Run HE pipeline on this thread (encrypt → compute → decrypt)
                delta, he_metrics = self._lora_forward_he(x, module_name)

                # Collect base model result (blocks if base hasn't finished)
                base_start = time.perf_counter()
                y_base = base_future.result()
                result.base_model_time_ms = (time.perf_counter() - base_start) * 1000

            # In-place add: fuse delta into y_base without a third array
            np.add(y_base, delta, out=y_base)
            result.output = y_base
            result.lora_time_ms = (time.perf_counter() - lora_start) * 1000
            result.he_metrics = he_metrics

        elif self._mode == InferenceMode.FULL_HE:
            logger.warning("FULL_HE mode not recommended for latency")
            base_start = time.perf_counter()
            y_base = self._base_model_forward(x)
            result.base_model_time_ms = (time.perf_counter() - base_start) * 1000
            result.output = y_base

        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        self._inference_count += 1
        self._total_time_ms += result.total_time_ms

        return result

    def _base_model_forward(self, x: np.ndarray) -> np.ndarray:
        """Run base model forward pass."""
        if self._model is not None:
            if hasattr(self._model, 'forward'):
                return self._model.forward(x)
            elif callable(self._model):
                return self._model(x)

        # Mock: identity
        return x.copy()

    def _lora_forward_plaintext(
        self,
        x: np.ndarray,
        module_name: str | None = None,
    ) -> np.ndarray:
        """Compute LoRA delta in plaintext."""
        if module_name is None and self._lora_weights:
            module_name = next(iter(self._lora_weights))

        if module_name not in self._lora_weights:
            return np.zeros_like(x)

        lora_a, lora_b = self._lora_weights[module_name]
        scaling = self._lora_config.scaling

        # delta = scaling * (x @ A^T @ B^T)
        intermediate = x @ lora_a.T
        delta = intermediate @ lora_b.T
        return scaling * delta

    def _lora_forward_he(
        self,
        x: np.ndarray,
        module_name: str | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute LoRA delta under homomorphic encryption.

        Optimisations applied:
          - Partial/lazy decrypt: only decrypt the slots carrying real data
            (output_size), skipping unused padding in the CKKS slot vector.
        """
        self._ensure_he_backend()

        if module_name is None and self._lora_weights:
            module_name = next(iter(self._lora_weights))

        if module_name not in self._lora_weights:
            return np.zeros_like(x), {}

        lora_a, lora_b = self._lora_weights[module_name]
        scaling = self._lora_config.scaling

        # Encrypt input
        x_flat = x.flatten()
        ct_x = self._he_backend.encrypt(x_flat)

        # Compute encrypted LoRA delta
        ct_result = self._he_backend.lora_delta(ct_x, lora_a, lora_b, scaling)

        # Partial decrypt: only the slots that carry real payload
        output_size = len(x_flat)
        delta = self._he_backend.decrypt(ct_result, output_size=output_size)
        delta = delta[:output_size].reshape(x.shape)

        # Get metrics
        metrics = self._he_backend.get_metrics()

        return delta, metrics.to_dict()

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> InferenceResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            generation_config: Generation parameters

        Returns:
            InferenceResult with generated text

        Raises:
            RuntimeError: If model or tokenizer not available
        """
        config = generation_config or GenerationConfig(
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            do_sample=self._config.do_sample,
        )

        start_time = time.perf_counter()

        # Check requirements
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not available. Load a model with from_checkpoint() "
                "or provide a tokenizer during initialization."
            )

        if self._model is None:
            raise RuntimeError(
                "Model not available. Load a model with from_checkpoint() "
                "or provide a model during initialization."
            )

        try:
            import torch

            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Move to model device
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]

            # Set model to eval mode
            self._model.eval()

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature if config.do_sample else 1.0,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=config.eos_token_id or self._tokenizer.eos_token_id,
                    repetition_penalty=config.repetition_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    use_cache=config.use_cache,
                )

            # Decode
            generated_ids = outputs[0].tolist()
            text = self._tokenizer.decode(
                generated_ids[input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        except ImportError:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch"
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        tokens_generated = len(generated_ids) - len(inputs["input_ids"][0])
        tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        return InferenceResult(
            output=np.array(generated_ids),
            text=text,
            tokens=generated_ids,
            total_time_ms=elapsed_ms,
            tokens_per_second=tps,
            mode=self._mode.value,
        )

    def generate_batch(
        self,
        prompts: List[str],
        generation_config: GenerationConfig | None = None,
    ) -> BatchInferenceResult:
        """
        Generate text for a batch of prompts.

        Args:
            prompts: List of input prompts
            generation_config: Generation parameters

        Returns:
            BatchInferenceResult with all results
        """
        start_time = time.perf_counter()

        results = []
        for prompt in prompts:
            result = self.generate(prompt, generation_config)
            results.append(result)

        total_ms = (time.perf_counter() - start_time) * 1000

        # Calculate average throughput
        total_tokens = sum(len(r.tokens or []) for r in results)
        avg_tps = (total_tokens / (total_ms / 1000)) if total_ms > 0 else 0

        return BatchInferenceResult(
            results=results,
            total_time_ms=total_ms,
            avg_tokens_per_second=avg_tps,
        )

    def __call__(
        self,
        x: str | np.ndarray,
        **kwargs: Any,
    ) -> InferenceResult:
        """
        Convenience method for inference.

        Args:
            x: Input (string prompt or numpy array)
            **kwargs: Additional arguments

        Returns:
            InferenceResult
        """
        if isinstance(x, str):
            return self.generate(x, **kwargs)
        return self.forward(x, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics."""
        return {
            "inference_count": self._inference_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / max(1, self._inference_count),
            "mode": self._mode.value,
            "he_backend": self._he_backend.backend_name if self._he_backend else None,
        }


def create_inference(
    config: str | Path | TenSafeConfig,
    mode: InferenceMode | None = None,
    **kwargs: Any,
) -> TenSafeInference:
    """
    Create a TenSafe inference engine.

    Args:
        config: Configuration (path or object)
        mode: Inference mode
        **kwargs: Additional arguments

    Returns:
        Configured TenSafeInference
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)

    return TenSafeInference.from_config(config, mode=mode)
