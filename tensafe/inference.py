"""
TenSafe Inference Module with HE-LoRA Support.

Provides inference with the following modes:
- plaintext: Standard inference without encryption
- he_only: Base model in plaintext, LoRA delta under HE (MOAI-optimized)
- full_he: Full encrypted inference (not recommended for latency)

The he_only mode follows the architecture:
    y = y_base + decrypt(HE_LoRA_Delta(encrypt(x)))

Where:
    - y_base = frozen_model(x) runs in plaintext (fast)
    - LoRA delta is computed under CKKS HE with MOAI optimizations

This achieves privacy for the LoRA contribution while keeping
base model inference fast.

IMPORTANT: TGSP Format Enforcement
----------------------------------
For HE-encrypted inference (he_only, full_he modes), ONLY TGSP-format
adapters are allowed. This is a security lock-in that ensures:
1. All adapters are cryptographically signed and verified
2. Audit trail is maintained for compliance
3. Adapters come from trusted sources

Use the TGSPAdapterRegistry for modular adapter hot-swapping:

    from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

    registry = TGSPAdapterRegistry()
    adapter_id = registry.load_tgsp_adapter("adapter.tgsp", key_path)
    registry.activate_adapter(adapter_id)

    # Use with inference
    inference = TenSafeInference.from_tgsp_registry(registry, model)
    output = inference(input_ids)

Usage:
    from tensafe.inference import TenSafeInference, LoRAMode

    inference = TenSafeInference(model, lora_mode=LoRAMode.HE_ONLY)
    output = inference(input_ids)
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LoRAMode(Enum):
    """LoRA inference modes."""

    # No LoRA, just base model
    NONE = "none"

    # LoRA in plaintext (standard PEFT)
    PLAINTEXT = "plaintext"

    # LoRA delta under HE, base model plaintext (MOAI-optimized)
    # REQUIRES TGSP format adapter when enforce_tgsp=True
    HE_ONLY = "he_only"

    # Full HE inference (not recommended)
    # REQUIRES TGSP format adapter when enforce_tgsp=True
    FULL_HE = "full_he"


class TGSPEnforcementError(Exception):
    """Raised when TGSP enforcement is violated."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(
            f"TGSP Enforcement Violation: {message}. "
            f"Encrypted inference (HE_ONLY/FULL_HE modes) requires TGSP-format adapters. "
            f"Use TGSPAdapterRegistry.load_tgsp_adapter() to load adapters."
        )


@dataclass
class InferenceConfig:
    """Configuration for TenSafe inference."""

    # LoRA mode
    lora_mode: LoRAMode = LoRAMode.PLAINTEXT

    # LoRA parameters (used for HE_ONLY mode)
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: List[str] = None

    # HE parameters (used for HE_ONLY and FULL_HE modes)
    he_poly_modulus_degree: int = 8192
    he_coeff_modulus_bits: List[int] = None
    he_scale_bits: int = 40

    # TGSP enforcement (CRITICAL for security lock-in)
    # When True, HE_ONLY and FULL_HE modes REQUIRE TGSP format adapters
    enforce_tgsp: bool = True

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        if self.he_coeff_modulus_bits is None:
            self.he_coeff_modulus_bits = [60, 40, 40, 60]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "InferenceConfig":
        """Create config from command line arguments."""
        return cls(
            lora_mode=LoRAMode(getattr(args, "lora_mode", "plaintext")),
            lora_rank=getattr(args, "lora_rank", 16),
            lora_alpha=getattr(args, "lora_alpha", 32.0),
            max_new_tokens=getattr(args, "max_new_tokens", 128),
            temperature=getattr(args, "temperature", 0.7),
            enforce_tgsp=getattr(args, "enforce_tgsp", True),
        )

    def requires_tgsp(self) -> bool:
        """Check if current mode requires TGSP format."""
        return self.enforce_tgsp and self.lora_mode in (LoRAMode.HE_ONLY, LoRAMode.FULL_HE)


@dataclass
class InferenceResult:
    """Result from TenSafe inference."""

    output: np.ndarray
    generated_tokens: Optional[List[int]] = None
    generated_text: Optional[str] = None

    # Timing
    total_time_ms: float = 0.0
    base_model_time_ms: float = 0.0
    lora_time_ms: float = 0.0

    # HE metrics (for HE_ONLY mode)
    he_metrics: Optional[Dict[str, Any]] = None

    # Mode used
    lora_mode: str = "plaintext"


class TenSafeInference:
    """
    TenSafe inference with HE-LoRA support.

    This class wraps a base model and LoRA adapter, providing
    inference with optional homomorphic encryption on the LoRA path.

    The key architecture for HE_ONLY mode:
        1. Base model forward pass (plaintext, fast)
        2. LoRA delta under HE (encrypted, MOAI-optimized)
        3. Add decrypted delta to base output

    This ensures the frozen model path is never encrypted, maintaining
    low latency while providing privacy for LoRA contributions.

    TGSP Enforcement:
        When enforce_tgsp=True (default), HE_ONLY and FULL_HE modes
        REQUIRE adapters to be loaded via TGSPAdapterRegistry. This
        ensures cryptographic verification and audit compliance.
    """

    def __init__(
        self,
        base_model: Optional[Any] = None,
        lora_weights: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        config: Optional[InferenceConfig] = None,
        tgsp_registry: Optional[Any] = None,
    ):
        """
        Initialize TenSafe inference.

        Args:
            base_model: Base model (or mock for testing)
            lora_weights: Dict of module_name -> (lora_a, lora_b)
            config: Inference configuration
            tgsp_registry: Optional TGSPAdapterRegistry for TGSP-format adapters
        """
        self.config = config or InferenceConfig()
        self._base_model = base_model
        self._lora_weights = lora_weights or {}
        self._tgsp_registry = tgsp_registry

        # Track if weights came from TGSP format
        self._weights_from_tgsp = tgsp_registry is not None

        # HE-LoRA adapter (initialized lazily)
        self._he_adapter = None

        # Metrics
        self._inference_count = 0
        self._total_time_ms = 0.0

        # TGSP enforcement check
        if self.config.requires_tgsp():
            if not self._weights_from_tgsp and self._lora_weights:
                raise TGSPEnforcementError(
                    f"Attempted to use HE mode ({self.config.lora_mode.value}) "
                    f"with non-TGSP adapter weights. Use TGSPAdapterRegistry to load adapters."
                )

        logger.info(
            f"TenSafeInference initialized: mode={self.config.lora_mode.value}, "
            f"enforce_tgsp={self.config.enforce_tgsp}, "
            f"weights_from_tgsp={self._weights_from_tgsp}"
        )

        # Initialize HE adapter if needed
        if self.config.lora_mode in (LoRAMode.HE_ONLY, LoRAMode.FULL_HE):
            self._init_he_adapter()

    @classmethod
    def from_tgsp_registry(
        cls,
        registry: Any,
        base_model: Optional[Any] = None,
        config: Optional[InferenceConfig] = None,
    ) -> "TenSafeInference":
        """
        Create TenSafeInference from a TGSP adapter registry.

        This is the RECOMMENDED method for creating inference with
        encrypted LoRA modes, as it ensures TGSP format compliance.

        Args:
            registry: TGSPAdapterRegistry with loaded adapter
            base_model: Optional base model
            config: Optional inference configuration

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

        # Create config if not provided
        if config is None:
            config = InferenceConfig(
                lora_mode=LoRAMode.HE_ONLY,
                lora_rank=active_adapter.metadata.lora_rank,
                lora_alpha=active_adapter.metadata.lora_alpha,
                lora_target_modules=active_adapter.metadata.target_modules,
                enforce_tgsp=True,
            )

        return cls(
            base_model=base_model,
            lora_weights=active_adapter.weights,
            config=config,
            tgsp_registry=registry,
        )

    def _init_he_adapter(self) -> None:
        """Initialize the HE-LoRA adapter."""
        try:
            from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

            he_config = HELoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                poly_modulus_degree=self.config.he_poly_modulus_degree,
                coeff_modulus_bits=self.config.he_coeff_modulus_bits,
                scale_bits=self.config.he_scale_bits,
            )

            self._he_adapter = HELoRAAdapter(he_config)

            # Register LoRA weights
            for module_name, (lora_a, lora_b) in self._lora_weights.items():
                self._he_adapter.register_weights(
                    module_name=module_name,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                )

            logger.info(
                f"HE-LoRA adapter initialized: "
                f"registered {len(self._lora_weights)} modules"
            )

        except ImportError as e:
            raise RuntimeError(
                f"HE-LoRA mode requires N2HE-HEXL backend: {e}\n"
                "Build with: ./scripts/build_n2he_hexl.sh"
            )

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
            from_tgsp: Flag indicating weights came from TGSP format

        Raises:
            TGSPEnforcementError: If HE mode requires TGSP but weights are not from TGSP
        """
        # Enforce TGSP for HE modes
        if self.config.requires_tgsp() and not from_tgsp:
            raise TGSPEnforcementError(
                f"Cannot register non-TGSP weights when using {self.config.lora_mode.value} mode. "
                f"Use TGSPAdapterRegistry to load adapters, or set enforce_tgsp=False (not recommended)."
            )

        self._lora_weights[module_name] = (lora_a, lora_b)

        # Track TGSP provenance
        if from_tgsp:
            self._weights_from_tgsp = True

        # Also register with HE adapter if initialized
        if self._he_adapter is not None:
            self._he_adapter.register_weights(
                module_name=module_name,
                lora_a=lora_a,
                lora_b=lora_b,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )

    def forward(
        self,
        x: np.ndarray,
        module_name: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run forward pass with configured LoRA mode.

        Args:
            x: Input activation [batch, hidden_dim] or [hidden_dim]
            module_name: Target module (for LoRA modes)

        Returns:
            InferenceResult with output and metrics
        """
        start_time = time.perf_counter()
        result = InferenceResult(output=x, lora_mode=self.config.lora_mode.value)

        # Step 1: Base model forward (always plaintext)
        base_start = time.perf_counter()
        if self._base_model is not None:
            y_base = self._base_model_forward(x)
        else:
            # Mock: identity for testing
            y_base = x.copy()
        result.base_model_time_ms = (time.perf_counter() - base_start) * 1000

        # Step 2: Apply LoRA based on mode
        if self.config.lora_mode == LoRAMode.NONE:
            result.output = y_base

        elif self.config.lora_mode == LoRAMode.PLAINTEXT:
            lora_start = time.perf_counter()
            delta = self._lora_forward_plaintext(x, module_name)
            result.output = y_base + delta
            result.lora_time_ms = (time.perf_counter() - lora_start) * 1000

        elif self.config.lora_mode == LoRAMode.HE_ONLY:
            # CRITICAL: Base model is plaintext, only LoRA delta is encrypted
            lora_start = time.perf_counter()
            delta = self._lora_forward_he(x, module_name)
            result.output = y_base + delta
            result.lora_time_ms = (time.perf_counter() - lora_start) * 1000

            # Get HE metrics
            if self._he_adapter is not None:
                result.he_metrics = self._he_adapter.get_last_metrics()
                if result.he_metrics:
                    result.he_metrics = result.he_metrics.to_log_dict()

        elif self.config.lora_mode == LoRAMode.FULL_HE:
            # Not recommended - encrypts everything
            logger.warning("FULL_HE mode not recommended for latency")
            result.output = y_base  # Placeholder

        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        self._inference_count += 1
        self._total_time_ms += result.total_time_ms

        return result

    def _base_model_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run base model forward pass (always plaintext).

        Args:
            x: Input

        Returns:
            Base model output
        """
        if hasattr(self._base_model, 'forward'):
            return self._base_model.forward(x)
        elif callable(self._base_model):
            return self._base_model(x)
        else:
            # Mock: simple linear transformation
            return x

    def _lora_forward_plaintext(
        self,
        x: np.ndarray,
        module_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute LoRA delta in plaintext.

        Args:
            x: Input activation
            module_name: Target module

        Returns:
            LoRA delta
        """
        if module_name is None and self._lora_weights:
            module_name = next(iter(self._lora_weights))

        if module_name not in self._lora_weights:
            return np.zeros_like(x)

        lora_a, lora_b = self._lora_weights[module_name]
        scaling = self.config.lora_alpha / self.config.lora_rank

        # delta = scaling * (x @ A^T @ B^T)
        intermediate = x @ lora_a.T
        delta = intermediate @ lora_b.T
        return scaling * delta

    def _lora_forward_he(
        self,
        x: np.ndarray,
        module_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute LoRA delta under homomorphic encryption.

        This is the MOAI-optimized path:
        1. Encrypt x
        2. Compute delta = scaling * (enc(x) @ A^T @ B^T) with column packing
        3. Decrypt and return

        Args:
            x: Input activation (plaintext)
            module_name: Target module

        Returns:
            Decrypted LoRA delta
        """
        if self._he_adapter is None:
            raise RuntimeError("HE adapter not initialized")

        return self._he_adapter.forward(x, module_name)

    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics."""
        return {
            "inference_count": self._inference_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / max(1, self._inference_count),
            "lora_mode": self.config.lora_mode.value,
            "enforce_tgsp": self.config.enforce_tgsp,
            "weights_from_tgsp": self._weights_from_tgsp,
            "tgsp_compliant": self._weights_from_tgsp or self.config.lora_mode == LoRAMode.PLAINTEXT,
            "he_adapter_metrics": (
                self._he_adapter.get_metrics() if self._he_adapter else None
            ),
        }

    def __call__(
        self,
        x: np.ndarray,
        module_name: Optional[str] = None,
    ) -> InferenceResult:
        """Convenience method for forward()."""
        return self.forward(x, module_name)


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """
    Add TenSafe inference arguments to parser.

    Args:
        parser: Argument parser to add to
    """
    group = parser.add_argument_group("TenSafe Inference")

    group.add_argument(
        "--lora_mode",
        type=str,
        choices=["none", "plaintext", "he_only", "full_he"],
        default="plaintext",
        help="LoRA inference mode: "
             "none=no LoRA, "
             "plaintext=standard LoRA, "
             "he_only=LoRA under HE with base model plaintext (recommended), "
             "full_he=everything encrypted (slow)",
    )

    group.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank",
    )

    group.add_argument(
        "--lora_alpha",
        type=float,
        default=32.0,
        help="LoRA alpha scaling",
    )

    group.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )

    group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    group.add_argument(
        "--enforce_tgsp",
        type=bool,
        default=True,
        help="Enforce TGSP format for encrypted inference modes (he_only, full_he). "
             "When enabled, only adapters loaded via TGSPAdapterRegistry are allowed. "
             "Default: True (RECOMMENDED for production)",
    )

    group.add_argument(
        "--tgsp_adapter",
        type=str,
        default=None,
        help="Path to TGSP-format adapter file (.tgsp) for encrypted inference",
    )


def create_inference_from_args(
    args: argparse.Namespace,
    base_model: Optional[Any] = None,
    lora_weights: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> TenSafeInference:
    """
    Create TenSafeInference from command line arguments.

    Args:
        args: Parsed command line arguments
        base_model: Optional base model
        lora_weights: Optional LoRA weights

    Returns:
        Configured TenSafeInference
    """
    config = InferenceConfig.from_args(args)
    return TenSafeInference(
        base_model=base_model,
        lora_weights=lora_weights,
        config=config,
    )


# CLI entry point
def main():
    """CLI for TenSafe inference."""
    parser = argparse.ArgumentParser(description="TenSafe Inference")
    add_inference_args(parser)
    parser.add_argument("--input", type=str, help="Input file or text")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Demo with mock data
    print(f"LoRA Mode: {args.lora_mode}")

    # Create mock LoRA weights
    hidden_dim = 64
    rank = args.lora_rank
    lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
    lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01

    lora_weights = {"q_proj": (lora_a, lora_b)}

    # Create inference
    inference = create_inference_from_args(args, lora_weights=lora_weights)

    # Run forward
    x = np.random.randn(hidden_dim).astype(np.float64)
    result = inference(x, "q_proj")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.output.shape}")
    print(f"Total time: {result.total_time_ms:.2f} ms")
    print(f"Base model time: {result.base_model_time_ms:.2f} ms")
    print(f"LoRA time: {result.lora_time_ms:.2f} ms")

    if result.he_metrics:
        print(f"HE Metrics: {json.dumps(result.he_metrics, indent=2)}")


if __name__ == "__main__":
    main()
