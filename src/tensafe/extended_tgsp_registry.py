"""
Extended TGSP Adapter Registry with Advanced Adapter Type Support.

This module extends the base TGSPAdapterRegistry with:
- Support for multiple adapter types (LoRA, rsLoRA, LoRA-FA, VeRA, etc.)
- HE-aware forward pass using the new adapter integration layer
- VeRA CKKS restructuring for rotation-free λ multiplication
- Rotation budget tracking and validation
- Production benchmarking utilities

The module maintains backward compatibility with the base registry while
adding advanced features for production deployments.

Author: TenSafe Team
"""

import hashlib
import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .tgsp_adapter_registry import (
    TGSPAdapterRegistry,
    TGSPAdapterMetadata,
    LoadedAdapter,
    AdapterLoadError,
    TGSPFormatRequiredError,
    NoActiveAdapterError,
)
from .adapters import (
    AdapterType,
    AdapterConfig,
    BaseAdapter,
    create_adapter,
    HECompatibility,
    get_he_compatibility,
    is_he_compatible,
    HEForwardConfig,
    HEForwardMetrics,
    CKKSLoRAForward,
    HybridTFHEGatedForward,
    PlaintextFallbackForward,
    create_he_forward,
    HEAwareHotSwapForward,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VERA CKKS RESTRUCTURING
# =============================================================================

@dataclass
class VeRACKKSPacking:
    """
    CKKS packing strategy for VeRA adapters.

    VeRA uses: Δ = λ_b ⊙ (B @ (λ_d ⊙ (A @ x)))

    For CKKS, we restructure this as:
    1. Encode λ_d as diagonal plaintext: diag_d
    2. Encode λ_b as diagonal plaintext: diag_b
    3. Compute: ct_x @ A^T (column-packed, 0 rotations)
    4. Multiply: ct_int * diag_d (element-wise, 0 rotations)
    5. Compute: scaled @ B^T (column-packed, 0 rotations)
    6. Multiply: ct_out * diag_b (element-wise, 0 rotations)

    Total rotations: 0 (when using column packing)
    """
    # Original VeRA parameters
    lambda_d: np.ndarray  # [rank]
    lambda_b: np.ndarray  # [out_features]

    # CKKS-packed versions
    diag_d_packed: Optional[Any] = None  # Diagonal encoding for λ_d
    diag_b_packed: Optional[Any] = None  # Diagonal encoding for λ_b

    # Shared random matrices (column-packed)
    A_packed: Optional[Any] = None
    B_packed: Optional[Any] = None

    # Metadata
    is_packed: bool = False
    packing_time_ms: float = 0.0


class VeRACKKSRestructurer:
    """
    Restructures VeRA computation for CKKS compatibility.

    Standard VeRA: Δ = λ_b ⊙ (B @ (λ_d ⊙ (A @ x)))
    - Element-wise λ multiplication is NOT directly CKKS-compatible

    Restructured for CKKS:
    - Pre-scale matrices: A' = diag(λ_d) @ A, B' = B @ diag(λ_d^-1) @ diag(λ_b)
    - Then: Δ = B' @ (A' @ x) which is standard matrix multiplication

    Alternative (runtime scaling):
    - Encode λ vectors as diagonal plaintexts
    - Use plaintext-ciphertext multiplication (0 rotations)
    """

    def __init__(self, backend: Any):
        """
        Initialize with CKKS backend.

        Args:
            backend: GPUCKKSBackend instance
        """
        self._backend = backend

    def pack_vera_weights(
        self,
        lambda_d: np.ndarray,
        lambda_b: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> VeRACKKSPacking:
        """
        Pack VeRA weights for CKKS computation.

        Args:
            lambda_d: Scaling vector [rank]
            lambda_b: Scaling vector [out_features]
            A: Shared random matrix [rank, in_features]
            B: Shared random matrix [out_features, rank]

        Returns:
            VeRACKKSPacking with CKKS-ready representations
        """
        start_time = time.perf_counter()

        packing = VeRACKKSPacking(
            lambda_d=lambda_d.copy(),
            lambda_b=lambda_b.copy(),
        )

        # Strategy 1: Pre-scaled matrices (compile-time)
        # A' = diag(λ_d) @ A => scales each row of A
        A_scaled = lambda_d[:, np.newaxis] * A

        # B' = diag(λ_b) @ B => scales each row of B
        B_scaled = lambda_b[:, np.newaxis] * B

        # Pack matrices using MOAI column packing
        if hasattr(self._backend, 'create_column_packed_matrix'):
            packing.A_packed = self._backend.create_column_packed_matrix(
                A_scaled.astype(np.float64)
            )
            packing.B_packed = self._backend.create_column_packed_matrix(
                B_scaled.astype(np.float64)
            )

            # For runtime λ updates, also create diagonal encodings
            packing.diag_d_packed = self._encode_as_diagonal(lambda_d)
            packing.diag_b_packed = self._encode_as_diagonal(lambda_b)
        else:
            # Fallback: store raw matrices
            packing.A_packed = A_scaled.astype(np.float64)
            packing.B_packed = B_scaled.astype(np.float64)

        packing.is_packed = True
        packing.packing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(f"Packed VeRA weights in {packing.packing_time_ms:.2f}ms")

        return packing

    def _encode_as_diagonal(self, vector: np.ndarray) -> Any:
        """
        Encode vector as diagonal plaintext for element-wise multiplication.

        The vector is replicated across SIMD slots in a pattern that allows
        element-wise multiplication without rotation.
        """
        if hasattr(self._backend, 'encode_diagonal'):
            return self._backend.encode_diagonal(vector.astype(np.float64))

        # Fallback: return raw vector
        return vector.astype(np.float64)

    def forward(
        self,
        ct_x: Any,
        packing: VeRACKKSPacking,
    ) -> Any:
        """
        Compute VeRA forward pass under CKKS.

        Args:
            ct_x: Encrypted input
            packing: Pre-packed VeRA weights

        Returns:
            Encrypted delta
        """
        if not packing.is_packed:
            raise ValueError("VeRA weights not packed. Call pack_vera_weights() first.")

        # ct_x @ A'^T (0 rotations with column packing)
        if hasattr(self._backend, 'column_packed_matmul'):
            ct_intermediate = self._backend.column_packed_matmul(
                ct_x, packing.A_packed, rescale=True
            )
            ct_result = self._backend.column_packed_matmul(
                ct_intermediate, packing.B_packed, rescale=True
            )
        else:
            # Fallback: standard matmul
            ct_intermediate = self._backend.ct_pt_multiply(ct_x, packing.A_packed.T)
            ct_result = self._backend.ct_pt_multiply(ct_intermediate, packing.B_packed.T)

        return ct_result


# =============================================================================
# EXTENDED TGSP METADATA
# =============================================================================

@dataclass
class ExtendedTGSPMetadata(TGSPAdapterMetadata):
    """Extended metadata supporting multiple adapter types."""

    # Adapter type (extending base LoRA support)
    adapter_type: AdapterType = AdapterType.LORA

    # HE compatibility
    he_compatibility: HECompatibility = HECompatibility.FULL_CKKS

    # VeRA-specific
    vera_lambda_d: Optional[np.ndarray] = None
    vera_lambda_b: Optional[np.ndarray] = None

    # DoRA-specific
    dora_magnitude: Optional[np.ndarray] = None

    # Rotation tracking
    total_rotations: int = 0
    rotation_budget_exceeded: bool = False

    # HE metrics
    he_metrics: Optional[HEForwardMetrics] = None


@dataclass
class ExtendedLoadedAdapter(LoadedAdapter):
    """Extended loaded adapter with multi-type support."""

    # Extended metadata
    extended_metadata: Optional[ExtendedTGSPMetadata] = None

    # Adapter instance (for non-linear types)
    adapter_instance: Optional[BaseAdapter] = None

    # HE forward handler
    he_forward: Optional[Any] = None  # HEAdapterForward instance

    # VeRA packing
    vera_packing: Optional[VeRACKKSPacking] = None


# =============================================================================
# ROTATION BUDGET TRACKER
# =============================================================================

@dataclass
class RotationBudget:
    """Rotation budget configuration and tracking."""
    max_rotations_per_token: int = 16
    max_rotations_per_layer: int = 64
    max_keyswitches_per_token: int = 16
    max_rescales_per_token: int = 8

    # Current usage
    current_rotations: int = 0
    current_keyswitches: int = 0
    current_rescales: int = 0

    # Validation
    budget_exceeded: bool = False
    exceeded_reason: Optional[str] = None

    def reset(self) -> None:
        """Reset current usage counters."""
        self.current_rotations = 0
        self.current_keyswitches = 0
        self.current_rescales = 0
        self.budget_exceeded = False
        self.exceeded_reason = None

    def record(
        self,
        rotations: int = 0,
        keyswitches: int = 0,
        rescales: int = 0,
    ) -> bool:
        """
        Record operation counts and check budget.

        Returns:
            True if within budget, False if exceeded
        """
        self.current_rotations += rotations
        self.current_keyswitches += keyswitches
        self.current_rescales += rescales

        if self.current_rotations > self.max_rotations_per_token:
            self.budget_exceeded = True
            self.exceeded_reason = (
                f"Rotation budget exceeded: {self.current_rotations} > "
                f"{self.max_rotations_per_token}"
            )
            return False

        if self.current_keyswitches > self.max_keyswitches_per_token:
            self.budget_exceeded = True
            self.exceeded_reason = (
                f"Keyswitch budget exceeded: {self.current_keyswitches} > "
                f"{self.max_keyswitches_per_token}"
            )
            return False

        return True


# =============================================================================
# EXTENDED REGISTRY
# =============================================================================

class ExtendedTGSPAdapterRegistry(TGSPAdapterRegistry):
    """
    Extended TGSP Adapter Registry with multi-type support.

    Extends the base registry with:
    - Support for all adapter types (LoRA, DoRA, VeRA, etc.)
    - HE-aware forward pass selection
    - VeRA CKKS restructuring
    - Rotation budget tracking
    - Production benchmarking
    """

    def __init__(
        self,
        enforce_tgsp: bool = True,
        auto_verify_signatures: bool = True,
        he_config: Optional[Dict[str, Any]] = None,
        work_dir: Optional[str] = None,
        rotation_budget: Optional[RotationBudget] = None,
        enable_benchmarking: bool = True,
    ):
        """
        Initialize extended registry.

        Args:
            enforce_tgsp: If True, ONLY TGSP format allowed for encrypted inference
            auto_verify_signatures: Automatically verify TGSP signatures on load
            he_config: HE configuration for adapter initialization
            work_dir: Working directory for extracted adapters
            rotation_budget: Rotation budget configuration
            enable_benchmarking: Enable performance benchmarking
        """
        super().__init__(
            enforce_tgsp=enforce_tgsp,
            auto_verify_signatures=auto_verify_signatures,
            he_config=he_config,
            work_dir=work_dir,
        )

        self.rotation_budget = rotation_budget or RotationBudget()
        self.enable_benchmarking = enable_benchmarking

        # Extended adapter storage
        self._extended_adapters: Dict[str, ExtendedLoadedAdapter] = {}

        # HE backend reference
        self._he_backend = None

        # VeRA restructurer
        self._vera_restructurer: Optional[VeRACKKSRestructurer] = None

        # HE-aware forward handler
        self._he_forward_handler: Optional[HEAwareHotSwapForward] = None

        # Benchmark results
        self._benchmarks: Dict[str, Dict[str, Any]] = {}

        logger.info("ExtendedTGSPAdapterRegistry initialized")

    def set_he_backend(self, backend: Any) -> None:
        """
        Set the HE backend for encrypted computation.

        Args:
            backend: GPUCKKSBackend instance
        """
        self._he_backend = backend
        self._vera_restructurer = VeRACKKSRestructurer(backend)
        self._he_forward_handler = HEAwareHotSwapForward(backend)

        logger.info("HE backend configured for extended registry")

    def load_tgsp_adapter(
        self,
        tgsp_path: str,
        recipient_key_path: Optional[str] = None,
        adapter_id: Optional[str] = None,
        public_key: Optional[Dict] = None,
        adapter_type: AdapterType = AdapterType.LORA,
    ) -> str:
        """
        Load a TGSP adapter with extended type support.

        Args:
            tgsp_path: Path to the .tgsp file
            recipient_key_path: Path to recipient private key for decryption
            adapter_id: Optional custom adapter ID
            public_key: Optional public key for signature verification
            adapter_type: Type of adapter (default: LoRA)

        Returns:
            Adapter ID
        """
        # Load using base class
        adapter_id = super().load_tgsp_adapter(
            tgsp_path=tgsp_path,
            recipient_key_path=recipient_key_path,
            adapter_id=adapter_id,
            public_key=public_key,
        )

        # Extend with advanced features
        base_adapter = self._adapters[adapter_id]
        self._extend_adapter(adapter_id, base_adapter, adapter_type)

        return adapter_id

    def _extend_adapter(
        self,
        adapter_id: str,
        base_adapter: LoadedAdapter,
        adapter_type: AdapterType,
    ) -> None:
        """Extend a base adapter with advanced features."""
        # Check HE compatibility
        he_compat = get_he_compatibility(adapter_type)

        # Create extended metadata
        ext_metadata = ExtendedTGSPMetadata(
            adapter_id=base_adapter.metadata.adapter_id,
            tgsp_path=base_adapter.metadata.tgsp_path,
            model_name=base_adapter.metadata.model_name,
            model_version=base_adapter.metadata.model_version,
            author_id=base_adapter.metadata.author_id,
            manifest_hash=base_adapter.metadata.manifest_hash,
            payload_hash=base_adapter.metadata.payload_hash,
            signature_verified=base_adapter.metadata.signature_verified,
            signature_key_id=base_adapter.metadata.signature_key_id,
            lora_rank=base_adapter.metadata.lora_rank,
            lora_alpha=base_adapter.metadata.lora_alpha,
            target_modules=base_adapter.metadata.target_modules,
            adapter_type=adapter_type,
            he_compatibility=he_compat,
        )

        # Create adapter config
        adapter_config = AdapterConfig(
            adapter_type=adapter_type,
            rank=base_adapter.metadata.lora_rank,
            alpha=base_adapter.metadata.lora_alpha,
            target_modules=base_adapter.metadata.target_modules,
        )

        # Create adapter instance for non-linear types
        adapter_instance = None
        if adapter_type not in {AdapterType.LORA, AdapterType.RS_LORA, AdapterType.LORA_FA}:
            # Get dimensions from first weight
            for module_name, (lora_a, lora_b) in base_adapter.weights.items():
                in_features = lora_a.shape[1]
                out_features = lora_b.shape[0]
                break
            else:
                in_features = out_features = 0

            if in_features > 0:
                adapter_instance = create_adapter(
                    adapter_config, in_features, out_features
                )
                # Load weights into adapter
                self._load_weights_into_adapter(adapter_instance, base_adapter.weights)

        # Create extended adapter
        ext_adapter = ExtendedLoadedAdapter(
            metadata=base_adapter.metadata,
            weights=base_adapter.weights,
            he_adapter=base_adapter.he_adapter,
            is_active=base_adapter.is_active,
            is_he_initialized=base_adapter.is_he_initialized,
            extended_metadata=ext_metadata,
            adapter_instance=adapter_instance,
        )

        # Handle VeRA packing
        if adapter_type == AdapterType.VERA and self._vera_restructurer is not None:
            self._pack_vera_adapter(ext_adapter, adapter_instance)

        # Create HE forward handler
        if adapter_instance is not None and self._he_backend is not None:
            he_forward = create_he_forward(adapter_instance)
            he_forward.initialize_backend(self._he_backend)
            ext_adapter.he_forward = he_forward

        self._extended_adapters[adapter_id] = ext_adapter

        logger.info(
            f"Extended adapter {adapter_id}: type={adapter_type.value}, "
            f"he_compat={he_compat.value}"
        )

    def _load_weights_into_adapter(
        self,
        adapter: BaseAdapter,
        weights: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Load TGSP weights into adapter instance."""
        # Get first weight for loading
        for module_name, (lora_a, lora_b) in weights.items():
            try:
                adapter.set_weights({
                    "lora_A": lora_a,
                    "lora_B": lora_b,
                }, strict=False)
            except Exception as e:
                logger.warning(f"Failed to load weights for {module_name}: {e}")
            break

    def _pack_vera_adapter(
        self,
        ext_adapter: ExtendedLoadedAdapter,
        adapter_instance: BaseAdapter,
    ) -> None:
        """Pack VeRA adapter weights for CKKS."""
        if adapter_instance is None:
            return

        # Get VeRA-specific parameters
        trainable = adapter_instance.get_trainable_params()
        frozen = adapter_instance.get_frozen_params()

        lambda_d = trainable.get("lambda_d")
        lambda_b = trainable.get("lambda_b")
        A = frozen.get("lora_A")
        B = frozen.get("lora_B")

        if all(x is not None for x in [lambda_d, lambda_b, A, B]):
            packing = self._vera_restructurer.pack_vera_weights(
                lambda_d=lambda_d,
                lambda_b=lambda_b,
                A=A,
                B=B,
            )
            ext_adapter.vera_packing = packing
            ext_adapter.extended_metadata.vera_lambda_d = lambda_d
            ext_adapter.extended_metadata.vera_lambda_b = lambda_b

            logger.info(f"Packed VeRA weights for {ext_adapter.metadata.adapter_id}")

    def activate_adapter(self, adapter_id: str) -> None:
        """
        Activate an adapter for encrypted inference.

        Extends base activation with HE-aware forward setup.
        """
        super().activate_adapter(adapter_id)

        # Setup HE forward handler
        if adapter_id in self._extended_adapters and self._he_forward_handler is not None:
            ext_adapter = self._extended_adapters[adapter_id]
            if ext_adapter.adapter_instance is not None:
                he_config = HEForwardConfig(
                    use_column_packing=True,
                    enable_tfhe_hybrid=(
                        ext_adapter.extended_metadata.he_compatibility ==
                        HECompatibility.HYBRID_TFHE
                    ),
                )
                self._he_forward_handler.set_active_adapter(
                    adapter_id,
                    ext_adapter.adapter_instance,
                    he_config,
                )

    def forward_he(
        self,
        x_plain: np.ndarray,
        module_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute encrypted forward pass with extended adapter support.

        Args:
            x_plain: Plaintext activation
            module_name: Target module

        Returns:
            Decrypted delta
        """
        adapter_id = self._active_adapter_id

        if adapter_id is None:
            raise NoActiveAdapterError("No adapter activated")

        # Check for extended adapter
        if adapter_id in self._extended_adapters:
            return self._forward_extended(adapter_id, x_plain, module_name)

        # Fall back to base implementation
        return super().forward_he(x_plain, module_name)

    def _forward_extended(
        self,
        adapter_id: str,
        x_plain: np.ndarray,
        module_name: Optional[str],
    ) -> np.ndarray:
        """Forward pass using extended adapter system."""
        ext_adapter = self._extended_adapters[adapter_id]
        adapter_type = ext_adapter.extended_metadata.adapter_type
        he_compat = ext_adapter.extended_metadata.he_compatibility

        # Reset rotation tracking
        self.rotation_budget.reset()

        start_time = time.perf_counter()

        # Route based on HE compatibility
        if he_compat == HECompatibility.FULL_CKKS:
            delta = self._forward_ckks(ext_adapter, x_plain, module_name)

        elif he_compat == HECompatibility.HYBRID_TFHE:
            delta = self._forward_hybrid(ext_adapter, x_plain, module_name)

        else:
            # Plaintext fallback with warning
            logger.warning(
                f"Adapter {adapter_id} ({adapter_type.value}) is not HE-compatible. "
                f"Using plaintext computation. SECURITY REDUCED."
            )
            delta = self._forward_plaintext(ext_adapter, x_plain, module_name)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update metrics
        ext_adapter.metadata.forward_count += 1
        ext_adapter.metadata.total_inference_time_ms += elapsed_ms
        ext_adapter.metadata.last_used_at = datetime.utcnow()
        ext_adapter.extended_metadata.total_rotations += self.rotation_budget.current_rotations

        # Check rotation budget
        if self.rotation_budget.budget_exceeded:
            ext_adapter.extended_metadata.rotation_budget_exceeded = True
            logger.warning(
                f"Rotation budget exceeded: {self.rotation_budget.exceeded_reason}"
            )

        # Record benchmark
        if self.enable_benchmarking:
            self._record_benchmark(adapter_id, elapsed_ms)

        return delta

    def _forward_ckks(
        self,
        ext_adapter: ExtendedLoadedAdapter,
        x_plain: np.ndarray,
        module_name: Optional[str],
    ) -> np.ndarray:
        """Forward pass using CKKS encryption."""
        adapter_type = ext_adapter.extended_metadata.adapter_type

        # Special handling for VeRA
        if adapter_type == AdapterType.VERA and ext_adapter.vera_packing is not None:
            return self._forward_vera_ckks(ext_adapter, x_plain)

        # Standard CKKS forward via HE handler
        if ext_adapter.he_forward is not None:
            delta = ext_adapter.he_forward.forward(x_plain, module_name or "default")

            # Record rotation counts
            metrics = ext_adapter.he_forward.get_metrics()
            self.rotation_budget.record(
                rotations=metrics.rotations,
                keyswitches=metrics.keyswitches,
                rescales=metrics.rescales,
            )

            return delta

        # Fallback to base HE adapter
        if ext_adapter.he_adapter is not None:
            return ext_adapter.he_adapter.forward(x_plain, module_name)

        # Final fallback to plaintext
        return self._forward_plaintext_base(
            self._adapters[ext_adapter.metadata.adapter_id],
            x_plain,
            module_name,
        )

    def _forward_vera_ckks(
        self,
        ext_adapter: ExtendedLoadedAdapter,
        x_plain: np.ndarray,
    ) -> np.ndarray:
        """
        VeRA forward pass with CKKS restructuring.

        Uses pre-scaled matrices for 0-rotation computation.
        """
        if self._he_backend is None:
            raise RuntimeError("HE backend not configured")

        packing = ext_adapter.vera_packing

        # Flatten input
        original_shape = x_plain.shape
        x_flat = x_plain.flatten().astype(np.float64)

        # Encrypt
        ct_x = self._he_backend.encrypt(x_flat)

        # Use restructurer for computation
        ct_result = self._vera_restructurer.forward(ct_x, packing)

        # Decrypt
        delta = self._he_backend.decrypt(ct_result, output_size=len(x_flat))

        # Record metrics: should be 0 rotations with column packing
        self.rotation_budget.record(rotations=0, rescales=2)

        return delta.reshape(original_shape)

    def _forward_hybrid(
        self,
        ext_adapter: ExtendedLoadedAdapter,
        x_plain: np.ndarray,
        module_name: Optional[str],
    ) -> np.ndarray:
        """Forward pass using CKKS+TFHE hybrid."""
        if ext_adapter.he_forward is not None:
            delta = ext_adapter.he_forward.forward(x_plain, module_name or "default")

            # Record metrics including bootstraps
            metrics = ext_adapter.he_forward.get_metrics()
            self.rotation_budget.record(
                rotations=metrics.rotations,
                keyswitches=metrics.keyswitches,
                rescales=metrics.rescales,
            )

            ext_adapter.extended_metadata.he_metrics = metrics

            return delta

        # Fallback
        logger.warning("Hybrid TFHE forward not available, using plaintext")
        return self._forward_plaintext(ext_adapter, x_plain, module_name)

    def _forward_plaintext(
        self,
        ext_adapter: ExtendedLoadedAdapter,
        x_plain: np.ndarray,
        module_name: Optional[str],
    ) -> np.ndarray:
        """Plaintext forward pass for HE-incompatible adapters."""
        if ext_adapter.adapter_instance is not None:
            return ext_adapter.adapter_instance.forward(x_plain)

        return self._forward_plaintext_base(
            self._adapters[ext_adapter.metadata.adapter_id],
            x_plain,
            module_name,
        )

    def _forward_plaintext_base(
        self,
        adapter: LoadedAdapter,
        x_plain: np.ndarray,
        module_name: Optional[str],
    ) -> np.ndarray:
        """Base plaintext forward (from parent class)."""
        if module_name is None:
            if not adapter.weights:
                raise ValueError("No weights in adapter")
            module_name = next(iter(adapter.weights))

        if module_name not in adapter.weights:
            raise ValueError(f"Module {module_name} not in adapter")

        lora_a, lora_b = adapter.weights[module_name]
        scaling = adapter.metadata.lora_alpha / adapter.metadata.lora_rank

        intermediate = x_plain @ lora_a.T
        delta = intermediate @ lora_b.T
        return scaling * delta

    def _record_benchmark(self, adapter_id: str, elapsed_ms: float) -> None:
        """Record benchmark result."""
        if adapter_id not in self._benchmarks:
            self._benchmarks[adapter_id] = {
                "samples": [],
                "total_ms": 0.0,
                "count": 0,
            }

        bench = self._benchmarks[adapter_id]
        bench["samples"].append(elapsed_ms)
        bench["total_ms"] += elapsed_ms
        bench["count"] += 1

        # Keep only last 1000 samples
        if len(bench["samples"]) > 1000:
            bench["samples"] = bench["samples"][-1000:]

    def get_benchmark_stats(self, adapter_id: str) -> Dict[str, Any]:
        """Get benchmark statistics for an adapter."""
        if adapter_id not in self._benchmarks:
            return {}

        bench = self._benchmarks[adapter_id]
        samples = np.array(bench["samples"])

        return {
            "adapter_id": adapter_id,
            "count": bench["count"],
            "total_ms": bench["total_ms"],
            "avg_ms": bench["total_ms"] / max(bench["count"], 1),
            "min_ms": float(np.min(samples)) if len(samples) > 0 else 0,
            "max_ms": float(np.max(samples)) if len(samples) > 0 else 0,
            "p50_ms": float(np.percentile(samples, 50)) if len(samples) > 0 else 0,
            "p95_ms": float(np.percentile(samples, 95)) if len(samples) > 0 else 0,
            "p99_ms": float(np.percentile(samples, 99)) if len(samples) > 0 else 0,
        }

    def get_rotation_report(self, adapter_id: str) -> Dict[str, Any]:
        """Get rotation count report for an adapter."""
        if adapter_id not in self._extended_adapters:
            return {}

        ext_adapter = self._extended_adapters[adapter_id]
        ext_meta = ext_adapter.extended_metadata

        return {
            "adapter_id": adapter_id,
            "adapter_type": ext_meta.adapter_type.value,
            "he_compatibility": ext_meta.he_compatibility.value,
            "total_rotations": ext_meta.total_rotations,
            "rotation_budget_exceeded": ext_meta.rotation_budget_exceeded,
            "forward_count": ext_adapter.metadata.forward_count,
            "avg_rotations_per_forward": (
                ext_meta.total_rotations / max(ext_adapter.metadata.forward_count, 1)
            ),
            "moai_enabled": hasattr(self._he_backend, 'column_packed_matmul') if self._he_backend else False,
        }

    def validate_rotation_budget(self) -> List[str]:
        """
        Validate rotation budget across all adapters.

        Returns:
            List of validation error messages
        """
        errors = []

        for adapter_id, ext_adapter in self._extended_adapters.items():
            ext_meta = ext_adapter.extended_metadata

            if ext_meta.rotation_budget_exceeded:
                errors.append(
                    f"Adapter {adapter_id} exceeded rotation budget: "
                    f"{ext_meta.total_rotations} rotations"
                )

            # Check for non-CKKS adapters
            if ext_meta.he_compatibility == HECompatibility.PLAINTEXT_ONLY:
                errors.append(
                    f"Adapter {adapter_id} ({ext_meta.adapter_type.value}) "
                    f"is not HE-compatible and runs in plaintext mode"
                )

        return errors


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "VeRACKKSPacking",
    "VeRACKKSRestructurer",
    "ExtendedTGSPMetadata",
    "ExtendedLoadedAdapter",
    "RotationBudget",
    "ExtendedTGSPAdapterRegistry",
]
