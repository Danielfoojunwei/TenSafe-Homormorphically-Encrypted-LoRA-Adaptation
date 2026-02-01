"""
TenSafe Unified Orchestrator.

This module provides a central orchestrator that connects all components:
- ML Backends (PyTorch)
- Privacy Accounting
- HE Backends
- Logging and Audit

The orchestrator provides a single entry point for:
- Client SDK operations
- Server API handlers
- CLI commands
- Direct Python usage

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              TenSafeOrchestrator                │
    │         (Central Control Point)                 │
    └───────────────────┬─────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌───────────┐    ┌──────────────┐    ┌─────────────┐
│ ML Backend │    │ HE Backend   │    │ Privacy     │
│ (Training) │    │ (Encryption) │    │ (Accounting)│
└───────────┘    └──────────────┘    └─────────────┘

Usage:
    from tensafe.core.orchestrator import TenSafeOrchestrator

    # Create orchestrator
    orchestrator = TenSafeOrchestrator(config)
    orchestrator.initialize()

    # Training
    result = orchestrator.forward_backward(batch)
    result = orchestrator.optim_step()

    # Inference
    samples = orchestrator.sample(prompts)

    # State management
    state = orchestrator.save_state()
    orchestrator.load_state(state)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Model configuration
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    model_revision: Optional[str] = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

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
    max_grad_norm: float = 1.0

    # DP configuration
    dp_enabled: bool = True
    dp_noise_multiplier: float = 1.0
    dp_target_epsilon: Optional[float] = 8.0
    dp_target_delta: float = 1e-5
    dp_accountant_type: str = "rdp"

    # HE configuration
    he_enabled: bool = False
    he_backend: str = "auto"
    he_poly_modulus_degree: int = 8192
    he_scale_bits: int = 40

    # Backend selection
    ml_backend: str = "torch"

    def to_ml_config(self):
        """Convert to MLBackendConfig."""
        from tensafe.backends.ml_backend import MLBackendConfig

        return MLBackendConfig(
            model_name_or_path=self.model_name_or_path,
            model_revision=self.model_revision,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            lora_enabled=self.lora_enabled,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_modules=self.lora_target_modules,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

    def to_dp_config(self):
        """Convert to DPConfig."""
        from tensafe.privacy.accountants import DPConfig

        return DPConfig(
            noise_multiplier=self.dp_noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            target_epsilon=self.dp_target_epsilon,
            target_delta=self.dp_target_delta,
            accountant_type=self.dp_accountant_type,
        )


class OrchestratorState(Enum):
    """States of the orchestrator."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    SAMPLING = "sampling"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class TrainingMetrics:
    """Metrics from training operations."""

    step: int
    loss: float
    grad_norm: float
    learning_rate: float
    tokens_processed: int
    time_ms: float

    # DP metrics
    epsilon_spent: Optional[float] = None
    total_epsilon: Optional[float] = None

    # HE metrics
    he_operations: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Unified Orchestrator
# ==============================================================================


class TenSafeOrchestrator:
    """
    Unified orchestrator for TenSafe operations.

    This class provides a single entry point for all training and inference
    operations, coordinating between ML backend, privacy accounting, and
    HE backends.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        orchestrator_id: Optional[str] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            orchestrator_id: Optional unique identifier
        """
        self.config = config or OrchestratorConfig()
        self.orchestrator_id = orchestrator_id or self._generate_id()

        self._state = OrchestratorState.CREATED
        self._lock = threading.RLock()

        # Components (initialized lazily)
        self._ml_backend = None
        self._privacy_accountant = None
        self._he_backend = None

        # Metrics
        self._step = 0
        self._total_tokens = 0
        self._start_time: Optional[float] = None

        # Callbacks
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        logger.info(f"Orchestrator created: {self.orchestrator_id}")

    def _generate_id(self) -> str:
        """Generate unique orchestrator ID."""
        import secrets

        return f"orch_{secrets.token_hex(8)}"

    # ===========================================================================
    # Lifecycle Methods
    # ===========================================================================

    def initialize(self) -> "TenSafeOrchestrator":
        """
        Initialize all backend components.

        This must be called before any training/inference operations.

        Returns:
            self for chaining
        """
        with self._lock:
            if self._state == OrchestratorState.READY:
                logger.info("Orchestrator already initialized")
                return self

            self._state = OrchestratorState.INITIALIZING
            self._emit_event("initializing", {})

            try:
                # Initialize ML backend
                self._init_ml_backend()

                # Initialize privacy accounting
                if self.config.dp_enabled:
                    self._init_privacy_accountant()

                # Initialize HE backend
                if self.config.he_enabled:
                    self._init_he_backend()

                self._state = OrchestratorState.READY
                self._start_time = time.time()
                self._emit_event("ready", {})

                logger.info(f"Orchestrator {self.orchestrator_id} ready")

            except Exception as e:
                self._state = OrchestratorState.ERROR
                self._emit_event("error", {"error": str(e)})
                raise

            return self

    def _init_ml_backend(self) -> None:
        """Initialize the ML backend."""
        from tensafe.backends.ml_backend import get_ml_backend

        ml_config = self.config.to_ml_config()

        self._ml_backend = get_ml_backend(
            backend_type=self.config.ml_backend,
            config=ml_config,
            initialize=True,
        )

        logger.info(f"ML backend initialized: {self._ml_backend.backend_name}")

    def _init_privacy_accountant(self) -> None:
        """Initialize privacy accounting."""
        from tensafe.privacy.accountants import get_privacy_accountant

        dp_config = self.config.to_dp_config()

        self._privacy_accountant = get_privacy_accountant(
            accountant_type=self.config.dp_accountant_type,
            config=dp_config,
        )

        logger.info(
            f"Privacy accountant initialized: {self._privacy_accountant.accountant_type}"
        )

    def _init_he_backend(self) -> None:
        """Initialize HE backend."""
        from tensafe.core.he_interface import get_backend, HEParams, HEBackendType

        params = HEParams(
            poly_modulus_degree=self.config.he_poly_modulus_degree,
            scale_bits=self.config.he_scale_bits,
        )

        backend_type = (
            HEBackendType.AUTO
            if self.config.he_backend == "auto"
            else HEBackendType(self.config.he_backend)
        )

        self._he_backend = get_backend(backend_type, params)

        logger.info(f"HE backend initialized: {self._he_backend.backend_name}")

    def shutdown(self) -> None:
        """Shutdown the orchestrator and release resources."""
        with self._lock:
            self._state = OrchestratorState.SHUTDOWN
            self._emit_event("shutdown", {})

            # Cleanup
            self._ml_backend = None
            self._privacy_accountant = None
            self._he_backend = None

            logger.info(f"Orchestrator {self.orchestrator_id} shutdown")

    # ===========================================================================
    # Training Methods
    # ===========================================================================

    def forward_backward(
        self,
        batch: Dict[str, Any],
        sample_rate: float = 0.01,
    ) -> TrainingMetrics:
        """
        Execute forward-backward pass.

        Args:
            batch: Training batch with input_ids, labels, attention_mask
            sample_rate: Batch sampling rate for DP

        Returns:
            TrainingMetrics with loss and gradient information
        """
        self._ensure_ready()

        with self._lock:
            self._state = OrchestratorState.TRAINING
            start_time = time.perf_counter()

            # Build DP config
            dp_config = None
            if self.config.dp_enabled:
                from tensafe.backends.ml_backend import DPConfig as MLDPConfig

                dp_config = MLDPConfig(
                    enabled=True,
                    noise_multiplier=self.config.dp_noise_multiplier,
                    max_grad_norm=self.config.max_grad_norm,
                    target_delta=self.config.dp_target_delta,
                )

            # Execute forward-backward
            result = self._ml_backend.forward_backward(batch, dp_config)

            # Update metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._total_tokens += result.tokens_processed

            # Build metrics
            metrics = TrainingMetrics(
                step=self._step,
                loss=result.loss,
                grad_norm=result.grad_norm,
                learning_rate=self._ml_backend._optimizer.param_groups[0]["lr"]
                if hasattr(self._ml_backend, "_optimizer")
                else self.config.learning_rate,
                tokens_processed=result.tokens_processed,
                time_ms=elapsed_ms,
            )

            # Add DP metrics if available
            if result.dp_metrics:
                metrics.extra.update(result.dp_metrics)

            self._emit_event(
                "forward_backward", {"step": self._step, "loss": result.loss}
            )
            self._state = OrchestratorState.READY

            return metrics

    def optim_step(
        self,
        apply_dp_noise: bool = True,
        sample_rate: float = 0.01,
    ) -> TrainingMetrics:
        """
        Execute optimizer step.

        Args:
            apply_dp_noise: Whether to apply DP noise
            sample_rate: Sampling rate for privacy accounting

        Returns:
            TrainingMetrics with step information
        """
        self._ensure_ready()

        with self._lock:
            self._state = OrchestratorState.TRAINING
            start_time = time.perf_counter()

            # Build DP config
            dp_config = None
            if self.config.dp_enabled:
                from tensafe.backends.ml_backend import DPConfig as MLDPConfig

                dp_config = MLDPConfig(
                    enabled=True,
                    noise_multiplier=self.config.dp_noise_multiplier,
                    max_grad_norm=self.config.max_grad_norm,
                    target_delta=self.config.dp_target_delta,
                )

            # Execute optimizer step
            result = self._ml_backend.optim_step(apply_dp_noise, dp_config)

            # Update privacy accounting
            epsilon_spent = None
            total_epsilon = None
            if self._privacy_accountant and apply_dp_noise:
                self._privacy_accountant.config.sample_rate = sample_rate
                epsilon_spent = self._privacy_accountant.step()
                spent = self._privacy_accountant.get_privacy_spent()
                total_epsilon = spent.epsilon

            self._step = result.step
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Build metrics
            metrics = TrainingMetrics(
                step=self._step,
                loss=0.0,  # Not computed in optim_step
                grad_norm=0.0,
                learning_rate=result.learning_rate,
                tokens_processed=0,
                time_ms=elapsed_ms,
                epsilon_spent=epsilon_spent,
                total_epsilon=total_epsilon,
            )

            if result.dp_metrics:
                metrics.extra.update(result.dp_metrics)

            self._emit_event("optim_step", {"step": self._step})
            self._state = OrchestratorState.READY

            return metrics

    def train_step(
        self,
        batch: Dict[str, Any],
        sample_rate: float = 0.01,
    ) -> TrainingMetrics:
        """
        Execute combined forward-backward and optimizer step.

        Convenience method that combines both operations.

        Args:
            batch: Training batch
            sample_rate: Batch sampling rate

        Returns:
            TrainingMetrics from the full step
        """
        fb_metrics = self.forward_backward(batch, sample_rate)
        opt_metrics = self.optim_step(sample_rate=sample_rate)

        # Combine metrics
        fb_metrics.learning_rate = opt_metrics.learning_rate
        fb_metrics.epsilon_spent = opt_metrics.epsilon_spent
        fb_metrics.total_epsilon = opt_metrics.total_epsilon
        fb_metrics.step = opt_metrics.step

        return fb_metrics

    # ===========================================================================
    # Sampling / Inference Methods
    # ===========================================================================

    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
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
            List of sample dicts with prompt, completion, etc.
        """
        self._ensure_ready()

        with self._lock:
            self._state = OrchestratorState.SAMPLING

            result = self._ml_backend.sample(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
            )

            self._emit_event("sample", {"num_prompts": len(prompts)})
            self._state = OrchestratorState.READY

            return result.samples

    # ===========================================================================
    # State Management
    # ===========================================================================

    def save_state(self, include_optimizer: bool = True) -> bytes:
        """
        Save model state to bytes.

        Args:
            include_optimizer: Whether to include optimizer state

        Returns:
            Serialized state bytes
        """
        self._ensure_ready()

        with self._lock:
            state = self._ml_backend.save_state(include_optimizer)

            self._emit_event("save_state", {"size_bytes": len(state)})

            return state

    def load_state(self, state_bytes: bytes) -> int:
        """
        Load model state from bytes.

        Args:
            state_bytes: Serialized state

        Returns:
            Loaded step number
        """
        self._ensure_ready()

        with self._lock:
            step = self._ml_backend.load_state(state_bytes)
            self._step = step

            self._emit_event("load_state", {"step": step})

            return step

    # ===========================================================================
    # HE Operations
    # ===========================================================================

    def get_lora_weights(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get LoRA weights for HE inference.

        Returns:
            Dict of module_name -> (lora_a, lora_b)
        """
        self._ensure_ready()
        return self._ml_backend.get_lora_weights()

    def encrypt_activation(self, x: np.ndarray) -> Any:
        """
        Encrypt an activation tensor.

        Args:
            x: Plaintext activation

        Returns:
            Ciphertext
        """
        if self._he_backend is None:
            raise RuntimeError("HE backend not initialized")

        return self._he_backend.encrypt(x)

    def decrypt_activation(self, ct: Any, output_size: int = 0) -> np.ndarray:
        """
        Decrypt a ciphertext.

        Args:
            ct: Ciphertext
            output_size: Number of elements to return

        Returns:
            Decrypted plaintext
        """
        if self._he_backend is None:
            raise RuntimeError("HE backend not initialized")

        return self._he_backend.decrypt(ct, output_size)

    def encrypted_lora_delta(
        self,
        ct_x: Any,
        module_name: Optional[str] = None,
    ) -> Any:
        """
        Compute encrypted LoRA delta.

        Args:
            ct_x: Encrypted activation
            module_name: Target module

        Returns:
            Encrypted delta
        """
        if self._he_backend is None:
            raise RuntimeError("HE backend not initialized")

        weights = self.get_lora_weights()

        if module_name is None:
            if not weights:
                raise ValueError("No LoRA weights available")
            module_name = next(iter(weights))

        if module_name not in weights:
            raise ValueError(f"Module {module_name} not found")

        lora_a, lora_b = weights[module_name]
        scaling = self.config.lora_alpha / self.config.lora_rank

        return self._he_backend.lora_delta(ct_x, lora_a, lora_b, scaling)

    # ===========================================================================
    # Privacy Methods
    # ===========================================================================

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta)
        """
        if self._privacy_accountant is None:
            return 0.0, 0.0

        spent = self._privacy_accountant.get_privacy_spent()
        return spent.epsilon, spent.delta

    def check_privacy_budget(self) -> bool:
        """
        Check if privacy budget is still available.

        Returns:
            True if budget OK, False if exceeded
        """
        if self._privacy_accountant is None:
            return True

        return self._privacy_accountant.check_budget()

    # ===========================================================================
    # Utility Methods
    # ===========================================================================

    def _ensure_ready(self) -> None:
        """Ensure orchestrator is ready for operations."""
        if self._state == OrchestratorState.CREATED:
            raise RuntimeError(
                "Orchestrator not initialized. Call initialize() first."
            )
        if self._state == OrchestratorState.ERROR:
            raise RuntimeError("Orchestrator in error state.")
        if self._state == OrchestratorState.SHUTDOWN:
            raise RuntimeError("Orchestrator has been shutdown.")

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit an event to callbacks."""
        payload["orchestrator_id"] = self.orchestrator_id
        payload["timestamp"] = datetime.utcnow().isoformat()

        for callback in self._callbacks:
            try:
                callback(event_type, payload)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def register_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """Register an event callback."""
        self._callbacks.append(callback)

    @property
    def state(self) -> OrchestratorState:
        """Get current state."""
        return self._state

    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self._step

    @property
    def is_ready(self) -> bool:
        """Check if orchestrator is ready."""
        return self._state == OrchestratorState.READY

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        eps, delta = self.get_privacy_spent()

        return {
            "orchestrator_id": self.orchestrator_id,
            "state": self._state.value,
            "step": self._step,
            "total_tokens": self._total_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self._total_tokens / max(1, elapsed),
            "privacy": {
                "epsilon": eps,
                "delta": delta,
                "budget_remaining": self.check_privacy_budget(),
            },
            "backends": {
                "ml": self._ml_backend.backend_name
                if self._ml_backend
                else None,
                "he": self._he_backend.backend_name
                if self._he_backend
                else None,
                "privacy": self._privacy_accountant.accountant_type
                if self._privacy_accountant
                else None,
            },
        }


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_orchestrator(
    config: Optional[OrchestratorConfig] = None,
    initialize: bool = True,
    **kwargs: Any,
) -> TenSafeOrchestrator:
    """
    Create a TenSafe orchestrator.

    Args:
        config: Configuration
        initialize: Whether to auto-initialize
        **kwargs: Additional config overrides

    Returns:
        Configured TenSafeOrchestrator
    """
    if config is None:
        config = OrchestratorConfig(**kwargs)
    else:
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    orchestrator = TenSafeOrchestrator(config)

    if initialize:
        orchestrator.initialize()

    return orchestrator
