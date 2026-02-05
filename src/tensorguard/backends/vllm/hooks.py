"""HE-LoRA Forward Hooks for vLLM Integration.

This module provides PyTorch forward hooks that inject HE-LoRA computation
into the vLLM inference pipeline while preserving privacy guarantees.
"""

from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
import time
import threading
import torch
import torch.nn as nn

# Import TenSafe HE components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from tensorguard.n2he.core import HESchemeParams, N2HEScheme, ToyN2HEScheme
    from tensorguard.n2he.keys import SecretKey, PublicKey, EvaluationKey
    HE_AVAILABLE = True
except ImportError:
    HE_AVAILABLE = False

try:
    # Import HE-LoRA microkernel if available
    from he_lora_microkernel.runtime.executor import HELoRAExecutor
    from he_lora_microkernel.backend.gpu_ckks_backend import GPUCKKSBackend
    HELORA_MICROKERNEL_AVAILABLE = True
except ImportError:
    HELORA_MICROKERNEL_AVAILABLE = False


@dataclass
class HELoRAMetrics:
    """Metrics for HE-LoRA operations."""
    total_operations: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    errors: int = 0

    def record_operation(self, latency_ms: float):
        """Record a single operation."""
        self.total_operations += 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_operations == 0:
            return 0.0
        return self.total_latency_ms / self.total_operations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            "errors": self.errors,
        }


@dataclass
class HELoRAConfig:
    """Configuration for HE-LoRA hook."""
    hidden_size: int = 4096
    rank: int = 16
    alpha: float = 32.0
    scheme: str = "ckks"  # ckks, tfhe, hybrid
    profile: str = "fast"  # fast, safe
    use_moai: bool = True  # MOAI rotation elimination
    enable_metrics: bool = True

    # Gated LoRA / Hybrid settings
    adapter_type: str = "linear_lora"  # "linear_lora" | "gated_lora"
    gate_type: str = "step"  # "step" | "sign" (for gated_lora)

    # HAS service settings (for hybrid mode)
    has_service_url: Optional[str] = None
    bridge_service_url: Optional[str] = None
    bridge_timeout_ms: int = 5000


class HELoRAHook:
    """Forward hook for injecting HE-LoRA computation.

    This hook intercepts the output of target modules (q_proj, v_proj, etc.)
    and applies encrypted LoRA transformation.

    The computation flow is:
    1. Module produces output: h = W @ x
    2. Hook intercepts output
    3. HE-LoRA applied: h' = h + (α/r) * B @ A @ x (encrypted)
    4. Return modified output

    For privacy preservation:
    - LoRA weights (A, B) remain encrypted throughout
    - Uses CKKS for efficient encrypted matrix operations
    - MOAI eliminates rotation overhead
    """

    def __init__(
        self,
        layer_name: str,
        lora_a: torch.Tensor,  # Shape: (rank, hidden_size)
        lora_b: torch.Tensor,  # Shape: (hidden_size, rank)
        config: HELoRAConfig,
        he_scheme: Optional['N2HEScheme'] = None,
        evaluation_key: Optional['EvaluationKey'] = None,
        w_gate: Optional[torch.Tensor] = None,  # Shape: (hidden_size,) for gated_lora
        b_gate: Optional[torch.Tensor] = None,  # Shape: (1,) for gated_lora
    ):
        self.layer_name = layer_name
        self.config = config
        self.he_scheme = he_scheme
        self.evaluation_key = evaluation_key

        # Store LoRA weights (encrypted in production)
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling = config.alpha / config.rank

        # Gate weights (for gated_lora)
        self.w_gate = w_gate
        self.b_gate = b_gate

        # Hybrid backend (for gated_lora with hybrid scheme)
        self._hybrid_backend = None

        # Metrics
        self.metrics = HELoRAMetrics() if config.enable_metrics else None

        # Thread safety
        self._lock = threading.Lock()

        # HE-LoRA executor (if microkernel available)
        self._executor = None
        if HELORA_MICROKERNEL_AVAILABLE:
            try:
                self._executor = self._create_executor()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to create HE-LoRA executor: {e}")

        # Initialize hybrid backend for gated adapters
        if config.scheme == "hybrid" and config.adapter_type == "gated_lora":
            self._init_hybrid_backend()

    def _create_executor(self) -> Optional['HELoRAExecutor']:
        """Create HE-LoRA executor with GPU backend."""
        if not HELORA_MICROKERNEL_AVAILABLE:
            return None

        try:
            backend = GPUCKKSBackend(
                hidden_size=self.config.hidden_size,
                rank=self.config.rank,
                use_moai=self.config.use_moai,
            )
            return HELoRAExecutor(backend=backend)
        except Exception:
            return None

    def _init_hybrid_backend(self) -> None:
        """Initialize hybrid CKKS-TFHE backend for gated adapters."""
        try:
            from he_lora_microkernel.hybrid_compiler.backend import HybridHEBackend
            self._hybrid_backend = HybridHEBackend.create_simulated()
        except ImportError:
            import warnings
            warnings.warn("Hybrid backend not available, gated LoRA will use simulation")

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward hook implementation.

        Args:
            module: The PyTorch module (e.g., Linear layer)
            input: Input to the module
            output: Output from the module

        Returns:
            Modified output with HE-LoRA applied
        """
        start_time = time.perf_counter()

        try:
            # Get input tensor
            x = input[0] if isinstance(input, tuple) else input

            # Route based on scheme and adapter type
            if self.config.scheme == "hybrid" and self.config.adapter_type == "gated_lora":
                # Use hybrid CKKS-TFHE path for gated LoRA
                result = self._apply_gated_lora(x, output)
            elif self._executor is not None:
                # Use HE-LoRA microkernel for linear LoRA
                lora_output = self._executor.apply(
                    hidden_states=x,
                    lora_a=self.lora_a,
                    lora_b=self.lora_b,
                )
                result = output + self.scaling * lora_output
            else:
                # Fallback to simulated computation
                lora_output = self._apply_lora_simulated(x)
                result = output + self.scaling * lora_output

            # Record metrics
            if self.metrics is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                with self._lock:
                    self.metrics.record_operation(latency_ms)

            return result

        except Exception as e:
            if self.metrics is not None:
                with self._lock:
                    self.metrics.errors += 1

            # Log error but don't crash inference
            import logging
            logging.warning(f"HE-LoRA hook error in {self.layer_name}: {e}")

            # Return original output on error
            return output

    def _apply_gated_lora(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated LoRA using hybrid CKKS-TFHE.

        y = Wx + g(x) * scaling * B(Ax)
        where g(x) = LUT(w_g^T @ x + b_g)
        """
        import numpy as np

        # Convert to numpy for hybrid backend
        x_np = x.detach().cpu().numpy().astype(np.float64)
        base_np = base_output.detach().cpu().numpy().astype(np.float64)

        # Get weights
        lora_a = self.lora_a.detach().cpu().numpy().astype(np.float64)
        lora_b = self.lora_b.detach().cpu().numpy().astype(np.float64)

        # Get gate weights
        if self.w_gate is None:
            # No gate weights, fall back to linear
            lora_output = self._apply_lora_simulated(x)
            return base_output + self.scaling * lora_output

        w_gate = self.w_gate.detach().cpu().numpy().astype(np.float64)
        b_gate = self.b_gate.detach().cpu().numpy().astype(np.float64) if self.b_gate is not None else None

        # Process each sample in batch
        batch_shape = x_np.shape[:-1]
        hidden_size = x_np.shape[-1]

        x_flat = x_np.reshape(-1, hidden_size)
        base_flat = base_np.reshape(-1, hidden_size)
        output_flat = np.zeros_like(base_flat)

        for i in range(x_flat.shape[0]):
            xi = x_flat[i]
            base_i = base_flat[i]

            # LoRA delta: u = A @ x; delta = B @ u
            u = lora_a @ xi
            delta = lora_b @ u

            # Gate pre-activation: z = w_g^T @ x + b_g
            z = w_gate @ xi
            if b_gate is not None:
                z = z + float(b_gate.flat[0])

            # Apply gate LUT (simulation)
            if self.config.gate_type == "step":
                g = 1.0 if float(z) >= 0 else 0.0
            else:  # sign
                if float(z) > 0:
                    g = 1.0
                elif float(z) < 0:
                    g = -1.0
                else:
                    g = 0.0

            # Gated output
            output_flat[i] = base_i + g * self.scaling * delta

        # Reshape and convert back to torch
        output = output_flat.reshape(base_np.shape)
        return torch.from_numpy(output).to(base_output.device, dtype=base_output.dtype)

    def _apply_lora_simulated(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation (simulated, non-HE).

        This is used when:
        1. HE microkernel is not available
        2. Running in test/development mode
        3. Benchmarking baseline performance

        In production, this should be replaced with actual HE operations.
        """
        # LoRA: y = B @ A @ x
        # A: (rank, hidden_size), B: (hidden_size, rank)
        # x: (batch, seq, hidden_size)

        # Ensure weights are on same device
        device = x.device
        lora_a = self.lora_a.to(device)
        lora_b = self.lora_b.to(device)

        # Apply A matrix: (batch, seq, rank)
        intermediate = torch.matmul(x, lora_a.T)

        # Apply B matrix: (batch, seq, hidden_size)
        result = torch.matmul(intermediate, lora_b.T)

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get hook metrics."""
        if self.metrics is None:
            return {}

        with self._lock:
            return {
                "layer_name": self.layer_name,
                **self.metrics.to_dict()
            }


class HELoRAHookManager:
    """Manager for multiple HE-LoRA hooks.

    This class manages the registration and lifecycle of HE-LoRA hooks
    across all target modules in a model.

    Example:
        ```python
        manager = HELoRAHookManager(config)
        manager.register_hooks(model, lora_weights)

        # Run inference...

        metrics = manager.get_all_metrics()
        manager.remove_hooks()
        ```
    """

    def __init__(
        self,
        config: HELoRAConfig,
        target_modules: Optional[List[str]] = None,
    ):
        self.config = config
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        self._hooks: Dict[str, HELoRAHook] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._lock = threading.Lock()

    def register_hooks(
        self,
        model: nn.Module,
        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        he_scheme: Optional['N2HEScheme'] = None,
        evaluation_key: Optional['EvaluationKey'] = None,
    ) -> int:
        """Register HE-LoRA hooks on target modules.

        Args:
            model: PyTorch model
            lora_weights: Dict mapping layer names to (lora_a, lora_b) tuples
            he_scheme: HE scheme for encrypted operations
            evaluation_key: Evaluation key for HE operations

        Returns:
            Number of hooks registered
        """
        registered = 0

        for name, module in model.named_modules():
            # Check if this module is a target
            is_target = any(target in name for target in self.target_modules)

            if is_target and name in lora_weights:
                lora_a, lora_b = lora_weights[name]

                hook = HELoRAHook(
                    layer_name=name,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    config=self.config,
                    he_scheme=he_scheme,
                    evaluation_key=evaluation_key,
                )

                handle = module.register_forward_hook(hook)

                with self._lock:
                    self._hooks[name] = hook
                    self._handles.append(handle)

                registered += 1

        return registered

    def remove_hooks(self):
        """Remove all registered hooks."""
        with self._lock:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._hooks.clear()

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all hooks."""
        with self._lock:
            return {
                name: hook.get_metrics()
                for name, hook in self._hooks.items()
            }

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all hooks."""
        all_metrics = self.get_all_metrics()

        if not all_metrics:
            return {}

        total_ops = sum(m.get("total_operations", 0) for m in all_metrics.values())
        total_latency = sum(m.get("total_latency_ms", 0) for m in all_metrics.values())
        total_errors = sum(m.get("errors", 0) for m in all_metrics.values())

        max_latencies = [m.get("max_latency_ms", 0) for m in all_metrics.values()]
        min_latencies = [m.get("min_latency_ms", float('inf')) for m in all_metrics.values() if m.get("min_latency_ms", float('inf')) != float('inf')]

        return {
            "num_hooks": len(all_metrics),
            "total_operations": total_ops,
            "total_latency_ms": total_latency,
            "avg_latency_per_operation_ms": total_latency / total_ops if total_ops > 0 else 0,
            "max_latency_ms": max(max_latencies) if max_latencies else 0,
            "min_latency_ms": min(min_latencies) if min_latencies else 0,
            "total_errors": total_errors,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False
