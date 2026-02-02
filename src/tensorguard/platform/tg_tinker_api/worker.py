"""
TG-Tinker background worker.

Processes jobs from the queue and executes training operations.
Uses the unified TenSafeOrchestrator for production ML operations.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .queue import Job, JobQueue, get_job_queue

logger = logging.getLogger(__name__)


# ==============================================================================
# Production ML Backend (uses TenSafeOrchestrator)
# ==============================================================================


class ProductionMLBackend:
    """
    Production ML backend using TenSafeOrchestrator.

    This backend integrates with PyTorch/Transformers for real training
    operations. Requires proper GPU and model setup.
    """

    def __init__(self):
        """Initialize production backend."""
        self._orchestrators: Dict[str, Any] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def initialize_model(
        self,
        training_client_id: str,
        model_ref: str,
        config: Dict[str, Any],
    ) -> None:
        """Initialize a model for training."""
        self._configs[training_client_id] = {
            "model_ref": model_ref,
            "config": config,
        }

        try:
            from tensafe.core.orchestrator import (
                TenSafeOrchestrator,
                OrchestratorConfig,
            )

            # Build config from request
            orch_config = OrchestratorConfig(
                model_name_or_path=model_ref,
                lora_enabled=config.get("lora", {}).get("enabled", True),
                lora_rank=config.get("lora", {}).get("rank", 16),
                lora_alpha=config.get("lora", {}).get("alpha", 32.0),
                learning_rate=config.get("optimizer", {}).get("learning_rate", 1e-4),
                dp_enabled=config.get("dp", {}).get("enabled", True),
                dp_noise_multiplier=config.get("dp", {}).get("noise_multiplier", 1.0),
                dp_target_epsilon=config.get("dp", {}).get("target_epsilon", 8.0),
                dp_target_delta=config.get("dp", {}).get("target_delta", 1e-5),
            )

            orchestrator = TenSafeOrchestrator(
                config=orch_config,
                orchestrator_id=training_client_id,
            )
            orchestrator.initialize()

            self._orchestrators[training_client_id] = orchestrator
            logger.info(f"Initialized model for {training_client_id}: {model_ref}")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(
                f"Failed to initialize model '{model_ref}': {e}\n"
                "Ensure PyTorch and Transformers are properly installed "
                "and GPU resources are available."
            )

    def forward_backward(
        self,
        training_client_id: str,
        batch: Dict[str, Any],
        dp_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute forward-backward pass."""
        orch = self._get_orchestrator(training_client_id)

        sample_rate = dp_config.get("sample_rate", 0.01) if dp_config else 0.01
        metrics = orch.forward_backward(batch, sample_rate)

        result = {
            "loss": metrics.loss,
            "grad_norm": metrics.grad_norm,
            "tokens_processed": metrics.tokens_processed,
        }

        if dp_config and dp_config.get("enabled"):
            result["dp_metrics"] = metrics.extra

        return result

    def optim_step(
        self,
        training_client_id: str,
        apply_dp_noise: bool = True,
        dp_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute optimizer step."""
        orch = self._get_orchestrator(training_client_id)

        sample_rate = dp_config.get("sample_rate", 0.01) if dp_config else 0.01
        metrics = orch.optim_step(apply_dp_noise, sample_rate)

        result = {
            "step": metrics.step,
            "learning_rate": metrics.learning_rate,
        }

        if dp_config and dp_config.get("enabled"):
            result["dp_metrics"] = {
                "noise_applied": apply_dp_noise,
                "epsilon_spent": metrics.epsilon_spent,
                "total_epsilon": metrics.total_epsilon,
                "delta": dp_config.get("target_delta", 1e-5),
            }

        return result

    def sample(
        self,
        training_client_id: str,
        prompts: list,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate samples from the model."""
        orch = self._get_orchestrator(training_client_id)

        samples = orch.sample(
            prompts=prompts,
            max_tokens=config.get("max_tokens", 128),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 50),
        )

        return {
            "samples": samples,
            "model_step": orch.current_step,
            "sampling_config": config,
        }

    def save_state(
        self,
        training_client_id: str,
        include_optimizer: bool = True,
    ) -> bytes:
        """Serialize model state."""
        orch = self._get_orchestrator(training_client_id)
        return orch.save_state(include_optimizer)

    def load_state(
        self,
        training_client_id: str,
        state_bytes: bytes,
    ) -> int:
        """Load model state from bytes."""
        orch = self._get_orchestrator(training_client_id)
        return orch.load_state(state_bytes)

    def _get_orchestrator(self, training_client_id: str):
        """Get orchestrator for a training client."""
        orch = self._orchestrators.get(training_client_id)
        if orch is None:
            raise ValueError(f"Model not found: {training_client_id}")
        return orch


class Worker:
    """
    Background worker that processes jobs from the queue.

    Runs in a separate thread and executes training operations.
    Uses ProductionMLBackend for real training with PyTorch/Transformers.
    """

    def __init__(
        self,
        queue: Optional[JobQueue] = None,
        ml_backend: Optional[ProductionMLBackend] = None,
        poll_interval: float = 0.1,
    ):
        """
        Initialize worker.

        Args:
            queue: Job queue (defaults to global queue)
            ml_backend: ML backend for executing operations (uses ProductionMLBackend)
            poll_interval: Interval between queue polls
        """
        self.queue = queue or get_job_queue()
        self.ml_backend = ml_backend or ProductionMLBackend()
        self.poll_interval = poll_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable[[Job], Dict[str, Any]]] = {
            "forward_backward": self._handle_forward_backward,
            "optim_step": self._handle_optim_step,
            "sample": self._handle_sample,
            "save_state": self._handle_save_state,
            "load_state": self._handle_load_state,
        }

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Worker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout)
            self._thread = None
        logger.info("Worker stopped")

    def _run_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                job = self.queue.get_next(timeout=self.poll_interval)
                if job is None:
                    continue

                self._process_job(job)

            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")
                time.sleep(1)  # Backoff on error

    def _process_job(self, job: Job) -> None:
        """Process a single job."""
        logger.info(f"Processing job {job.job_id}: {job.operation}")

        handler = self._handlers.get(job.operation)
        if handler is None:
            self.queue.fail(job.job_id, f"Unknown operation: {job.operation}")
            return

        try:
            result = handler(job)
            self.queue.complete(job.job_id, result)
            logger.info(f"Job {job.job_id} completed")

        except Exception as e:
            logger.exception(f"Job {job.job_id} failed: {e}")
            self.queue.fail(job.job_id, str(e))

    def _handle_forward_backward(self, job: Job) -> Dict[str, Any]:
        """Handle forward_backward operation."""
        payload = job.payload
        return self.ml_backend.forward_backward(
            training_client_id=job.training_client_id,
            batch=payload.get("batch", {}),
            dp_config=payload.get("dp_config"),
        )

    def _handle_optim_step(self, job: Job) -> Dict[str, Any]:
        """Handle optim_step operation."""
        payload = job.payload
        return self.ml_backend.optim_step(
            training_client_id=job.training_client_id,
            apply_dp_noise=payload.get("apply_dp_noise", True),
            dp_config=payload.get("dp_config"),
        )

    def _handle_sample(self, job: Job) -> Dict[str, Any]:
        """Handle sample operation."""
        payload = job.payload
        return self.ml_backend.sample(
            training_client_id=job.training_client_id,
            prompts=payload.get("prompts", []),
            config={
                "max_tokens": payload.get("max_tokens", 128),
                "temperature": payload.get("temperature", 0.7),
                "top_p": payload.get("top_p", 0.9),
                "top_k": payload.get("top_k", 50),
                "stop_sequences": payload.get("stop_sequences", []),
            },
        )

    def _handle_save_state(self, job: Job) -> Dict[str, Any]:
        """Handle save_state operation."""
        payload = job.payload
        state_bytes = self.ml_backend.save_state(
            training_client_id=job.training_client_id,
            include_optimizer=payload.get("include_optimizer", True),
        )

        # Note: Actual storage/encryption is done in the route handler
        # This just returns the serialized state
        return {
            "state_bytes": state_bytes,
            "size_bytes": len(state_bytes),
        }

    def _handle_load_state(self, job: Job) -> Dict[str, Any]:
        """Handle load_state operation."""
        payload = job.payload
        state_bytes = payload.get("state_bytes", b"")

        step = self.ml_backend.load_state(
            training_client_id=job.training_client_id,
            state_bytes=state_bytes,
        )

        return {
            "step": step,
        }


# Global worker instance
_worker: Optional[Worker] = None


def get_worker() -> Worker:
    """Get the global worker instance."""
    global _worker
    if _worker is None:
        _worker = Worker()
    return _worker


def start_worker() -> None:
    """Start the global worker."""
    get_worker().start()


def stop_worker() -> None:
    """Stop the global worker."""
    global _worker
    if _worker:
        _worker.stop()
        _worker = None
