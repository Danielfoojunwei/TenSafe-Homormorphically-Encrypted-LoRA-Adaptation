"""
TG-Tinker background worker.

Processes jobs from the queue and executes training operations.
"""

import logging
import secrets
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .queue import Job, JobQueue, JobStatus, get_job_queue

logger = logging.getLogger(__name__)


class MockMLBackend:
    """
    Mock ML backend for development and testing.

    In production, this would integrate with actual ML frameworks
    (PyTorch, JAX, etc.) for training operations.
    """

    def __init__(self):
        """Initialize mock backend."""
        self._models: Dict[str, Dict[str, Any]] = {}
        self._gradients: Dict[str, Any] = {}

    def initialize_model(
        self,
        training_client_id: str,
        model_ref: str,
        config: Dict[str, Any],
    ) -> None:
        """Initialize a model for training."""
        self._models[training_client_id] = {
            "model_ref": model_ref,
            "config": config,
            "step": 0,
            "weights": secrets.token_bytes(1024),  # Mock weights
            "optimizer_state": secrets.token_bytes(512),  # Mock optimizer state
        }
        logger.info(f"Initialized model for {training_client_id}: {model_ref}")

    def forward_backward(
        self,
        training_client_id: str,
        batch: Dict[str, Any],
        dp_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute forward-backward pass.

        Args:
            training_client_id: Training client ID
            batch: Training batch
            dp_config: Optional DP configuration

        Returns:
            Result dict with loss, grad_norm, etc.
        """
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Simulate computation time
        time.sleep(0.1)

        # Generate mock results
        batch_size = len(batch.get("input_ids", []))
        seq_len = len(batch.get("input_ids", [[]])[0]) if batch.get("input_ids") else 128

        loss = 2.5 - (model["step"] * 0.01)  # Decreasing loss
        grad_norm = 1.5 + secrets.randbelow(100) / 1000

        # Store gradients for optim_step
        self._gradients[training_client_id] = {
            "grad_norm": grad_norm,
            "computed_at": datetime.utcnow(),
        }

        result = {
            "loss": max(0.1, loss),
            "grad_norm": grad_norm,
            "tokens_processed": batch_size * seq_len,
        }

        # Apply DP if configured
        if dp_config and dp_config.get("enabled"):
            max_grad_norm = dp_config.get("max_grad_norm", 1.0)
            clipped_norm = min(grad_norm, max_grad_norm)
            num_clipped = 1 if grad_norm > max_grad_norm else 0

            result["dp_metrics"] = {
                "noise_applied": False,
                "epsilon_spent": 0.0,  # Computed in optim_step
                "total_epsilon": 0.0,
                "delta": dp_config.get("target_delta", 1e-5),
                "grad_norm_before_clip": grad_norm,
                "grad_norm_after_clip": clipped_norm,
                "num_clipped": num_clipped,
            }

            self._gradients[training_client_id]["clipped_norm"] = clipped_norm

        return result

    def optim_step(
        self,
        training_client_id: str,
        apply_dp_noise: bool = True,
        dp_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute optimizer step.

        Args:
            training_client_id: Training client ID
            apply_dp_noise: Whether to apply DP noise
            dp_config: Optional DP configuration

        Returns:
            Result dict with step, learning_rate, etc.
        """
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Simulate computation time
        time.sleep(0.05)

        # Increment step
        model["step"] += 1

        # Get learning rate from config
        optim_config = model["config"].get("optimizer", {})
        learning_rate = optim_config.get("learning_rate", 1e-4)

        result = {
            "step": model["step"],
            "learning_rate": learning_rate,
        }

        # Apply DP if configured
        if dp_config and dp_config.get("enabled") and apply_dp_noise:
            noise_multiplier = dp_config.get("noise_multiplier", 1.0)
            max_grad_norm = dp_config.get("max_grad_norm", 1.0)

            # Simple epsilon calculation (placeholder - real impl uses RDP)
            # This is a rough approximation: epsilon ≈ sqrt(2 * ln(1.25/delta)) / noise_multiplier
            delta = dp_config.get("target_delta", 1e-5)
            import math

            epsilon_spent = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier

            result["dp_metrics"] = {
                "noise_applied": True,
                "epsilon_spent": epsilon_spent,
                "total_epsilon": model.get("total_epsilon", 0) + epsilon_spent,
                "delta": delta,
                "grad_norm_before_clip": None,
                "grad_norm_after_clip": None,
                "num_clipped": None,
            }

            model["total_epsilon"] = result["dp_metrics"]["total_epsilon"]

        return result

    def sample(
        self,
        training_client_id: str,
        prompts: list,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate samples from the model.

        Args:
            training_client_id: Training client ID
            prompts: List of prompts
            config: Sampling configuration

        Returns:
            Result dict with samples
        """
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Simulate generation time
        time.sleep(0.1 * len(prompts))

        max_tokens = config.get("max_tokens", 128)

        samples = []
        for prompt in prompts:
            # Generate mock completion
            completion = f" [Mock completion for step {model['step']}]"
            tokens_generated = min(len(completion.split()), max_tokens)

            samples.append({
                "prompt": prompt,
                "completion": completion,
                "tokens_generated": tokens_generated,
                "finish_reason": "stop",
            })

        return {
            "samples": samples,
            "model_step": model["step"],
            "sampling_config": config,
        }

    def save_state(
        self,
        training_client_id: str,
        include_optimizer: bool = True,
    ) -> bytes:
        """
        Serialize model state.

        Args:
            training_client_id: Training client ID
            include_optimizer: Whether to include optimizer state

        Returns:
            Serialized state bytes
        """
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        import json

        state = {
            "model_ref": model["model_ref"],
            "step": model["step"],
            "weights_hash": secrets.token_hex(32),
            "config": model["config"],
        }

        if include_optimizer:
            state["optimizer_state_hash"] = secrets.token_hex(16)

        # In reality, this would be the actual serialized weights
        state_json = json.dumps(state)
        return state_json.encode() + model["weights"]

    def load_state(
        self,
        training_client_id: str,
        state_bytes: bytes,
    ) -> int:
        """
        Load model state from bytes.

        Args:
            training_client_id: Training client ID
            state_bytes: Serialized state bytes

        Returns:
            Loaded step number
        """
        model = self._models.get(training_client_id)
        if model is None:
            raise ValueError(f"Model not found: {training_client_id}")

        # Parse state header (JSON portion)
        import json

        try:
            # Find JSON boundary (simplistic approach)
            json_end = state_bytes.find(b"}")
            if json_end > 0:
                state_json = state_bytes[: json_end + 1].decode()
                state = json.loads(state_json)
                model["step"] = state.get("step", 0)
        except Exception:
            pass

        return model["step"]


class Worker:
    """
    Background worker that processes jobs from the queue.

    Runs in a separate thread and executes training operations.
    """

    def __init__(
        self,
        queue: Optional[JobQueue] = None,
        ml_backend: Optional[MockMLBackend] = None,
        poll_interval: float = 0.1,
    ):
        """
        Initialize worker.

        Args:
            queue: Job queue (defaults to global queue)
            ml_backend: ML backend for executing operations
            poll_interval: Interval between queue polls
        """
        self.queue = queue or get_job_queue()
        self.ml_backend = ml_backend or MockMLBackend()
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
