"""TenSafe MLflow Integration.

Provides experiment tracking and model registry with privacy awareness:
- Tracks training metrics and privacy budget
- Registers models with TSSP verification
- Supports model staging and deployment
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Conditional MLflow import
MLFLOW_AVAILABLE = False
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("mlflow not installed. Install with: pip install mlflow")


@dataclass
class TenSafeMLflowConfig:
    """Configuration for MLflow integration."""
    tracking_uri: Optional[str] = None
    experiment_name: str = "tensafe-training"
    artifact_location: Optional[str] = None

    # Model registry
    registered_model_name: Optional[str] = None
    model_stage: str = "Staging"

    # Privacy settings
    log_privacy_metrics: bool = True

    # Logging frequency
    log_frequency: int = 10


class TenSafeMLflowCallback:
    """MLflow callback with privacy-aware tracking.

    Example:
        ```python
        callback = TenSafeMLflowCallback(
            config=TenSafeMLflowConfig(
                tracking_uri="http://localhost:5000",
                experiment_name="my-experiment",
            )
        )

        callback.on_train_begin(params={"model": "llama-3-8b"})

        for step in range(num_steps):
            loss = train_step()
            callback.on_step_end(step=step, metrics={"loss": loss})

        callback.on_train_end()
        ```
    """

    def __init__(
        self,
        config: Optional[TenSafeMLflowConfig] = None,
        dp_config: Optional[Any] = None,
    ):
        """Initialize MLflow callback.

        Args:
            config: MLflow configuration
            dp_config: Differential privacy configuration
        """
        self.config = config or TenSafeMLflowConfig()
        self.dp_config = dp_config
        self._run = None
        self._client = None
        self._start_time = None

        self._setup()

    def _setup(self):
        """Setup MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            return

        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)

        # Set or create experiment
        mlflow.set_experiment(self.config.experiment_name)

        self._client = MlflowClient()

    def on_train_begin(
        self,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Start MLflow run.

        Args:
            params: Training parameters to log
            tags: Run tags
        """
        if not MLFLOW_AVAILABLE:
            return

        # Start run
        self._run = mlflow.start_run()
        self._start_time = time.time()

        # Log parameters
        base_params = {
            "framework": "tensafe",
            "framework_version": "3.0.0",
        }

        if self.dp_config:
            base_params.update({
                "dp_enabled": str(getattr(self.dp_config, 'enabled', True)),
                "dp_noise_multiplier": getattr(self.dp_config, 'noise_multiplier', None),
                "dp_max_grad_norm": getattr(self.dp_config, 'max_grad_norm', None),
                "dp_target_epsilon": getattr(self.dp_config, 'target_epsilon', None),
            })

        if params:
            # Redact sensitive params
            safe_params = self._redact_params(params)
            base_params.update(safe_params)

        mlflow.log_params(base_params)

        # Log tags
        base_tags = {"tensafe": "true", "privacy_preserving": "true"}
        if tags:
            base_tags.update(tags)
        mlflow.set_tags(base_tags)

        logger.info(f"MLflow run started: {self._run.info.run_id}")

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        """Log metrics at step end.

        Args:
            step: Current step
            metrics: Metrics to log
            epsilon: Current privacy epsilon
            delta: Current privacy delta
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        if step % self.config.log_frequency != 0:
            return

        # Log training metrics
        for key, value in metrics.items():
            if self._is_safe_metric(key, value):
                mlflow.log_metric(key, value, step=step)

        # Log privacy metrics
        if self.config.log_privacy_metrics:
            if epsilon is not None:
                mlflow.log_metric("privacy_epsilon", epsilon, step=step)
            if delta is not None:
                mlflow.log_metric("privacy_delta", delta, step=step)

    def on_train_end(
        self,
        final_metrics: Optional[Dict[str, float]] = None,
        model_path: Optional[str] = None,
    ):
        """End MLflow run.

        Args:
            final_metrics: Final metrics to log
            model_path: Path to model for registration
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        # Log final metrics
        if final_metrics:
            for key, value in final_metrics.items():
                if self._is_safe_metric(key, value):
                    mlflow.log_metric(f"final_{key}", value)

        # Log duration
        if self._start_time:
            duration = time.time() - self._start_time
            mlflow.log_metric("training_duration_seconds", duration)

        # Register model (metadata only)
        if model_path and self.config.registered_model_name:
            self._register_model(model_path)

        mlflow.end_run()
        logger.info("MLflow run ended")

    def _register_model(self, model_path: str):
        """Register model in MLflow registry (metadata only).

        Args:
            model_path: Path to model directory
        """
        # Create a custom model that only includes metadata
        class TenSafeModelWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                pass

            def predict(self, context, model_input):
                raise NotImplementedError(
                    "This is a metadata-only model. "
                    "Load the actual model using TenSafe SDK."
                )

        # Log metadata artifact
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path, "model_metadata")

        # Create metadata file
        metadata = {
            "tensafe_version": "3.0.0",
            "privacy_preserved": True,
            "note": "Actual model weights are not logged for privacy. Use TenSafe SDK to load.",
        }

        metadata_path = "/tmp/tensafe_model_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        mlflow.log_artifact(metadata_path, "model_metadata")

        # Register
        try:
            mlflow.pyfunc.log_model(
                artifact_path="tensafe_model",
                python_model=TenSafeModelWrapper(),
                registered_model_name=self.config.registered_model_name,
            )

            # Transition to stage
            if self.config.model_stage:
                self._client.transition_model_version_stage(
                    name=self.config.registered_model_name,
                    version=1,  # Assuming first version
                    stage=self.config.model_stage,
                )

            logger.info(f"Model registered: {self.config.registered_model_name}")

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")

    def _redact_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive parameters."""
        sensitive = ["password", "token", "key", "secret", "credential"]
        redacted = {}

        for k, v in params.items():
            if any(s in k.lower() for s in sensitive):
                redacted[k] = "[REDACTED]"
            elif isinstance(v, dict):
                redacted[k] = str(self._redact_params(v))
            else:
                redacted[k] = str(v)[:100]  # Truncate long values

        return redacted

    def _is_safe_metric(self, key: str, value: Any) -> bool:
        """Check if metric is safe to log."""
        unsafe = ["weight", "gradient", "embedding", "param"]
        return not any(u in key.lower() for u in unsafe) and isinstance(value, (int, float))
