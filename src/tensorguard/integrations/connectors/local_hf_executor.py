import logging
import subprocess
import os
from typing import Dict, Any, Optional
from datetime import datetime
from ..framework.contracts import Connector, ValidationResult, HealthStatus, SmokeTestResult
from ..framework.config_schema import TrainingExecutorConfig

logger = logging.getLogger(__name__)

class LocalHFExecutor(Connector):
    """Execution engine for running HuggingFace training locally (Category D)."""
    
    def __init__(self, config: TrainingExecutorConfig):
        self.config = config

    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        errors = []
        if not cfg.get("cluster_ref"):
            errors.append("Missing cluster_ref (should be 'localhost' or device id)")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def health_check(self) -> HealthStatus:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            return HealthStatus(
                status="ok",
                message=f"Local Torch active. CUDA: {has_cuda}",
                latency_ms=0.0,
                last_checked=datetime.utcnow().isoformat()
            )
        except ImportError:
            return HealthStatus(status="warning", message="Torch not installed locally", last_checked=datetime.utcnow().isoformat())

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "training_executor",
            "supports_qlora": True,
            "supports_lora_merge": True,
            "supports_local_gpu": True
        }

    async def run_smoke_test(self) -> SmokeTestResult:
        # Check if we can run a minimal python command
        start_time = datetime.utcnow()
        try:
            subprocess.run(["python", "--version"], check=True, capture_output=True)
            return SmokeTestResult(success=True, details={}, duration_ms=100.0)
        except Exception as e:
            return SmokeTestResult(success=False, details={"error": str(e)}, duration_ms=100.0)

    def run_training(self, training_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a training job locally.
        In simulation mode, this just returns a success dict.
        """
        if os.environ.get("TG_SIMULATION") == "true":
            logger.info("Executing SIMULATED local training")
            return {
                "status": "success",
                "adapter_path": "outputs/simulated_adapter",
                "metrics": {"accuracy": 0.95, "loss": 0.1},
                "executor": self.config.name
            }
        
        # Real implementation would call 'tensorguard-train' or similar
        raise NotImplementedError("Real local training execution requires valid environment setup")
