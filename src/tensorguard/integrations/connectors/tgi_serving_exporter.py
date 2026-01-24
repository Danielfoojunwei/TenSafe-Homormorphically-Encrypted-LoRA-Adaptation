import yaml
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..framework.contracts import Connector, ValidationResult, HealthStatus, SmokeTestResult
from ..framework.config_schema import ServingConfig

logger = logging.getLogger(__name__)

class TgiServingExporter(Connector):
    """Config generator for HuggingFace Text-Generation-Inference (Category F)."""
    
    def __init__(self, config: ServingConfig):
        self.config = config

    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        errors = []
        if cfg.get("type") != "tgi":
            errors.append("Invalid connector type for TgiServingExporter")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            status="ok",
            message="TGI Template engine ready",
            last_checked=datetime.utcnow().isoformat()
        )

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "serving_exporter",
            "target": "tgi",
            "formats": ["yaml", "env"],
            "supports_sharding": True
        }

    async def run_smoke_test(self) -> SmokeTestResult:
        try:
            self.export_serving_pack("tgi-test", {"model": "phi-2"})
            return SmokeTestResult(success=True, details={}, duration_ms=1.0)
        except Exception as e:
            return SmokeTestResult(success=False, details={"error": str(e)}, duration_ms=1.0)

    def export_serving_pack(self, adapter_id: str, model_metadata: Dict[str, Any]) -> str:
        """
        Generates a TGI-compatible Docker Compose or CLI spec.
        """
        config = {
            "model_id": model_metadata.get("base_model", "unknown"),
            "revision": "main",
            "port": 8080,
            "sharded": self.config.params.get("sharded", "false"),
            "num_shard": self.config.params.get("num_shard", 1),
            "env": {
                "ADAPTER_ID": adapter_id,
                "TENANT_ID": self.config.tenant_id
            }
        }
        return yaml.dump(config)
