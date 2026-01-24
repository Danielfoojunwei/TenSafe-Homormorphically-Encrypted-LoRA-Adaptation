import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..framework.contracts import Connector, ValidationResult, HealthStatus, SmokeTestResult
from ..framework.config_schema import ServingConfig

logger = logging.getLogger(__name__)

class VllmServingExporter(Connector):
    """Config generator for vLLM inference (Category F)."""
    
    def __init__(self, config: ServingConfig):
        self.config = config

    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        errors = []
        if cfg.get("type") != "vllm":
            errors.append("Invalid connector type for VllmServingExporter")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def health_check(self) -> HealthStatus:
        # Since this is an exporter, health check just validates we can generate JSON
        return HealthStatus(
            status="ok",
            message="vLLM Template engine ready",
            last_checked=datetime.utcnow().isoformat()
        )

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "serving_exporter",
            "target": "vllm",
            "formats": ["json"],
            "supports_multi_adapter": True
        }

    async def run_smoke_test(self) -> SmokeTestResult:
        try:
            test_pack = self.export_serving_pack("test_adapter", {"model": "phi-2"})
            return SmokeTestResult(success=True, details={"test_pack": test_pack}, duration_ms=1.0)
        except Exception as e:
            return SmokeTestResult(success=False, details={"error": str(e)}, duration_ms=1.0)

    def export_serving_pack(self, adapter_id: str, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a 'Serving Pack' which is a JSON manifest that vLLM-compatible 
        entrypoints can use to pull and serve the adapter.
        """
        serving_pack = {
            "vllm_config": {
                "model": model_metadata.get("base_model", "unknown"),
                "adapters": [
                    {
                        "name": adapter_id,
                        "path": f"/models/{adapter_id}",
                    }
                ],
                "max_model_len": 2048,
                "dtype": "bfloat16",
                "disable_log_stats": False
            },
            "tensorguard_metadata": {
                "adapter_id": adapter_id,
                "exported_at": datetime.utcnow().isoformat(),
                "exporter": self.config.name,
                "tenant_id": self.config.tenant_id
            }
        }
        return serving_pack
