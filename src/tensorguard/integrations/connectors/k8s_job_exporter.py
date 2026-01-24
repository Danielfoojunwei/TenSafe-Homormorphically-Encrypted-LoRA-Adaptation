import yaml
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..framework.contracts import Connector, ValidationResult, HealthStatus, SmokeTestResult
from ..framework.config_schema import TrainingExecutorConfig

logger = logging.getLogger(__name__)

class K8sJobExporter(Connector):
    """Generates Kubernetes Job manifests for training (Category D)."""
    
    def __init__(self, config: TrainingExecutorConfig):
        self.config = config

    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        errors = []
        if not cfg.get("cluster_ref"):
            errors.append("Missing cluster_ref")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            status="ok",
            message="K8s Template generator active",
            last_checked=datetime.utcnow().isoformat()
        )

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "training_exporter",
            "target": "kubernetes",
            "supports_gpu_scheduling": True,
            "supports_node_selectors": True
        }

    async def run_smoke_test(self) -> SmokeTestResult:
        # Validate that we can generate a valid YAML
        try:
            self.export_job_spec("test-job", {"image": "tensorguard/trainer:latest"})
            return SmokeTestResult(success=True, details={}, duration_ms=1.0)
        except Exception as e:
            return SmokeTestResult(success=False, details={"error": str(e)}, duration_ms=1.0)

    def export_job_spec(self, job_id: str, training_spec: Dict[str, Any]) -> str:
        """Generates a Kubernetes Job YAML string."""
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_id,
                "namespace": self.config.env_vars.get("NAMESPACE", "default"),
                "labels": {
                    "tensorguard-route": training_spec.get("route_key", "unknown"),
                    "tenant": self.config.tenant_id
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "trainer",
                            "image": training_spec.get("image", "tensorguard/trainer:latest"),
                            "env": [
                                {"name": "TG_ROUTE_KEY", "value": training_spec.get("route_key")},
                                {"name": "TG_TENANT_ID", "value": self.config.tenant_id}
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": 1
                                }
                            }
                        }],
                        "restartPolicy": "Never"
                    }
                },
                "backoffLimit": 0
            }
        }
        return yaml.dump(manifest)
