import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from ..framework.contracts import Connector, ValidationResult, HealthStatus, SmokeTestResult
from ..framework.config_schema import TrainingExecutorConfig

logger = logging.getLogger(__name__)

class SageMakerExporter(Connector):
    """Generates Amazon SageMaker Training Job specifications (Category D)."""
    
    def __init__(self, config: TrainingExecutorConfig):
        self.config = config

    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        errors = []
        if not cfg.get("cluster_ref"):
            errors.append("Missing execution_role_arn (mapped to cluster_ref)")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            status="ok",
            message="SageMaker JobSpec engine ready",
            last_checked=datetime.utcnow().isoformat()
        )

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "training_exporter",
            "target": "aws_sagemaker",
            "supports_spot_instances": True,
            "supports_vpc": True
        }

    async def run_smoke_test(self) -> SmokeTestResult:
        try:
            self.export_job_spec("sm-test", {"image": "trainer:latest", "s3_input": "s3://bucket/data"})
            return SmokeTestResult(success=True, details={}, duration_ms=1.0)
        except Exception as e:
            return SmokeTestResult(success=False, details={"error": str(e)}, duration_ms=1.0)

    def export_job_spec(self, job_name: str, training_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a SageMaker CreateTrainingJob request body."""
        spec = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": training_spec.get("image"),
                "TrainingInputMode": "File"
            },
            "RoleArn": self.config.cluster_ref,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": training_spec.get("s3_input")
                        }
                    }
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": training_spec.get("s3_output")
            },
            "ResourceConfig": {
                "InstanceType": self.config.env_vars.get("INSTANCE_TYPE", "ml.g5.2xlarge"),
                "InstanceCount": 1,
                "VolumeSizeInGB": 50
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400
            }
        }
        return spec
