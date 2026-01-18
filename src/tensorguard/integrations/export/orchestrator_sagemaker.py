"""
AWS SageMaker Orchestrator

Exports TensorGuardFlow PEFT runs as SageMaker Training Job specs.
Supports N2HE via SageMaker Processing or co-located container.
"""

from typing import Dict, Any, Optional
from .orchestrator_base import BaseOrchestrator, JobSpec, ExportResult


class SageMakerOrchestrator(BaseOrchestrator):
    """AWS SageMaker Training Job exporter."""
    
    DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
    
    @property
    def backend_id(self) -> str:
        return "sagemaker_export"
    
    def export(
        self,
        run_spec: Dict[str, Any],
        output_dir: str,
        role_arn: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: int = 1,
        s3_output_path: Optional[str] = None,
        **kwargs
    ) -> ExportResult:
        """
        Export SageMaker Training Job config.
        
        Args:
            run_spec: TGFlowRunSpec as dict
            output_dir: Output directory
            role_arn: IAM role ARN for SageMaker
            instance_type: EC2 instance type
            s3_output_path: S3 path for outputs
        """
        job_spec = self._build_base_spec(run_spec)
        
        job_name = f"tgflow-peft-{run_spec.get('run_name', 'job')[:40]}"
        
        # Build SageMaker Training Job config
        sm_config = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": job_spec.image,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": job_spec.command,
                "ContainerArguments": job_spec.args,
            },
            "RoleArn": role_arn or "${SAGEMAKER_ROLE_ARN}",
            "ResourceConfig": {
                "InstanceType": instance_type or self.DEFAULT_INSTANCE_TYPE,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": 100,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400,  # 24 hours
            },
            "OutputDataConfig": {
                "S3OutputPath": s3_output_path or "${S3_OUTPUT_PATH}",
            },
            "Environment": job_spec.env_vars,
            "Tags": [{"Key": k, "Value": v} for k, v in job_spec.labels.items()],
        }
        
        # N2HE configuration
        if job_spec.n2he_sidecar_enabled:
            sm_config["Environment"]["N2HE_MODE"] = "embedded"
            sm_config["Environment"]["TENSEAL_AVAILABLE"] = "true"
        elif job_spec.n2he_service_endpoint:
            sm_config["Environment"]["N2HE_SERVICE_URL"] = job_spec.n2he_service_endpoint
        
        return self._save_spec(job_spec, sm_config, run_spec, output_dir, "sagemaker_job.json")
