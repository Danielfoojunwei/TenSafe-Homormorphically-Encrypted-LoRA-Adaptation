"""
Azure Machine Learning Orchestrator

Exports TensorGuardFlow PEFT runs as Azure ML Job specs.
"""

from typing import Dict, Any, Optional
from .orchestrator_base import BaseOrchestrator, JobSpec, ExportResult


class AzureMLOrchestrator(BaseOrchestrator):
    """Azure Machine Learning Job exporter."""
    
    DEFAULT_VM_SIZE = "Standard_NC4as_T4_v3"
    
    @property
    def backend_id(self) -> str:
        return "azureml_export"
    
    def export(
        self,
        run_spec: Dict[str, Any],
        output_dir: str,
        workspace_name: Optional[str] = None,
        resource_group: Optional[str] = None,
        compute_target: Optional[str] = None,
        vm_size: Optional[str] = None,
        **kwargs
    ) -> ExportResult:
        """
        Export Azure ML Command Job config.
        
        Args:
            run_spec: TGFlowRunSpec as dict
            output_dir: Output directory
            workspace_name: Azure ML workspace name
            resource_group: Azure resource group
            compute_target: Compute cluster name
            vm_size: VM size for compute
        """
        job_spec = self._build_base_spec(run_spec)
        
        job_name = f"tgflow-peft-{run_spec.get('run_name', 'job')[:40]}"
        
        # Build Azure ML Job config (SDK v2 format)
        azureml_config = {
            "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
            "type": "command",
            "display_name": job_name,
            "experiment_name": "tensorguardflow-peft",
            "compute": compute_target or "${AZUREML_COMPUTE_TARGET}",
            "environment": {
                "image": job_spec.image,
            },
            "command": " ".join(job_spec.command + job_spec.args),
            "environment_variables": job_spec.env_vars,
            "resources": {
                "instance_type": vm_size or self.DEFAULT_VM_SIZE,
                "instance_count": 1,
            },
            "tags": job_spec.labels,
        }
        
        # N2HE configuration via environment
        if job_spec.n2he_sidecar_enabled:
            azureml_config["environment_variables"]["N2HE_MODE"] = "embedded"
        elif job_spec.n2he_service_endpoint:
            azureml_config["environment_variables"]["N2HE_SERVICE_URL"] = job_spec.n2he_service_endpoint
        
        return self._save_spec(job_spec, azureml_config, run_spec, output_dir, "azureml_job.yaml")
