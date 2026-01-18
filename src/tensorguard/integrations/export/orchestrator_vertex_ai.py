"""
Google Cloud Vertex AI Orchestrator

Exports TensorGuardFlow PEFT runs as Vertex AI Custom Training Job specs.
"""

from typing import Dict, Any, Optional
from .orchestrator_base import BaseOrchestrator, JobSpec, ExportResult


class VertexAIOrchestrator(BaseOrchestrator):
    """GCP Vertex AI Custom Training Job exporter."""
    
    DEFAULT_MACHINE_TYPE = "n1-standard-4"
    DEFAULT_ACCELERATOR = "NVIDIA_TESLA_T4"
    
    @property
    def backend_id(self) -> str:
        return "vertex_export"
    
    def export(
        self,
        run_spec: Dict[str, Any],
        output_dir: str,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: int = 1,
        **kwargs
    ) -> ExportResult:
        """
        Export Vertex AI Custom Training Job config.
        
        Args:
            run_spec: TGFlowRunSpec as dict
            output_dir: Output directory
            project_id: GCP project ID
            location: GCP region
            machine_type: Compute Engine machine type
            accelerator_type: GPU accelerator type
            accelerator_count: Number of GPUs
        """
        job_spec = self._build_base_spec(run_spec)
        
        display_name = f"tgflow-peft-{run_spec.get('run_name', 'job')[:40]}"
        
        # Build Vertex AI Custom Job spec
        vertex_config = {
            "displayName": display_name,
            "jobSpec": {
                "workerPoolSpecs": [
                    {
                        "machineSpec": {
                            "machineType": machine_type or self.DEFAULT_MACHINE_TYPE,
                            "acceleratorType": accelerator_type or self.DEFAULT_ACCELERATOR,
                            "acceleratorCount": accelerator_count,
                        },
                        "replicaCount": 1,
                        "containerSpec": {
                            "imageUri": job_spec.image,
                            "command": job_spec.command,
                            "args": job_spec.args,
                            "env": [{"name": k, "value": v} for k, v in job_spec.env_vars.items()],
                        },
                    }
                ],
            },
            "labels": {k.replace(".", "_"): v for k, v in job_spec.labels.items()},
        }
        
        # Add N2HE sidecar as secondary worker pool
        if job_spec.n2he_sidecar_enabled:
            vertex_config["jobSpec"]["workerPoolSpecs"].append({
                "machineSpec": {"machineType": "n1-standard-2"},
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": job_spec.n2he_sidecar_image,
                    "env": [{"name": "N2HE_MODE", "value": "sidecar"}],
                },
            })
        
        return self._save_spec(job_spec, vertex_config, run_spec, output_dir, "vertex_job.json")
