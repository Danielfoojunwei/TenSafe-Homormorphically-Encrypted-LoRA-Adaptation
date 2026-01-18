"""
Kubernetes Job Orchestrator

Exports TensorGuardFlow PEFT runs as Kubernetes Job specs.
Supports N2HE sidecar co-scheduling.
"""

from typing import Dict, Any, Optional
from .orchestrator_base import BaseOrchestrator, JobSpec, ExportResult


class K8sJobOrchestrator(BaseOrchestrator):
    """Kubernetes Job exporter."""
    
    @property
    def backend_id(self) -> str:
        return "k8s_job_export"
    
    def export(
        self,
        run_spec: Dict[str, Any],
        output_dir: str,
        namespace: str = "default",
        service_account: Optional[str] = None,
        **kwargs
    ) -> ExportResult:
        """
        Export Kubernetes Job YAML.
        
        Args:
            run_spec: TGFlowRunSpec as dict
            output_dir: Output directory
            namespace: K8s namespace
            service_account: Service account name
        """
        job_spec = self._build_base_spec(run_spec)
        
        # Build K8s Job manifest
        job_name = f"tgflow-peft-{run_spec.get('run_name', 'job')[:40]}"
        
        containers = [
            {
                "name": "peft-runner",
                "image": job_spec.image,
                "command": job_spec.command,
                "args": job_spec.args,
                "env": [{"name": k, "value": v} for k, v in job_spec.env_vars.items()],
                "resources": {
                    "limits": {
                        "cpu": job_spec.cpu,
                        "memory": job_spec.memory,
                    },
                },
                "volumeMounts": [
                    {"name": "run-spec", "mountPath": "/app/run_spec.json", "subPath": "run_spec.json"},
                ],
            }
        ]
        
        # Add GPU resources
        if job_spec.gpu:
            containers[0]["resources"]["limits"]["nvidia.com/gpu"] = job_spec.gpu
        
        # Add N2HE sidecar
        if job_spec.n2he_sidecar_enabled:
            containers.append({
                "name": "n2he-sidecar",
                "image": job_spec.n2he_sidecar_image,
                "ports": [{"containerPort": 8765}],
                "env": [{"name": "N2HE_MODE", "value": "sidecar"}],
            })
        
        k8s_job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "labels": job_spec.labels,
            },
            "spec": {
                "backoffLimit": 2,
                "template": {
                    "metadata": {"labels": job_spec.labels},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": containers,
                        "volumes": [
                            {
                                "name": "run-spec",
                                "configMap": {"name": f"{job_name}-config"},
                            }
                        ],
                    },
                },
            },
        }
        
        if service_account:
            k8s_job["spec"]["template"]["spec"]["serviceAccountName"] = service_account
        
        return self._save_spec(job_spec, k8s_job, run_spec, output_dir, "job.yaml")
