"""
Databricks Orchestrator

Exports TensorGuardFlow PEFT runs as Databricks Job specs.
Supports Unity Catalog integration and MLflow tracking.
"""

from typing import Dict, Any, Optional, List
from .orchestrator_base import BaseOrchestrator, JobSpec, ExportResult


class DatabricksOrchestrator(BaseOrchestrator):
    """Databricks Job exporter."""
    
    DEFAULT_NODE_TYPE = "Standard_NC4as_T4_v3"  # Azure
    DEFAULT_SPARK_VERSION = "13.3.x-gpu-ml-scala2.12"
    
    @property
    def backend_id(self) -> str:
        return "databricks_export"
    
    def export(
        self,
        run_spec: Dict[str, Any],
        output_dir: str,
        workspace_url: Optional[str] = None,
        node_type_id: Optional[str] = None,
        spark_version: Optional[str] = None,
        num_workers: int = 0,
        cluster_id: Optional[str] = None,
        **kwargs
    ) -> ExportResult:
        """
        Export Databricks Job config.
        
        Args:
            run_spec: TGFlowRunSpec as dict
            output_dir: Output directory
            workspace_url: Databricks workspace URL
            node_type_id: Node type for cluster
            spark_version: Databricks Runtime version
            num_workers: Number of worker nodes (0 for single-node)
            cluster_id: Existing cluster ID (if using existing cluster)
        """
        job_spec = self._build_base_spec(run_spec)
        
        job_name = f"tgflow-peft-{run_spec.get('run_name', 'job')[:40]}"
        
        # Build cluster config
        cluster_spec = {
            "spark_version": spark_version or self.DEFAULT_SPARK_VERSION,
            "node_type_id": node_type_id or self.DEFAULT_NODE_TYPE,
            "num_workers": num_workers,
            "spark_env_vars": job_spec.env_vars,
            "docker_image": {
                "url": job_spec.image,
            },
        }
        
        # Build Databricks Job config
        databricks_config = {
            "name": job_name,
            "tags": job_spec.labels,
            "tasks": [
                {
                    "task_key": "peft_training",
                    "description": "TensorGuardFlow PEFT Training",
                    "new_cluster": cluster_spec if not cluster_id else None,
                    "existing_cluster_id": cluster_id,
                    "spark_python_task": {
                        "python_file": "dbfs:/tensorguardflow/run_peft.py",
                        "parameters": ["--spec", "run_spec.json"],
                    },
                    "libraries": [
                        {"pypi": {"package": "tensorguardflow"}},
                        {"pypi": {"package": "peft"}},
                        {"pypi": {"package": "transformers"}},
                    ],
                }
            ],
            "max_concurrent_runs": 1,
            "format": "MULTI_TASK",
        }
        
        # Remove None cluster config
        if cluster_id:
            databricks_config["tasks"][0].pop("new_cluster", None)
        else:
            databricks_config["tasks"][0].pop("existing_cluster_id", None)
        
        # N2HE configuration
        if job_spec.n2he_sidecar_enabled or job_spec.n2he_service_endpoint:
            cluster_spec["spark_env_vars"]["N2HE_MODE"] = "embedded" if job_spec.n2he_sidecar_enabled else "remote"
            if job_spec.n2he_service_endpoint:
                cluster_spec["spark_env_vars"]["N2HE_SERVICE_URL"] = job_spec.n2he_service_endpoint
            databricks_config["tasks"][0]["libraries"].append({"pypi": {"package": "tenseal"}})
        
        return self._save_spec(job_spec, databricks_config, run_spec, output_dir, "databricks_job.json")
