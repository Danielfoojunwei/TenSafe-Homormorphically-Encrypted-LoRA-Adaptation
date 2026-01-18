"""
Portability Exporters Module

Generates portable job specs for running TensorGuardFlow PEFT workflows
on various backends without platform lock-in.

Each exporter generates:
1. Job spec (YAML/JSON configuration)
2. Container image reference
3. Required environment variables
4. N2HE sidecar configuration (if privacy.mode=n2he)

Remote submission is optional behind TG_ENABLE_REMOTE_SUBMIT=true.
"""

from .orchestrator_base import BaseOrchestrator, JobSpec, ExportResult
from .orchestrator_k8s_job import K8sJobOrchestrator
from .orchestrator_sagemaker import SageMakerOrchestrator
from .orchestrator_vertex_ai import VertexAIOrchestrator
from .orchestrator_azureml import AzureMLOrchestrator
from .orchestrator_databricks import DatabricksOrchestrator

__all__ = [
    "BaseOrchestrator",
    "JobSpec",
    "ExportResult",
    "K8sJobOrchestrator",
    "SageMakerOrchestrator",
    "VertexAIOrchestrator",
    "AzureMLOrchestrator",
    "DatabricksOrchestrator",
]
