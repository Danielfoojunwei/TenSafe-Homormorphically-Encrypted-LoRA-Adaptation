from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, SecretStr
from enum import Enum

class DataSourceType(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    HF = "hf"

class TrainingExecutorType(str, Enum):
    LOCAL = "local"
    K8S = "k8s"
    SAGEMAKER = "sagemaker"
    VERTEX = "vertex"
    AZUREML = "azureml"
    DATABRICKS = "databricks"

class ServingTargetType(str, Enum):
    VLLM = "vllm"
    TGI = "tgi"
    TRITON = "triton"
    SAGEMAKER = "sagemaker_endpoint"
    BEDROCK = "bedrock_import"

# Base Config
class BaseIntegrationConfig(BaseModel):
    tenant_id: str = Field(..., description="Owner tenant for this integration")
    name: str = Field(..., description="Unique name for this instance")
    is_active: bool = Field(default=True)

# C: Data Sources
class DataSourceConfig(BaseIntegrationConfig):
    type: DataSourceType
    uri: str
    credentials_secret_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

# D: Training Executors
class TrainingExecutorConfig(BaseIntegrationConfig):
    type: TrainingExecutorType
    cluster_ref: str
    max_parallel_jobs: int = 1
    job_spec_template: Optional[str] = None
    env_vars: Dict[str, str] = Field(default_factory=dict)

# E: Model Registry (Internal is default, but could be external MLflow)
class RegistryConfig(BaseIntegrationConfig):
    type: str = "internal" # "internal", "mlflow", "wandb"
    endpoint: Optional[str] = None
    api_key: Optional[SecretStr] = None

# F: Serving Targets
class ServingConfig(BaseIntegrationConfig):
    type: ServingTargetType
    export_format: str = "json" # "json", "yaml", "env"
    target_bucket: Optional[str] = None
    promotion_webhook: Optional[str] = None

# G: Trust & Privacy
class TrustConfig(BaseIntegrationConfig):
    type: str = "nitro" # "nitro", "plain"
    public_key_id: str
    attestation_required: bool = True

class PrivacyConfig(BaseIntegrationConfig):
    type: str = "n2he"
    n2he_profile: str = "router_only" # "router_only", "full"
    receipt_storage: str = "local"

# Combined Integration Profile
class IntegrationProfile(BaseModel):
    tenant_id: str
    data_sources: List[DataSourceConfig] = []
    training_executors: List[TrainingExecutorConfig] = []
    registries: List[RegistryConfig] = []
    serving_targets: List[ServingConfig] = []
    trust_settings: Optional[TrustConfig] = None
    privacy_settings: Optional[PrivacyConfig] = None
