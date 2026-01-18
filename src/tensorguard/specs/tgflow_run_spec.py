"""
TensorGuardFlow Canonical Run Spec

Defines the portable run specification for PEFT workflows that can be executed
across multiple backends (local, K8s, SageMaker, Vertex AI, Azure ML, Databricks).
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ExecutionBackend(str, Enum):
    """Supported execution backends for PEFT training."""
    LOCAL = "local"
    K8S_JOB_EXPORT = "k8s_job_export"
    SAGEMAKER_EXPORT = "sagemaker_export"
    VERTEX_EXPORT = "vertex_export"
    AZUREML_EXPORT = "azureml_export"
    DATABRICKS_EXPORT = "databricks_export"


class TrainingMethod(str, Enum):
    """Supported PEFT training methods."""
    LORA = "lora"
    QLORA = "qlora"


class TrustCore(str, Enum):
    """Trust execution environment."""
    SOFTWARE = "software"
    NITRO_ENCLAVE = "nitro_enclave"


class PrivacyMode(str, Enum):
    """Privacy mode for routing and evaluation."""
    OFF = "off"
    N2HE = "n2he"


class N2HEProfile(str, Enum):
    """N2HE feature profile."""
    ROUTER_ONLY = "router_only"
    ROUTER_PLUS_EVAL = "router_plus_eval"


class N2HESidecar(str, Enum):
    """N2HE sidecar deployment mode."""
    ENABLED = "enabled"
    DISABLED = "disabled"


# --- Nested Config Models ---

class PromotionPolicy(BaseModel):
    """Policy for promoting adapters to release channels."""
    threshold: float = Field(default=0.9, description="Minimum primary metric for promotion")
    forgetting_budget: float = Field(default=0.1, description="Maximum allowed forgetting score")
    regression_budget: float = Field(default=0.05, description="Maximum allowed regression on held-out tasks")


class ReleasePolicy(BaseModel):
    """Policy for releasing adapters to routing."""
    route_key: str = Field(..., description="Routing key for adapter selection")
    channel: str = Field(default="canary", description="Release channel (canary, stable, etc.)")
    rollback_enabled: bool = Field(default=True, description="Enable instant rollback capability")


class TrustConfig(BaseModel):
    """Trust and attestation configuration."""
    trust_core: TrustCore = Field(default=TrustCore.SOFTWARE, description="Trust execution environment")
    attestation_required: bool = Field(default=False, description="Require TEE attestation for signing")


class PrivacyConfig(BaseModel):
    """Privacy configuration for N2HE integration."""
    mode: PrivacyMode = Field(default=PrivacyMode.OFF, description="Privacy mode (off or n2he)")
    n2he_profile: N2HEProfile = Field(default=N2HEProfile.ROUTER_ONLY, description="N2HE feature profile")
    n2he_sidecar: N2HESidecar = Field(default=N2HESidecar.DISABLED, description="N2HE sidecar deployment")
    n2he_encrypted_logs: bool = Field(default=True, description="Encrypt logs when N2HE enabled")
    n2he_service_endpoint: Optional[str] = Field(default=None, description="External N2HE service URL (if sidecar disabled)")


class OutputsConfig(BaseModel):
    """Output artifact requirements."""
    tgsp_required: bool = Field(default=True, description="Generate TGSP package")
    evidence_required: bool = Field(default=True, description="Emit evidence chain events")
    registry_required: bool = Field(default=True, description="Register adapter in registry")
    merged_export: bool = Field(default=False, description="Export merged model (base + adapter)")


class DatasetSource(BaseModel):
    """Dataset source configuration."""
    source_type: Literal["local", "huggingface", "s3", "gcs", "azure_blob"] = "local"
    path: str = Field(..., description="Path or identifier for the dataset")
    split: str = Field(default="train", description="Dataset split to use")
    text_column: str = Field(default="text", description="Column containing text data")


class LoRAConfig(BaseModel):
    """LoRA-specific training configuration."""
    r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, description="Dropout for LoRA layers")
    target_modules: List[str] = Field(default=["q_proj", "v_proj"], description="Target modules for LoRA")
    bias: str = Field(default="none", description="Bias training mode")


class TrainingParams(BaseModel):
    """Training hyperparameters."""
    epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=4, description="Training batch size")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    warmup_steps: int = Field(default=100, description="Warmup steps")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation steps")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    fp16: bool = Field(default=True, description="Use FP16 mixed precision")


# --- Main Run Spec ---

class TGFlowRunSpec(BaseModel):
    """
    TensorGuardFlow Canonical Run Specification.
    
    Portable across all supported backends. Answers:
    1) Which adapter should be used for this request?
    2) What changed since last week?
    3) Did we forget old tasks?
    4) Can we roll back instantly?
    5) Can we deploy into YOUR stack without rewrites?
    """
    
    # Identification
    spec_version: str = Field(default="1.0", description="Run spec schema version")
    run_name: str = Field(..., description="Human-readable run name")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Execution
    execution_backend: ExecutionBackend = Field(
        default=ExecutionBackend.LOCAL,
        description="Target backend for execution"
    )
    training_method: TrainingMethod = Field(
        default=TrainingMethod.LORA,
        description="PEFT training method"
    )
    
    # Model
    base_model_ref: str = Field(..., description="Base model identifier (HF hub or path)")
    
    # Data
    dataset_source: DatasetSource = Field(..., description="Training dataset configuration")
    evaluation_suite_id: Optional[str] = Field(default=None, description="Evaluation suite for quality gates")
    
    # Training
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")
    training_params: TrainingParams = Field(default_factory=TrainingParams, description="Training parameters")
    
    # Policies
    promotion_policy: PromotionPolicy = Field(
        default_factory=PromotionPolicy,
        description="Promotion gate policy"
    )
    release_policy: Optional[ReleasePolicy] = Field(
        default=None,
        description="Release routing policy"
    )
    
    # Trust
    trust: TrustConfig = Field(
        default_factory=TrustConfig,
        description="Trust and attestation configuration"
    )
    
    # Privacy (N2HE)
    privacy: PrivacyConfig = Field(
        default_factory=PrivacyConfig,
        description="Privacy mode configuration (N2HE)"
    )
    
    # Outputs
    outputs: OutputsConfig = Field(
        default_factory=OutputsConfig,
        description="Output artifact requirements"
    )
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict, description="Custom labels")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="Custom annotations")
    
    def is_privacy_enabled(self) -> bool:
        """Check if privacy mode (N2HE) is enabled."""
        return self.privacy.mode == PrivacyMode.N2HE
    
    def requires_sidecar(self) -> bool:
        """Check if N2HE sidecar is required."""
        return (
            self.privacy.mode == PrivacyMode.N2HE and 
            self.privacy.n2he_sidecar == N2HESidecar.ENABLED
        )
    
    def to_exporter_config(self) -> Dict[str, Any]:
        """Generate config for backend exporters."""
        return {
            "backend": self.execution_backend.value,
            "base_model": self.base_model_ref,
            "method": self.training_method.value,
            "privacy_mode": self.privacy.mode.value,
            "sidecar_required": self.requires_sidecar(),
            "trust_core": self.trust.trust_core.value,
            "outputs": self.outputs.model_dump(),
        }
