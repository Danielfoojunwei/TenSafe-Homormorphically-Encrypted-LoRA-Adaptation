from enum import Enum
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from pydantic import BaseModel

class IntegrationType(str, Enum):
    DATA_SOURCE = "data_source"
    TRAINING_EXECUTOR = "training_executor"
    MODEL_REGISTRY = "model_registry"
    SERVING_TARGET = "serving_target"
    OBSERVABILITY = "observability"
    SECRETS_KMS = "secrets_kms"
    VECTOR_STORE = "vector_store"
    TRUST_ANCHOR = "trust_anchor"
    PRIVACY_PROVIDER = "privacy_provider"

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

class HealthStatus(BaseModel):
    status: str  # "ok", "warning", "error", "unreachable"
    message: str
    latency_ms: Optional[float] = None
    last_checked: str

class SmokeTestResult(BaseModel):
    success: bool
    details: Dict[str, Any]
    duration_ms: float

@runtime_checkable
class Connector(Protocol):
    """Base interface for all TensorGuard integration connectors."""
    
    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        """Validate the provided configuration against the connector's schema."""
        ...

    async def health_check(self) -> HealthStatus:
        """Perform a real-time health check (connectivity, permissions)."""
        ...

    def describe_capabilities(self) -> Dict[str, Any]:
        """Return a dictionary of supported features and limitations."""
        ...

    async def run_smoke_test(self) -> SmokeTestResult:
        """Execute a minimal, non-destructive end-to-end operation."""
        ...

class ConnectionHandle:
    """Generic wrapper for native client handles (e.g., S3 client, K8s api client)."""
    def __init__(self, client: Any, metadata: Dict[str, Any] = None):
        self.client = client
        self.metadata = metadata or {}
        self.connected_at = "" # To be set by connector
