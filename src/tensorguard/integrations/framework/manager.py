import logging
import importlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
from .contracts import Connector, IntegrationType, ValidationResult, HealthStatus
from .config_schema import IntegrationProfile, DataSourceConfig, TrainingExecutorConfig, ServingConfig

logger = logging.getLogger(__name__)

class IntegrationManager:
    """
    Registry and lifecycle manager for all 3rd-party integrations.
    
    Loads configuration for a specific tenant, instantiates connectors,
    and provides unified access to the C-D-E-F-G pipeline adapters.
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.connectors: Dict[str, Connector] = {}
        self.profile: Optional[IntegrationProfile] = None
        self._load_profile()

    def _load_profile(self):
        """
        Loads the integration profile from the database or environment.
        Currently defaults to a mock profile for local development.
        """
        # In production, this would query the DB for the tenant's IntegrationConfig
        # For now, we'll return a minimal profile with local defaults
        self.profile = IntegrationProfile(
            tenant_id=self.tenant_id,
            data_sources=[
                DataSourceConfig(tenant_id=self.tenant_id, name="local_feed", type="local", uri="data/raw")
            ],
            training_executors=[
                TrainingExecutorConfig(tenant_id=self.tenant_id, name="local_trainer", type="local", cluster_ref="localhost")
            ],
            serving_targets=[
                ServingConfig(tenant_id=self.tenant_id, name="vllm_exporter", type="vllm")
            ]
        )
        logger.info(f"Loaded integration profile for tenant={self.tenant_id}")

    def get_connector(self, integration_name: str) -> Optional[Connector]:
        """Returns an instantiated connector by its configured name."""
        if integration_name in self.connectors:
            return self.connectors[integration_name]
        
        # Look up in profile
        config = self._find_config(integration_name)
        if not config:
            logger.error(f"Connector config not found: {integration_name}")
            return None
        
        # Instantiate
        try:
            connector = self._instantiate_connector(config)
            self.connectors[integration_name] = connector
            return connector
        except Exception as e:
            logger.exception(f"Failed to instantiate connector {integration_name}")
            return None

    def _find_config(self, name: str) -> Optional[Any]:
        """Search across all config categories for a named connector."""
        all_configs = (
            self.profile.data_sources + 
            self.profile.training_executors + 
            self.profile.registries + 
            self.profile.serving_targets
        )
        for cfg in all_configs:
            if cfg.name == name:
                return cfg
        return None

    def _instantiate_connector(self, config: Any) -> Connector:
        """
        Dynamic mapping of connector classes based on type.
        """
        from ..connectors.local_filesystem import LocalFilesystemConnector
        from ..connectors.local_hf_executor import LocalHFExecutor
        from ..connectors.vllm_serving_exporter import VllmServingExporter
        from ..connectors.s3_feed import S3FeedConnector
        from ..connectors.k8s_job_exporter import K8sJobExporter
        from ..connectors.sagemaker_exporter import SageMakerExporter
        from ..connectors.tgi_serving_exporter import TgiServingExporter
        from ..connectors.n2he_privacy_connector import N2HEPrivacyConnector
        from ..connectors.gcs_feed import GCSFeedConnector
        from ..connectors.azure_blob import AzureBlobConnector
        from ..connectors.hf_dataset import HFDatasetConnector
        from ..connectors.vertex_ai_exporter import VertexAIExporter
        from ..connectors.triton_serving_exporter import TritonServingExporter
        from ..connectors.nitro_trust import NitroTrustConnector
        from ..connectors.mlflow_connector import MLflowConnector

        mapping = {
            "local": LocalFilesystemConnector,
            "local_hf": LocalHFExecutor, # Special case for trainer
            "vllm": VllmServingExporter,
            "s3": S3FeedConnector,
            "gcs": GCSFeedConnector,
            "azure": AzureBlobConnector,
            "hf": HFDatasetConnector,
            "k8s": K8sJobExporter,
            "sagemaker": SageMakerExporter,
            "sagemaker_endpoint": SageMakerExporter, # Shared for now
            "vertex": VertexAIExporter,
            "tgi": TgiServingExporter,
            "triton": TritonServingExporter,
            "n2he": N2HEPrivacyConnector,
            "nitro": NitroTrustConnector,
            "mlflow": MLflowConnector
        }
        
        # Determine the key
        type_val = config.type.value if hasattr(config.type, "value") else config.type
        
        key = type_val
        if type_val == "local" and hasattr(config, "cluster_ref"):
            key = "local_hf"
            
        connector_class = mapping.get(key)
        if not connector_class:
            logger.error(f"No connector implementation for type: {key} (type_val: {type_val})")
            raise ValueError(f"No connector implementation for type: {key}")
            
        return connector_class(config)

    async def get_compatibility_snapshot(self) -> Dict[str, Any]:
        """
        Generates a summary of all active integrations, their health, 
        and capabilities. Used for TGSP evidence fingerprints.
        """
        print(f"DEBUG: get_compatibility_snapshot for tenant={self.tenant_id}")
        snapshot = {
            "tenant_id": self.tenant_id,
            "integrations": {},
            "timestamp": datetime.utcnow().isoformat() 
        }
        
        all_configs = (
            self.profile.data_sources + 
            self.profile.training_executors + 
            self.profile.registries + 
            self.profile.serving_targets
        )
        print(f"DEBUG: all_configs count={len(all_configs)}")
        
        for cfg in all_configs:
            print(f"DEBUG: Processing config name={cfg.name}, type={cfg.type}")
            connector = self.get_connector(cfg.name)
            if connector:
                print(f"DEBUG: Found connector for {cfg.name}")
                try:
                    health = await connector.health_check()
                    snapshot["integrations"][cfg.name] = {
                        "type": cfg.type.value if hasattr(cfg.type, "value") else cfg.type,
                        "status": health.status,
                        "capabilities": connector.describe_capabilities()
                    }
                except Exception as e:
                    print(f"DEBUG: Health check failed for {cfg.name}: {e}")
            else:
                print(f"DEBUG: Connector NOT FOUND for {cfg.name}")
        
        return snapshot

class MockConnector(Connector):
    """Stub connector for Phase 0 framework validation."""
    def __init__(self, config: Any):
        self.config = config
        
    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        return ValidationResult(is_valid=True)
        
    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="ok", message="Simulation active", last_checked="now")
        
    def describe_capabilities(self) -> Dict[str, Any]:
        return {"simulated": True}
        
    async def run_smoke_test(self) -> Any:
        return {"success": True}
