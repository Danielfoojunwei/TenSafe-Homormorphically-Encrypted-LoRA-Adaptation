import os
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..framework.contracts import Connector, ValidationResult, HealthStatus, SmokeTestResult
from ..framework.config_schema import DataSourceConfig

logger = logging.getLogger(__name__)

class LocalFilesystemConnector(Connector):
    """Connector for ingesting data from the local filesystem (Category C)."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.base_path = config.uri

    def validate_config(self, cfg: Dict[str, Any]) -> ValidationResult:
        errors = []
        uri = cfg.get("uri")
        if not uri:
            errors.append("Missing required field: uri")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def health_check(self) -> HealthStatus:
        start_time = datetime.utcnow()
        if os.path.exists(self.base_path):
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return HealthStatus(
                status="ok",
                message=f"Path exists: {self.base_path}",
                latency_ms=latency,
                last_checked=datetime.utcnow().isoformat()
            )
        return HealthStatus(
            status="error",
            message=f"Path not found: {self.base_path}",
            last_checked=datetime.utcnow().isoformat()
        )

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "data_source",
            "subtype": "local_filesystem",
            "supports_streaming": False,
            "supports_versioning": False
        }

    async def run_smoke_test(self) -> SmokeTestResult:
        start_time = datetime.utcnow()
        try:
            # Check if directory is readable
            files = os.listdir(self.base_path)
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return SmokeTestResult(
                success=True,
                details={"file_count": len(files)},
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return SmokeTestResult(
                success=False,
                details={"error": str(e)},
                duration_ms=duration
            )

    def ingest_snapshot(self, relative_uri: str) -> Dict[str, Any]:
        """Reads data and returns a fingerprint for the evidence chain."""
        full_path = os.path.join(self.base_path, relative_uri)

        # Handle simulation mode gracefully
        if os.getenv("TG_SIMULATION", "false").lower() == "true":
            if not os.path.exists(full_path):
                # Return simulated data in simulation mode
                simulated_hash = hashlib.sha256(f"simulated-{full_path}-{datetime.utcnow().isoformat()}".encode()).hexdigest()
                logger.info(f"Simulation mode: returning simulated snapshot for {full_path}")
                return {
                    "path": full_path,
                    "content_hash": f"sha256:{simulated_hash}",
                    "data_hash": f"sha256:{simulated_hash}",
                    "ingested_at": datetime.utcnow().isoformat(),
                    "connector": self.config.name,
                    "count": 100,
                    "content": f"simulated_content_{full_path}",
                    "simulated": True
                }

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data snapshot not found at {full_path}")

        # Calculate hash for evidence integrity
        sha256 = hashlib.sha256()
        with open(full_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        return {
            "path": full_path,
            "data_hash": f"sha256:{sha256.hexdigest()}",
            "content_hash": f"sha256:{sha256.hexdigest()}",
            "ingested_at": datetime.utcnow().isoformat(),
            "connector": self.config.name,
            "count": 100,
            "content": full_path
        }
