"""
Base Orchestrator - Common Exporter Interface

Defines the base class and data structures for all portability exporters.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class JobSpec:
    """Portable job specification."""
    spec_version: str = "1.0"
    backend: str = ""
    
    # Container
    image: str = ""
    image_digest: Optional[str] = None
    
    # Resources
    cpu: str = "4"
    memory: str = "16Gi"
    gpu: Optional[str] = None
    gpu_type: Optional[str] = None
    
    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    
    # N2HE Sidecar
    n2he_sidecar_enabled: bool = False
    n2he_sidecar_image: Optional[str] = None
    n2he_service_endpoint: Optional[str] = None
    
    # Volumes
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Run config
    run_spec_path: Optional[str] = None
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "backend": self.backend,
            "container": {
                "image": self.image,
                "image_digest": self.image_digest,
            },
            "resources": {
                "cpu": self.cpu,
                "memory": self.memory,
                "gpu": self.gpu,
                "gpu_type": self.gpu_type,
            },
            "environment": {
                "env_vars": self.env_vars,
                "secrets": self.secrets,
            },
            "n2he_sidecar": {
                "enabled": self.n2he_sidecar_enabled,
                "image": self.n2he_sidecar_image,
                "service_endpoint": self.n2he_service_endpoint,
            },
            "volumes": self.volumes,
            "execution": {
                "run_spec_path": self.run_spec_path,
                "command": self.command,
                "args": self.args,
            },
            "metadata": {
                "labels": self.labels,
                "annotations": self.annotations,
                "generated_at": self.generated_at,
            },
        }


@dataclass
class ExportResult:
    """Result of job spec export."""
    success: bool
    job_spec: Optional[JobSpec] = None
    output_path: Optional[str] = None
    native_spec: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "job_spec": self.job_spec.to_dict() if self.job_spec else None,
            "error": self.error,
        }


class BaseOrchestrator(ABC):
    """Base class for backend orchestrators."""
    
    # Default TensorGuardFlow PEFT image
    DEFAULT_IMAGE = "tensorguardflow/peft-runner:latest"
    DEFAULT_N2HE_SIDECAR = "tensorguardflow/n2he-sidecar:latest"
    
    def __init__(
        self,
        image: Optional[str] = None,
        n2he_sidecar_image: Optional[str] = None,
    ):
        self.image = image or self.DEFAULT_IMAGE
        self.n2he_sidecar_image = n2he_sidecar_image or self.DEFAULT_N2HE_SIDECAR
        self.remote_submit_enabled = os.getenv("TG_ENABLE_REMOTE_SUBMIT", "false").lower() == "true"
    
    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Backend identifier."""
        pass
    
    @abstractmethod
    def export(
        self,
        run_spec: Dict[str, Any],
        output_dir: str,
        **kwargs
    ) -> ExportResult:
        """
        Export job spec for this backend.
        
        Args:
            run_spec: TGFlowRunSpec as dict
            output_dir: Directory to write output files
            **kwargs: Backend-specific options
            
        Returns:
            ExportResult with job spec and output path
        """
        pass
    
    def submit(self, job_spec: JobSpec, **kwargs) -> Dict[str, Any]:
        """
        Submit job to backend (if remote submit enabled).
        
        Args:
            job_spec: Job specification to submit
            **kwargs: Backend-specific options
            
        Returns:
            Submission result with job ID
        """
        if not self.remote_submit_enabled:
            return {
                "submitted": False,
                "reason": "Remote submit disabled. Set TG_ENABLE_REMOTE_SUBMIT=true to enable.",
            }
        return self._submit_impl(job_spec, **kwargs)
    
    def _submit_impl(self, job_spec: JobSpec, **kwargs) -> Dict[str, Any]:
        """Backend-specific submit implementation."""
        return {"submitted": False, "reason": "Not implemented for this backend"}
    
    def _build_base_spec(
        self,
        run_spec: Dict[str, Any],
        gpu_required: bool = True,
    ) -> JobSpec:
        """Build base job spec from run spec."""
        privacy = run_spec.get("privacy", {})
        privacy_mode = privacy.get("mode", "off")
        
        spec = JobSpec(
            backend=self.backend_id,
            image=self.image,
            env_vars={
                "TG_RUN_SPEC": "run_spec.json",
                "TG_PRIVACY_MODE": privacy_mode,
            },
            command=["python", "-m", "tensorguard.cli"],
            args=["peft", "run", "--spec", "run_spec.json"],
        )
        
        # GPU configuration
        if gpu_required:
            spec.gpu = "1"
            spec.gpu_type = "nvidia-tesla-t4"
        
        # N2HE sidecar configuration
        if privacy_mode == "n2he":
            sidecar_mode = privacy.get("n2he_sidecar", "disabled")
            if sidecar_mode == "enabled":
                spec.n2he_sidecar_enabled = True
                spec.n2he_sidecar_image = self.n2he_sidecar_image
                spec.env_vars["N2HE_SIDECAR_URL"] = "http://localhost:8765"
            else:
                # Use external N2HE service
                spec.n2he_service_endpoint = privacy.get("n2he_service_endpoint")
                if spec.n2he_service_endpoint:
                    spec.env_vars["N2HE_SERVICE_URL"] = spec.n2he_service_endpoint
        
        # Labels
        spec.labels = {
            "app": "tensorguardflow",
            "component": "peft-runner",
            "privacy-mode": privacy_mode,
        }
        
        return spec
    
    def _save_spec(
        self,
        job_spec: JobSpec,
        native_spec: Dict[str, Any],
        run_spec: Dict[str, Any],
        output_dir: str,
        native_filename: str,
    ) -> ExportResult:
        """Save job spec and native spec to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save run spec
        run_spec_path = os.path.join(output_dir, "run_spec.json")
        with open(run_spec_path, "w") as f:
            json.dump(run_spec, f, indent=2, default=str)
        job_spec.run_spec_path = run_spec_path
        
        # Save job spec
        job_spec_path = os.path.join(output_dir, "job_spec.json")
        with open(job_spec_path, "w") as f:
            json.dump(job_spec.to_dict(), f, indent=2)
        
        # Save native spec
        native_path = os.path.join(output_dir, native_filename)
        with open(native_path, "w") as f:
            if native_filename.endswith(".json"):
                json.dump(native_spec, f, indent=2)
            else:
                import yaml
                yaml.safe_dump(native_spec, f, default_flow_style=False)
        
        return ExportResult(
            success=True,
            job_spec=job_spec,
            output_path=output_dir,
            native_spec=native_spec,
        )
