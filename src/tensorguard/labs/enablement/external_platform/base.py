"""
External Platform Adapter Interface

Defines the contract for integrating with RobOps platforms (e.g. Foxglove, Formant).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class RunContext(BaseModel):
    """Context for a job run."""
    run_id: str
    robot_id: str
    site_id: Optional[str] = None
    job_type: str
    config: Dict[str, Any] = {}

class ExternalPlatformAdapter(ABC):
    """
    Abstract Base Class for Platform Adapters.
    """
    
    @abstractmethod
    def fetch_inputs(self, ctx: RunContext) -> str:
        """
        Download/Locate input data (bag/mcap) for the run.
        Returns local path to the file.
        """
        pass

    @abstractmethod
    def publish_artifact(self, ctx: RunContext, file_path: str, artifact_type: str):
        """
        Upload an artifact (report, model, log) to the platform.
        """
        pass

    @abstractmethod
    def record_metric(self, ctx: RunContext, name: str, value: float, unit: str = ""):
        """
        Report a scalar metric (e.g. privacy budget spent).
        """
        pass
        
    @abstractmethod
    def update_status(self, ctx: RunContext, status: str, message: str = ""):
        """
        Update run status (RUNNING, SUCCESS, FAILED).
        """
        pass
