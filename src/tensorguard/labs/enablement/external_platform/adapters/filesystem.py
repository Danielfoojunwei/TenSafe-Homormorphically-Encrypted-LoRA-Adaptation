"""
Filesystem Adapter

Implementation of ExternalPlatformAdapter for local/offline execution.
Writes outputs to a local directory structure.
"""

import os
import shutil
import logging
import json
from pathlib import Path
from typing import Dict, Any

from .base import ExternalPlatformAdapter, RunContext

logger = logging.getLogger(__name__)

class FilesystemAdapter(ExternalPlatformAdapter):
    def __init__(self, base_dir: str = "./runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_run_dir(self, run_id: str) -> Path:
        p = self.base_dir / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def fetch_inputs(self, ctx: RunContext) -> str:
        """
        Expects input path in config ['input_path'].
        """
        path = ctx.config.get("input_path")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Input path not found: {path}")
        return path

    def publish_artifact(self, ctx: RunContext, file_path: str, artifact_type: str):
        run_dir = self._get_run_dir(ctx.run_id)
        dest = run_dir / Path(file_path).name
        shutil.copy2(file_path, dest)
        logger.info(f"[FS] Published artifact: {dest} ({artifact_type})")

    def record_metric(self, ctx: RunContext, name: str, value: float, unit: str = ""):
        run_dir = self._get_run_dir(ctx.run_id)
        metrics_file = run_dir / "metrics.json"
        
        data = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
        
        data[name] = {"value": value, "unit": unit}
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def update_status(self, ctx: RunContext, status: str, message: str = ""):
        run_dir = self._get_run_dir(ctx.run_id)
        status_file = run_dir / "status.txt"
        with open(status_file, 'w') as f:
            f.write(f"{status}: {message}\n")
        logger.info(f"[FS] Status Update: {status} - {message}")
