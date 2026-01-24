"""
MOAI Model Exporter - Production Hardened

Converts training checkpoints into FHE-servable ModelPacks.

SECURITY NOTE: Uses safe serialization (msgpack) instead of pickle.

Production behavior:
- Loads real checkpoints from disk
- Validates target modules exist
- Never generates mock/random weights
"""

import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from ..core.adapters import FHEExportAdapter
from ..utils.logging import get_logger
from ..utils.serialization import safe_dumps
from ..utils.production_gates import is_production, ProductionGateError
from .modelpack import ModelPack, ModelPackMetadata
from .moai_config import MoaiConfig

logger = get_logger(__name__)


class MoaiExporter:
    """
    Handles the pipeline of:
    1. Loading a trained model checkpoint
    2. Extracting specific submodules (FHEExportAdapter)
    3. Packing/Quantizing per MOAI config
    4. Serializing to ModelPack

    Production behavior:
    - Requires real checkpoint file
    - Validates target modules exist in checkpoint
    - Never generates random weights
    """

    def __init__(self, config: MoaiConfig):
        self.config = config
        self._torch_available = False
        self._safetensors_available = False

        try:
            import torch
            self._torch = torch
            self._torch_available = True
        except ImportError:
            if is_production():
                raise ProductionGateError(
                    gate_name="MOAI_TORCH",
                    message="PyTorch is required for MOAI export in production.",
                    remediation="Install PyTorch: pip install torch",
                )
            logger.warning("PyTorch not found. MOAI export limited in development mode.")

        try:
            import safetensors.torch as safetensors
            self._safetensors = safetensors
            self._safetensors_available = True
        except ImportError:
            logger.info("safetensors not found. Will use torch.load for checkpoints.")

    def _load_checkpoint(self, model_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from file.

        Supports:
        - .safetensors files (preferred, safe)
        - .pt / .pth / .bin files (PyTorch format)

        Args:
            model_path: Path to checkpoint file

        Returns:
            State dict with model weights

        Raises:
            FileNotFoundError: If checkpoint not found
            ProductionGateError: If unable to load in production
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        logger.info(f"Loading checkpoint: {model_path}")

        # Determine format from extension
        ext = os.path.splitext(model_path)[1].lower()

        if ext == ".safetensors":
            if not self._safetensors_available:
                raise ProductionGateError(
                    gate_name="SAFETENSORS",
                    message="safetensors library required for .safetensors files.",
                    remediation="Install safetensors: pip install safetensors",
                )
            state_dict = self._safetensors.load_file(model_path)
            logger.info(f"Loaded safetensors checkpoint with {len(state_dict)} tensors")
            return state_dict

        elif ext in (".pt", ".pth", ".bin"):
            if not self._torch_available:
                raise ProductionGateError(
                    gate_name="TORCH_LOAD",
                    message="PyTorch required for .pt/.pth/.bin checkpoint files.",
                    remediation="Install PyTorch: pip install torch",
                )
            # Load with weights_only=True for security (prevents pickle attacks)
            state_dict = self._torch.load(model_path, map_location="cpu", weights_only=True)

            # Handle nested state dicts (e.g., from Trainer.save_model)
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

            # Convert tensors to numpy for MOAI processing
            numpy_dict = {}
            for k, v in state_dict.items():
                if hasattr(v, "numpy"):
                    numpy_dict[k] = v.cpu().numpy()
                elif isinstance(v, np.ndarray):
                    numpy_dict[k] = v
                else:
                    logger.warning(f"Skipping non-tensor value: {k} ({type(v)})")

            logger.info(f"Loaded PyTorch checkpoint with {len(numpy_dict)} tensors")
            return numpy_dict

        else:
            raise ValueError(
                f"Unsupported checkpoint format: {ext}. "
                f"Supported: .safetensors, .pt, .pth, .bin"
            )

    def _validate_target_modules(
        self, state_dict: Dict[str, Any], target_modules: List[str]
    ) -> None:
        """
        Validate that target modules exist in the checkpoint.

        Args:
            state_dict: Loaded state dict
            target_modules: List of module names to extract

        Raises:
            ValueError: If target modules not found
        """
        found_modules = set()
        missing_modules = []

        for target in target_modules:
            # Check for exact match or prefix match
            found = False
            for key in state_dict.keys():
                if key.startswith(f"{target}.") or key == target:
                    found = True
                    found_modules.add(target)
                    break

            if not found:
                missing_modules.append(target)

        if missing_modules:
            available = sorted(set(k.split(".")[0] for k in state_dict.keys()))
            raise ValueError(
                f"Target modules not found in checkpoint: {missing_modules}. "
                f"Available top-level modules: {available[:20]}..."
            )

        logger.info(f"Validated target modules: {list(found_modules)}")

    def export(
        self,
        model_path: str,
        model_id: str,
        target_modules: List[str],
        git_commit: str = "unknown",
        tokenizer_config: Optional[Dict[str, Any]] = None,
    ) -> ModelPack:
        """
        Export a ModelPack from a checkpoint.

        Args:
            model_path: Path to model checkpoint file
            model_id: Unique identifier for the model
            target_modules: List of module names to extract
            git_commit: Git commit hash for provenance
            tokenizer_config: Optional tokenizer configuration

        Returns:
            ModelPack ready for FHE serving

        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If target modules not found
            ProductionGateError: If dependencies unavailable in production
        """
        logger.info(f"Starting MOAI export for {model_id} targeting {target_modules}")

        # 1. Load real checkpoint
        state_dict = self._load_checkpoint(model_path)

        # 2. Validate target modules exist
        self._validate_target_modules(state_dict, target_modules)

        # 3. Use Adapter to extract target weights
        adapter = FHEExportAdapter(model_path, target_modules)
        extracted_weights = adapter.extract_submodules(state_dict)

        if not extracted_weights:
            raise ValueError(
                f"No weights extracted for target modules: {target_modules}"
            )

        logger.info(f"Extracted {len(extracted_weights)} tensors from checkpoint")

        # 4. Quantization & Packing
        # SECURITY: Uses safe serialization instead of pickle
        packed_weights = {}
        weight_hashes = {}

        for k, v in extracted_weights.items():
            if isinstance(v, np.ndarray):
                packed_weights[k] = safe_dumps(v)
                # Compute hash for integrity verification
                weight_hashes[k] = hashlib.sha256(v.tobytes()).hexdigest()[:16]
            else:
                logger.warning(f"Skipping non-array weight: {k}")

        # 5. Create Metadata with provenance
        checkpoint_hash = hashlib.sha256(
            open(model_path, "rb").read()
        ).hexdigest()[:32]

        base_config = self.config.__dict__.copy() if hasattr(self.config, "__dict__") else {}
        meta = ModelPackMetadata(
            model_id=model_id,
            version="1.0.0",
            base_model=os.path.basename(model_path),
            target_modules=target_modules,
            created_at=datetime.utcnow().isoformat(),
            git_commit_hash=git_commit,
            config={
                **base_config,
                "source_checkpoint": model_path,
                "checkpoint_hash": checkpoint_hash,
                "weight_hashes": weight_hashes,
            },
        )

        # 6. Assemble Package
        pack = ModelPack(
            meta=meta,
            weights=packed_weights,
            tokenizer_config=tokenizer_config or {},
        )

        logger.info(
            f"Exported ModelPack: {model_id} with {len(packed_weights)} tensors "
            f"(checkpoint hash: {checkpoint_hash})"
        )
        return pack


def export_moai_modelpack(
    model_path: str,
    output_path: str,
    target_modules: Optional[List[str]] = None,
    model_id: Optional[str] = None,
) -> str:
    """
    CLI Entrypoint for MOAI export.

    Args:
        model_path: Path to input checkpoint
        output_path: Path for output ModelPack
        target_modules: Modules to extract (default: ["policy_head"])
        model_id: Model identifier (default: derived from filename)

    Returns:
        Path to saved ModelPack

    Raises:
        FileNotFoundError: If checkpoint not found
        ProductionGateError: In production without dependencies
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    config = MoaiConfig()
    exporter = MoaiExporter(config)

    targets = target_modules or ["policy_head"]
    mid = model_id or os.path.splitext(os.path.basename(model_path))[0]

    pack = exporter.export(model_path, mid, targets)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pack.save(output_path)
    logger.info(f"Saved ModelPack to {output_path}")

    return output_path
