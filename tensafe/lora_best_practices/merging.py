"""
LoRA Merging Utilities

Implements various methods for merging multiple LoRA adapters:
- Linear averaging (Model Soup)
- Task vectors
- TIES-Merging
- DARE

Based on research from:
- "LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks"
- "Model soups: averaging weights of multiple fine-tuned models"
- "TIES-Merging: Resolving Interference When Merging Models"
- "DARE: Language Models are Super Mario"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import copy
import math


class MergeMethod(Enum):
    """Available LoRA merging methods."""

    LINEAR = "linear"
    """Simple weighted average of adapter weights."""

    TASK_ARITHMETIC = "task_arithmetic"
    """Task vector arithmetic (add/subtract task vectors)."""

    TIES = "ties"
    """TIES-Merging: Trim, Elect Sign, Merge."""

    DARE = "dare"
    """DARE: Drop And REscale for reduced interference."""

    CAT = "cat"
    """Concatenation of LoRAs (from LoRA Soups paper)."""

    SLERP = "slerp"
    """Spherical linear interpolation."""


@dataclass
class MergeConfig:
    """Configuration for LoRA merging."""

    method: MergeMethod = MergeMethod.LINEAR
    weights: Optional[List[float]] = None  # Weights for each adapter
    normalize_weights: bool = True

    # TIES-specific parameters
    ties_density: float = 0.5  # Fraction of parameters to keep

    # DARE-specific parameters
    dare_drop_rate: float = 0.9  # Fraction to drop

    # Task arithmetic parameters
    scaling_coefficient: float = 1.0


@dataclass
class MergeResult:
    """Result of a merge operation."""

    merged_state_dict: Dict[str, Any]
    method: MergeMethod
    num_adapters_merged: int
    weights_used: List[float]
    metadata: Dict[str, Any]


class LoRAMerger:
    """
    Utility class for merging multiple LoRA adapters.

    Supports various merging strategies based on recent research
    on model merging and LoRA composition.
    """

    def __init__(self, base_model_name: Optional[str] = None):
        """
        Initialize the merger.

        Args:
            base_model_name: Name of the base model (for validation)
        """
        self.base_model_name = base_model_name

    def merge(
        self,
        adapters: List[Dict[str, Any]],
        config: Optional[MergeConfig] = None,
    ) -> MergeResult:
        """
        Merge multiple LoRA adapters.

        Args:
            adapters: List of adapter state dicts
            config: Merge configuration

        Returns:
            MergeResult with merged adapter
        """
        if config is None:
            config = MergeConfig()

        if len(adapters) == 0:
            raise ValueError("No adapters provided for merging")

        if len(adapters) == 1:
            return MergeResult(
                merged_state_dict=copy.deepcopy(adapters[0]),
                method=config.method,
                num_adapters_merged=1,
                weights_used=[1.0],
                metadata={"single_adapter": True},
            )

        # Set default weights if not provided
        weights = config.weights
        if weights is None:
            weights = [1.0 / len(adapters)] * len(adapters)
        elif config.normalize_weights:
            total = sum(weights)
            weights = [w / total for w in weights]

        # Dispatch to appropriate method
        if config.method == MergeMethod.LINEAR:
            merged = self._merge_linear(adapters, weights)
        elif config.method == MergeMethod.TASK_ARITHMETIC:
            merged = self._merge_task_arithmetic(adapters, weights, config.scaling_coefficient)
        elif config.method == MergeMethod.TIES:
            merged = self._merge_ties(adapters, weights, config.ties_density)
        elif config.method == MergeMethod.DARE:
            merged = self._merge_dare(adapters, weights, config.dare_drop_rate)
        elif config.method == MergeMethod.SLERP:
            if len(adapters) != 2:
                raise ValueError("SLERP requires exactly 2 adapters")
            merged = self._merge_slerp(adapters[0], adapters[1], weights[0])
        else:
            raise ValueError(f"Unknown merge method: {config.method}")

        return MergeResult(
            merged_state_dict=merged,
            method=config.method,
            num_adapters_merged=len(adapters),
            weights_used=weights,
            metadata={
                "config": {
                    "method": config.method.value,
                    "ties_density": config.ties_density,
                    "dare_drop_rate": config.dare_drop_rate,
                }
            },
        )

    def _merge_linear(
        self,
        adapters: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Linear weighted average merge."""
        merged = {}

        # Get all keys from first adapter
        keys = set(adapters[0].keys())

        for key in keys:
            # Check key exists in all adapters
            if not all(key in adapter for adapter in adapters):
                continue

            # Get tensors
            tensors = [adapter[key] for adapter in adapters]

            # Weighted average
            merged_tensor = None
            for tensor, weight in zip(tensors, weights):
                if merged_tensor is None:
                    merged_tensor = tensor * weight
                else:
                    merged_tensor = merged_tensor + tensor * weight

            merged[key] = merged_tensor

        return merged

    def _merge_task_arithmetic(
        self,
        adapters: List[Dict[str, Any]],
        weights: List[float],
        scaling: float,
    ) -> Dict[str, Any]:
        """
        Task arithmetic merge.

        Treats each adapter as a "task vector" and combines them.
        """
        merged = {}
        keys = set(adapters[0].keys())

        for key in keys:
            if not all(key in adapter for adapter in adapters):
                continue

            tensors = [adapter[key] for adapter in adapters]

            # Sum weighted task vectors
            merged_tensor = None
            for tensor, weight in zip(tensors, weights):
                scaled = tensor * weight * scaling
                if merged_tensor is None:
                    merged_tensor = scaled
                else:
                    merged_tensor = merged_tensor + scaled

            merged[key] = merged_tensor

        return merged

    def _merge_ties(
        self,
        adapters: List[Dict[str, Any]],
        weights: List[float],
        density: float,
    ) -> Dict[str, Any]:
        """
        TIES-Merging: Trim, Elect Sign, Merge.

        Steps:
        1. Trim: Keep only top-k% parameters by magnitude
        2. Elect: Resolve sign conflicts by majority vote
        3. Merge: Average the trimmed, sign-resolved parameters
        """
        merged = {}
        keys = set(adapters[0].keys())

        for key in keys:
            if not all(key in adapter for adapter in adapters):
                continue

            tensors = [adapter[key] for adapter in adapters]

            # Step 1: Trim - keep top density% by magnitude
            trimmed_tensors = []
            for tensor in tensors:
                flat = tensor.flatten() if hasattr(tensor, 'flatten') else tensor
                if hasattr(flat, 'abs'):
                    magnitudes = flat.abs()
                    k = int(len(magnitudes) * density)
                    if k > 0:
                        threshold = sorted(magnitudes, reverse=True)[min(k, len(magnitudes)-1)]
                        mask = magnitudes >= threshold
                        trimmed = tensor.clone() if hasattr(tensor, 'clone') else tensor.copy()
                        if hasattr(trimmed, 'flatten'):
                            trimmed_flat = trimmed.flatten()
                            trimmed_flat[~mask] = 0
                            trimmed = trimmed_flat.reshape(tensor.shape)
                        trimmed_tensors.append(trimmed)
                    else:
                        trimmed_tensors.append(tensor * 0)
                else:
                    trimmed_tensors.append(tensor)

            # Step 2: Elect sign by majority
            if len(trimmed_tensors) > 0 and hasattr(trimmed_tensors[0], 'sign'):
                signs = [t.sign() for t in trimmed_tensors]
                sign_sum = sum(s * w for s, w in zip(signs, weights))
                majority_sign = sign_sum.sign()

                # Step 3: Merge with sign alignment
                merged_tensor = None
                for tensor, weight in zip(trimmed_tensors, weights):
                    aligned = tensor.abs() * majority_sign * weight
                    if merged_tensor is None:
                        merged_tensor = aligned
                    else:
                        merged_tensor = merged_tensor + aligned

                merged[key] = merged_tensor
            else:
                # Fallback to linear merge
                merged_tensor = sum(t * w for t, w in zip(tensors, weights))
                merged[key] = merged_tensor

        return merged

    def _merge_dare(
        self,
        adapters: List[Dict[str, Any]],
        weights: List[float],
        drop_rate: float,
    ) -> Dict[str, Any]:
        """
        DARE: Drop And REscale.

        Randomly drops parameters and rescales remaining ones
        to reduce interference between adapters.
        """
        merged = {}
        keys = set(adapters[0].keys())
        rescale = 1.0 / (1.0 - drop_rate)

        for key in keys:
            if not all(key in adapter for adapter in adapters):
                continue

            tensors = [adapter[key] for adapter in adapters]

            # Apply DARE to each tensor
            dare_tensors = []
            for tensor in tensors:
                if hasattr(tensor, 'shape'):
                    # Create random mask
                    try:
                        import torch
                        mask = torch.rand_like(tensor.float()) > drop_rate
                        dare_tensor = tensor * mask.to(tensor.dtype) * rescale
                    except ImportError:
                        # Fallback: no dropping
                        dare_tensor = tensor
                else:
                    dare_tensor = tensor
                dare_tensors.append(dare_tensor)

            # Linear merge of DARE-processed tensors
            merged_tensor = sum(t * w for t, w in zip(dare_tensors, weights))
            merged[key] = merged_tensor

        return merged

    def _merge_slerp(
        self,
        adapter1: Dict[str, Any],
        adapter2: Dict[str, Any],
        t: float,
    ) -> Dict[str, Any]:
        """
        Spherical Linear Interpolation (SLERP).

        Interpolates between two adapters along a great circle.
        """
        merged = {}

        for key in adapter1.keys():
            if key not in adapter2:
                continue

            t1 = adapter1[key]
            t2 = adapter2[key]

            if hasattr(t1, 'flatten') and hasattr(t2, 'flatten'):
                # Flatten for computation
                v1 = t1.flatten().float()
                v2 = t2.flatten().float()

                # Compute angle
                try:
                    import torch
                    dot = torch.dot(v1, v2)
                    norm1 = torch.norm(v1)
                    norm2 = torch.norm(v2)

                    if norm1 > 0 and norm2 > 0:
                        cos_theta = (dot / (norm1 * norm2)).clamp(-1, 1)
                        theta = torch.acos(cos_theta)

                        if theta.abs() > 1e-6:
                            sin_theta = torch.sin(theta)
                            interp = (
                                torch.sin((1 - t) * theta) / sin_theta * v1 +
                                torch.sin(t * theta) / sin_theta * v2
                            )
                        else:
                            # Nearly parallel, use linear interpolation
                            interp = (1 - t) * v1 + t * v2

                        merged[key] = interp.reshape(t1.shape).to(t1.dtype)
                    else:
                        merged[key] = (1 - t) * t1 + t * t2
                except ImportError:
                    # Fallback to linear
                    merged[key] = (1 - t) * t1 + t * t2
            else:
                # Non-tensor, use linear
                merged[key] = (1 - t) * t1 + t * t2

        return merged


def merge_lora_adapters(
    adapter_paths: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    method: MergeMethod = MergeMethod.LINEAR,
    weights: Optional[List[float]] = None,
    **kwargs,
) -> MergeResult:
    """
    High-level function to merge LoRA adapters from files.

    Args:
        adapter_paths: Paths to adapter directories or files
        output_path: Path to save merged adapter (optional)
        method: Merging method to use
        weights: Weights for each adapter
        **kwargs: Additional arguments for MergeConfig

    Returns:
        MergeResult with merged adapter

    Example:
        >>> result = merge_lora_adapters(
        ...     ["adapter1/", "adapter2/", "adapter3/"],
        ...     method=MergeMethod.TIES,
        ...     weights=[0.5, 0.3, 0.2],
        ...     ties_density=0.7,
        ... )
        >>> # Save manually or use output_path
    """
    # Load adapters
    adapters = []
    for path in adapter_paths:
        path = Path(path)

        # Try different loading methods
        try:
            import torch

            if path.is_dir():
                # Look for adapter files
                for pattern in ["adapter_model.safetensors", "adapter_model.bin"]:
                    adapter_file = path / pattern
                    if adapter_file.exists():
                        if pattern.endswith(".safetensors"):
                            try:
                                from safetensors.torch import load_file
                                state_dict = load_file(str(adapter_file))
                            except ImportError:
                                state_dict = torch.load(adapter_file, map_location="cpu")
                        else:
                            state_dict = torch.load(adapter_file, map_location="cpu")
                        adapters.append(state_dict)
                        break
            elif path.exists():
                if str(path).endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(str(path))
                    except ImportError:
                        state_dict = torch.load(path, map_location="cpu")
                else:
                    state_dict = torch.load(path, map_location="cpu")
                adapters.append(state_dict)
        except ImportError:
            raise ImportError("PyTorch is required for loading adapters")

    if len(adapters) == 0:
        raise ValueError("No adapters could be loaded")

    # Create config
    config = MergeConfig(method=method, weights=weights, **kwargs)

    # Merge
    merger = LoRAMerger()
    result = merger.merge(adapters, config)

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from safetensors.torch import save_file
            save_file(result.merged_state_dict, output_path / "adapter_model.safetensors")
        except ImportError:
            import torch
            torch.save(result.merged_state_dict, output_path / "adapter_model.bin")

    return result


# Documentation
MERGE_METHODS_GUIDE = """
LoRA Merging Methods Guide
==========================

LINEAR (Simple Averaging)
-------------------------
- Weighted average of adapter weights
- Best for: Similar tasks, same base model
- Pros: Simple, fast, predictable
- Cons: Can dilute specialized knowledge

TASK_ARITHMETIC
---------------
- Treats adapters as "task vectors"
- Can add or subtract capabilities
- Best for: Combining complementary skills
- Pros: Flexible composition
- Cons: May require tuning scaling coefficient

TIES (Trim, Elect, Merge)
-------------------------
- Resolves parameter conflicts by:
  1. Keeping only top-k parameters
  2. Majority vote for sign
  3. Merging aligned parameters
- Best for: Merging conflicting adapters
- Pros: Handles interference well
- Cons: More complex, density tuning needed

DARE (Drop And REscale)
-----------------------
- Randomly drops parameters to reduce interference
- Rescales remaining parameters
- Best for: Many adapters with potential conflicts
- Pros: Simple noise injection reduces interference
- Cons: Stochastic, may need multiple attempts

CAT (Concatenation)
-------------------
- From "LoRA Soups" paper
- Concatenates adapters rather than averaging
- Best for: Skill composition tasks
- Pros: Preserves individual adapter knowledge
- Cons: Increases adapter size

SLERP (Spherical Interpolation)
-------------------------------
- Interpolates along great circle
- Only for 2 adapters
- Best for: Smooth interpolation between styles
- Pros: Maintains magnitude, smooth transition
- Cons: Only 2 adapters at a time

Recommendation
--------------
Start with LINEAR for simple cases.
Use TIES or DARE when LINEAR shows interference.
Use CAT when preserving distinct skills is critical.
"""
