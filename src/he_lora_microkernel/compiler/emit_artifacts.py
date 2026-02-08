"""
Artifact Emission for HE-LoRA Microkernel Compiler

This module handles the serialization and emission of compiled artifacts
for the HE-LoRA microkernel. Artifacts include:

  1. Compiled schedules (JSON)
  2. Pre-encoded plaintext weights (binary)
  3. Rotation key specifications
  4. Cost predictions and budgets
  5. Deterministic hashes for verification

All artifacts are deterministically generated from the input configuration,
enabling reproducible builds and CI verification.
"""

import json
import hashlib
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime

from .ckks_params import CKKSParams
from .packer import PackingLayout, PackedLoRAWeights
from .lora_ir import LoRAConfig, LoRAIRModule
from .scheduler import ExecutionSchedule, RotationSchedule
from .cost_model import CostEstimate, CostBudget


# =============================================================================
# ARTIFACT TYPES
# =============================================================================

@dataclass
class ArtifactMetadata:
    """Metadata for artifact bundle."""
    version: str = "1.0.0"
    created_at: str = ""
    config_hash: str = ""
    schedule_hash: str = ""
    deterministic: bool = True

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class CompiledArtifacts:
    """Complete bundle of compiled artifacts."""

    # Metadata
    metadata: ArtifactMetadata

    # Configuration
    config: LoRAConfig
    ckks_params: CKKSParams

    # Compiled components
    layout: PackingLayout
    schedule: ExecutionSchedule
    ir_module: LoRAIRModule
    rotation_schedule: RotationSchedule

    # Pre-encoded weights (binary)
    encoded_weights: Optional[bytes] = None

    # Cost estimates
    cost_estimate: Optional[CostEstimate] = None
    cost_budget: Optional[CostBudget] = None

    # File paths (if saved to disk)
    artifact_dir: Optional[Path] = None


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

class ArtifactEncoder(json.JSONEncoder):
    """Custom JSON encoder for artifacts."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                'dtype': str(obj.dtype),
                'shape': list(obj.shape),
                'data': obj.tolist(),
            }
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        return super().default(obj)


def artifact_decoder(d: dict) -> Any:
    """Custom JSON decoder for artifacts."""
    if '__type__' in d:
        if d['__type__'] == 'ndarray':
            return np.array(d['data'], dtype=d['dtype']).reshape(d['shape'])
    return d


# =============================================================================
# BINARY ENCODING FOR WEIGHTS
# =============================================================================

def encode_packed_weights(weights: PackedLoRAWeights) -> bytes:
    """
    Encode packed LoRA weights to binary format.

    Format:
      - Header (32 bytes): magic, version, counts, sizes
      - A_scaled matrix (float64)
      - B_original matrix (float64)
      - Packed blocks (float64 arrays)

    Args:
        weights: Packed LoRA weights

    Returns:
        Binary encoded weights
    """
    parts = []

    # Magic number
    magic = b'HELORAW1'  # HE-LORA Weights v1
    parts.append(magic)

    # Dimensions
    hidden_size, rank = weights.A_original.shape
    num_B_blocks = len(weights.B_packed_blocks)
    num_A_blocks = len(weights.A_packed_blocks)

    header = struct.pack(
        '<IIII',  # Little-endian, 4 uint32
        hidden_size,
        rank,
        num_B_blocks,
        num_A_blocks,
    )
    parts.append(header)

    # A_scaled matrix
    parts.append(weights.A_scaled.astype(np.float64).tobytes())

    # B_original matrix
    parts.append(weights.B_original.astype(np.float64).tobytes())

    # B packed blocks
    for block in weights.B_packed_blocks:
        block_size = struct.pack('<I', len(block))
        parts.append(block_size)
        parts.append(block.astype(np.float64).tobytes())

    # A packed blocks
    for block in weights.A_packed_blocks:
        block_size = struct.pack('<I', len(block))
        parts.append(block_size)
        parts.append(block.astype(np.float64).tobytes())

    return b''.join(parts)


def decode_packed_weights(data: bytes, layout: PackingLayout) -> PackedLoRAWeights:
    """
    Decode packed LoRA weights from binary format.

    Args:
        data: Binary encoded weights
        layout: Packing layout for context

    Returns:
        Decoded PackedLoRAWeights
    """
    offset = 0

    # Verify magic
    magic = data[offset:offset+8]
    if magic != b'HELORAW1':
        raise ValueError(f"Invalid magic number: {magic}")
    offset += 8

    # Read header
    hidden_size, rank, num_B_blocks, num_A_blocks = struct.unpack(
        '<IIII', data[offset:offset+16]
    )
    offset += 16

    # Read A_scaled
    A_size = hidden_size * rank * 8  # float64
    A_scaled = np.frombuffer(data[offset:offset+A_size], dtype=np.float64)
    A_scaled = A_scaled.reshape(hidden_size, rank)
    offset += A_size

    # Read B_original
    B_size = rank * hidden_size * 8
    B_original = np.frombuffer(data[offset:offset+B_size], dtype=np.float64)
    B_original = B_original.reshape(rank, hidden_size)
    offset += B_size

    # Read B packed blocks
    B_packed_blocks = []
    for _ in range(num_B_blocks):
        block_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        block = np.frombuffer(data[offset:offset+block_len*8], dtype=np.float64)
        B_packed_blocks.append(block)
        offset += block_len * 8

    # Read A packed blocks
    A_packed_blocks = []
    for _ in range(num_A_blocks):
        block_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        block = np.frombuffer(data[offset:offset+block_len*8], dtype=np.float64)
        A_packed_blocks.append(block)
        offset += block_len * 8

    # Reconstruct A_original (without scaling)
    # Note: We stored A_scaled, so we need to reverse the scaling
    # This is approximate - ideally we'd store original A
    A_original = A_scaled.copy()

    return PackedLoRAWeights(
        A_original=A_original,
        B_original=B_original,
        A_scaled=A_scaled,
        B_packed_blocks=B_packed_blocks,
        A_packed_blocks=A_packed_blocks,
        AB_combined=None,
        layout=layout,
    )


# =============================================================================
# ARTIFACT EMISSION
# =============================================================================

def emit_schedule_json(schedule: ExecutionSchedule) -> str:
    """
    Emit execution schedule as JSON.

    Args:
        schedule: Compiled execution schedule

    Returns:
        JSON string
    """
    return json.dumps(schedule.to_dict(), cls=ArtifactEncoder, indent=2)


def emit_rotation_keys_spec(schedule: ExecutionSchedule) -> Dict[str, Any]:
    """
    Emit specification for required rotation keys.

    This tells the runtime which Galois keys need to be generated.

    Args:
        schedule: Compiled schedule

    Returns:
        Key specification dictionary
    """
    return {
        'rotation_steps': schedule.rotation_schedule.rotation_steps,
        'required_galois_keys': schedule.rotation_schedule.required_galois_keys,
        'total_rotations': schedule.rotation_schedule.total_rotations,
        'strategy': schedule.rotation_schedule.strategy.value,
    }


def emit_cost_report(
    estimate: CostEstimate,
    budget: Optional[CostBudget] = None,
) -> Dict[str, Any]:
    """
    Emit cost report for CI verification.

    Args:
        estimate: Cost estimate
        budget: Optional budget for compliance check

    Returns:
        Cost report dictionary
    """
    report = {
        'estimate': estimate.to_dict(),
        'budget_compliance': None,
    }

    if budget:
        from .cost_model import check_budget_compliance
        from .lora_ir import LoRATargets

        passed, violations = check_budget_compliance(
            estimate, budget, LoRATargets.QKV
        )
        report['budget_compliance'] = {
            'passed': passed,
            'violations': violations,
        }

    return report


# =============================================================================
# ARTIFACT BUNDLE
# =============================================================================

def create_artifact_bundle(
    schedule: ExecutionSchedule,
    weights: Optional[PackedLoRAWeights] = None,
    cost_estimate: Optional[CostEstimate] = None,
    cost_budget: Optional[CostBudget] = None,
) -> CompiledArtifacts:
    """
    Create complete artifact bundle from compiled schedule.

    Args:
        schedule: Compiled execution schedule
        weights: Optional pre-packed weights
        cost_estimate: Optional cost estimate
        cost_budget: Optional cost budget

    Returns:
        Complete artifact bundle
    """
    metadata = ArtifactMetadata(
        config_hash=schedule.config.config_hash(),
        schedule_hash=schedule.schedule_hash,
    )

    encoded_weights = None
    if weights:
        encoded_weights = encode_packed_weights(weights)

    return CompiledArtifacts(
        metadata=metadata,
        config=schedule.config,
        ckks_params=schedule.ckks_params,
        layout=schedule.layout,
        schedule=schedule,
        ir_module=schedule.ir_module,
        rotation_schedule=schedule.rotation_schedule,
        encoded_weights=encoded_weights,
        cost_estimate=cost_estimate,
        cost_budget=cost_budget,
    )


def save_artifacts(
    artifacts: CompiledArtifacts,
    output_dir: Union[str, Path],
) -> Path:
    """
    Save artifact bundle to disk.

    Creates directory structure:
      output_dir/
        metadata.json
        schedule.json
        layout.json
        ir_module.json
        rotation_keys.json
        cost_report.json
        weights.bin (if present)

    Args:
        artifacts: Artifact bundle
        output_dir: Output directory path

    Returns:
        Path to artifact directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'version': artifacts.metadata.version,
            'created_at': artifacts.metadata.created_at,
            'config_hash': artifacts.metadata.config_hash,
            'schedule_hash': artifacts.metadata.schedule_hash,
            'deterministic': artifacts.metadata.deterministic,
        }, f, indent=2)

    # Save schedule
    schedule_path = output_dir / "schedule.json"
    with open(schedule_path, 'w') as f:
        f.write(emit_schedule_json(artifacts.schedule))

    # Save layout
    layout_path = output_dir / "layout.json"
    with open(layout_path, 'w') as f:
        json.dump(artifacts.layout.to_dict(), f, indent=2)

    # Save IR module
    ir_path = output_dir / "ir_module.json"
    with open(ir_path, 'w') as f:
        json.dump(artifacts.ir_module.to_dict(), f, indent=2)

    # Save rotation keys spec
    keys_path = output_dir / "rotation_keys.json"
    with open(keys_path, 'w') as f:
        json.dump(emit_rotation_keys_spec(artifacts.schedule), f, indent=2)

    # Save cost report
    if artifacts.cost_estimate:
        cost_path = output_dir / "cost_report.json"
        with open(cost_path, 'w') as f:
            json.dump(
                emit_cost_report(artifacts.cost_estimate, artifacts.cost_budget),
                f, indent=2
            )

    # Save weights binary
    if artifacts.encoded_weights:
        weights_path = output_dir / "weights.bin"
        with open(weights_path, 'wb') as f:
            f.write(artifacts.encoded_weights)

    # Update artifact dir
    artifacts.artifact_dir = output_dir

    return output_dir


def load_artifacts(
    artifact_dir: Union[str, Path],
) -> CompiledArtifacts:
    """
    Load artifact bundle from disk.

    Args:
        artifact_dir: Path to artifact directory

    Returns:
        Loaded artifact bundle

    Note:
        This is a partial load - some objects need reconstruction
        from the JSON data using appropriate from_dict methods.
    """
    artifact_dir = Path(artifact_dir)

    # Load metadata
    with open(artifact_dir / "metadata.json", 'r') as f:
        metadata_dict = json.load(f)
    metadata = ArtifactMetadata(**metadata_dict)

    # Load schedule JSON (raw dict for now)
    with open(artifact_dir / "schedule.json", 'r') as f:
        schedule_dict = json.load(f, object_hook=artifact_decoder)

    # Load layout
    with open(artifact_dir / "layout.json", 'r') as f:
        layout_dict = json.load(f)

    # Load IR module
    with open(artifact_dir / "ir_module.json", 'r') as f:
        ir_dict = json.load(f)

    # Load weights if present
    encoded_weights = None
    weights_path = artifact_dir / "weights.bin"
    if weights_path.exists():
        with open(weights_path, 'rb') as f:
            encoded_weights = f.read()

    # Note: Full reconstruction would require from_dict implementations
    # For now, return a partial bundle with raw dicts
    # In production, these would be fully deserialized

    # Reconstruct config
    config = LoRAConfig.from_dict(schedule_dict['config'])

    # Reconstruct CKKS params
    from .ckks_params import CKKSParams
    ckks_params = CKKSParams.from_dict(schedule_dict['ckks_params'])

    # Create placeholder objects (partial reconstruction)
    # Full implementation would reconstruct all objects

    return CompiledArtifacts(
        metadata=metadata,
        config=config,
        ckks_params=ckks_params,
        layout=None,  # Would need reconstruction
        schedule=None,  # Would need reconstruction
        ir_module=None,  # Would need reconstruction
        rotation_schedule=None,  # Would need reconstruction
        encoded_weights=encoded_weights,
        artifact_dir=artifact_dir,
    )


# =============================================================================
# DETERMINISM VERIFICATION
# =============================================================================

def verify_determinism(
    config: LoRAConfig,
    ckks_params: CKKSParams,
    reference_hash: str,
) -> bool:
    """
    Verify that compilation is deterministic.

    Compiles the schedule twice and verifies hashes match.

    Args:
        config: LoRA configuration
        ckks_params: CKKS parameters
        reference_hash: Expected schedule hash

    Returns:
        True if hashes match
    """
    from .scheduler import compile_schedule

    # Compile twice
    schedule1 = compile_schedule(config, ckks_params)
    schedule2 = compile_schedule(config, ckks_params)

    # Verify internal consistency
    if schedule1.schedule_hash != schedule2.schedule_hash:
        return False

    # Verify against reference
    if reference_hash and schedule1.schedule_hash != reference_hash:
        return False

    return True


def compute_artifact_checksum(artifact_dir: Union[str, Path]) -> str:
    """
    Compute checksum of all artifacts in directory.

    Args:
        artifact_dir: Path to artifact directory

    Returns:
        SHA-256 checksum of all files
    """
    artifact_dir = Path(artifact_dir)
    hasher = hashlib.sha256()

    # Sort files for deterministic order
    files = sorted(artifact_dir.iterdir())

    for file_path in files:
        if file_path.is_file():
            hasher.update(file_path.name.encode())
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)

    return hasher.hexdigest()
