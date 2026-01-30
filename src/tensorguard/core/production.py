"""
TensorGuard Production Optimization Components

This module implements the production-grade features for TensorGuard:
- Operating envelope enforcement
- Canonical UpdatePackage format
- Separate privacy/training controls
- Enterprise key management
- Evaluation gating
- IL/RL pipeline separation
- SRE observability

Based on the TensorGuard Production Blueprint.
Updated with Workflow Architect Hardening (v2.1).
"""

from enum import Enum, auto

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def numpy_json_serializer(obj):
    """Helper for JSON serialization of numpy types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, datetime)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


# ============================================================================
# 1. Production Operating Envelope
# ============================================================================

class PEFTStrategy(Enum):
    """Supported PEFT strategies"""
    LORA = "lora"
    ADAPTER = "adapter"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"


@dataclass
class OperatingEnvelope:
    """
    Production operating envelope that enforces explicit, hard constraints.

    This is the #1 optimization: lock down the scope to prevent performance
    and reliability drift.
    """

    # Trainable parameters constraint
    peft_strategy: PEFTStrategy = PEFTStrategy.LORA
    trainable_modules: List[str] = field(default_factory=lambda: ["policy_head", "last_4_blocks"])
    max_trainable_params: int = 10_000_000  # 10M parameters max

    # Round cadence
    round_interval_seconds: int = 3600  # Default: hourly
    min_round_interval_seconds: int = 600  # Minimum: 10 minutes
    max_round_interval_seconds: int = 86400  # Maximum: daily

    # Update size constraints
    target_update_size_kb: int = 500  # Target: 500KB
    min_update_size_kb: int = 10  # Minimum: 10KB
    max_update_size_kb: int = 5120  # Maximum: 5MB

    # Server capabilities
    server_operations: List[str] = field(default_factory=lambda: ["ciphertext_sum", "ciphertext_average"])
    allow_plaintext_inspection: bool = False  # Production default: False

    # Deployment controls
    enable_canary: bool = True
    enable_rollback: bool = True
    canary_percentage: float = 0.1  # 10% canary rollout

    def validate(self) -> bool:
        """Validate envelope constraints"""
        errors = []

        if self.round_interval_seconds < self.min_round_interval_seconds:
            errors.append(f"Round interval {self.round_interval_seconds}s < minimum {self.min_round_interval_seconds}s")

        if self.round_interval_seconds > self.max_round_interval_seconds:
            errors.append(f"Round interval {self.round_interval_seconds}s > maximum {self.max_round_interval_seconds}s")

        if self.target_update_size_kb < self.min_update_size_kb:
            errors.append(f"Target update size {self.target_update_size_kb}KB < minimum {self.min_update_size_kb}KB")

        if self.target_update_size_kb > self.max_update_size_kb:
            errors.append(f"Target update size {self.target_update_size_kb}KB > maximum {self.max_update_size_kb}KB")

        if self.canary_percentage < 0 or self.canary_percentage > 1:
            errors.append(f"Canary percentage {self.canary_percentage} must be in [0, 1]")

        if errors:
            logger.error(f"Operating envelope validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
            return False

        logger.info("Operating envelope validated successfully")
        return True

    def enforce_update_size(self, update_size_bytes: int) -> bool:
        """Check if update size is within constraints"""
        size_kb = update_size_bytes / 1024

        if size_kb < self.min_update_size_kb:
            logger.warning(f"Update size {size_kb:.2f}KB below minimum {self.min_update_size_kb}KB")
            return False

        if size_kb > self.max_update_size_kb:
            logger.error(f"Update size {size_kb:.2f}KB exceeds maximum {self.max_update_size_kb}KB")
            return False

        return True


# ============================================================================
# 2. Canonical UpdatePackage Format
# ============================================================================

class ObjectiveType(Enum):
    """Training objective types"""
    IMITATION_LEARNING = "il"
    OFFLINE_RL = "offline_rl"
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"


@dataclass
class ModelTargetMap:
    """Identifies which modules/tensors are updated"""
    module_names: List[str]
    adapter_ids: List[str]
    tensor_shapes: Dict[str, Tuple[int, ...]]

    def fingerprint(self) -> str:
        """Generate fingerprint for this target map"""
        data = json.dumps({
            "modules": sorted(self.module_names),
            "adapters": sorted(self.adapter_ids),
            "shapes": {k: list(v) for k, v in sorted(self.tensor_shapes.items())}
        }, sort_keys=True, default=numpy_json_serializer)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class SafetyStatistics:
    """Safety and quality metrics from training"""
    constraint_violations: int = 0
    ood_score: float = 0.0
    kl_divergence: float = 0.0
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0
    dp_epsilon_consumed: float = 0.0
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None


@dataclass
class TrainingMetadata:
    """Training configuration and statistics"""
    steps: int
    learning_rate: float
    objective_type: ObjectiveType
    reward_version_id: Optional[str] = None
    dataset_hash: Optional[str] = None
    num_demonstrations: int = 0
    training_duration_seconds: float = 0.0


@dataclass
class UpdatePackage:
    """
    Canonical update package format with strict versioning.

    This enables:
    - Deterministic application of deltas
    - Forward/backward compatibility
    - Reproducible audits
    - Safe rollback
    """

    # Version and identity
    schema_version: str = "1.0.0"
    package_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])
    client_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Model target
    target_map: ModelTargetMap = None
    base_model_fingerprint: str = ""
    adapter_schema_version: str = "1.0.0"

    # Delta tensors (compressed representation)
    delta_tensors: Dict[str, bytes] = field(default_factory=dict)  # Serialized compressed deltas
    compression_metadata: Dict[str, Any] = field(default_factory=dict)

    # Training metadata
    training_meta: TrainingMetadata = None

    # Safety statistics
    safety_stats: SafetyStatistics = field(default_factory=SafetyStatistics)
    
    # FedMoE Metadata (v2.0)
    expert_weights: Dict[str, float] = field(default_factory=dict)

    # Compatibility
    tensorguard_version: str = "2.0.0-fedmoe"

    def serialize(self) -> bytes:
        """Serialize to bytes for transmission"""
        package_dict = {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "client_id": self.client_id,
            "timestamp": self.timestamp,
            "target_map": {
                "module_names": self.target_map.module_names,
                "adapter_ids": self.target_map.adapter_ids,
                "tensor_shapes": {k: list(v) for k, v in self.target_map.tensor_shapes.items()}
            } if self.target_map else None,
            "base_model_fingerprint": self.base_model_fingerprint,
            "adapter_schema_version": self.adapter_schema_version,
            "compression_metadata": self.compression_metadata,
            "training_meta": asdict(self.training_meta) if self.training_meta else None,
            "safety_stats": asdict(self.safety_stats),
            "expert_weights": self.expert_weights,
            "tensorguard_version": self.tensorguard_version
        }

        # Serialize metadata as JSON
        metadata_json = json.dumps(package_dict, sort_keys=True, default=numpy_json_serializer).encode()
        metadata_size = len(metadata_json).to_bytes(4, 'big')

        # Serialize delta tensors
        delta_data = []
        for name, tensor_bytes in self.delta_tensors.items():
            name_bytes = name.encode()
            name_size = len(name_bytes).to_bytes(4, 'big')
            tensor_size = len(tensor_bytes).to_bytes(4, 'big')
            delta_data.append(name_size + name_bytes + tensor_size + tensor_bytes)

        delta_bytes = b''.join(delta_data)

        # Combine: [metadata_size][metadata][delta_tensors]
        return metadata_size + metadata_json + delta_bytes

    @classmethod
    def deserialize(cls, data: bytes) -> 'UpdatePackage':
        """Deserialize from bytes with robust bounds checking"""
        if len(data) < 4:
            raise ValueError("Payload too small for metadata size")
            
        # Parse metadata size
        metadata_size = int.from_bytes(data[:4], 'big')
        if metadata_size > len(data) - 4:
            raise ValueError(f"Invalid metadata size: {metadata_size}")
            
        metadata_json = data[4:4+metadata_size]
        try:
            package_dict = json.loads(metadata_json.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse metadata JSON: {e}")

        # Parse delta tensors
        delta_tensors = {}
        offset = 4 + metadata_size
        while offset < len(data):
            if len(data) - offset < 4:
                break # Trailing bytes
                
            name_size = int.from_bytes(data[offset:offset+4], 'big')
            offset += 4
            
            if len(data) - offset < name_size:
                raise ValueError(f"Corrupt payload: name_size {name_size} exceeds remaining data")
                
            name = data[offset:offset+name_size].decode()
            offset += name_size
            
            if len(data) - offset < 4:
                raise ValueError(f"Corrupt payload: expected tensor_size after name '{name}'")
                
            tensor_size = int.from_bytes(data[offset:offset+4], 'big')
            offset += 4
            
            if len(data) - offset < tensor_size:
                raise ValueError(f"Corrupt payload: tensor_size {tensor_size} for '{name}' exceeds remaining data")
                
            tensor_bytes = data[offset:offset+tensor_size]
            offset += tensor_size
            delta_tensors[name] = tensor_bytes

        # Reconstruct UpdatePackage
        package = cls(
            schema_version=package_dict["schema_version"],
            package_id=package_dict["package_id"],
            client_id=package_dict["client_id"],
            timestamp=package_dict["timestamp"],
            base_model_fingerprint=package_dict["base_model_fingerprint"],
            adapter_schema_version=package_dict["adapter_schema_version"],
            compression_metadata=package_dict["compression_metadata"],
            tensorguard_version=package_dict.get("tensorguard_version", "1.0.0"),
            expert_weights=package_dict.get("expert_weights", {}),
            delta_tensors=delta_tensors
        )

        # Reconstruct target map
        if package_dict["target_map"]:
            package.target_map = ModelTargetMap(
                module_names=package_dict["target_map"]["module_names"],
                adapter_ids=package_dict["target_map"]["adapter_ids"],
                tensor_shapes={k: tuple(v) for k, v in package_dict["target_map"]["tensor_shapes"].items()}
            )

        # Reconstruct training metadata
        if package_dict["training_meta"]:
            tm = package_dict["training_meta"]
            package.training_meta = TrainingMetadata(
                steps=tm["steps"],
                learning_rate=tm["learning_rate"],
                objective_type=ObjectiveType(tm["objective_type"]),
                reward_version_id=tm.get("reward_version_id"),
                dataset_hash=tm.get("dataset_hash"),
                num_demonstrations=tm.get("num_demonstrations", 0),
                training_duration_seconds=tm.get("training_duration_seconds", 0.0)
            )

        # Reconstruct safety stats
        ss = package_dict["safety_stats"]
        package.safety_stats = SafetyStatistics(**ss)

        return package

    def fingerprint(self) -> str:
        """Generate unique fingerprint for this update"""
        fp_data = {
            "package_id": self.package_id,
            "client_id": self.client_id,
            "timestamp": self.timestamp,
            "target_map_fp": self.target_map.fingerprint() if self.target_map else "",
            "base_model_fp": self.base_model_fingerprint
        }
        data_str = json.dumps(fp_data, sort_keys=True, default=numpy_json_serializer)
        return hashlib.sha256(data_str.encode()).hexdigest()


# ============================================================================
# 3. Separate Privacy and Training Controls
# ============================================================================

@dataclass
class DPPolicyProfile:
    """Differential Privacy policy profile (per customer/site)"""

    profile_name: str
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    epsilon_budget: float = 10.0
    delta: float = 1e-5
    epsilon_consumed: float = 0.0
    hard_stop_enabled: bool = True

    # Accounting
    accounting_method: str = "rdp"  # Renyi DP or GDP

    def consume_epsilon(self, amount: float) -> bool:
        """
        Consume epsilon budget. Returns False if budget exhausted.
        """
        if self.hard_stop_enabled and self.epsilon_consumed + amount > self.epsilon_budget:
            logger.error(f"DP budget exhausted: {self.epsilon_consumed:.3f} + {amount:.3f} > {self.epsilon_budget:.3f}")
            return False

        self.epsilon_consumed += amount
        remaining = self.epsilon_budget - self.epsilon_consumed
        logger.info(f"DP epsilon consumed: {amount:.3f}, remaining: {remaining:.3f}/{self.epsilon_budget:.3f}")
        return True

    def reset_budget(self):
        """Reset epsilon budget (e.g., new privacy period)"""
        logger.info(f"Resetting DP budget from {self.epsilon_consumed:.3f} to 0.0")
        self.epsilon_consumed = 0.0


@dataclass
class EncryptionPolicyProfile:
    """Encryption policy profile"""

    profile_name: str
    security_level: int = 128  # 128 or 192 bits
    key_rotation_schedule_hours: int = 24
    last_key_rotation: Optional[datetime] = None
    ciphertext_parameter_set: str = "N2HE_128"
    aggregation_quorum_threshold: int = 2  # Minimum clients for aggregation

    def needs_key_rotation(self) -> bool:
        """Check if key rotation is due"""
        if self.last_key_rotation is None:
            return True

        elapsed = datetime.utcnow() - self.last_key_rotation
        due = elapsed > timedelta(hours=self.key_rotation_schedule_hours)

        if due:
            logger.warning(f"Key rotation overdue: {elapsed.total_seconds()/3600:.1f}h > {self.key_rotation_schedule_hours}h")

        return due


@dataclass
class TrainingPolicyProfile:
    """Training policy profile (separate from privacy)"""

    profile_name: str
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    warmup_steps: int = 100

    # Compression
    compression_ratio: int = 32
    sparsity: float = 0.01

    # Quality thresholds
    max_quality_mse: float = 0.05
    min_grad_norm: float = 1e-6
    max_grad_norm: float = 10.0


# ============================================================================
# 4. Enterprise Key Management
# ============================================================================

@dataclass
class KeyMetadata:
    """Metadata for encryption key"""
    key_id: str
    created_at: datetime
    rotated_at: Optional[datetime] = None
    version: int = 1
    purpose: str = "encryption"
    owner: str = "customer"
    key_type: str = "N2HE"
    security_level: int = 128


class KeyManagementSystem:
    """
    Enterprise-ready key management.

    Production requirements:
    - Cloud never holds decryption keys
    - Customer-controlled decryption
    - Automatic key rotation
    - Disaster recovery
    - Break-glass policies
    - Full audit logs
    """

    def __init__(self, audit_log_path: Optional[Path] = None):
        self.keys: Dict[str, KeyMetadata] = {}
        self.audit_log_path = audit_log_path or Path("key_audit.log")
        self._ensure_audit_log()

    def _ensure_audit_log(self):
        """Create audit log if it doesn't exist"""
        if not self.audit_log_path.exists():
            self.audit_log_path.touch(mode=0o600)  # Restricted permissions

    def _audit(self, event: str, key_id: str, details: Dict[str, Any]):
        """Write to audit log"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event": event,
            "key_id": key_id,
            "details": details
        }

        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(log_entry, default=numpy_json_serializer) + "\n")

        logger.info(f"Key audit: {event} for key {key_id}")

    def register_key(self, key_id: str, metadata: KeyMetadata):
        """Register a new encryption key"""
        self.keys[key_id] = metadata
        self._audit("KEY_REGISTERED", key_id, asdict(metadata))

    def rotate_key(self, old_key_id: str, new_key_id: str, new_metadata: KeyMetadata) -> bool:
        """
        Rotate encryption key.

        Returns:
            True if rotation successful
        """
        if old_key_id not in self.keys:
            logger.error(f"Cannot rotate unknown key: {old_key_id}")
            return False

        old_meta = self.keys[old_key_id]
        old_meta.rotated_at = datetime.utcnow()

        self.keys[new_key_id] = new_metadata

        self._audit("KEY_ROTATED", new_key_id, {
            "old_key_id": old_key_id,
            "new_key_metadata": asdict(new_metadata)
        })

        logger.info(f"Key rotated: {old_key_id} -> {new_key_id}")
        return True

    def revoke_key(self, key_id: str, reason: str):
        """Revoke a key (break-glass scenario)"""
        if key_id not in self.keys:
            logger.error(f"Cannot revoke unknown key: {key_id}")
            return

        self._audit("KEY_REVOKED", key_id, {"reason": reason})
        del self.keys[key_id]
        logger.warning(f"Key revoked: {key_id}, reason: {reason}")

    def get_key_for_round(self, round_number: int) -> Optional[str]:
        """Map round number to key ID for auditing"""
        # In production, this would query a secure mapping table
        # For now, return the most recent active key
        active_keys = [k for k, v in self.keys.items() if v.rotated_at is None]
        return active_keys[0] if active_keys else None

    def disaster_recovery_export(self, export_path: Path, authorized_by: str) -> bool:
        """
        Export key metadata for disaster recovery.

        NOTE: This does NOT export actual key material, only metadata.
        Key material must be recovered from customer KMS/HSM.
        """
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "authorized_by": authorized_by,
            "keys": {k: asdict(v) for k, v in self.keys.items()}
        }

        try:
            export_path.write_text(json.dumps(export_data, indent=2, default=str))
            self._audit("DISASTER_RECOVERY_EXPORT", "ALL_KEYS", {
                "export_path": str(export_path),
                "authorized_by": authorized_by
            })
            logger.info(f"Disaster recovery export completed: {export_path}")
            return True
        except Exception as e:
            logger.error(f"Disaster recovery export failed: {e}")
            return False


# ============================================================================
# 5. Production-Grade Evaluation Gating
# ============================================================================

@dataclass
class EvaluationMetrics:
    """Metrics from evaluation gate"""
    success_rate: float = 0.0
    constraint_violations: int = 0
    time_to_complete_mean: float = 0.0
    collision_proxy_score: float = 0.0
    kl_divergence_vs_baseline: float = 0.0
    ood_robustness_score: float = 0.0

    # Regression checks
    regression_detected: bool = False
    regression_metric: Optional[str] = None
    regression_delta: Optional[float] = None


@dataclass
class SafetyThresholds:
    """Safety thresholds for evaluation gating"""
    min_success_rate: float = 0.8
    max_constraint_violations: int = 5
    max_kl_divergence: float = 0.5
    max_regression_delta: float = 0.05
    min_ood_robustness: float = 0.6


class EvaluationGate:
    """
    Production-grade evaluation gating.

    Non-negotiable for RL deployments. Includes:
    - Offline evaluation tasks
    - Policy drift checks
    - Robustness validation
    - Regression detection
    """

    def __init__(self, thresholds: SafetyThresholds):
        self.thresholds = thresholds
        self.baseline_metrics: Optional[EvaluationMetrics] = None

    def set_baseline(self, metrics: EvaluationMetrics):
        """Set baseline metrics for regression detection"""
        self.baseline_metrics = metrics
        logger.info(f"Baseline set: success_rate={metrics.success_rate:.3f}")

    def evaluate(self, metrics: EvaluationMetrics) -> Tuple[bool, List[str]]:
        """
        Evaluate whether update passes safety gate.

        Returns:
            (passed, reasons) - True if passed, list of failure reasons
        """
        failures = []

        # Check success rate
        if metrics.success_rate < self.thresholds.min_success_rate:
            failures.append(f"Success rate {metrics.success_rate:.3f} < threshold {self.thresholds.min_success_rate:.3f}")

        # Check constraint violations
        if metrics.constraint_violations > self.thresholds.max_constraint_violations:
            failures.append(f"Constraint violations {metrics.constraint_violations} > threshold {self.thresholds.max_constraint_violations}")

        # Check KL divergence
        if metrics.kl_divergence_vs_baseline > self.thresholds.max_kl_divergence:
            failures.append(f"KL divergence {metrics.kl_divergence_vs_baseline:.3f} > threshold {self.thresholds.max_kl_divergence:.3f}")

        # Check OOD robustness
        if metrics.ood_robustness_score < self.thresholds.min_ood_robustness:
            failures.append(f"OOD robustness {metrics.ood_robustness_score:.3f} < threshold {self.thresholds.min_ood_robustness:.3f}")

        # Check regression
        if self.baseline_metrics and metrics.regression_detected:
            failures.append(f"Regression detected in {metrics.regression_metric}: delta={metrics.regression_delta:.3f}")

        passed = len(failures) == 0

        if passed:
            logger.info("✓ Evaluation gate PASSED")
        else:
            logger.error(f"✗ Evaluation gate FAILED:\n" + "\n".join(f"  - {f}" for f in failures))

        return passed, failures


# ============================================================================
# 6. Clean IL + RL Pipeline Separation
# ============================================================================

class TrainingStage(Enum):
    """Training pipeline stages"""
    IL_PEFT_BASELINE = "il_peft_baseline"
    OFFLINE_RL_PEFT = "offline_rl_peft"
    ONPOLICY_RL_PEFT = "onpolicy_rl_peft"


@dataclass
class StageConfig:
    """Configuration for a training stage"""
    stage: TrainingStage
    enabled: bool = True
    requires_approval: bool = False  # Require explicit approval before running
    max_rounds: int = 10
    evaluation_gate: Optional[EvaluationGate] = None


class TrainingPipeline:
    """
    Clean separation of IL + RL stages.

    Production-friendly sequence:
    1. IL PEFT baseline (stable supervised adaptation)
    2. Offline RL PEFT (improvement from logs, conservative)
    3. Optional on-policy RL PEFT (only where exploration acceptable)
    """

    def __init__(self):
        self.stages: Dict[TrainingStage, StageConfig] = {
            TrainingStage.IL_PEFT_BASELINE: StageConfig(
                stage=TrainingStage.IL_PEFT_BASELINE,
                enabled=True,
                requires_approval=False,
                max_rounds=5
            ),
            TrainingStage.OFFLINE_RL_PEFT: StageConfig(
                stage=TrainingStage.OFFLINE_RL_PEFT,
                enabled=True,
                requires_approval=False,
                max_rounds=10
            ),
            TrainingStage.ONPOLICY_RL_PEFT: StageConfig(
                stage=TrainingStage.ONPOLICY_RL_PEFT,
                enabled=False,  # Disabled by default
                requires_approval=True,
                max_rounds=20
            )
        }

        self.current_stage: Optional[TrainingStage] = None
        self.stage_round_counts: Dict[TrainingStage, int] = {s: 0 for s in TrainingStage}

    def start_stage(self, stage: TrainingStage, approved: bool = False) -> bool:
        """Start a training stage"""
        config = self.stages[stage]

        if not config.enabled:
            logger.error(f"Cannot start disabled stage: {stage.value}")
            return False

        if config.requires_approval and not approved:
            logger.error(f"Stage {stage.value} requires explicit approval")
            return False

        self.current_stage = stage
        self.stage_round_counts[stage] = 0
        logger.info(f"Started training stage: {stage.value}")
        return True

    def record_round(self) -> bool:
        """Record a completed round for current stage"""
        if self.current_stage is None:
            logger.error("No active training stage")
            return False

        self.stage_round_counts[self.current_stage] += 1
        current_count = self.stage_round_counts[self.current_stage]
        max_rounds = self.stages[self.current_stage].max_rounds

        logger.info(f"Stage {self.current_stage.value}: round {current_count}/{max_rounds}")

        if current_count >= max_rounds:
            logger.warning(f"Stage {self.current_stage.value} reached max rounds")
            return False

        return True

    def complete_stage(self, stage: TrainingStage):
        """Mark stage as complete"""
        logger.info(f"Completed training stage: {stage.value} ({self.stage_round_counts[stage]} rounds)")
        if self.current_stage == stage:
            self.current_stage = None


# ============================================================================
# 7. SRE Observability and Instrumentation
# ============================================================================

@dataclass
class RoundLatencyBreakdown:
    """Latency breakdown for a federated learning round"""
    train_ms: float = 0.0
    compress_ms: float = 0.0
    encrypt_ms: float = 0.0
    upload_ms: float = 0.0
    aggregate_ms: float = 0.0
    decrypt_ms: float = 0.0
    apply_ms: float = 0.0

    def total_ms(self) -> float:
        return sum([
            self.train_ms, self.compress_ms, self.encrypt_ms,
            self.upload_ms, self.aggregate_ms, self.decrypt_ms, self.apply_ms
        ])


@dataclass
class CompressionMetrics:
    """Compression efficiency metrics"""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0

    def effective_ratio(self) -> float:
        if self.compressed_size_bytes == 0:
            return float('inf')
        return self.original_size_bytes / self.compressed_size_bytes


@dataclass
class ModelQualityMetrics:
    """Model quality KPIs"""
    success_rate: float = 0.0
    average_reward: float = 0.0
    kl_divergence: float = 0.0
    update_norm: float = 0.0
    is_outlier: bool = False


class ObservabilityCollector:
    """
    SRE-grade observability for TensorGuard.

    Treats training like a distributed system problem with full visibility.
    """

    def __init__(self, metrics_file: Optional[Path] = None):
        self.metrics_file = metrics_file or Path("tensorguard_metrics.jsonl")
        self.current_round_metrics: Dict[str, Any] = {}

    def record_latency(self, breakdown: RoundLatencyBreakdown, round_number: int):
        """Record latency breakdown for a round"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "latency",
            "round": round_number,
            "train_ms": breakdown.train_ms,
            "compress_ms": breakdown.compress_ms,
            "encrypt_ms": breakdown.encrypt_ms,
            "upload_ms": breakdown.upload_ms,
            "aggregate_ms": breakdown.aggregate_ms,
            "decrypt_ms": breakdown.decrypt_ms,
            "apply_ms": breakdown.apply_ms,
            "total_ms": breakdown.total_ms()
        }
        self._write_metric(metric)
        logger.info(f"Round {round_number} latency: {breakdown.total_ms():.0f}ms")

    def record_compression(self, metrics: CompressionMetrics, round_number: int):
        """Record compression metrics"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "compression",
            "round": round_number,
            "original_bytes": metrics.original_size_bytes,
            "compressed_bytes": metrics.compressed_size_bytes,
            "ratio": metrics.compression_ratio,
            "effective_ratio": metrics.effective_ratio()
        }
        self._write_metric(metric)
        logger.info(f"Compression: {metrics.original_size_bytes/1024:.1f}KB -> {metrics.compressed_size_bytes/1024:.1f}KB ({metrics.effective_ratio():.1f}x)")

    def record_dp_epsilon(self, consumed: float, budget: float, round_number: int):
        """Record DP epsilon consumption"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "privacy",
            "round": round_number,
            "epsilon_consumed": consumed,
            "epsilon_budget": budget,
            "epsilon_remaining": budget - consumed,
            "consumption_rate": consumed / budget if budget > 0 else 0
        }
        self._write_metric(metric)

    def record_quality(self, quality: ModelQualityMetrics, round_number: int):
        """Record model quality metrics"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "quality",
            "round": round_number,
            "success_rate": quality.success_rate,
            "average_reward": quality.average_reward,
            "kl_divergence": quality.kl_divergence,
            "update_norm": quality.update_norm,
            "is_outlier": quality.is_outlier
        }
        self._write_metric(metric)

    def record_expert_weights(self, weights: Dict[str, float], round_number: int):
        """Record MoI expert distribution (FedMoE v2.0)"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "moi",
            "round": round_number,
            "weights": weights
        }
        self._write_metric(metric)
        logger.info(f"MoI Distribution: {weights}")

    def record_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Record an alert event"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "alert",
            "alert_type": alert_type,
            "message": message,
            "severity": severity
        }
        self._write_metric(metric)

        if severity == "critical":
            logger.critical(f"ALERT [{alert_type}]: {message}")
        elif severity == "error":
            logger.error(f"ALERT [{alert_type}]: {message}")
        else:
            logger.warning(f"ALERT [{alert_type}]: {message}")

    def _write_metric(self, metric: Dict[str, Any]):
        """Write metric to JSONL file with numpy serialization handling."""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric, default=numpy_json_serializer) + "\n")


# ============================================================================
# 8. Resilient Aggregation
# ============================================================================

@dataclass
class ClientContribution:
    """A client's contribution to a federated round"""
    client_id: str
    update_package: UpdatePackage
    received_at: datetime
    staleness_seconds: float = 0.0
    weight: float = 1.0
    health_score: float = 1.0


class ResilientAggregator:
    """
    Production-grade aggregation that handles:
    - Stragglers
    - Dropouts
    - Staleness
    - Client reputation

    Essential for scaling beyond 3-5 sites.
    """

    def __init__(
        self,
        quorum_threshold: int = 2,
        max_staleness_seconds: float = 3600,
        enable_async: bool = False
    ):
        self.quorum_threshold = quorum_threshold
        self.max_staleness_seconds = max_staleness_seconds
        self.enable_async = enable_async

        self.contributions: List[ClientContribution] = []
        self.received_client_ids: set = set()
        self.client_health: Dict[str, float] = {}  # Client ID -> health score
        self.round_start_time: Optional[datetime] = None

    def start_round(self):
        """Start a new aggregation round"""
        self.contributions.clear()
        self.received_client_ids.clear()
        self.round_start_time = datetime.utcnow()
        logger.info(f"Started aggregation round at {self.round_start_time.isoformat()}")

    def add_contribution(self, contribution: ClientContribution) -> bool:
        """
        Add a client contribution with unique client enforcement.

        Returns:
            True if contribution accepted, False if rejected
        """
        if contribution.client_id in self.received_client_ids:
            logger.warning(f"Rejecting duplicate contribution from {contribution.client_id}")
            return False
        # Calculate staleness
        if self.round_start_time:
            contribution.staleness_seconds = (contribution.received_at - self.round_start_time).total_seconds()

        # Check staleness threshold
        if contribution.staleness_seconds > self.max_staleness_seconds:
            logger.warning(f"Rejecting stale contribution from {contribution.client_id}: {contribution.staleness_seconds:.0f}s > {self.max_staleness_seconds:.0f}s")
            return False

        # Apply staleness weighting (exponential decay)
        staleness_weight = np.exp(-contribution.staleness_seconds / self.max_staleness_seconds)

        # Get client health score
        client_health = self.client_health.get(contribution.client_id, 1.0)

        # Combine weights
        contribution.weight = staleness_weight * client_health
        contribution.health_score = client_health

        self.contributions.append(contribution)
        self.received_client_ids.add(contribution.client_id)
        logger.info(f"Accepted contribution from {contribution.client_id} (weight={contribution.weight:.3f}, staleness={contribution.staleness_seconds:.0f}s)")
        return True

    def can_aggregate(self) -> bool:
        """Check if we have enough contributions to aggregate"""
        return len(self.contributions) >= self.quorum_threshold

    def update_client_health(self, client_id: str, health_score: float):
        """
        Update client health/reputation score.

        Lower scores down-weight contributions from unreliable clients.
        """
        self.client_health[client_id] = max(0.0, min(1.0, health_score))
        logger.info(f"Updated client {client_id} health: {self.client_health[client_id]:.3f}")

    def detect_outliers(self) -> List[str]:
        """
        Detect outlier contributions based on update norms.

        Returns list of client IDs flagged as outliers.
        """
        if len(self.contributions) < 3:
            return []  # Need at least 3 for outlier detection

        # Extract update norms from safety stats
        norms = [c.update_package.safety_stats.grad_norm_max for c in self.contributions]

        # Simple outlier detection: >3 sigma from median
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        threshold = median_norm + 3 * 1.4826 * mad  # 1.4826 = 1/norm.ppf(0.75)

        outliers = []
        for contrib, norm in zip(self.contributions, norms):
            if norm > threshold:
                outliers.append(contrib.client_id)
                logger.warning(f"Outlier detected: {contrib.client_id} (norm={norm:.3f} > threshold={threshold:.3f})")

        return outliers

    def get_aggregation_weights(self) -> Dict[str, float]:
        """Get normalized aggregation weights for all contributions"""
        total_weight = sum(c.weight for c in self.contributions)

        if total_weight == 0:
            # Fallback to uniform weights
            return {c.client_id: 1.0 / len(self.contributions) for c in self.contributions}

        return {c.client_id: c.weight / total_weight for c in self.contributions}


# ============================================================================
# 9. Unified Pipeline Workflow Management (v2.1)
# ============================================================================

class PipelineStage(Enum):
    """The 7-stage TensorGuard workflow Formalized."""
    CAPTURE = "capture"   # Raw telemetry ingest
    EMBED   = "embed"     # Feature extraction
    GATE    = "gate"      # Expert routing (IOSP)
    PEFT    = "peft"      # LoRA update calculation
    SHIELD  = "shield"    # N2HE + Skellam DP
    SYNC    = "sync"      # Federated Aggregation
    PULL    = "pull"      # Global consensus update

@dataclass
class StageResult:
    """Telemetric record of a single pipeline stage execution."""
    stage: PipelineStage
    status: str  # "ok", "error", "degraded"
    latency_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BoundaryGuard:
    """Validates data integrity between pipeline stages to prevent malformed updates."""
    
    @staticmethod
    def validate_stage_input(stage: PipelineStage, data: Any):
        """Pre-execution check for each stage."""
        if stage == PipelineStage.SHIELD:
            if not isinstance(data, dict) or not all(isinstance(v, np.ndarray) for v in data.values()):
                raise ValidationError(f"Invalid input to SHIELD: Expected gradient dict, got {type(data)}")
        
        elif stage == PipelineStage.SYNC:
            if not hasattr(data, 'serialize'):
                raise ValidationError(f"Invalid input to SYNC: Expected UpdatePackage, got {type(data)}")
                
        # Additional guards can be expanded here for 'Enterprise' precision
        logger.debug(f"[BoundaryGuard] Stage {stage.value} input validated.")

class UnifiedPipelineManager:
    """
    Orchestrates the 7-stage TensorGuard workflow with surgical hardening.
    
    Implements:
    - Circuit Breakers: Immediate stop if critical security fails.
    - Graceful Degradation: Reverts to 'Safe Mode' policies on non-critical errors.
    - Observability: Full telemetry for the 'Command Center' UI.
    """
    
    def __init__(self, fleet_id: str, observability: ObservabilityCollector):
        self.fleet_id = fleet_id
        self.obs = observability
        self.history: List[StageResult] = []
        self.safe_mode_active = False

    def run_stage(self, stage: PipelineStage, func: callable, *args, **kwargs) -> Any:
        """Executes a stage with guard-rails and timing."""
        start_time = time.time()
        try:
            # 1. Pre-Execution Guard
            BoundaryGuard.validate_stage_input(stage, args[0] if args else None)
            
            # 2. Execution
            result = func(*args, **kwargs)
            
            # 3. Success Logging
            latency = (time.time() - start_time) * 1000
            self.history.append(StageResult(stage, "ok", latency))
            return result
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # 4. Graceful Degradation / Circuit Breaker Logic
            if stage in [PipelineStage.SHIELD, PipelineStage.GATE]:
                # Critical security stages - Activate Circuit Breaker
                logger.critical(f"Circuit Breaker TRIPPED at {stage.value}: {error_msg}")
                self.history.append(StageResult(stage, "error", latency, error_msg))
                self.safe_mode_active = True
                raise RuntimeError(f"Workflow Halted: Security Fault in {stage.value}")
            else:
                # Non-critical stages - Degradation
                logger.warning(f"Stage {stage.value} degraded: {error_msg}. Falling back to Safe Default.")
                self.history.append(StageResult(stage, "degraded", latency, error_msg))
                return self._get_safe_fallback(stage)

    def _get_safe_fallback(self, stage: PipelineStage) -> Any:
        """Returns non-breaking default results for degraded stages."""
        if stage == PipelineStage.EMBED:
            return np.zeros((1, 128)) # Empty embedding
        if stage == PipelineStage.PEFT:
            return {} # No update
        return None

    def export_telemetry(self) -> str:
        """Export JSON trace for the Platform UI."""
        trace = {
            "fleet_id": self.fleet_id,
            "timestamp": datetime.utcnow().isoformat(),
            "safe_mode": self.safe_mode_active,
            "workflow": [asdict(r) for r in self.history]
        }
        return json.dumps(trace, default=numpy_json_serializer)

# ============================================================================
# Production Blueprint Summary
# ============================================================================

def print_production_status():
    """Print production readiness status"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         TensorGuard Production Optimization Blueprint        ║
╚══════════════════════════════════════════════════════════════╝

✓ Operating Envelope Enforcement
✓ Canonical UpdatePackage Format
✓ Separated Privacy/Training Controls
✓ Enterprise Key Management
✓ Production Evaluation Gating
✓ Clean IL/RL Pipeline Separation
✓ SRE Observability & Instrumentation
✓ Resilient Aggregation (Quorum/Staleness/Health)

Production readiness: ENABLED
Ready for secure post-training at scale.
    """)
