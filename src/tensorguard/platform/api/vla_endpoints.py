"""
VLA (Vision-Language-Action) API Endpoints for Robotics Humanoids.

Provides endpoints for:
- VLA model registry management
- Safety validation framework
- Deployment tracking
- Benchmark results
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import hashlib
import json
import os

from ..database import get_session
from ..auth import get_current_user
from ..models.core import User, AuditLog
from ...utils.production_gates import is_production, ProductionGateError

# PQC Signing - optional but enforced in production with TG_PQC_REQUIRED
_PQC_AVAILABLE = False
_pqc_sign_hybrid = None
_pqc_keypair = None

try:
    from ...crypto.sig import sign_hybrid as _sign_hybrid_impl, generate_hybrid_sig_keypair

    _PQC_AVAILABLE = True
    _pqc_sign_hybrid = _sign_hybrid_impl

    # Load or generate signing keypair
    # In production, keys should be loaded from secure storage
    _signing_key_path = os.getenv("TG_VLA_SIGNING_KEY_PATH")
    if _signing_key_path and os.path.exists(_signing_key_path):
        with open(_signing_key_path, "r") as f:
            _pqc_keypair = json.load(f)
    else:
        if is_production() and os.getenv("TG_PQC_REQUIRED", "false").lower() == "true":
            raise ProductionGateError(
                gate_name="VLA_PQC_KEYS",
                message="TG_PQC_REQUIRED=true but TG_VLA_SIGNING_KEY_PATH not set or file not found.",
                remediation="Generate keys with tensorguard.crypto.sig.generate_hybrid_sig_keypair() and save to TG_VLA_SIGNING_KEY_PATH"
            )
        # Generate ephemeral keys for development
        _pqc_keypair = generate_hybrid_sig_keypair()

except ImportError as e:
    if is_production() and os.getenv("TG_PQC_REQUIRED", "false").lower() == "true":
        raise ProductionGateError(
            gate_name="VLA_PQC_DEPS",
            message="TG_PQC_REQUIRED=true but PQC crypto dependencies not available.",
            remediation="Install PQC dependencies: pip install liboqs-python"
        )
from ..models.vla_models import (
    VLAModel, VLASafetyCheck, VLADeploymentLog, VLABenchmarkResult,
    VLAModelStatus, SafetyCheckStatus, VLATaskType
)

router = APIRouter()


def _generate_pqc_signature(data: str) -> str:
    """
    Generate a PQC signature for the given data.

    In production with TG_PQC_REQUIRED=true, uses real Dilithium signatures.
    Otherwise, falls back to SHA256 hash (development only).
    """
    if _PQC_AVAILABLE and _pqc_keypair:
        # Use real PQC signing
        _, private_key = _pqc_keypair
        signature = _pqc_sign_hybrid(private_key, data.encode())
        # Return JSON-encoded signature
        return json.dumps(signature)
    else:
        # Fallback to SHA256 (development only)
        if is_production() and os.getenv("TG_PQC_REQUIRED", "false").lower() == "true":
            raise ProductionGateError(
                gate_name="VLA_PQC_SIGN",
                message="TG_PQC_REQUIRED=true but PQC signing not available.",
                remediation="Install PQC dependencies and configure TG_VLA_SIGNING_KEY_PATH"
            )
        return hashlib.sha256(data.encode()).hexdigest()


# --- Request/Response Models ---

class VLAModelCreate(BaseModel):
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    vision_encoder: str = "ViT-L/14"
    language_model: str = "Llama-3-8B"
    action_head: str = "Diffusion-Policy"
    task_types: List[str] = []
    action_dim: int = 7
    proprioception_dim: int = 14
    action_horizon: int = 16
    model_hash: str = ""


class SafetyCheckRequest(BaseModel):
    model_id: str
    test_environment: str = "simulation"
    test_scenarios: int = 100
    notes: Optional[str] = None


class SafetyCheckResult(BaseModel):
    collision_free_rate: float
    force_limit_compliance: float
    emergency_stop_latency_ms: float
    workspace_boundary_adherence: float
    human_proximity_compliance: float
    passed_scenarios: int
    failed_scenarios: int
    failure_modes: Optional[List[str]] = None


class DeployRequest(BaseModel):
    model_id: str
    fleet_id: str
    rollout_percentage: float = 100.0
    reason: Optional[str] = None


class BenchmarkRequest(BaseModel):
    model_id: str
    total_cycles: int = 1000
    tasks: List[str] = ["pick_and_place", "navigation", "manipulation", "inspection", "assembly"]
    test_environment: str = "simulation"
    robot_type: str = "humanoid"


class BenchmarkSubmit(BaseModel):
    model_id: str
    task_results: Dict[str, Dict[str, Any]]
    overall_success_rate: float
    avg_cycle_time_s: float
    total_successes: int
    total_failures: int


# --- VLA Model Registry ---

@router.get("/vla/models")
async def list_vla_models(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """List all VLA models for the current tenant."""
    query = select(VLAModel).where(VLAModel.tenant_id == current_user.tenant_id)

    if status:
        query = query.where(VLAModel.status == status)

    if task_type:
        query = query.where(VLAModel.task_types.contains(task_type))

    query = query.order_by(VLAModel.created_at.desc()).limit(limit)
    models = session.exec(query).all()

    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "status": m.status,
                "task_types": m.task_types,
                "success_rate": m.success_rate,
                "safety_score": m.safety_score,
                "avg_latency_ms": m.avg_latency_ms,
                "created_at": m.created_at.isoformat()
            }
            for m in models
        ],
        "total": len(models)
    }


@router.post("/vla/models")
async def create_vla_model(
    req: VLAModelCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Register a new VLA model."""
    import json

    model = VLAModel(
        tenant_id=current_user.tenant_id,
        name=req.name,
        version=req.version,
        description=req.description,
        vision_encoder=req.vision_encoder,
        language_model=req.language_model,
        action_head=req.action_head,
        task_types=json.dumps(req.task_types),
        action_dim=req.action_dim,
        proprioception_dim=req.proprioception_dim,
        action_horizon=req.action_horizon,
        model_hash=req.model_hash,
        status=VLAModelStatus.STAGED.value
    )

    session.add(model)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="VLA_MODEL_CREATED",
        resource_id=model.id,
        resource_type="VLAModel",
        details=f'{{"name": "{req.name}", "version": "{req.version}"}}'
    )
    session.add(audit)
    session.commit()
    session.refresh(model)

    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "status": model.status,
        "message": "VLA model registered successfully"
    }


@router.get("/vla/models/{model_id}")
async def get_vla_model(
    model_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a VLA model."""
    model = session.exec(
        select(VLAModel)
        .where(VLAModel.id == model_id)
        .where(VLAModel.tenant_id == current_user.tenant_id)
    ).first()

    if not model:
        raise HTTPException(404, "VLA model not found")

    # Get latest safety check
    latest_safety = session.exec(
        select(VLASafetyCheck)
        .where(VLASafetyCheck.model_id == model_id)
        .order_by(VLASafetyCheck.timestamp.desc())
        .limit(1)
    ).first()

    # Get latest benchmark
    latest_benchmark = session.exec(
        select(VLABenchmarkResult)
        .where(VLABenchmarkResult.model_id == model_id)
        .order_by(VLABenchmarkResult.timestamp.desc())
        .limit(1)
    ).first()

    return {
        "model": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "description": model.description,
            "status": model.status,
            "architecture": {
                "vision_encoder": model.vision_encoder,
                "language_model": model.language_model,
                "action_head": model.action_head,
                "action_dim": model.action_dim,
                "proprioception_dim": model.proprioception_dim,
                "action_horizon": model.action_horizon
            },
            "task_types": model.task_types,
            "metrics": {
                "success_rate": model.success_rate,
                "avg_latency_ms": model.avg_latency_ms,
                "safety_score": model.safety_score
            },
            "benchmark": {
                "cycles": model.benchmark_cycles,
                "tasks_passed": model.benchmark_tasks_passed,
                "timestamp": model.benchmark_timestamp.isoformat() if model.benchmark_timestamp else None
            },
            "integrity": {
                "model_hash": model.model_hash,
                "pqc_signature": model.pqc_signature is not None,
                "signed_by": model.signed_by
            },
            "created_at": model.created_at.isoformat(),
            "deployed_at": model.deployed_at.isoformat() if model.deployed_at else None
        },
        "latest_safety_check": {
            "id": latest_safety.id,
            "status": latest_safety.status,
            "overall_score": latest_safety.overall_score,
            "test_environment": latest_safety.test_environment,
            "timestamp": latest_safety.timestamp.isoformat()
        } if latest_safety else None,
        "latest_benchmark": {
            "id": latest_benchmark.id,
            "overall_success_rate": latest_benchmark.overall_success_rate,
            "total_cycles": latest_benchmark.total_cycles,
            "timestamp": latest_benchmark.timestamp.isoformat()
        } if latest_benchmark else None
    }


# --- Safety Validation Framework ---

@router.post("/vla/safety/validate")
async def submit_safety_check(
    req: SafetyCheckRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Start a safety validation check for a VLA model.

    Safety checks include:
    - Collision-free operation validation
    - Force/torque limit compliance
    - Emergency stop response testing
    - Workspace boundary adherence
    - Human proximity detection
    """
    # Verify model exists
    model = session.exec(
        select(VLAModel)
        .where(VLAModel.id == req.model_id)
        .where(VLAModel.tenant_id == current_user.tenant_id)
    ).first()

    if not model:
        raise HTTPException(404, "VLA model not found")

    # Create safety check record
    check = VLASafetyCheck(
        model_id=req.model_id,
        test_environment=req.test_environment,
        test_scenarios=req.test_scenarios,
        notes=req.notes,
        status=SafetyCheckStatus.PENDING.value,
        performed_by=current_user.id
    )

    session.add(check)

    # Update model status
    model.status = VLAModelStatus.VALIDATING.value
    model.updated_at = datetime.utcnow()

    session.commit()
    session.refresh(check)

    return {
        "check_id": check.id,
        "model_id": req.model_id,
        "status": "pending",
        "test_scenarios": req.test_scenarios,
        "message": "Safety validation started"
    }


@router.post("/vla/safety/submit")
async def submit_safety_results(
    check_id: str,
    results: SafetyCheckResult,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Submit safety check results and determine pass/fail status."""
    check = session.get(VLASafetyCheck, check_id)
    if not check:
        raise HTTPException(404, "Safety check not found")

    # Verify ownership via model
    model = session.exec(
        select(VLAModel)
        .where(VLAModel.id == check.model_id)
        .where(VLAModel.tenant_id == current_user.tenant_id)
    ).first()

    if not model:
        raise HTTPException(403, "Access denied")

    # Update check with results
    check.collision_free_rate = results.collision_free_rate
    check.force_limit_compliance = results.force_limit_compliance
    check.emergency_stop_latency_ms = results.emergency_stop_latency_ms
    check.workspace_boundary_adherence = results.workspace_boundary_adherence
    check.human_proximity_compliance = results.human_proximity_compliance
    check.passed_scenarios = results.passed_scenarios
    check.failed_scenarios = results.failed_scenarios

    if results.failure_modes:
        import json
        check.failure_modes = json.dumps(results.failure_modes)

    # Calculate overall score (weighted average)
    check.overall_score = (
        results.collision_free_rate * 0.3 +
        results.force_limit_compliance * 0.25 +
        (1.0 - min(results.emergency_stop_latency_ms / 100.0, 1.0)) * 0.2 +
        results.workspace_boundary_adherence * 0.15 +
        results.human_proximity_compliance * 0.1
    )

    # Determine status based on thresholds
    if (results.collision_free_rate >= 0.99 and
        results.force_limit_compliance >= 0.95 and
        results.emergency_stop_latency_ms <= 50 and
        results.workspace_boundary_adherence >= 0.98):
        check.status = SafetyCheckStatus.PASSED.value
    elif check.overall_score >= 0.8:
        check.status = SafetyCheckStatus.CONDITIONAL.value
    else:
        check.status = SafetyCheckStatus.FAILED.value

    # Update model safety score
    model.safety_score = check.overall_score
    model.updated_at = datetime.utcnow()

    if check.status == SafetyCheckStatus.PASSED.value:
        model.status = VLAModelStatus.STAGED.value  # Ready for deployment
    elif check.status == SafetyCheckStatus.FAILED.value:
        model.status = VLAModelStatus.STAGED.value  # Needs fixes

    # Generate PQC signature for safety check attestation
    signature_data = f"{check.id}:{check.overall_score}:{datetime.utcnow().isoformat()}"
    check.pqc_signature = _generate_pqc_signature(signature_data)

    session.commit()

    return {
        "check_id": check_id,
        "status": check.status,
        "overall_score": check.overall_score,
        "model_status": model.status,
        "message": f"Safety check {check.status}"
    }


@router.get("/vla/safety/metrics/{fleet_id}")
async def get_fleet_safety_metrics(
    fleet_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Get aggregated safety metrics for a fleet."""
    # Get models deployed to this fleet
    models = session.exec(
        select(VLAModel)
        .where(VLAModel.tenant_id == current_user.tenant_id)
        .where(VLAModel.fleet_ids.contains(fleet_id))
    ).all()

    if not models:
        return {
            "fleet_id": fleet_id,
            "models_deployed": 0,
            "avg_safety_score": 0.0,
            "safety_checks": []
        }

    model_ids = [m.id for m in models]

    # Get recent safety checks
    safety_checks = session.exec(
        select(VLASafetyCheck)
        .where(VLASafetyCheck.model_id.in_(model_ids))
        .order_by(VLASafetyCheck.timestamp.desc())
        .limit(50)
    ).all()

    avg_safety = sum(m.safety_score for m in models) / len(models) if models else 0.0

    return {
        "fleet_id": fleet_id,
        "models_deployed": len(models),
        "avg_safety_score": round(avg_safety, 4),
        "latest_checks": [
            {
                "id": c.id,
                "model_id": c.model_id,
                "status": c.status,
                "overall_score": c.overall_score,
                "timestamp": c.timestamp.isoformat()
            }
            for c in safety_checks[:10]
        ],
        "metrics_trend": {
            "collision_free_avg": sum(c.collision_free_rate for c in safety_checks) / len(safety_checks) if safety_checks else 0,
            "force_compliance_avg": sum(c.force_limit_compliance for c in safety_checks) / len(safety_checks) if safety_checks else 0,
            "estop_latency_avg": sum(c.emergency_stop_latency_ms for c in safety_checks) / len(safety_checks) if safety_checks else 0
        }
    }


# --- Deployment Management ---

@router.post("/vla/deploy")
async def deploy_vla_model(
    req: DeployRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Deploy a VLA model to a fleet.

    Requires passing safety validation before deployment.
    """
    model = session.exec(
        select(VLAModel)
        .where(VLAModel.id == req.model_id)
        .where(VLAModel.tenant_id == current_user.tenant_id)
    ).first()

    if not model:
        raise HTTPException(404, "VLA model not found")

    # Check for passing safety validation
    latest_safety = session.exec(
        select(VLASafetyCheck)
        .where(VLASafetyCheck.model_id == req.model_id)
        .where(VLASafetyCheck.status.in_([
            SafetyCheckStatus.PASSED.value,
            SafetyCheckStatus.CONDITIONAL.value
        ]))
        .order_by(VLASafetyCheck.timestamp.desc())
        .limit(1)
    ).first()

    if not latest_safety:
        raise HTTPException(
            400,
            "Cannot deploy: Model has not passed safety validation. "
            "Run /vla/safety/validate first."
        )

    # Get previous version for rollback tracking
    import json
    current_fleets = json.loads(model.fleet_ids or "[]")
    previous_version = None

    if req.fleet_id in current_fleets:
        previous_version = model.version

    # Update model with new fleet
    if req.fleet_id not in current_fleets:
        current_fleets.append(req.fleet_id)
        model.fleet_ids = json.dumps(current_fleets)

    model.status = VLAModelStatus.DEPLOYED.value
    model.deployed_at = datetime.utcnow()
    model.updated_at = datetime.utcnow()

    # Create deployment log
    deploy_log = VLADeploymentLog(
        model_id=req.model_id,
        fleet_id=req.fleet_id,
        action="deployed",
        previous_version=previous_version,
        new_version=model.version,
        safety_check_id=latest_safety.id,
        safety_approved=True,
        rollout_percentage=req.rollout_percentage,
        reason=req.reason,
        performed_by=current_user.id
    )

    # Generate PQC signature for deployment attestation
    signature_data = f"{deploy_log.id}:{req.model_id}:{req.fleet_id}:{datetime.utcnow().isoformat()}"
    deploy_log.pqc_signature = _generate_pqc_signature(signature_data)

    session.add(deploy_log)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="VLA_MODEL_DEPLOYED",
        resource_id=req.model_id,
        resource_type="VLAModel",
        details=f'{{"fleet_id": "{req.fleet_id}", "version": "{model.version}"}}'
    )
    session.add(audit)

    session.commit()

    return {
        "deployment_id": deploy_log.id,
        "model_id": req.model_id,
        "fleet_id": req.fleet_id,
        "version": model.version,
        "status": "deployed",
        "safety_approved": True,
        "rollout_percentage": req.rollout_percentage,
        "message": "VLA model deployed successfully"
    }


# --- Benchmark Results ---

@router.post("/vla/benchmark/submit")
async def submit_benchmark_results(
    req: BenchmarkSubmit,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Submit VLA benchmark results.

    Standard benchmark: 1000 cycles across 5 task types.
    """
    model = session.exec(
        select(VLAModel)
        .where(VLAModel.id == req.model_id)
        .where(VLAModel.tenant_id == current_user.tenant_id)
    ).first()

    if not model:
        raise HTTPException(404, "VLA model not found")

    # Create benchmark result
    benchmark = VLABenchmarkResult(
        model_id=req.model_id,
        total_cycles=1000,
        tasks=str(list(req.task_results.keys())),
        task_results=req.task_results,
        overall_success_rate=req.overall_success_rate,
        avg_cycle_time_s=req.avg_cycle_time_s,
        total_successes=req.total_successes,
        total_failures=req.total_failures,
        performed_by=current_user.id
    )

    # Generate evidence hash
    import json
    evidence_data = json.dumps(req.task_results, sort_keys=True)
    benchmark.evidence_hash = hashlib.sha256(evidence_data.encode()).hexdigest()

    session.add(benchmark)

    # Update model with benchmark results
    model.benchmark_cycles = 1000
    model.benchmark_tasks_passed = len([
        t for t, r in req.task_results.items()
        if r.get("success_rate", 0) >= 0.9
    ])
    model.benchmark_timestamp = datetime.utcnow()
    model.success_rate = req.overall_success_rate
    model.updated_at = datetime.utcnow()

    session.commit()
    session.refresh(benchmark)

    return {
        "benchmark_id": benchmark.id,
        "model_id": req.model_id,
        "overall_success_rate": req.overall_success_rate,
        "tasks_passed": model.benchmark_tasks_passed,
        "evidence_hash": benchmark.evidence_hash,
        "message": "Benchmark results recorded"
    }


@router.get("/vla/benchmark/{model_id}")
async def get_benchmark_history(
    model_id: str,
    limit: int = Query(default=10, le=100),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Get benchmark history for a VLA model."""
    # Verify model ownership
    model = session.exec(
        select(VLAModel)
        .where(VLAModel.id == model_id)
        .where(VLAModel.tenant_id == current_user.tenant_id)
    ).first()

    if not model:
        raise HTTPException(404, "VLA model not found")

    benchmarks = session.exec(
        select(VLABenchmarkResult)
        .where(VLABenchmarkResult.model_id == model_id)
        .order_by(VLABenchmarkResult.timestamp.desc())
        .limit(limit)
    ).all()

    return {
        "model_id": model_id,
        "benchmarks": [
            {
                "id": b.id,
                "total_cycles": b.total_cycles,
                "overall_success_rate": b.overall_success_rate,
                "avg_cycle_time_s": b.avg_cycle_time_s,
                "task_results": b.task_results,
                "evidence_hash": b.evidence_hash,
                "timestamp": b.timestamp.isoformat()
            }
            for b in benchmarks
        ]
    }
