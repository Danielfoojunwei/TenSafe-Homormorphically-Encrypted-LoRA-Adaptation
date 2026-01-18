from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlmodel import Session, select
from typing import List, Any, Dict, Optional
from datetime import datetime
import hashlib
import json

from ..database import get_session
from ..models.core import User
from ..auth import get_current_user
from ..models.peft_models import PeftRun, PeftWizardDraft, IntegrationConfig, PeftRunStatus
from ...integrations.peft_hub.catalog import ConnectorCatalog, discover_connectors
from ...integrations.peft_hub.schemas import PeftWizardState, TrainingConfig
from ..dependencies import require_tenant_context

# Ensure connectors are registered
discover_connectors()

router = APIRouter()

@router.get("/connectors")
async def list_connectors(current_user: User = Depends(get_current_user)):
    return ConnectorCatalog.list_connectors()

@router.post("/connectors/test")
async def test_connector(connector_id: str, config: Dict[str, Any], current_user: User = Depends(get_current_user)):
    try:
        connector = ConnectorCatalog.get_connector(connector_id)
        result = connector.validate_config(config)
        return {
            "ok": result.ok,
            "details": result.details,
            "remediation": result.remediation,
            "installed": connector.check_installed()
        }
    except Exception as e:
        return {"ok": False, "details": str(e)}

@router.get("/profiles")
async def list_profiles(current_user: User = Depends(get_current_user)):
    return [
        {"id": "local-hf", "name": "Local HF Studio (No Accounts)", "description": "Uses local transformers and local filesystem storage."},
        {"id": "hf-mlflow-minio", "name": "MLOps Stack (MLflow + MinIO)", "description": "Standard enterprise stack using Docker Compose."},
        {"id": "k8s-template", "name": "Kubernetes Template Output", "description": "Generates YAML for remote training."}
    ]

@router.post("/wizard/compile")
async def compile_wizard(state: PeftWizardState, current_user: User = Depends(get_current_user)):
    # Derived defaults and validation
    config = state.dict()
    config_blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_hash = hashlib.sha256(config_blob).hexdigest()
    config["derived_info"] = {
        "estimated_memory": "8GB (approx)",
        "config_hash": config_hash
    }
    return config

from ...integrations.peft_hub.workflow import PeftWorkflow

async def _run_workflow_task(run_id: str):
    # We need a new session in the background task
    from ..database import SessionLocal
    with SessionLocal() as session:
        workflow = PeftWorkflow(run_id, session)
        async for _ in workflow.execute():
            pass

@router.post("/runs")
async def start_run(
    state: PeftWizardState, 
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
    tenant_id: str = Depends(require_tenant_context)
):
    run = PeftRun(
        tenant_id=tenant_id,
        created_by_user_id=current_user.id,
        config_json=state.dict(),
        status=PeftRunStatus.PENDING,
        stage="INIT"
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    background_tasks.add_task(_run_workflow_task, run.id)

    return {"run_id": run.id, "status": run.status}

@router.get("/runs")
async def list_runs(
    session: Session = Depends(get_session), 
    tenant_id: str = Depends(require_tenant_context)
):
    statement = select(PeftRun).where(PeftRun.tenant_id == tenant_id).order_by(PeftRun.created_at.desc())
    return session.exec(statement).all()

@router.get("/runs/{run_id}")
async def get_run(run_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    run = session.get(PeftRun, run_id)
    if not run or run.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.post("/runs/{run_id}/promote")
async def promote_run(
    run_id: str, 
    channel: str, 
    session: Session = Depends(get_session), 
    current_user: User = Depends(get_current_user)
):
    """
    Promote a PEFT run to a release channel.
    
    Promotion gates check:
    - eval.primary_metric >= threshold (default 0.9)
    - eval.forgetting_score <= forgetting_budget (default 0.1)
    - trust.signature_verified == true (if required)
    - privacy.mode in {"off", "n2he"} (valid modes only)
    """
    run = session.get(PeftRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Build structured diagnosis report for failures
    diagnosis = {"gates": [], "failed": []}
    
    # Gate 1: Metrics must exist
    if not run.metrics_json:
        return {
            "ok": False, 
            "reason": "Policy Gate: Missing metrics for promotion.",
            "diagnosis": {"error": "metrics_json is empty or null"}
        }
    
    metrics = run.metrics_json
    
    # Gate 2: Eval metrics
    eval_metrics = metrics.get("eval", {})
    primary_metric = eval_metrics.get("primary_metric", 0.0)
    forgetting_score = eval_metrics.get("forgetting_score", 1.0)
    threshold = metrics.get("promotion_threshold", 0.9)
    forgetting_budget = metrics.get("forgetting_budget", 0.1)
    
    diagnosis["gates"].append({
        "name": "primary_metric",
        "value": primary_metric,
        "threshold": threshold,
        "passed": primary_metric >= threshold
    })
    if primary_metric < threshold:
        diagnosis["failed"].append(f"primary_metric ({primary_metric:.3f}) < threshold ({threshold})")
    
    diagnosis["gates"].append({
        "name": "forgetting_score",
        "value": forgetting_score,
        "budget": forgetting_budget,
        "passed": forgetting_score <= forgetting_budget
    })
    if forgetting_score > forgetting_budget:
        diagnosis["failed"].append(f"forgetting_score ({forgetting_score:.3f}) > budget ({forgetting_budget})")
    
    # Gate 3: Trust verification (if required)
    trust_info = metrics.get("trust", {})
    signature_required = trust_info.get("signature_required", False)
    signature_verified = trust_info.get("signature_verified", False)
    
    if signature_required:
        diagnosis["gates"].append({
            "name": "trust_signature",
            "required": signature_required,
            "verified": signature_verified,
            "passed": signature_verified
        })
        if not signature_verified:
            diagnosis["failed"].append("trust.signature_verified is false but signature is required")
    
    # Gate 4: Privacy mode validation
    privacy_info = metrics.get("privacy", {})
    privacy_mode = privacy_info.get("mode", "off")
    valid_modes = {"off", "n2he"}
    
    diagnosis["gates"].append({
        "name": "privacy_mode",
        "value": privacy_mode,
        "valid_modes": list(valid_modes),
        "passed": privacy_mode in valid_modes
    })
    if privacy_mode not in valid_modes:
        diagnosis["failed"].append(f"privacy.mode '{privacy_mode}' not in valid modes {valid_modes}")
    
    # Final decision
    if diagnosis["failed"]:
        return {
            "ok": False,
            "reason": f"Policy Gate: {len(diagnosis['failed'])} gate(s) failed.",
            "diagnosis": diagnosis
        }
    
    # All gates passed - promote
    run.stage = f"PROMOTED_{channel.upper()}"
    run.policy_verdict = "PASS"
    run.policy_details_json = {
        "channel": channel,
        "promoted_at": datetime.utcnow().isoformat(),
        "gates": diagnosis["gates"]
    }
    session.add(run)
    session.commit()
    
    return {"ok": True, "channel": channel, "diagnosis": diagnosis}
