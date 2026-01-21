from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any
from ..database import get_session
from ..models.evidence_models import Run, RunArtifact, RunPolicyResult
from ..policy_engine import PolicyEngine
import json
import os
import shutil
import uuid

router = APIRouter()

# Lazy-loaded policy engine to avoid database access at import time
_policy_engine = None

def get_policy_engine() -> PolicyEngine:
    """Lazy-load the policy engine on first use."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    return _policy_engine

@router.post("/runs", response_model=Run)
def create_run(run_data: Dict[str, Any], session: Session = Depends(get_session)):
    """Register a new benchmark run."""
    # Expects report.json content as body mostly, or a subset
    # Let's assume the client sends the full report.json structure or just metadata
    
    # Extract keys
    rid = run_data.get("run_id") or str(uuid.uuid4())
    
    # Check duplicate
    existing = session.get(Run, rid)
    if existing:
        return existing
        
    run = Run(
        run_id=rid,
        sdk_version=run_data.get("sdk_version", "unknown"),
        git_commit=run_data.get("git_commit"),
        env_json=json.dumps(run_data.get("environment", {})),
        config_json=json.dumps(run_data.get("configs", {})),
        metrics_json=json.dumps(run_data.get("metrics", {})),
        status="registered"
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run

@router.post("/runs/{run_id}/artifacts")
async def upload_artifact(
    run_id: str, 
    file: UploadFile = File(...), 
    artifact_type: str = "report.json",
    sha256: str = "",
    session: Session = Depends(get_session)
):
    """Upload an artifact for a run."""
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
        
    # storage path
    upload_dir = f"artifacts_storage/{run_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    path = f"{upload_dir}/{file.filename}"
    
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create record
    artifact = RunArtifact(
        run_id=run_id,
        artifact_type=artifact_type,
        path=path,
        sha256=sha256
    )
    session.add(artifact)
    session.commit()
    
    return {"status": "uploaded", "path": path}

@router.post("/runs/{run_id}/evaluate")
def evaluate_run(
    run_id: str, 
    pack_id: str = "soc2-evidence-pack",
    session: Session = Depends(get_session)
):
    """Evaluate a run against a policy pack."""
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
        
    # Need report.json content. 
    # Option A: It's in the DB `metrics_json` (partial)
    # Option B: Load from artifacts.
    
    # Try to find report.json artifact
    artifact = session.exec(select(RunArtifact).where(RunArtifact.run_id == run_id, RunArtifact.artifact_type == "report.json")).first()
    
    report_data = {}
    if artifact and os.path.exists(artifact.path):
        with open(artifact.path, 'r') as f:
            report_data = json.load(f)
    else:
        # Fallback to DB stored metrics
        report_data = {"metrics": json.loads(run.metrics_json)}
        
    try:
        result = get_policy_engine().evaluate(run_id, report_data, pack_id)
        session.add(result)
        run.status = "evaluated"
        session.add(run)
        session.commit()
        session.refresh(result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")

@router.get("/runs", response_model=List[Run])
def list_runs(
    session: Session = Depends(get_session),
    limit: int = Query(default=100, le=1000, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """List runs with pagination."""
    return list(session.exec(
        select(Run).order_by(Run.created_at.desc()).offset(offset).limit(limit)
    ).all())

@router.get("/runs/{run_id}")
def get_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.get("/runs/{run_id}/results")
def get_run_results(
    run_id: str,
    session: Session = Depends(get_session),
    limit: int = Query(default=100, le=1000, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """Get run results with pagination."""
    return list(session.exec(
        select(RunPolicyResult).where(RunPolicyResult.run_id == run_id).offset(offset).limit(limit)
    ).all())
