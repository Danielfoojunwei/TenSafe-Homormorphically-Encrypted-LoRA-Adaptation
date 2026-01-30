from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, func
from typing import List, Any, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import secrets
import hashlib

from ..database import get_session
from ..models.core import Tenant, User, Fleet, Job, UserRole
from ..models.telemetry_models import FleetDevice, TelemetryStageEvent, StageStatus
from ..auth import get_current_user, create_access_token, verify_password, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
from .identity_endpoints import verify_fleet_auth
from ...utils.production_gates import is_demo_mode

router = APIRouter()

# --- Auth ---
class Token(BaseModel):
    access_token: str
    token_type: str

class LoginData(BaseModel):
    username: str
    password: str

@router.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: LoginData, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role, "tenant_id": user.tenant_id}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- Tenants ---
@router.post("/onboarding/init", response_model=Tenant)
async def init_tenant(name: str, admin_email: str, admin_pass: str, session: Session = Depends(get_session)):
    """Initialize a new tenant and admin user."""
    try:
        # Check if user exists
        existing_user = session.exec(select(User).where(User.email == admin_email)).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
            
        tenant = Tenant(name=name, plan="Enterprise")
        session.add(tenant)
        session.commit()
        session.refresh(tenant)
        
        user = User(
            email=admin_email, 
            hashed_password=get_password_hash(admin_pass),
            role=UserRole.ORG_ADMIN,
            tenant_id=tenant.id
        )
        session.add(user)
        session.commit()
        
        return tenant
    except Exception as e:
        print(f"ERROR in init_tenant: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Fleets ---
@router.get("/fleets", response_model=List[Fleet])
async def get_fleets(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return session.exec(select(Fleet).where(Fleet.tenant_id == current_user.tenant_id)).all()


@router.get("/fleets/extended")
async def get_fleets_extended(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    """
    Get fleets with extended metrics (device counts, trust scores, region info).

    All metrics are computed from real database data:
    - Device counts from FleetDevice table
    - Online status from last_seen_at within 5 minutes
    - Error rates from TelemetryStageEvent table
    - Trust score derived from error rates

    No random/simulated data in production paths.
    """
    fleets = session.exec(select(Fleet).where(Fleet.tenant_id == current_user.tenant_id)).all()

    result = []
    now = datetime.utcnow()
    online_threshold = now - timedelta(minutes=5)
    telemetry_window = now - timedelta(hours=1)

    for fleet in fleets:
        # Query real device counts
        devices_total = session.exec(
            select(func.count(FleetDevice.id))
            .where(FleetDevice.fleet_id == fleet.id)
        ).one() or 0

        devices_online = session.exec(
            select(func.count(FleetDevice.id))
            .where(
                FleetDevice.fleet_id == fleet.id,
                FleetDevice.last_seen_at >= online_threshold
            )
        ).one() or 0

        # Query error rate from telemetry
        total_events = session.exec(
            select(func.count(TelemetryStageEvent.id))
            .where(
                TelemetryStageEvent.fleet_id == fleet.id,
                TelemetryStageEvent.ts >= telemetry_window
            )
        ).one() or 0

        error_events = session.exec(
            select(func.count(TelemetryStageEvent.id))
            .where(
                TelemetryStageEvent.fleet_id == fleet.id,
                TelemetryStageEvent.ts >= telemetry_window,
                TelemetryStageEvent.status == StageStatus.ERROR.value
            )
        ).one() or 0

        # Compute trust score from error rate (100% if no errors, decreases with errors)
        error_rate = error_events / total_events if total_events > 0 else 0
        trust_score = max(0, 100 - (error_rate * 100))

        # Determine status
        if error_rate > 0.1:
            status = "Error"
        elif error_rate > 0.05:
            status = "Degraded"
        else:
            status = "Healthy"

        result.append({
            "id": fleet.id,
            "name": fleet.name,
            "region": fleet.region or "default",
            "status": status,
            "devices_total": devices_total,
            "devices_online": devices_online,
            "trust": round(trust_score, 1),
            "is_active": fleet.is_active,
            "metrics": {
                "error_rate": round(error_rate, 4),
                "total_events": total_events,
                "error_events": error_events,
            }
        })

    # Only show demo data if explicitly in demo mode (NOT in production)
    if not result and is_demo_mode():
        result = [
            {"id": "demo-f1", "name": "US-East-1 Cluster", "region": "us-east-1", "status": "Healthy", "devices_total": 0, "devices_online": 0, "trust": 100.0, "is_active": True, "metrics": {"error_rate": 0, "total_events": 0, "error_events": 0}},
            {"id": "demo-f2", "name": "Berlin Gigafactory", "region": "eu-central-1", "status": "Healthy", "devices_total": 0, "devices_online": 0, "trust": 100.0, "is_active": True, "metrics": {"error_rate": 0, "total_events": 0, "error_events": 0}}
        ]

    return result

@router.post("/fleets", response_model=Dict[str, Any])
async def create_fleet(name: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Generate a real secure API key
    raw_key = f"tg_{secrets.token_hex(16)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    fleet = Fleet(name=name, tenant_id=current_user.tenant_id, api_key_hash=key_hash)
    session.add(fleet)
    session.commit()
    session.refresh(fleet)
    
    # Return the raw key ONLY once
    return {
        "id": fleet.id,
        "name": fleet.name,
        "api_key": raw_key,
        "instruction": "Save this key! It will not be shown again."
    }

# --- Jobs ---
@router.get("/jobs", response_model=List[Job])
async def get_jobs(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Join with Fleet to check Tenant ID
    statement = select(Job).join(Fleet).where(Fleet.tenant_id == current_user.tenant_id)
    return session.exec(statement).all()

@router.post("/jobs", response_model=Job)
async def create_job(fleet_id: str, type: str, config: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Verify fleet ownership
    fleet = session.get(Fleet, fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Fleet not found")
        
    job = Job(fleet_id=fleet_id, type=type, config_json=config, status="pending")
    session.add(job)
    session.commit()
    session.refresh(job)
    return job

# --- Attestation & Key Release ---
class AttestationRequest(BaseModel):
    agent_id: str
    fleet_id: str
    claims: Dict[str, Any]
    nonce: str
    signature: str

@router.post("/attestation/verify")
async def verify_attestation(req: AttestationRequest, fleet: Fleet = Depends(verify_fleet_auth)):
    """Verify device integrity claims. Scoped by HMAC auth."""
    """Verify device integrity claims."""
    # MVP: Log claims and return success
    # In real world: check signature against device public key, validate TPM quotes
    print(f"Verifying attestation for agent {req.agent_id}")
    return {
        "attestation_id": "att_" + secrets.token_hex(8),
        "result": "allow",
        "reason": "software_claims_valid_mvp",
        "claims_hash": hashlib.sha256(str(req.claims).encode()).hexdigest()
    }

class KeyReleaseRequest(BaseModel):
    package_id: str
    recipient_id: str
    tgsp_version: str
    manifest_hash: str
    claims_hash: str
    device_hpke_pubkey: Optional[str] = None # Hex encoded

@router.post("/tgsp/key-release")
async def release_key(req: KeyReleaseRequest, fleet: Fleet = Depends(verify_fleet_auth)):
    """
    Release a re-wrapped DEK for a device.
    In production, this verifies attestation state and unwraps the master key from KMS.
    """
    return {
        "result": "allow",
        "rewrapped": {
            "alg": "V03_HPKE_MVP",
            "ephemeral_pub": secrets.token_hex(32),
            "ct": secrets.token_hex(64),
            "tag": secrets.token_hex(16)
        }
    }

# --- Unified Telemetry (v2.1) ---
# NOTE: The main /telemetry/pipeline endpoint is now in telemetry_endpoints.py
# This legacy endpoint redirects to the new implementation for backwards compatibility

@router.get("/telemetry/pipeline/legacy")
async def get_pipeline_telemetry_legacy(
    fleet_id: Optional[str] = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    DEPRECATED: Use GET /api/v1/telemetry/pipeline instead.

    This endpoint provides backwards compatibility but returns real DB-backed data.
    """
    # Import the real implementation
    from .telemetry_endpoints import get_pipeline_telemetry as real_pipeline_telemetry

    # Call the real implementation
    return await real_pipeline_telemetry(
        fleet_id=fleet_id,
        time_range="15m",
        stage=None,
        session=session,
        current_user=current_user
    )
