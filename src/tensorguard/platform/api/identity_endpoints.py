"""
Identity API Endpoints - FastAPI Routes for Machine Identity Guard

Provides REST API for certificate lifecycle management.
Integrates with existing platform auth (JWT + Fleet API keys).
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import hashlib
import hmac
import time
import json
import logging

from ..database import get_session
from ..auth import get_current_user
from ..models.core import User, Fleet, ReplayNonce
from ..models.identity_models import (
    IdentityEndpoint, IdentityCertificate, IdentityPolicy,
    IdentityRenewalJob, IdentityAuditLog, IdentityAgent,
    EndpointType, Criticality, CertificateType, RenewalJobStatus, AuditAction,
)
from ...identity.policy_engine import PolicyEngine
from ...identity.inventory import InventoryService
from ...identity.audit import AuditService
from ...identity.scheduler import RenewalScheduler

logger = logging.getLogger(__name__)
router = APIRouter()


# === Request/Response Models ===

class EndpointCreate(BaseModel):
    name: str
    hostname: str
    port: int = 443
    endpoint_type: str = "kubernetes"
    environment: str = "production"
    criticality: str = "medium"
    fleet_id: str
    k8s_namespace: Optional[str] = None
    k8s_secret_name: Optional[str] = None
    k8s_ingress_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class CertificateReport(BaseModel):
    fingerprint_sha256: str
    serial_number: str
    subject_dn: str
    issuer_dn: str
    sans: List[str]
    not_before: datetime
    not_after: datetime
    key_type: str
    key_size: int
    signature_algorithm: str
    eku_server_auth: bool = True
    eku_client_auth: bool = False
    is_public_trust: bool = True


class PolicyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    max_validity_days: int = 90
    renewal_window_days: int = 30
    allow_server_auth: bool = True
    allow_client_auth: bool = False
    require_eku_separation: bool = True
    require_public_trust: bool = True
    acme_challenge_type: str = "http-01"
    preset_name: Optional[str] = None  # Use preset instead of custom


class RenewalRequest(BaseModel):
    endpoint_ids: List[str]
    policy_id: str
    scheduled_at: Optional[datetime] = None


class AgentRegister(BaseModel):
    name: str
    hostname: str
    fleet_id: str
    supported_types: Optional[List[str]] = None
    supported_challenges: Optional[List[str]] = None
    public_key_pem: Optional[str] = None
    version: Optional[str] = None


class AgentCSRSubmit(BaseModel):
    job_id: str
    csr_pem: str


class AgentChallengeComplete(BaseModel):
    job_id: str
    token: str


class AgentDeployConfirm(BaseModel):
    job_id: str


# === Fleet API Key Auth (for agents) ===

async def verify_fleet_auth(
    request: Request,
    x_tg_fleet_id: str = Header(...),
    x_tg_timestamp: str = Header(...),
    x_tg_nonce: str = Header(...),
    x_tg_signature: str = Header(...),
    session: Session = Depends(get_session),
) -> Fleet:
    """
    Verify agent request using Fleet API key + HMAC signature.
    
    Signature format: HMAC-SHA256(fleet_api_key, timestamp + nonce + body_hash)
    """
    # Check replay (timestamp within 5 minutes)
    try:
        request_time = int(x_tg_timestamp)
        current_time = int(time.time())
        if abs(current_time - request_time) > 300:
            raise HTTPException(status_code=401, detail="Request expired")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid timestamp")
    
    # Get fleet
    fleet = session.get(Fleet, x_tg_fleet_id)
    if not fleet or not fleet.is_active:
        raise HTTPException(status_code=401, detail="Invalid fleet")
    
    # Compute expected signature
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{x_tg_timestamp}:{x_tg_nonce}:{body_hash}"
    
    # We need the raw API key, but we only store the hash
    # In production, agents would use the key they stored at creation
    # For verification, we compare the signature using the stored hash as key
    # This is a simplification - in production, use a proper key derivation
    expected_sig = hmac.new(
        fleet.api_key_hash.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_sig, x_tg_signature):
        logger.warning(f"Signature mismatch for fleet {x_tg_fleet_id}")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Check for nonce reuse (Replay attack prevention)
    statement = select(ReplayNonce).where(ReplayNonce.nonce == x_tg_nonce, ReplayNonce.fleet_id == x_tg_fleet_id)
    if session.exec(statement).first():
        logger.warning(f"Replay attack detected: Nonce reuse for fleet {x_tg_fleet_id}")
        raise HTTPException(status_code=401, detail="Invalid request (nonce reuse)")

    # Record nonce usage
    # Note: In production, nonces should be periodically purged based on expires_at
    new_nonce = ReplayNonce(
        nonce=x_tg_nonce,
        fleet_id=x_tg_fleet_id,
        timestamp=request_time,
        expires_at=datetime.utcnow() + timedelta(minutes=10)
    )
    session.add(new_nonce)
    session.commit()
    
    return fleet


# === Inventory Routes ===

@router.get("/inventory", response_model=Dict[str, Any])
async def get_inventory(
    fleet_id: Optional[str] = None,
    environment: Optional[str] = None,
    expiry_within_days: Optional[int] = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get certificate inventory for tenant."""
    inventory = InventoryService(session)
    
    # Get endpoints
    endpoints = inventory.list_endpoints(
        tenant_id=current_user.tenant_id,
        fleet_id=fleet_id,
        environment=environment,
    )
    
    # Get certificates
    certificates = inventory.list_certificates(
        tenant_id=current_user.tenant_id,
        fleet_id=fleet_id,
        expiry_within_days=expiry_within_days,
    )
    
    # Get expiry summary
    expiry_summary = inventory.get_expiry_summary(
        tenant_id=current_user.tenant_id,
        fleet_id=fleet_id,
    )
    
    return {
        "endpoints": [
            {
                "id": e.id,
                "name": e.name,
                "hostname": e.hostname,
                "endpoint_type": e.endpoint_type.value,
                "environment": e.environment,
                "criticality": e.criticality.value,
            }
            for e in endpoints
        ],
        "certificates": [
            {
                "id": c.id,
                "endpoint_id": c.endpoint_id,
                "subject": c.subject_dn,
                "issuer": c.issuer_dn,
                "not_after": c.not_after.isoformat(),
                "days_to_expiry": c.days_to_expiry,
                "has_eku_conflict": c.has_eku_conflict,
            }
            for c in certificates
        ],
        "expiry_summary": expiry_summary,
    }


@router.post("/endpoints", response_model=Dict[str, Any])
async def create_endpoint(
    data: EndpointCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Create a new endpoint."""
    # Verify fleet belongs to tenant
    fleet = session.get(Fleet, data.fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Fleet not found")
    
    inventory = InventoryService(session)
    
    endpoint = inventory.create_endpoint(
        tenant_id=current_user.tenant_id,
        fleet_id=data.fleet_id,
        name=data.name,
        hostname=data.hostname,
        port=data.port,
        endpoint_type=EndpointType(data.endpoint_type),
        environment=data.environment,
        criticality=Criticality(data.criticality),
        k8s_namespace=data.k8s_namespace,
        k8s_secret_name=data.k8s_secret_name,
        k8s_ingress_name=data.k8s_ingress_name,
        tags=data.tags,
    )
    
    return {"id": endpoint.id, "name": endpoint.name, "hostname": endpoint.hostname}


# === Scan Routes ===

@router.post("/scan/request", response_model=Dict[str, Any])
async def request_scan(
    fleet_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Request a certificate scan from agents in a fleet."""
    fleet = session.get(Fleet, fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Fleet not found")
    
    # In production: send scan request to agents via message queue
    # For now: return a scan job ID
    
    import uuid
    scan_id = str(uuid.uuid4())
    
    AuditService(session).log(
        tenant_id=current_user.tenant_id,
        fleet_id=fleet_id,
        action=AuditAction.ENDPOINT_DISCOVERED,
        actor_type="user",
        actor_id=current_user.id,
        payload={"scan_id": scan_id},
    )
    
    return {"scan_id": scan_id, "status": "requested", "fleet_id": fleet_id}


# === Policy Routes ===

@router.post("/policies", response_model=Dict[str, Any])
async def create_policy(
    data: PolicyCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Create or apply a certificate policy."""
    if data.preset_name:
        # Use preset
        policy = IdentityPolicy.create_preset(data.preset_name, current_user.tenant_id)
    else:
        # Custom policy
        policy = IdentityPolicy(
            tenant_id=current_user.tenant_id,
            name=data.name,
            description=data.description,
            max_validity_days=data.max_validity_days,
            renewal_window_days=data.renewal_window_days,
            allow_server_auth=data.allow_server_auth,
            allow_client_auth=data.allow_client_auth,
            require_eku_separation=data.require_eku_separation,
            require_public_trust=data.require_public_trust,
            acme_challenge_type=data.acme_challenge_type,
        )
    
    session.add(policy)
    session.commit()
    session.refresh(policy)
    
    AuditService(session).log(
        tenant_id=current_user.tenant_id,
        action=AuditAction.POLICY_CREATED,
        actor_type="user",
        actor_id=current_user.id,
        target_type="policy",
        target_id=policy.id,
        payload={"name": policy.name},
    )
    
    return {"id": policy.id, "name": policy.name}


@router.get("/policies", response_model=List[Dict[str, Any]])
async def list_policies(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """List all policies for tenant."""
    statement = select(IdentityPolicy).where(
        IdentityPolicy.tenant_id == current_user.tenant_id,
        IdentityPolicy.is_active == True
    )
    policies = session.exec(statement).all()
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "max_validity_days": p.max_validity_days,
            "renewal_window_days": p.renewal_window_days,
            "is_preset": p.is_preset,
            "preset_name": p.preset_name,
        }
        for p in policies
    ]


@router.get("/policies/{policy_id}", response_model=Dict[str, Any])
async def get_policy(
    policy_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get policy details."""
    policy = session.get(IdentityPolicy, policy_id)
    if not policy or policy.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Policy not found")
    
    return {
        "id": policy.id,
        "name": policy.name,
        "description": policy.description,
        "max_validity_days": policy.max_validity_days,
        "renewal_window_days": policy.renewal_window_days,
        "allow_server_auth": policy.allow_server_auth,
        "allow_client_auth": policy.allow_client_auth,
        "require_eku_separation": policy.require_eku_separation,
        "acme_challenge_type": policy.acme_challenge_type,
        "alert_days_critical": policy.alert_days_critical,
        "alert_days_warning": policy.alert_days_warning,
    }


# === Renewal Routes ===

@router.post("/renewals/run", response_model=Dict[str, Any])
async def run_renewals(
    data: RenewalRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Execute renewal for specified endpoints."""
    scheduler = RenewalScheduler(session)
    jobs = []
    
    for endpoint_id in data.endpoint_ids:
        endpoint = session.get(IdentityEndpoint, endpoint_id)
        if not endpoint or endpoint.tenant_id != current_user.tenant_id:
            continue
        
        job = scheduler.schedule_renewal(
            tenant_id=current_user.tenant_id,
            fleet_id=endpoint.fleet_id,
            endpoint_id=endpoint_id,
            policy_id=data.policy_id,
            scheduled_at=data.scheduled_at,
        )
        jobs.append({"job_id": job.id, "endpoint_id": endpoint_id, "status": job.status.value})
    
    return {"jobs": jobs, "scheduled_count": len(jobs)}


@router.get("/renewals", response_model=List[Dict[str, Any]])
async def list_renewals(
    fleet_id: Optional[str] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """List renewal jobs."""
    scheduler = RenewalScheduler(session)
    
    status_enum = RenewalJobStatus(status) if status else None
    jobs = scheduler.list_jobs(
        tenant_id=current_user.tenant_id,
        fleet_id=fleet_id,
        status=status_enum,
    )
    
    return [
        {
            "id": j.id,
            "endpoint_id": j.endpoint_id,
            "status": j.status.value,
            "status_message": j.status_message,
            "retry_count": j.retry_count,
            "created_at": j.created_at.isoformat(),
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        for j in jobs
    ]


@router.get("/renewals/{job_id}", response_model=Dict[str, Any])
async def get_renewal(
    job_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get renewal job details."""
    job = session.get(IdentityRenewalJob, job_id)
    if not job or job.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "id": job.id,
        "endpoint_id": job.endpoint_id,
        "policy_id": job.policy_id,
        "status": job.status.value,
        "status_message": job.status_message,
        "retry_count": job.retry_count,
        "max_retries": job.max_retries,
        "last_error": job.last_error,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "can_rollback": job.can_rollback,
    }


# === Migration Routes ===

@router.post("/migrations/eku-split", response_model=Dict[str, Any])
async def execute_eku_migration(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Execute EKU split migration for all conflicting certificates.
    
    Chrome Jun 2026: public certs cannot have both serverAuth + clientAuth.
    This executor:
    1. Finds violations
    2. For each, schedules a renewal for public TLS (serverAuth only)
    3. For each, schedules a renewal for private mTLS (clientAuth only)
    """
    inventory = InventoryService(session)
    scheduler = RenewalScheduler(session)
    violations = inventory.detect_eku_violations(current_user.tenant_id)
    
    # Get/Create mTLS policy
    mtls_policy = session.exec(
        select(IdentityPolicy).where(
            IdentityPolicy.tenant_id == current_user.tenant_id,
            IdentityPolicy.preset_name == "mtls"
        )
    ).first()
    
    if not mtls_policy:
        mtls_policy = IdentityPolicy.create_preset("mtls", current_user.tenant_id)
        session.add(mtls_policy)
        session.commit()
        session.refresh(mtls_policy)
    
    jobs_created = 0
    for v in violations:
        cert_id = v["certificate_id"]
        cert = session.get(IdentityCertificate, cert_id)
        if not cert:
             continue
             
        # 1. Schedule Public Renewal (serverAuth)
        # Assuming current policy is public
        scheduler.schedule_renewal(
            tenant_id=current_user.tenant_id,
            fleet_id=cert.endpoint.fleet_id,
            endpoint_id=cert.endpoint_id,
            policy_id=cert.policy_id
        )
        
        # 2. Schedule Private Renewal (clientAuth)
        scheduler.schedule_renewal(
            tenant_id=current_user.tenant_id,
            fleet_id=cert.endpoint.fleet_id,
            endpoint_id=cert.endpoint_id,
            policy_id=mtls_policy.id
        )
        jobs_created += 2
        
    AuditService(session).log(
        tenant_id=current_user.tenant_id,
        action=AuditAction.RENEWAL_STARTED,
        actor_type="user",
        actor_id=current_user.id,
        payload={"migration_type": "eku-split", "jobs_created": jobs_created},
    )
    
    return {
        "status": "migration_started",
        "violations_processed": len(violations),
        "jobs_created": jobs_created,
        "recommendation": "Update client trust anchors for the new private CA certificates."
    }


# === Risk Analysis ===

@router.get("/risk", response_model=Dict[str, Any])
async def get_risk_analysis(
    cert_id: Optional[str] = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get blast radius / risk analysis."""
    inventory = InventoryService(session)
    return inventory.get_risk_assessment(
        tenant_id=current_user.tenant_id,
        cert_id=cert_id,
    )


# === Audit Routes ===

@router.get("/audit", response_model=List[Dict[str, Any]])
async def get_audit_log(
    fleet_id: Optional[str] = None,
    action: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get audit log entries."""
    audit = AuditService(session)
    
    since_dt = datetime.fromisoformat(since) if since else None
    action_enum = AuditAction(action) if action else None
    
    entries = audit.get_entries(
        tenant_id=current_user.tenant_id,
        fleet_id=fleet_id,
        action=action_enum,
        since=since_dt,
        limit=limit,
    )
    
    return [
        {
            "id": e.id,
            "sequence": e.sequence_number,
            "action": e.action.value,
            "action_detail": e.action_detail,
            "actor_type": e.actor_type,
            "actor_id": e.actor_id,
            "target_type": e.target_type,
            "target_id": e.target_id,
            "timestamp": e.timestamp.isoformat(),
            "entry_hash": e.entry_hash[:16] + "...",
        }
        for e in entries
    ]


@router.get("/audit/verify", response_model=Dict[str, Any])
async def verify_audit_chain(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Verify audit log integrity."""
    audit = AuditService(session)
    return audit.verify_chain(current_user.tenant_id)


# === Agent Routes (Fleet API Key Auth) ===

@router.post("/agent/register", response_model=Dict[str, Any])
async def register_agent(
    data: AgentRegister,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Register a new identity agent."""
    fleet = session.get(Fleet, data.fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Fleet not found")
    
    inventory = InventoryService(session)
    agent = inventory.register_agent(
        tenant_id=current_user.tenant_id,
        fleet_id=data.fleet_id,
        name=data.name,
        hostname=data.hostname,
        supported_types=data.supported_types,
        supported_challenges=data.supported_challenges,
        public_key_pem=data.public_key_pem,
        version=data.version,
    )
    
    AuditService(session).log(
        tenant_id=current_user.tenant_id,
        fleet_id=data.fleet_id,
        action=AuditAction.AGENT_ENROLLED,
        actor_type="user",
        actor_id=current_user.id,
        target_type="agent",
        target_id=agent.id,
    )
    
    return {"agent_id": agent.id, "name": agent.name}


@router.get("/agent/jobs", response_model=List[Dict[str, Any]])
async def list_agent_jobs(
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """List pending jobs for a specific agent/fleet."""
    scheduler = RenewalScheduler(session)
    # Return jobs that need agent action
    jobs = scheduler.list_jobs(
        tenant_id=None, # Filter by fleet_id instead
        fleet_id=fleet.id,
        status=None # We'll filter in the list
    )
    
    agent_action_statuses = [
        RenewalJobStatus.PENDING,
        RenewalJobStatus.CSR_REQUESTED,
        RenewalJobStatus.CHALLENGE_PENDING,
        RenewalJobStatus.ISSUED,
        RenewalJobStatus.DEPLOYING
    ]
    
    return [
        {
            "id": j.id,
            "status": j.status.value,
            "endpoint_id": j.endpoint_id,
            "challenge_type": j.challenge_type,
            "challenge_token": j.challenge_token,
            "challenge_url": j.challenge_url,
            "issued_cert_pem": j.issued_cert_pem if j.status == RenewalJobStatus.ISSUED else None,
            "hostname": endpoint.hostname if (endpoint := session.get(IdentityEndpoint, j.endpoint_id)) else None,
            "endpoint": (
                {
                    "id": endpoint.id,
                    "hostname": endpoint.hostname,
                    "port": endpoint.port,
                    "endpoint_type": endpoint.endpoint_type.value,
                    "k8s_namespace": endpoint.k8s_namespace,
                    "k8s_secret_name": endpoint.k8s_secret_name,
                    "k8s_ingress_name": endpoint.k8s_ingress_name,
                }
                if (endpoint := session.get(IdentityEndpoint, j.endpoint_id))
                else None
            ),
        }
        for j in jobs if j.status in agent_action_statuses
    ]


@router.post("/agent/csr", response_model=Dict[str, Any])
async def submit_csr(
    data: AgentCSRSubmit,
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """Agent submits CSR for a renewal job."""
    scheduler = RenewalScheduler(session)
    
    job = scheduler.receive_csr(data.job_id, data.csr_pem)
    
    return {"job_id": job.id, "status": job.status.value}


@router.post("/agent/report", response_model=Dict[str, Any])
async def report_certificates(
    data: List[CertificateReport],
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """Agent reports discovered certificates."""
    inventory = InventoryService(session)
    
    for cert_data in data:
        inventory.register_certificate(
            endpoint_id="auto-discovered",
            tenant_id=fleet.tenant_id,
            fingerprint_sha256=cert_data.fingerprint_sha256,
            serial_number=cert_data.serial_number,
            subject_dn=cert_data.subject_dn,
            issuer_dn=cert_data.issuer_dn,
            sans=cert_data.sans,
            not_before=cert_data.not_before,
            not_after=cert_data.not_after,
            key_type=cert_data.key_type,
            key_size=cert_data.key_size,
            signature_algorithm=cert_data.signature_algorithm,
            eku_server_auth=cert_data.eku_server_auth,
            eku_client_auth=cert_data.eku_client_auth,
            is_public_trust=cert_data.is_public_trust,
        )
        
    return {"status": "ok", "reported_count": len(data)}


@router.post("/agent/challenge-complete", response_model=Dict[str, Any])
async def complete_challenge(
    data: AgentChallengeComplete,
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """Agent confirms challenge completion."""
    scheduler = RenewalScheduler(session)
    
    job = scheduler.complete_challenge(data.job_id, data.token)
    
    return {"job_id": job.id, "status": job.status.value}


@router.post("/agent/deploy-confirm", response_model=Dict[str, Any])
async def confirm_deploy(
    data: AgentDeployConfirm,
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """Agent confirms certificate deployment."""
    scheduler = RenewalScheduler(session)
    
    job = scheduler.confirm_deployment(data.job_id)
    
    return {"job_id": job.id, "status": job.status.value}


@router.post("/agent/heartbeat", response_model=Dict[str, str])
async def agent_heartbeat(
    agent_id: str,
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """Agent heartbeat to report status."""
    inventory = InventoryService(session)
    agent = inventory.update_agent_heartbeat(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {"status": "ok", "agent_id": agent_id}
