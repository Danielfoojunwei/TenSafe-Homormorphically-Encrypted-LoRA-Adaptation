"""
Identity Models - Machine Identity Guard Database Schema

Extends TensorGuardFlow with certificate lifecycle management tables.
All identity entities are scoped to Tenant + Fleet for multi-tenancy.
"""

from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from sqlalchemy import Index
from datetime import datetime
from enum import Enum
import uuid
import hashlib


# === Enums ===

class EndpointType(str, Enum):
    """Type of endpoint hosting certificates."""
    KUBERNETES = "kubernetes"
    NGINX = "nginx"
    ENVOY = "envoy"
    APACHE = "apache"
    HAPROXY = "haproxy"
    VDA5050 = "vda5050"
    OPEN_RMF = "open_rmf"
    CUSTOM = "custom"


class Criticality(str, Enum):
    """Business criticality level."""
    CRITICAL = "critical"  # Production, customer-facing
    HIGH = "high"          # Internal production
    MEDIUM = "medium"      # Staging, pre-prod
    LOW = "low"            # Dev, test


class CertificateType(str, Enum):
    """Certificate type/purpose."""
    PUBLIC_TLS = "public_tls"        # Public CA (Let's Encrypt, etc.)
    PRIVATE_TLS = "private_tls"      # Private CA for internal TLS
    MTLS_CLIENT = "mtls_client"      # Client auth (mTLS)
    MTLS_SERVER = "mtls_server"      # Server with client verification
    CODE_SIGNING = "code_signing"    # Code/artifact signing


class RenewalJobStatus(str, Enum):
    """Renewal job state machine."""
    PENDING = "pending"
    CSR_REQUESTED = "csr_requested"
    CSR_RECEIVED = "csr_received"
    CHALLENGE_PENDING = "challenge_pending"
    CHALLENGE_COMPLETE = "challenge_complete"
    ISSUING = "issuing"
    ISSUED = "issued"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class AuditAction(str, Enum):
    """Auditable actions in the identity system."""
    ENDPOINT_DISCOVERED = "endpoint_discovered"
    ENDPOINT_UPDATED = "endpoint_updated"
    CERT_DISCOVERED = "cert_discovered"
    CERT_EXPIRY_WARNING = "cert_expiry_warning"
    POLICY_CREATED = "policy_created"
    POLICY_UPDATED = "policy_updated"
    RENEWAL_STARTED = "renewal_started"
    RENEWAL_SUCCEEDED = "renewal_succeeded"
    RENEWAL_FAILED = "renewal_failed"
    RENEWAL_ROLLED_BACK = "renewal_rolled_back"
    CSR_GENERATED = "csr_generated"
    CHALLENGE_STARTED = "challenge_started"
    CHALLENGE_COMPLETED = "challenge_completed"
    CERT_DEPLOYED = "cert_deployed"
    EKU_VIOLATION_DETECTED = "eku_violation_detected"
    AGENT_ENROLLED = "agent_enrolled"
    AGENT_HEARTBEAT = "agent_heartbeat"


# === Models ===

class IdentityEndpoint(SQLModel, table=True):
    """
    A discoverable endpoint that hosts TLS certificates.

    Can be a K8s Ingress, Nginx server, Envoy listener, etc.
    Scoped to tenant + fleet for multi-tenancy.
    """
    __tablename__ = "identity_endpoints"
    __table_args__ = (
        Index('ix_endpoint_tenant_env_type', 'tenant_id', 'environment', 'endpoint_type'),
        Index('ix_endpoint_tenant_fleet_active', 'tenant_id', 'fleet_id', 'is_active'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    fleet_id: str = Field(index=True)
    
    # Identification
    name: str = Field(index=True)  # Human-readable name
    hostname: str = Field(index=True)  # FQDN or IP
    port: int = Field(default=443)
    endpoint_type: EndpointType = Field(default=EndpointType.CUSTOM)
    
    # K8s-specific (if applicable)
    k8s_namespace: Optional[str] = None
    k8s_secret_name: Optional[str] = None
    k8s_ingress_name: Optional[str] = None
    
    # Environment and criticality
    environment: str = Field(default="production", index=True)  # production, staging, dev
    criticality: Criticality = Field(default=Criticality.MEDIUM)
    tags: Optional[str] = Field(default=None)  # JSON-encoded tags
    
    # Agent association
    agent_id: Optional[str] = None  # Which agent manages this endpoint
    last_scan_at: Optional[datetime] = None
    
    # Lifecycle
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    certificates: List["IdentityCertificate"] = Relationship(back_populates="endpoint")


class IdentityCertificate(SQLModel, table=True):
    """
    A discovered or managed TLS/mTLS certificate.

    Stores metadata only - private keys are NEVER stored in the control plane.
    """
    __tablename__ = "identity_certificates"
    __table_args__ = (
        Index('ix_cert_tenant_current_expiry', 'tenant_id', 'is_current', 'not_after'),
        Index('ix_cert_endpoint_type', 'endpoint_id', 'certificate_type'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    endpoint_id: str = Field(foreign_key="identity_endpoints.id", index=True)
    tenant_id: str = Field(index=True)
    
    # Certificate identity
    fingerprint_sha256: str = Field(unique=True, index=True)  # SHA-256 of DER
    serial_number: str
    subject_dn: str  # Distinguished Name
    issuer_dn: str
    
    # SANs (Subject Alternative Names) - stored as JSON array
    sans_json: str  # ["example.com", "*.example.com", "192.168.1.1"]
    
    # Validity
    not_before: datetime
    not_after: datetime = Field(index=True)  # Critical for expiry tracking
    
    # Key and signature info
    key_type: str = Field(default="RSA")  # RSA, ECDSA, Ed25519
    key_size: int = Field(default=2048)  # bits for RSA, curve size for EC
    signature_algorithm: str  # SHA256WithRSA, etc.
    
    # Extended Key Usage
    eku_server_auth: bool = Field(default=False)
    eku_client_auth: bool = Field(default=False)
    eku_other: Optional[str] = None  # Other EKUs as JSON
    
    # Trust and type
    certificate_type: CertificateType = Field(default=CertificateType.PUBLIC_TLS)
    is_public_trust: bool = Field(default=True)  # Publicly trusted CA chain
    is_self_signed: bool = Field(default=False)
    
    # Chain info
    chain_depth: int = Field(default=0)  # 0 = leaf, 1+ = intermediates
    issuer_fingerprint: Optional[str] = None  # Link to issuer cert
    
    # Policy linkage
    policy_id: Optional[str] = Field(default=None, foreign_key="identity_policies.id")
    
    # Status
    is_current: bool = Field(default=True)  # Currently deployed (vs. historical)
    renewal_job_id: Optional[str] = None  # If being renewed
    
    # Lifecycle
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    endpoint: IdentityEndpoint = Relationship(back_populates="certificates")
    policy: Optional["IdentityPolicy"] = Relationship(back_populates="certificates")
    
    @property
    def days_to_expiry(self) -> int:
        """Days until certificate expires."""
        delta = self.not_after - datetime.utcnow()
        return max(0, delta.days)

    @property
    def has_eku_conflict(self) -> bool:
        """
        Conflict: public cert (serverAuth) also has clientAuth.
        Chrome 2026: public certs must be single-use.
        """
        return self.is_public_trust and self.eku_server_auth and self.eku_client_auth
    
    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.not_after


class IdentityPolicy(SQLModel, table=True):
    """
    Certificate lifecycle policy defining renewal rules and constraints.
    
    Supports 47-day, 100-day, 200-day presets plus custom configurations.
    """
    __tablename__ = "identity_policies"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Identity
    name: str = Field(index=True)
    description: Optional[str] = None
    
    # === Validity Rules ===
    max_validity_days: int = Field(default=90)  # Max cert lifetime
    renewal_window_days: int = Field(default=30)  # Renew X days before expiry
    min_remaining_days: int = Field(default=7)   # Alert if below this
    
    # === EKU Rules ===
    allow_server_auth: bool = Field(default=True)
    allow_client_auth: bool = Field(default=False)  # Separate by default
    require_eku_separation: bool = Field(default=True)  # No serverAuth+clientAuth together for public
    
    # === CA Constraints ===
    allowed_issuers_json: Optional[str] = None  # JSON list of allowed issuer DNs
    require_public_trust: bool = Field(default=True)
    preferred_acme_provider: str = Field(default="letsencrypt")  # letsencrypt, zerossl, buypass
    
    # === Algorithm Constraints ===
    min_key_size_rsa: int = Field(default=2048)
    min_key_size_ec: int = Field(default=256)
    allowed_key_types_json: str = Field(default='["RSA", "ECDSA"]')
    allowed_sig_algs_json: str = Field(default='["SHA256", "SHA384", "SHA512"]')
    
    # === ACME / DCV Settings ===
    acme_challenge_type: str = Field(default="http-01")  # http-01, dns-01
    dcv_reuse_window_hours: int = Field(default=720)  # 30 days default
    force_fresh_dcv: bool = Field(default=False)  # Always fresh challenge
    
    # === Alert Thresholds ===
    alert_days_critical: int = Field(default=7)
    alert_days_warning: int = Field(default=30)
    alert_days_info: int = Field(default=60)
    
    # === Presets ===
    is_preset: bool = Field(default=False)
    preset_name: Optional[str] = None  # "47-day", "100-day", "200-day", "mtls"
    
    # Lifecycle
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    certificates: List[IdentityCertificate] = Relationship(back_populates="policy")
    
    @classmethod
    def create_preset(cls, preset_name: str, tenant_id: str) -> "IdentityPolicy":
        """Create a policy from a preset."""
        presets = {
            "47-day": {
                "name": "47-Day Public TLS (Mar 2029+)",
                "max_validity_days": 47,
                "renewal_window_days": 10,
                "min_remaining_days": 3,
                "allow_client_auth": False,
                "require_eku_separation": True,
            },
            "100-day": {
                "name": "100-Day Public TLS (Mar 2027+)",
                "max_validity_days": 100,
                "renewal_window_days": 20,
                "min_remaining_days": 5,
                "allow_client_auth": False,
                "require_eku_separation": True,
            },
            "200-day": {
                "name": "200-Day Public TLS (Mar 2026+)",
                "max_validity_days": 200,
                "renewal_window_days": 30,
                "min_remaining_days": 7,
                "allow_client_auth": False,
                "require_eku_separation": True,
            },
            "mtls": {
                "name": "mTLS Client Auth (Private CA)",
                "max_validity_days": 365,
                "renewal_window_days": 30,
                "min_remaining_days": 7,
                "allow_server_auth": False,
                "allow_client_auth": True,
                "require_public_trust": False,
                "require_eku_separation": False,
            },
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        config = presets[preset_name]
        return cls(
            tenant_id=tenant_id,
            is_preset=True,
            preset_name=preset_name,
            **config
        )


class IdentityRenewalJob(SQLModel, table=True):
    """
    State machine for certificate renewal workflows.

    Tracks the full lifecycle: CSR → Challenge → Issue → Deploy → Validate.
    Supports rollback on failure.
    """
    __tablename__ = "identity_renewal_jobs"
    __table_args__ = (
        Index('ix_renewal_tenant_status', 'tenant_id', 'status'),
        Index('ix_renewal_endpoint_status', 'endpoint_id', 'status'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    fleet_id: str = Field(index=True)
    
    # Target
    endpoint_id: str = Field(foreign_key="identity_endpoints.id")
    old_cert_id: Optional[str] = None  # The cert being replaced
    new_cert_id: Optional[str] = None  # The new cert after issuance
    policy_id: str = Field(foreign_key="identity_policies.id")
    
    # State machine
    status: RenewalJobStatus = Field(default=RenewalJobStatus.PENDING, index=True)
    status_message: Optional[str] = None
    
    # Progress tracking
    csr_pem: Optional[str] = None  # CSR from agent (no private key!)
    challenge_token: Optional[str] = None
    challenge_type: Optional[str] = None
    issued_cert_pem: Optional[str] = None  # Full chain after issuance
    
    # ACME persistence
    acme_order_url: Optional[str] = None
    acme_finalize_url: Optional[str] = None
    acme_authz_urls_json: Optional[str] = None
    acme_cert_url: Optional[str] = None
    challenge_url: Optional[str] = None
    challenge_domain: Optional[str] = None
    last_acme_status: Optional[str] = None
    
    # Retry logic
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    last_error: Optional[str] = None
    
    # Scheduling
    scheduled_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    
    # Rollback info
    rollback_cert_id: Optional[str] = None  # Cert to restore on rollback
    can_rollback: bool = Field(default=True)
    deployment_snapshot: Optional[str] = Field(default=None, sa_column=Column(JSON))
    
    # Idempotency
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [
            RenewalJobStatus.SUCCEEDED,
            RenewalJobStatus.FAILED,
            RenewalJobStatus.ROLLED_BACK
        ]
    
    @property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and not self.is_terminal


class IdentityAuditLog(SQLModel, table=True):
    """
    Tamper-evident audit log with hash chaining.
    
    Every entry includes prev_hash and entry_hash for integrity verification.
    Immutable after creation - append-only.
    """
    __tablename__ = "identity_audit_log"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    sequence_number: int = Field(index=True)  # Monotonic sequence within tenant
    tenant_id: str = Field(index=True)
    fleet_id: Optional[str] = Field(default=None, index=True)
    
    # Actor
    actor_type: str  # "user", "agent", "system", "scheduler"
    actor_id: str    # User ID, Agent ID, or "system"
    actor_ip: Optional[str] = None
    
    # Action
    action: AuditAction
    action_detail: Optional[str] = None
    
    # Target
    target_type: Optional[str] = None  # "endpoint", "certificate", "policy", "job"
    target_id: Optional[str] = None
    
    # Payload
    payload_json: Optional[str] = None  # Action-specific data (sanitized)
    payload_hash: str  # SHA-256 of payload for integrity
    
    # Evidence
    evidence_uri: Optional[str] = None  # Link to evidence pack (CSR, cert, etc.)
    
    # Hash chain for tamper-evidence
    prev_hash: str  # Hash of previous entry (or "GENESIS" for first)
    entry_hash: str  # SHA-256(prev_hash + action + payload_hash + timestamp)
    pqc_signature: Optional[str] = None # Dilithium-3 hex signature
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    @classmethod
    def compute_entry_hash(
        cls,
        prev_hash: str,
        action: str,
        payload_hash: str,
        timestamp: datetime
    ) -> str:
        """Compute the tamper-evident hash for this entry."""
        data = f"{prev_hash}:{action}:{payload_hash}:{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    @classmethod
    def compute_payload_hash(cls, payload: dict) -> str:
        """Compute hash of the payload for integrity."""
        import json
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()


class IdentityAgent(SQLModel, table=True):
    """
    Registered identity agent for certificate management.
    
    Agents run on fleet nodes and handle local key generation,
    challenge responses, and certificate deployment.
    """
    __tablename__ = "identity_agents"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    fleet_id: str = Field(foreign_key="fleet.id", index=True)
    
    # Identity
    name: str
    hostname: str = Field(index=True)
    
    # Enrollment
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)
    enrollment_token_hash: Optional[str] = None  # One-time enrollment token
    
    # Authentication
    # Agent uses fleet API key for auth, with additional request signing
    public_key_pem: Optional[str] = None  # Agent's signing public key
    
    # Capabilities
    supported_types_json: str = Field(default='["kubernetes", "nginx", "envoy"]')
    supported_challenges_json: str = Field(default='["http-01"]')
    has_pkcs11: bool = Field(default=False)
    has_tpm: bool = Field(default=False)
    
    # Status
    is_active: bool = Field(default=True)
    last_heartbeat_at: Optional[datetime] = None
    last_scan_at: Optional[datetime] = None
    version: Optional[str] = None
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def is_online(self) -> bool:
        """Check if agent has reported within last 5 minutes."""
        if not self.last_heartbeat_at:
            return False
        delta = datetime.utcnow() - self.last_heartbeat_at
        return delta.total_seconds() < 300
