
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import time
import uuid

class EventType(str, Enum):
    # Benchmark / System
    BENCH_REPORT_GENERATED = "BENCH_REPORT_GENERATED"
    
    # TGSP Lifecycle
    TGSP_BUILT = "TGSP_BUILT"
    TGSP_VERIFY_SUCCEEDED = "TGSP_VERIFY_SUCCEEDED"
    TGSP_VERIFY_FAILED = "TGSP_VERIFY_FAILED"
    TGSP_OPEN_ALLOWED = "TGSP_OPEN_ALLOWED"
    TGSP_OPEN_DENIED = "TGSP_OPEN_DENIED"
    TGSP_DEK_UNWRAP_SUCCEEDED = "TGSP_DEK_UNWRAP_SUCCEEDED"
    TGSP_DEK_UNWRAP_FAILED = "TGSP_DEK_UNWRAP_FAILED"
    
    # Attestation
    ATTESTATION_SUBMITTED = "ATTESTATION_SUBMITTED"
    ATTESTATION_VERIFIED = "ATTESTATION_VERIFIED"
    KEY_RELEASE_GRANTED = "KEY_RELEASE_GRANTED"
    KEY_RELEASE_DENIED = "KEY_RELEASE_DENIED"
    
    # Platform
    POLICY_CHANGED = "POLICY_CHANGED"
    
    # Privacy / N2HE Events
    PRIVACY_MODE_ENABLED = "PRIVACY_MODE_ENABLED"
    PRIVACY_MODE_DISABLED = "PRIVACY_MODE_DISABLED"
    N2HE_ROUTING_DECISION = "N2HE_ROUTING_DECISION"
    N2HE_ROUTING_DECISION_HASH = "N2HE_ROUTING_DECISION_HASH"
    N2HE_PROVIDER_VERSION = "N2HE_PROVIDER_VERSION"
    N2HE_SCHEME_PROFILE_HASH = "N2HE_SCHEME_PROFILE_HASH"
    N2HE_EVAL_COMPLETED = "N2HE_EVAL_COMPLETED"
    PRIVACY_RECEIPT_GENERATED = "PRIVACY_RECEIPT_GENERATED"
    
    # Adapter Lifecycle
    ADAPTER_REGISTERED = "ADAPTER_REGISTERED"
    ADAPTER_PROMOTED = "ADAPTER_PROMOTED"
    ADAPTER_ROLLBACK = "ADAPTER_ROLLBACK"
    ADAPTER_ARCHIVED = "ADAPTER_ARCHIVED"
    
class EvidenceSubject(BaseModel):
    tgsp_ref: Optional[str] = None
    model_ref: Optional[str] = None
    round_id: Optional[str] = None
    policy_id: Optional[str] = None
    
class EvidenceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    
    tenant_id: str = "default"
    fleet_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    subject: Dict[str, Any] = Field(default_factory=dict)
    
    # Hashes for Immutable Context
    manifest_hash: Optional[str] = None
    recipients_hash: Optional[str] = None # or meta_hash for v0.3
    payload_hash: Optional[str] = None
    policy_hash: Optional[str] = None
    claims_hash: Optional[str] = None
    attestation_id: Optional[str] = None
    
    result: Dict[str, Any] = Field(default_factory=dict) # {status: "allow", reason: "..."}
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, str]] = Field(default_factory=list) # [{name, sha256}]
    
    signature: Optional[Dict[str, str]] = None # {key_id, alg, sig}
