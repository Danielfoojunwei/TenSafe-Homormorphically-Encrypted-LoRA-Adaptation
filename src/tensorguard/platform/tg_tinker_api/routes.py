"""
TG-Tinker API routes.

FastAPI routers for the TG-Tinker training API.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from .audit import get_audit_logger
from .dp import DPConfig, DPTrainer
from .models import (
    TinkerArtifact,
    TinkerFuture,
    TinkerTrainingClient,
    generate_future_id,
    generate_tc_id,
)
from .queue import get_job_queue
from .storage import EncryptedArtifactStore, KeyManager, LocalStorageBackend
from .worker import get_worker, start_worker

logger = logging.getLogger(__name__)

# Initialize routers
router = APIRouter(prefix="/v1", tags=["tg-tinker"])

# In-memory storage for demo (in production, use database)
_training_clients: Dict[str, TinkerTrainingClient] = {}
_futures: Dict[str, TinkerFuture] = {}
_artifacts: Dict[str, TinkerArtifact] = {}
_dp_trainers: Dict[str, DPTrainer] = {}

# Initialize storage
_key_manager = KeyManager()
_storage_backend = LocalStorageBackend()
_artifact_store = EncryptedArtifactStore(_storage_backend, _key_manager)


# ==============================================================================
# Request/Response Models
# ==============================================================================


class LoRAConfigModel(BaseModel):
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    bias: str = "none"


class OptimizerConfigModel(BaseModel):
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


class DPConfigModel(BaseModel):
    enabled: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_epsilon: Optional[float] = 8.0
    target_delta: Optional[float] = 1e-5
    accountant_type: str = "rdp"


class CreateTrainingClientRequest(BaseModel):
    model_ref: str
    lora_config: Optional[LoRAConfigModel] = None
    optimizer: OptimizerConfigModel = Field(default_factory=OptimizerConfigModel)
    dp_config: Optional[DPConfigModel] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingClientResponse(BaseModel):
    training_client_id: str
    tenant_id: str
    model_ref: str
    status: str
    step: int
    created_at: datetime
    config: Dict[str, Any]
    dp_metrics: Optional[Dict[str, Any]] = None


class BatchDataModel(BaseModel):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: Optional[List[List[int]]] = None


class ForwardBackwardRequest(BaseModel):
    batch: BatchDataModel
    batch_hash: Optional[str] = None


class OptimStepRequest(BaseModel):
    apply_dp_noise: bool = True


class SampleRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = Field(default_factory=list)


class SaveStateRequest(BaseModel):
    include_optimizer: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoadStateRequest(BaseModel):
    artifact_id: str


class FutureResponse(BaseModel):
    future_id: str
    status: str
    created_at: datetime
    training_client_id: str
    operation: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class FutureResultResponse(BaseModel):
    future_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SampleCompletionModel(BaseModel):
    prompt: str
    completion: str
    tokens_generated: int
    finish_reason: str


class SampleResultResponse(BaseModel):
    samples: List[SampleCompletionModel]
    model_step: int
    sampling_config: Dict[str, Any]


class EncryptionInfoModel(BaseModel):
    algorithm: str
    key_id: str


class SaveStateResponse(BaseModel):
    artifact_id: str
    artifact_type: str
    size_bytes: int
    encryption: EncryptionInfoModel
    content_hash: str
    metadata: Dict[str, Any]
    created_at: datetime
    dp_metrics: Optional[Dict[str, Any]] = None


class LoadStateResponse(BaseModel):
    training_client_id: str
    loaded_artifact_id: str
    step: int
    status: str


class AuditLogEntryResponse(BaseModel):
    entry_id: str
    tenant_id: str
    training_client_id: str
    operation: str
    request_hash: str
    request_size_bytes: int
    artifact_ids_produced: List[str]
    artifact_ids_consumed: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    prev_hash: str
    record_hash: str
    dp_metrics: Optional[Dict[str, Any]]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ==============================================================================
# Dependency: Get tenant from API key
# ==============================================================================


async def get_tenant_id(
    authorization: str = Header(..., description="Bearer token"),
) -> str:
    """Extract tenant ID from authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "AUTHENTICATION_REQUIRED", "message": "Invalid authorization header"}},
        )

    token = authorization[7:]

    # In production, validate token and extract tenant
    # For demo, derive tenant from token hash
    tenant_id = f"tenant-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
    return tenant_id


# ==============================================================================
# Training Client Endpoints
# ==============================================================================


@router.post(
    "/training_clients",
    response_model=TrainingClientResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_training_client(
    request: CreateTrainingClientRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> TrainingClientResponse:
    """Create a new training client."""
    # Start worker if not running
    start_worker()

    tc_id = generate_tc_id()

    # Build config dict
    config_dict = {
        "model_ref": request.model_ref,
        "lora_config": request.lora_config.model_dump() if request.lora_config else None,
        "optimizer": request.optimizer.model_dump(),
        "dp_config": request.dp_config.model_dump() if request.dp_config else None,
        "batch_size": request.batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "max_steps": request.max_steps,
        "metadata": request.metadata,
    }

    # Create training client
    tc = TinkerTrainingClient(
        id=tc_id,
        tenant_id=tenant_id,
        model_ref=request.model_ref,
        status="ready",
        step=0,
        config_json=config_dict,
        dp_enabled=request.dp_config is not None and request.dp_config.enabled,
    )

    _training_clients[tc_id] = tc

    # Initialize ML backend
    worker = get_worker()
    worker.ml_backend.initialize_model(tc_id, request.model_ref, config_dict)

    # Initialize DP trainer if needed
    if request.dp_config and request.dp_config.enabled:
        dp_config = DPConfig(
            enabled=request.dp_config.enabled,
            noise_multiplier=request.dp_config.noise_multiplier,
            max_grad_norm=request.dp_config.max_grad_norm,
            target_epsilon=request.dp_config.target_epsilon,
            target_delta=request.dp_config.target_delta,
            accountant_type=request.dp_config.accountant_type,
        )
        _dp_trainers[tc_id] = DPTrainer(dp_config)

    # Log to audit
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id=tc_id,
        operation="create_training_client",
        request_hash=f"sha256:{hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()}",
        request_size_bytes=len(json.dumps(config_dict)),
        success=True,
    )

    return TrainingClientResponse(
        training_client_id=tc.id,
        tenant_id=tc.tenant_id,
        model_ref=tc.model_ref,
        status=tc.status,
        step=tc.step,
        created_at=tc.created_at,
        config=config_dict,
    )


@router.get("/training_clients", response_model=List[TrainingClientResponse])
async def list_training_clients(
    tenant_id: str = Depends(get_tenant_id),
) -> List[TrainingClientResponse]:
    """List all training clients for the tenant."""
    clients = [tc for tc in _training_clients.values() if tc.tenant_id == tenant_id]

    return [
        TrainingClientResponse(
            training_client_id=tc.id,
            tenant_id=tc.tenant_id,
            model_ref=tc.model_ref,
            status=tc.status,
            step=tc.step,
            created_at=tc.created_at,
            config=tc.config_json,
        )
        for tc in clients
    ]


@router.get("/training_clients/{tc_id}", response_model=TrainingClientResponse)
async def get_training_client(
    tc_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> TrainingClientResponse:
    """Get a training client by ID."""
    tc = _training_clients.get(tc_id)
    if tc is None or tc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "TRAINING_CLIENT_NOT_FOUND",
                    "message": f"Training client '{tc_id}' not found",
                    "details": {"training_client_id": tc_id},
                }
            },
        )

    dp_metrics = None
    if tc_id in _dp_trainers:
        eps, delta = _dp_trainers[tc_id].get_privacy_spent()
        dp_metrics = {
            "total_epsilon": eps,
            "delta": delta,
            "num_steps": _dp_trainers[tc_id].state.num_steps,
        }

    return TrainingClientResponse(
        training_client_id=tc.id,
        tenant_id=tc.tenant_id,
        model_ref=tc.model_ref,
        status=tc.status,
        step=tc.step,
        created_at=tc.created_at,
        config=tc.config_json,
        dp_metrics=dp_metrics,
    )


# ==============================================================================
# Training Primitive Endpoints
# ==============================================================================


@router.post(
    "/training_clients/{tc_id}/forward_backward",
    response_model=FutureResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def forward_backward(
    tc_id: str,
    request: ForwardBackwardRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> FutureResponse:
    """Queue a forward-backward pass."""
    tc = _training_clients.get(tc_id)
    if tc is None or tc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "TRAINING_CLIENT_NOT_FOUND", "message": f"Training client '{tc_id}' not found"}},
        )

    # Create future
    future_id = generate_future_id()
    payload = {
        "batch": request.batch.model_dump(),
        "dp_config": tc.config_json.get("dp_config"),
    }
    request_json = json.dumps(payload, sort_keys=True)
    request_hash = f"sha256:{hashlib.sha256(request_json.encode()).hexdigest()}"

    future = TinkerFuture(
        id=future_id,
        training_client_id=tc_id,
        tenant_id=tenant_id,
        operation="forward_backward",
        status="pending",
        request_hash=request_hash,
        request_size_bytes=len(request_json),
    )
    _futures[future_id] = future

    # Submit job to queue
    queue = get_job_queue()
    queue.submit(
        job_id=future_id,
        tenant_id=tenant_id,
        training_client_id=tc_id,
        operation="forward_backward",
        payload=payload,
    )

    return FutureResponse(
        future_id=future.id,
        status=future.status,
        created_at=future.created_at,
        training_client_id=future.training_client_id,
        operation=future.operation,
    )


@router.post(
    "/training_clients/{tc_id}/optim_step",
    response_model=FutureResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def optim_step(
    tc_id: str,
    request: OptimStepRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> FutureResponse:
    """Queue an optimizer step."""
    tc = _training_clients.get(tc_id)
    if tc is None or tc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "TRAINING_CLIENT_NOT_FOUND", "message": f"Training client '{tc_id}' not found"}},
        )

    # Create future
    future_id = generate_future_id()
    payload = {
        "apply_dp_noise": request.apply_dp_noise,
        "dp_config": tc.config_json.get("dp_config"),
    }
    request_json = json.dumps(payload, sort_keys=True)
    request_hash = f"sha256:{hashlib.sha256(request_json.encode()).hexdigest()}"

    future = TinkerFuture(
        id=future_id,
        training_client_id=tc_id,
        tenant_id=tenant_id,
        operation="optim_step",
        status="pending",
        request_hash=request_hash,
        request_size_bytes=len(request_json),
    )
    _futures[future_id] = future

    # Submit job to queue
    queue = get_job_queue()
    queue.submit(
        job_id=future_id,
        tenant_id=tenant_id,
        training_client_id=tc_id,
        operation="optim_step",
        payload=payload,
    )

    return FutureResponse(
        future_id=future.id,
        status=future.status,
        created_at=future.created_at,
        training_client_id=future.training_client_id,
        operation=future.operation,
    )


@router.post(
    "/training_clients/{tc_id}/sample",
    response_model=SampleResultResponse,
)
async def sample(
    tc_id: str,
    request: SampleRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> SampleResultResponse:
    """Generate samples from the model (synchronous)."""
    tc = _training_clients.get(tc_id)
    if tc is None or tc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "TRAINING_CLIENT_NOT_FOUND", "message": f"Training client '{tc_id}' not found"}},
        )

    # Execute sampling directly (synchronous)
    worker = get_worker()
    result = worker.ml_backend.sample(
        training_client_id=tc_id,
        prompts=request.prompts,
        config={
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop_sequences": request.stop_sequences,
        },
    )

    # Log to audit
    request_json = json.dumps(request.model_dump(), sort_keys=True)
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id=tc_id,
        operation="sample",
        request_hash=f"sha256:{hashlib.sha256(request_json.encode()).hexdigest()}",
        request_size_bytes=len(request_json),
        success=True,
    )

    return SampleResultResponse(
        samples=[
            SampleCompletionModel(
                prompt=s["prompt"],
                completion=s["completion"],
                tokens_generated=s["tokens_generated"],
                finish_reason=s["finish_reason"],
            )
            for s in result["samples"]
        ],
        model_step=result["model_step"],
        sampling_config=result["sampling_config"],
    )


@router.post(
    "/training_clients/{tc_id}/save_state",
    response_model=SaveStateResponse,
)
async def save_state(
    tc_id: str,
    request: SaveStateRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> SaveStateResponse:
    """Save training state as encrypted checkpoint."""
    tc = _training_clients.get(tc_id)
    if tc is None or tc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "TRAINING_CLIENT_NOT_FOUND", "message": f"Training client '{tc_id}' not found"}},
        )

    # Get state from ML backend
    worker = get_worker()
    state_bytes = worker.ml_backend.save_state(
        training_client_id=tc_id,
        include_optimizer=request.include_optimizer,
    )

    # Add DP metrics to metadata if available
    metadata = dict(request.metadata)
    dp_metrics = None
    if tc_id in _dp_trainers:
        eps, delta = _dp_trainers[tc_id].get_privacy_spent()
        dp_metrics = {
            "total_epsilon": eps,
            "delta": delta,
            "num_steps": _dp_trainers[tc_id].state.num_steps,
        }
        metadata["dp_metrics"] = dp_metrics

    # Encrypt and store
    artifact = _artifact_store.save_artifact(
        data=state_bytes,
        tenant_id=tenant_id,
        training_client_id=tc_id,
        artifact_type="checkpoint",
        metadata=metadata,
    )

    _artifacts[artifact.id] = artifact

    # Log to audit
    request_json = json.dumps(request.model_dump(), sort_keys=True)
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id=tc_id,
        operation="save_state",
        request_hash=f"sha256:{hashlib.sha256(request_json.encode()).hexdigest()}",
        request_size_bytes=len(request_json),
        artifact_ids_produced=[artifact.id],
        success=True,
        dp_metrics=dp_metrics,
    )

    return SaveStateResponse(
        artifact_id=artifact.id,
        artifact_type=artifact.artifact_type,
        size_bytes=artifact.size_bytes,
        encryption=EncryptionInfoModel(
            algorithm=artifact.encryption_algorithm,
            key_id=artifact.encryption_key_id,
        ),
        content_hash=artifact.content_hash,
        metadata=artifact.metadata_json,
        created_at=artifact.created_at,
        dp_metrics=dp_metrics,
    )


@router.post(
    "/training_clients/{tc_id}/load_state",
    response_model=LoadStateResponse,
)
async def load_state(
    tc_id: str,
    request: LoadStateRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> LoadStateResponse:
    """Load training state from encrypted checkpoint."""
    tc = _training_clients.get(tc_id)
    if tc is None or tc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "TRAINING_CLIENT_NOT_FOUND", "message": f"Training client '{tc_id}' not found"}},
        )

    artifact = _artifacts.get(request.artifact_id)
    if artifact is None or artifact.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "ARTIFACT_NOT_FOUND", "message": f"Artifact '{request.artifact_id}' not found"}},
        )

    # Decrypt and load
    state_bytes = _artifact_store.load_artifact(artifact)

    # Load into ML backend
    worker = get_worker()
    step = worker.ml_backend.load_state(tc_id, state_bytes)

    # Update training client
    tc.step = step
    tc.updated_at = datetime.utcnow()

    # Log to audit
    request_json = json.dumps(request.model_dump(), sort_keys=True)
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id=tc_id,
        operation="load_state",
        request_hash=f"sha256:{hashlib.sha256(request_json.encode()).hexdigest()}",
        request_size_bytes=len(request_json),
        artifact_ids_consumed=[artifact.id],
        success=True,
    )

    return LoadStateResponse(
        training_client_id=tc_id,
        loaded_artifact_id=artifact.id,
        step=step,
        status=tc.status,
    )


# ==============================================================================
# Future Endpoints
# ==============================================================================


@router.get("/futures/{future_id}", response_model=FutureResponse)
async def get_future(
    future_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> FutureResponse:
    """Get future status."""
    future = _futures.get(future_id)
    if future is None or future.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "FUTURE_NOT_FOUND", "message": f"Future '{future_id}' not found"}},
        )

    # Sync status from queue
    queue = get_job_queue()
    job = queue.get_status(future_id)
    if job:
        future.status = job.status.value
        future.started_at = job.started_at
        future.completed_at = job.completed_at
        if job.result:
            future.result_json = job.result
        if job.error:
            future.error_message = job.error

    return FutureResponse(
        future_id=future.id,
        status=future.status,
        created_at=future.created_at,
        training_client_id=future.training_client_id,
        operation=future.operation,
        started_at=future.started_at,
        completed_at=future.completed_at,
    )


@router.get("/futures/{future_id}/result", response_model=FutureResultResponse)
async def get_future_result(
    future_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> FutureResultResponse:
    """Get future result."""
    future = _futures.get(future_id)
    if future is None or future.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "FUTURE_NOT_FOUND", "message": f"Future '{future_id}' not found"}},
        )

    # Sync status from queue
    queue = get_job_queue()
    job = queue.get_status(future_id)
    if job:
        future.status = job.status.value
        future.started_at = job.started_at
        future.completed_at = job.completed_at
        if job.result:
            future.result_json = job.result
        if job.error:
            future.error_message = job.error

    if future.status not in ("completed", "failed"):
        return FutureResultResponse(
            future_id=future.id,
            status=future.status,
            result=None,
            error=None,
        )

    # Update training client step if optim_step completed
    if future.status == "completed" and future.operation == "optim_step":
        tc = _training_clients.get(future.training_client_id)
        if tc and future.result_json:
            tc.step = future.result_json.get("step", tc.step)
            tc.updated_at = datetime.utcnow()

    # Log to audit
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id=future.training_client_id,
        operation=future.operation,
        request_hash=future.request_hash,
        request_size_bytes=future.request_size_bytes,
        success=future.status == "completed",
        error_message=future.error_message,
        started_at=future.started_at,
        completed_at=future.completed_at,
        dp_metrics=future.result_json.get("dp_metrics") if future.result_json else None,
    )

    return FutureResultResponse(
        future_id=future.id,
        status=future.status,
        result=future.result_json,
        error=future.error_message,
    )


@router.post("/futures/{future_id}/cancel")
async def cancel_future(
    future_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> Dict[str, Any]:
    """Cancel a pending future."""
    future = _futures.get(future_id)
    if future is None or future.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "FUTURE_NOT_FOUND", "message": f"Future '{future_id}' not found"}},
        )

    queue = get_job_queue()
    success = queue.cancel(future_id)

    if success:
        future.status = "cancelled"
        future.completed_at = datetime.utcnow()

    return {"success": success}


# ==============================================================================
# Audit Log Endpoints
# ==============================================================================


@router.get("/audit_logs", response_model=List[AuditLogEntryResponse])
async def get_audit_logs(
    training_client_id: Optional[str] = Query(None),
    operation: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    tenant_id: str = Depends(get_tenant_id),
) -> List[AuditLogEntryResponse]:
    """Retrieve audit logs."""
    audit = get_audit_logger()
    logs = audit.get_logs(
        tenant_id=tenant_id,
        training_client_id=training_client_id,
        operation=operation,
        limit=limit,
        offset=offset,
    )

    return [
        AuditLogEntryResponse(
            entry_id=log.id,
            tenant_id=log.tenant_id,
            training_client_id=log.training_client_id,
            operation=log.operation,
            request_hash=log.request_hash,
            request_size_bytes=log.request_size_bytes,
            artifact_ids_produced=log.artifact_ids_produced,
            artifact_ids_consumed=log.artifact_ids_consumed,
            started_at=log.started_at,
            completed_at=log.completed_at,
            duration_ms=log.duration_ms,
            success=log.success,
            error_code=log.error_code,
            error_message=log.error_message,
            prev_hash=log.prev_hash,
            record_hash=log.record_hash,
            dp_metrics=log.dp_metrics_json,
        )
        for log in logs
    ]


# ==============================================================================
# Artifact Endpoints
# ==============================================================================


@router.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: str,
    tenant_id: str = Depends(get_tenant_id),
):
    """Download artifact content (encrypted)."""
    artifact = _artifacts.get(artifact_id)
    if artifact is None or artifact.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "ARTIFACT_NOT_FOUND", "message": f"Artifact '{artifact_id}' not found"}},
        )

    # Return encrypted bytes directly
    from fastapi.responses import Response

    content = _storage_backend.read(artifact.storage_key)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{artifact_id}.enc"',
            "X-Encryption-Algorithm": artifact.encryption_algorithm,
            "X-Encryption-Key-Id": artifact.encryption_key_id,
        },
    )


# ==============================================================================
# Health Check
# ==============================================================================


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "tg-tinker"}


# ==============================================================================
# TGSP Adapter Management Endpoints (Encrypted Inference Lock-In)
# ==============================================================================

# TGSP adapter registry storage (per-tenant)
_tgsp_registries: Dict[str, Any] = {}

# Request ID counter for audit correlation (SOC 2 CC4.1)
import uuid
import threading
_request_id_counter = 0
_request_id_lock = threading.Lock()


def generate_request_id() -> str:
    """Generate unique request ID for audit correlation."""
    global _request_id_counter
    with _request_id_lock:
        _request_id_counter += 1
        return f"tgsp-{uuid.uuid4().hex[:8]}-{_request_id_counter}"


# Simple rate limiter for TGSP inference (SOC 2 CC6.1)
import threading
from collections import defaultdict
import time

_rate_limit_buckets: Dict[str, List[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()
TGSP_RATE_LIMIT_REQUESTS = 100  # Max requests per window
TGSP_RATE_LIMIT_WINDOW = 60  # Window in seconds


def check_rate_limit(tenant_id: str) -> bool:
    """Check if tenant is within rate limits. Returns True if allowed."""
    now = time.time()
    with _rate_limit_lock:
        # Clean old entries
        _rate_limit_buckets[tenant_id] = [
            t for t in _rate_limit_buckets[tenant_id]
            if now - t < TGSP_RATE_LIMIT_WINDOW
        ]
        # Check limit
        if len(_rate_limit_buckets[tenant_id]) >= TGSP_RATE_LIMIT_REQUESTS:
            return False
        # Record request
        _rate_limit_buckets[tenant_id].append(now)
        return True


class TGSPAdapterLoadRequest(BaseModel):
    """Request to load a TGSP adapter."""
    tgsp_path: str = Field(..., description="Path to .tgsp file")
    adapter_id: Optional[str] = Field(None, description="Custom adapter ID")
    recipient_key_path: Optional[str] = Field(None, description="Path to recipient private key")


class TGSPAdapterResponse(BaseModel):
    """Response containing TGSP adapter details."""
    adapter_id: str
    tgsp_path: str
    model_name: str
    model_version: str
    signature_verified: bool
    lora_rank: int
    lora_alpha: float
    target_modules: List[str]
    is_active: bool
    forward_count: int
    loaded_at: datetime


class TGSPAdapterActivateRequest(BaseModel):
    """Request to activate a TGSP adapter."""
    adapter_id: str


class TGSPInferenceRequest(BaseModel):
    """Request for encrypted inference with TGSP adapter."""
    inputs: List[List[float]]  # Batch of input activations
    module_name: Optional[str] = Field(None, description="Target module for LoRA")


class TGSPInferenceResponse(BaseModel):
    """Response from encrypted inference."""
    outputs: List[List[float]]
    adapter_id: str
    inference_time_ms: float
    he_metrics: Optional[Dict[str, Any]] = None
    tgsp_compliant: bool = True
    request_id: Optional[str] = None  # For audit correlation (SOC 2 CC4.1)


def _get_tenant_registry(tenant_id: str) -> Any:
    """Get or create TGSP registry for tenant."""
    if tenant_id not in _tgsp_registries:
        try:
            from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry
            _tgsp_registries[tenant_id] = TGSPAdapterRegistry(
                enforce_tgsp=True,
                auto_verify_signatures=True,
            )
        except ImportError:
            # Fallback for testing
            _tgsp_registries[tenant_id] = None
    return _tgsp_registries[tenant_id]


@router.post(
    "/tgsp/adapters",
    response_model=TGSPAdapterResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["tgsp-adapters"],
)
async def load_tgsp_adapter(
    request: TGSPAdapterLoadRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> TGSPAdapterResponse:
    """
    Load a LoRA adapter from a TGSP package.

    This endpoint ONLY accepts TGSP-format adapters (.tgsp files).
    The TGSP format ensures:
    - Cryptographic signature verification
    - Audit trail for compliance
    - Secure key management

    This is part of the encrypted inference lock-in - TGSP format is REQUIRED
    for all adapters used with HE-encrypted inference.
    """
    registry = _get_tenant_registry(tenant_id)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error": {"code": "TGSP_NOT_AVAILABLE", "message": "TGSP registry not available"}},
        )

    # Validate file extension
    if not request.tgsp_path.lower().endswith('.tgsp'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "TGSP_FORMAT_REQUIRED",
                    "message": "Only TGSP-format adapters (.tgsp) are allowed for encrypted inference",
                    "details": {
                        "provided_path": request.tgsp_path,
                        "required_extension": ".tgsp",
                        "help": "Use 'tgsp build' to create a TGSP package from your adapter",
                    },
                }
            },
        )

    try:
        adapter_id = registry.load_tgsp_adapter(
            tgsp_path=request.tgsp_path,
            recipient_key_path=request.recipient_key_path,
            adapter_id=request.adapter_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "TGSP_LOAD_FAILED",
                    "message": str(e),
                }
            },
        )

    # Get adapter info
    info = registry.get_adapter_info(adapter_id)

    # Log to audit
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id="tgsp-registry",
        operation="load_tgsp_adapter",
        request_hash=f"sha256:{hashlib.sha256(request.tgsp_path.encode()).hexdigest()}",
        request_size_bytes=len(request.tgsp_path),
        artifact_ids_produced=[adapter_id],
        success=True,
    )

    return TGSPAdapterResponse(
        adapter_id=info["adapter_id"],
        tgsp_path=info["tgsp_path"],
        model_name=info["model_name"],
        model_version=info["model_version"],
        signature_verified=info["signature_verified"],
        lora_rank=info["lora_rank"],
        lora_alpha=info["lora_alpha"],
        target_modules=info["target_modules"],
        is_active=info["is_active"],
        forward_count=info["forward_count"],
        loaded_at=datetime.fromisoformat(info["loaded_at"]),
    )


@router.get(
    "/tgsp/adapters",
    response_model=List[TGSPAdapterResponse],
    tags=["tgsp-adapters"],
)
async def list_tgsp_adapters(
    tenant_id: str = Depends(get_tenant_id),
) -> List[TGSPAdapterResponse]:
    """
    List all loaded TGSP adapters for the tenant.

    Returns information about each adapter including:
    - Adapter ID and source path
    - Model information
    - Signature verification status
    - LoRA configuration
    - Usage statistics
    """
    registry = _get_tenant_registry(tenant_id)
    if registry is None:
        return []

    adapters = registry.list_adapters()

    return [
        TGSPAdapterResponse(
            adapter_id=a["adapter_id"],
            tgsp_path=a["tgsp_path"],
            model_name=a["model_name"],
            model_version=a["model_version"],
            signature_verified=a["signature_verified"],
            lora_rank=a["lora_rank"],
            lora_alpha=a["lora_alpha"],
            target_modules=a["target_modules"],
            is_active=a["is_active"],
            forward_count=a["forward_count"],
            loaded_at=datetime.fromisoformat(a["loaded_at"]),
        )
        for a in adapters
    ]


@router.post(
    "/tgsp/adapters/activate",
    response_model=TGSPAdapterResponse,
    tags=["tgsp-adapters"],
)
async def activate_tgsp_adapter(
    request: TGSPAdapterActivateRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> TGSPAdapterResponse:
    """
    Activate a TGSP adapter for encrypted inference.

    This enables hot-swapping of adapters without restarting the inference
    engine. The activated adapter will be used for all subsequent encrypted
    inference requests.

    Only one adapter can be active at a time per tenant.
    """
    registry = _get_tenant_registry(tenant_id)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error": {"code": "TGSP_NOT_AVAILABLE", "message": "TGSP registry not available"}},
        )

    try:
        registry.activate_adapter(request.adapter_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "ADAPTER_NOT_FOUND",
                    "message": str(e),
                }
            },
        )

    info = registry.get_adapter_info(request.adapter_id)

    # Log to audit
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id="tgsp-registry",
        operation="activate_tgsp_adapter",
        request_hash=f"sha256:{hashlib.sha256(request.adapter_id.encode()).hexdigest()}",
        request_size_bytes=len(request.adapter_id),
        success=True,
    )

    return TGSPAdapterResponse(
        adapter_id=info["adapter_id"],
        tgsp_path=info["tgsp_path"],
        model_name=info["model_name"],
        model_version=info["model_version"],
        signature_verified=info["signature_verified"],
        lora_rank=info["lora_rank"],
        lora_alpha=info["lora_alpha"],
        target_modules=info["target_modules"],
        is_active=info["is_active"],
        forward_count=info["forward_count"],
        loaded_at=datetime.fromisoformat(info["loaded_at"]),
    )


@router.delete(
    "/tgsp/adapters/{adapter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["tgsp-adapters"],
)
async def unload_tgsp_adapter(
    adapter_id: str,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Unload a TGSP adapter from the registry.

    This removes the adapter from memory and cleans up any temporary files.
    If the adapter is currently active, it will be deactivated first.
    """
    registry = _get_tenant_registry(tenant_id)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error": {"code": "TGSP_NOT_AVAILABLE", "message": "TGSP registry not available"}},
        )

    success = registry.unload_adapter(adapter_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "ADAPTER_NOT_FOUND",
                    "message": f"Adapter '{adapter_id}' not found",
                }
            },
        )

    # Log to audit
    audit = get_audit_logger()
    audit.log_operation(
        tenant_id=tenant_id,
        training_client_id="tgsp-registry",
        operation="unload_tgsp_adapter",
        request_hash=f"sha256:{hashlib.sha256(adapter_id.encode()).hexdigest()}",
        request_size_bytes=len(adapter_id),
        artifact_ids_consumed=[adapter_id],
        success=True,
    )


@router.post(
    "/tgsp/inference",
    response_model=TGSPInferenceResponse,
    tags=["tgsp-adapters"],
)
async def tgsp_encrypted_inference(
    request: TGSPInferenceRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> TGSPInferenceResponse:
    """
    Run encrypted inference using the active TGSP adapter.

    This endpoint REQUIRES a TGSP-format adapter to be loaded and activated.
    It enforces the TGSP format lock-in for all encrypted inference operations.

    The inference computes LoRA deltas under homomorphic encryption (CKKS),
    providing privacy for user activations while maintaining fast base model
    inference in plaintext.

    Rate limited to 100 requests per 60 seconds per tenant (SOC 2 CC6.1).

    Returns:
        Decrypted LoRA deltas for each input in the batch
    """
    import time
    import numpy as np

    # Generate request ID for audit correlation (SOC 2 CC4.1)
    request_id = generate_request_id()

    # Check rate limit (SOC 2 CC6.1)
    if not check_rate_limit(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Max {TGSP_RATE_LIMIT_REQUESTS} requests per {TGSP_RATE_LIMIT_WINDOW} seconds.",
                    "request_id": request_id,
                }
            },
        )

    registry = _get_tenant_registry(tenant_id)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error": {"code": "TGSP_NOT_AVAILABLE", "message": "TGSP registry not available", "request_id": request_id}},
        )

    # Check for active adapter
    active = registry.get_active_adapter()
    if active is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "NO_ACTIVE_ADAPTER",
                    "message": "No TGSP adapter is activated. Use POST /tgsp/adapters/activate first.",
                    "request_id": request_id,
                    "details": {
                        "help": "Encrypted inference requires a TGSP-format adapter to be loaded and activated",
                    },
                }
            },
        )

    start_time = time.perf_counter()

    try:
        # Convert inputs to numpy
        inputs = np.array(request.inputs, dtype=np.float64)

        # Run encrypted forward pass
        outputs = []
        for i in range(len(inputs)):
            delta = registry.forward_he(inputs[i], request.module_name)
            outputs.append(delta.tolist())

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Log to audit with request_id (SOC 2 CC4.1)
        audit = get_audit_logger()
        audit.log_operation(
            tenant_id=tenant_id,
            training_client_id="tgsp-registry",
            operation="tgsp_encrypted_inference",
            request_hash=f"sha256:{hashlib.sha256(str(len(request.inputs)).encode()).hexdigest()}",
            request_size_bytes=len(request.inputs) * 8 * (len(request.inputs[0]) if request.inputs else 0),
            success=True,
        )

        return TGSPInferenceResponse(
            outputs=outputs,
            adapter_id=active.metadata.adapter_id,
            inference_time_ms=elapsed_ms,
            tgsp_compliant=True,
            request_id=request_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": str(e),
                    "request_id": request_id,
                }
            },
        )


@router.get(
    "/tgsp/audit",
    response_model=List[Dict[str, Any]],
    tags=["tgsp-adapters"],
)
async def get_tgsp_audit_log(
    tenant_id: str = Depends(get_tenant_id),
) -> List[Dict[str, Any]]:
    """
    Get the TGSP adapter audit log for compliance.

    Returns all adapter-related operations including:
    - Adapter loading and unloading
    - Activation changes
    - Inference requests
    - Signature verification results
    """
    registry = _get_tenant_registry(tenant_id)
    if registry is None:
        return []

    return registry.get_audit_log()


# ==============================================================================
# LoRA to TGSP Conversion Endpoints
# ==============================================================================

class LoRAConvertRequest(BaseModel):
    """Request to convert a LoRA adapter to TGSP format."""
    input_path: str = Field(..., description="Path to LoRA adapter file or directory")
    output_path: Optional[str] = Field(None, description="Output TGSP file path (auto-generated if not provided)")
    model_name: Optional[str] = Field(None, description="Model name (auto-detected if not provided)")
    model_version: str = Field(default="1.0.0", description="Model version")
    validate_weights: bool = Field(default=True, description="Validate LoRA weights before conversion")
    auto_generate_keys: bool = Field(default=True, description="Auto-generate missing cryptographic keys")


class LoRAConvertResponse(BaseModel):
    """Response from LoRA to TGSP conversion."""
    success: bool
    output_path: str
    adapter_id: str
    model_name: str
    model_version: str

    # Cryptographic info
    manifest_hash: str
    payload_hash: str
    signature_key_id: str

    # LoRA config
    lora_rank: int
    lora_alpha: float
    target_modules: List[str]

    # Statistics
    input_format: str
    input_size_bytes: int
    output_size_bytes: int
    conversion_time_ms: float

    # Error (if any)
    error: Optional[str] = None


class BatchLoRAConvertRequest(BaseModel):
    """Request to convert multiple LoRA adapters to TGSP format."""
    input_paths: List[str] = Field(..., description="List of paths to LoRA adapters")
    output_dir: str = Field(..., description="Output directory for TGSP files")
    model_version: str = Field(default="1.0.0", description="Model version for all adapters")
    validate_weights: bool = Field(default=True, description="Validate LoRA weights")
    auto_generate_keys: bool = Field(default=True, description="Auto-generate missing keys")


class BatchLoRAConvertResponse(BaseModel):
    """Response from batch LoRA to TGSP conversion."""
    total: int
    successful: int
    failed: int
    results: List[LoRAConvertResponse]


@router.post(
    "/tgsp/convert",
    response_model=LoRAConvertResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["tgsp-conversion"],
)
async def convert_lora_to_tgsp(
    request: LoRAConvertRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> LoRAConvertResponse:
    """
    Convert a standard LoRA adapter to TGSP format.

    This endpoint allows users to convert their existing LoRA adapters
    (safetensors, PyTorch .bin/.pt, or Hugging Face directories) into
    the TGSP (TensorGuard Secure Package) format for use with HE-encrypted
    inference.

    Supported input formats:
    - safetensors (.safetensors)
    - PyTorch (.bin, .pt)
    - Hugging Face adapter directory (with adapter_config.json)

    The conversion:
    1. Detects and validates the input format
    2. Extracts LoRA weights and configuration
    3. Validates compatibility with HE operations
    4. Creates a TGSP package with cryptographic signatures

    The output .tgsp file can then be loaded using POST /tgsp/adapters
    for encrypted inference.
    """
    try:
        from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter

        # Determine output path
        import os
        from pathlib import Path

        if request.output_path:
            output_path = request.output_path
        else:
            # Auto-generate output path
            input_stem = Path(request.input_path).stem
            if Path(request.input_path).is_dir():
                input_stem = Path(request.input_path).name
            output_path = f"/tmp/tgsp_outputs/{tenant_id}/{input_stem}.tgsp"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ensure output has .tgsp extension
        if not output_path.lower().endswith('.tgsp'):
            output_path += '.tgsp'

        converter = LoRAToTGSPConverter(
            auto_generate_keys=request.auto_generate_keys,
        )

        try:
            result = converter.convert(
                input_path=request.input_path,
                output_path=output_path,
                model_name=request.model_name,
                model_version=request.model_version,
                validate=request.validate_weights,
            )

            # Log to audit
            audit = get_audit_logger()
            audit.log_operation(
                tenant_id=tenant_id,
                training_client_id="tgsp-converter",
                operation="convert_lora_to_tgsp",
                request_hash=f"sha256:{hashlib.sha256(request.input_path.encode()).hexdigest()}",
                request_size_bytes=result.input_size_bytes,
                artifact_ids_produced=[result.adapter_id] if result.success else [],
                success=result.success,
                error_message=result.error,
            )

            return LoRAConvertResponse(
                success=result.success,
                output_path=result.output_path,
                adapter_id=result.adapter_id,
                model_name=result.model_name,
                model_version=result.model_version,
                manifest_hash=result.manifest_hash,
                payload_hash=result.payload_hash,
                signature_key_id=result.signature_key_id,
                lora_rank=result.lora_config.rank,
                lora_alpha=result.lora_config.alpha,
                target_modules=result.lora_config.target_modules,
                input_format=result.input_format.value,
                input_size_bytes=result.input_size_bytes,
                output_size_bytes=result.output_size_bytes,
                conversion_time_ms=result.conversion_time_ms,
                error=result.error,
            )

        finally:
            converter.cleanup()

    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "error": {
                    "code": "CONVERTER_NOT_AVAILABLE",
                    "message": f"LoRA to TGSP converter not available: {e}",
                }
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "CONVERSION_FAILED",
                    "message": str(e),
                }
            },
        )


@router.post(
    "/tgsp/convert/batch",
    response_model=BatchLoRAConvertResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["tgsp-conversion"],
)
async def batch_convert_lora_to_tgsp(
    request: BatchLoRAConvertRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> BatchLoRAConvertResponse:
    """
    Convert multiple LoRA adapters to TGSP format in batch.

    This endpoint converts multiple LoRA adapters at once, which is
    useful for migrating a collection of adapters to the TGSP format.

    Each input adapter will be converted to a separate .tgsp file
    in the specified output directory.
    """
    import os

    try:
        from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter

        os.makedirs(request.output_dir, exist_ok=True)

        converter = LoRAToTGSPConverter(
            auto_generate_keys=request.auto_generate_keys,
        )

        try:
            results = converter.batch_convert(
                input_paths=request.input_paths,
                output_dir=request.output_dir,
                model_version=request.model_version,
                validate=request.validate_weights,
            )

            # Convert to response models
            response_results = []
            for result in results:
                response_results.append(LoRAConvertResponse(
                    success=result.success,
                    output_path=result.output_path,
                    adapter_id=result.adapter_id,
                    model_name=result.model_name,
                    model_version=result.model_version,
                    manifest_hash=result.manifest_hash,
                    payload_hash=result.payload_hash,
                    signature_key_id=result.signature_key_id,
                    lora_rank=result.lora_config.rank,
                    lora_alpha=result.lora_config.alpha,
                    target_modules=result.lora_config.target_modules,
                    input_format=result.input_format.value,
                    input_size_bytes=result.input_size_bytes,
                    output_size_bytes=result.output_size_bytes,
                    conversion_time_ms=result.conversion_time_ms,
                    error=result.error,
                ))

            successful = sum(1 for r in results if r.success)

            # Log to audit
            audit = get_audit_logger()
            audit.log_operation(
                tenant_id=tenant_id,
                training_client_id="tgsp-converter",
                operation="batch_convert_lora_to_tgsp",
                request_hash=f"sha256:{hashlib.sha256(str(request.input_paths).encode()).hexdigest()}",
                request_size_bytes=sum(r.input_size_bytes for r in results),
                artifact_ids_produced=[r.adapter_id for r in results if r.success],
                success=successful == len(results),
            )

            return BatchLoRAConvertResponse(
                total=len(results),
                successful=successful,
                failed=len(results) - successful,
                results=response_results,
            )

        finally:
            converter.cleanup()

    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "error": {
                    "code": "CONVERTER_NOT_AVAILABLE",
                    "message": f"LoRA to TGSP converter not available: {e}",
                }
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "BATCH_CONVERSION_FAILED",
                    "message": str(e),
                }
            },
        )


@router.get(
    "/tgsp/convert/formats",
    response_model=Dict[str, Any],
    tags=["tgsp-conversion"],
)
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get information about supported LoRA formats for conversion.

    Returns details about which input formats are supported and
    what configuration options are available.
    """
    return {
        "supported_formats": [
            {
                "format": "safetensors",
                "extensions": [".safetensors"],
                "description": "Safetensors format (recommended)",
                "requires_config": False,
            },
            {
                "format": "pytorch_bin",
                "extensions": [".bin"],
                "description": "PyTorch binary format",
                "requires_config": False,
            },
            {
                "format": "pytorch_pt",
                "extensions": [".pt"],
                "description": "PyTorch checkpoint format",
                "requires_config": False,
            },
            {
                "format": "huggingface_dir",
                "extensions": ["(directory)"],
                "description": "Hugging Face adapter directory with adapter_config.json",
                "requires_config": True,
            },
        ],
        "output_format": {
            "format": "tgsp",
            "version": "1.0",
            "extension": ".tgsp",
            "description": "TensorGuard Secure Package - Post-quantum hybrid encrypted container",
            "features": [
                "Hybrid post-quantum signatures (Ed25519 + Dilithium)",
                "Hybrid encryption (Kyber + ChaCha20Poly1305)",
                "Manifest with integrity hashes",
                "Audit trail support",
                "Compatible with HE-encrypted inference",
            ],
        },
        "auto_detection": {
            "config_file": "adapter_config.json",
            "weight_patterns": ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"],
        },
    }
