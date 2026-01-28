"""
TG-Tinker API routes.

FastAPI routers for the TG-Tinker training API.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from pydantic import BaseModel, Field

from .audit import get_audit_logger
from .dp import DPConfig, DPTrainer
from .models import (
    TinkerArtifact,
    TinkerFuture,
    TinkerTrainingClient,
    generate_artifact_id,
    generate_future_id,
    generate_tc_id,
)
from .queue import Job, JobStatus, get_job_queue
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
    clients = [
        tc for tc in _training_clients.values() if tc.tenant_id == tenant_id
    ]

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
            detail={"error": {"code": "TRAINING_CLIENT_NOT_FOUND", "message": f"Training client '{tc_id}' not found", "details": {"training_client_id": tc_id}}},
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
