"""
Confidential Inference API Endpoints.

Provides TEE-protected endpoints for confidential inference:
- GET  /v1/confidential/attestation  - Get TEE attestation + session key
- POST /v1/confidential/session      - Create encrypted session
- POST /v1/confidential/chat/completions - Encrypted chat completion
- POST /v1/confidential/completions  - Encrypted text completion
- DELETE /v1/confidential/session/{id} - Destroy session
- GET  /v1/confidential/session/{id}/status - Session status

These wrap the existing OpenAI-compatible endpoints with HPKE
encryption and privacy receipts.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from .session import ConfidentialSessionManager
from .middleware import ConfidentialInferenceMiddleware
from .receipt import PrivacyReceiptGenerator

logger = logging.getLogger(__name__)


# --- Pydantic Models ---


class AttestationResponse(BaseModel):
    """Response from attestation endpoint."""

    tee_platform: str = Field(..., description="TEE platform type")
    attestation_quote: str = Field(..., description="Hex-encoded attestation quote")
    quote_signature: str = Field(..., description="Hex-encoded quote signature")
    server_public_key: str = Field(
        ..., description="Hex-encoded X25519 ephemeral public key"
    )
    session_id: str = Field(..., description="Session ID for subsequent requests")
    gpu_attestation: Optional[str] = Field(
        None, description="NVIDIA GPU attestation token"
    )


class CreateSessionRequest(BaseModel):
    """Request to create a confidential session."""

    ttl_seconds: int = Field(3600, description="Session TTL in seconds", ge=60, le=86400)
    client_nonce: Optional[str] = Field(
        None, description="Client nonce for attestation freshness (hex)"
    )


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    server_public_key: str
    attestation_quote: str
    quote_signature: str
    tee_platform: str
    expires_at: str
    gpu_attestation: Optional[str] = None


class EncryptedRequest(BaseModel):
    """Encrypted inference request envelope."""

    session_id: str = Field(..., description="Confidential session ID")
    encrypted_payload: Dict[str, str] = Field(
        ..., description="HPKE sealed payload {enc, ciphertext}"
    )
    client_public_key: str = Field(
        ..., description="Client X25519 public key (hex) for response encryption"
    )
    aad: Optional[str] = Field(None, description="Additional authenticated data (hex)")


class EncryptedResponse(BaseModel):
    """Encrypted inference response envelope."""

    session_id: str
    encrypted_response: Dict[str, str] = Field(
        ..., description="HPKE sealed response {enc, ciphertext}"
    )
    privacy_receipt: Dict[str, Any]
    aad: Optional[str] = None


class SessionStatusResponse(BaseModel):
    """Session status."""

    session_id: str
    is_active: bool
    created_at: str
    expires_at: str
    requests_processed: int
    bytes_decrypted: int
    bytes_encrypted: int


# --- API Router Factory ---


def create_confidential_router(
    engine: Any = None,
    attestation_provider: Any = None,
    require_api_key: bool = True,
    allowed_api_keys: Optional[List[str]] = None,
) -> APIRouter:
    """
    Create FastAPI router for confidential inference endpoints.

    Args:
        engine: TenSafeVLLMEngine instance (or None for testing)
        attestation_provider: AttestationProvider instance
        require_api_key: Whether to require API key auth
        allowed_api_keys: List of valid API keys

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/v1/confidential", tags=["Confidential Inference"])

    # Initialize components
    session_manager = ConfidentialSessionManager()
    middleware = ConfidentialInferenceMiddleware(session_manager=session_manager)

    # Determine TEE platform
    tee_platform = "simulation"
    if attestation_provider is not None:
        tee_type = getattr(attestation_provider, "attestation_type", None)
        if tee_type is not None:
            tee_platform = tee_type.value

    receipt_generator = PrivacyReceiptGenerator(
        tee_platform=tee_platform,
    )

    # --- Auth dependency ---

    async def verify_api_key(
        authorization: Optional[str] = Header(None),
    ) -> Optional[str]:
        if not require_api_key:
            return None
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Invalid Authorization format"
            )
        api_key = authorization[7:]
        if allowed_api_keys and api_key not in allowed_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key

    # --- Endpoints ---

    @router.post("/session", response_model=CreateSessionResponse)
    async def create_session(
        request: CreateSessionRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Create a new confidential inference session.

        Returns the TEE attestation quote and a server ephemeral public key.
        The client should verify the quote before sending encrypted requests.
        """
        # Generate attestation quote with client nonce
        nonce = None
        if request.client_nonce:
            nonce = bytes.fromhex(request.client_nonce)

        attestation_quote = None
        quote_data_hex = "simulation"
        quote_sig_hex = "simulation"
        quote_hash = "simulation"

        if attestation_provider is not None:
            try:
                attestation_quote = attestation_provider.generate_quote(nonce=nonce)
                quote_data_hex = attestation_quote.quote_data.hex()
                quote_sig_hex = attestation_quote.signature.hex()
                import hashlib

                quote_hash = hashlib.sha256(attestation_quote.quote_data).hexdigest()
            except Exception as e:
                logger.warning(f"Attestation generation failed: {e}")
                quote_hash = "unavailable"

        # Create session bound to attestation
        session = session_manager.create_session(
            ttl=request.ttl_seconds,
            attestation_quote_hash=quote_hash,
        )

        # Update receipt generator with quote hash
        receipt_generator._attestation_quote_hash = quote_hash

        return CreateSessionResponse(
            session_id=session.session_id,
            server_public_key=session.server_public_key_hex,
            attestation_quote=quote_data_hex,
            quote_signature=quote_sig_hex,
            tee_platform=tee_platform,
            expires_at=session.expires_at.isoformat(),
        )

    @router.get("/attestation", response_model=AttestationResponse)
    async def get_attestation(api_key: str = Depends(verify_api_key)):
        """
        Get TEE attestation and create a session in one step.

        Convenience endpoint that combines session creation with
        attestation retrieval.
        """
        # Create session
        session = session_manager.create_session()

        quote_data_hex = "simulation"
        quote_sig_hex = "simulation"

        if attestation_provider is not None:
            try:
                quote = attestation_provider.generate_quote()
                quote_data_hex = quote.quote_data.hex()
                quote_sig_hex = quote.signature.hex()
            except Exception as e:
                logger.warning(f"Attestation failed: {e}")

        return AttestationResponse(
            tee_platform=tee_platform,
            attestation_quote=quote_data_hex,
            quote_signature=quote_sig_hex,
            server_public_key=session.server_public_key_hex,
            session_id=session.session_id,
        )

    @router.post("/chat/completions", response_model=EncryptedResponse)
    async def confidential_chat_completion(
        request: EncryptedRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Confidential chat completion.

        Accepts an HPKE-encrypted request, decrypts inside the TEE,
        runs inference, encrypts the response, and returns a privacy receipt.
        """
        start_time = time.time()

        # Unwrap encrypted request
        try:
            plaintext_request, session, _ = middleware.unwrap_request(
                request.model_dump()
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Run inference (plaintext inside TEE)
        inference_response = await _run_chat_inference(engine, plaintext_request)

        # Generate privacy receipt
        latency_ms = (time.time() - start_time) * 1000
        receipt = receipt_generator.generate(
            session_id=session.session_id,
            he_mode=_get_he_mode(engine),
            he_backend=_get_he_backend(engine),
            adapter_encrypted=_is_adapter_encrypted(engine),
            he_metrics=_get_he_metrics(engine),
            tssp_hash=_get_tssp_hash(engine),
            dp_certificate=_get_dp_certificate(engine),
            latency_ms=latency_ms,
        )

        # Wrap encrypted response
        aad = bytes.fromhex(request.aad) if request.aad else b""
        try:
            envelope = middleware.wrap_response(
                response_data=inference_response,
                session=session,
                privacy_receipt=receipt.to_dict(),
                aad=aad,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return EncryptedResponse(**envelope)

    @router.post("/completions", response_model=EncryptedResponse)
    async def confidential_completion(
        request: EncryptedRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Confidential text completion (legacy format)."""
        start_time = time.time()

        try:
            plaintext_request, session, _ = middleware.unwrap_request(
                request.model_dump()
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))

        inference_response = await _run_completion_inference(
            engine, plaintext_request
        )

        latency_ms = (time.time() - start_time) * 1000
        receipt = receipt_generator.generate(
            session_id=session.session_id,
            he_mode=_get_he_mode(engine),
            latency_ms=latency_ms,
        )

        aad = bytes.fromhex(request.aad) if request.aad else b""
        try:
            envelope = middleware.wrap_response(
                response_data=inference_response,
                session=session,
                privacy_receipt=receipt.to_dict(),
                aad=aad,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return EncryptedResponse(**envelope)

    @router.get("/session/{session_id}", response_model=SessionStatusResponse)
    async def get_session_status(
        session_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Get session status."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"Session not found: {session_id}"
            )

        return SessionStatusResponse(
            session_id=session.session_id,
            is_active=session.is_active,
            created_at=session.created_at.isoformat(),
            expires_at=session.expires_at.isoformat(),
            requests_processed=session.requests_processed,
            bytes_decrypted=session.bytes_decrypted,
            bytes_encrypted=session.bytes_encrypted,
        )

    @router.delete("/session/{session_id}")
    async def destroy_session(
        session_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Destroy a confidential session."""
        destroyed = session_manager.destroy_session(session_id)
        if not destroyed:
            raise HTTPException(
                status_code=404, detail=f"Session not found: {session_id}"
            )
        return {"status": "destroyed", "session_id": session_id}

    @router.get("/stats")
    async def get_stats(api_key: str = Depends(verify_api_key)):
        """Get confidential inference statistics."""
        return session_manager.get_statistics()

    return router


# --- Helpers ---


async def _run_chat_inference(engine: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """Run chat inference, returning OpenAI-compatible response."""
    if engine is None:
        # Mock response for testing
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "tensafe-confidential"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[Confidential response - processed inside TEE]",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

    # Real inference via vLLM engine
    messages = request.get("messages", [])
    prompt = "\n".join(
        f"<|{m['role']}|>\n{m['content']}</s>" for m in messages
    )
    prompt += "\n<|assistant|>\n"

    sampling_params = None
    try:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=request.get("max_tokens", 1024),
            temperature=request.get("temperature", 1.0),
            top_p=request.get("top_p", 1.0),
        )
    except ImportError:
        pass

    results = engine.generate([prompt], sampling_params)

    choices = []
    total_tokens = 0
    for i, result in enumerate(results):
        for j, output in enumerate(result.outputs):
            choices.append(
                {
                    "index": j,
                    "message": {"role": "assistant", "content": output["text"]},
                    "finish_reason": output.get("finish_reason", "stop"),
                }
            )
            total_tokens += len(output.get("token_ids", output["text"].split()))

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model", "tensafe"),
        "choices": choices,
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": total_tokens,
            "total_tokens": len(prompt.split()) + total_tokens,
        },
    }


async def _run_completion_inference(
    engine: Any, request: Dict[str, Any]
) -> Dict[str, Any]:
    """Run text completion inference."""
    if engine is None:
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.get("model", "tensafe-confidential"),
            "choices": [
                {
                    "text": "[Confidential completion - processed inside TEE]",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            },
        }

    prompt = request.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0]

    sampling_params = None
    try:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=request.get("max_tokens", 100),
            temperature=request.get("temperature", 1.0),
        )
    except ImportError:
        pass

    results = engine.generate([prompt], sampling_params)

    choices = []
    for result in results:
        for j, output in enumerate(result.outputs):
            choices.append(
                {
                    "text": output["text"],
                    "index": j,
                    "finish_reason": output.get("finish_reason", "stop"),
                }
            )

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.get("model", "tensafe"),
        "choices": choices,
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 10,
            "total_tokens": len(prompt.split()) + 10,
        },
    }


def _get_he_mode(engine: Any) -> str:
    if engine is None:
        return "DISABLED"
    config = getattr(engine, "config", None)
    if config and getattr(config, "enable_he_lora", False):
        return "HE_ONLY"
    return "DISABLED"


def _get_he_backend(engine: Any) -> Optional[str]:
    if engine is None:
        return None
    config = getattr(engine, "config", None)
    if config and getattr(config, "enable_he_lora", False):
        return "CKKS-MOAI"
    return None


def _is_adapter_encrypted(engine: Any) -> bool:
    if engine is None:
        return False
    config = getattr(engine, "config", None)
    return bool(config and getattr(config, "enable_he_lora", False))


def _get_he_metrics(engine: Any) -> Optional[Dict[str, Any]]:
    if engine is None:
        return None
    get_metrics = getattr(engine, "get_metrics", None)
    if get_metrics:
        metrics = get_metrics()
        return metrics.get("he_lora", None)
    return None


def _get_tssp_hash(engine: Any) -> Optional[str]:
    if engine is None:
        return None
    tssp = getattr(engine, "tssp_package", None)
    if tssp:
        manifest = getattr(tssp, "manifest", None)
        if manifest:
            return getattr(manifest, "package_hash", None)
    return None


def _get_dp_certificate(engine: Any) -> Optional[Dict[str, Any]]:
    if engine is None:
        return None
    tssp = getattr(engine, "tssp_package", None)
    if tssp:
        manifest = getattr(tssp, "manifest", None)
        if manifest:
            return getattr(manifest, "dp_certificate", None)
    return None
