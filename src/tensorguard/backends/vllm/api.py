"""OpenAI-Compatible API for TenSafe vLLM Backend.

Provides REST API endpoints compatible with OpenAI's API format,
enabling drop-in replacement for existing OpenAI integrations.
"""

import logging
import time
import uuid
from typing import Any, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for OpenAI API compatibility
class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: str = Field(..., description="Role: system, user, assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name")


class CompletionRequest(BaseModel):
    """Completion request (legacy format)."""
    model: str = Field(..., description="Model ID")
    prompt: Union[str, List[str]] = Field(..., description="Prompt(s)")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0, le=1, description="Nucleus sampling")
    n: int = Field(1, ge=1, le=10, description="Number of completions")
    stream: bool = Field(False, description="Stream responses")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)
    user: Optional[str] = Field(None, description="User ID for tracking")


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str = Field(..., description="Model ID")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)
    user: Optional[str] = Field(None)


class CompletionChoice(BaseModel):
    """Completion choice."""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: str


class CompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tensafe"


class ModelList(BaseModel):
    """List of models."""
    object: str = "list"
    data: List[ModelInfo]


class TenSafeAPIContext:
    """Context for API requests including privacy tracking."""

    def __init__(self, engine, api_key: Optional[str] = None, user_id: Optional[str] = None):
        self.engine = engine
        self.api_key = api_key
        self.user_id = user_id
        self.request_id = str(uuid.uuid4())
        self.start_time = time.time()


def create_openai_router(
    engine,
    require_api_key: bool = True,
    allowed_api_keys: Optional[List[str]] = None,
) -> APIRouter:
    """Create OpenAI-compatible API router.

    Args:
        engine: TenSafeVLLMEngine instance
        require_api_key: Whether to require API key authentication
        allowed_api_keys: List of allowed API keys (if None, accept any)

    Returns:
        FastAPI router with OpenAI-compatible endpoints
    """
    router = APIRouter(prefix="/v1", tags=["OpenAI Compatible API"])

    async def verify_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
        """Verify API key from Authorization header."""
        if not require_api_key:
            return None

        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid Authorization format")

        api_key = authorization[7:]

        if allowed_api_keys and api_key not in allowed_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return api_key

    @router.get("/models", response_model=ModelList)
    async def list_models(api_key: str = Depends(verify_api_key)):
        """List available models."""
        models = [
            ModelInfo(
                id=engine.config.model_path,
                created=int(time.time()),
            )
        ]

        # Add TSSP package as a "model" if loaded
        if engine.tssp_package:
            models.append(ModelInfo(
                id=f"tensafe:{engine.tssp_package.manifest.package_id}",
                created=int(time.time()),
            ))

        return ModelList(data=models)

    @router.get("/models/{model_id}", response_model=ModelInfo)
    async def get_model(model_id: str, api_key: str = Depends(verify_api_key)):
        """Get model information."""
        return ModelInfo(
            id=model_id,
            created=int(time.time()),
        )

    @router.post("/completions", response_model=CompletionResponse)
    async def create_completion(
        request: CompletionRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Create text completion (legacy endpoint)."""
        try:
            # Convert to sampling params
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
        except ImportError:
            sampling_params = None

        # Handle single or multiple prompts
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

        if request.stream:
            # Streaming response
            async def generate_stream():
                for prompt in prompts:
                    async for token in engine.generate_stream(prompt, sampling_params):
                        chunk = {
                            "id": f"cmpl-{uuid.uuid4()}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "text": token,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )

        # Non-streaming response
        results = engine.generate(prompts, sampling_params)

        # Build response
        choices = []
        total_completion_tokens = 0

        for i, result in enumerate(results):
            for j, output in enumerate(result.outputs):
                choices.append(CompletionChoice(
                    text=output["text"],
                    index=i * request.n + j,
                    finish_reason=output.get("finish_reason", "stop"),
                ))
                total_completion_tokens += len(output.get("token_ids", output["text"].split()))

        # Estimate prompt tokens
        prompt_tokens = sum(len(p.split()) for p in prompts)

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=prompt_tokens + total_completion_tokens,
            ),
        )

    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(
        request: ChatCompletionRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Create chat completion."""
        # Convert chat messages to prompt
        prompt = _format_chat_messages(request.messages)

        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
        except ImportError:
            sampling_params = None

        if request.stream:
            async def generate_stream():
                async for token in engine.generate_stream(prompt, sampling_params):
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )

        # Non-streaming
        results = engine.generate([prompt], sampling_params)

        choices = []
        total_completion_tokens = 0

        for i, result in enumerate(results):
            for j, output in enumerate(result.outputs):
                choices.append(ChatCompletionChoice(
                    index=j,
                    message=ChatMessage(role="assistant", content=output["text"]),
                    finish_reason=output.get("finish_reason", "stop"),
                ))
                total_completion_tokens += len(output.get("token_ids", output["text"].split()))

        prompt_tokens = len(prompt.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=prompt_tokens + total_completion_tokens,
            ),
        )

    @router.get("/metrics")
    async def get_metrics(api_key: str = Depends(verify_api_key)):
        """Get engine metrics including HE-LoRA performance."""
        return engine.get_metrics()

    @router.get("/privacy")
    async def get_privacy_info(api_key: str = Depends(verify_api_key)):
        """Get privacy information including TSSP package details."""
        info = {
            "he_lora_enabled": engine.config.enable_he_lora,
            "scheme": engine.config.he_scheme.value,
            "profile": engine.config.ckks_profile.value,
        }

        if engine.tssp_package:
            info["tssp_package"] = {
                "package_id": engine.tssp_package.manifest.package_id,
                "verified": True,
            }

        return info

    return router


def _format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into a single prompt string.

    Uses a simple format compatible with instruction-tuned models.
    """
    formatted = []

    for msg in messages:
        if msg.role == "system":
            formatted.append(f"<|system|>\n{msg.content}</s>")
        elif msg.role == "user":
            formatted.append(f"<|user|>\n{msg.content}</s>")
        elif msg.role == "assistant":
            formatted.append(f"<|assistant|>\n{msg.content}</s>")

    # Add assistant prompt
    formatted.append("<|assistant|>\n")

    return "\n".join(formatted)
