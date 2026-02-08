"""
MSS (Model Serving Service) Server

OpenAI-compatible API server for secure LoRA inference with HE-LoRA support.

Endpoints:
- POST /v1/completions - Text completion
- POST /v1/chat/completions - Chat completion
- POST /v1/helora/adapters - Load HE-LoRA adapter
- DELETE /v1/helora/adapters/{adapter_id} - Unload adapter
- GET /v1/models - List available models
- GET /health - Health check
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def create_app(
    model_id: Optional[str] = None,
    backend: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """
    Create the FastAPI application.

    Args:
        model_id: HuggingFace model identifier
        backend: Backend type ("vllm", "tensorrt_llm", "sglang", "mock")
        config_path: Path to configuration file

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        raise ImportError("FastAPI required for MSS server")

    from .router import BackendType, RequestRouter, RouterConfig
    from .schemas import (
        ChatCompletionRequest,
        CompletionRequest,
        ErrorResponse,
        InsertionPointSchema,
    )

    # Global router instance
    router: Optional[RequestRouter] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        nonlocal router

        # Load configuration
        config = RouterConfig()
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config_data = json.load(f)
                if 'default_backend' in config_data:
                    config.default_backend = BackendType(config_data['default_backend'])
                if 'max_batch_size' in config_data:
                    config.max_batch_size = config_data['max_batch_size']

        # Override with arguments
        if backend:
            config.default_backend = BackendType(backend)

        # Initialize router
        router = RequestRouter(config)

        if model_id:
            router.initialize(model_id, config.default_backend)
            logger.info(f"MSS server started with model: {model_id}")
        else:
            logger.info("MSS server started (no model loaded)")

        yield

        # Shutdown
        if router:
            router.shutdown()
        logger.info("MSS server shutdown")

    app = FastAPI(
        title="HE-LoRA Model Serving Service",
        description="OpenAI-compatible API for secure LoRA inference",
        version="1.0.0",
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # Pydantic models for API
    # -------------------------------------------------------------------------

    class CompletionRequestModel(BaseModel):
        model: str
        prompt: Union[str, list]
        max_tokens: int = 16
        temperature: float = 1.0
        top_p: float = 1.0
        n: int = 1
        stream: bool = False
        logprobs: Optional[int] = None
        echo: bool = False
        stop: Optional[Union[str, list]] = None
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        best_of: int = 1
        logit_bias: Optional[Dict[str, float]] = None
        user: Optional[str] = None
        helora_config: Optional[Dict[str, Any]] = None

    class ChatCompletionRequestModel(BaseModel):
        model: str
        messages: list
        max_tokens: Optional[int] = None
        temperature: float = 1.0
        top_p: float = 1.0
        n: int = 1
        stream: bool = False
        stop: Optional[Union[str, list]] = None
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        logit_bias: Optional[Dict[str, float]] = None
        user: Optional[str] = None
        helora_config: Optional[Dict[str, Any]] = None

    class LoadAdapterRequest(BaseModel):
        adapter_id: str
        model_id: str
        adapter_path: Optional[str] = None
        rank: int = 16
        alpha: float = 32.0
        targets: str = "qkv"
        layers: Optional[list] = None

    # -------------------------------------------------------------------------
    # COMPLETION ENDPOINTS
    # -------------------------------------------------------------------------

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequestModel):
        """Create a text completion."""
        if router is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        try:
            # Convert to internal request format
            helora_config = None
            if request.helora_config:
                helora_config = InsertionPointSchema.from_dict(request.helora_config)

            completion_request = CompletionRequest(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stream=request.stream,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                best_of=request.best_of,
                logit_bias=request.logit_bias,
                user=request.user,
                helora_config=helora_config,
            )

            # Process request
            response = router.process_completion(completion_request)
            return JSONResponse(content=response.to_dict())

        except Exception as e:
            logger.error(f"Completion error: {e}")
            error = ErrorResponse.create(
                message=str(e),
                type="server_error",
            )
            return JSONResponse(content=error.to_dict(), status_code=500)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequestModel):
        """Create a chat completion."""
        if router is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        try:
            # Convert messages
            from .schemas import ChatMessage
            messages = [
                ChatMessage(
                    role=m.get('role', 'user'),
                    content=m.get('content', ''),
                    name=m.get('name'),
                )
                for m in request.messages
            ]

            # Convert to internal request format
            helora_config = None
            if request.helora_config:
                helora_config = InsertionPointSchema.from_dict(request.helora_config)

            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stream=request.stream,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logit_bias=request.logit_bias,
                user=request.user,
                helora_config=helora_config,
            )

            # Process request
            response = router.process_chat_completion(chat_request)
            return JSONResponse(content=response.to_dict())

        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            error = ErrorResponse.create(
                message=str(e),
                type="server_error",
            )
            return JSONResponse(content=error.to_dict(), status_code=500)

    # -------------------------------------------------------------------------
    # HE-LORA ADAPTER ENDPOINTS
    # -------------------------------------------------------------------------

    @app.post("/v1/helora/adapters")
    async def load_adapter(request: LoadAdapterRequest):
        """Load an HE-LoRA adapter."""
        if router is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        try:
            success = router.load_helora_adapter(
                adapter_id=request.adapter_id,
                model_id=request.model_id,
                adapter_path=request.adapter_path,
                rank=request.rank,
                alpha=request.alpha,
                targets=request.targets,
                layers=request.layers,
            )

            if success:
                return JSONResponse(content={
                    "success": True,
                    "adapter_id": request.adapter_id,
                    "message": f"Adapter {request.adapter_id} loaded successfully",
                })
            else:
                raise HTTPException(status_code=500, detail="Failed to load adapter")

        except Exception as e:
            logger.error(f"Load adapter error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/helora/adapters/{adapter_id}")
    async def unload_adapter(adapter_id: str):
        """Unload an HE-LoRA adapter."""
        if router is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        success = router.unload_helora_adapter(adapter_id)
        if success:
            return JSONResponse(content={
                "success": True,
                "adapter_id": adapter_id,
                "message": f"Adapter {adapter_id} unloaded",
            })
        else:
            raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")

    @app.get("/v1/helora/adapters")
    async def list_adapters():
        """List loaded HE-LoRA adapters."""
        if router is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        status = router.get_status()
        has_stats = status.get('has_stats', {})
        adapters = has_stats.get('loaded_adapters', []) if has_stats else []

        return JSONResponse(content={
            "adapters": adapters,
            "count": len(adapters),
        })

    # -------------------------------------------------------------------------
    # MODEL ENDPOINTS
    # -------------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        if router is None:
            return JSONResponse(content={"object": "list", "data": []})

        status = router.get_status()
        models = [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization",
            }
            for model_id in status.get('loaded_models', [])
        ]

        return JSONResponse(content={"object": "list", "data": models})

    @app.post("/v1/models/load")
    async def load_model(request: Request):
        """Load a model into the server."""
        if router is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        data = await request.json()
        model_id = data.get('model_id')
        backend = data.get('backend')

        if not model_id:
            raise HTTPException(status_code=400, detail="model_id required")

        try:
            backend_type = BackendType(backend) if backend else None
            router.initialize(model_id, backend_type)
            return JSONResponse(content={
                "success": True,
                "model_id": model_id,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # HEALTH & STATUS ENDPOINTS
    # -------------------------------------------------------------------------

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        status = "healthy" if router else "initializing"
        return JSONResponse(content={
            "status": status,
            "timestamp": time.time(),
        })

    @app.get("/v1/status")
    async def get_status():
        """Get detailed server status."""
        if router is None:
            return JSONResponse(content={
                "initialized": False,
                "status": "not_initialized",
            })

        return JSONResponse(content=router.get_status())

    return app


class MSSServer:
    """
    MSS Server wrapper for programmatic usage.

    Usage:
        server = MSSServer(model_id="meta-llama/Llama-2-7b-hf")
        server.start(host="0.0.0.0", port=8000)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        backend: str = "mock",
        config_path: Optional[str] = None,
    ):
        self._model_id = model_id
        self._backend = backend
        self._config_path = config_path
        self._app = None

    def create_app(self):
        """Create the FastAPI application."""
        self._app = create_app(
            model_id=self._model_id,
            backend=self._backend,
            config_path=self._config_path,
        )
        return self._app

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info",
    ) -> None:
        """Start the server."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn required. Install with: pip install uvicorn")

        if self._app is None:
            self.create_app()

        uvicorn.run(
            self._app,
            host=host,
            port=port,
            log_level=log_level,
        )


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="HE-LoRA Model Serving Service")
    parser.add_argument("--model", "-m", type=str, help="Model ID to load")
    parser.add_argument("--backend", "-b", type=str, default="mock",
                       choices=["vllm", "tensorrt_llm", "sglang", "mock"],
                       help="Inference backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    parser.add_argument("--log-level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start server
    server = MSSServer(
        model_id=args.model,
        backend=args.backend,
        config_path=args.config,
    )
    server.start(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
