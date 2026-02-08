"""
Model Serving Service (MSS)

OpenAI-compatible API for secure LoRA inference with HE-LoRA support.
The MSS holds no HE secret keys - those remain in the HE Adapter Service (HAS).

Components:
- API server with OpenAI-compatible endpoints
- Insertion point configuration schema
- Request routing to appropriate backend adapter
- Communication with HAS for delta injection
"""

from .server import MSSServer, create_app
from .schemas import (
    InsertionPointSchema,
    CompletionRequest,
    ChatCompletionRequest,
    CompletionResponse,
    ChatCompletionResponse,
)
from .router import RequestRouter
from .has_client import HASClient

__all__ = [
    'MSSServer',
    'create_app',
    'InsertionPointSchema',
    'CompletionRequest',
    'ChatCompletionRequest',
    'CompletionResponse',
    'ChatCompletionResponse',
    'RequestRouter',
    'HASClient',
]
