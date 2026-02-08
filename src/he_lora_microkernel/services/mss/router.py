"""
MSS Request Router

Routes inference requests to appropriate backend adapters
and coordinates with HAS for delta injection.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .has_client import HASClient, HASConfig, RequestContext
from .schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    CompletionRequest,
    CompletionResponse,
    InsertionPointSchema,
    Usage,
)

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Supported inference backend types."""
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    SGLANG = "sglang"
    MOCK = "mock"


@dataclass
class RouterConfig:
    """Configuration for the request router."""
    # Backend settings
    default_backend: BackendType = BackendType.MOCK
    backend_timeout_ms: int = 30000

    # HAS settings
    has_config: HASConfig = field(default_factory=HASConfig)
    enable_helora: bool = True

    # Generation settings
    default_max_tokens: int = 256
    max_batch_size: int = 32


class RequestRouter:
    """
    Routes requests to backend adapters and coordinates HE-LoRA.

    Handles:
    - Backend adapter selection and initialization
    - HAS client management for delta injection
    - Request lifecycle (prepare, generate, cleanup)
    - Streaming support (future)
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self._config = config or RouterConfig()

        # Backend adapters
        self._adapters: Dict[str, Any] = {}
        self._default_adapter = None

        # HAS client
        self._has_client: Optional[HASClient] = None

        # Active requests
        self._active_requests: Dict[str, Dict] = {}

        # Tokenizer (for prompt processing)
        self._tokenizer = None

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def initialize(
        self,
        model_id: str,
        backend: Optional[BackendType] = None,
        **backend_kwargs,
    ) -> None:
        """
        Initialize the router with a model and backend.

        Args:
            model_id: HuggingFace model identifier
            backend: Backend type to use (default from config)
            **backend_kwargs: Additional backend-specific arguments
        """
        backend = backend or self._config.default_backend

        logger.info(f"Initializing router with model={model_id}, backend={backend.value}")

        # Initialize backend adapter
        adapter = self._create_adapter(model_id, backend, **backend_kwargs)
        metadata = adapter.init()

        self._adapters[model_id] = {
            'adapter': adapter,
            'metadata': metadata,
            'backend': backend,
        }
        self._default_adapter = adapter

        # Initialize HAS client if HE-LoRA enabled
        if self._config.enable_helora:
            self._has_client = HASClient(self._config.has_config)
            self._has_client.connect()

        logger.info(f"Router initialized: {metadata.num_layers} layers, "
                   f"hidden_size={metadata.hidden_size}")

    def _create_adapter(
        self,
        model_id: str,
        backend: BackendType,
        **kwargs,
    ) -> Any:
        """Create a backend adapter instance."""
        from he_lora_microkernel.backend.base_adapter import BatchConfig, get_adapter

        batch_config = BatchConfig(
            max_batch_size=self._config.max_batch_size,
            max_context_length=kwargs.pop('max_context_length', 4096),
            max_generation_length=kwargs.pop('max_generation_length', 2048),
        )

        # Get adapter class
        try:
            adapter_cls = get_adapter(backend.value)
        except ValueError:
            adapter_cls = None

        if adapter_cls is None:
            # Fall back to mock
            from he_lora_microkernel.backend.base_adapter import BaseRuntimeAdapter, ModelMetadata
            logger.warning(f"Backend {backend.value} not found, using mock")

            # Create mock adapter
            class MockAdapter(BaseRuntimeAdapter):
                def init(self):
                    self._metadata = ModelMetadata(
                        model_id=model_id,
                        num_layers=32,
                        hidden_size=1024,  # Match executor hidden_size
                        num_attention_heads=32,
                        head_dim=32,  # 1024 / 32 = 32
                        vocab_size=32000,
                        max_position_embeddings=4096,
                        architecture='mock',
                        has_output_projection=True,
                    )
                    self._initialized = True
                    return self._metadata

                def shutdown(self):
                    self._initialized = False

                def prefill(self, input_ids, attention_mask=None):
                    return {'input_ids': input_ids}

                def decode_one_step(self, input_ids_last, kv_cache):
                    import torch
                    batch_size = input_ids_last.shape[0]
                    logits = torch.randn(batch_size, self._metadata.vocab_size)
                    return logits, {}

                def apply_deltas(self, layer_idx, deltas):
                    pass

                def get_layer_module(self, layer_idx):
                    return None

                def check_quantization(self):
                    return False

            return MockAdapter(model_id, batch_config, **kwargs)

        return adapter_cls(model_id, batch_config, **kwargs)


    def shutdown(self) -> None:
        """Shutdown router and release resources."""
        # Cleanup active requests
        for req_id in list(self._active_requests.keys()):
            self._cleanup_request(req_id)

        # Shutdown HAS client
        if self._has_client is not None:
            self._has_client.disconnect()
            self._has_client = None

        # Shutdown adapters
        for model_id, info in self._adapters.items():
            info['adapter'].shutdown()
        self._adapters.clear()
        self._default_adapter = None

        logger.info("Router shutdown complete")

    # -------------------------------------------------------------------------
    # REQUEST PROCESSING
    # -------------------------------------------------------------------------

    def process_completion(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """
        Process a completion request.

        Args:
            request: CompletionRequest with prompt and parameters

        Returns:
            CompletionResponse with generated text
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Setup HE-LoRA if configured
            has_ctx = None
            if request.helora_config and self._has_client:
                has_ctx = self._setup_helora(request_id, request.helora_config)

            # Tokenize prompt
            input_ids = self._tokenize(request.prompt)

            # Generate
            output_ids, stats = self._generate(
                input_ids=input_ids,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                has_ctx=has_ctx,
            )

            # Decode output
            output_text = self._decode(output_ids)

            # Build response
            choice = Choice(
                text=output_text,
                index=0,
                finish_reason="stop",
            )

            usage = Usage(
                prompt_tokens=input_ids.shape[-1],
                completion_tokens=len(output_ids) - input_ids.shape[-1],
                total_tokens=len(output_ids),
                helora_stats=stats.get('helora') if stats else None,
            )

            return CompletionResponse(
                id=f"cmpl-{request_id}",
                object="text_completion",
                created=int(start_time),
                model=request.model,
                choices=[choice],
                usage=usage,
            )

        finally:
            self._cleanup_request(request_id)

    def process_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Process a chat completion request.

        Args:
            request: ChatCompletionRequest with messages

        Returns:
            ChatCompletionResponse with assistant message
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Setup HE-LoRA if configured
            has_ctx = None
            if request.helora_config and self._has_client:
                has_ctx = self._setup_helora(request_id, request.helora_config)

            # Format messages to prompt
            prompt = self._format_chat_prompt(request.messages)

            # Tokenize
            input_ids = self._tokenize(prompt)

            # Generate
            max_tokens = request.max_tokens or self._config.default_max_tokens
            output_ids, stats = self._generate(
                input_ids=input_ids,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                has_ctx=has_ctx,
            )

            # Decode output
            output_text = self._decode(output_ids)

            # Extract assistant response
            assistant_text = self._extract_assistant_response(output_text)

            # Build response
            choice = ChatChoice(
                message=ChatMessage(role="assistant", content=assistant_text),
                index=0,
                finish_reason="stop",
            )

            usage = Usage(
                prompt_tokens=input_ids.shape[-1],
                completion_tokens=len(output_ids) - input_ids.shape[-1],
                total_tokens=len(output_ids),
                helora_stats=stats.get('helora') if stats else None,
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                object="chat.completion",
                created=int(start_time),
                model=request.model,
                choices=[choice],
                usage=usage,
            )

        finally:
            self._cleanup_request(request_id)

    # -------------------------------------------------------------------------
    # HE-LORA SETUP
    # -------------------------------------------------------------------------

    def _setup_helora(
        self,
        request_id: str,
        config: InsertionPointSchema,
    ) -> Optional[RequestContext]:
        """Setup HE-LoRA for a request."""
        if self._has_client is None or not config.enabled:
            return None

        # Ensure adapter is loaded
        adapter_info = self._has_client.get_adapter_info(config.adapter_id)
        if adapter_info is None:
            # Try to load adapter
            # In production, adapter should be pre-loaded
            logger.warning(f"Adapter {config.adapter_id} not loaded")
            return None

        # Get batch size and seq len from request context
        batch_size = 1  # Will be updated during tokenization
        seq_len = 0

        # Prepare request
        ctx = self._has_client.prepare_request(
            request_id=request_id,
            adapter_id=config.adapter_id,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Store in active requests
        self._active_requests[request_id] = {
            'has_ctx': ctx,
            'helora_config': config,
        }

        # Configure adapter insertion points
        if self._default_adapter:
            from ..backend.base_adapter import InsertionConfig, InsertionPoint, LoRATargets

            # Map schema to adapter config
            targets = LoRATargets.QKV if config.targets.value == "qkv" else LoRATargets.QKVO

            adapter_metadata = self._adapters.get(
                self._default_adapter._model_id, {}
            ).get('metadata')
            num_layers = adapter_metadata.num_layers if adapter_metadata else 32

            layers = config.layer_selection.get_layers(num_layers)

            insertion_config = InsertionConfig(
                targets=targets,
                layers=layers,
                insertion_point=InsertionPoint(config.insertion_point.value),
            )
            self._default_adapter.set_insertion_config(insertion_config)

            # Set delta callback
            callback = self._has_client.create_delta_callback(ctx)
            self._default_adapter.set_delta_callback(callback)

        return ctx

    def _cleanup_request(self, request_id: str) -> None:
        """Cleanup request resources."""
        if request_id in self._active_requests:
            req_info = self._active_requests[request_id]

            # Release HAS context
            if 'has_ctx' in req_info and self._has_client:
                self._has_client.release_request(req_info['has_ctx'])

            del self._active_requests[request_id]

    # -------------------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------------------

    def _generate(
        self,
        input_ids: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
        has_ctx: Optional[RequestContext] = None,
    ) -> Tuple[Any, Dict]:
        """
        Generate tokens using the backend adapter.

        Returns:
            Tuple of (output_ids, statistics)
        """
        import torch

        if self._default_adapter is None:
            raise RuntimeError("Router not initialized")

        stats = {'helora': {'tokens_processed': 0, 'layers_patched': 0}}

        # Prefill
        kv_cache = self._default_adapter.prefill(input_ids)

        # Decode loop
        output_ids = input_ids.clone()
        generated = 0

        while generated < max_tokens:
            # Get last token
            last_token = output_ids[:, -1:]

            # Decode one step
            logits, layer_states = self._default_adapter.decode_one_step(
                last_token, kv_cache
            )

            # Sample next token
            next_token = self._sample(logits, temperature, top_p)

            # Append to output
            output_ids = torch.cat([output_ids, next_token], dim=-1)
            generated += 1

            # Update stats
            stats['helora']['tokens_processed'] += 1

            # Check for EOS
            if self._is_eos(next_token):
                break

        return output_ids, stats

    def _sample(
        self,
        logits: Any,
        temperature: float,
        top_p: float,
    ) -> Any:
        """Sample next token from logits."""
        import torch

        if temperature == 0:
            # Greedy
            return logits.argmax(dim=-1, keepdim=True)

        # Apply temperature
        logits = logits / temperature

        # Apply top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _is_eos(self, token: Any) -> bool:
        """Check if token is end-of-sequence."""
        # Common EOS token IDs
        eos_ids = {0, 1, 2, 50256, 128001}  # Various tokenizers
        return token.item() in eos_ids

    # -------------------------------------------------------------------------
    # TOKENIZATION
    # -------------------------------------------------------------------------

    def _tokenize(self, prompt: Union[str, List[str], List[int]]) -> Any:
        """Tokenize a prompt."""
        import torch

        if isinstance(prompt, str):
            # Simple mock tokenization
            # In production, use actual tokenizer
            tokens = [ord(c) % 32000 for c in prompt]
            return torch.tensor([tokens], dtype=torch.long)

        elif isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                return torch.tensor([prompt], dtype=torch.long)
            else:
                # List of strings
                tokens_list = []
                for p in prompt:
                    tokens = [ord(c) % 32000 for c in p]
                    tokens_list.append(tokens)
                # Pad to same length
                max_len = max(len(t) for t in tokens_list)
                padded = [t + [0] * (max_len - len(t)) for t in tokens_list]
                return torch.tensor(padded, dtype=torch.long)

        return torch.tensor([[0]], dtype=torch.long)

    def _decode(self, token_ids: Any) -> str:
        """Decode token IDs to text."""
        # Simple mock decoding
        # In production, use actual tokenizer
        if hasattr(token_ids, 'tolist'):
            ids = token_ids[0].tolist() if token_ids.dim() > 1 else token_ids.tolist()
        else:
            ids = token_ids

        # Mock: just return placeholder
        return f"[Generated text from {len(ids)} tokens]"

    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string."""
        # Simple chat template
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _extract_assistant_response(self, output_text: str) -> str:
        """Extract assistant response from generated text."""
        # Simple extraction - look for last "Assistant:" and take everything after
        if "Assistant:" in output_text:
            parts = output_text.split("Assistant:")
            return parts[-1].strip()
        return output_text

    # -------------------------------------------------------------------------
    # ADAPTER MANAGEMENT
    # -------------------------------------------------------------------------

    def load_helora_adapter(
        self,
        adapter_id: str,
        model_id: str,
        adapter_path: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Load an HE-LoRA adapter for use in requests.

        Args:
            adapter_id: Unique identifier for this adapter
            model_id: Base model identifier
            adapter_path: Path to adapter weights

        Returns:
            True if adapter loaded successfully
        """
        if self._has_client is None:
            logger.error("HAS client not initialized")
            return False

        try:
            self._has_client.load_adapter(
                adapter_id=adapter_id,
                model_id=model_id,
                adapter_path=adapter_path,
                **kwargs,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            return False

    def unload_helora_adapter(self, adapter_id: str) -> bool:
        """Unload an HE-LoRA adapter."""
        if self._has_client is None:
            return False
        return self._has_client.unload_adapter(adapter_id)

    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            'initialized': self._default_adapter is not None,
            'backend': self._config.default_backend.value,
            'loaded_models': list(self._adapters.keys()),
            'active_requests': len(self._active_requests),
            'helora_enabled': self._has_client is not None and self._has_client.is_connected,
            'has_stats': self._has_client.get_statistics() if self._has_client else None,
        }
