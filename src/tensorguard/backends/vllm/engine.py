"""TenSafe vLLM Engine with HE-LoRA Support.

High-throughput LLM inference with privacy-preserving LoRA computation.
"""

from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass
import asyncio
import time
import logging
import os

import torch

from .config import TenSafeVLLMConfig, HESchemeType
from .hooks import HELoRAHook, HELoRAHookManager, HELoRAConfig

# Setup logging
logger = logging.getLogger(__name__)

# Conditional vLLM imports
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not installed. Install with: pip install vllm")

# TenSafe imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from tensorguard.tgsp.service import TSGPService
    from tensorguard.tgsp.format import TSGPPackage
    TSSP_AVAILABLE = True
except ImportError:
    TSSP_AVAILABLE = False
    logger.warning("TSSP not available")

try:
    from tensorguard.platform.tg_tinker_api.audit import AuditLogger
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


@dataclass
class GenerationResult:
    """Result from a generation request."""
    request_id: str
    prompt: str
    outputs: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    privacy_info: Optional[Dict[str, Any]] = None


class TenSafeVLLMEngine:
    """vLLM engine with TenSafe privacy features.

    This engine wraps vLLM to provide:
    - High-throughput inference via PagedAttention and continuous batching
    - Privacy-preserving HE-LoRA injection
    - TSSP package verification
    - Audit trail logging
    - OpenAI-compatible API

    Example:
        ```python
        config = TenSafeVLLMConfig(
            model_path="meta-llama/Llama-3-8B",
            tssp_package_path="/path/to/adapter.tssp",
        )
        engine = TenSafeVLLMEngine(config)

        # Synchronous generation
        results = engine.generate(["Hello, world!"])

        # Async generation
        async for output in engine.generate_stream(["Hello, world!"]):
            print(output)
        ```
    """

    def __init__(self, config: TenSafeVLLMConfig):
        """Initialize TenSafe vLLM engine.

        Args:
            config: Engine configuration
        """
        self.config = config
        self._initialized = False

        # TSSP package
        self.tssp_package: Optional['TSGPPackage'] = None
        self.tssp_service: Optional['TSGPService'] = None

        # HE-LoRA
        self.he_lora_manager: Optional[HELoRAHookManager] = None
        self._lora_weights: Dict[str, tuple] = {}

        # Audit
        self.audit_logger = None

        # vLLM engine (lazy initialization)
        self._llm: Optional['LLM'] = None
        self._async_engine: Optional['AsyncLLMEngine'] = None

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._start_time = time.time()

        # Initialize components
        self._initialize()

    def _initialize(self):
        """Initialize engine components."""
        # Load TSSP package if provided
        if self.config.tssp_package_path and TSSP_AVAILABLE:
            self._load_tssp_package()

        # Initialize audit logger
        if self.config.enable_audit_logging and AUDIT_AVAILABLE:
            try:
                self.audit_logger = AuditLogger()
            except Exception as e:
                logger.warning(f"Failed to initialize audit logger: {e}")

        self._initialized = True

    def _load_tssp_package(self):
        """Load and verify TSSP package."""
        if not TSSP_AVAILABLE:
            raise RuntimeError("TSSP not available")

        logger.info(f"Loading TSSP package: {self.config.tssp_package_path}")

        self.tssp_service = TSGPService()

        # Load package
        self.tssp_package = self.tssp_service.load_package(
            self.config.tssp_package_path
        )

        # Verify package signatures
        verification = self.tssp_service.verify_package(self.tssp_package)
        if not verification.valid:
            raise ValueError(f"TSSP verification failed: {verification.reason}")

        logger.info(f"TSSP package verified: {self.tssp_package.manifest.package_id}")

        # Extract LoRA weights
        self._lora_weights = self._extract_lora_weights()

        # Log to audit trail
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="TSSP_PACKAGE_LOADED",
                package_id=self.tssp_package.manifest.package_id,
                verification_status="PASSED",
            )

    def _extract_lora_weights(self) -> Dict[str, tuple]:
        """Extract LoRA weights from TSSP package.

        Returns:
            Dict mapping layer names to (lora_a, lora_b) tuples
        """
        if not self.tssp_package:
            return {}

        weights = {}

        # This would normally decrypt and extract weights from TSSP
        # For now, create placeholder weights for testing
        for target in self.config.lora_target_modules:
            # Create random weights for simulation
            # In production, these come from encrypted TSSP payload
            hidden_size = 4096  # Default for Llama-3-8B
            rank = 16

            lora_a = torch.randn(rank, hidden_size) * 0.01
            lora_b = torch.zeros(hidden_size, rank)

            # Find all layers with this target
            for layer_idx in range(32):  # Typical transformer layers
                layer_name = f"model.layers.{layer_idx}.self_attn.{target}"
                weights[layer_name] = (lora_a.clone(), lora_b.clone())

        return weights

    def _get_llm(self) -> 'LLM':
        """Get or create synchronous vLLM engine."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed")

        if self._llm is None:
            logger.info("Initializing vLLM engine...")

            vllm_args = self.config.to_vllm_args()
            self._llm = LLM(**vllm_args)

            # Register HE-LoRA hooks if enabled
            if self.config.enable_he_lora and self._lora_weights:
                self._register_he_lora_hooks(self._llm.llm_engine.model_executor)

            logger.info("vLLM engine initialized")

        return self._llm

    async def _get_async_engine(self) -> 'AsyncLLMEngine':
        """Get or create async vLLM engine."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed")

        if self._async_engine is None:
            logger.info("Initializing async vLLM engine...")

            vllm_args = self.config.to_vllm_args()
            engine_args = EngineArgs(**vllm_args)
            self._async_engine = AsyncLLMEngine.from_engine_args(engine_args)

            logger.info("Async vLLM engine initialized")

        return self._async_engine

    def _register_he_lora_hooks(self, model_executor):
        """Register HE-LoRA hooks on model."""
        if not self._lora_weights:
            return

        he_config = HELoRAConfig(
            hidden_size=4096,
            rank=16,
            alpha=32.0,
            scheme=self.config.he_scheme.value,
            profile=self.config.ckks_profile.value,
            enable_metrics=True,
        )

        self.he_lora_manager = HELoRAHookManager(
            config=he_config,
            target_modules=self.config.lora_target_modules,
        )

        # Get model from executor
        try:
            model = model_executor.driver_worker.model_runner.model
            num_hooks = self.he_lora_manager.register_hooks(
                model=model,
                lora_weights=self._lora_weights,
            )
            logger.info(f"Registered {num_hooks} HE-LoRA hooks")
        except Exception as e:
            logger.warning(f"Failed to register HE-LoRA hooks: {e}")

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional['SamplingParams'] = None,
        use_tqdm: bool = False,
    ) -> List[GenerationResult]:
        """Generate completions synchronously.

        Args:
            prompts: List of prompts to generate from
            sampling_params: vLLM sampling parameters
            use_tqdm: Show progress bar

        Returns:
            List of generation results
        """
        if not VLLM_AVAILABLE:
            # Fallback simulation for testing
            return self._simulate_generation(prompts, sampling_params)

        start_time = time.perf_counter()

        # Default sampling params
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=self.config.max_tokens_per_request,
                temperature=0.7,
            )

        # Get engine and generate
        llm = self._get_llm()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

        # Convert to results
        results = []
        total_tokens = 0

        for i, output in enumerate(outputs):
            output_texts = []
            for completion in output.outputs:
                output_texts.append({
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "token_ids": list(completion.token_ids) if hasattr(completion, 'token_ids') else [],
                })
                total_tokens += len(completion.token_ids) if hasattr(completion, 'token_ids') else len(completion.text.split())

            result = GenerationResult(
                request_id=output.request_id,
                prompt=prompts[i],
                outputs=output_texts,
                metrics={
                    "latency_ms": (time.perf_counter() - start_time) * 1000,
                },
            )
            results.append(result)

        # Update metrics
        self._total_requests += len(prompts)
        self._total_tokens += total_tokens

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="INFERENCE_COMPLETE",
                num_prompts=len(prompts),
                total_tokens=total_tokens,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        return results

    async def generate_async(
        self,
        prompts: List[str],
        sampling_params: Optional['SamplingParams'] = None,
    ) -> List[GenerationResult]:
        """Generate completions asynchronously.

        Args:
            prompts: List of prompts
            sampling_params: Sampling parameters

        Returns:
            List of generation results
        """
        if not VLLM_AVAILABLE:
            return self._simulate_generation(prompts, sampling_params)

        # Default params
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=self.config.max_tokens_per_request,
                temperature=0.7,
            )

        engine = await self._get_async_engine()

        # Submit all requests
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_id = f"req-{i}-{time.time()}"
            await engine.add_request(request_id, prompt, sampling_params)
            request_ids.append(request_id)

        # Collect results
        results = []
        async for output in engine.generate(None):
            if output.finished:
                idx = request_ids.index(output.request_id)
                result = GenerationResult(
                    request_id=output.request_id,
                    prompt=prompts[idx],
                    outputs=[{
                        "text": o.text,
                        "finish_reason": o.finish_reason,
                    } for o in output.outputs],
                    metrics={},
                )
                results.append(result)

                if len(results) == len(prompts):
                    break

        return results

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional['SamplingParams'] = None,
    ) -> AsyncIterator[str]:
        """Stream generation token by token.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters

        Yields:
            Generated tokens
        """
        if not VLLM_AVAILABLE:
            # Simulate streaming
            for word in "This is a simulated response.".split():
                yield word + " "
                await asyncio.sleep(0.1)
            return

        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=self.config.max_tokens_per_request,
                temperature=0.7,
            )

        engine = await self._get_async_engine()
        request_id = f"stream-{time.time()}"

        await engine.add_request(request_id, prompt, sampling_params)

        prev_text = ""
        async for output in engine.generate(None):
            if output.request_id == request_id:
                for completion in output.outputs:
                    new_text = completion.text[len(prev_text):]
                    if new_text:
                        yield new_text
                        prev_text = completion.text

                if output.finished:
                    break

    def _simulate_generation(
        self,
        prompts: List[str],
        sampling_params: Optional[Any] = None,
    ) -> List[GenerationResult]:
        """Simulate generation when vLLM is not available.

        This is used for testing the integration code without GPU.
        """
        results = []
        for i, prompt in enumerate(prompts):
            simulated_response = f"[Simulated response to: {prompt[:50]}...] This is a test response from the TenSafe vLLM integration layer."

            result = GenerationResult(
                request_id=f"sim-{i}",
                prompt=prompt,
                outputs=[{
                    "text": simulated_response,
                    "finish_reason": "stop",
                    "token_ids": [],
                }],
                metrics={"simulated": True},
            )
            results.append(result)

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics.

        Returns:
            Dict containing performance and privacy metrics
        """
        uptime = time.time() - self._start_time

        metrics = {
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
            "tokens_per_second": self._total_tokens / uptime if uptime > 0 else 0,
        }

        # Add HE-LoRA metrics
        if self.he_lora_manager:
            metrics["he_lora"] = self.he_lora_manager.get_aggregate_metrics()

        return metrics

    def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        logger.info("Shutting down TenSafe vLLM engine...")

        # Remove HE-LoRA hooks
        if self.he_lora_manager:
            self.he_lora_manager.remove_hooks()

        # Cleanup vLLM
        self._llm = None
        self._async_engine = None

        logger.info("TenSafe vLLM engine shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
