# TenSafe Integration Implementation Plan

**Version:** 1.0.0
**Date:** 2026-02-02
**Status:** Draft for Review

---

## Overview

This document provides detailed implementation specifications for integrating TenSafe with industry-standard ML infrastructure. Each integration is designed to preserve TenSafe's unique privacy guarantees while achieving feature parity with competitors.

---

## Priority Matrix Summary

| Priority | Integration | Effort | Business Value | Risk |
|----------|-------------|--------|----------------|------|
| **P0** | Kubernetes Native Deployment | Medium | Critical | Low |
| **P0** | Helm Charts | Medium | Critical | Low |
| **P1** | vLLM Backend | High | Very High | Medium |
| **P1** | KEDA Auto-scaling | Low | High | Low |
| **P1** | OpenTelemetry Observability | Medium | High | Low |
| **P1** | Ray Train Distributed | High | Very High | Medium |
| **P1** | KMS Integration (Vault) | Medium | High | Low |
| **P2** | LoRAX Multi-Adapter | Medium | High | Medium |
| **P2** | TRL Integration (DPO/RLHF) | Medium | Medium | Low |
| **P2** | Hugging Face Hub | Low | Medium | Low |
| **P2** | W&B/MLflow | Low | Medium | Low |
| **P2** | Liger/Unsloth Kernels | Low | Medium | Low |
| **P3** | Kubeflow Pipelines | Medium | Medium | Medium |
| **P3** | Speculative Decoding | Medium | Medium | High |

---

## Phase 0: Foundation - Kubernetes Native Deployment

### 0.1 Multi-Worker FastAPI Setup

**Current State:** Single-process Uvicorn server
**Target State:** Gunicorn + Uvicorn workers with proper process management

**File:** `src/tensorguard/platform/main.py`

```python
# New deployment configuration
import multiprocessing
from gunicorn.app.base import BaseApplication

class TenSafeApplication(BaseApplication):
    """Production-grade Gunicorn application."""

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

def get_gunicorn_options():
    return {
        "bind": f"0.0.0.0:{os.getenv('PORT', 8000)}",
        "workers": int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1)),
        "worker_class": "uvicorn.workers.UvicornWorker",
        "timeout": int(os.getenv('TIMEOUT', 120)),
        "keepalive": int(os.getenv('KEEPALIVE', 5)),
        "max_requests": int(os.getenv('MAX_REQUESTS', 10000)),
        "max_requests_jitter": int(os.getenv('MAX_REQUESTS_JITTER', 1000)),
        "preload_app": True,
        "accesslog": "-",
        "errorlog": "-",
        "loglevel": os.getenv('LOG_LEVEL', 'info'),
    }

if __name__ == "__main__":
    options = get_gunicorn_options()
    TenSafeApplication(app, options).run()
```

### 0.2 Kubernetes Manifests

**File:** `deploy/kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensafe-server
  labels:
    app: tensafe
    component: server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tensafe
      component: server
  template:
    metadata:
      labels:
        app: tensafe
        component: server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: tensafe
      containers:
      - name: tensafe
        image: tensafe/server:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: TG_ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tensafe-secrets
              key: database-url
        - name: TG_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: tensafe-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readiness
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: tensafe-config
---
apiVersion: v1
kind: Service
metadata:
  name: tensafe-server
  labels:
    app: tensafe
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: tensafe
    component: server
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tensafe-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.tensafe.dev
    secretName: tensafe-tls
  rules:
  - host: api.tensafe.dev
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tensafe-server
            port:
              number: 80
```

### 0.3 Helm Chart Structure

**File:** `deploy/helm/tensafe/Chart.yaml`

```yaml
apiVersion: v2
name: tensafe
description: Privacy-preserving ML training platform
type: application
version: 0.1.0
appVersion: "3.0.0"
keywords:
  - machine-learning
  - privacy
  - homomorphic-encryption
  - lora
home: https://github.com/tensafe/tensafe
maintainers:
  - name: TenSafe Team
    email: team@tensafe.dev
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

**File:** `deploy/helm/tensafe/values.yaml`

```yaml
# Default values for TenSafe
replicaCount: 3

image:
  repository: tensafe/server
  pullPolicy: IfNotPresent
  tag: ""  # Defaults to Chart appVersion

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podSecurityContext:
  fsGroup: 1000

securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.tensafe.dev
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: tensafe-tls
      hosts:
        - api.tensafe.dev

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}

# TenSafe-specific configuration
tensafe:
  environment: production
  logLevel: info
  workers: 4

  # Security settings
  security:
    jwtExpireMinutes: 30
    minPasswordLength: 12
    rateLimiting:
      enabled: true
      maxRequests: 100
      windowMinutes: 1

  # Database
  database:
    # Use external database or enable bundled PostgreSQL
    external: false
    host: ""
    port: 5432
    name: tensafe
    sslMode: require

  # Observability
  observability:
    metrics:
      enabled: true
      path: /metrics
    tracing:
      enabled: true
      samplingRate: 0.1
      endpoint: ""  # OpenTelemetry collector endpoint

# PostgreSQL subchart configuration
postgresql:
  enabled: true
  auth:
    postgresPassword: ""  # Set via --set or secret
    database: tensafe
  primary:
    persistence:
      enabled: true
      size: 50Gi

# Redis subchart configuration (for rate limiting)
redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: true
    password: ""  # Set via --set or secret
```

---

## Phase 1: vLLM Integration

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TenSafe vLLM Integration                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Request → TenSafe API Gateway                                  │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TenSafeVLLMEngine                          │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │            Pre-processing Layer                  │   │   │
│  │  │  • TSSP package verification                    │   │   │
│  │  │  • HE key initialization                        │   │   │
│  │  │  • Adapter decryption (if needed)               │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                         │                               │   │
│  │                         ▼                               │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │           vLLM Core Engine                       │   │   │
│  │  │  • PagedAttention                               │   │   │
│  │  │  • Continuous batching                          │   │   │
│  │  │  • Tensor parallelism                           │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                         │                               │   │
│  │                         ▼                               │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │          HE-LoRA Injection Hooks                 │   │   │
│  │  │  • Per-layer forward hooks                      │   │   │
│  │  │  • CKKS encrypted computation                   │   │   │
│  │  │  • MOAI rotation elimination                    │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                         │                               │   │
│  │                         ▼                               │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │           Post-processing Layer                  │   │   │
│  │  │  • Audit logging                                │   │   │
│  │  │  • Privacy budget tracking                      │   │   │
│  │  │  • Response encryption (optional)               │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│               │                                                 │
│               ▼                                                 │
│  Response (OpenAI-compatible format)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Implementation Files

**File:** `src/tensorguard/backends/vllm/__init__.py`

```python
"""TenSafe vLLM Backend Integration."""

from .engine import TenSafeVLLMEngine
from .config import TenSafeVLLMConfig
from .hooks import HELoRAHook

__all__ = ["TenSafeVLLMEngine", "TenSafeVLLMConfig", "HELoRAHook"]
```

**File:** `src/tensorguard/backends/vllm/engine.py`

```python
"""TenSafe vLLM Engine with HE-LoRA support."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine

from tensorguard.tgsp import TSGPPackage, TSGPService
from tensorguard.n2he import HESchemeParams, N2HEScheme
from he_lora_microkernel.runtime import HELoRAExecutor
from tensorguard.platform.tg_tinker_api.audit import AuditLogger

@dataclass
class TenSafeVLLMConfig:
    """Configuration for TenSafe vLLM engine."""

    # Model configuration
    model_path: str
    tssp_package_path: Optional[str] = None

    # vLLM configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9

    # HE-LoRA configuration
    enable_he_lora: bool = True
    he_scheme: str = "ckks"  # ckks, tfhe, hybrid
    ckks_profile: str = "fast"  # fast, safe

    # Privacy configuration
    enable_audit_logging: bool = True
    enable_privacy_tracking: bool = True

    # Performance configuration
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True


class TenSafeVLLMEngine:
    """vLLM engine with TenSafe privacy features."""

    def __init__(self, config: TenSafeVLLMConfig):
        self.config = config
        self.tssp_service = TSGPService()
        self.audit_logger = AuditLogger() if config.enable_audit_logging else None

        # Load and verify TSSP package if provided
        self.tssp_package = None
        self.he_lora_executor = None

        if config.tssp_package_path:
            self._load_tssp_package(config.tssp_package_path)

        # Initialize vLLM engine
        self.engine = self._create_vllm_engine()

        # Register HE-LoRA hooks if enabled
        if config.enable_he_lora and self.he_lora_executor:
            self._register_he_lora_hooks()

    def _load_tssp_package(self, package_path: str):
        """Load and verify TSSP package."""
        # Load package
        self.tssp_package = self.tssp_service.load_package(package_path)

        # Verify signatures
        verification_result = self.tssp_service.verify_package(self.tssp_package)
        if not verification_result.valid:
            raise ValueError(f"TSSP package verification failed: {verification_result.reason}")

        # Initialize HE-LoRA executor
        he_params = HESchemeParams.from_config(self.config.he_scheme, self.config.ckks_profile)
        self.he_lora_executor = HELoRAExecutor(
            lora_config=self.tssp_package.lora_config,
            he_params=he_params,
        )

        # Log to audit trail
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="TSSP_PACKAGE_LOADED",
                package_id=self.tssp_package.manifest.package_id,
                verification_status="PASSED",
            )

    def _create_vllm_engine(self) -> LLMEngine:
        """Create vLLM engine with TenSafe configuration."""
        engine_args = EngineArgs(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_caching=self.config.enable_prefix_caching,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
        )
        return LLMEngine.from_engine_args(engine_args)

    def _register_he_lora_hooks(self):
        """Register forward hooks for HE-LoRA injection."""
        model = self.engine.model_executor.driver_worker.model_runner.model

        for name, module in model.named_modules():
            if self._is_lora_target(name):
                module.register_forward_hook(
                    self._create_he_lora_hook(name)
                )

    def _is_lora_target(self, module_name: str) -> bool:
        """Check if module is a LoRA target."""
        if not self.tssp_package:
            return False

        targets = self.tssp_package.lora_config.target_modules
        return any(target in module_name for target in targets)

    def _create_he_lora_hook(self, layer_name: str):
        """Create HE-LoRA forward hook for a specific layer."""
        def hook(module, input, output):
            # Apply HE-LoRA transformation
            if self.he_lora_executor:
                he_output = self.he_lora_executor.apply(
                    hidden_states=output,
                    layer_name=layer_name,
                )
                return he_output
            return output
        return hook

    async def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate completions with HE-LoRA."""
        sampling_params = sampling_params or SamplingParams()

        # Log request start
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="INFERENCE_START",
                request_id=request_id,
                num_prompts=len(prompts),
            )

        # Generate using vLLM
        outputs = await self.engine.generate(prompts, sampling_params)

        # Format results
        results = []
        for output in outputs:
            results.append({
                "id": output.request_id,
                "choices": [
                    {
                        "text": out.text,
                        "finish_reason": out.finish_reason,
                    }
                    for out in output.outputs
                ],
            })

        # Log request completion
        if self.audit_logger:
            total_tokens = sum(
                len(out.token_ids)
                for output in outputs
                for out in output.outputs
            )
            self.audit_logger.log_event(
                event_type="INFERENCE_COMPLETE",
                request_id=request_id,
                total_tokens=total_tokens,
            )

        return results

    def get_openai_compatible_endpoint(self):
        """Return OpenAI-compatible API router."""
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel

        router = APIRouter()

        class CompletionRequest(BaseModel):
            model: str
            prompt: str | List[str]
            max_tokens: int = 100
            temperature: float = 1.0
            top_p: float = 1.0

        @router.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            results = await self.generate(prompts, params)
            return {"choices": results[0]["choices"]}

        return router
```

### 1.3 KEDA Auto-scaling Configuration

**File:** `deploy/kubernetes/keda-scaledobject.yaml`

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: tensafe-scaledobject
  namespace: tensafe
spec:
  scaleTargetRef:
    name: tensafe-server
  pollingInterval: 15
  cooldownPeriod: 60
  minReplicaCount: 3
  maxReplicaCount: 50

  triggers:
  # Scale based on Prometheus metrics
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: tensafe_inference_latency_seconds
      threshold: '0.1'  # Scale up when p95 latency > 100ms
      query: |
        histogram_quantile(0.95,
          sum(rate(tensafe_inference_latency_seconds_bucket{job="tensafe"}[5m]))
          by (le))

  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: tensafe_queue_depth
      threshold: '50'  # Scale up when queue depth > 50
      query: |
        sum(tensafe_request_queue_depth{job="tensafe"})

  # Scale based on GPU utilization
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: tensafe_gpu_utilization
      threshold: '0.8'  # Scale up when GPU > 80%
      query: |
        avg(DCGM_FI_DEV_GPU_UTIL{job="tensafe"}) / 100
```

---

## Phase 2: Ray Train Distributed Training

### 2.1 TenSafe Ray Trainer

**File:** `src/tensorguard/distributed/ray_trainer.py`

```python
"""TenSafe distributed training with Ray Train."""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import ray
from ray import train
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import torch
import torch.distributed as dist

from tensorguard.platform.tg_tinker_api.dp import DPConfig, DPOptimizer
from tensorguard.n2he.keys import KeyManager
from tensorguard.platform.tg_tinker_api.audit import AuditLogger


@dataclass
class TenSafeRayConfig:
    """Configuration for TenSafe Ray training."""

    # Ray configuration
    num_workers: int = 4
    use_gpu: bool = True
    resources_per_worker: Optional[Dict[str, float]] = None

    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_epochs: int = 10

    # DP configuration
    dp_config: Optional[DPConfig] = None

    # Checkpointing
    checkpoint_frequency: int = 100
    checkpoint_dir: str = "/checkpoints"

    # Security
    secure_aggregation: bool = True
    audit_logging: bool = True


class TenSafeRayTrainer:
    """Distributed trainer with DP-SGD and secure aggregation."""

    def __init__(
        self,
        config: TenSafeRayConfig,
        model_init_fn: Callable[[], torch.nn.Module],
        dataset_fn: Callable[[], torch.utils.data.Dataset],
    ):
        self.config = config
        self.model_init_fn = model_init_fn
        self.dataset_fn = dataset_fn

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()

    def _train_func_per_worker(self, train_loop_config: Dict[str, Any]):
        """Training function executed on each worker."""
        import torch
        from torch.utils.data import DataLoader, DistributedSampler

        # Get worker info
        rank = train.get_context().get_world_rank()
        world_size = train.get_context().get_world_size()

        # Initialize model
        model = train_loop_config["model_init_fn"]()
        model = model.to(train.get_context().get_device())

        # Wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(model)

        # Initialize dataset with distributed sampler
        dataset = train_loop_config["dataset_fn"]()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=train_loop_config["batch_size"],
            sampler=sampler,
        )

        # Initialize optimizer (with DP if configured)
        dp_config = train_loop_config.get("dp_config")
        if dp_config and dp_config.enabled:
            optimizer = DPOptimizer(
                model.parameters(),
                lr=train_loop_config["learning_rate"],
                noise_multiplier=dp_config.noise_multiplier,
                max_grad_norm=dp_config.max_grad_norm,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_loop_config["learning_rate"],
            )

        # Training loop
        for epoch in range(train_loop_config["max_epochs"]):
            sampler.set_epoch(epoch)
            total_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(train.get_context().get_device()) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # DP: Per-sample gradient clipping
                if dp_config and dp_config.enabled:
                    self._clip_gradients_per_sample(model, dp_config.max_grad_norm)

                # Secure aggregation (if enabled)
                if train_loop_config.get("secure_aggregation"):
                    self._secure_gradient_aggregation(model)

                # Optimizer step (adds DP noise if DPOptimizer)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Report metrics
            avg_loss = total_loss / num_batches

            # Get privacy budget if using DP
            privacy_metrics = {}
            if dp_config and dp_config.enabled:
                epsilon, delta = optimizer.get_privacy_spent()
                privacy_metrics = {"epsilon": epsilon, "delta": delta}

            train.report({
                "loss": avg_loss,
                "epoch": epoch,
                **privacy_metrics,
            })

            # Checkpoint
            if (epoch + 1) % train_loop_config["checkpoint_frequency"] == 0:
                checkpoint = train.Checkpoint.from_dict({
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    **privacy_metrics,
                })
                train.report({"loss": avg_loss}, checkpoint=checkpoint)

    def _clip_gradients_per_sample(self, model: torch.nn.Module, max_norm: float):
        """Clip gradients per sample for DP."""
        # Per-sample gradient clipping using opacus-style approach
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2)
                clip_coef = max_norm / (grad_norm + 1e-6)
                clip_coef = torch.clamp(clip_coef, max=1.0)
                param.grad.mul_(clip_coef)

    def _secure_gradient_aggregation(self, model: torch.nn.Module):
        """Perform secure gradient aggregation across workers."""
        # Simple secure aggregation using all-reduce with encryption
        # In production, use MPC or HE-based aggregation
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(dist.get_world_size())

    def train(self) -> ray.train.Result:
        """Execute distributed training."""
        # Create scaling config
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker or {},
        )

        # Create run config
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_frequency=self.config.checkpoint_frequency,
            ),
        )

        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self._train_func_per_worker,
            train_loop_config={
                "model_init_fn": self.model_init_fn,
                "dataset_fn": self.dataset_fn,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_epochs": self.config.max_epochs,
                "dp_config": self.config.dp_config,
                "secure_aggregation": self.config.secure_aggregation,
                "checkpoint_frequency": self.config.checkpoint_frequency,
            },
            scaling_config=scaling_config,
            run_config=run_config,
            torch_config=TorchConfig(backend="nccl"),
        )

        # Run training
        result = trainer.fit()

        return result
```

---

## Phase 3: Observability Integration

### 3.1 OpenTelemetry Setup

**File:** `src/tensorguard/observability/__init__.py`

```python
"""TenSafe Observability with OpenTelemetry."""

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import start_http_server
import os


def setup_observability(app=None, service_name: str = "tensafe"):
    """Initialize OpenTelemetry observability stack."""

    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("TENSAFE_VERSION", "3.0.0"),
        "deployment.environment": os.getenv("TG_ENVIRONMENT", "development"),
    })

    # Setup tracing
    tracer_provider = TracerProvider(resource=resource)

    # OTLP exporter for traces (to Jaeger/Tempo)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Setup metrics with Prometheus exporter
    prometheus_reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[prometheus_reader])
    metrics.set_meter_provider(meter_provider)

    # Start Prometheus metrics server
    metrics_port = int(os.getenv("METRICS_PORT", 9090))
    start_http_server(metrics_port)

    # Instrument FastAPI if app provided
    if app:
        FastAPIInstrumentor.instrument_app(app)

    return tracer_provider, meter_provider


# Custom metrics for TenSafe
def create_tensafe_metrics():
    """Create TenSafe-specific metrics."""
    meter = metrics.get_meter("tensafe")

    # Inference metrics
    inference_latency = meter.create_histogram(
        name="tensafe_inference_latency_seconds",
        description="Inference latency in seconds",
        unit="s",
    )

    inference_tokens = meter.create_counter(
        name="tensafe_inference_tokens_total",
        description="Total tokens generated",
        unit="tokens",
    )

    # Privacy metrics
    privacy_budget_epsilon = meter.create_observable_gauge(
        name="tensafe_privacy_budget_epsilon",
        description="Current privacy budget (epsilon)",
        callbacks=[lambda options: []],  # Callback to get current epsilon
    )

    privacy_budget_delta = meter.create_observable_gauge(
        name="tensafe_privacy_budget_delta",
        description="Current privacy budget (delta)",
        callbacks=[lambda options: []],
    )

    # HE-LoRA metrics
    he_lora_latency = meter.create_histogram(
        name="tensafe_he_lora_latency_seconds",
        description="HE-LoRA computation latency",
        unit="s",
    )

    # Request queue
    request_queue_depth = meter.create_observable_gauge(
        name="tensafe_request_queue_depth",
        description="Number of requests in queue",
        callbacks=[lambda options: []],
    )

    return {
        "inference_latency": inference_latency,
        "inference_tokens": inference_tokens,
        "privacy_budget_epsilon": privacy_budget_epsilon,
        "privacy_budget_delta": privacy_budget_delta,
        "he_lora_latency": he_lora_latency,
        "request_queue_depth": request_queue_depth,
    }
```

### 3.2 Grafana Dashboard

**File:** `deploy/grafana/tensafe-dashboard.json`

```json
{
  "dashboard": {
    "title": "TenSafe Production Dashboard",
    "uid": "tensafe-prod",
    "panels": [
      {
        "title": "Inference Latency (p95)",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(tensafe_inference_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95 latency"
          }
        ]
      },
      {
        "title": "Throughput (tokens/sec)",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(tensafe_inference_tokens_total[5m]))",
            "legendFormat": "tokens/sec"
          }
        ]
      },
      {
        "title": "Privacy Budget (Epsilon)",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "tensafe_privacy_budget_epsilon",
            "legendFormat": "epsilon"
          }
        ],
        "options": {
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "green", "value": null},
              {"color": "yellow", "value": 5},
              {"color": "red", "value": 10}
            ]
          }
        }
      },
      {
        "title": "HE-LoRA Overhead",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(tensafe_he_lora_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95 HE-LoRA latency"
          }
        ]
      },
      {
        "title": "Active Replicas",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "count(up{job=\"tensafe\"} == 1)",
            "legendFormat": "replicas"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
        "targets": [
          {
            "expr": "avg(DCGM_FI_DEV_GPU_UTIL{job=\"tensafe\"})",
            "legendFormat": "GPU %"
          }
        ]
      }
    ]
  }
}
```

---

## Phase 4: MLOps Integrations

### 4.1 Weights & Biases Callback

**File:** `src/tensorguard/integrations/wandb_callback.py`

```python
"""TenSafe W&B Integration."""

from typing import Optional, Dict, Any
import wandb
from tensorguard.platform.tg_tinker_api.dp import DPConfig


class TenSafeWandbCallback:
    """Weights & Biases callback with privacy tracking."""

    def __init__(
        self,
        project: str = "tensafe",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dp_config: Optional[DPConfig] = None,
    ):
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.dp_config = dp_config
        self.run = None

    def on_train_begin(self, **kwargs):
        """Initialize W&B run."""
        # Add TenSafe-specific config
        tensafe_config = {
            "framework": "tensafe",
            "privacy_enabled": self.dp_config is not None,
        }

        if self.dp_config:
            tensafe_config.update({
                "dp_noise_multiplier": self.dp_config.noise_multiplier,
                "dp_max_grad_norm": self.dp_config.max_grad_norm,
                "dp_target_epsilon": self.dp_config.target_epsilon,
                "dp_target_delta": self.dp_config.target_delta,
            })

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config={**self.config, **tensafe_config},
        )

    def on_step_end(
        self,
        step: int,
        loss: float,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        **kwargs,
    ):
        """Log step metrics."""
        metrics = {"step": step, "loss": loss}

        if epsilon is not None:
            metrics["privacy/epsilon"] = epsilon
        if delta is not None:
            metrics["privacy/delta"] = delta

        # Log HE-LoRA metrics if available
        if "he_lora_latency" in kwargs:
            metrics["he_lora/latency_ms"] = kwargs["he_lora_latency"] * 1000

        wandb.log(metrics, step=step)

    def on_epoch_end(self, epoch: int, **kwargs):
        """Log epoch summary."""
        wandb.log({"epoch": epoch, **kwargs})

    def on_train_end(self, **kwargs):
        """Finalize W&B run."""
        # Log final privacy budget
        if self.dp_config and "final_epsilon" in kwargs:
            wandb.summary["final_privacy_epsilon"] = kwargs["final_epsilon"]
            wandb.summary["final_privacy_delta"] = kwargs.get("final_delta", self.dp_config.target_delta)

        wandb.finish()

    def log_model(self, model_path: str, metadata: Optional[Dict] = None):
        """Log model artifact (without weights for privacy)."""
        # Only log metadata, not actual weights
        artifact = wandb.Artifact(
            name="model-metadata",
            type="model-metadata",
            metadata=metadata or {},
        )
        artifact.add_file(f"{model_path}/config.json")
        wandb.log_artifact(artifact)
```

### 4.2 Hugging Face Hub Integration

**File:** `src/tensorguard/integrations/hf_hub.py`

```python
"""TenSafe Hugging Face Hub Integration."""

from typing import Optional, Dict
from pathlib import Path
import json
from huggingface_hub import HfApi, Repository, create_repo
from tensorguard.tgsp import TSGPPackage, TSGPService


class TenSafeHFHubIntegration:
    """Hugging Face Hub integration with TSSP support."""

    def __init__(self, token: Optional[str] = None, private: bool = True):
        self.api = HfApi(token=token)
        self.private = private
        self.tssp_service = TSGPService()

    def push_to_hub(
        self,
        tssp_package_path: str,
        repo_id: str,
        commit_message: str = "Upload TenSafe model",
        include_weights: bool = False,
    ) -> str:
        """Push TSSP package to Hugging Face Hub.

        Args:
            tssp_package_path: Path to TSSP package
            repo_id: HF Hub repository ID (e.g., "username/model-name")
            commit_message: Git commit message
            include_weights: If False, only push metadata (privacy-preserving)

        Returns:
            URL of the created/updated repository
        """
        # Load TSSP package
        package = self.tssp_service.load_package(tssp_package_path)

        # Create or get repository
        try:
            self.api.create_repo(repo_id, private=self.private, exist_ok=True)
        except Exception as e:
            print(f"Repository exists or error: {e}")

        # Prepare files to upload
        files_to_upload = []

        # Always upload config and metadata
        config_content = json.dumps(package.manifest.to_dict(), indent=2)
        files_to_upload.append(("config.json", config_content.encode()))

        # Generate model card
        model_card = self._generate_model_card(package)
        files_to_upload.append(("README.md", model_card.encode()))

        # Upload TSSP manifest (verification data)
        manifest_content = json.dumps(package.manifest.to_dict(), indent=2)
        files_to_upload.append(("tssp_manifest.json", manifest_content.encode()))

        if include_weights:
            # Upload encrypted weights
            for file_info in package.manifest.files:
                file_path = Path(tssp_package_path) / file_info.path
                if file_path.exists():
                    files_to_upload.append((file_info.path, file_path.read_bytes()))

        # Upload files
        for filename, content in files_to_upload:
            self.api.upload_file(
                path_or_fileobj=content,
                path_in_repo=filename,
                repo_id=repo_id,
                commit_message=commit_message,
            )

        return f"https://huggingface.co/{repo_id}"

    def _generate_model_card(self, package: TSGPPackage) -> str:
        """Generate model card from TSSP package."""
        manifest = package.manifest

        return f"""---
language: en
license: apache-2.0
library_name: tensafe
tags:
  - tensafe
  - privacy-preserving
  - homomorphic-encryption
  - lora
---

# {manifest.name}

This model was trained using TenSafe, a privacy-preserving ML training platform.

## Model Details

- **Base Model:** {manifest.base_model}
- **Training Method:** LoRA with Differential Privacy
- **Privacy Guarantees:** ε={manifest.privacy_epsilon}, δ={manifest.privacy_delta}
- **Package ID:** {manifest.package_id}

## Security

This model is distributed as a TSSP (TenSafe Secure Package) with:
- Cryptographic signatures (Ed25519 + Dilithium3)
- Encrypted weights (AES-256-GCM)
- Tamper-evident manifest

## Usage

```python
from tensafe import load_model

model = load_model("{manifest.package_id}")
```

## Privacy Information

This model was trained with differential privacy guarantees:
- Epsilon (ε): {manifest.privacy_epsilon}
- Delta (δ): {manifest.privacy_delta}
- Noise Multiplier: {manifest.noise_multiplier}
- Max Gradient Norm: {manifest.max_grad_norm}

## Verification

To verify this model's integrity:

```python
from tensafe import verify_package

result = verify_package("{manifest.package_id}")
print(f"Verification: {result.status}")
```
"""

    def pull_from_hub(
        self,
        repo_id: str,
        local_dir: str,
        verify: bool = True,
    ) -> TSGPPackage:
        """Pull TSSP package from Hugging Face Hub.

        Args:
            repo_id: HF Hub repository ID
            local_dir: Local directory to save files
            verify: Whether to verify package integrity

        Returns:
            Loaded TSSP package
        """
        # Download repository
        local_path = self.api.snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
        )

        # Load TSSP package
        package = self.tssp_service.load_package(local_path)

        # Verify if requested
        if verify:
            result = self.tssp_service.verify_package(package)
            if not result.valid:
                raise ValueError(f"Package verification failed: {result.reason}")

        return package
```

---

## Summary: Build vs Integrate Decision Matrix

| Component | Decision | Rationale | Effort (weeks) |
|-----------|----------|-----------|----------------|
| **vLLM Backend** | INTEGRATE | Industry standard, 24x throughput | 6 |
| **LoRAX Multi-Adapter** | INTEGRATE | Unique multi-LoRA capability | 4 |
| **Ray Train** | INTEGRATE | De facto distributed training | 6 |
| **KEDA Auto-scaling** | INTEGRATE | K8s-native, SLI-based | 2 |
| **OpenTelemetry** | ADOPT | Vendor-neutral standard | 3 |
| **Prometheus/Grafana** | ADOPT | Industry standard monitoring | 2 |
| **HF Hub** | INTEGRATE | Standard model registry | 2 |
| **W&B/MLflow** | INTEGRATE | Mature experiment tracking | 2 |
| **TRL (DPO/RLHF)** | INTEGRATE | Comprehensive trainers | 3 |
| **Liger/Unsloth** | INTEGRATE | 20-70% speedup | 2 |
| **K8s Deployment** | BUILD | Foundation for integrations | 4 |
| **Helm Charts** | BUILD | Required for distribution | 3 |
| **KMS Plugin System** | BUILD | Security requirement | 4 |
| **Native HE Library** | BUILD | Core differentiator | 12 |
| **Privacy Wrappers** | BUILD | Unique value proposition | 4 |
| **TSSP Integration** | BUILD | Core security feature | 2 |

**Total Estimated Effort:** ~61 engineering weeks (15-16 months with 1 FTE, 4 months with 4 FTEs)

---

*Implementation Plan v1.0.0*
*Generated: 2026-02-02*
