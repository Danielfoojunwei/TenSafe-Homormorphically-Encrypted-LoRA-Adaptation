# Kubernetes Deployment Guide

**Version**: 4.0.0
**Last Updated**: 2026-02-02

This guide covers deploying TenSafe to Kubernetes using the official Helm chart.

## Overview

TenSafe's Kubernetes deployment provides:
- **Production-Ready Helm Chart**: Configurable deployment templates
- **KEDA Auto-scaling**: SLI-based horizontal pod autoscaling
- **Database Integration**: PostgreSQL and Redis subcharts
- **Secret Management**: HashiCorp Vault integration
- **GPU Support**: NVIDIA device plugin scheduling

## Prerequisites

```bash
# Kubernetes cluster (1.24+)
kubectl version --client

# Helm 3.0+
helm version

# Optional: KEDA for auto-scaling
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml

# Optional: NVIDIA device plugin for GPU support
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

## Quick Start

### Install from Helm Chart

```bash
# Add TenSafe Helm repository
helm repo add tensafe https://tensafe.github.io/helm-charts
helm repo update

# Install with default values
helm install tensafe tensafe/tensafe \
  --namespace tensafe \
  --create-namespace

# Or install from local chart
helm install tensafe ./deploy/helm/tensafe \
  --namespace tensafe \
  --create-namespace
```

### Verify Installation

```bash
# Check pods
kubectl get pods -n tensafe

# Check services
kubectl get svc -n tensafe

# Check ingress
kubectl get ingress -n tensafe
```

## Helm Chart Structure

```
deploy/helm/tensafe/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Default configuration
├── templates/
│   ├── _helpers.tpl        # Template helpers
│   ├── deployment.yaml     # Main deployment
│   ├── service.yaml        # Service definitions
│   ├── ingress.yaml        # Ingress rules
│   ├── configmap.yaml      # Configuration
│   ├── secrets.yaml        # Secret references
│   ├── serviceaccount.yaml # RBAC
│   ├── hpa.yaml            # Horizontal Pod Autoscaler
│   └── keda-scaledobject.yaml # KEDA auto-scaling
└── charts/                 # Subcharts
    ├── postgresql/
    └── redis/
```

## Configuration

### values.yaml Overview

```yaml
# Image configuration
image:
  repository: tensafe/tensafe
  tag: "4.0.0"
  pullPolicy: IfNotPresent

# Replicas
replicaCount: 2

# Resources
resources:
  limits:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: 1
  requests:
    cpu: "2"
    memory: "8Gi"

# Service configuration
service:
  type: ClusterIP
  port: 8000

# Ingress
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: tensafe.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: tensafe-tls
      hosts:
        - tensafe.example.com

# Auto-scaling
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

# KEDA scaling
keda:
  enabled: true
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: tensafe_inference_latency_p95
        threshold: "100"
        query: histogram_quantile(0.95, sum(rate(tensafe_inference_latency_seconds_bucket[5m])) by (le))

# PostgreSQL
postgresql:
  enabled: true
  auth:
    database: tensafe
    username: tensafe
    existingSecret: tensafe-db-secret
  primary:
    persistence:
      size: 100Gi

# Redis
redis:
  enabled: true
  architecture: standalone
  auth:
    existingSecret: tensafe-redis-secret

# Vault integration
vault:
  enabled: true
  address: https://vault.example.com
  role: tensafe
  secretPath: secret/data/tensafe

# Observability
observability:
  enabled: true
  prometheus:
    serviceMonitor:
      enabled: true
  jaeger:
    enabled: true
    agent:
      host: jaeger-agent
      port: 6831
```

### Environment-Specific Values

#### Development

```yaml
# values-dev.yaml
replicaCount: 1

resources:
  limits:
    cpu: "2"
    memory: "4Gi"
  requests:
    cpu: "1"
    memory: "2Gi"

postgresql:
  enabled: true
  primary:
    persistence:
      size: 10Gi

autoscaling:
  enabled: false
```

#### Production

```yaml
# values-prod.yaml
replicaCount: 3

resources:
  limits:
    cpu: "8"
    memory: "32Gi"
    nvidia.com/gpu: 2
  requests:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: 2

postgresql:
  enabled: true
  architecture: replication
  primary:
    persistence:
      size: 500Gi
  readReplicas:
    replicaCount: 2

redis:
  architecture: replication
  replica:
    replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 50

keda:
  enabled: true
```

### Installing with Custom Values

```bash
# Development
helm install tensafe ./deploy/helm/tensafe \
  -f values-dev.yaml \
  --namespace tensafe-dev

# Production
helm install tensafe ./deploy/helm/tensafe \
  -f values-prod.yaml \
  --namespace tensafe-prod \
  --set postgresql.auth.password=$DB_PASSWORD \
  --set redis.auth.password=$REDIS_PASSWORD
```

## Auto-scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
# In values.yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### KEDA SLI-Based Scaling

```yaml
# In values.yaml
keda:
  enabled: true
  pollingInterval: 30
  cooldownPeriod: 300
  triggers:
    # Scale on P95 latency
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: tensafe_inference_latency_p95
        threshold: "100"  # 100ms
        query: |
          histogram_quantile(0.95,
            sum(rate(tensafe_inference_latency_seconds_bucket[5m])) by (le)
          )

    # Scale on queue depth
    - type: prometheus
      metadata:
        metricName: tensafe_job_queue_depth
        threshold: "100"
        query: tensafe_job_queue_depth

    # Scale down on low GPU utilization
    - type: prometheus
      metadata:
        metricName: nvidia_gpu_utilization
        threshold: "30"
        query: avg(nvidia_gpu_utilization)
```

### GPU-Aware Scaling

```yaml
# Request GPUs in resources
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1

# Node selector for GPU nodes
nodeSelector:
  nvidia.com/gpu.present: "true"

# Tolerations for GPU taints
tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

## Secret Management

### Kubernetes Secrets

```yaml
# secrets.yaml (use external-secrets or sealed-secrets)
apiVersion: v1
kind: Secret
metadata:
  name: tensafe-secrets
type: Opaque
stringData:
  TS_SECRET_KEY: "your-secret-key"
  TS_KEY_MASTER: "your-master-key"
  DATABASE_URL: "postgresql://user:pass@host:5432/db"
```

### HashiCorp Vault Integration

```yaml
# In values.yaml
vault:
  enabled: true
  address: https://vault.example.com
  role: tensafe
  secretPath: secret/data/tensafe

  # Annotations for vault-agent-injector
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "tensafe"
    vault.hashicorp.com/agent-inject-secret-config: "secret/data/tensafe"
```

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: tensafe-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: tensafe-secrets
  data:
    - secretKey: TS_SECRET_KEY
      remoteRef:
        key: secret/tensafe
        property: secret_key
    - secretKey: TS_KEY_MASTER
      remoteRef:
        key: secret/tensafe
        property: key_master
```

## Networking

### Ingress Configuration

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.tensafe.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: tensafe-tls
      hosts:
        - api.tensafe.example.com
```

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tensafe-network-policy
spec:
  podSelector:
    matchLabels:
      app: tensafe
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
        - podSelector:
            matchLabels:
              app: prometheus
      ports:
        - port: 8000
        - port: 9090  # Metrics
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
        - podSelector:
            matchLabels:
              app: redis
        - podSelector:
            matchLabels:
              app: vault
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
# Enabled in values.yaml
observability:
  prometheus:
    serviceMonitor:
      enabled: true
      interval: 30s
      scrapeTimeout: 10s
```

### Grafana Dashboards

```bash
# Import TenSafe dashboard
kubectl create configmap tensafe-dashboard \
  --from-file=dashboard.json=./deploy/grafana/tensafe-dashboard.json \
  -n monitoring

# Label for Grafana sidecar
kubectl label configmap tensafe-dashboard \
  grafana_dashboard=1 \
  -n monitoring
```

### Jaeger Tracing

```yaml
# In values.yaml
observability:
  jaeger:
    enabled: true
    agent:
      host: jaeger-agent.monitoring
      port: 6831
```

## Upgrading

### Helm Upgrade

```bash
# Update chart repository
helm repo update

# Upgrade release
helm upgrade tensafe tensafe/tensafe \
  --namespace tensafe \
  --values values-prod.yaml

# Rollback if needed
helm rollback tensafe 1 -n tensafe
```

### Blue-Green Deployment

```bash
# Deploy new version (green)
helm install tensafe-green ./deploy/helm/tensafe \
  --namespace tensafe \
  --set image.tag=4.1.0

# Verify green deployment
kubectl get pods -n tensafe -l app=tensafe-green

# Switch traffic (update ingress)
kubectl patch ingress tensafe-ingress \
  -n tensafe \
  --type=json \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "tensafe-green"}]'

# Remove old deployment (blue)
helm uninstall tensafe -n tensafe
```

## Troubleshooting

### Common Issues

**Pods Pending - No GPU Nodes**
```bash
kubectl describe pod tensafe-xxx -n tensafe
# Events: FailedScheduling... Insufficient nvidia.com/gpu
```
Solution: Check GPU node availability and taints

**Database Connection Failed**
```bash
kubectl logs tensafe-xxx -n tensafe
# Error: could not connect to PostgreSQL
```
Solution: Verify database secret and network policy

**Vault Authentication Failed**
```bash
kubectl logs tensafe-xxx -c vault-agent-init -n tensafe
# Error: permission denied
```
Solution: Check Vault role and policy configuration

### Debug Commands

```bash
# Check pod status
kubectl get pods -n tensafe -o wide

# View logs
kubectl logs -f deployment/tensafe -n tensafe

# Exec into pod
kubectl exec -it deployment/tensafe -n tensafe -- /bin/bash

# Check events
kubectl get events -n tensafe --sort-by='.lastTimestamp'

# Describe deployment
kubectl describe deployment tensafe -n tensafe
```

### Health Checks

```bash
# Liveness probe
curl -f http://tensafe-svc:8000/health

# Readiness probe
curl -f http://tensafe-svc:8000/readiness

# Check from within cluster
kubectl run curl --image=curlimages/curl -it --rm -- \
  curl http://tensafe.tensafe.svc.cluster.local:8000/health
```

## Security Best Practices

1. **RBAC**: Use minimal service account permissions
2. **Network Policies**: Restrict pod-to-pod communication
3. **Pod Security**: Enable pod security standards
4. **Secrets**: Use external secret management (Vault, External Secrets)
5. **TLS**: Enable TLS termination at ingress
6. **Image Security**: Use signed images, scan for vulnerabilities

```yaml
# Pod security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

# Container security context
containerSecurityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
```

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [observability.md](observability.md) - Monitoring setup
- [ray-train.md](ray-train.md) - Distributed training on K8s
