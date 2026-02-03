# TenSafe Scaling Operations Runbook

**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Document Owner**: Platform Operations Team

---

## Table of Contents

1. [Overview](#overview)
2. [Scaling Architecture](#scaling-architecture)
3. [Horizontal Scaling](#horizontal-scaling)
4. [Vertical Scaling](#vertical-scaling)
5. [GPU Node Management](#gpu-node-management)
6. [Auto-Scaling Configuration](#auto-scaling-configuration)
7. [Capacity Planning](#capacity-planning)
8. [Scaling Procedures](#scaling-procedures)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This runbook covers scaling operations for TenSafe, including horizontal pod scaling, vertical resource adjustments, GPU node management, and auto-scaling configuration.

### Key Scaling Metrics

| Metric | Scale Up Threshold | Scale Down Threshold |
|--------|-------------------|---------------------|
| CPU Utilization | > 70% | < 30% |
| Memory Utilization | > 80% | < 40% |
| P95 Latency | > 100ms | < 50ms |
| Queue Depth | > 50 requests | < 10 requests |
| GPU Utilization | > 80% | < 30% |
| HE-LoRA Queue | > 20 operations | < 5 operations |

### Scaling Limits

| Resource | Minimum | Maximum | Notes |
|----------|---------|---------|-------|
| API Pods | 3 | 50 | Production minimum for HA |
| Training Workers | 0 | 10 | Scale to 0 when idle |
| PostgreSQL Replicas | 1 | 5 | Primary + read replicas |
| Redis Replicas | 1 | 6 | Cluster mode |
| GPU Nodes | 1 | 20 | Cost consideration |

---

## Scaling Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                  Load Balancer                   │
                    └─────────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┼───────────────────────┐
                    │                         │                        │
            ┌───────▼───────┐   ┌─────────────▼─────────┐   ┌────────▼────────┐
            │  API Pod 1    │   │      API Pod 2        │   │   API Pod N     │
            │  (CPU/Memory) │   │     (CPU/Memory)      │   │  (CPU/Memory)   │
            └───────┬───────┘   └─────────────┬─────────┘   └────────┬────────┘
                    │                         │                      │
                    └─────────────────────────┼──────────────────────┘
                                              │
                    ┌─────────────────────────┼───────────────────────┐
                    │                         │                        │
            ┌───────▼───────┐   ┌─────────────▼─────────┐   ┌────────▼────────┐
            │   GPU Worker  │   │      GPU Worker       │   │    GPU Worker   │
            │   (A100/H100) │   │      (A100/H100)      │   │   (A100/H100)   │
            └───────────────┘   └───────────────────────┘   └─────────────────┘
                                              │
                    ┌─────────────────────────┼───────────────────────┐
                    │                         │                        │
            ┌───────▼───────┐           ┌─────▼─────┐           ┌──────▼──────┐
            │  PostgreSQL   │           │   Redis   │           │   Object    │
            │   (Primary)   │           │  Cluster  │           │   Storage   │
            └───────────────┘           └───────────┘           └─────────────┘
```

---

## Horizontal Scaling

### Manual Horizontal Scaling

#### Scale API Pods

```bash
# Check current replicas
kubectl get deployment tensafe-server -n tensafe

# Scale to specific replica count
kubectl scale deployment tensafe-server -n tensafe --replicas=10

# Verify scaling
kubectl get pods -n tensafe -l app.kubernetes.io/name=tensafe
kubectl rollout status deployment/tensafe-server -n tensafe
```

**Expected Outcome**: All pods in Running state within 2-5 minutes

#### Scale Training Workers

```bash
# Check current training worker count
kubectl get deployment tensafe-training-worker -n tensafe

# Scale up for batch training job
kubectl scale deployment tensafe-training-worker -n tensafe --replicas=5

# Monitor scaling
watch kubectl get pods -n tensafe -l app.kubernetes.io/component=training-worker

# Scale down after training completes
kubectl scale deployment tensafe-training-worker -n tensafe --replicas=0
```

### Horizontal Scaling Triggers

| Trigger | Condition | Action | Cooldown |
|---------|-----------|--------|----------|
| High CPU | Avg CPU > 70% for 5m | Add 2 pods | 3 minutes |
| High Memory | Avg Memory > 80% for 5m | Add 2 pods | 3 minutes |
| High Latency | P95 > 100ms for 5m | Add 4 pods | 1 minute |
| Queue Backup | Depth > 50 for 2m | Add 4 pods | 1 minute |
| Low Load | Avg CPU < 30% for 15m | Remove 1 pod | 5 minutes |

### Scaling Best Practices

1. **Never scale below minimum replicas** (3 for production)
2. **Scale up aggressively, scale down conservatively**
3. **Monitor after scaling** - wait 5 minutes before additional scaling
4. **Check PodDisruptionBudget** - ensure minimum availability

```bash
# Check PDB status
kubectl get pdb tensafe-pdb -n tensafe

# Verify before scaling down
kubectl describe pdb tensafe-pdb -n tensafe
```

---

## Vertical Scaling

### Adjust Resource Requests/Limits

```bash
# View current resource allocation
kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Update resources using patch
kubectl patch deployment tensafe-server -n tensafe --type='json' \
  -p='[
    {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/memory", "value": "4Gi"},
    {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/cpu", "value": "2000m"},
    {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "8Gi"},
    {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu", "value": "4000m"}
  ]'

# Monitor rollout
kubectl rollout status deployment/tensafe-server -n tensafe
```

### Resource Profiles

| Profile | CPU Request | CPU Limit | Memory Request | Memory Limit | Use Case |
|---------|-------------|-----------|----------------|--------------|----------|
| Small | 1000m | 2000m | 2Gi | 4Gi | Development |
| Medium | 2000m | 4000m | 4Gi | 8Gi | Staging |
| Large | 4000m | 8000m | 8Gi | 16Gi | Production |
| XLarge | 8000m | 16000m | 16Gi | 32Gi | High-load |

### Vertical Scaling Procedure

1. **Assess current usage**
   ```bash
   kubectl top pods -n tensafe
   kubectl top nodes
   ```

2. **Check if nodes have capacity**
   ```bash
   kubectl describe nodes | grep -A 5 "Allocated resources"
   ```

3. **Update deployment**
   ```bash
   # Using Helm
   helm upgrade tensafe ./deploy/helm/tensafe \
     --namespace tensafe \
     --set resources.requests.memory=8Gi \
     --set resources.limits.memory=16Gi \
     --set resources.requests.cpu=4000m \
     --set resources.limits.cpu=8000m
   ```

4. **Monitor rollout**
   ```bash
   kubectl rollout status deployment/tensafe-server -n tensafe
   ```

**Expected Outcome**: Pods restart with new resource allocation

**Rollback if issues**:
```bash
kubectl rollout undo deployment/tensafe-server -n tensafe
```

---

## GPU Node Management

### GPU Node Types

| Node Type | GPU | Memory | vCPUs | Use Case |
|-----------|-----|--------|-------|----------|
| p4d.24xlarge | 8x A100 40GB | 1152 GB | 96 | Training |
| p5.48xlarge | 8x H100 80GB | 2048 GB | 192 | Large models |
| g5.xlarge | 1x A10G | 16 GB | 4 | Inference |
| g5.12xlarge | 4x A10G | 192 GB | 48 | Batch inference |

### Check GPU Node Status

```bash
# List GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true

# Check GPU availability
kubectl describe nodes -l nvidia.com/gpu.present=true | grep -A 5 "Allocatable:" | grep nvidia

# Check GPU utilization
kubectl exec -it deployment/tensafe-server -n tensafe -- nvidia-smi

# Check DCGM metrics (if DCGM exporter is installed)
kubectl get pods -n monitoring -l app=dcgm-exporter
```

### Add GPU Nodes (AWS EKS Example)

```bash
# Scale up GPU node group
aws eks update-nodegroup-config \
  --cluster-name tensafe-prod \
  --nodegroup-name gpu-workers \
  --scaling-config minSize=2,maxSize=10,desiredSize=4

# Verify nodes joining cluster
watch kubectl get nodes -l nvidia.com/gpu.present=true

# Check node status
kubectl describe node <new-gpu-node-name>
```

### Remove GPU Nodes

```bash
# Cordon node (prevent new pods)
kubectl cordon <gpu-node-name>

# Drain node (evict pods)
kubectl drain <gpu-node-name> --ignore-daemonsets --delete-emptydir-data

# Scale down node group
aws eks update-nodegroup-config \
  --cluster-name tensafe-prod \
  --nodegroup-name gpu-workers \
  --scaling-config minSize=1,maxSize=10,desiredSize=2

# Delete node from cluster (if needed)
kubectl delete node <gpu-node-name>
```

### GPU Scheduling Configuration

```yaml
# Pod spec for GPU workloads
spec:
  containers:
  - name: tensafe-training
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
  nodeSelector:
    nvidia.com/gpu.present: "true"
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

---

## Auto-Scaling Configuration

### HPA (Horizontal Pod Autoscaler)

```yaml
# Check current HPA status
kubectl get hpa tensafe-hpa -n tensafe

# Describe HPA for details
kubectl describe hpa tensafe-hpa -n tensafe

# Update HPA configuration
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tensafe-hpa
  namespace: tensafe
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tensafe-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
EOF
```

### KEDA (Kubernetes Event-Driven Autoscaling)

```bash
# Check KEDA ScaledObject status
kubectl get scaledobject tensafe-scaledobject -n tensafe

# Describe for detailed status
kubectl describe scaledobject tensafe-scaledobject -n tensafe

# Check KEDA operator logs
kubectl logs -l app=keda-operator -n keda --tail=100
```

### Update KEDA Configuration

```yaml
# Apply updated ScaledObject
kubectl apply -f - <<EOF
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: tensafe-scaledobject
  namespace: tensafe
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tensafe-server
  pollingInterval: 15
  cooldownPeriod: 60
  minReplicaCount: 3
  maxReplicaCount: 50
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: tensafe_inference_latency_p95
      threshold: '0.1'
      query: |
        histogram_quantile(0.95,
          sum(rate(tensafe_inference_latency_seconds_bucket{job="tensafe"}[5m]))
          by (le)
        )
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: tensafe_queue_depth
      threshold: '50'
      query: sum(tensafe_request_queue_depth{job="tensafe"})
EOF
```

### Disable Auto-Scaling (Emergency)

```bash
# Disable HPA
kubectl delete hpa tensafe-hpa -n tensafe

# Or pause KEDA
kubectl annotate scaledobject tensafe-scaledobject -n tensafe \
  autoscaling.keda.sh/paused="true"

# Set fixed replica count
kubectl scale deployment tensafe-server -n tensafe --replicas=10
```

### Re-Enable Auto-Scaling

```bash
# Unpause KEDA
kubectl annotate scaledobject tensafe-scaledobject -n tensafe \
  autoscaling.keda.sh/paused-

# Or reapply HPA
kubectl apply -f deploy/kubernetes/hpa.yaml
```

---

## Capacity Planning

### Sizing Guidelines

| Workload Type | Pods | CPU/Pod | Memory/Pod | GPU/Pod |
|---------------|------|---------|------------|---------|
| Light (< 100 RPS) | 3 | 2 cores | 4Gi | 0 |
| Medium (100-500 RPS) | 5-10 | 4 cores | 8Gi | 0-1 |
| Heavy (500-1000 RPS) | 10-20 | 4 cores | 16Gi | 1 |
| HE-LoRA Training | 3-10 | 8 cores | 32Gi | 1-2 |

### Capacity Monitoring Queries

```promql
# CPU headroom
100 - (avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) / avg(kube_pod_container_resource_limits{namespace="tensafe",resource="cpu"})) * 100

# Memory headroom
100 - (avg(container_memory_working_set_bytes{namespace="tensafe"}) / avg(kube_pod_container_resource_limits{namespace="tensafe",resource="memory"})) * 100

# GPU headroom
100 - avg(DCGM_FI_DEV_GPU_UTIL{namespace="tensafe"})

# Requests per second
sum(rate(tensafe_http_requests_total{namespace="tensafe"}[5m]))
```

### Capacity Planning Checklist

- [ ] Review current resource utilization (CPU, Memory, GPU)
- [ ] Analyze traffic patterns (peak hours, seasonal trends)
- [ ] Calculate projected growth (10%, 50%, 100% increase)
- [ ] Plan buffer capacity (20% above projected peak)
- [ ] Verify node capacity for scaling
- [ ] Check cost implications
- [ ] Update auto-scaling thresholds

### Scaling Decision Matrix

| Current Load | P95 Latency | Action |
|--------------|-------------|--------|
| < 50% capacity | < 50ms | Monitor, consider scaling down |
| 50-70% capacity | < 100ms | Optimal, no action |
| 70-85% capacity | < 100ms | Pre-scale to add buffer |
| > 85% capacity | > 100ms | **Immediate scale up** |
| Any | > 200ms | **Emergency scale up** |

---

## Scaling Procedures

### Procedure: Scale for Expected Load Increase

**Trigger**: Scheduled event, marketing campaign, known traffic spike

1. **Calculate required capacity**
   ```bash
   # Current capacity
   CURRENT_PODS=$(kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.replicas}')

   # Calculate new capacity (e.g., 2x current)
   NEW_PODS=$((CURRENT_PODS * 2))
   ```

2. **Pre-scale 30 minutes before event**
   ```bash
   kubectl scale deployment tensafe-server -n tensafe --replicas=$NEW_PODS
   ```

3. **Verify scaling complete**
   ```bash
   kubectl rollout status deployment/tensafe-server -n tensafe
   kubectl get pods -n tensafe
   ```

4. **Monitor during event**
   - Watch Grafana dashboards
   - Set up alert thresholds

5. **Scale down after event** (wait 15-30 minutes after load decrease)
   ```bash
   kubectl scale deployment tensafe-server -n tensafe --replicas=$CURRENT_PODS
   ```

### Procedure: Emergency Scale Up

**Trigger**: Alert firing, high latency, error spike

1. **Immediately double capacity**
   ```bash
   CURRENT=$(kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.replicas}')
   kubectl scale deployment tensafe-server -n tensafe --replicas=$((CURRENT * 2))
   ```

2. **Pause auto-scaler to prevent interference**
   ```bash
   kubectl annotate scaledobject tensafe-scaledobject -n tensafe \
     autoscaling.keda.sh/paused="true"
   ```

3. **Monitor**
   ```bash
   watch kubectl get pods -n tensafe
   ```

4. **Once stable, re-enable auto-scaler**
   ```bash
   kubectl annotate scaledobject tensafe-scaledobject -n tensafe \
     autoscaling.keda.sh/paused-
   ```

---

## Troubleshooting

### Pods Not Scaling Up

```bash
# Check HPA status
kubectl describe hpa tensafe-hpa -n tensafe

# Check if metrics are available
kubectl top pods -n tensafe

# Check metrics-server
kubectl get apiservices | grep metrics

# Check KEDA status
kubectl describe scaledobject tensafe-scaledobject -n tensafe
kubectl logs -l app=keda-operator -n keda --tail=100
```

### Pods Pending After Scale Up

```bash
# Check pod events
kubectl describe pod <pending-pod> -n tensafe

# Common causes:
# 1. Insufficient node resources
kubectl describe nodes | grep -A 10 "Allocated resources"

# 2. Pod affinity rules
kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.template.spec.affinity}'

# 3. GPU unavailable
kubectl get nodes -l nvidia.com/gpu.present=true
```

### Auto-Scaler Oscillating (Thrashing)

```bash
# Increase cooldown period
kubectl patch hpa tensafe-hpa -n tensafe --type='json' \
  -p='[{"op": "replace", "path": "/spec/behavior/scaleDown/stabilizationWindowSeconds", "value": 600}]'

# Check scaling history
kubectl describe hpa tensafe-hpa -n tensafe | grep -A 20 "Events"
```

### GPU Not Detected

```bash
# Check NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check device plugin logs
kubectl logs -l name=nvidia-device-plugin-ds -n kube-system

# Verify GPU on node
kubectl debug node/<gpu-node> -it --image=nvidia/cuda:12.0-base -- nvidia-smi
```

---

## Related Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment procedures
- [MONITORING_ALERTS.md](MONITORING_ALERTS.md) - Alert response
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [docs/guides/kubernetes.md](../guides/kubernetes.md) - Kubernetes guide

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Platform Ops | Initial runbook |
