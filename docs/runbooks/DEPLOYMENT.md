# TenSafe Deployment Runbook

**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Document Owner**: Platform Operations Team

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Standard Deployment](#standard-deployment)
4. [Blue/Green Deployment](#bluegreen-deployment)
5. [Canary Deployment](#canary-deployment)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Rollback Procedures](#rollback-procedures)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This runbook covers deployment procedures for TenSafe, a privacy-first ML platform with homomorphic encryption capabilities. TenSafe is deployed on Kubernetes with the following components:

| Component | Description | Critical? |
|-----------|-------------|-----------|
| tensafe-server | Main API server | Yes |
| PostgreSQL | Primary database | Yes |
| Redis | Caching and job queues | Yes |
| OTEL Collector | Observability | No |

### Deployment Environments

| Environment | Cluster | Namespace | Approval Required |
|-------------|---------|-----------|-------------------|
| Development | dev-cluster | tensafe-dev | No |
| Staging | staging-cluster | tensafe-staging | No |
| Production | prod-cluster | tensafe-prod | Yes (2 approvers) |

---

## Prerequisites

### Required Access

- [ ] Kubernetes cluster access (kubectl configured)
- [ ] Helm 3.x installed
- [ ] Access to container registry (tensafe/server)
- [ ] Vault access for secrets (production)
- [ ] PagerDuty/On-call access (production)

### Pre-Deployment Checklist

- [ ] Release notes reviewed and approved
- [ ] Changelog updated
- [ ] All CI/CD checks passing
- [ ] Security scan completed (no critical vulnerabilities)
- [ ] Database migrations tested in staging
- [ ] Load testing completed (for major releases)
- [ ] Rollback plan documented
- [ ] On-call engineer notified (production)

### Verify Prerequisites

```bash
# Check kubectl access
kubectl cluster-info
kubectl get nodes

# Check Helm
helm version

# Verify namespace exists
kubectl get namespace tensafe

# Check current deployment status
kubectl get deployments -n tensafe
kubectl get pods -n tensafe

# Verify secrets are configured
kubectl get secrets -n tensafe
```

---

## Standard Deployment

### Step 1: Announce Deployment

**For Production Only**

```bash
# Post to #ops-announcements Slack channel
# Format: [DEPLOYMENT] TenSafe vX.Y.Z deployment starting in 15 minutes
```

### Step 2: Backup Current State

```bash
# Record current deployment state
kubectl get deployment tensafe-server -n tensafe -o yaml > /tmp/tensafe-backup-$(date +%Y%m%d-%H%M%S).yaml

# Record current image version
CURRENT_VERSION=$(kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "Current version: $CURRENT_VERSION"
```

### Step 3: Update Container Image

```bash
# Set the new version
NEW_VERSION="4.1.0"

# Using kubectl set image
kubectl set image deployment/tensafe-server \
  tensafe=tensafe/server:${NEW_VERSION} \
  -n tensafe

# Or using Helm upgrade
helm upgrade tensafe ./deploy/helm/tensafe \
  --namespace tensafe \
  --set image.tag=${NEW_VERSION} \
  --wait \
  --timeout 10m
```

### Step 4: Monitor Rollout

```bash
# Watch rollout status
kubectl rollout status deployment/tensafe-server -n tensafe --timeout=5m

# Monitor pod status
watch kubectl get pods -n tensafe -l app.kubernetes.io/name=tensafe

# Check for any failing pods
kubectl get pods -n tensafe | grep -v Running
```

### Step 5: Verify Deployment

```bash
# Check deployment is healthy
kubectl get deployment tensafe-server -n tensafe

# Verify new image version
kubectl get pods -n tensafe -l app.kubernetes.io/name=tensafe -o jsonpath='{.items[0].spec.containers[0].image}'

# Run health check
kubectl exec -it deployment/tensafe-server -n tensafe -- curl -s http://localhost:8000/health

# Check logs for errors
kubectl logs -l app.kubernetes.io/name=tensafe -n tensafe --tail=100 | grep -i error
```

**Expected Outcomes**:
- All pods in Running state
- Health check returns `{"status": "healthy"}`
- No error logs related to startup

---

## Blue/Green Deployment

Blue/Green deployment enables zero-downtime deployments by running two identical environments.

### Step 1: Deploy Green Environment

```bash
# Create green deployment
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensafe-server-green
  namespace: tensafe
  labels:
    app.kubernetes.io/name: tensafe
    app.kubernetes.io/instance: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app.kubernetes.io/name: tensafe
      app.kubernetes.io/instance: green
  template:
    metadata:
      labels:
        app.kubernetes.io/name: tensafe
        app.kubernetes.io/instance: green
    spec:
      containers:
      - name: tensafe
        image: tensafe/server:${NEW_VERSION}
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: tensafe-config
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tensafe-secrets
              key: database-url
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
EOF
```

### Step 2: Wait for Green Ready

```bash
# Wait for green deployment to be ready
kubectl rollout status deployment/tensafe-server-green -n tensafe --timeout=10m

# Verify all green pods are running
kubectl get pods -n tensafe -l app.kubernetes.io/instance=green
```

### Step 3: Test Green Environment

```bash
# Port-forward to green deployment for testing
kubectl port-forward deployment/tensafe-server-green 8001:8000 -n tensafe &

# Run smoke tests against green
curl http://localhost:8001/health
curl http://localhost:8001/ready

# Run integration tests (if available)
TENSAFE_BASE_URL=http://localhost:8001 pytest tests/integration/test_smoke.py -v

# Kill port-forward
pkill -f "port-forward.*tensafe-server-green"
```

### Step 4: Switch Traffic

```bash
# Update service selector to point to green
kubectl patch service tensafe-server -n tensafe \
  -p '{"spec":{"selector":{"app.kubernetes.io/instance":"green"}}}'

# Verify traffic switch
kubectl get endpoints tensafe-server -n tensafe
```

### Step 5: Verify and Cleanup

```bash
# Monitor for 5-10 minutes
watch kubectl get pods -n tensafe

# If everything is healthy, remove blue deployment
kubectl delete deployment tensafe-server-blue -n tensafe

# Rename green to become the new production
kubectl patch deployment tensafe-server-green -n tensafe \
  --type='json' \
  -p='[{"op": "replace", "path": "/metadata/name", "value": "tensafe-server"}]'
```

**Rollback (if issues)**:
```bash
# Switch traffic back to blue
kubectl patch service tensafe-server -n tensafe \
  -p '{"spec":{"selector":{"app.kubernetes.io/instance":"blue"}}}'

# Delete failed green deployment
kubectl delete deployment tensafe-server-green -n tensafe
```

---

## Canary Deployment

Canary deployment gradually shifts traffic to the new version.

### Step 1: Deploy Canary (10% traffic)

```bash
# Deploy canary with 1 replica (while production has 9)
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensafe-server-canary
  namespace: tensafe
  labels:
    app.kubernetes.io/name: tensafe
    app.kubernetes.io/instance: canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: tensafe
      app.kubernetes.io/instance: canary
  template:
    metadata:
      labels:
        app.kubernetes.io/name: tensafe
        app.kubernetes.io/instance: canary
        version: canary
    spec:
      containers:
      - name: tensafe
        image: tensafe/server:${NEW_VERSION}
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: tensafe-config
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tensafe-secrets
              key: database-url
EOF

# Scale production to 9 replicas
kubectl scale deployment tensafe-server -n tensafe --replicas=9
```

### Step 2: Monitor Canary

```bash
# Monitor canary for 15-30 minutes
# Check error rates
kubectl logs -l app.kubernetes.io/instance=canary -n tensafe --tail=100

# Compare metrics between canary and stable
# Check Grafana dashboards for:
# - Error rate comparison
# - Latency comparison
# - Resource utilization
```

### Step 3: Gradual Rollout

```bash
# Increase canary to 30% (3 replicas)
kubectl scale deployment tensafe-server-canary -n tensafe --replicas=3
kubectl scale deployment tensafe-server -n tensafe --replicas=7

# Monitor for 15 minutes...

# Increase canary to 50% (5 replicas)
kubectl scale deployment tensafe-server-canary -n tensafe --replicas=5
kubectl scale deployment tensafe-server -n tensafe --replicas=5

# Monitor for 15 minutes...

# Increase canary to 100%
kubectl scale deployment tensafe-server-canary -n tensafe --replicas=10
kubectl scale deployment tensafe-server -n tensafe --replicas=0
```

### Step 4: Finalize Deployment

```bash
# Update main deployment to new version
kubectl set image deployment/tensafe-server \
  tensafe=tensafe/server:${NEW_VERSION} \
  -n tensafe

# Scale main deployment back up
kubectl scale deployment tensafe-server -n tensafe --replicas=10

# Delete canary
kubectl delete deployment tensafe-server-canary -n tensafe
```

**Rollback (if issues during canary)**:
```bash
# Scale down canary immediately
kubectl scale deployment tensafe-server-canary -n tensafe --replicas=0

# Scale up stable
kubectl scale deployment tensafe-server -n tensafe --replicas=10

# Delete failed canary
kubectl delete deployment tensafe-server-canary -n tensafe
```

---

## Post-Deployment Verification

### Automated Verification Script

```bash
#!/bin/bash
# post_deploy_verify.sh

NAMESPACE="tensafe"
DEPLOYMENT="tensafe-server"

echo "=== Post-Deployment Verification ==="

# 1. Check deployment status
echo "[1/7] Checking deployment status..."
READY=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
DESIRED=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.replicas}')
if [ "$READY" != "$DESIRED" ]; then
    echo "FAIL: Only $READY/$DESIRED replicas ready"
    exit 1
fi
echo "PASS: $READY/$DESIRED replicas ready"

# 2. Check all pods running
echo "[2/7] Checking pod status..."
FAILED_PODS=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=tensafe --no-headers | grep -v Running | wc -l)
if [ "$FAILED_PODS" -gt 0 ]; then
    echo "FAIL: $FAILED_PODS pods not in Running state"
    kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=tensafe
    exit 1
fi
echo "PASS: All pods running"

# 3. Health endpoint check
echo "[3/7] Checking health endpoint..."
POD=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=tensafe -o jsonpath='{.items[0].metadata.name}')
HEALTH=$(kubectl exec $POD -n $NAMESPACE -- curl -s http://localhost:8000/health)
if [[ $HEALTH != *"healthy"* ]]; then
    echo "FAIL: Health check failed: $HEALTH"
    exit 1
fi
echo "PASS: Health check passed"

# 4. Ready endpoint check
echo "[4/7] Checking ready endpoint..."
READY_CHECK=$(kubectl exec $POD -n $NAMESPACE -- curl -s http://localhost:8000/ready)
if [[ $READY_CHECK != *"ready"* ]]; then
    echo "FAIL: Readiness check failed: $READY_CHECK"
    exit 1
fi
echo "PASS: Readiness check passed"

# 5. Database connectivity
echo "[5/7] Checking database connectivity..."
DB_CHECK=$(kubectl exec $POD -n $NAMESPACE -- curl -s http://localhost:8000/health/db)
if [[ $DB_CHECK != *"connected"* ]]; then
    echo "FAIL: Database connectivity check failed"
    exit 1
fi
echo "PASS: Database connected"

# 6. Check for error logs
echo "[6/7] Checking for error logs..."
ERROR_COUNT=$(kubectl logs $POD -n $NAMESPACE --since=5m | grep -i "error\|exception\|fatal" | wc -l)
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "WARN: $ERROR_COUNT errors in last 5 minutes"
fi
echo "INFO: $ERROR_COUNT errors in logs"

# 7. Verify version
echo "[7/7] Verifying deployed version..."
DEPLOYED_VERSION=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "Deployed version: $DEPLOYED_VERSION"

echo ""
echo "=== Verification Complete ==="
```

### Manual Verification Checklist

- [ ] All pods are in Running state
- [ ] Health endpoint returns healthy
- [ ] Ready endpoint returns ready
- [ ] Database connectivity confirmed
- [ ] API endpoints responding (test /v1/training)
- [ ] Metrics endpoint accessible (/metrics on port 9090)
- [ ] Logs show no critical errors
- [ ] Grafana dashboards show normal metrics
- [ ] Alerts are not firing

---

## Rollback Procedures

### Immediate Rollback

```bash
# Rollback to previous revision
kubectl rollout undo deployment/tensafe-server -n tensafe

# Verify rollback
kubectl rollout status deployment/tensafe-server -n tensafe

# Check rolled back version
kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.template.spec.containers[0].image}'
```

### Rollback to Specific Version

```bash
# List rollout history
kubectl rollout history deployment/tensafe-server -n tensafe

# Rollback to specific revision
kubectl rollout undo deployment/tensafe-server -n tensafe --to-revision=3

# Or specify image directly
kubectl set image deployment/tensafe-server \
  tensafe=tensafe/server:4.0.0 \
  -n tensafe
```

### Rollback Using Helm

```bash
# List Helm history
helm history tensafe -n tensafe

# Rollback to previous release
helm rollback tensafe -n tensafe

# Rollback to specific revision
helm rollback tensafe 5 -n tensafe
```

### Post-Rollback Actions

1. **Verify rollback success**
   ```bash
   kubectl get pods -n tensafe
   kubectl rollout status deployment/tensafe-server -n tensafe
   ```

2. **Notify stakeholders**
   - Post to #ops-announcements: "[ROLLBACK] TenSafe rolled back to vX.Y.Z due to [reason]"

3. **Create incident ticket**
   - Document the issue that caused rollback
   - Link relevant logs and metrics

4. **Conduct post-mortem** (if production rollback)

---

## Troubleshooting

### Deployment Stuck in Pending

```bash
# Check pod events
kubectl describe pod -l app.kubernetes.io/name=tensafe -n tensafe

# Common causes:
# - Insufficient resources -> Check resource quotas
# - Image pull errors -> Check image name and registry credentials
# - Node affinity issues -> Check node labels
```

### Pods CrashLooping

```bash
# Check logs from crashed pod
kubectl logs -l app.kubernetes.io/name=tensafe -n tensafe --previous

# Check events
kubectl get events -n tensafe --sort-by='.lastTimestamp'

# Common causes:
# - Configuration errors -> Check ConfigMap/Secrets
# - Database migration issues -> Check DB connectivity
# - Memory issues -> Check resource limits
```

### Health Check Failures

```bash
# Check health endpoint directly
kubectl exec -it deployment/tensafe-server -n tensafe -- curl -v http://localhost:8000/health

# Check readiness probe configuration
kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.template.spec.containers[0].readinessProbe}'

# Check application logs
kubectl logs -l app.kubernetes.io/name=tensafe -n tensafe --tail=200
```

### Slow Rollout

```bash
# Check rollout status
kubectl rollout status deployment/tensafe-server -n tensafe

# Check for pod scheduling issues
kubectl get pods -n tensafe -o wide

# Check node resources
kubectl top nodes
kubectl describe nodes | grep -A 5 "Allocated resources"
```

---

## Related Documentation

- [SCALING.md](SCALING.md) - Scaling operations
- [ROLLBACK.md](ROLLBACK.md) - Detailed rollback procedures
- [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) - Incident handling
- [docs/guides/kubernetes.md](../guides/kubernetes.md) - Kubernetes deployment guide

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Platform Ops | Initial runbook |
