# Continuous Learning Exports

This guide explains how to run TensorGuardFlow continuous learning loops on external infrastructure.

## Overview

TensorGuardFlow is a **control plane**, not a compute platform. You can export continuous loop specs to run on:

- Kubernetes
- AWS SageMaker
- GCP Vertex AI
- Azure Machine Learning
- Databricks

---

## Export a Route

```bash
curl -X POST "http://localhost:8000/api/v1/tgflow/routes/customer-support/export?backend=k8s"
```

**Response:**
```json
{
  "ok": true,
  "backend": "k8s",
  "run_spec": {
    "route_key": "customer-support",
    "base_model_ref": "microsoft/phi-2",
    "feed": { "type": "hf_dataset", "uri": "tatsu-lab/alpaca" },
    "policy": { "novelty_threshold": 0.3, "forgetting_budget": 0.1 },
    "privacy": { "mode": "off" }
  }
}
```

---

## Kubernetes

Use the K8s Job orchestrator to generate Kubernetes manifests:

```python
from tensorguard.integrations.export import K8sJobOrchestrator

orchestrator = K8sJobOrchestrator()
result = orchestrator.export(run_spec, output_dir="./k8s-export")
```

**Output Files:**
- `job.yaml` - Kubernetes Job manifest
- `run_spec.json` - TGFlow run specification
- `job_spec.json` - Portable job specification

### Running on K8s

```bash
# Create ConfigMap with run spec
kubectl create configmap tgflow-customer-support-config --from-file=run_spec.json

# Apply the Job
kubectl apply -f job.yaml
```

---

## AWS SageMaker

```python
from tensorguard.integrations.export import SageMakerOrchestrator

orchestrator = SageMakerOrchestrator()
result = orchestrator.export(
    run_spec,
    output_dir="./sagemaker-export",
    role_arn="arn:aws:iam::123456789:role/SageMakerRole",
    instance_type="ml.g4dn.xlarge"
)
```

### Running on SageMaker

```python
import boto3

client = boto3.client("sagemaker")
client.create_training_job(**result.native_spec)
```

---

## GCP Vertex AI

```python
from tensorguard.integrations.export import VertexAIOrchestrator

orchestrator = VertexAIOrchestrator()
result = orchestrator.export(
    run_spec,
    output_dir="./vertex-export",
    project_id="my-gcp-project",
    location="us-central1"
)
```

---

## Azure Machine Learning

```python
from tensorguard.integrations.export import AzureMLOrchestrator

orchestrator = AzureMLOrchestrator()
result = orchestrator.export(
    run_spec,
    output_dir="./azureml-export",
    compute_target="gpu-cluster",
    vm_size="Standard_NC4as_T4_v3"
)
```

---

## Databricks

```python
from tensorguard.integrations.export import DatabricksOrchestrator

orchestrator = DatabricksOrchestrator()
result = orchestrator.export(
    run_spec,
    output_dir="./databricks-export",
    spark_version="13.3.x-gpu-ml-scala2.12"
)
```

---

## N2HE Sidecar

When `privacy.mode = n2he` and `privacy.n2he_sidecar = enabled`:

- K8s: Sidecar container co-scheduled
- SageMaker: Embedded TenSEAL mode
- Vertex AI: Secondary worker pool
- Others: External N2HE service URL

---

## Pushing Results Back

After external execution, push results back to TensorGuardFlow:

```bash
# 1. Upload TGSP package
curl -X POST http://localhost:8000/api/v1/tgsp/upload \
  -F "package=@adapter-001.tgsp"

# 2. Register adapter
curl -X POST http://localhost:8000/api/v1/tgflow/routes/customer-support/register \
  -d '{"adapter_id": "adapter-001", "tgsp_path": "/path/to/adapter-001.tgsp"}'

# 3. Run evaluation
curl -X POST http://localhost:8000/api/v1/tgflow/routes/customer-support/eval \
  -d '{"adapter_id": "adapter-001"}'
```

---

## Cadence Triggers

For scheduled continuous learning on external infra:

| Platform | Trigger Mechanism |
|----------|------------------|
| Kubernetes | CronJob |
| SageMaker | EventBridge Rules |
| Vertex AI | Cloud Scheduler |
| Azure ML | Azure Automation |
| Databricks | Workflow Schedules |

Example K8s CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tgflow-customer-support-daily
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: tgflow-runner
            image: tensorguardflow/peft-runner:latest
            args: ["peft", "run", "--spec", "run_spec.json"]
```

---

## Evidence Chain Integrity

When running externally, maintain evidence chain by:

1. Signing TGSP packages with TrustCore
2. Including evidence events in package
3. Verifying signatures on upload

```bash
# Sign before upload
tgflow tgsp sign --package adapter-001.tgsp --key-id software-dev

# Verify on server
curl -X POST http://localhost:8000/api/v1/tgsp/verify \
  -F "package=@adapter-001.tgsp"
```
