# Integration Config Reference

This document details the configuration schemas for the TensorGuardFlow Pipeline Integration Framework.

## Core Concepts

All integrations are grouped by **Category** and defined via **Connectors**. Configurations are strict Pydantic models.

### Integration Profile
The `IntegrationProfile` is a tenant-level aggregation of all configured connectors.

```yaml
tenant_id: "acme-corp"
data_sources: [...]
training_executors: [...]
serving_targets: [...]
privacy_settings: {...}
```

---

## Category C: Data Sources
Connectors for ingesting raw data for continuous learning.

### Local Filesystem (`type: local`)
Ingests data from a local directory or mount point.
- `uri`: Path to the data (e.g., `/data/raw` or `data/raw/`).
- `uri` (strict): Must be a valid local path.

### S3 Feed (`type: s3`)
Ingests data from AWS S3.
- `uri`: S3 URI (e.g., `s3://my-bucket/prefix/`).
- `credentials_secret_id` (optional): Reference to vault for AWS keys.

---

## Category D: Training Executors
Connectors for executing or exporting training jobs.

### Local Hugging Face (`type: local`)
Executes training on the local control plane (mainly for simulation/small models).
- `cluster_ref`: Must be `localhost`.

### K8s Job Exporter (`type: k8s`)
Generates Kubernetes `Job` YAML for external execution.
- `cluster_ref`: Kubernetes context or namespace.
- `env_vars`: Dictionary of environment variables to inject.

### SageMaker Exporter (`type: sagemaker`)
Generates AWS SageMaker Training Job specifications.
- `cluster_ref`: ARN of the execution role.

---

## Category F: Serving Targets
Exporters for generating infrastructure-ready serving packages.

### vLLM Exporter (`type: vllm`)
Generates a `_serving_pack.json` for vLLM Multi-Adapter serving.

### TGI Exporter (`type: tgi`)
Generates a `tgi-config.yaml` for Text-Generation-Inference.

---

## Category G: Trust & Privacy

### N2HE Privacy (`type: n2he`)
Provides encryption and receipt generation for privacy-preserving data flows.
- `n2he_profile`: `router_only` or `full`.
- `api_key`: Secret key for the N2HE provider.

---

## Example: Full Tenant Profile

```yaml
tenant_id: "default"
data_sources:
  - name: "daily_ingest"
    type: "s3"
    uri: "s3://tensorguard-feeds/acme/"
training_executors:
  - name: "spot_cluster"
    type: "k8s"
    cluster_ref: "gpu-spot-nodepool"
serving_targets:
  - name: "prod_vllm"
    type: "vllm"
```
