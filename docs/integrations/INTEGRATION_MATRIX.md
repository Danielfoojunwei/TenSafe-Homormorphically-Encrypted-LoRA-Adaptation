# Integration Matrix

This document outlines the Official Integration Strategy for TensorGuardFlow Core, mapping how it interacts with the broader ML/LLM ecosystem (Categories C-D-E-F-G).

## C: Data Sources + Governance

| Tool | Integration Style | Required for Core? | Config Keys | Health Check |
| :--- | :--- | :--- | :--- | :--- |
| **Local File** | Direct API (`jsonl`) | **Yes** | `uri` | Path exists |
| **AWS S3** | Direct API (boto3) | No | `bucket`, `prefix`, `iam_role` | `HeadBucket` |
| **GCS** | Direct API (google-cloud-storage) | No | `bucket`, `prefix`, `project_id` | `get_bucket` |
| **Azure Blob** | Direct API (azure-storage-blob) | No | `container`, `prefix`, `connection_string` | `get_container_client` |
| **HF Dataset** | Direct API (`datasets`) | No | `repo_id`, `subset`, `split` | `load_dataset_builder` |

## D: Training Execution

| Tool | Integration Style | Required for Core? | Config Keys | Health Check |
| :--- | :--- | :--- | :--- | :--- |
| **Local HF** | Direct Subprocess | **Yes** | `device`, `max_steps` | Torch CUDA check |
| **K8s Job** | **Export Template** (YAML) | No | `image`, `resource_limits`, `namespace` | API connectivity |
| **SageMaker** | **Export Template** (JobSpec) | No | `instance_type`, `execution_role`, `vpc_config` | Boto3 SM check |
| **Vertex AI** | **Export Template** (JobSpec) | No | `machine_type`, `service_account`, `region` | AIPlatform check |
| **Databricks** | **Export Template** (JSON) | No | `cluster_id`, `notebook_path` | Databricks CLI check |

## E: Evaluation + Registry

| Tool | Integration Style | Required for Core? | Config Keys | Health Check |
| :--- | :--- | :--- | :--- | :--- |
| **Internal** | SQLModel (Default) | **Yes** | N/A (Included) | DB Health check |
| **MLflow** | Optional Export | No | `tracking_uri`, `experiment_id` | `mlflow.get_experiment` |
| **W&B** | Optional Metrics Push | No | `project`, `entity`, `api_key` | `wandb.login` |

## F: Serving Targets

| Tool | Integration Style | Required for Core? | Config Keys | Health Check |
| :--- | :--- | :--- | :--- | :--- |
| **vLLM** | **Serving Pack** (JSON) | **Yes** | `model_id`, `dtype`, `max_model_len` | Config validation |
| **TGI** | **Serving Pack** (YAML) | No | `model_id`, `sharded`, `num_shard` | Template validation |
| **Triton** | **Serving Pack** (pbtxt) | No | `model_name`, `backend`, `max_batch_size` | Schema validation |
| **Bedrock** | **Import Guidance** | No | `import_arn`, `bucket_location` | Documentation-only |

## G: Trust + Privacy

| Tool | Integration Style | Required for Core? | Config Keys | Health Check |
| :--- | :--- | :--- | :--- | :--- |
| **Nitro Enclave** | Sidecar / Signer | No | `enclave_id`, `pcr0` | `attestation_check` |
| **N2HE** | **Privacy Provider** | **Yes** | `kms_key_id`, `n2he_profile` | Receipt verification |

---

> [!IMPORTANT]
> TensorGuardFlow remains the **Control Plane**. It generates the "evidence chain" and "serving packs" that these tools consume. It does NOT host the training or the inference itself.
