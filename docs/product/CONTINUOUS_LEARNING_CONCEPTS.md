# Continuous Learning Concepts

This document explains the core concepts behind TensorGuardFlow's continuous learning system.

---

## Core Objects

### Route

A **Route** is the unit of continuous learning. One route = one evolving adapter family.

Think of a route as a "topic" your system needs to handle:
- `customer-support` - Customer service chat
- `finance-qa` - Financial question answering
- `warehouse-pick` - Robotic picking actions

Each route has:
- A base model reference
- An active (stable) adapter
- A fallback adapter for rollback
- An optional canary adapter

---

### Feed

A **Feed** is a reference pointer to your data source. TensorGuardFlow does NOT store your data.

Supported feed types:
- `s3` - AWS S3 bucket
- `gcs` - Google Cloud Storage
- `azure_blob` - Azure Blob Storage
- `hf_dataset` - Hugging Face dataset
- `local` - Local filesystem

The feed tracks:
- Last ingest hash (for novelty detection)
- Schema hash (for drift detection)
- Privacy mode setting

---

### Policy

A **Policy** controls the stability vs plasticity tradeoff:

| Setting | Purpose |
|---------|---------|
| `novelty_threshold` | Below this = skip training (save compute) |
| `forgetting_budget` | Max allowed forgetting of old knowledge |
| `regression_budget` | Max regression on held-out evaluation |
| `max_fast_adapters` | Cap on FAST lane adapters |
| `max_total_adapters` | Total adapter cap before archival |
| `update_cadence` | How often to check for updates |

---

### AdapterLifecycleState

Each adapter moves through a lifecycle:

```
CANDIDATE → SHADOW → CANARY → STABLE → ARCHIVED
```

**Lanes:**
- **FAST** - Recent adaptations, more plastic
- **SLOW** - Consolidated knowledge, more stable

---

## The Continuous Loop

Every `run_once()` executes this loop:

```
INGEST → NOVELTY_CHECK → TRAIN → EVAL → PACKAGE → REGISTER → PROMOTE → CONSOLIDATE
```

### 1. INGEST

Read the feed reference and compute content hash. We don't pull actual data - just metadata for tracking.

### 2. NOVELTY_CHECK

Compare current data against last stable snapshot. If novelty is below threshold, skip training entirely.

**Novelty Detection Methods:**
- Hash change (40%) - Did content change?
- Centroid drift (30%) - Did embedding space shift?
- Keyword drift (30%) - Did vocabulary change?

### 3. TRAIN

Fine-tune a new adapter using PEFT/LoRA. This is real training (or simulated with `TG_SIMULATION=true`).

### 4. EVAL

Check three gates:
1. **Primary Metric** - Does it meet quality threshold?
2. **Forgetting Score** - Did it forget old tasks?
3. **Regression Score** - Did it regress on held-out tasks?

### 5. PACKAGE

Build a deterministic TGSP bundle with:
- Adapter weights
- Manifest with privacy claims
- Evidence chain link

### 6. REGISTER

Add adapter to route's registry with full provenance.

### 7. PROMOTE

Based on policy:
- Auto-promote to CANARY
- Auto-promote to STABLE
- Or await manual promotion

### 8. CONSOLIDATE

Enforce adapter caps:
- Archive oldest FAST lane adapters
- Merge FAST → SLOW on cadence

---

## Forgetting Score

**Forgetting** measures how much the model has lost knowledge of previous tasks.

```
forgetting = (old_task_score_before - old_task_score_after) / old_task_score_before
```

A forgetting score of 0.05 means 5% performance loss on old tasks.

**Why It Matters:**
- Catastrophic forgetting is the primary risk in continual learning
- TensorGuardFlow makes forgetting a first-class gate - if exceeded, candidate is rejected

---

## FAST vs SLOW Lanes

Inspired by Complementary Learning Systems (CLS) theory:

| Lane | Purpose |
|------|---------|
| **FAST** | Recent adaptations, quickly updated, higher plasticity |
| **SLOW** | Consolidated knowledge, rarely updated, higher stability |

Adapters start in FAST lane and may be consolidated to SLOW on cadence.

---

## Release Channels

| Channel | Purpose |
|---------|---------|
| **CANDIDATE** | Just trained, awaiting evaluation |
| **SHADOW** | Evaluated, running in background (not serving traffic) |
| **CANARY** | Active for small traffic or explicit testing |
| **STABLE** | Production traffic |
| **ARCHIVED** | No longer active, kept for rollback |

---

## Privacy Mode (N2HE)

When `privacy_mode = n2he`:
- Routing decisions computed on encrypted features
- No plaintext prompts exposed
- Privacy receipts in evidence chain
- Timeline shows "Routing encrypted"

**Profiles:**
- `router_only` - Encrypted routing decisions (default)
- `router_plus_eval` - Encrypted routing AND evaluation

---

## Rollback

TensorGuardFlow maintains a fallback adapter for instant rollback:

1. When promoting to STABLE, previous stable becomes FALLBACK
2. Rollback swaps STABLE ← FALLBACK
3. One API call, instant recovery

---

## Export/Portability

Routes can be exported for external execution on:
- Kubernetes
- AWS SageMaker
- GCP Vertex AI
- Azure ML
- Databricks

Exports are **templates** - TensorGuardFlow is the control plane, not a replacement for cloud compute.
