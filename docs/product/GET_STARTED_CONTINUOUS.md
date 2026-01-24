# Get Started with Continuous Learning

This guide will have you running continuous learning in under 30 minutes.

## What You'll Do

1. Create a Route (a continuous-learning "topic")
2. Connect a data feed reference
3. Configure stability/plasticity budgets
4. Run the continuous loop
5. See adapter candidates appear
6. Promote to stable with one click
7. Rollback instantly if needed

---

## Prerequisites

- TensorGuardFlow Core running locally or deployed
- Access to a data source (HuggingFace dataset, S3, GCS, etc.)
- Python 3.10+

---

## Step 1: Create a Route

A **Route** is the unit of continuous learning. One route = one evolving adapter family.

```bash
curl -X POST http://localhost:8000/api/v1/tgflow/routes \
  -H "Content-Type: application/json" \
  -d '{
    "route_key": "customer-support",
    "base_model_ref": "microsoft/phi-2",
    "description": "Customer support chat adapter"
  }'
```

**Response:**
```json
{
  "ok": true,
  "route_key": "customer-support",
  "message": "Route 'customer-support' created. Next: connect a feed."
}
```

---

## Step 2: Connect a Feed

A **Feed** is a reference pointer to your data. TensorGuardFlow doesn't store your data.

```bash
curl -X POST http://localhost:8000/api/v1/tgflow/routes/customer-support/feed \
  -H "Content-Type: application/json" \
  -d '{
    "feed_type": "hf_dataset",
    "feed_uri": "tatsu-lab/alpaca",
    "privacy_mode": "off"
  }'
```

**Privacy Mode Options:**
- `off` - Standard operation
- `n2he` - Encrypted routing (recommended for sensitive data)

---

## Step 3: Set Policy

A **Policy** controls the stability vs plasticity dial.

```bash
curl -X POST http://localhost:8000/api/v1/tgflow/routes/customer-support/policy \
  -H "Content-Type: application/json" \
  -d '{
    "novelty_threshold": 0.3,
    "forgetting_budget": 0.1,
    "regression_budget": 0.05,
    "update_cadence": "daily",
    "auto_promote_to_canary": true
  }'
```

**Key Settings:**
| Setting | Default | Meaning |
|---------|---------|---------|
| `novelty_threshold` | 0.3 | Below this = skip training |
| `forgetting_budget` | 0.1 | Max allowed forgetting |
| `regression_budget` | 0.05 | Max regression on held-out tasks |
| `update_cadence` | daily | How often to check for updates |

---

## Step 4: Run the Loop

Execute one complete continuous learning loop:

```bash
curl -X POST http://localhost:8000/api/v1/tgflow/routes/customer-support/run_once
```

**Response:**
```json
{
  "ok": true,
  "loop_id": "loop-abc123",
  "verdict": "success",
  "summary": "Adapter adapter-001 produced and registered",
  "adapter_produced": "adapter-001",
  "promoted_to": "canary"
}
```

**What Happens:**
1. **INGEST** - Read feed reference, compute content hash
2. **NOVELTY_CHECK** - Compare against previous snapshot
3. **TRAIN** - Fine-tune adapter using PEFT/LoRA
4. **EVAL** - Check quality, forgetting, regression gates
5. **PACKAGE** - Build deterministic TGSP bundle
6. **REGISTER** - Add to route's adapter registry
7. **PROMOTE** - Move to CANARY (if auto-promote enabled)

---

## Step 5: View the Timeline

See what happened in human-readable format:

```bash
curl http://localhost:8000/api/v1/tgflow/routes/customer-support/timeline
```

**Response Example:**
```json
{
  "route_key": "customer-support",
  "timeline": [
    {
      "loop_id": "loop-abc123",
      "verdict": "success",
      "summary": "Adapter adapter-001 produced and registered",
      "events": [
        {
          "stage": "novelty_check",
          "headline": "Update proposed",
          "explanation": "Data novelty (0.65) exceeds threshold (0.30).",
          "verdict": "success"
        },
        {
          "stage": "eval",
          "headline": "Evaluation passed",
          "explanation": "Quality: 94%. Forgetting: 3%. Regression: 1%.",
          "verdict": "success"
        }
      ]
    }
  ]
}
```

---

## Step 6: Promote to Stable

Once you're confident in a candidate:

```bash
curl -X POST "http://localhost:8000/api/v1/tgflow/routes/customer-support/promote?adapter_id=adapter-001&target=stable"
```

---

## Step 7: Rollback (If Needed)

One-click rollback to previous stable:

```bash
curl -X POST http://localhost:8000/api/v1/tgflow/routes/customer-support/rollback
```

**Response:**
```json
{
  "ok": true,
  "headline": "Rollback executed",
  "explanation": "Rolled back from adapter-001 to adapter-000."
}
```

---

## Step 8: Compare Changes

"What changed since last week?"

```bash
curl http://localhost:8000/api/v1/tgflow/routes/customer-support/diff
```

**Response:**
```json
{
  "diff_available": true,
  "changes": [
    {
      "field": "Primary Metric",
      "from": "92%",
      "to": "94%",
      "summary": "Quality improved by 2%"
    },
    {
      "field": "Forgetting Score",
      "from": "5%",
      "to": "3%",
      "summary": "Forgetting decreased by 2%"
    }
  ]
}
```

---

## Using the UI

Open the **Continuous Learning Console** in your browser:

1. **Dashboard** - See all active routes, stats, and quick actions
2. **Route Wizard** - Create routes with guided 4-step process
3. **Timeline** - Visual timeline of loop executions
4. **Releases** - Manage CANARY/STABLE/SHADOW channels

---

## Next Steps

- Read [Continuous Learning Concepts](CONTINUOUS_LEARNING_CONCEPTS.md)
- Configure [Privacy Mode (N2HE)](../n2he/N2HE_PRIVACY.md)
- Set up [Export for External Execution](CONTINUOUS_EXPORTS.md)
