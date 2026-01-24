# Dashboard User Guide: Analytics Console

Welcome to the **TensorGuardFlow Analytics Console**. This guide helps you navigate the continuous learning performance and pipeline health of your routes.

## 1. Route Health Score
Located at the top of each route dashboard, the **Health Score (0-100)** provides an instant assessment of route status:
- **Emerald (80-100)**: All systems nominal. Accuracy high, forgetting minimal.
- **Amber (50-79)**: Warning. Check score reasons (e.g., elevated forgetting or missing evidence).
- **Rose (<50)**: Critical. Automatic loop paused or manual intervention required.

## 2. Tabs & Views

### Learning Tab
Visualizes the **Accuracy Trend** over time. 
- Use this to monitor for "Catastrophic Forgetting."
- The **BWT/FWT** meters show how well the route balances new knowledge with old.

### PEFT Tab
Monitors parameter and storage efficiency.
- **Trainable Params**: Target <1% for optimal efficiency.
- **Adapter Growth**: High growth with low accuracy gains suggests "Data Saturation."

### Ops Tab
Operational breakdown of the continuous learning pipeline.
- **Execution Timeline**: Live updates of Ingest → Train → Eval steps.
- **Pipeline Topology**: A visual map of how tools (S3, MLflow, Triton) are interconnected.

### Trust Tab
Compliance and privacy overhead monitoring.
- **Privacy Overhead**: Monitors the N2HE latency tax.
- **Audit Evidence**: Direct links to cryptographically signed TGSP packages.

## 3. Best Practices
1. **Enable Autopromote**: Use for mature routes where the Health Score consistently stays >90.
2. **Review Rollback Readiness**: Ensure fallback adapters are registered before major data ingestions.
3. **Monitor Saturation**: If Forgetting Mean exceeds 10% frequently, consider increasing LoRA rank or adding a Consolidation stage.
