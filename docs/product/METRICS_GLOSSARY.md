# Metrics Glossary: TGFlow Continuous Learning

This document defines the key performance indicators (KPIs) used in the TensorGuardFlow Analytics Console to evaluate continuous learning loops.

## 1. Learning Retention (CL Metrics)

### Average Accuracy (ACC)
The mean performance across all observed tasks at the current time step.
- **Formula**: $(1/N) \sum_{i=1}^N R_{N,i}$
- **Goal**: Maximize. Indicates generalist capability.

### Forgetting Mean / Max
Measure of how much performance on previous tasks has degraded after learning new data.
- **Formula**: $f_{t,i} = \max_{j < t} R_{j,i} - R_{t,i}$
- **Goal**: Minimize. Values >10% typically trigger architectural "Skill Lock" or replay buffer increases.

### Backward Transfer (BWT)
The influence that learning a new task has on the performance of previous tasks.
- **Positive BWT**: Learning Task B improved Task A (synergy).
- **Negative BWT**: Learning Task B degraded Task A (interference/forgetting).

---

## 2. Parameter Efficiency (PEFT)

### Trainable Parameter %
The ratio of adapter parameters (e.g., LoRA weights) to the total base model parameters.
- **Typical Range**: 0.01% - 1.0%
- **Goal**: Minimize while maintaining ACC.

### Adapter Storage (MB)
The disk/memory footprint of the serialized adapter artifact.
- **Goal**: Minimize for edge deployment efficiency.

---

## 3. Resource & Operational (Ops)

### End-to-End Time (sec)
Total wall time for one full iteration (Ingest → Promote).
- **Goal**: Optimize for high-cadence streaming updates.

### Peak GPU Memory (MB)
The maximum VRAM consumed during the `TRAIN` or `EVAL` stage.
- **Goal**: Stay within fleet hardware constraints.

---

## 4. Trust & Privacy

### Evidence Completeness
Ratio of required compliance artifacts (Lineage, Training Logs, Evaluation Proofs) present in the TGSP manifest.
- **Goal**: 1.0 (100%).

### N2HE Privacy Overhead
The Relative latency increase when performing inference/evaluation through the Shielded Privacy Layer compared to plaintext.
- **Formula**: $(Latency_{Encrypted} - Latency_{Plain}) / Latency_{Plain}$
