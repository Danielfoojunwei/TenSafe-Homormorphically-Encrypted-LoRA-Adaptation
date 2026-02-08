"""
Hybrid Scheduler for CKKS-TFHE Compiler

Schedules IR operations to minimize:
1. Scheme switching overhead
2. TFHE bootstrap count
3. CKKS rotation count (via MOAI)

Key optimizations:
- Fuse consecutive CKKS linear ops
- Hoist TFHE ops to minimize switching
- Cost-aware scheduling based on operation estimates
"""

from .cost_model import (
    CostModel,
    OpCost,
    estimate_program_cost,
)
from .scheduler import (
    ExecutionPlan,
    HybridScheduler,
    ScheduleConfig,
    SchedulePhase,
    schedule_gated_lora,
)

__all__ = [
    "HybridScheduler",
    "ScheduleConfig",
    "ExecutionPlan",
    "SchedulePhase",
    "schedule_gated_lora",
    "CostModel",
    "OpCost",
    "estimate_program_cost",
]
