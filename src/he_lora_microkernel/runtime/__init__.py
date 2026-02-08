"""
HE-LoRA Microkernel Runtime

This package provides the runtime execution environment for HE-LoRA
inference with every-token execution.

Key components:
  - executor: Main execution engine for HE-LoRA computation
  - batching: Dynamic batch management and adjustment
  - telemetry: Performance monitoring and CI enforcement

Usage:
    from he_lora_microkernel.runtime import (
        HELoRAExecutor,
        BatchManager,
        TelemetryCollector,
        InvariantChecker,
    )

    # Create executor from schedule
    executor = HELoRAExecutor(schedule, backend_type)
    executor.load_weights(A, B, alpha)

    # Execute for every token
    for token in generation:
        delta = executor.execute_token(activations)
"""

# Executor
from .executor import (
    ExecutionMode,
    ExecutionContext,
    HELoRAExecutor,
    LoRAAdapterExecutor,
)

# Batching
from .batching import (
    BatchConfig,
    BatchManager,
    DynamicBatchExecutor,
    pad_activations,
    unpad_activations,
)

# Telemetry
from .telemetry import (
    MetricType,
    MetricValue,
    TelemetryCollector,
    PerformanceReporter,
    InvariantChecker,
    get_global_collector,
    reset_global_collector,
)


__all__ = [
    # Executor
    'ExecutionMode',
    'ExecutionContext',
    'HELoRAExecutor',
    'LoRAAdapterExecutor',
    # Batching
    'BatchConfig',
    'BatchManager',
    'DynamicBatchExecutor',
    'pad_activations',
    'unpad_activations',
    # Telemetry
    'MetricType',
    'MetricValue',
    'TelemetryCollector',
    'PerformanceReporter',
    'InvariantChecker',
    'get_global_collector',
    'reset_global_collector',
]
