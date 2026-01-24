from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from datetime import datetime

class MetricCategory(str, Enum):
    LEARNING_RETENTION = "LEARNING_RETENTION"
    PARAMETER_EFFICIENCY = "PARAMETER_EFFICIENCY"
    RESOURCE_OPERATIONAL = "RESOURCE_OPERATIONAL"
    QUALITATIVE_DISTRIBUTIONAL = "QUALITATIVE_DISTRIBUTIONAL"
    TRUST_PRIVACY = "TRUST_PRIVACY"

class MetricName(str, Enum):
    # LEARNING_RETENTION
    AVG_ACCURACY = "avg_accuracy"
    FORGETTING_MEAN = "forgetting_mean"
    FORGETTING_MAX = "forgetting_max"
    BWT = "bwt"
    FWT = "fwt"
    ACCURACY_INIT = "accuracy_init"
    ACCURACY_FINAL = "accuracy_final"
    
    # PARAMETER_EFFICIENCY
    TRAINABLE_PARAM_COUNT = "trainable_param_count"
    TRAINABLE_PARAM_PERCENT = "trainable_param_percent"
    ADAPTER_STORAGE_MB = "adapter_storage_mb"
    ADAPTER_COUNT = "adapter_count"
    ADAPTER_GROWTH_RATE = "adapter_growth_rate"
    MERGE_STABILITY_DELTA = "merge_stability_delta"
    
    # RESOURCE_OPERATIONAL
    TRAIN_WALL_TIME_SEC = "train_wall_time_sec"
    EVAL_WALL_TIME_SEC = "eval_wall_time_sec"
    PACKAGE_WALL_TIME_SEC = "package_wall_time_sec"
    END_TO_END_TIME_SEC = "end_to_end_time_sec"
    PEAK_GPU_MEM_MB = "peak_gpu_mem_mb"
    PEAK_CPU_MEM_MB = "peak_cpu_mem_mb"
    TOKENS_PER_SEC = "tokens_per_sec"
    
    # QUALITATIVE_DISTRIBUTIONAL
    CROSS_DOMAIN_ACCURACY = "cross_domain_accuracy"
    PERPLEXITY = "perplexity"
    LOSS_FLATNESS_PROXY = "loss_flatness_proxy"
    
    # TRUST_PRIVACY
    EVIDENCE_COMPLETENESS = "evidence_completeness"
    SIGNATURE_VERIFIED_RATE = "signature_verified_rate"
    ROLLBACK_TIME_MS = "rollback_time_ms"
    RESOLVE_LATENCY_MS = "resolve_latency_ms"
    RESOLVE_LATENCY_N2HE_MS = "resolve_latency_n2he_ms"
    PRIVACY_RECEIPT_SUCCESS_RATE = "privacy_receipt_success_rate"

class MetricUnit(str, Enum):
    PERCENT = "%"
    SECONDS = "sec"
    MILLISECONDS = "ms"
    MEGABYTES = "MB"
    COUNT = "count"
    RATIO = "ratio"
    PER_SEC = "/sec"
    JOULES = "J"
    NONE = "none"

class MetricData(BaseModel):
    """Structured metric point for ingestion and transfer."""
    name: MetricName
    category: MetricCategory
    value: Union[float, int, str]
    unit: MetricUnit = MetricUnit.NONE
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    route_key: str
    adapter_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
