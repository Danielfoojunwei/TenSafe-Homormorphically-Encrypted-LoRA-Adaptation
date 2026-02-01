# TenSafe System Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to consolidate the TenSafe codebase, remove all mock/stub/simulated implementations, and create a unified production-ready pipeline.

---

## Current Architecture Issues

### 1. Fragmented Pipeline Architecture

The system currently has **three disconnected execution paths**:

```
Path 1: Client SDK (tg_tinker) → Server API → MockMLBackend (MOCK)
Path 2: TenSafePipeline → _MockModel + _MockOptimizer (MOCK)
Path 3: Direct RLVR Training → MockRolloutSampler (MOCK)
```

**Problem**: No unified flow from configuration to actual ML operations.

### 2. Mock/Stub Inventory (Production Code)

| File | Mock Component | Severity | Description |
|------|---------------|----------|-------------|
| `worker.py:19` | `MockMLBackend` | **CRITICAL** | Server uses mock for ALL training operations |
| `dp.py:203` | `MomentsAccountant` | **CRITICAL** | Privacy accounting is placeholder |
| `dp.py:242` | `PRVAccountant` | **CRITICAL** | Privacy accounting is placeholder |
| `pipeline.py:923` | `_MockModel` | **HIGH** | Training uses mock model |
| `pipeline.py:945` | `_MockOptimizer` | **HIGH** | Training uses mock optimizer |
| `pipeline.py:898` | `_mock_dataloader()` | **HIGH** | Generates random data |
| `pipeline.py:519` | `_generate_rollouts()` | **HIGH** | RLVR placeholder |
| `tgsp_adapter_registry.py:373` | `_mock_extract_tgsp()` | **HIGH** | Falls back to fake extraction |
| `tgsp_adapter_registry.py:417` | Random weights fallback | **HIGH** | Creates random LoRA weights |
| `inference.py:304` | Identity forward | **MEDIUM** | No actual model inference |
| `inference.py:392` | Mock tokenization | **MEDIUM** | Placeholder generation |
| `he_interface.py:613` | HEXL NotImplementedError | **MEDIUM** | Missing operations |
| `rollout.py:309` | `MockRolloutSampler` | **MEDIUM** | Used in examples |

### 3. Organizational Issues

1. **Duplicate Components**: Both `tensafe/` and `src/tensorguard/` have overlapping functionality
2. **No Central Registry**: ML backends, privacy accountants scattered across modules
3. **Inconsistent Interfaces**: Client SDK and pipeline have different APIs
4. **Missing Integration Layer**: No connection between server API and actual training

---

## Refactoring Plan

### Phase 1: Core Backend Consolidation

#### 1.1 Create Unified ML Backend Interface

**Location**: `tensafe/backends/ml_backend.py`

```python
class MLBackendInterface(ABC):
    """Unified interface for ML operations."""

    @abstractmethod
    def initialize_model(self, config: ModelConfig) -> None: ...

    @abstractmethod
    def forward_backward(self, batch: Batch, dp_config: DPConfig) -> ForwardBackwardResult: ...

    @abstractmethod
    def optim_step(self, dp_config: DPConfig) -> OptimStepResult: ...

    @abstractmethod
    def sample(self, prompts: List[str], config: SamplingConfig) -> List[Sample]: ...

    @abstractmethod
    def save_state(self) -> bytes: ...

    @abstractmethod
    def load_state(self, state: bytes) -> int: ...
```

**Implementations**:
- `TorchMLBackend` - PyTorch/Transformers implementation
- `MockMLBackend` - For testing only (moved, not in production path)

#### 1.2 Integrate Production Privacy Accountants

**Location**: `tensafe/privacy/accountants.py`

Replace placeholder implementations with:
- **dp-accounting library** (Google's production-grade implementation)
- **Opacus** for PyTorch integration

```python
class ProductionPrivacyAccountant:
    """Production-grade privacy accounting using dp-accounting."""

    def __init__(self, accountant_type: str = "rdp"):
        if accountant_type == "prv":
            from dp_accounting.pld import PLDAccountant
            self._accountant = PLDAccountant(...)
        elif accountant_type == "rdp":
            from dp_accounting.rdp import RdpAccountant
            self._accountant = RdpAccountant(...)
```

### Phase 2: Unified Pipeline Architecture

#### 2.1 Create Central Pipeline Orchestrator

**Location**: `tensafe/core/orchestrator.py`

```
                    ┌─────────────────────────┐
                    │   TenSafeOrchestrator   │
                    │  (Central Entry Point)  │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  Config Layer │      │  Backend Layer │      │  Privacy Layer│
│   (Unified)   │      │  (ML + HE)     │      │  (DP + Audit) │
└───────────────┘      └───────────────┘      └───────────────┘
```

#### 2.2 Unified Entry Points

1. **CLI**: `tensafe train`, `tensafe serve`, `tensafe inference`
2. **SDK**: `TenSafeClient`, `TrainingSession`, `InferenceSession`
3. **Server API**: `/api/v1/training`, `/api/v1/inference`

All entry points use the same underlying `TenSafeOrchestrator`.

### Phase 3: Remove Mock Fallbacks

#### 3.1 TGSP Adapter Registry

**Changes**:
- Remove `_mock_extract_tgsp()`
- Remove random weights fallback
- Make TGSP service a required dependency for encrypted inference
- Add clear error messages when dependencies missing

#### 3.2 Inference Engine

**Changes**:
- Remove mock tokenization
- Remove mock generation
- Integrate with HuggingFace tokenizers/models
- Add proper model loading from checkpoints

#### 3.3 HE Backend

**Changes**:
- Implement missing HEXL operations
- Add proper error handling for unavailable backends
- Remove toy backend from production code paths

### Phase 4: Backend Registry System

**Location**: `tensafe/backends/registry.py`

```python
class BackendRegistry:
    """Central registry for all backends."""

    _ml_backends: Dict[str, Type[MLBackendInterface]]
    _he_backends: Dict[str, Type[HEBackendInterface]]
    _privacy_accountants: Dict[str, Type[PrivacyAccountant]]

    @classmethod
    def register_ml_backend(cls, name: str, backend_class: Type): ...

    @classmethod
    def get_ml_backend(cls, name: str, config: Dict) -> MLBackendInterface: ...
```

---

## Implementation Order

### Step 1: Create Backend Interfaces (No Breaking Changes)
1. Create `tensafe/backends/` directory structure
2. Define `MLBackendInterface`
3. Implement `TorchMLBackend`
4. Create backend registry

### Step 2: Integrate Production Privacy Accountants
1. Add dp-accounting to dependencies
2. Implement `ProductionRDPAccountant`
3. Implement `ProductionPRVAccountant`
4. Update `DPTrainer` to use new accountants

### Step 3: Create Unified Orchestrator
1. Create `TenSafeOrchestrator`
2. Wire up backend registry
3. Connect to existing pipeline components

### Step 4: Refactor Server Worker
1. Replace `MockMLBackend` with `TorchMLBackend`
2. Update job handlers
3. Add proper error handling

### Step 5: Fix TGSP and Inference
1. Remove mock fallbacks from TGSP
2. Implement real model loading
3. Complete HEXL operations

### Step 6: Update Entry Points
1. Update CLI to use orchestrator
2. Update SDK to use orchestrator
3. Verify all paths work

---

## File Changes Summary

### New Files
- `tensafe/backends/__init__.py`
- `tensafe/backends/ml_backend.py` - ML backend interface and implementations
- `tensafe/backends/registry.py` - Backend registry
- `tensafe/privacy/accountants.py` - Production privacy accountants
- `tensafe/core/orchestrator.py` - Central orchestrator

### Modified Files
- `src/tensorguard/platform/tg_tinker_api/worker.py` - Use real backend
- `src/tensorguard/platform/tg_tinker_api/dp.py` - Use production accountants
- `tensafe/core/pipeline.py` - Remove mock classes
- `tensafe/tgsp_adapter_registry.py` - Remove mock fallbacks
- `tensafe/core/inference.py` - Real model loading
- `tensafe/core/he_interface.py` - Complete implementations

### Files to Keep (Test Only)
- Mock classes moved to `tests/mocks/`

---

## Testing Strategy

1. **Unit Tests**: Each new backend implementation
2. **Integration Tests**: Full pipeline from config to training
3. **E2E Tests**: Complete training run with real models
4. **Regression Tests**: Ensure existing functionality works

---

## Success Criteria

1. ✅ Zero mock implementations in production code paths
2. ✅ Single unified entry point for all operations
3. ✅ Production-grade privacy accounting
4. ✅ Real ML backend with PyTorch/Transformers
5. ✅ Complete HEXL operations
6. ✅ Clear separation between production and test code
7. ✅ All existing tests pass

---

## Timeline

| Phase | Description | Est. Effort |
|-------|-------------|-------------|
| 1 | Backend Consolidation | Core |
| 2 | Unified Pipeline | High |
| 3 | Remove Mocks | Medium |
| 4 | Registry System | Low |
| 5 | Testing & Validation | Medium |
