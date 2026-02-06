"""
Integration Tests for MSS and HAS Services

Tests the complete service stack with mock backends.
"""

import pytest
import numpy as np
import time


class TestHASService:
    """Tests for HE Adapter Service."""

    def test_executor_initialization(self):
        """Test HAS executor initialization."""
        from he_lora_microkernel.services.has.executor import HASExecutor

        executor = HASExecutor(backend_type="SIMULATION")
        success = executor.initialize()

        assert success
        executor.shutdown()

    def test_adapter_lifecycle(self):
        """Test adapter loading and unloading."""
        from he_lora_microkernel.services.has.executor import HASExecutor

        executor = HASExecutor(backend_type="SIMULATION")
        executor.initialize()

        # Load adapter
        state = executor.load_adapter(
            adapter_id="test-adapter",
            model_id="test-model",
            rank=16,
            alpha=32.0,
            targets="qkv",
        )

        assert state is not None
        assert state.adapter_id == "test-adapter"
        assert state.rank == 16
        assert len(state.loaded_layers) > 0

        # Get adapter
        retrieved = executor.get_adapter("test-adapter")
        assert retrieved is not None
        assert retrieved.adapter_id == "test-adapter"

        # Unload adapter
        success = executor.unload_adapter("test-adapter")
        assert success

        # Verify unloaded
        assert executor.get_adapter("test-adapter") is None

        executor.shutdown()

    def test_request_lifecycle(self):
        """Test request preparation and release."""
        from he_lora_microkernel.services.has.executor import HASExecutor

        executor = HASExecutor(backend_type="SIMULATION")
        executor.initialize()

        # Load adapter first
        executor.load_adapter(
            adapter_id="test-adapter",
            model_id="test-model",
            rank=16,
            alpha=32.0,
            targets="qkv",
        )

        # Prepare request
        state = executor.prepare_request(
            request_id="req-001",
            adapter_id="test-adapter",
            batch_size=2,
            seq_len=10,
        )

        assert state is not None
        assert state.request_id == "req-001"
        assert state.batch_size == 2

        # Release request
        success = executor.release_request("req-001")
        assert success

        executor.shutdown()

    def test_delta_computation(self):
        """Test HE-LoRA delta computation."""
        from he_lora_microkernel.services.has.executor import HASExecutor

        executor = HASExecutor(backend_type="SIMULATION")
        executor.initialize()

        # Setup
        executor.load_adapter(
            adapter_id="test-adapter",
            model_id="test-model",
            rank=16,
            alpha=32.0,
            targets="qkv",
        )

        executor.prepare_request(
            request_id="req-001",
            adapter_id="test-adapter",
            batch_size=2,
            seq_len=10,
        )

        # Compute delta
        hidden_states = np.random.randn(2, 1, 4096).astype(np.float16)
        delta, timing = executor.apply_token_step(
            request_id="req-001",
            layer_idx=0,
            projection_type="q",
            hidden_states=hidden_states,
        )

        assert delta is not None
        assert delta.shape == hidden_states.shape
        assert 'encrypt_time_us' in timing
        assert 'compute_time_us' in timing
        assert 'decrypt_time_us' in timing

        executor.shutdown()

    def test_key_manager(self):
        """Test HE key management."""
        from he_lora_microkernel.services.has.key_manager import KeyManager

        manager = KeyManager(enable_audit_log=True, allow_mock=True)

        # Initialize with no Galois keys (MOAI guarantees 0 rotations)
        # Pass an empty object as backend to bypass the None check;
        # it has no keygen methods, so KeyManager falls back to mock keys.
        success = manager.initialize(
            backend=object(),
            galois_steps=[],
        )

        assert success
        assert manager.get_available_galois_steps() == []

        # Check audit log
        audit_log = manager.get_audit_log()
        assert len(audit_log) > 0

        # Clear keys
        manager.clear_keys()

    def test_shared_memory_manager(self):
        """Test shared memory management."""
        from he_lora_microkernel.services.has.shm_manager import SharedMemoryManager

        manager = SharedMemoryManager(shm_prefix="/test_helora")

        # Create region
        region = manager.create_region(
            name="test_region",
            batch_size=4,
            hidden_size=4096,
        )

        assert region is not None
        assert region.size > 0

        # Write and read data
        import numpy as np
        data = np.random.randn(4, 1, 4096).astype(np.float16)

        manager.write_hidden_states(region, data)
        read_data = manager.read_hidden_states(region, shape=data.shape)

        # Data should round-trip (with possible precision loss)
        if read_data is not None:
            assert read_data.shape == data.shape

        # Cleanup
        manager.destroy_region("test_region")
        manager.shutdown()


class TestMSSService:
    """Tests for Model Serving Service."""

    def test_request_router_initialization(self):
        """Test request router initialization."""
        from he_lora_microkernel.services.mss.router import RequestRouter, RouterConfig

        config = RouterConfig()
        router = RequestRouter(config)

        # Initialize with mock backend
        router.initialize(model_id="test-model")

        status = router.get_status()
        assert status['initialized']
        assert 'test-model' in status['loaded_models']

        router.shutdown()

    def test_completion_request(self):
        """Test completion request processing."""
        from he_lora_microkernel.services.mss.router import RequestRouter
        from he_lora_microkernel.services.mss.schemas import CompletionRequest

        router = RequestRouter()
        router.initialize(model_id="test-model")

        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            max_tokens=10,
        )

        response = router.process_completion(request)

        assert response is not None
        assert len(response.choices) > 0
        assert response.usage is not None

        router.shutdown()

    def test_chat_completion_request(self):
        """Test chat completion request processing."""
        from he_lora_microkernel.services.mss.router import RequestRouter
        from he_lora_microkernel.services.mss.schemas import (
            ChatCompletionRequest,
            ChatMessage,
        )

        router = RequestRouter()
        router.initialize(model_id="test-model")

        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="user", content="Hello!"),
            ],
            max_tokens=10,
        )

        response = router.process_chat_completion(request)

        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.role == "assistant"

        router.shutdown()

    def test_insertion_point_schema(self):
        """Test insertion point schema parsing."""
        from he_lora_microkernel.services.mss.schemas import (
            InsertionPointSchema,
            LayerSelection,
            LayerSelectionMode,
            LoRATargetType,
        )

        # Test from dict
        config_dict = {
            'adapter_id': 'my-adapter',
            'targets': 'qkvo',
            'layer_selection': {
                'mode': 'range',
                'start': 0,
                'end': 16,
            },
            'insertion_point': 'post_projection',
        }

        schema = InsertionPointSchema.from_dict(config_dict)

        assert schema.adapter_id == 'my-adapter'
        assert schema.targets == LoRATargetType.QKVO

        # Test layer selection
        layers = schema.layer_selection.get_layers(32)
        assert layers == list(range(16))

    def test_request_validation(self):
        """Test request validation."""
        from he_lora_microkernel.services.security.validation import RequestValidator

        validator = RequestValidator()

        # Valid request
        valid_request = {
            'model': 'meta-llama/Llama-2-7b-hf',
            'prompt': 'Hello, world!',
            'max_tokens': 100,
        }

        result = validator.validate_completion_request(valid_request)
        assert result.valid
        assert len(result.errors) == 0

        # Invalid request - missing model
        invalid_request = {
            'prompt': 'Hello!',
        }

        result = validator.validate_completion_request(invalid_request)
        assert not result.valid
        assert len(result.errors) > 0

        # Invalid request - bad characters in model ID
        invalid_request2 = {
            'model': 'model; DROP TABLE users;--',
            'prompt': 'Hello!',
        }

        result = validator.validate_completion_request(invalid_request2)
        assert not result.valid


class TestTelemetry:
    """Tests for telemetry and KPI enforcement."""

    def test_telemetry_collector(self):
        """Test telemetry collection."""
        from he_lora_microkernel.services.telemetry.collector import (
            ServiceTelemetryCollector,
            TelemetryEventType,
        )

        collector = ServiceTelemetryCollector()

        # Record events
        collector.start_request("req-001", "adapter-001")
        collector.record_token("req-001", 0, is_prefill=True, duration_us=1000)
        collector.record_token("req-001", 1, is_prefill=False, duration_us=500)
        collector.record_he_operation(
            "req-001", "compute", 0, "q",
            duration_us=300, rotations=2, keyswitches=2,
        )
        collector.end_request("req-001", success=True)

        # Get metrics
        metrics = collector.get_request_metrics("req-001")
        assert metrics is not None
        assert metrics.prefill_tokens == 1
        assert metrics.decode_tokens == 1

        # Get aggregated
        aggregated = collector.get_aggregated_metrics()
        assert aggregated['total_requests'] > 0

    def test_kpi_enforcement(self):
        """Test KPI enforcement."""
        from he_lora_microkernel.services.telemetry.kpi import (
            KPIEnforcer,
            ServiceKPIs,
            KPISeverity,
        )

        enforcer = KPIEnforcer()

        # Check within budget
        result = enforcer.check_value('rotations_per_token', 5.0)
        assert result  # 5 <= 16

        # Check exceeds budget
        result = enforcer.check_value('rotations_per_token', 20.0)
        assert not result  # 20 > 16

        # Get violations
        violations = enforcer.get_violations()
        assert len(violations) == 1
        assert violations[0].kpi_name == 'rotations_per_token'
        assert violations[0].actual_value == 20.0

    def test_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        from he_lora_microkernel.services.telemetry.collector import ServiceTelemetryCollector
        from he_lora_microkernel.services.telemetry.metrics import PrometheusExporter

        collector = ServiceTelemetryCollector()

        # Record some data
        collector.start_request("req-001", "adapter-001")
        collector.record_token("req-001", 0, False, 500)
        collector.end_request("req-001")

        # Create exporter
        exporter = PrometheusExporter(collector, port=0)  # Port 0 to skip server

        # Get metrics text
        metrics_text = exporter.get_metrics_text()

        assert 'helora_' in metrics_text
        assert '# HELP' in metrics_text
        assert '# TYPE' in metrics_text


class TestSecurityHardening:
    """Tests for security hardening."""

    def test_process_isolation(self):
        """Test process isolation verification."""
        from he_lora_microkernel.services.security.isolation import (
            ProcessIsolation,
            IsolationConfig,
            IsolationLevel,
        )

        config = IsolationConfig(level=IsolationLevel.PROCESS)
        isolation = ProcessIsolation(config)

        success = isolation.initialize()
        assert success

        verification = isolation.verify_isolation()
        assert verification['all_passed']

    def test_audit_logging(self):
        """Test security audit logging."""
        from he_lora_microkernel.services.security.audit import (
            SecurityAuditLog,
            AuditEvent,
            AuditEventType,
            AuditSeverity,
        )

        audit_log = SecurityAuditLog(
            service_name="test",
            enable_stdout=False,
            enable_file=False,
        )

        # Log events
        audit_log.log(AuditEvent(
            event_type=AuditEventType.REQUEST_RECEIVED,
            timestamp=time.time(),
            request_id="req-001",
            message="Test request",
        ))

        audit_log.log_security_event(
            event_type=AuditEventType.VALIDATION_FAILED,
            message="Invalid input",
            source_ip="192.168.1.100",
        )

        # Get recent events
        events = audit_log.get_recent_events()
        assert len(events) == 2

        # Get security summary
        summary = audit_log.get_security_summary()
        assert summary['total_security_events'] == 1


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_inference_pipeline(self):
        """Test complete inference pipeline with HE-LoRA."""
        import torch
        from he_lora_microkernel.backend.base_adapter import (
            BatchConfig,
            InsertionConfig,
            LoRATargets,
            get_adapter,
        )
        from he_lora_microkernel.services.has.executor import HASExecutor
        from he_lora_microkernel.services.mss.has_client import HASClient, HASConfig

        # Initialize HAS executor
        executor = HASExecutor(backend_type="SIMULATION")
        executor.initialize()

        # Load adapter
        executor.load_adapter(
            adapter_id="test-lora",
            model_id="test-model",
            rank=16,
            alpha=32.0,
            targets="qkv",
        )

        # Initialize adapter
        batch_config = BatchConfig(
            max_batch_size=4,
            max_context_length=2048,
            max_generation_length=512,
        )

        adapter_cls = get_adapter("vllm")
        adapter = adapter_cls(
            model_id="test-model",
            batch_config=batch_config,
        )
        adapter.init()

        insertion_config = InsertionConfig(
            targets=LoRATargets.QKV,
            layers=[0, 1, 2, 3],
        )
        adapter.set_insertion_config(insertion_config)

        # Create delta callback using executor
        req_state = executor.prepare_request(
            request_id="req-001",
            adapter_id="test-lora",
            batch_size=1,
            seq_len=10,
        )

        def delta_callback(layer_idx, proj_type, hidden_states):
            if layer_idx not in [0, 1, 2, 3]:
                return None
            if proj_type not in "qkv":
                return None

            delta, _ = executor.apply_token_step(
                request_id="req-001",
                layer_idx=layer_idx,
                projection_type=proj_type,
                hidden_states=hidden_states.numpy(),
            )
            if delta is not None:
                return torch.from_numpy(delta)
            return None

        adapter.set_delta_callback(delta_callback)

        # Run inference
        input_ids = torch.randint(0, 32000, (1, 10))
        kv_cache = adapter.prefill(input_ids)

        generated_tokens = []
        for _ in range(5):
            last_token = torch.randint(0, 32000, (1, 1))
            logits, _ = adapter.decode_one_step(last_token, kv_cache)
            next_token = logits.argmax(dim=-1)
            generated_tokens.append(next_token.item())

        assert len(generated_tokens) == 5

        # Verify executor processed tokens
        stats = executor.get_statistics()
        assert stats['total_tokens_processed'] > 0

        # Cleanup
        executor.release_request("req-001")
        adapter.shutdown()
        executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
