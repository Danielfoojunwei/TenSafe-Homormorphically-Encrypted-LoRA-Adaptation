"""
Observability Tests

Verifies that:
1. Correlation IDs are properly propagated
2. Metrics are recorded correctly
3. Health checks work
4. Prometheus export format is correct
"""

import sys
import time

import pytest

# Import directly to avoid tensorguard's crypto imports
sys.path.insert(0, "src")


def import_observability_module():
    """Import observability module avoiding crypto conflicts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "observability",
        "src/tensorguard/platform/tg_tinker_api/observability.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["observability"] = module
    spec.loader.exec_module(module)
    return module


obs = import_observability_module()


class TestCorrelationId:
    """Test correlation ID functionality."""

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        cid1 = obs.generate_correlation_id()
        cid2 = obs.generate_correlation_id()

        assert cid1.startswith("req-")
        assert cid2.startswith("req-")
        assert cid1 != cid2  # Should be unique

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        cid = "req-test-123"
        obs.set_correlation_id(cid)

        assert obs.get_correlation_id() == cid

    def test_set_and_get_tenant_id(self):
        """Test setting and getting tenant ID."""
        tid = "tenant-test-123"
        obs.set_tenant_id(tid)

        assert obs.get_tenant_id() == tid

    def test_set_and_get_operation_id(self):
        """Test setting and getting operation ID."""
        oid = "op-test-123"
        obs.set_operation_id(oid)

        assert obs.get_operation_id() == oid


class TestMetrics:
    """Test metrics functionality."""

    def test_counter_increment(self):
        """Test counter metric increment."""
        registry = obs.MetricsRegistry()
        counter = registry.counter(
            "test_counter",
            "Test counter help",
            labels=["label1"],
        )

        counter.inc({"label1": "value1"})
        counter.inc({"label1": "value1"})
        counter.inc({"label1": "value2"})

        assert counter.values[(("label1", "value1"),)] == 2
        assert counter.values[(("label1", "value2"),)] == 1

    def test_counter_no_labels(self):
        """Test counter without labels."""
        registry = obs.MetricsRegistry()
        counter = registry.counter(
            "test_counter_no_labels",
            "Test counter help",
        )

        counter.inc()
        counter.inc(value=5)

        assert counter.values[()] == 6

    def test_gauge_set(self):
        """Test gauge metric set."""
        registry = obs.MetricsRegistry()
        gauge = registry.gauge(
            "test_gauge",
            "Test gauge help",
            labels=["service"],
        )

        gauge.set(42, {"service": "api"})
        gauge.set(100, {"service": "worker"})

        assert gauge.values[(("service", "api"),)] == 42
        assert gauge.values[(("service", "worker"),)] == 100

    def test_histogram_observe(self):
        """Test histogram metric observe."""
        registry = obs.MetricsRegistry()
        histogram = registry.histogram(
            "test_histogram",
            "Test histogram help",
            labels=["endpoint"],
            buckets=[0.1, 0.5, 1.0],
        )

        # Observe some values
        histogram.observe(0.05, {"endpoint": "/api"})  # < 0.1
        histogram.observe(0.3, {"endpoint": "/api"})   # < 0.5
        histogram.observe(0.8, {"endpoint": "/api"})   # < 1.0
        histogram.observe(2.0, {"endpoint": "/api"})   # > 1.0

        labels = (("endpoint", "/api"),)

        # Check sum and count
        assert histogram.values[labels + ("_sum",)] == pytest.approx(3.15)
        assert histogram.values[labels + ("_count",)] == 4

        # Check buckets
        assert histogram.values[labels + ("_bucket_le_0.1",)] == 1
        assert histogram.values[labels + ("_bucket_le_0.5",)] == 2
        assert histogram.values[labels + ("_bucket_le_1.0",)] == 3
        assert histogram.values[labels + ("_bucket_le_+Inf",)] == 4

    def test_metrics_export(self):
        """Test Prometheus export format."""
        registry = obs.MetricsRegistry()

        # Create various metrics
        counter = registry.counter("http_requests", "HTTP requests", labels=["method"])
        counter.inc({"method": "GET"}, 10)
        counter.inc({"method": "POST"}, 5)

        gauge = registry.gauge("active_connections", "Active connections")
        gauge.set(42)

        # Export
        output = registry.export()

        # Verify format
        assert "# HELP http_requests HTTP requests" in output
        assert "# TYPE http_requests counter" in output
        assert 'http_requests{method="GET"} 10' in output
        assert 'http_requests{method="POST"} 5' in output

        assert "# HELP active_connections Active connections" in output
        assert "# TYPE active_connections gauge" in output
        assert "active_connections 42" in output


class TestHealthChecker:
    """Test health checker functionality."""

    def test_register_and_check(self):
        """Test registering and running health checks."""
        checker = obs.HealthChecker()

        def healthy_check():
            return obs.HealthCheckResult(
                name="healthy",
                healthy=True,
                message="All good",
            )

        checker.register(healthy_check)

        result = checker.check_all()

        assert result["healthy"] is True
        assert len(result["checks"]) == 1
        assert result["checks"][0]["name"] == "healthy"
        assert result["checks"][0]["healthy"] is True

    def test_unhealthy_check(self):
        """Test unhealthy check result."""
        checker = obs.HealthChecker()

        def unhealthy_check():
            return obs.HealthCheckResult(
                name="database",
                healthy=False,
                message="Connection refused",
            )

        checker.register(unhealthy_check)

        result = checker.check_all()

        assert result["healthy"] is False
        assert result["checks"][0]["healthy"] is False
        assert "Connection refused" in result["checks"][0]["message"]

    def test_mixed_health_checks(self):
        """Test multiple checks with mixed results."""
        checker = obs.HealthChecker()

        def healthy_check():
            return obs.HealthCheckResult(
                name="api",
                healthy=True,
                message="OK",
            )

        def unhealthy_check():
            return obs.HealthCheckResult(
                name="database",
                healthy=False,
                message="Timeout",
            )

        checker.register(healthy_check)
        checker.register(unhealthy_check)

        result = checker.check_all()

        # Overall should be unhealthy if any check fails
        assert result["healthy"] is False
        assert len(result["checks"]) == 2

    def test_check_exception_handling(self):
        """Test that exceptions in checks are handled."""
        checker = obs.HealthChecker()

        def failing_check():
            raise RuntimeError("Check failed")

        checker.register(failing_check)

        result = checker.check_all()

        assert result["healthy"] is False
        assert "Check failed" in result["checks"][0]["message"]

    def test_check_latency_recorded(self):
        """Test that check latency is recorded."""
        checker = obs.HealthChecker()

        def slow_check():
            time.sleep(0.05)  # 50ms
            return obs.HealthCheckResult(
                name="slow",
                healthy=True,
                message="OK",
            )

        checker.register(slow_check)

        result = checker.check_all()

        assert result["checks"][0]["latency_ms"] >= 50


class TestTracedDecorator:
    """Test the traced decorator."""

    def test_traced_logs_operation(self, caplog):
        """Test that traced decorator logs operation."""
        import logging

        caplog.set_level(logging.INFO)

        @obs.traced("test_operation")
        def my_function():
            return "result"

        result = my_function()

        assert result == "result"
        assert "Starting operation: test_operation" in caplog.text
        assert "Completed operation: test_operation" in caplog.text

    def test_traced_logs_exception(self, caplog):
        """Test that traced decorator logs exceptions."""
        import logging

        caplog.set_level(logging.INFO)

        @obs.traced("failing_operation")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        assert "Starting operation: failing_operation" in caplog.text
        assert "Failed operation: failing_operation" in caplog.text


class TestStructuredLogFormatter:
    """Test the structured log formatter."""

    def test_format_includes_context(self):
        """Test that formatter includes context variables."""
        import logging

        # Set context
        obs.set_correlation_id("req-test-456")
        obs.set_tenant_id("tenant-test")
        obs.set_operation_id("op-test")

        # Create logger with our formatter
        handler = logging.StreamHandler()
        handler.setFormatter(obs.StructuredLogFormatter())

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = handler.format(record)

        assert "Test message" in output
        assert "req-test-456" in output
        assert "tenant-test" in output
        assert "op-test" in output


class TestPredefinedMetrics:
    """Test predefined metrics."""

    def test_request_count_exists(self):
        """Test that request count metric exists."""
        assert obs.request_count is not None
        assert obs.request_count.name == "tg_tinker_requests_total"

    def test_request_duration_exists(self):
        """Test that request duration metric exists."""
        assert obs.request_duration is not None
        assert obs.request_duration.type == obs.MetricType.HISTOGRAM

    def test_queue_depth_exists(self):
        """Test that queue depth metric exists."""
        assert obs.queue_depth is not None
        assert obs.queue_depth.type == obs.MetricType.GAUGE

    def test_dp_epsilon_exists(self):
        """Test that DP epsilon metric exists."""
        assert obs.dp_epsilon_total is not None
        assert "tenant_id" in obs.dp_epsilon_total.labels

    def test_he_operations_exists(self):
        """Test that HE operations metric exists."""
        assert obs.he_operations_total is not None
        assert "operation" in obs.he_operations_total.labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
