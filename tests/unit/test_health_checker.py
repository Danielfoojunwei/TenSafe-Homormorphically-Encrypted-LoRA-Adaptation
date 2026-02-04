"""
Unit tests for the health_checker module.

Tests cover:
- Enum values and completeness
- HealthCheck dataclass properties
- HealthReport dataclass and serialization
- ProductHealthChecker scoring and status determination
- Recommendations generation
"""

import importlib.util
import json
import os
from datetime import datetime
from unittest.mock import patch

import pytest

# Import directly from the module file to avoid cryptography import issues
spec = importlib.util.spec_from_file_location(
    'health_checker',
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'tensorguard', 'analytics', 'health_checker.py')
)
health_checker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health_checker_module)

HealthStatus = health_checker_module.HealthStatus
CheckCategory = health_checker_module.CheckCategory
HealthCheck = health_checker_module.HealthCheck
HealthReport = health_checker_module.HealthReport
ProductHealthChecker = health_checker_module.ProductHealthChecker


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_enum_values(self):
        """Verify all expected status values exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_enum_completeness(self):
        """Verify enum has exactly 4 values."""
        assert len(HealthStatus) == 4


class TestCheckCategory:
    """Tests for CheckCategory enum."""

    def test_enum_values(self):
        """Verify all expected category values exist."""
        assert CheckCategory.SERVICE.value == "service"
        assert CheckCategory.PERFORMANCE.value == "performance"
        assert CheckCategory.RESOURCES.value == "resources"
        assert CheckCategory.SECURITY.value == "security"
        assert CheckCategory.BUSINESS.value == "business"
        assert CheckCategory.CUSTOMER.value == "customer"

    def test_enum_completeness(self):
        """Verify enum has exactly 6 values."""
        assert len(CheckCategory) == 6


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_healthy_check_properties(self):
        """Test properties of a healthy check."""
        check = HealthCheck(
            name="API Health",
            category=CheckCategory.SERVICE,
            status=HealthStatus.HEALTHY,
            value=200,
            threshold_warning=None,
            threshold_critical=None,
            message="API responding",
        )
        assert check.is_healthy is True
        assert check.needs_attention is False

    def test_warning_check_properties(self):
        """Test properties of a warning check."""
        check = HealthCheck(
            name="CPU Usage",
            category=CheckCategory.RESOURCES,
            status=HealthStatus.WARNING,
            value="75%",
            threshold_warning="60%",
            threshold_critical="80%",
            message="CPU at 75%",
        )
        assert check.is_healthy is False
        assert check.needs_attention is True

    def test_critical_check_properties(self):
        """Test properties of a critical check."""
        check = HealthCheck(
            name="API Down",
            category=CheckCategory.SERVICE,
            status=HealthStatus.CRITICAL,
            value=None,
            threshold_warning=None,
            threshold_critical=None,
            message="API unreachable",
        )
        assert check.is_healthy is False
        assert check.needs_attention is True

    def test_degraded_check_properties(self):
        """Test properties of a degraded check."""
        check = HealthCheck(
            name="Latency",
            category=CheckCategory.PERFORMANCE,
            status=HealthStatus.DEGRADED,
            value="120ms",
            threshold_warning="100ms",
            threshold_critical="200ms",
            message="Latency elevated",
        )
        assert check.is_healthy is False
        assert check.needs_attention is False  # Degraded doesn't need immediate attention

    def test_check_with_details(self):
        """Test check with additional details."""
        check = HealthCheck(
            name="API Health",
            category=CheckCategory.SERVICE,
            status=HealthStatus.HEALTHY,
            value=200,
            threshold_warning=None,
            threshold_critical=None,
            message="API responding",
            details={"latency_ms": 50, "response_size": 1024},
        )
        assert check.details["latency_ms"] == 50
        assert check.details["response_size"] == 1024

    def test_check_timestamp(self):
        """Test that timestamp is automatically set."""
        before = datetime.utcnow()
        check = HealthCheck(
            name="Test",
            category=CheckCategory.SERVICE,
            status=HealthStatus.HEALTHY,
            value=1,
            threshold_warning=None,
            threshold_critical=None,
            message="Test",
        )
        after = datetime.utcnow()
        assert before <= check.timestamp <= after


class TestHealthReport:
    """Tests for HealthReport dataclass."""

    def test_report_creation(self):
        """Test basic report creation."""
        check = HealthCheck(
            name="Test",
            category=CheckCategory.SERVICE,
            status=HealthStatus.HEALTHY,
            value=1,
            threshold_warning=None,
            threshold_critical=None,
            message="OK",
        )
        report = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            overall_score=100.0,
            checks=[check],
            summary={"total": 1},
            recommendations=[],
        )
        assert report.overall_score == 100.0
        assert len(report.checks) == 1
        assert report.overall_status == HealthStatus.HEALTHY

    def test_to_dict(self):
        """Test to_dict conversion."""
        check = HealthCheck(
            name="Test",
            category=CheckCategory.SERVICE,
            status=HealthStatus.HEALTHY,
            value=1,
            threshold_warning=None,
            threshold_critical=None,
            message="OK",
        )
        report = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            overall_score=95.5,
            checks=[check],
            summary={"total": 1, "healthy": 1},
            recommendations=["Consider scaling"],
        )
        d = report.to_dict()

        assert d["overall_status"] == "healthy"
        assert d["overall_score"] == 95.5
        assert len(d["checks"]) == 1
        assert d["summary"]["total"] == 1
        assert "Consider scaling" in d["recommendations"]
        assert "generated_at" in d

    def test_to_json(self):
        """Test JSON serialization."""
        check = HealthCheck(
            name="Test",
            category=CheckCategory.SERVICE,
            status=HealthStatus.WARNING,
            value="high",
            threshold_warning="medium",
            threshold_critical="critical",
            message="Warning",
        )
        report = HealthReport(
            overall_status=HealthStatus.WARNING,
            overall_score=72.5,
            checks=[check],
            summary={"total": 1},
            recommendations=[],
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["overall_status"] == "warning"
        assert parsed["overall_score"] == 72.5
        assert parsed["checks"][0]["name"] == "Test"


class TestProductHealthChecker:
    """Tests for ProductHealthChecker class."""

    def test_instantiation(self):
        """Test checker instantiation with custom URLs."""
        checker = ProductHealthChecker(
            prometheus_url="http://prometheus:9090",
            api_url="http://api:8000",
        )
        assert checker.prometheus_url == "http://prometheus:9090"
        assert checker.api_url == "http://api:8000"

    def test_instantiation_with_env_vars(self):
        """Test checker uses environment variables."""
        with patch.dict(os.environ, {
            "PROMETHEUS_URL": "http://env-prometheus:9090",
            "TENSAFE_API_URL": "http://env-api:8000",
        }):
            checker = ProductHealthChecker()
            assert checker.prometheus_url == "http://env-prometheus:9090"
            assert checker.api_url == "http://env-api:8000"

    def test_calculate_health_score_all_healthy(self):
        """Test health score calculation with all healthy checks."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test2", CheckCategory.PERFORMANCE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test3", CheckCategory.RESOURCES, HealthStatus.HEALTHY, 1, None, None, "OK"),
        ]
        score = checker._calculate_health_score()
        assert score == 100.0

    def test_calculate_health_score_mixed(self):
        """Test health score with mixed check statuses."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test2", CheckCategory.PERFORMANCE, HealthStatus.WARNING, 1, None, None, "WARN"),
            HealthCheck("Test3", CheckCategory.RESOURCES, HealthStatus.CRITICAL, 1, None, None, "CRIT"),
        ]
        score = checker._calculate_health_score()
        assert 0 < score < 100

    def test_calculate_health_score_empty(self):
        """Test health score with no checks."""
        checker = ProductHealthChecker()
        checker._checks = []
        score = checker._calculate_health_score()
        assert score == 100.0  # Default to healthy if no checks

    def test_determine_overall_status_healthy(self):
        """Test status determination when all healthy."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test2", CheckCategory.PERFORMANCE, HealthStatus.HEALTHY, 1, None, None, "OK"),
        ]
        status = checker._determine_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_determine_overall_status_warning(self):
        """Test status determination with single warning."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test2", CheckCategory.PERFORMANCE, HealthStatus.WARNING, 1, None, None, "WARN"),
        ]
        status = checker._determine_overall_status()
        assert status == HealthStatus.WARNING

    def test_determine_overall_status_critical_on_critical_check(self):
        """Test status is critical when any check is critical."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test2", CheckCategory.PERFORMANCE, HealthStatus.CRITICAL, 1, None, None, "CRIT"),
        ]
        status = checker._determine_overall_status()
        assert status == HealthStatus.CRITICAL

    def test_determine_overall_status_critical_on_multiple_warnings(self):
        """Test status is critical when 3+ warnings exist."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.WARNING, 1, None, None, "WARN"),
            HealthCheck("Test2", CheckCategory.PERFORMANCE, HealthStatus.WARNING, 1, None, None, "WARN"),
            HealthCheck("Test3", CheckCategory.RESOURCES, HealthStatus.WARNING, 1, None, None, "WARN"),
        ]
        status = checker._determine_overall_status()
        assert status == HealthStatus.CRITICAL

    def test_generate_summary(self):
        """Test summary generation."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Test1", CheckCategory.SERVICE, HealthStatus.HEALTHY, 1, None, None, "OK"),
            HealthCheck("Test2", CheckCategory.SERVICE, HealthStatus.WARNING, 1, None, None, "WARN"),
            HealthCheck("Test3", CheckCategory.PERFORMANCE, HealthStatus.CRITICAL, 1, None, None, "CRIT"),
        ]
        summary = checker._generate_summary()

        assert summary["total_checks"] == 3
        assert summary["healthy"] == 1
        assert summary["warning"] == 1
        assert summary["critical"] == 1
        assert "by_category" in summary

    def test_generate_recommendations_cpu_critical(self):
        """Test recommendations for critical CPU."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("CPU Utilization", CheckCategory.RESOURCES, HealthStatus.CRITICAL, "95%", "60%", "80%", "High"),
        ]
        recs = checker._generate_recommendations()
        assert any("CPU" in r for r in recs)

    def test_generate_recommendations_error_critical(self):
        """Test recommendations for critical error rate."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("Error Rate", CheckCategory.PERFORMANCE, HealthStatus.CRITICAL, "10%", "0.1%", "1%", "High"),
        ]
        recs = checker._generate_recommendations()
        assert any("error" in r.lower() for r in recs)

    def test_generate_recommendations_latency_warning(self):
        """Test recommendations for latency warning."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("API Latency", CheckCategory.PERFORMANCE, HealthStatus.WARNING, "150ms", "100ms", "200ms", "Elevated"),
        ]
        recs = checker._generate_recommendations()
        assert any("latency" in r.lower() or "capacity" in r.lower() for r in recs)

    def test_generate_recommendations_deduplication(self):
        """Test that duplicate recommendations are removed."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("CPU 1", CheckCategory.RESOURCES, HealthStatus.CRITICAL, "95%", None, None, "High"),
            HealthCheck("CPU 2", CheckCategory.RESOURCES, HealthStatus.CRITICAL, "96%", None, None, "High"),
        ]
        recs = checker._generate_recommendations()
        # Should not have duplicates
        assert len(recs) == len(set(recs))


class TestHealthCheckerIntegration:
    """Integration tests for health checker components."""

    def test_full_report_generation(self):
        """Test generating a complete health report."""
        checker = ProductHealthChecker(
            prometheus_url="http://fake:9090",
            api_url="http://fake:8000",
        )

        # Mock checks to avoid actual network calls
        checker._checks = [
            HealthCheck("API", CheckCategory.SERVICE, HealthStatus.HEALTHY, 200, None, None, "OK"),
            HealthCheck("CPU", CheckCategory.RESOURCES, HealthStatus.WARNING, "75%", "60%", "80%", "High"),
            HealthCheck("Error Rate", CheckCategory.PERFORMANCE, HealthStatus.HEALTHY, "0.01%", "0.1%", "1%", "OK"),
        ]

        # Manually generate report components
        score = checker._calculate_health_score()
        status = checker._determine_overall_status()
        summary = checker._generate_summary()
        recommendations = checker._generate_recommendations()

        report = HealthReport(
            overall_status=status,
            overall_score=score,
            checks=checker._checks,
            summary=summary,
            recommendations=recommendations,
        )

        # Verify report
        assert report.overall_status == HealthStatus.WARNING
        assert 0 < report.overall_score < 100
        assert len(report.checks) == 3
        assert report.summary["total_checks"] == 3

        # Verify serialization works
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["overall_status"] == "warning"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_checks_list(self):
        """Test behavior with no checks."""
        checker = ProductHealthChecker()
        checker._checks = []

        assert checker._calculate_health_score() == 100.0
        assert checker._determine_overall_status() == HealthStatus.HEALTHY
        assert checker._generate_summary()["total_checks"] == 0
        assert checker._generate_recommendations() == []

    def test_all_critical_checks(self):
        """Test behavior when all checks are critical."""
        checker = ProductHealthChecker()
        checker._checks = [
            HealthCheck("API", CheckCategory.SERVICE, HealthStatus.CRITICAL, None, None, None, "Down"),
            HealthCheck("DB", CheckCategory.SERVICE, HealthStatus.CRITICAL, None, None, None, "Down"),
            HealthCheck("CPU", CheckCategory.RESOURCES, HealthStatus.CRITICAL, "99%", None, None, "Full"),
        ]

        score = checker._calculate_health_score()
        status = checker._determine_overall_status()

        assert score == 0.0
        assert status == HealthStatus.CRITICAL

    def test_check_with_none_value(self):
        """Test check can have None value."""
        check = HealthCheck(
            name="Test",
            category=CheckCategory.SERVICE,
            status=HealthStatus.CRITICAL,
            value=None,
            threshold_warning=None,
            threshold_critical=None,
            message="Unreachable",
        )
        assert check.value is None
        assert check.is_healthy is False

    def test_check_with_complex_details(self):
        """Test check with nested details."""
        check = HealthCheck(
            name="Complex",
            category=CheckCategory.PERFORMANCE,
            status=HealthStatus.HEALTHY,
            value=100,
            threshold_warning=80,
            threshold_critical=90,
            message="OK",
            details={
                "nested": {"level": 2},
                "list": [1, 2, 3],
                "string": "test",
            },
        )
        assert check.details["nested"]["level"] == 2
        assert len(check.details["list"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
