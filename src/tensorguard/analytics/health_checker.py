"""
TenSafe Automated Health Check System

Provides automated health monitoring, scoring, and alerting for product operations.
Run this as a scheduled job (cron) or integrate with your monitoring stack.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"      # All systems operational
    DEGRADED = "degraded"    # Minor issues, monitoring required
    WARNING = "warning"      # Needs attention soon
    CRITICAL = "critical"    # Immediate action required


class CheckCategory(Enum):
    """Categories of health checks."""
    SERVICE = "service"           # API, database, cache availability
    PERFORMANCE = "performance"   # Latency, throughput
    RESOURCES = "resources"       # CPU, memory, disk, GPU
    SECURITY = "security"         # Auth, certificates, privacy
    BUSINESS = "business"         # Revenue, users, engagement
    CUSTOMER = "customer"         # Customer health, churn risk


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    category: CheckCategory
    status: HealthStatus
    value: Any
    threshold_warning: Any
    threshold_critical: Any
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def needs_attention(self) -> bool:
        return self.status in (HealthStatus.WARNING, HealthStatus.CRITICAL)


@dataclass
class HealthReport:
    """Complete health report."""
    overall_status: HealthStatus
    overall_score: float  # 0-100
    checks: List[HealthCheck]
    summary: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "checks": [
                {
                    "name": c.name,
                    "category": c.category.value,
                    "status": c.status.value,
                    "value": c.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ProductHealthChecker:
    """
    Automated health checker for TenSafe product.

    Usage:
        checker = ProductHealthChecker()
        report = checker.run_all_checks()

        if report.overall_status == HealthStatus.CRITICAL:
            send_alert(report)
    """

    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        db_session=None,
        api_url: Optional[str] = None,
    ):
        self.prometheus_url = prometheus_url or os.environ.get(
            "PROMETHEUS_URL", "http://prometheus:9090"
        )
        self.api_url = api_url or os.environ.get(
            "TENSAFE_API_URL", "http://localhost:8000"
        )
        self.db_session = db_session
        self._checks: List[HealthCheck] = []

    def run_all_checks(self) -> HealthReport:
        """Run all health checks and generate report."""
        self._checks = []

        # Run checks by category
        self._run_service_checks()
        self._run_performance_checks()
        self._run_resource_checks()
        self._run_security_checks()
        self._run_business_checks()
        self._run_customer_checks()

        # Calculate overall score and status
        score = self._calculate_health_score()
        status = self._determine_overall_status()
        summary = self._generate_summary()
        recommendations = self._generate_recommendations()

        return HealthReport(
            overall_status=status,
            overall_score=score,
            checks=self._checks,
            summary=summary,
            recommendations=recommendations,
        )

    # ==================== SERVICE CHECKS ====================

    def _run_service_checks(self):
        """Check service availability."""

        # API Health
        self._check_api_health()

        # Database Health
        self._check_database_health()

        # Redis Health
        self._check_redis_health()

        # Pod Count
        self._check_pod_count()

    def _check_api_health(self):
        """Check API endpoint health."""
        try:
            response = httpx.get(f"{self.api_url}/health", timeout=5.0)
            is_healthy = response.status_code == 200
            latency_ms = response.elapsed.total_seconds() * 1000

            self._checks.append(HealthCheck(
                name="API Health",
                category=CheckCategory.SERVICE,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.CRITICAL,
                value=response.status_code,
                threshold_warning=None,
                threshold_critical=None,
                message=f"API responding (status={response.status_code}, latency={latency_ms:.0f}ms)" if is_healthy else "API not responding",
                details={"latency_ms": latency_ms, "status_code": response.status_code}
            ))
        except Exception as e:
            self._checks.append(HealthCheck(
                name="API Health",
                category=CheckCategory.SERVICE,
                status=HealthStatus.CRITICAL,
                value=None,
                threshold_warning=None,
                threshold_critical=None,
                message=f"API health check failed: {e}",
            ))

    def _check_database_health(self):
        """Check database connectivity."""
        try:
            result = self._query_prometheus('pg_up')
            is_up = result and float(result) == 1

            self._checks.append(HealthCheck(
                name="Database Health",
                category=CheckCategory.SERVICE,
                status=HealthStatus.HEALTHY if is_up else HealthStatus.CRITICAL,
                value=1 if is_up else 0,
                threshold_warning=None,
                threshold_critical=0,
                message="Database connected" if is_up else "Database unreachable",
            ))
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            self._checks.append(HealthCheck(
                name="Database Health",
                category=CheckCategory.SERVICE,
                status=HealthStatus.WARNING,
                value=None,
                threshold_warning=None,
                threshold_critical=None,
                message=f"Could not check database: {e}",
            ))

    def _check_redis_health(self):
        """Check Redis connectivity."""
        try:
            result = self._query_prometheus('redis_up')
            is_up = result and float(result) == 1

            self._checks.append(HealthCheck(
                name="Redis Health",
                category=CheckCategory.SERVICE,
                status=HealthStatus.HEALTHY if is_up else HealthStatus.WARNING,
                value=1 if is_up else 0,
                threshold_warning=None,
                threshold_critical=0,
                message="Redis connected" if is_up else "Redis unreachable",
            ))
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")

    def _check_pod_count(self):
        """Check number of running pods."""
        try:
            result = self._query_prometheus(
                'count(kube_pod_status_ready{namespace="tensafe",condition="true"})'
            )
            pod_count = int(float(result)) if result else 0

            if pod_count >= 3:
                status = HealthStatus.HEALTHY
                message = f"{pod_count} pods running (healthy)"
            elif pod_count >= 2:
                status = HealthStatus.WARNING
                message = f"Only {pod_count} pods running (minimum 3 recommended)"
            else:
                status = HealthStatus.CRITICAL
                message = f"Only {pod_count} pods running (critical)"

            self._checks.append(HealthCheck(
                name="Pod Count",
                category=CheckCategory.SERVICE,
                status=status,
                value=pod_count,
                threshold_warning=3,
                threshold_critical=2,
                message=message,
            ))
        except Exception as e:
            logger.warning(f"Pod count check failed: {e}")

    # ==================== PERFORMANCE CHECKS ====================

    def _run_performance_checks(self):
        """Check performance metrics."""

        # Error Rate
        self._check_error_rate()

        # Latency P95
        self._check_latency()

        # Request Rate
        self._check_request_rate()

        # Queue Depth
        self._check_queue_depth()

    def _check_error_rate(self):
        """Check API error rate."""
        try:
            result = self._query_prometheus(
                'sum(rate(tensafe_http_requests_total{status=~"5.."}[5m])) / '
                'sum(rate(tensafe_http_requests_total[5m])) * 100'
            )
            error_rate = float(result) if result else 0

            if error_rate < 0.1:
                status = HealthStatus.HEALTHY
            elif error_rate < 1.0:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Error Rate (5xx)",
                category=CheckCategory.PERFORMANCE,
                status=status,
                value=f"{error_rate:.2f}%",
                threshold_warning="0.1%",
                threshold_critical="1.0%",
                message=f"Error rate is {error_rate:.2f}%",
                details={"error_rate_percent": error_rate}
            ))
        except Exception as e:
            logger.warning(f"Error rate check failed: {e}")

    def _check_latency(self):
        """Check API latency P95."""
        try:
            result = self._query_prometheus(
                'histogram_quantile(0.95, '
                'sum(rate(tensafe_http_request_duration_seconds_bucket[5m])) by (le)) * 1000'
            )
            latency_ms = float(result) if result else 0

            if latency_ms < 100:
                status = HealthStatus.HEALTHY
            elif latency_ms < 200:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="API Latency (P95)",
                category=CheckCategory.PERFORMANCE,
                status=status,
                value=f"{latency_ms:.0f}ms",
                threshold_warning="100ms",
                threshold_critical="200ms",
                message=f"P95 latency is {latency_ms:.0f}ms",
                details={"latency_ms": latency_ms}
            ))
        except Exception as e:
            logger.warning(f"Latency check failed: {e}")

    def _check_request_rate(self):
        """Check request throughput."""
        try:
            result = self._query_prometheus(
                'sum(rate(tensafe_http_requests_total[5m]))'
            )
            rps = float(result) if result else 0

            # This is informational - no threshold
            self._checks.append(HealthCheck(
                name="Request Rate",
                category=CheckCategory.PERFORMANCE,
                status=HealthStatus.HEALTHY,
                value=f"{rps:.1f}/s",
                threshold_warning=None,
                threshold_critical=None,
                message=f"Handling {rps:.1f} requests per second",
                details={"requests_per_second": rps}
            ))
        except Exception as e:
            logger.warning(f"Request rate check failed: {e}")

    def _check_queue_depth(self):
        """Check request queue depth."""
        try:
            result = self._query_prometheus('tensafe_request_queue_depth')
            depth = int(float(result)) if result else 0

            if depth < 20:
                status = HealthStatus.HEALTHY
            elif depth < 50:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Queue Depth",
                category=CheckCategory.PERFORMANCE,
                status=status,
                value=depth,
                threshold_warning=20,
                threshold_critical=50,
                message=f"Request queue depth: {depth}",
                details={"queue_depth": depth}
            ))
        except Exception as e:
            logger.warning(f"Queue depth check failed: {e}")

    # ==================== RESOURCE CHECKS ====================

    def _run_resource_checks(self):
        """Check resource utilization."""

        # CPU
        self._check_cpu()

        # Memory
        self._check_memory()

        # GPU
        self._check_gpu()

        # Disk
        self._check_disk()

        # Database Connections
        self._check_db_connections()

    def _check_cpu(self):
        """Check CPU utilization."""
        try:
            result = self._query_prometheus(
                'avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) / '
                'avg(kube_pod_container_resource_limits{namespace="tensafe",resource="cpu"}) * 100'
            )
            cpu_percent = float(result) if result else 0

            if cpu_percent < 60:
                status = HealthStatus.HEALTHY
            elif cpu_percent < 80:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="CPU Utilization",
                category=CheckCategory.RESOURCES,
                status=status,
                value=f"{cpu_percent:.1f}%",
                threshold_warning="60%",
                threshold_critical="80%",
                message=f"CPU at {cpu_percent:.1f}%",
                details={"cpu_percent": cpu_percent}
            ))
        except Exception as e:
            logger.warning(f"CPU check failed: {e}")

    def _check_memory(self):
        """Check memory utilization."""
        try:
            result = self._query_prometheus(
                'avg(container_memory_working_set_bytes{namespace="tensafe"}) / '
                'avg(kube_pod_container_resource_limits{namespace="tensafe",resource="memory"}) * 100'
            )
            mem_percent = float(result) if result else 0

            if mem_percent < 70:
                status = HealthStatus.HEALTHY
            elif mem_percent < 85:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Memory Utilization",
                category=CheckCategory.RESOURCES,
                status=status,
                value=f"{mem_percent:.1f}%",
                threshold_warning="70%",
                threshold_critical="85%",
                message=f"Memory at {mem_percent:.1f}%",
                details={"memory_percent": mem_percent}
            ))
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

    def _check_gpu(self):
        """Check GPU utilization."""
        try:
            result = self._query_prometheus('avg(DCGM_FI_DEV_GPU_UTIL)')
            gpu_percent = float(result) if result else 0

            if gpu_percent < 70:
                status = HealthStatus.HEALTHY
            elif gpu_percent < 90:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="GPU Utilization",
                category=CheckCategory.RESOURCES,
                status=status,
                value=f"{gpu_percent:.1f}%",
                threshold_warning="70%",
                threshold_critical="90%",
                message=f"GPU at {gpu_percent:.1f}%",
                details={"gpu_percent": gpu_percent}
            ))
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")

    def _check_disk(self):
        """Check disk usage."""
        try:
            result = self._query_prometheus(
                '(1 - node_filesystem_avail_bytes{mountpoint="/"} / '
                'node_filesystem_size_bytes{mountpoint="/"}) * 100'
            )
            disk_percent = float(result) if result else 0

            if disk_percent < 70:
                status = HealthStatus.HEALTHY
            elif disk_percent < 85:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Disk Usage",
                category=CheckCategory.RESOURCES,
                status=status,
                value=f"{disk_percent:.1f}%",
                threshold_warning="70%",
                threshold_critical="85%",
                message=f"Disk at {disk_percent:.1f}%",
                details={"disk_percent": disk_percent}
            ))
        except Exception as e:
            logger.warning(f"Disk check failed: {e}")

    def _check_db_connections(self):
        """Check database connection pool."""
        try:
            result = self._query_prometheus(
                'pg_stat_activity_count / pg_settings_max_connections * 100'
            )
            conn_percent = float(result) if result else 0

            if conn_percent < 60:
                status = HealthStatus.HEALTHY
            elif conn_percent < 80:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="DB Connections",
                category=CheckCategory.RESOURCES,
                status=status,
                value=f"{conn_percent:.1f}%",
                threshold_warning="60%",
                threshold_critical="80%",
                message=f"Database connections at {conn_percent:.1f}%",
                details={"connection_percent": conn_percent}
            ))
        except Exception as e:
            logger.warning(f"DB connection check failed: {e}")

    # ==================== SECURITY CHECKS ====================

    def _run_security_checks(self):
        """Check security-related metrics."""

        # Privacy Budget
        self._check_privacy_budget()

        # Certificate Expiry
        self._check_cert_expiry()

        # Auth Failures
        self._check_auth_failures()

    def _check_privacy_budget(self):
        """Check differential privacy budget."""
        try:
            result = self._query_prometheus('max(tensafe_dp_epsilon_spent)')
            epsilon = float(result) if result else 0

            if epsilon < 5:
                status = HealthStatus.HEALTHY
            elif epsilon < 8:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Privacy Budget (Epsilon)",
                category=CheckCategory.SECURITY,
                status=status,
                value=f"{epsilon:.2f}/10",
                threshold_warning="5",
                threshold_critical="8",
                message=f"Privacy budget: {epsilon:.2f} of 10 spent",
                details={"epsilon_spent": epsilon, "epsilon_max": 10}
            ))
        except Exception as e:
            logger.warning(f"Privacy budget check failed: {e}")

    def _check_cert_expiry(self):
        """Check TLS certificate expiry."""
        try:
            result = self._query_prometheus(
                '(probe_ssl_earliest_cert_expiry - time()) / 86400'
            )
            days_until_expiry = float(result) if result else 365

            if days_until_expiry > 30:
                status = HealthStatus.HEALTHY
            elif days_until_expiry > 7:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="TLS Certificate",
                category=CheckCategory.SECURITY,
                status=status,
                value=f"{days_until_expiry:.0f} days",
                threshold_warning="30 days",
                threshold_critical="7 days",
                message=f"Certificate expires in {days_until_expiry:.0f} days",
                details={"days_until_expiry": days_until_expiry}
            ))
        except Exception as e:
            logger.warning(f"Certificate check failed: {e}")

    def _check_auth_failures(self):
        """Check authentication failure rate."""
        try:
            result = self._query_prometheus(
                'sum(rate(tensafe_auth_failures_total[5m])) * 60'
            )
            failures_per_min = float(result) if result else 0

            if failures_per_min < 5:
                status = HealthStatus.HEALTHY
            elif failures_per_min < 20:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Auth Failures",
                category=CheckCategory.SECURITY,
                status=status,
                value=f"{failures_per_min:.1f}/min",
                threshold_warning="5/min",
                threshold_critical="20/min",
                message=f"Auth failures: {failures_per_min:.1f} per minute",
                details={"failures_per_minute": failures_per_min}
            ))
        except Exception as e:
            logger.warning(f"Auth failures check failed: {e}")

    # ==================== BUSINESS CHECKS ====================

    def _run_business_checks(self):
        """Check business metrics."""

        # DAU trend
        self._check_dau_trend()

        # Error spike (customer-impacting)
        self._check_customer_errors()

    def _check_dau_trend(self):
        """Check DAU trend."""
        try:
            # This would normally query your analytics database
            # For now, use a placeholder
            self._checks.append(HealthCheck(
                name="DAU Trend",
                category=CheckCategory.BUSINESS,
                status=HealthStatus.HEALTHY,
                value="Stable",
                threshold_warning="-10%",
                threshold_critical="-20%",
                message="Daily active users stable",
            ))
        except Exception as e:
            logger.warning(f"DAU trend check failed: {e}")

    def _check_customer_errors(self):
        """Check customer-impacting errors."""
        try:
            result = self._query_prometheus(
                'sum(increase(tensafe_customer_errors_total[1h]))'
            )
            errors = int(float(result)) if result else 0

            if errors < 10:
                status = HealthStatus.HEALTHY
            elif errors < 50:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            self._checks.append(HealthCheck(
                name="Customer Errors (1h)",
                category=CheckCategory.BUSINESS,
                status=status,
                value=errors,
                threshold_warning=10,
                threshold_critical=50,
                message=f"{errors} customer-impacting errors in last hour",
                details={"error_count": errors}
            ))
        except Exception as e:
            logger.warning(f"Customer errors check failed: {e}")

    # ==================== CUSTOMER CHECKS ====================

    def _run_customer_checks(self):
        """Check customer health metrics."""

        # At-risk customers
        self._check_at_risk_customers()

    def _check_at_risk_customers(self):
        """Check number of at-risk customers."""
        try:
            # This would normally query your customer health database
            self._checks.append(HealthCheck(
                name="At-Risk Customers",
                category=CheckCategory.CUSTOMER,
                status=HealthStatus.HEALTHY,
                value=0,
                threshold_warning=3,
                threshold_critical=5,
                message="No at-risk customers detected",
            ))
        except Exception as e:
            logger.warning(f"At-risk customers check failed: {e}")

    # ==================== HELPER METHODS ====================

    def _query_prometheus(self, query: str) -> Optional[str]:
        """Query Prometheus and return result."""
        try:
            response = httpx.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10.0,
            )
            data = response.json()
            if data["status"] == "success" and data["data"]["result"]:
                return data["data"]["result"][0]["value"][1]
            return None
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return None

    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if not self._checks:
            return 100.0

        # Weight by category
        weights = {
            CheckCategory.SERVICE: 30,
            CheckCategory.PERFORMANCE: 25,
            CheckCategory.RESOURCES: 20,
            CheckCategory.SECURITY: 15,
            CheckCategory.BUSINESS: 5,
            CheckCategory.CUSTOMER: 5,
        }

        # Score by status
        status_scores = {
            HealthStatus.HEALTHY: 100,
            HealthStatus.DEGRADED: 75,
            HealthStatus.WARNING: 50,
            HealthStatus.CRITICAL: 0,
        }

        # Calculate weighted score per category
        category_scores = {}
        category_counts = {}

        for check in self._checks:
            cat = check.category
            score = status_scores[check.status]

            if cat not in category_scores:
                category_scores[cat] = 0
                category_counts[cat] = 0

            category_scores[cat] += score
            category_counts[cat] += 1

        # Average per category
        for cat in category_scores:
            if category_counts[cat] > 0:
                category_scores[cat] /= category_counts[cat]

        # Weighted total
        total_score = 0
        total_weight = 0
        for cat, weight in weights.items():
            if cat in category_scores:
                total_score += category_scores[cat] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 100.0

    def _determine_overall_status(self) -> HealthStatus:
        """Determine overall status from checks."""
        if not self._checks:
            return HealthStatus.HEALTHY

        # Any critical check = critical overall
        if any(c.status == HealthStatus.CRITICAL for c in self._checks):
            return HealthStatus.CRITICAL

        # Multiple warnings = critical
        warning_count = sum(1 for c in self._checks if c.status == HealthStatus.WARNING)
        if warning_count >= 3:
            return HealthStatus.CRITICAL

        # Any warning = warning overall
        if warning_count > 0:
            return HealthStatus.WARNING

        # Any degraded = degraded
        if any(c.status == HealthStatus.DEGRADED for c in self._checks):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_checks": len(self._checks),
            "healthy": sum(1 for c in self._checks if c.status == HealthStatus.HEALTHY),
            "degraded": sum(1 for c in self._checks if c.status == HealthStatus.DEGRADED),
            "warning": sum(1 for c in self._checks if c.status == HealthStatus.WARNING),
            "critical": sum(1 for c in self._checks if c.status == HealthStatus.CRITICAL),
            "by_category": {
                cat.value: {
                    "total": sum(1 for c in self._checks if c.category == cat),
                    "healthy": sum(1 for c in self._checks if c.category == cat and c.status == HealthStatus.HEALTHY),
                    "issues": sum(1 for c in self._checks if c.category == cat and c.status != HealthStatus.HEALTHY),
                }
                for cat in CheckCategory
            }
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on checks."""
        recommendations = []

        for check in self._checks:
            if check.status == HealthStatus.CRITICAL:
                if "CPU" in check.name:
                    recommendations.append("URGENT: Scale up pods immediately to reduce CPU pressure")
                elif "Memory" in check.name:
                    recommendations.append("URGENT: Scale up pods or investigate memory leak")
                elif "GPU" in check.name:
                    recommendations.append("URGENT: Add GPU nodes or reduce training workload")
                elif "Error" in check.name:
                    recommendations.append("URGENT: Investigate error spike - check recent deployments")
                elif "API" in check.name:
                    recommendations.append("URGENT: API is down - check pods and restart if needed")
                elif "Privacy" in check.name:
                    recommendations.append("URGENT: Privacy budget nearly exhausted - stop training")

            elif check.status == HealthStatus.WARNING:
                if "CPU" in check.name:
                    recommendations.append("Plan to scale up - CPU utilization trending high")
                elif "Memory" in check.name:
                    recommendations.append("Monitor memory usage - consider scaling soon")
                elif "Latency" in check.name:
                    recommendations.append("Latency elevated - consider adding capacity")
                elif "Disk" in check.name:
                    recommendations.append("Disk space running low - clean up or expand")
                elif "Certificate" in check.name:
                    recommendations.append("Renew TLS certificate soon")

        # Deduplicate
        return list(dict.fromkeys(recommendations))


# ==================== CLI INTERFACE ====================

def run_health_check(output_format: str = "text") -> int:
    """Run health check from CLI."""
    checker = ProductHealthChecker()
    report = checker.run_all_checks()

    if output_format == "json":
        print(report.to_json())
    else:
        # Text format
        print("\n" + "=" * 60)
        print(f"  TENSAFE HEALTH REPORT - {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 60)

        status_emoji = {
            HealthStatus.HEALTHY: "OK",
            HealthStatus.DEGRADED: "DEGRADED",
            HealthStatus.WARNING: "WARNING",
            HealthStatus.CRITICAL: "CRITICAL",
        }

        print(f"\n  Overall Status: [{status_emoji[report.overall_status]}]")
        print(f"  Health Score: {report.overall_score:.0f}/100")

        print(f"\n  Summary: {report.summary['healthy']} healthy, "
              f"{report.summary['warning']} warnings, "
              f"{report.summary['critical']} critical")

        # Group by category
        print("\n" + "-" * 60)
        for category in CheckCategory:
            category_checks = [c for c in report.checks if c.category == category]
            if not category_checks:
                continue

            print(f"\n  {category.value.upper()}")
            for check in category_checks:
                status_marker = {
                    HealthStatus.HEALTHY: "[OK]",
                    HealthStatus.DEGRADED: "[DEGRADED]",
                    HealthStatus.WARNING: "[WARN]",
                    HealthStatus.CRITICAL: "[CRIT]",
                }[check.status]
                print(f"    {status_marker} {check.name}: {check.value}")

        if report.recommendations:
            print("\n" + "-" * 60)
            print("\n  RECOMMENDATIONS:")
            for rec in report.recommendations:
                print(f"    - {rec}")

        print("\n" + "=" * 60 + "\n")

    # Return exit code based on status
    if report.overall_status == HealthStatus.CRITICAL:
        return 2
    elif report.overall_status == HealthStatus.WARNING:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "text"
    exit_code = run_health_check(output)
    sys.exit(exit_code)
