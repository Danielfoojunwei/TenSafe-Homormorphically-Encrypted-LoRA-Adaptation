"""
Dashboard Data Aggregation Service.

Provides aggregated metrics and analytics for the admin dashboard:
- Total users, tenants, requests
- Usage trends (7 day, 30 day)
- Top tenants by usage
- Error rates
- Active training jobs
"""

import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    MetricDataPoint,
    ServiceHealth,
    ServiceStatus,
    SystemHealth,
    SystemMetrics,
    TenantUsage,
    TopTenantMetric,
    UsageTrend,
)
from .permissions import AdminUserContext

logger = logging.getLogger(__name__)


class DashboardService:
    """
    Service for aggregating dashboard metrics and analytics.

    In production, this would query:
    - Time-series database (e.g., Prometheus, InfluxDB)
    - Application database
    - Cache layer (Redis)

    For the foundation, we provide the interface with mock data.
    """

    def __init__(self):
        """Initialize the dashboard service."""
        self._start_time = datetime.now(timezone.utc)
        self._metrics_cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update: Optional[datetime] = None

        # Mock data stores (replace with real database queries)
        self._tenants: Dict[str, Dict[str, Any]] = {}
        self._users: Dict[str, Dict[str, Any]] = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._requests: List[Dict[str, Any]] = []

    # ==========================================================================
    # System Health
    # ==========================================================================

    async def get_system_health(self) -> SystemHealth:
        """
        Get comprehensive system health status.

        Checks health of all services and aggregates into overall status.

        Returns:
            SystemHealth with component statuses and resource utilization
        """
        services = await self._check_all_services()

        # Determine overall status
        overall_status = ServiceStatus.HEALTHY
        for service in services:
            if service.status == ServiceStatus.UNHEALTHY:
                overall_status = ServiceStatus.UNHEALTHY
                break
            elif service.status == ServiceStatus.DEGRADED:
                overall_status = ServiceStatus.DEGRADED

        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        # Get resource utilization (mock data - replace with psutil in production)
        cpu_percent = self._get_cpu_usage()
        memory_percent = self._get_memory_usage()
        disk_percent = self._get_disk_usage()

        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version=self._get_version(),
            uptime_seconds=uptime,
            services=services,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            active_connections=self._get_active_connections(),
            pending_jobs=self._get_pending_jobs_count(),
        )

    async def _check_all_services(self) -> List[ServiceHealth]:
        """Check health of all services."""
        services = []
        now = datetime.now(timezone.utc)

        # Database
        services.append(await self._check_database_health())

        # Redis/Cache
        services.append(await self._check_cache_health())

        # Message Queue
        services.append(await self._check_queue_health())

        # KMS
        services.append(await self._check_kms_health())

        # HE Service
        services.append(await self._check_he_service_health())

        # Training Worker
        services.append(await self._check_worker_health())

        return services

    async def _check_database_health(self) -> ServiceHealth:
        """Check database health."""
        # Mock implementation - replace with actual DB health check
        return ServiceHealth(
            name="database",
            status=ServiceStatus.HEALTHY,
            latency_ms=1.5,
            last_check=datetime.now(timezone.utc),
            details={"type": "postgresql", "connections": 10, "max_connections": 100},
        )

    async def _check_cache_health(self) -> ServiceHealth:
        """Check cache (Redis) health."""
        return ServiceHealth(
            name="cache",
            status=ServiceStatus.HEALTHY,
            latency_ms=0.5,
            last_check=datetime.now(timezone.utc),
            details={"type": "redis", "memory_used_mb": 256},
        )

    async def _check_queue_health(self) -> ServiceHealth:
        """Check message queue health."""
        return ServiceHealth(
            name="queue",
            status=ServiceStatus.HEALTHY,
            latency_ms=2.0,
            last_check=datetime.now(timezone.utc),
            details={"type": "redis", "pending_messages": 15},
        )

    async def _check_kms_health(self) -> ServiceHealth:
        """Check KMS health."""
        return ServiceHealth(
            name="kms",
            status=ServiceStatus.HEALTHY,
            latency_ms=5.0,
            last_check=datetime.now(timezone.utc),
            details={"provider": "vault", "keys_cached": 50},
        )

    async def _check_he_service_health(self) -> ServiceHealth:
        """Check homomorphic encryption service health."""
        return ServiceHealth(
            name="he_service",
            status=ServiceStatus.HEALTHY,
            latency_ms=10.0,
            last_check=datetime.now(timezone.utc),
            details={"backend": "tenseal", "contexts_loaded": 3},
        )

    async def _check_worker_health(self) -> ServiceHealth:
        """Check training worker health."""
        return ServiceHealth(
            name="training_worker",
            status=ServiceStatus.HEALTHY,
            latency_ms=3.0,
            last_check=datetime.now(timezone.utc),
            details={"active_workers": 4, "gpu_available": True},
        )

    # ==========================================================================
    # System Metrics
    # ==========================================================================

    async def get_system_metrics(
        self,
        period: str = "24h",
        include_trends: bool = True,
    ) -> SystemMetrics:
        """
        Get system-wide metrics.

        Args:
            period: Time period ("1h", "24h", "7d", "30d")
            include_trends: Whether to include trend data

        Returns:
            SystemMetrics with aggregated metrics
        """
        now = datetime.now(timezone.utc)
        period_delta = self._parse_period(period)

        # Get tenant metrics
        total_tenants, active_tenants, new_tenants = await self._get_tenant_metrics(
            period_delta
        )

        # Get user metrics
        total_users, active_users, new_users = await self._get_user_metrics(period_delta)

        # Get request metrics
        request_metrics = await self._get_request_metrics(period_delta)

        # Get training metrics
        training_metrics = await self._get_training_metrics(period_delta)

        # Get storage metrics
        storage_metrics = await self._get_storage_metrics()

        # Get privacy metrics
        privacy_metrics = await self._get_privacy_metrics(period_delta)

        # Build trends if requested
        trends = []
        if include_trends:
            trends = await self._get_usage_trends(period)

        return SystemMetrics(
            timestamp=now,
            period=period,
            # Tenant metrics
            total_tenants=total_tenants,
            active_tenants=active_tenants,
            new_tenants_period=new_tenants,
            # User metrics
            total_users=total_users,
            active_users=active_users,
            new_users_period=new_users,
            # Request metrics
            total_requests=request_metrics["total"],
            requests_per_second=request_metrics["rps"],
            error_rate_percent=request_metrics["error_rate"],
            avg_latency_ms=request_metrics["avg_latency"],
            p99_latency_ms=request_metrics["p99_latency"],
            # Training metrics
            active_training_jobs=training_metrics["active"],
            completed_jobs_period=training_metrics["completed"],
            failed_jobs_period=training_metrics["failed"],
            total_compute_hours=training_metrics["compute_hours"],
            # Storage metrics
            total_storage_used_gb=storage_metrics["used_gb"],
            total_artifacts=storage_metrics["artifacts"],
            # Privacy metrics
            total_dp_budget_consumed=privacy_metrics["dp_consumed"],
            total_he_operations=privacy_metrics["he_operations"],
            # Trends
            trends=trends,
        )

    async def _get_tenant_metrics(
        self, period: timedelta
    ) -> Tuple[int, int, int]:
        """Get tenant metrics."""
        # Mock data - replace with actual database queries
        total = len(self._tenants) or 25
        active = int(total * 0.8)
        new = int(total * 0.1)
        return total, active, new

    async def _get_user_metrics(self, period: timedelta) -> Tuple[int, int, int]:
        """Get user metrics."""
        # Mock data
        total = len(self._users) or 150
        active = int(total * 0.6)
        new = int(total * 0.05)
        return total, active, new

    async def _get_request_metrics(self, period: timedelta) -> Dict[str, Any]:
        """Get request metrics."""
        # Mock data
        return {
            "total": 125000,
            "rps": 45.5,
            "error_rate": 0.15,
            "avg_latency": 125.0,
            "p99_latency": 450.0,
        }

    async def _get_training_metrics(self, period: timedelta) -> Dict[str, Any]:
        """Get training job metrics."""
        # Mock data
        return {
            "active": 12,
            "completed": 45,
            "failed": 3,
            "compute_hours": 234.5,
        }

    async def _get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics."""
        # Mock data
        return {
            "used_gb": 456.7,
            "artifacts": 1234,
        }

    async def _get_privacy_metrics(self, period: timedelta) -> Dict[str, Any]:
        """Get privacy metrics."""
        # Mock data
        return {
            "dp_consumed": 45.6,
            "he_operations": 15000,
        }

    async def _get_usage_trends(self, period: str) -> List[UsageTrend]:
        """Get usage trends for various metrics."""
        trends = []

        # Requests trend
        trends.append(
            await self._build_trend("api_requests", period, "count")
        )

        # Active users trend
        trends.append(
            await self._build_trend("active_users", period, "count")
        )

        # Training jobs trend
        trends.append(
            await self._build_trend("training_jobs", period, "count")
        )

        # Error rate trend
        trends.append(
            await self._build_trend("error_rate", period, "percent")
        )

        return trends

    async def _build_trend(
        self,
        metric_name: str,
        period: str,
        unit: str,
    ) -> UsageTrend:
        """Build a usage trend for a metric."""
        period_delta = self._parse_period(period)
        now = datetime.now(timezone.utc)

        # Generate mock data points
        num_points = 24 if period in ("1h", "24h") else 30
        data_points = []
        values = []

        for i in range(num_points):
            timestamp = now - (period_delta / num_points) * (num_points - i)
            # Generate realistic-looking data
            base_value = 100 + (i * 2)  # Upward trend
            noise = random.uniform(-10, 10)
            value = max(0, base_value + noise)
            values.append(value)

            data_points.append(
                MetricDataPoint(
                    timestamp=timestamp,
                    value=value,
                    labels={"period": period},
                )
            )

        # Calculate statistics
        total = sum(values)
        average = total / len(values) if values else 0
        min_value = min(values) if values else 0
        max_value = max(values) if values else 0

        # Calculate trend (compare first half to second half)
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0
        trend_percent = (
            ((second_half_avg - first_half_avg) / first_half_avg * 100)
            if first_half_avg > 0
            else 0
        )

        return UsageTrend(
            metric_name=metric_name,
            period=period,
            data_points=data_points,
            total=total,
            average=average,
            min_value=min_value,
            max_value=max_value,
            trend_percent=round(trend_percent, 2),
        )

    # ==========================================================================
    # Top Tenants
    # ==========================================================================

    async def get_top_tenants_by_usage(
        self,
        metric: str = "requests",
        limit: int = 10,
        period: str = "30d",
    ) -> List[TopTenantMetric]:
        """
        Get top tenants by a specific metric.

        Args:
            metric: Metric to rank by ("requests", "compute_hours", "storage", "users")
            limit: Maximum number of tenants to return
            period: Time period for the metric

        Returns:
            List of top tenants with their metric values
        """
        # Mock data - replace with actual aggregation query
        mock_tenants = [
            ("tenant-001", "Acme Corp", 45000, "requests"),
            ("tenant-002", "GlobalTech", 38000, "requests"),
            ("tenant-003", "DataFlow Inc", 32000, "requests"),
            ("tenant-004", "AI Solutions", 28000, "requests"),
            ("tenant-005", "SecureML", 22000, "requests"),
        ]

        metric_units = {
            "requests": "requests",
            "compute_hours": "hours",
            "storage": "GB",
            "users": "users",
        }

        results = []
        for tenant_id, tenant_name, value, _ in mock_tenants[:limit]:
            # Adjust value based on metric
            if metric == "compute_hours":
                value = value / 1000
            elif metric == "storage":
                value = value / 5000
            elif metric == "users":
                value = value / 1000

            results.append(
                TopTenantMetric(
                    tenant_id=tenant_id,
                    tenant_name=tenant_name,
                    metric_value=value,
                    metric_unit=metric_units.get(metric, "count"),
                )
            )

        return results

    # ==========================================================================
    # Tenant Usage
    # ==========================================================================

    async def get_tenant_usage(
        self,
        tenant_id: str,
        period: str = "30d",
    ) -> TenantUsage:
        """
        Get detailed usage metrics for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            period: Time period ("7d", "30d", "90d")

        Returns:
            TenantUsage with detailed metrics
        """
        period_delta = self._parse_period(period)
        now = datetime.now(timezone.utc)
        period_start = now - period_delta

        # Mock data - replace with actual tenant-specific queries
        return TenantUsage(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=now,
            total_users=15,
            active_users=10,
            training_jobs_count=25,
            training_jobs_completed=22,
            compute_hours_used=45.5,
            storage_used_gb=12.3,
            artifacts_count=156,
            api_requests_count=5600,
            api_errors_count=23,
            dp_budget_consumed=2.5,
            he_operations_count=1200,
            estimated_cost_usd=125.50,
        )

    # ==========================================================================
    # Active Training Jobs
    # ==========================================================================

    async def get_active_training_jobs(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get list of active training jobs.

        Args:
            tenant_id: Filter by tenant (None for all tenants)
            limit: Maximum number of jobs to return

        Returns:
            List of active training jobs
        """
        # Mock data - replace with actual job query
        jobs = [
            {
                "job_id": "job-001",
                "tenant_id": "tenant-001",
                "tenant_name": "Acme Corp",
                "model": "llama-2-7b",
                "status": "running",
                "progress_percent": 45.5,
                "started_at": datetime.now(timezone.utc) - timedelta(hours=2),
                "estimated_completion": datetime.now(timezone.utc) + timedelta(hours=3),
                "compute_hours_used": 2.0,
                "dp_budget_consumed": 0.5,
            },
            {
                "job_id": "job-002",
                "tenant_id": "tenant-002",
                "tenant_name": "GlobalTech",
                "model": "mistral-7b",
                "status": "running",
                "progress_percent": 78.2,
                "started_at": datetime.now(timezone.utc) - timedelta(hours=5),
                "estimated_completion": datetime.now(timezone.utc) + timedelta(hours=1),
                "compute_hours_used": 5.0,
                "dp_budget_consumed": 1.2,
            },
        ]

        # Filter by tenant if specified
        if tenant_id:
            jobs = [j for j in jobs if j["tenant_id"] == tenant_id]

        return jobs[:limit]

    # ==========================================================================
    # Error Analytics
    # ==========================================================================

    async def get_error_analytics(
        self,
        period: str = "24h",
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get error analytics and breakdown.

        Args:
            period: Time period
            tenant_id: Filter by tenant (None for all)

        Returns:
            Error analytics with breakdown by type, endpoint, etc.
        """
        # Mock data - replace with actual error analysis
        return {
            "total_errors": 156,
            "error_rate_percent": 0.12,
            "errors_by_type": {
                "validation_error": 45,
                "authentication_error": 32,
                "rate_limit_exceeded": 28,
                "internal_error": 15,
                "not_found": 36,
            },
            "errors_by_endpoint": {
                "/v1/training_clients": 25,
                "/v1/inference": 42,
                "/v1/artifacts": 18,
                "/auth/token": 32,
            },
            "errors_by_tenant": {
                "tenant-001": 23,
                "tenant-002": 18,
                "tenant-003": 15,
            },
            "trend": [
                {"hour": 0, "count": 5},
                {"hour": 1, "count": 3},
                {"hour": 2, "count": 8},
                # ... more hours
            ],
        }

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _parse_period(self, period: str) -> timedelta:
        """Parse period string to timedelta."""
        periods = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
        }
        return periods.get(period, timedelta(days=1))

    def _get_version(self) -> str:
        """Get application version."""
        return "1.0.0"

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 35.5  # Mock value

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 62.3  # Mock value

    def _get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        try:
            import psutil
            return psutil.disk_usage("/").percent
        except ImportError:
            return 45.8  # Mock value

    def _get_active_connections(self) -> int:
        """Get number of active connections."""
        return 125  # Mock value

    def _get_pending_jobs_count(self) -> int:
        """Get number of pending jobs."""
        return 8  # Mock value


# Singleton instance
_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """Get or create the dashboard service singleton."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service
