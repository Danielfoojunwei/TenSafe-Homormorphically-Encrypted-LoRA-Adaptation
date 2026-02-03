"""
Founder Dashboard

Real-time visibility into business health, user behavior, and operations.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from .metrics import BusinessMetrics, UserMetrics, OperationalMetrics

logger = logging.getLogger(__name__)


@dataclass
class DashboardSection:
    """A section of the dashboard."""

    title: str
    metrics: Dict[str, Any]
    alerts: List[str] = None
    trend: str = None  # "up", "down", "stable"


class FounderDashboard:
    """
    Executive dashboard for founders and leadership.

    Provides at-a-glance visibility into:
    - Revenue and growth metrics
    - User engagement and retention
    - Product health and performance
    - Operational status
    """

    def __init__(self, db_session=None, prometheus_url: Optional[str] = None):
        self.business = BusinessMetrics(db_session)
        self.users = UserMetrics(db_session)
        self.ops = OperationalMetrics(prometheus_url)

    def get_executive_summary(self) -> Dict[str, Any]:
        """
        Get high-level executive summary.

        This is what you'd look at every morning.
        """
        now = datetime.utcnow()

        return {
            "generated_at": now.isoformat(),
            "summary": {
                "mrr": self.business.get_mrr().formatted_value,
                "mrr_growth": self._calculate_mrr_growth(),
                "active_users": self.users.get_active_users("day").value,
                "churn_rate": self.business.get_churn_rate(
                    now - timedelta(days=30), now
                ).formatted_value,
                "uptime": self.ops.get_uptime().formatted_value,
            },
            "alerts": self._get_active_alerts(),
            "status": self._get_overall_status(),
        }

    def get_revenue_dashboard(self) -> DashboardSection:
        """Revenue and monetization metrics."""
        now = datetime.utcnow()
        month_ago = now - timedelta(days=30)

        metrics = self.business.get_dashboard_metrics()

        return DashboardSection(
            title="Revenue",
            metrics={
                "mrr": metrics["mrr"].formatted_value,
                "arr": metrics["arr"].formatted_value,
                "arpu": metrics["arpu"].formatted_value,
                "ltv": metrics["ltv"].formatted_value,
                "nrr": metrics["nrr"].formatted_value,
            },
            alerts=self._get_revenue_alerts(),
            trend=self._calculate_revenue_trend(),
        )

    def get_growth_dashboard(self) -> DashboardSection:
        """User growth and acquisition metrics."""
        journey = self.users.get_user_journey_metrics()

        return DashboardSection(
            title="Growth",
            metrics={
                "signups_7d": journey["signups_7d"],
                "activated_7d": journey["activated_7d"],
                "converted_7d": journey["converted_7d"],
                "activation_rate": f"{journey['activation_rate']:.1f}%",
                "conversion_rate": f"{journey['conversion_rate']:.1f}%",
            },
            alerts=self._get_growth_alerts(journey),
        )

    def get_engagement_dashboard(self) -> DashboardSection:
        """User engagement metrics."""
        return DashboardSection(
            title="Engagement",
            metrics={
                "dau": self.users.get_active_users("day").value,
                "wau": self.users.get_active_users("week").value,
                "mau": self.users.get_active_users("month").value,
                "dau_wau_ratio": self._calculate_stickiness(),
                "d7_retention": self.users.get_retention_rate(
                    datetime.utcnow() - timedelta(days=7), 7
                ).formatted_value,
            },
        )

    def get_product_dashboard(self) -> DashboardSection:
        """Product usage and feature adoption."""
        privacy_features = self.users.get_privacy_feature_usage()

        return DashboardSection(
            title="Product",
            metrics={
                "dp_training_adoption": privacy_features["dp_sgd"].formatted_value,
                "he_inference_adoption": privacy_features["he_lora"].formatted_value,
                "tgsp_adoption": privacy_features["tgsp"].formatted_value,
                "pqc_adoption": privacy_features["pqc"].formatted_value,
            },
            alerts=self._get_product_alerts(),
        )

    def get_operations_dashboard(self) -> DashboardSection:
        """System health and operations."""
        api_metrics = self.ops.get_api_metrics()
        infra_metrics = self.ops.get_infrastructure_metrics()

        return DashboardSection(
            title="Operations",
            metrics={
                "api_rpm": api_metrics["requests_per_minute"].value,
                "error_rate": api_metrics["error_rate"].formatted_value,
                "p99_latency": f"{api_metrics['p99_latency'].value:.0f}ms",
                "uptime_30d": self.ops.get_uptime(30).formatted_value,
                "cpu_util": infra_metrics["cpu_utilization"].formatted_value,
                "gpu_util": infra_metrics["gpu_utilization"].formatted_value,
            },
            alerts=self._get_ops_alerts(api_metrics, infra_metrics),
        )

    def get_full_dashboard(self) -> Dict[str, Any]:
        """Get all dashboard sections."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "executive_summary": self.get_executive_summary(),
            "revenue": asdict(self.get_revenue_dashboard()),
            "growth": asdict(self.get_growth_dashboard()),
            "engagement": asdict(self.get_engagement_dashboard()),
            "product": asdict(self.get_product_dashboard()),
            "operations": asdict(self.get_operations_dashboard()),
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics for external tools."""
        data = self.get_full_dashboard()

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        elif format == "csv":
            return self._to_csv(data)
        else:
            raise ValueError(f"Unknown format: {format}")

    # Private helper methods

    def _calculate_mrr_growth(self) -> str:
        """Calculate MRR month-over-month growth."""
        now = datetime.utcnow()
        current = self.business.get_mrr(now).value
        previous = self.business.get_mrr(now - timedelta(days=30)).value

        if previous == 0:
            return "N/A"

        growth = ((current - previous) / previous) * 100
        sign = "+" if growth >= 0 else ""
        return f"{sign}{growth:.1f}%"

    def _calculate_stickiness(self) -> str:
        """Calculate DAU/WAU ratio (stickiness)."""
        dau = self.users.get_active_users("day").value
        wau = self.users.get_active_users("week").value

        if wau == 0:
            return "N/A"

        stickiness = (dau / wau) * 100
        return f"{stickiness:.1f}%"

    def _calculate_revenue_trend(self) -> str:
        """Determine revenue trend."""
        now = datetime.utcnow()
        current = self.business.get_mrr(now).value
        previous = self.business.get_mrr(now - timedelta(days=30)).value

        if current > previous * 1.05:
            return "up"
        elif current < previous * 0.95:
            return "down"
        else:
            return "stable"

    def _get_active_alerts(self) -> List[str]:
        """Get active alerts requiring attention."""
        alerts = []

        # Check churn
        churn = self.business.get_churn_rate(
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        ).value
        if churn > 5:
            alerts.append(f"High churn rate: {churn:.1f}%")

        # Check error rate
        error_rate = self.ops.get_api_metrics()["error_rate"].value
        if error_rate > 1:
            alerts.append(f"Elevated error rate: {error_rate:.2f}%")

        return alerts

    def _get_overall_status(self) -> str:
        """Determine overall system status."""
        alerts = self._get_active_alerts()
        if len(alerts) == 0:
            return "healthy"
        elif len(alerts) <= 2:
            return "warning"
        else:
            return "critical"

    def _get_revenue_alerts(self) -> List[str]:
        return []

    def _get_growth_alerts(self, journey: Dict) -> List[str]:
        alerts = []
        if journey["activation_rate"] < 30:
            alerts.append("Low activation rate - review onboarding")
        if journey["conversion_rate"] < 5:
            alerts.append("Low conversion rate - review pricing/value prop")
        return alerts

    def _get_product_alerts(self) -> List[str]:
        return []

    def _get_ops_alerts(self, api: Dict, infra: Dict) -> List[str]:
        alerts = []
        if api["error_rate"].value > 1:
            alerts.append("Error rate above SLA threshold")
        if api["p99_latency"].value > 500:
            alerts.append("P99 latency above target")
        if infra["cpu_utilization"].value > 80:
            alerts.append("High CPU utilization")
        if infra["gpu_utilization"].value > 90:
            alerts.append("GPU near capacity")
        return alerts

    def _to_csv(self, data: Dict) -> str:
        """Convert metrics to CSV format."""
        lines = ["metric,value,timestamp"]
        # Flatten and add metrics
        return "\n".join(lines)
