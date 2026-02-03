"""
Business and User Metrics

KPI tracking and reporting for founders and operators.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """A point-in-time metric value."""

    name: str
    value: float
    timestamp: datetime
    dimensions: Dict[str, str] = None

    @property
    def formatted_value(self) -> str:
        """Format value based on metric type."""
        if "rate" in self.name.lower() or "percentage" in self.name.lower():
            return f"{self.value:.1f}%"
        elif "revenue" in self.name.lower() or "mrr" in self.name.lower():
            return f"${self.value:,.2f}"
        elif "time" in self.name.lower() and "ms" not in self.name.lower():
            return f"{self.value:.1f}s"
        else:
            return f"{self.value:,.0f}"


class BusinessMetrics:
    """
    Business metrics tracking for SaaS KPIs.

    Key metrics tracked:
    - MRR (Monthly Recurring Revenue)
    - ARR (Annual Recurring Revenue)
    - Churn rate
    - ARPU (Average Revenue Per User)
    - LTV (Lifetime Value)
    - CAC (Customer Acquisition Cost)
    - Net Revenue Retention
    """

    def __init__(self, db_session=None):
        self.db = db_session

    def get_mrr(self, as_of: Optional[datetime] = None) -> MetricSnapshot:
        """
        Calculate Monthly Recurring Revenue.

        MRR = Sum of all active subscription monthly values
        """
        as_of = as_of or datetime.utcnow()

        # Query active subscriptions
        # In production, query from billing tables
        mrr = self._calculate_mrr_from_db(as_of)

        return MetricSnapshot(
            name="MRR",
            value=mrr,
            timestamp=as_of,
        )

    def get_arr(self, as_of: Optional[datetime] = None) -> MetricSnapshot:
        """Calculate Annual Recurring Revenue (MRR * 12)."""
        mrr = self.get_mrr(as_of)
        return MetricSnapshot(
            name="ARR",
            value=mrr.value * 12,
            timestamp=mrr.timestamp,
        )

    def get_churn_rate(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> MetricSnapshot:
        """
        Calculate churn rate for a period.

        Churn Rate = (Customers Lost / Customers at Start) * 100
        """
        customers_start = self._count_customers_at(period_start)
        customers_lost = self._count_churned_customers(period_start, period_end)

        churn_rate = (customers_lost / max(customers_start, 1)) * 100

        return MetricSnapshot(
            name="Churn Rate",
            value=churn_rate,
            timestamp=period_end,
        )

    def get_net_revenue_retention(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> MetricSnapshot:
        """
        Calculate Net Revenue Retention (NRR).

        NRR = (Starting MRR + Expansion - Contraction - Churn) / Starting MRR * 100
        """
        starting_mrr = self._calculate_mrr_from_db(period_start)
        ending_mrr = self._calculate_mrr_from_db(period_end)

        # From existing customers only
        nrr = (ending_mrr / max(starting_mrr, 1)) * 100

        return MetricSnapshot(
            name="Net Revenue Retention",
            value=nrr,
            timestamp=period_end,
        )

    def get_arpu(self, as_of: Optional[datetime] = None) -> MetricSnapshot:
        """
        Calculate Average Revenue Per User.

        ARPU = MRR / Active Paying Customers
        """
        as_of = as_of or datetime.utcnow()
        mrr = self.get_mrr(as_of).value
        paying_customers = self._count_paying_customers(as_of)

        arpu = mrr / max(paying_customers, 1)

        return MetricSnapshot(
            name="ARPU",
            value=arpu,
            timestamp=as_of,
        )

    def get_ltv(self, as_of: Optional[datetime] = None) -> MetricSnapshot:
        """
        Calculate Customer Lifetime Value.

        LTV = ARPU / Monthly Churn Rate
        """
        as_of = as_of or datetime.utcnow()
        arpu = self.get_arpu(as_of).value

        # Get trailing 3-month churn rate
        period_start = as_of - timedelta(days=90)
        monthly_churn = self.get_churn_rate(period_start, as_of).value / 3

        ltv = arpu / max(monthly_churn / 100, 0.01)  # Avoid division by zero

        return MetricSnapshot(
            name="Customer LTV",
            value=ltv,
            timestamp=as_of,
        )

    def get_dashboard_metrics(self) -> Dict[str, MetricSnapshot]:
        """Get all key metrics for founder dashboard."""
        now = datetime.utcnow()
        month_ago = now - timedelta(days=30)

        return {
            "mrr": self.get_mrr(),
            "arr": self.get_arr(),
            "churn_rate": self.get_churn_rate(month_ago, now),
            "nrr": self.get_net_revenue_retention(month_ago, now),
            "arpu": self.get_arpu(),
            "ltv": self.get_ltv(),
        }

    # Private methods for DB queries (implement based on your schema)

    def _calculate_mrr_from_db(self, as_of: datetime) -> float:
        """Query MRR from database."""
        # TODO: Implement actual query
        # SELECT SUM(monthly_value) FROM subscriptions WHERE status = 'active'
        return 0.0

    def _count_customers_at(self, date: datetime) -> int:
        """Count customers at a point in time."""
        # TODO: Implement actual query
        return 0

    def _count_paying_customers(self, date: datetime) -> int:
        """Count paying customers."""
        # TODO: Implement actual query
        return 0

    def _count_churned_customers(self, start: datetime, end: datetime) -> int:
        """Count customers who churned in period."""
        # TODO: Implement actual query
        return 0


class UserMetrics:
    """
    User behavior and engagement metrics.

    Key metrics:
    - DAU/WAU/MAU (Daily/Weekly/Monthly Active Users)
    - Retention rates
    - Feature adoption
    - Session metrics
    """

    def __init__(self, db_session=None):
        self.db = db_session

    def get_active_users(
        self,
        period: str = "day",
        as_of: Optional[datetime] = None,
    ) -> MetricSnapshot:
        """
        Get active users for a period.

        Args:
            period: 'day', 'week', or 'month'
            as_of: Point in time (default: now)
        """
        as_of = as_of or datetime.utcnow()

        period_days = {"day": 1, "week": 7, "month": 30}
        days = period_days.get(period, 1)

        start = as_of - timedelta(days=days)
        count = self._count_active_users(start, as_of)

        name_map = {"day": "DAU", "week": "WAU", "month": "MAU"}

        return MetricSnapshot(
            name=name_map.get(period, "Active Users"),
            value=count,
            timestamp=as_of,
        )

    def get_retention_rate(
        self,
        cohort_date: datetime,
        days_after: int,
    ) -> MetricSnapshot:
        """
        Calculate retention rate for a cohort.

        Args:
            cohort_date: When users signed up
            days_after: How many days after signup to check
        """
        cohort_size = self._get_cohort_size(cohort_date)
        retained = self._get_retained_users(cohort_date, days_after)

        rate = (retained / max(cohort_size, 1)) * 100

        return MetricSnapshot(
            name=f"D{days_after} Retention",
            value=rate,
            timestamp=cohort_date + timedelta(days=days_after),
        )

    def get_feature_adoption(
        self,
        feature_name: str,
        period_days: int = 30,
    ) -> MetricSnapshot:
        """
        Calculate feature adoption rate.

        Adoption = Users who used feature / Total active users
        """
        now = datetime.utcnow()
        start = now - timedelta(days=period_days)

        total_active = self._count_active_users(start, now)
        feature_users = self._count_feature_users(feature_name, start, now)

        adoption = (feature_users / max(total_active, 1)) * 100

        return MetricSnapshot(
            name=f"{feature_name} Adoption",
            value=adoption,
            timestamp=now,
            dimensions={"feature": feature_name},
        )

    def get_privacy_feature_usage(self) -> Dict[str, MetricSnapshot]:
        """Get adoption of privacy features (key differentiator)."""
        return {
            "dp_sgd": self.get_feature_adoption("dp_training"),
            "he_lora": self.get_feature_adoption("he_inference"),
            "tgsp": self.get_feature_adoption("tgsp_packaging"),
            "pqc": self.get_feature_adoption("pqc_encryption"),
        }

    def get_user_journey_metrics(self) -> Dict[str, Any]:
        """Get metrics across the user journey."""
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)

        return {
            "signups_7d": self._count_signups(week_ago, now),
            "activated_7d": self._count_activated(week_ago, now),  # First API call
            "converted_7d": self._count_converted(week_ago, now),  # Free -> Paid
            "activation_rate": self._calculate_activation_rate(week_ago, now),
            "conversion_rate": self._calculate_conversion_rate(week_ago, now),
        }

    # Private methods

    def _count_active_users(self, start: datetime, end: datetime) -> int:
        """Count users with activity in period."""
        # TODO: Query from events table
        return 0

    def _get_cohort_size(self, date: datetime) -> int:
        """Get number of users who signed up on date."""
        return 0

    def _get_retained_users(self, cohort_date: datetime, days: int) -> int:
        """Get retained users from cohort."""
        return 0

    def _count_feature_users(self, feature: str, start: datetime, end: datetime) -> int:
        """Count users who used a feature."""
        return 0

    def _count_signups(self, start: datetime, end: datetime) -> int:
        return 0

    def _count_activated(self, start: datetime, end: datetime) -> int:
        return 0

    def _count_converted(self, start: datetime, end: datetime) -> int:
        return 0

    def _calculate_activation_rate(self, start: datetime, end: datetime) -> float:
        signups = self._count_signups(start, end)
        activated = self._count_activated(start, end)
        return (activated / max(signups, 1)) * 100

    def _calculate_conversion_rate(self, start: datetime, end: datetime) -> float:
        activated = self._count_activated(start, end)
        converted = self._count_converted(start, end)
        return (converted / max(activated, 1)) * 100


class OperationalMetrics:
    """
    System operational metrics for reliability and performance.
    """

    def __init__(self, prometheus_url: Optional[str] = None):
        self.prometheus_url = prometheus_url

    def get_api_metrics(self) -> Dict[str, MetricSnapshot]:
        """Get API health metrics."""
        now = datetime.utcnow()

        return {
            "requests_per_minute": MetricSnapshot("API RPM", self._get_rpm(), now),
            "error_rate": MetricSnapshot("Error Rate", self._get_error_rate(), now),
            "p50_latency": MetricSnapshot("P50 Latency (ms)", self._get_latency(50), now),
            "p99_latency": MetricSnapshot("P99 Latency (ms)", self._get_latency(99), now),
        }

    def get_infrastructure_metrics(self) -> Dict[str, MetricSnapshot]:
        """Get infrastructure metrics."""
        now = datetime.utcnow()

        return {
            "cpu_utilization": MetricSnapshot("CPU %", self._get_cpu_util(), now),
            "memory_utilization": MetricSnapshot("Memory %", self._get_memory_util(), now),
            "gpu_utilization": MetricSnapshot("GPU %", self._get_gpu_util(), now),
            "disk_utilization": MetricSnapshot("Disk %", self._get_disk_util(), now),
        }

    def get_uptime(self, days: int = 30) -> MetricSnapshot:
        """Calculate uptime percentage."""
        # Query from incidents/downtime tracking
        uptime = 99.95  # TODO: Calculate from actual data
        return MetricSnapshot(
            name=f"{days}d Uptime",
            value=uptime,
            timestamp=datetime.utcnow(),
        )

    # Private methods - implement based on monitoring stack

    def _get_rpm(self) -> float:
        return 0.0

    def _get_error_rate(self) -> float:
        return 0.0

    def _get_latency(self, percentile: int) -> float:
        return 0.0

    def _get_cpu_util(self) -> float:
        return 0.0

    def _get_memory_util(self) -> float:
        return 0.0

    def _get_gpu_util(self) -> float:
        return 0.0

    def _get_disk_util(self) -> float:
        return 0.0
