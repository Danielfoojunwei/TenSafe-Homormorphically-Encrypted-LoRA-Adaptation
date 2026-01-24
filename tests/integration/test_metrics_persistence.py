import unittest
import asyncio
import os
import uuid
from datetime import datetime
from sqlmodel import Session, create_engine, SQLModel, select

from tensorguard.platform.database import engine as db_engine
from tensorguard.platform.services.metrics_collector import MetricsCollector
from tensorguard.metrics.schemas import MetricName, MetricUnit
from tensorguard.platform.models.metrics_models import RouteMetricSeries, AdapterMetricSnapshot
from tensorguard.platform.models.continuous_models import Route

class TestMetricsPersistence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Use development/demo DB for testing
        os.environ["TG_DEMO_MODE"] = "true"
        os.environ["TG_SIMULATION"] = "true"
        SQLModel.metadata.create_all(db_engine)
        self.session = Session(db_engine)
        self.collector = MetricsCollector(self.session)
        self.tenant_id = "test-tenant"
        self.route_key = f"test-route-{uuid.uuid4()}"
        
        # Create a test route
        route = Route(
            tenant_id=self.tenant_id,
            route_key=self.route_key,
            base_model_ref="test-model"
        )
        self.session.add(route)
        self.session.commit()

    async def test_record_route_series(self):
        """Test appending points to route time series."""
        self.collector.append_route_series(
            self.tenant_id, self.route_key,
            {MetricName.AVG_ACCURACY: 0.95, MetricName.FORGETTING_MEAN: 0.02},
            {MetricName.AVG_ACCURACY: "%", MetricName.FORGETTING_MEAN: "%"}
        )
        
        # Verify
        series = self.session.exec(
            select(RouteMetricSeries).where(RouteMetricSeries.route_key == self.route_key)
        ).all()
        self.assertEqual(len(series), 2)
        metrics = {s.metric_name: s.value for s in series}
        self.assertEqual(metrics["avg_accuracy"], 0.95)
        self.assertEqual(metrics["forgetting_mean"], 0.02)

    async def test_record_adapter_snapshot(self):
        """Test recording adapter snapshots."""
        adapter_id = str(uuid.uuid4())
        self.collector.write_adapter_snapshot(
            self.tenant_id, adapter_id, self.route_key,
            {MetricName.ACCURACY_FINAL: 0.94, MetricName.ADAPTER_STORAGE_MB: 12.5},
            {MetricName.ACCURACY_FINAL: "%", MetricName.ADAPTER_STORAGE_MB: "MB"}
        )
        
        # Verify
        snapshots = self.session.exec(
            select(AdapterMetricSnapshot).where(AdapterMetricSnapshot.adapter_id == adapter_id)
        ).all()
        self.assertEqual(len(snapshots), 2)
        metrics = {s.metric_name: s.value for s in snapshots}
        self.assertEqual(metrics["accuracy_final"], 0.94)
        self.assertEqual(metrics["adapter_storage_mb"], 12.5)

    async def asyncTearDown(self):
        self.session.close()

if __name__ == "__main__":
    unittest.main()
