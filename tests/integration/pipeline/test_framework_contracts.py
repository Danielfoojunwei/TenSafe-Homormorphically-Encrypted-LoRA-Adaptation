import unittest
import asyncio
from tensorguard.integrations.framework.contracts import IntegrationType, Connector, ValidationResult
from tensorguard.integrations.framework.config_schema import DataSourceConfig, TrainingExecutorConfig, ServingConfig
from tensorguard.integrations.connectors.local_filesystem import LocalFilesystemConnector
from tensorguard.integrations.connectors.local_hf_executor import LocalHFExecutor
from tensorguard.integrations.connectors.vllm_serving_exporter import VllmServingExporter
from tensorguard.integrations.connectors.n2he_privacy_connector import N2HEPrivacyConnector
from tensorguard.integrations.connectors.s3_feed import S3FeedConnector
from tensorguard.integrations.connectors.k8s_job_exporter import K8sJobExporter
from tensorguard.integrations.connectors.sagemaker_exporter import SageMakerExporter
from tensorguard.integrations.connectors.gcs_feed import GCSFeedConnector
from tensorguard.integrations.connectors.azure_blob import AzureBlobConnector
from tensorguard.integrations.connectors.hf_dataset import HFDatasetConnector
from tensorguard.integrations.connectors.vertex_ai_exporter import VertexAIExporter
from tensorguard.integrations.connectors.triton_serving_exporter import TritonServingExporter
from tensorguard.integrations.connectors.nitro_trust import NitroTrustConnector

class TestIntegrationContracts(unittest.TestCase):
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_local_filesystem_contract(self):
        config = DataSourceConfig(tenant_id="t1", name="local", type="local", uri="data/raw")
        connector = LocalFilesystemConnector(config)
        
        # 1. Interface check
        self.assertTrue(isinstance(connector, Connector))
        
        # 2. Capabilities
        caps = connector.describe_capabilities()
        self.assertEqual(caps["subtype"], "local_filesystem")
        
        # 3. Health check (mocked or static)
        health = self.loop.run_until_complete(connector.health_check())
        self.assertIn(health.status, ["ok", "error"])

    def test_local_hf_executor_contract(self):
        config = TrainingExecutorConfig(tenant_id="t1", name="trainer", type="local", cluster_ref="localhost")
        connector = LocalHFExecutor(config)
        
        self.assertTrue(isinstance(connector, Connector))
        caps = connector.describe_capabilities()
        self.assertTrue(caps["supports_qlora"])
        
        health = self.loop.run_until_complete(connector.health_check())
        self.assertEqual(health.status, "ok")

    def test_vllm_exporter_contract(self):
        config = ServingConfig(tenant_id="t1", name="vllm", type="vllm")
        connector = VllmServingExporter(config)
        
        self.assertTrue(isinstance(connector, Connector))
        pack = connector.export_serving_pack("adapter-123", {"base_model": "phi-2"})
        self.assertEqual(pack["vllm_config"]["adapters"][0]["name"], "adapter-123")

    def test_n2he_privacy_contract(self):
        from tensorguard.integrations.framework.config_schema import PrivacyConfig
        config = PrivacyConfig(tenant_id="t1", name="n2he", type="n2he", n2he_profile="router_only")
        connector = N2HEPrivacyConnector(config)
        
        self.assertTrue(isinstance(connector, Connector))
        health = self.loop.run_until_complete(connector.health_check())
        self.assertEqual(health.status, "ok")
        
        receipt = connector.generate_privacy_receipt("evt-1", {})
        self.assertTrue(receipt["receipt_id"].startswith("n2he_rcpt_"))

    def test_s3_feed_contract(self):
        config = DataSourceConfig(tenant_id="t1", name="s3", type="s3", uri="s3://bucket/data")
        connector = S3FeedConnector(config)
        self.assertTrue(isinstance(connector, Connector))
        self.assertEqual(connector.describe_capabilities()["subtype"], "s3")

    def test_k8s_exporter_contract(self):
        config = TrainingExecutorConfig(tenant_id="t1", name="k8s", type="k8s", cluster_ref="k8s-ctx")
        connector = K8sJobExporter(config)
        self.assertTrue(isinstance(connector, Connector))
        yaml_out = connector.export_job_spec("job-1", {"route_key": "r1"})
        self.assertIn("kind: Job", yaml_out)

    def test_sagemaker_exporter_contract(self):
        config = TrainingExecutorConfig(tenant_id="t1", name="sm", type="sagemaker", cluster_ref="arn:aws:iam::role")
        connector = SageMakerExporter(config)
        self.assertTrue(isinstance(connector, Connector))
        spec = connector.export_job_spec("sm-job", {"image": "trainer:latest"})
        self.assertEqual(spec["TrainingJobName"], "sm-job")

    def test_gcs_feed_contract(self):
        config = DataSourceConfig(tenant_id="t1", name="gcs", type="gcs", uri="gs://bucket/data")
        connector = GCSFeedConnector(config)
        self.assertTrue(isinstance(connector, Connector))
        self.assertEqual(connector.describe_capabilities()["subtype"], "gcs")

    def test_azure_feed_contract(self):
        config = DataSourceConfig(tenant_id="t1", name="az", type="azure", uri="https://st.blob.core.windows.net/cnt")
        connector = AzureBlobConnector(config)
        self.assertTrue(isinstance(connector, Connector))

    def test_hf_feed_contract(self):
        config = DataSourceConfig(tenant_id="t1", name="hf", type="hf", uri="org/dataset")
        connector = HFDatasetConnector(config)
        self.assertTrue(isinstance(connector, Connector))

    def test_vertex_exporter_contract(self):
        config = TrainingExecutorConfig(tenant_id="t1", name="vertex", type="vertex", cluster_ref="p/r")
        connector = VertexAIExporter(config)
        self.assertTrue(isinstance(connector, Connector))
        spec = connector.export_job_spec("v-job", {})
        self.assertEqual(spec["displayName"], "v-job")

    def test_triton_exporter_contract(self):
        config = ServingConfig(tenant_id="t1", name="triton", type="triton")
        connector = TritonServingExporter(config)
        self.assertTrue(isinstance(connector, Connector))
        pack = connector.export_serving_pack("adapter-1", {})
        self.assertIn('name: "adapter-1"', pack)

    def test_nitro_trust_contract(self):
        from tensorguard.integrations.framework.config_schema import TrustConfig
        config = TrustConfig(tenant_id="t1", name="nitro", type="nitro", public_key_id="k-1")
        connector = NitroTrustConnector(config)
        self.assertTrue(isinstance(connector, Connector))
        self.assertTrue(connector.verify_attestation("dG9rZW4="))

if __name__ == "__main__":
    unittest.main()
