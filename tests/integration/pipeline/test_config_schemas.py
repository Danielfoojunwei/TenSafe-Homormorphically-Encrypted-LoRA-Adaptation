import unittest
from pydantic import ValidationError
from tensorguard.integrations.framework.config_schema import (
    DataSourceConfig, TrainingExecutorConfig, ServingConfig, 
    TrustConfig, PrivacyConfig, IntegrationProfile, DataSourceType
)

class TestConfigSchemas(unittest.TestCase):

    def test_datasource_config_valid(self):
        cfg = DataSourceConfig(
            tenant_id="acme", 
            name="s3_input", 
            type=DataSourceType.S3, 
            uri="s3://my-bucket/data"
        )
        self.assertEqual(cfg.type, "s3")

    def test_datasource_config_invalid(self):
        with self.assertRaises(ValidationError):
            # Missing uri
            DataSourceConfig(tenant_id="acme", name="bad", type="local")

    def test_training_executor_config(self):
        cfg = TrainingExecutorConfig(
            tenant_id="acme",
            name="k8s_spot",
            type="k8s",
            cluster_ref="production-cluster",
            env_vars={"NAMESPACE": "ml-jobs"}
        )
        self.assertEqual(cfg.env_vars["NAMESPACE"], "ml-jobs")

    def test_integration_profile_aggregation(self):
        profile = IntegrationProfile(
            tenant_id="acme",
            data_sources=[
                DataSourceConfig(tenant_id="acme", name="f1", type="local", uri="/tmp")
            ],
            privacy_settings=PrivacyConfig(tenant_id="acme", name="hidden", type="n2he", n2he_profile="full")
        )
        self.assertEqual(len(profile.data_sources), 1)
        self.assertEqual(profile.privacy_settings.n2he_profile, "full")

if __name__ == "__main__":
    unittest.main()
