"""
Kubernetes Deployment Example

Demonstrates deploying TenSafe models to Kubernetes.

Requirements:
- TenSafe account and API key
- kubectl configured
- Kubernetes cluster access

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python kubernetes_deploy.py
"""

import os
from tensafe import TenSafeClient
from tensafe.deployment import KubernetesDeployer


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Configure Kubernetes deployment
    deployer = KubernetesDeployer(
        namespace="tensafe",
        registry="your-registry.io/tensafe",
    )

    # Deploy model
    deployment = deployer.deploy(
        model_id="your-trained-model",
        replicas=3,
        resources={
            "requests": {"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
        },
        autoscaling={
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70,
            "target_latency_ms": 100,
        },
        env={
            "TENSAFE_API_KEY": os.environ.get("TENSAFE_API_KEY"),
            "DP_EPSILON_LIMIT": "8.0",
        },
    )

    print(f"Deployment created: {deployment.name}")
    print(f"  Namespace: {deployment.namespace}")
    print(f"  Replicas: {deployment.replicas}")
    print(f"  Endpoint: {deployment.endpoint}")

    # Check deployment status
    status = deployer.get_status(deployment.name)
    print(f"\nStatus: {status.phase}")
    print(f"  Ready replicas: {status.ready_replicas}/{status.replicas}")


if __name__ == "__main__":
    main()
