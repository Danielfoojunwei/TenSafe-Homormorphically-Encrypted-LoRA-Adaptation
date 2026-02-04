"""
Federated Learning Example

Demonstrates privacy-preserving federated learning across multiple clients.

Requirements:
- TenSafe account and API key
- Multiple data sources/clients

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python federated_learning.py
"""

import os
from tensafe import TenSafeClient
from tensafe.federated import FederatedConfig, SecureAggregator


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Configure federated learning
    fed_config = FederatedConfig(
        num_clients=5,
        rounds=10,
        local_epochs=3,
        aggregation="fedavg",  # Federated Averaging
        secure_aggregation=True,  # Enable secure aggregation
        dp_per_client=True,  # Local DP at each client
    )

    aggregator = SecureAggregator(
        client=client,
        config=fed_config,
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
    )

    print(f"Federated Learning Setup:")
    print(f"  Clients: {fed_config.num_clients}")
    print(f"  Rounds: {fed_config.rounds}")
    print(f"  Local epochs: {fed_config.local_epochs}")
    print(f"  Secure aggregation: {fed_config.secure_aggregation}")
    print()

    # Simulate federated training
    for round_num in range(fed_config.rounds):
        print(f"=== Round {round_num + 1}/{fed_config.rounds} ===")

        # Each client trains locally
        client_updates = []
        for client_id in range(fed_config.num_clients):
            # Simulate local training
            local_update = aggregator.train_local(
                client_id=client_id,
                epochs=fed_config.local_epochs,
            )
            client_updates.append(local_update)
            print(f"  Client {client_id}: local loss={local_update.loss:.4f}")

        # Secure aggregation
        global_update = aggregator.aggregate(client_updates)
        print(f"  Global model updated (avg loss: {global_update.avg_loss:.4f})")

    # Get final model
    final_model = aggregator.get_global_model()
    print(f"\nFederated learning complete!")
    print(f"Final global model: {final_model.id}")


if __name__ == "__main__":
    main()
