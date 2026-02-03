"""
Webhook Event Handling Example

Demonstrates receiving and processing TenSafe webhook events
for training status updates, checkpoint notifications, etc.

Requirements:
- TenSafe account and API key
- Web server to receive webhooks
- pip install flask

Usage:
    export TENSAFE_API_KEY="your-api-key"
    export WEBHOOK_SECRET="your-webhook-secret"
    python webhook_events.py
"""

import os
import hmac
import hashlib
from flask import Flask, request, jsonify
from tensafe import TenSafeClient

app = Flask(__name__)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify webhook signature using HMAC-SHA256."""
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.route("/webhooks/tensafe", methods=["POST"])
def handle_webhook():
    """Handle incoming TenSafe webhooks."""
    # Verify signature
    signature = request.headers.get("X-TenSafe-Signature", "")
    if not verify_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401

    event = request.json
    event_type = event.get("type")
    data = event.get("data", {})

    print(f"Received event: {event_type}")

    # Handle different event types
    if event_type == "training.started":
        handle_training_started(data)
    elif event_type == "training.completed":
        handle_training_completed(data)
    elif event_type == "training.failed":
        handle_training_failed(data)
    elif event_type == "checkpoint.saved":
        handle_checkpoint_saved(data)
    elif event_type == "quota.warning":
        handle_quota_warning(data)
    elif event_type == "quota.exceeded":
        handle_quota_exceeded(data)
    else:
        print(f"Unknown event type: {event_type}")

    return jsonify({"received": True})


def handle_training_started(data):
    """Handle training.started event."""
    print(f"Training started: {data['training_client_id']}")
    print(f"  Model: {data['model_ref']}")
    print(f"  DP enabled: {data.get('dp_enabled', False)}")


def handle_training_completed(data):
    """Handle training.completed event."""
    print(f"Training completed: {data['training_client_id']}")
    print(f"  Steps: {data['total_steps']}")
    print(f"  Final epsilon: {data.get('final_epsilon', 'N/A')}")

    # Optionally download checkpoint
    # client = TenSafeClient()
    # client.download_artifact(data['checkpoint_id'], "/tmp/model.pt")


def handle_training_failed(data):
    """Handle training.failed event."""
    print(f"Training FAILED: {data['training_client_id']}")
    print(f"  Error: {data.get('error_message', 'Unknown')}")
    print(f"  Step: {data.get('failed_at_step', 'Unknown')}")

    # Send alert
    # send_slack_alert(f"Training failed: {data['error_message']}")


def handle_checkpoint_saved(data):
    """Handle checkpoint.saved event."""
    print(f"Checkpoint saved: {data['artifact_id']}")
    print(f"  Size: {data['size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Step: {data['step']}")
    print(f"  DP metrics: {data.get('dp_metrics', {})}")


def handle_quota_warning(data):
    """Handle quota.warning event."""
    print(f"QUOTA WARNING: {data['resource']}")
    print(f"  Usage: {data['current_usage']} / {data['limit']}")
    print(f"  Percentage: {data['percentage']}%")

    # Send notification
    # send_email_alert(f"Approaching quota limit for {data['resource']}")


def handle_quota_exceeded(data):
    """Handle quota.exceeded event."""
    print(f"QUOTA EXCEEDED: {data['resource']}")
    print(f"  Usage: {data['current_usage']} / {data['limit']}")

    # Critical alert
    # send_pagerduty_alert(f"Quota exceeded for {data['resource']}")


def setup_webhooks():
    """Register webhooks with TenSafe."""
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Create webhook
    webhook = client.webhooks.create(
        url="https://your-server.com/webhooks/tensafe",
        events=[
            "training.started",
            "training.completed",
            "training.failed",
            "checkpoint.saved",
            "quota.warning",
            "quota.exceeded",
        ],
        secret=WEBHOOK_SECRET,
    )
    print(f"Webhook created: {webhook.id}")

    # Test webhook
    client.webhooks.test(webhook.id)
    print("Test event sent!")


if __name__ == "__main__":
    # First, set up webhooks
    # setup_webhooks()

    # Then run the webhook receiver
    print("Starting webhook receiver on port 5000...")
    app.run(host="0.0.0.0", port=5000)
