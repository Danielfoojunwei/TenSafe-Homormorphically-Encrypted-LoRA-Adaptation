"""
Integration Adapters for ChatOps and Facade Systems.
Standardizes outgoing notifications for Slack, Jira, PagerDuty.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class IntegrationAdapter:
    def send_event(self, event_type: str, payload: Dict[str, Any]):
        raise NotImplementedError

class WebhookAdapter(IntegrationAdapter):
    def __init__(self, url: str, secret: Optional[str] = None):
        self.url = url
        self.secret = secret

    def send_event(self, event_type: str, payload: Dict[str, Any]):
        try:
            headers = {"Content-Type": "application/json"}
            if self.secret:
                headers["X-Hub-Signature"] = self.secret # Simplified
            
            requests.post(self.url, json={"type": event_type, "data": payload}, headers=headers, timeout=5)
            logger.info(f"Webhook sent to {self.url}")
        except Exception as e:
            logger.error(f"Webhook failed: {e}")

class SlackAdapter(IntegrationAdapter):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_event(self, event_type: str, payload: Dict[str, Any]):
        """Formats event as a Slack Block Kit message."""
        color = "#2eb886" if "success" in event_type.lower() else "#a30200"
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"TensorGuard: {event_type.upper()}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Run ID:*\n{payload.get('run_id', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Robot:*\n{payload.get('robot_id', 'N/A')}"}
                ]
            }
        ]
        
        # Add context/actions if needed
        
        try:
            requests.post(self.webhook_url, json={"blocks": blocks}, timeout=5)
        except Exception as e:
            logger.error(f"Slack send failed: {e}")

class PagerDutyAdapter(IntegrationAdapter):
    def __init__(self, routing_key: str):
        self.routing_key = routing_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    def send_event(self, event_type: str, payload: Dict[str, Any]):
        """Triggers PagerDuty incident for failures."""
        if "fail" not in event_type.lower() and "violation" not in event_type.lower():
             return

        pd_payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"TensorGuard Alert: {event_type}",
                "severity": "critical" if "violation" in event_type else "error",
                "source": payload.get("robot_id", "tensorguard"),
                "custom_details": payload
            }
        }
        try:
            requests.post(self.api_url, json=pd_payload, timeout=5)
            logger.info("PagerDuty alert sent")
        except Exception as e:
            logger.error(f"PagerDuty failed: {e}")
