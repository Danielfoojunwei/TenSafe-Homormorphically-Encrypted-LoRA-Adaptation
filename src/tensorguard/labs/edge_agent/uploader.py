"""
Edge Agent Uploader

Reliably uploads spooled telemetry messages to the Control Plane.
Implements HMAC authentication, exponential backoff, and batching.

Uses the production telemetry ingestion endpoint with proper HMAC signatures
as required by verify_fleet_auth.
"""

import time
import threading
import logging
import requests
import json
import hashlib
import hmac
import secrets
import os
from typing import Optional

from .spooler import Spooler

logger = logging.getLogger(__name__)


class Uploader(threading.Thread):
    """
    Batched uploader with HMAC authentication for secure telemetry ingestion.

    Uses verify_fleet_auth signature format:
    HMAC-SHA256(fleet_api_key, timestamp:nonce:body_hash)
    """

    def __init__(
        self,
        spooler: Spooler,
        target_url: str,
        api_key: str,
        fleet_id: str,
        device_id: Optional[str] = None,
        batch_size: int = 50,
        interval: float = 1.0
    ):
        super().__init__(daemon=True)
        self.spooler = spooler
        self.target_url = target_url
        self.api_key = api_key
        self.fleet_id = fleet_id
        self.device_id = device_id or os.environ.get("TG_DEVICE_ID", f"device-{secrets.token_hex(4)}")
        self.batch_size = batch_size
        self.interval = interval
        self.running = False

        # Device info for registration
        self.agent_version = os.environ.get("TG_AGENT_VERSION", "1.0.0")
        self.runtime_version = os.environ.get("TG_RUNTIME_VERSION")
        self.ros_distro = os.environ.get("ROS_DISTRO")
        self.firmware_version = os.environ.get("TG_FIRMWARE_VERSION")

    def _compute_hmac_signature(self, timestamp: str, nonce: str, body: bytes) -> str:
        """
        Compute HMAC-SHA256 signature for verify_fleet_auth.

        Signature format: HMAC-SHA256(api_key, timestamp:nonce:body_hash)
        """
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{timestamp}:{nonce}:{body_hash}"
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _build_headers(self, body: bytes) -> dict:
        """Build request headers with HMAC authentication."""
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signature = self._compute_hmac_signature(timestamp, nonce, body)

        return {
            "Content-Type": "application/json",
            "X-TG-Fleet-Id": self.fleet_id,
            "X-TG-Timestamp": timestamp,
            "X-TG-Nonce": nonce,
            "X-TG-Signature": signature,
        }

    def _transform_to_telemetry_format(self, batch: list) -> dict:
        """
        Transform spooled messages to telemetry ingestion format.

        Converts from spooler format to TelemetryBatch schema.
        """
        messages = []
        for msg in batch:
            # Map spooler message to TelemetryMessage format
            topic = msg.get("topic", "telemetry.stage")
            timestamp_ns = msg.get("timestamp_ns", int(time.time() * 1_000_000_000))
            payload = msg.get("payload", {})

            # Ensure device_id is in payload
            if "device_id" not in payload:
                payload["device_id"] = self.device_id

            messages.append({
                "topic": topic,
                "timestamp_ns": timestamp_ns,
                "payload": payload,
                "priority": msg.get("priority", 0),
            })

        return {
            "batch_id": f"batch-{int(time.time())}-{secrets.token_hex(4)}",
            "device_info": {
                "device_id": self.device_id,
                "agent_version": self.agent_version,
                "runtime_version": self.runtime_version,
                "ros_distro": self.ros_distro,
                "firmware_version": self.firmware_version,
            },
            "messages": messages,
        }

    def run(self):
        self.running = True
        logger.info(f"Uploader started. Target: {self.target_url}/ingest, Fleet: {self.fleet_id}")

        failures = 0

        while self.running:
            try:
                # 1. Peek batch from spooler
                batch = self.spooler.peek_batch(self.batch_size)
                if not batch:
                    time.sleep(self.interval)
                    continue

                # 2. Transform to telemetry format
                payload = self._transform_to_telemetry_format(batch)
                body = json.dumps(payload).encode()

                # 3. Build headers with HMAC auth
                headers = self._build_headers(body)

                # 4. Upload to telemetry ingest endpoint
                response = requests.post(
                    f"{self.target_url}/ingest",
                    data=body,
                    headers=headers,
                    timeout=10
                )

                if response.status_code in [200, 201, 202]:
                    # 5. Ack (Delete) processed messages
                    ids = [m['id'] for m in batch]
                    self.spooler.ack_batch(ids)

                    result = response.json()
                    accepted = result.get("accepted", len(ids))
                    rejected = result.get("rejected", 0)

                    if rejected > 0:
                        logger.warning(f"Uploaded batch: {accepted} accepted, {rejected} rejected")
                        for rej in result.get("rejections", []):
                            logger.debug(f"  Rejection: {rej}")
                    else:
                        logger.debug(f"Uploaded {accepted} messages")

                    failures = 0  # Reset backoff
                else:
                    logger.warning(f"Upload failed: {response.status_code} {response.text}")
                    failures += 1
                    self._backoff(failures)

            except requests.exceptions.Timeout:
                logger.warning("Upload timeout")
                failures += 1
                self._backoff(failures)

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                failures += 1
                self._backoff(failures)

            except Exception as e:
                logger.error(f"Upload error: {e}")
                failures += 1
                self._backoff(failures)

    def _backoff(self, failures: int):
        """Exponential backoff with max 60s delay."""
        sleep_time = min(self.interval * (2 ** (failures - 1)), 60)
        logger.info(f"Retrying in {sleep_time:.1f}s (failure #{failures})...")
        time.sleep(sleep_time)

    def stop(self):
        """Stop the uploader thread gracefully."""
        self.running = False
        if self.is_alive():
            self.join(timeout=5)
