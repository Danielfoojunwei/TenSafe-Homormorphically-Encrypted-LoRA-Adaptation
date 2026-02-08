"""
TenSafe Webhook Signature Utilities.

Provides HMAC-SHA256 signature generation and verification for webhook payloads.
Follows industry standards (similar to Stripe, GitHub webhooks).

Security Features:
- HMAC-SHA256 for payload authentication
- Timestamp validation to prevent replay attacks
- Constant-time comparison to prevent timing attacks
- Multiple signature algorithm support for future-proofing
"""

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# Signature validity window (5 minutes) - prevents replay attacks
DEFAULT_TIMESTAMP_TOLERANCE_SECONDS = 300

# Signature header format version
SIGNATURE_VERSION = "v1"


class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""

    HMAC_SHA256 = "sha256"
    HMAC_SHA384 = "sha384"
    HMAC_SHA512 = "sha512"


@dataclass
class SignatureComponents:
    """Components of a webhook signature."""

    timestamp: int
    signature: str
    algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256
    version: str = SIGNATURE_VERSION


class WebhookSignatureError(Exception):
    """Base exception for signature-related errors."""

    pass


class InvalidSignatureError(WebhookSignatureError):
    """Raised when signature verification fails."""

    pass


class ExpiredSignatureError(WebhookSignatureError):
    """Raised when timestamp is outside acceptable window."""

    pass


class MalformedSignatureError(WebhookSignatureError):
    """Raised when signature format is invalid."""

    pass


def generate_secret(length: int = 32) -> str:
    """
    Generate a cryptographically secure webhook secret.

    Args:
        length: Length of the secret in bytes (default 32 = 256 bits)

    Returns:
        Hex-encoded secret string
    """
    return secrets.token_hex(length)


def compute_signature(
    payload: bytes,
    secret: str,
    timestamp: int,
    algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
) -> str:
    """
    Compute HMAC signature for webhook payload.

    The signed message format is: "{timestamp}.{payload}"
    This binds the timestamp to the payload to prevent replay attacks.

    Args:
        payload: Raw payload bytes
        secret: Webhook secret (hex-encoded or raw string)
        timestamp: Unix timestamp of the request
        algorithm: Hash algorithm to use

    Returns:
        Hex-encoded signature string
    """
    # Convert secret to bytes
    if all(c in "0123456789abcdefABCDEF" for c in secret) and len(secret) % 2 == 0:
        # Hex-encoded secret
        secret_bytes = bytes.fromhex(secret)
    else:
        # Raw string secret
        secret_bytes = secret.encode("utf-8")

    # Create signed payload: timestamp.payload
    signed_payload = f"{timestamp}.".encode() + payload

    # Select hash function
    hash_funcs = {
        SignatureAlgorithm.HMAC_SHA256: hashlib.sha256,
        SignatureAlgorithm.HMAC_SHA384: hashlib.sha384,
        SignatureAlgorithm.HMAC_SHA512: hashlib.sha512,
    }
    hash_func = hash_funcs[algorithm]

    # Compute HMAC
    signature = hmac.new(secret_bytes, signed_payload, hash_func).hexdigest()

    return signature


def generate_signature_header(
    payload: bytes,
    secret: str,
    timestamp: Optional[int] = None,
    algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
) -> Tuple[str, int]:
    """
    Generate the complete signature header for a webhook request.

    Header format: "t={timestamp},v1={signature}"
    Multiple signatures can be included for key rotation: "t={timestamp},v1={sig1},v1={sig2}"

    Args:
        payload: Raw payload bytes
        secret: Webhook secret
        timestamp: Unix timestamp (default: current time)
        algorithm: Hash algorithm to use

    Returns:
        Tuple of (signature_header, timestamp)
    """
    if timestamp is None:
        timestamp = int(time.time())

    signature = compute_signature(payload, secret, timestamp, algorithm)

    # Format: t=timestamp,v1=signature
    # Using v1 prefix for versioning (similar to Stripe)
    header = f"t={timestamp},{SIGNATURE_VERSION}={signature}"

    return header, timestamp


def parse_signature_header(header: str) -> SignatureComponents:
    """
    Parse a webhook signature header.

    Args:
        header: Signature header string

    Returns:
        SignatureComponents with parsed values

    Raises:
        MalformedSignatureError: If header format is invalid
    """
    if not header:
        raise MalformedSignatureError("Empty signature header")

    parts = {}
    for item in header.split(","):
        if "=" not in item:
            raise MalformedSignatureError(f"Invalid signature component: {item}")
        key, value = item.split("=", 1)
        parts[key.strip()] = value.strip()

    # Extract timestamp
    if "t" not in parts:
        raise MalformedSignatureError("Missing timestamp in signature header")

    try:
        timestamp = int(parts["t"])
    except ValueError:
        raise MalformedSignatureError(f"Invalid timestamp: {parts['t']}")

    # Extract signature (look for v1, v2, etc.)
    signature = None
    version = None
    for key in parts:
        if key.startswith("v") and key[1:].isdigit():
            signature = parts[key]
            version = key
            break

    if signature is None:
        raise MalformedSignatureError("Missing signature in header")

    return SignatureComponents(
        timestamp=timestamp,
        signature=signature,
        version=version or SIGNATURE_VERSION,
    )


def verify_signature(
    payload: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
) -> bool:
    """
    Verify webhook signature.

    Performs:
    1. Header parsing and validation
    2. Timestamp freshness check (replay prevention)
    3. Signature verification with constant-time comparison

    Args:
        payload: Raw payload bytes
        signature_header: Signature header from request
        secret: Webhook secret
        tolerance_seconds: Maximum age of timestamp (default: 5 minutes)
        algorithm: Expected hash algorithm

    Returns:
        True if signature is valid

    Raises:
        MalformedSignatureError: If header format is invalid
        ExpiredSignatureError: If timestamp is too old or in the future
        InvalidSignatureError: If signature doesn't match
    """
    # Parse header
    components = parse_signature_header(signature_header)

    # Validate timestamp
    current_time = int(time.time())
    time_diff = abs(current_time - components.timestamp)

    if time_diff > tolerance_seconds:
        raise ExpiredSignatureError(
            f"Timestamp is {time_diff} seconds from current time, "
            f"exceeds tolerance of {tolerance_seconds} seconds"
        )

    # Compute expected signature
    expected_signature = compute_signature(
        payload, secret, components.timestamp, algorithm
    )

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(expected_signature, components.signature):
        raise InvalidSignatureError("Signature verification failed")

    return True


def verify_signature_safe(
    payload: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
) -> Tuple[bool, Optional[str]]:
    """
    Safe signature verification that returns a tuple instead of raising.

    Args:
        payload: Raw payload bytes
        signature_header: Signature header from request
        secret: Webhook secret
        tolerance_seconds: Maximum age of timestamp

    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, error_description)
    """
    try:
        verify_signature(payload, signature_header, secret, tolerance_seconds)
        return True, None
    except MalformedSignatureError as e:
        return False, f"Malformed signature: {str(e)}"
    except ExpiredSignatureError as e:
        return False, f"Expired signature: {str(e)}"
    except InvalidSignatureError as e:
        return False, f"Invalid signature: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"


class WebhookSigner:
    """
    Webhook signer utility class.

    Provides convenient methods for signing and verifying webhook payloads.
    Maintains state for secret and algorithm configuration.
    """

    def __init__(
        self,
        secret: str,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
        tolerance_seconds: int = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    ):
        """
        Initialize webhook signer.

        Args:
            secret: Webhook secret
            algorithm: Hash algorithm to use
            tolerance_seconds: Timestamp tolerance for verification
        """
        self.secret = secret
        self.algorithm = algorithm
        self.tolerance_seconds = tolerance_seconds

    def sign(self, payload: Dict[str, Any], timestamp: Optional[int] = None) -> Dict[str, str]:
        """
        Sign a payload and return headers for the webhook request.

        Args:
            payload: Payload dictionary to sign
            timestamp: Optional timestamp (default: current time)

        Returns:
            Dictionary of headers to include in the request
        """
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

        signature_header, ts = generate_signature_header(
            payload_bytes, self.secret, timestamp, self.algorithm
        )

        return {
            "X-TenSafe-Signature": signature_header,
            "X-TenSafe-Timestamp": str(ts),
        }

    def verify(self, payload: bytes, signature_header: str) -> bool:
        """
        Verify a webhook signature.

        Args:
            payload: Raw payload bytes
            signature_header: X-TenSafe-Signature header value

        Returns:
            True if valid

        Raises:
            WebhookSignatureError: If verification fails
        """
        return verify_signature(
            payload, signature_header, self.secret, self.tolerance_seconds, self.algorithm
        )

    def verify_safe(self, payload: bytes, signature_header: str) -> Tuple[bool, Optional[str]]:
        """
        Safely verify a webhook signature.

        Args:
            payload: Raw payload bytes
            signature_header: X-TenSafe-Signature header value

        Returns:
            Tuple of (is_valid, error_message)
        """
        return verify_signature_safe(
            payload, signature_header, self.secret, self.tolerance_seconds
        )


def construct_event(
    payload: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
) -> Dict[str, Any]:
    """
    Construct and verify a webhook event from request data.

    This is the recommended method for processing incoming webhooks.
    Similar to Stripe's stripe.Webhook.construct_event().

    Args:
        payload: Raw request body bytes
        signature_header: X-TenSafe-Signature header value
        secret: Webhook secret
        tolerance_seconds: Timestamp tolerance

    Returns:
        Parsed and verified event payload

    Raises:
        WebhookSignatureError: If verification fails
        json.JSONDecodeError: If payload is not valid JSON
    """
    # Verify signature first
    verify_signature(payload, signature_header, secret, tolerance_seconds)

    # Parse and return payload
    return json.loads(payload)


# Header names for webhook requests
SIGNATURE_HEADER_NAME = "X-TenSafe-Signature"
TIMESTAMP_HEADER_NAME = "X-TenSafe-Timestamp"
EVENT_ID_HEADER_NAME = "X-TenSafe-Event-Id"
EVENT_TYPE_HEADER_NAME = "X-TenSafe-Event-Type"
WEBHOOK_ID_HEADER_NAME = "X-TenSafe-Webhook-Id"
DELIVERY_ID_HEADER_NAME = "X-TenSafe-Delivery-Id"
