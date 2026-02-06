"""
Confidential Inference Middleware.

FastAPI middleware that handles the encrypted request/response envelope
for confidential inference. Sits between the client and the vLLM API
endpoints, transparently decrypting prompts and encrypting outputs
inside the TEE boundary.

Envelope format (request):
    {
        "session_id": "cs-...",
        "encrypted_payload": {"enc": "<hex>", "ciphertext": "<hex>"},
        "client_public_key": "<hex>",  // for response encryption
        "aad": "<hex>"                 // optional additional authenticated data
    }

Envelope format (response):
    {
        "session_id": "cs-...",
        "encrypted_response": {"enc": "<hex>", "ciphertext": "<hex>"},
        "privacy_receipt": { ... },
        "aad": "<hex>"
    }
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.asymmetric import x25519

from .session import ConfidentialSession, ConfidentialSessionManager

logger = logging.getLogger(__name__)


class ConfidentialInferenceMiddleware:
    """
    Middleware for encrypted prompt/output handling.

    This is NOT a Starlette middleware (which would buffer entire
    request/response bodies). Instead, it's a service layer called
    by the confidential API endpoints to unwrap/wrap the encrypted
    envelope.
    """

    def __init__(
        self,
        session_manager: ConfidentialSessionManager,
    ):
        self._session_manager = session_manager

    def unwrap_request(
        self, envelope: Dict[str, Any]
    ) -> tuple[Dict[str, Any], ConfidentialSession, Optional[bytes]]:
        """
        Unwrap an encrypted request envelope.

        Decrypts the payload inside the TEE and returns the plaintext
        request body as a parsed dict.

        Args:
            envelope: Encrypted request envelope

        Returns:
            Tuple of (plaintext_request_dict, session, client_public_key_bytes)

        Raises:
            ValueError: If envelope is malformed or decryption fails
            RuntimeError: If session is invalid or expired
        """
        # Validate envelope structure
        session_id = envelope.get("session_id")
        if not session_id:
            raise ValueError("Missing session_id in envelope")

        encrypted_payload = envelope.get("encrypted_payload")
        if not encrypted_payload:
            raise ValueError("Missing encrypted_payload in envelope")

        if "enc" not in encrypted_payload or "ciphertext" not in encrypted_payload:
            raise ValueError(
                "encrypted_payload must contain 'enc' and 'ciphertext' fields"
            )

        # Get session
        session = self._session_manager.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found or expired: {session_id}")

        # Parse AAD
        aad = b""
        if envelope.get("aad"):
            aad = bytes.fromhex(envelope["aad"])

        # Decrypt inside TEE
        try:
            plaintext_bytes = session.decrypt_request(encrypted_payload, aad=aad)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

        # Parse plaintext as JSON
        try:
            plaintext_request = json.loads(plaintext_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Decrypted payload is not valid JSON: {e}")

        # Store client public key if provided
        client_pub_bytes = None
        if "client_public_key" in envelope:
            client_pub_hex = envelope["client_public_key"]
            client_pub_bytes = bytes.fromhex(client_pub_hex)
            session.client_public_key = x25519.X25519PublicKey.from_public_bytes(
                client_pub_bytes
            )

        logger.debug(
            f"Unwrapped request for session {session_id}: "
            f"{len(plaintext_bytes)} bytes decrypted"
        )

        return plaintext_request, session, client_pub_bytes

    def wrap_response(
        self,
        response_data: Dict[str, Any],
        session: ConfidentialSession,
        privacy_receipt: Optional[Dict[str, Any]] = None,
        aad: bytes = b"",
    ) -> Dict[str, Any]:
        """
        Wrap a response in an encrypted envelope.

        Encrypts the response inside the TEE before returning to the client.

        Args:
            response_data: Plaintext response dict to encrypt
            session: Confidential session for this request
            privacy_receipt: Optional privacy receipt to include
            aad: Additional authenticated data

        Returns:
            Encrypted response envelope

        Raises:
            RuntimeError: If client public key is not set
        """
        if session.client_public_key is None:
            raise RuntimeError(
                "Client public key not set. Include 'client_public_key' "
                "in the request envelope."
            )

        # Serialize response
        response_bytes = json.dumps(response_data).encode("utf-8")

        # Encrypt response for the client
        encrypted_response = session.encrypt_response(
            plaintext=response_bytes,
            recipient_public_key=session.client_public_key,
            aad=aad,
        )

        envelope = {
            "session_id": session.session_id,
            "encrypted_response": encrypted_response,
        }

        if aad:
            envelope["aad"] = aad.hex()

        if privacy_receipt is not None:
            envelope["privacy_receipt"] = privacy_receipt

        logger.debug(
            f"Wrapped response for session {session.session_id}: "
            f"{len(response_bytes)} bytes encrypted"
        )

        return envelope


def create_confidential_middleware(
    session_manager: Optional[ConfidentialSessionManager] = None,
) -> ConfidentialInferenceMiddleware:
    """Factory function to create confidential middleware."""
    if session_manager is None:
        session_manager = ConfidentialSessionManager()
    return ConfidentialInferenceMiddleware(session_manager=session_manager)
