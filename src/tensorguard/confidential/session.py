"""
Confidential Session Manager.

Manages HPKE-based encrypted sessions between clients and the TEE-protected
inference server. Each session has an ephemeral key pair bound to the TEE
attestation quote.

Protocol:
    1. Client requests attestation: GET /v1/confidential/attestation
    2. Server returns TEE quote + ephemeral public key
    3. Client verifies quote, encrypts prompt with HPKE
    4. Server decrypts inside TEE, runs inference, encrypts response
    5. Client decrypts response with session key
"""

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.asymmetric import x25519

from ..tgsp.hpke_v03 import (
    generate_keypair,
    hpke_open,
    hpke_seal,
    public_key_to_bytes,
)

logger = logging.getLogger(__name__)

# Session defaults
DEFAULT_SESSION_TTL_SECONDS = 3600  # 1 hour
MAX_SESSION_TTL_SECONDS = 86400  # 24 hours
SESSION_CLEANUP_INTERVAL = 300  # 5 minutes


@dataclass
class ConfidentialSession:
    """
    A single confidential inference session.

    Each session has an ephemeral X25519 key pair used for HPKE
    encryption/decryption of prompts and outputs.
    """

    session_id: str
    created_at: datetime
    expires_at: datetime

    # Server-side ephemeral key pair (private key stays in TEE)
    server_private_key: x25519.X25519PrivateKey
    server_public_key: x25519.X25519PublicKey

    # Client's public key (set after first encrypted request)
    client_public_key: Optional[x25519.X25519PublicKey] = None

    # Attestation binding
    attestation_quote_hash: Optional[str] = None

    # Session metrics
    requests_processed: int = 0
    bytes_decrypted: int = 0
    bytes_encrypted: int = 0
    last_activity: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_active(self) -> bool:
        """Check if session is active and not expired."""
        return not self.is_expired

    @property
    def server_public_key_bytes(self) -> bytes:
        """Get server public key as raw bytes."""
        return public_key_to_bytes(self.server_public_key)

    @property
    def server_public_key_hex(self) -> str:
        """Get server public key as hex string."""
        return self.server_public_key_bytes.hex()

    def decrypt_request(self, sealed_data: Dict[str, str], aad: bytes = b"") -> bytes:
        """
        Decrypt an HPKE-sealed request from the client.

        Args:
            sealed_data: HPKE sealed data dict with 'enc' and 'ciphertext'
            aad: Additional authenticated data

        Returns:
            Decrypted plaintext bytes
        """
        plaintext = hpke_open(
            sealed=sealed_data,
            recipient_private_key=self.server_private_key,
            info=self.session_id.encode(),
            aad=aad,
        )

        self.requests_processed += 1
        self.bytes_decrypted += len(plaintext)
        self.last_activity = datetime.utcnow()

        return plaintext

    def encrypt_response(
        self,
        plaintext: bytes,
        recipient_public_key: x25519.X25519PublicKey,
        aad: bytes = b"",
    ) -> Dict[str, str]:
        """
        Encrypt a response for the client using HPKE.

        Args:
            plaintext: Response data to encrypt
            recipient_public_key: Client's X25519 public key
            aad: Additional authenticated data

        Returns:
            HPKE sealed data dict
        """
        sealed = hpke_seal(
            plaintext=plaintext,
            recipient_public_key=recipient_public_key,
            info=self.session_id.encode(),
            aad=aad,
        )

        self.bytes_encrypted += len(plaintext)
        self.last_activity = datetime.utcnow()

        return sealed

    def get_metadata(self) -> Dict[str, Any]:
        """Get session metadata (safe to expose)."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "requests_processed": self.requests_processed,
            "bytes_decrypted": self.bytes_decrypted,
            "bytes_encrypted": self.bytes_encrypted,
            "attestation_bound": self.attestation_quote_hash is not None,
        }


class ConfidentialSessionManager:
    """
    Manages confidential inference sessions.

    Handles session lifecycle: creation, lookup, expiration, and cleanup.
    Thread-safe for concurrent access from API handlers.

    Usage:
        manager = ConfidentialSessionManager()
        session = manager.create_session(attestation_quote_hash="abc123")
        # ... client sends encrypted request ...
        plaintext = session.decrypt_request(sealed_data)
        # ... run inference ...
        sealed_response = session.encrypt_response(response_bytes, client_pub)
    """

    def __init__(
        self,
        default_ttl: int = DEFAULT_SESSION_TTL_SECONDS,
        max_sessions: int = 10000,
        cleanup_interval: int = SESSION_CLEANUP_INTERVAL,
    ):
        """
        Initialize session manager.

        Args:
            default_ttl: Default session TTL in seconds
            max_sessions: Maximum concurrent sessions
            cleanup_interval: Interval for expired session cleanup
        """
        self._default_ttl = min(default_ttl, MAX_SESSION_TTL_SECONDS)
        self._max_sessions = max_sessions
        self._cleanup_interval = cleanup_interval

        self._sessions: Dict[str, ConfidentialSession] = {}
        self._lock = threading.Lock()

        # Metrics
        self._total_sessions_created = 0
        self._total_sessions_expired = 0

    def create_session(
        self,
        ttl: Optional[int] = None,
        attestation_quote_hash: Optional[str] = None,
    ) -> ConfidentialSession:
        """
        Create a new confidential session.

        Generates an ephemeral X25519 key pair for HPKE operations.

        Args:
            ttl: Session TTL in seconds (uses default if None)
            attestation_quote_hash: Hash of the attestation quote binding

        Returns:
            New ConfidentialSession
        """
        with self._lock:
            # Enforce max sessions
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_expired()
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError(
                        f"Maximum sessions ({self._max_sessions}) reached"
                    )

            # Generate session ID
            session_id = f"cs-{hashlib.sha256(os.urandom(32)).hexdigest()[:24]}"

            # Generate ephemeral key pair
            private_key, public_key = generate_keypair()

            effective_ttl = min(ttl or self._default_ttl, MAX_SESSION_TTL_SECONDS)
            now = datetime.utcnow()

            session = ConfidentialSession(
                session_id=session_id,
                created_at=now,
                expires_at=now + timedelta(seconds=effective_ttl),
                server_private_key=private_key,
                server_public_key=public_key,
                attestation_quote_hash=attestation_quote_hash,
            )

            self._sessions[session_id] = session
            self._total_sessions_created += 1

            logger.info(
                f"Created confidential session {session_id} "
                f"(TTL={effective_ttl}s, attestation_bound={attestation_quote_hash is not None})"
            )

            return session

    def get_session(self, session_id: str) -> Optional[ConfidentialSession]:
        """
        Get an active session by ID.

        Returns None if session doesn't exist or is expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                self._total_sessions_expired += 1
                return None
            return session

    def destroy_session(self, session_id: str) -> bool:
        """
        Explicitly destroy a session.

        Returns True if session was found and destroyed.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Destroyed confidential session {session_id}")
                return True
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        with self._lock:
            active_count = sum(
                1 for s in self._sessions.values() if s.is_active
            )
            return {
                "active_sessions": active_count,
                "total_sessions": len(self._sessions),
                "total_created": self._total_sessions_created,
                "total_expired": self._total_sessions_expired,
                "max_sessions": self._max_sessions,
                "default_ttl": self._default_ttl,
            }

    def cleanup(self) -> int:
        """Run cleanup of expired sessions. Returns count of removed sessions."""
        with self._lock:
            return self._cleanup_expired()

    def _cleanup_expired(self) -> int:
        """Remove expired sessions (must hold lock)."""
        expired = [
            sid for sid, session in self._sessions.items() if session.is_expired
        ]
        for sid in expired:
            del self._sessions[sid]
        self._total_sessions_expired += len(expired)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)
