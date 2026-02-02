"""
Request Signing and Replay Attack Prevention Module.

Provides request integrity verification and replay protection:
- HMAC-based request signing
- Timestamp validation
- Nonce-based replay prevention
- Request fingerprinting
"""

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class SignedRequest:
    """A signed request with verification data."""

    timestamp: float
    nonce: str
    signature: str
    key_id: Optional[str] = None


class NonceStore:
    """
    Store for tracking used nonces to prevent replay attacks.

    Supports both in-memory and Redis backends for distributed
    deployments.
    """

    DEFAULT_TTL = 300  # 5 minutes

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize nonce store.

        Args:
            ttl_seconds: How long to remember nonces
            redis_url: Redis URL for distributed deployments
        """
        self.ttl = ttl_seconds
        self.redis_url = redis_url or os.getenv("TG_REDIS_URL")
        self._redis_client = None
        self._local_store: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        if self.redis_url:
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Nonce store using Redis backend")
        except ImportError:
            logger.warning("Redis not available for nonce store")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None and self._redis_client is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up expired nonces."""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Nonce cleanup error: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired nonces from local store."""
        now = time.time()
        async with self._lock:
            expired = [k for k, v in self._local_store.items() if v < now]
            for key in expired:
                del self._local_store[key]

    async def is_used(self, nonce: str) -> bool:
        """
        Check if a nonce has been used.

        Args:
            nonce: Nonce to check

        Returns:
            True if nonce was already used
        """
        if self._redis_client:
            try:
                exists = await self._redis_client.exists(f"nonce:{nonce}")
                return bool(exists)
            except Exception as e:
                logger.warning(f"Redis error in nonce check: {e}")

        async with self._lock:
            if nonce in self._local_store:
                return self._local_store[nonce] > time.time()
            return False

    async def mark_used(self, nonce: str) -> bool:
        """
        Mark a nonce as used.

        Args:
            nonce: Nonce to mark

        Returns:
            True if marked successfully, False if already used
        """
        if await self.is_used(nonce):
            return False

        expires = time.time() + self.ttl

        if self._redis_client:
            try:
                # Use SET NX to atomically check and set
                result = await self._redis_client.set(
                    f"nonce:{nonce}",
                    "1",
                    ex=self.ttl,
                    nx=True,
                )
                return result is not None
            except Exception as e:
                logger.warning(f"Redis error in nonce mark: {e}")

        async with self._lock:
            if nonce in self._local_store:
                return False
            self._local_store[nonce] = expires
            return True


class RequestSigner:
    """
    Signs and verifies requests using HMAC.

    Provides request integrity verification to prevent tampering
    and ensure requests come from authorized sources.
    """

    ALGORITHM = "sha256"
    SIGNATURE_HEADER = "X-Request-Signature"
    TIMESTAMP_HEADER = "X-Request-Timestamp"
    NONCE_HEADER = "X-Request-Nonce"
    KEY_ID_HEADER = "X-Key-ID"

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        timestamp_tolerance: int = 300,
        nonce_store: Optional[NonceStore] = None,
    ):
        """
        Initialize request signer.

        Args:
            secret_key: HMAC secret key (32 bytes recommended)
            timestamp_tolerance: Maximum age of request in seconds
            nonce_store: Store for replay prevention
        """
        if secret_key is None:
            env_key = os.getenv("TG_REQUEST_SIGNING_KEY")
            if env_key:
                secret_key = bytes.fromhex(env_key)
            else:
                secret_key = secrets.token_bytes(32)
                logger.warning("Generated ephemeral request signing key")

        self.secret_key = secret_key
        self.timestamp_tolerance = timestamp_tolerance
        self.nonce_store = nonce_store or NonceStore()

        # Support for multiple keys (key rotation)
        self._keys: Dict[str, bytes] = {"default": secret_key}

    def add_key(self, key_id: str, key: bytes) -> None:
        """
        Add a signing key.

        Args:
            key_id: Key identifier
            key: HMAC key bytes
        """
        self._keys[key_id] = key

    def remove_key(self, key_id: str) -> bool:
        """
        Remove a signing key.

        Args:
            key_id: Key identifier

        Returns:
            True if key was removed
        """
        if key_id == "default":
            return False
        return self._keys.pop(key_id, None) is not None

    def _get_key(self, key_id: Optional[str] = None) -> bytes:
        """Get signing key by ID."""
        if key_id is None:
            return self.secret_key
        return self._keys.get(key_id, self.secret_key)

    def _compute_signature(
        self,
        method: str,
        path: str,
        timestamp: float,
        nonce: str,
        body: bytes = b"",
        key: Optional[bytes] = None,
    ) -> str:
        """
        Compute HMAC signature for request.

        Args:
            method: HTTP method
            path: Request path
            timestamp: Request timestamp
            nonce: Request nonce
            body: Request body
            key: HMAC key

        Returns:
            Hex-encoded signature
        """
        key = key or self.secret_key

        # Create canonical request string
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = f"{method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"

        # Compute HMAC
        signature = hmac.new(
            key,
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def sign_request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        key_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Sign a request.

        Args:
            method: HTTP method
            path: Request path
            body: Request body
            key_id: Optional key identifier

        Returns:
            Headers to add to request
        """
        timestamp = time.time()
        nonce = secrets.token_hex(16)
        key = self._get_key(key_id)

        signature = self._compute_signature(
            method, path, timestamp, nonce, body, key
        )

        headers = {
            self.TIMESTAMP_HEADER: str(timestamp),
            self.NONCE_HEADER: nonce,
            self.SIGNATURE_HEADER: signature,
        }

        if key_id:
            headers[self.KEY_ID_HEADER] = key_id

        return headers

    async def verify_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes = b"",
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a signed request.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body

        Returns:
            Tuple of (valid, error_message)
        """
        # Extract signature components
        timestamp_str = headers.get(self.TIMESTAMP_HEADER)
        nonce = headers.get(self.NONCE_HEADER)
        signature = headers.get(self.SIGNATURE_HEADER)
        key_id = headers.get(self.KEY_ID_HEADER)

        if not all([timestamp_str, nonce, signature]):
            return False, "Missing signature headers"

        # Validate timestamp
        try:
            timestamp = float(timestamp_str)
        except ValueError:
            return False, "Invalid timestamp"

        now = time.time()
        age = abs(now - timestamp)

        if age > self.timestamp_tolerance:
            return False, f"Request timestamp too old ({age:.0f}s)"

        # Check nonce
        if not await self.nonce_store.mark_used(nonce):
            return False, "Nonce already used (possible replay attack)"

        # Verify signature
        key = self._get_key(key_id)
        expected_signature = self._compute_signature(
            method, path, timestamp, nonce, body, key
        )

        # Constant-time comparison
        if not hmac.compare_digest(signature, expected_signature):
            return False, "Invalid signature"

        return True, None


class ReplayProtection:
    """
    Middleware for replay attack protection.

    Validates request signatures and prevents replay attacks
    using nonces and timestamps.
    """

    def __init__(
        self,
        signer: Optional[RequestSigner] = None,
        require_signature: bool = False,
        exempt_paths: Optional[List[str]] = None,
    ):
        """
        Initialize replay protection.

        Args:
            signer: Request signer instance
            require_signature: Require signatures on all requests
            exempt_paths: Paths to exempt from signature verification
        """
        self.signer = signer or RequestSigner()
        self.require_signature = require_signature
        self.exempt_paths = set(exempt_paths or ["/health", "/ready", "/live"])

    async def verify(self, request: Request) -> Tuple[bool, Optional[str]]:
        """
        Verify request signature.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (valid, error_message)
        """
        path = request.url.path

        # Check exemptions
        if any(path.startswith(exempt) for exempt in self.exempt_paths):
            return True, None

        # Check if signature headers are present
        has_signature = request.headers.get(RequestSigner.SIGNATURE_HEADER)

        if not has_signature:
            if self.require_signature:
                return False, "Request signature required"
            return True, None

        # Read body
        body = await request.body()

        # Verify signature
        return await self.signer.verify_request(
            method=request.method,
            path=path,
            headers=dict(request.headers),
            body=body,
        )


class RequestSigningMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request signature verification.

    Automatically verifies request signatures and rejects
    invalid or replayed requests.
    """

    def __init__(
        self,
        app,
        signer: Optional[RequestSigner] = None,
        require_signature: bool = False,
        exempt_paths: Optional[List[str]] = None,
    ):
        """
        Initialize request signing middleware.

        Args:
            app: FastAPI application
            signer: Request signer instance
            require_signature: Require signatures on all requests
            exempt_paths: Paths to exempt from verification
        """
        super().__init__(app)
        self.protection = ReplayProtection(
            signer=signer,
            require_signature=require_signature,
            exempt_paths=exempt_paths,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verify request signature before processing."""
        valid, error = await self.protection.verify(request)

        if not valid:
            logger.warning(
                f"Request signature verification failed: {error}",
                extra={
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                    "error": error,
                },
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Request verification failed: {error}",
            )

        return await call_next(request)


def generate_request_fingerprint(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes = b"",
    include_headers: Optional[List[str]] = None,
) -> str:
    """
    Generate a fingerprint for a request.

    Useful for request deduplication and caching.

    Args:
        method: HTTP method
        path: Request path
        headers: Request headers
        body: Request body
        include_headers: Headers to include in fingerprint

    Returns:
        Hex-encoded fingerprint
    """
    include_headers = include_headers or ["Content-Type", "Accept", "Authorization"]

    parts = [method.upper(), path]

    # Add selected headers
    for header in sorted(include_headers):
        value = headers.get(header, "")
        parts.append(f"{header}:{value}")

    # Add body hash
    body_hash = hashlib.sha256(body).hexdigest()
    parts.append(body_hash)

    # Compute fingerprint
    canonical = "\n".join(parts)
    return hashlib.sha256(canonical.encode()).hexdigest()
