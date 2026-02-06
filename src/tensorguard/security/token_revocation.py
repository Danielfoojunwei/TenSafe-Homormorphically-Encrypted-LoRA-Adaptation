"""
Token Revocation List (TRL) Module.

Provides JWT token invalidation capabilities:
- JTI-based revocation for individual tokens
- User-based revocation for all tokens of a user
- Session-based revocation
- Redis backend for distributed deployments
- Automatic cleanup of expired revocations
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import HTTPException, Request, Response, status
import jwt
from jwt.exceptions import PyJWTError as JWTError
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RevocationReason(str, Enum):
    """Reasons for token revocation."""

    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    SECURITY_INCIDENT = "security_incident"
    ADMIN_REVOKE = "admin_revoke"
    SESSION_EXPIRED = "session_expired"
    USER_DISABLED = "user_disabled"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    KEY_ROTATION = "key_rotation"


@dataclass
class RevocationEntry:
    """Record of a revoked token or session."""

    identifier: str  # JTI or user ID
    revocation_type: str  # "jti", "user", "session"
    reason: RevocationReason
    revoked_at: float
    expires_at: float
    revoked_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TokenRevocationList:
    """
    Manages revoked tokens and sessions.

    Features:
    - JTI-based revocation for individual tokens
    - User-based revocation (invalidates all tokens for a user)
    - Session-based revocation
    - Automatic cleanup of expired revocations
    - Redis support for distributed deployments
    """

    CLEANUP_INTERVAL = 300  # 5 minutes

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl_hours: int = 24,
    ):
        """
        Initialize token revocation list.

        Args:
            redis_url: Redis connection URL for distributed deployments
            default_ttl_hours: Default TTL for revocation entries
        """
        self.redis_url = redis_url or os.getenv("TG_REDIS_URL")
        self.default_ttl = default_ttl_hours * 3600
        self._redis_client = None
        self._lock = asyncio.Lock()

        # In-memory storage
        self._revoked_jtis: Dict[str, RevocationEntry] = {}
        self._revoked_users: Dict[str, RevocationEntry] = {}
        self._revoked_sessions: Dict[str, RevocationEntry] = {}

        # User revocation timestamps (for "revoke all tokens issued before X")
        self._user_revoke_timestamps: Dict[str, float] = {}

        # Initialize Redis if available
        if self.redis_url:
            self._init_redis()

        # Start cleanup task
        self._cleanup_task = None

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Token revocation list using Redis backend")
        except ImportError:
            logger.warning(
                "Redis not available for token revocation. "
                "Using in-memory storage (not suitable for distributed deployments)."
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
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
        """Background loop to clean up expired revocations."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in revocation cleanup: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired revocation entries."""
        now = time.time()

        if self._redis_client:
            # Redis handles TTL automatically via EXPIRE
            return

        async with self._lock:
            # Clean JTIs
            expired_jtis = [
                k for k, v in self._revoked_jtis.items() if v.expires_at < now
            ]
            for jti in expired_jtis:
                del self._revoked_jtis[jti]

            # Clean users
            expired_users = [
                k for k, v in self._revoked_users.items() if v.expires_at < now
            ]
            for user in expired_users:
                del self._revoked_users[user]

            # Clean sessions
            expired_sessions = [
                k for k, v in self._revoked_sessions.items() if v.expires_at < now
            ]
            for session in expired_sessions:
                del self._revoked_sessions[session]

            if expired_jtis or expired_users or expired_sessions:
                logger.debug(
                    f"Cleaned up revocations: {len(expired_jtis)} JTIs, "
                    f"{len(expired_users)} users, {len(expired_sessions)} sessions"
                )

    async def revoke_token(
        self,
        jti: str,
        reason: RevocationReason = RevocationReason.LOGOUT,
        ttl_seconds: Optional[int] = None,
        revoked_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Revoke a specific token by JTI.

        Args:
            jti: JWT token ID
            reason: Reason for revocation
            ttl_seconds: How long to keep the revocation entry
            revoked_by: ID of admin/user who revoked the token
            metadata: Additional metadata
        """
        now = time.time()
        ttl = ttl_seconds or self.default_ttl
        expires_at = now + ttl

        entry = RevocationEntry(
            identifier=jti,
            revocation_type="jti",
            reason=reason,
            revoked_at=now,
            expires_at=expires_at,
            revoked_by=revoked_by,
            metadata=metadata,
        )

        if self._redis_client:
            await self._revoke_redis("jti", jti, entry, ttl)
        else:
            async with self._lock:
                self._revoked_jtis[jti] = entry

        logger.info(
            f"Revoked token JTI={jti[:16]}... reason={reason.value}",
            extra={"jti": jti, "reason": reason.value, "revoked_by": revoked_by},
        )

    async def revoke_user_tokens(
        self,
        user_id: str,
        reason: RevocationReason = RevocationReason.PASSWORD_CHANGE,
        ttl_seconds: Optional[int] = None,
        revoked_by: Optional[str] = None,
        issued_before: Optional[float] = None,
    ) -> None:
        """
        Revoke all tokens for a user.

        Args:
            user_id: User identifier
            reason: Reason for revocation
            ttl_seconds: How long to keep the revocation entry
            revoked_by: ID of admin who revoked
            issued_before: Only revoke tokens issued before this timestamp
        """
        now = time.time()
        ttl = ttl_seconds or self.default_ttl
        expires_at = now + ttl

        # Store the revocation timestamp
        revoke_timestamp = issued_before or now

        entry = RevocationEntry(
            identifier=user_id,
            revocation_type="user",
            reason=reason,
            revoked_at=now,
            expires_at=expires_at,
            revoked_by=revoked_by,
            metadata={"issued_before": revoke_timestamp},
        )

        if self._redis_client:
            await self._revoke_redis("user", user_id, entry, ttl)
            await self._redis_client.set(
                f"trl:user_timestamp:{user_id}",
                str(revoke_timestamp),
                ex=ttl,
            )
        else:
            async with self._lock:
                self._revoked_users[user_id] = entry
                self._user_revoke_timestamps[user_id] = revoke_timestamp

        logger.info(
            f"Revoked all tokens for user={user_id} reason={reason.value}",
            extra={"user_id": user_id, "reason": reason.value, "revoked_by": revoked_by},
        )

    async def revoke_session(
        self,
        session_id: str,
        reason: RevocationReason = RevocationReason.LOGOUT,
        ttl_seconds: Optional[int] = None,
        revoked_by: Optional[str] = None,
    ) -> None:
        """
        Revoke a session.

        Args:
            session_id: Session identifier
            reason: Reason for revocation
            ttl_seconds: How long to keep the revocation entry
            revoked_by: ID of user who ended the session
        """
        now = time.time()
        ttl = ttl_seconds or self.default_ttl
        expires_at = now + ttl

        entry = RevocationEntry(
            identifier=session_id,
            revocation_type="session",
            reason=reason,
            revoked_at=now,
            expires_at=expires_at,
            revoked_by=revoked_by,
        )

        if self._redis_client:
            await self._revoke_redis("session", session_id, entry, ttl)
        else:
            async with self._lock:
                self._revoked_sessions[session_id] = entry

        logger.info(
            f"Revoked session={session_id[:16]}... reason={reason.value}",
            extra={"session_id": session_id, "reason": reason.value},
        )

    async def _revoke_redis(
        self,
        revocation_type: str,
        identifier: str,
        entry: RevocationEntry,
        ttl: int,
    ) -> None:
        """Store revocation in Redis."""
        key = f"trl:{revocation_type}:{identifier}"
        data = {
            "identifier": entry.identifier,
            "type": entry.revocation_type,
            "reason": entry.reason.value,
            "revoked_at": str(entry.revoked_at),
            "expires_at": str(entry.expires_at),
            "revoked_by": entry.revoked_by or "",
        }
        if entry.metadata:
            data["metadata"] = str(entry.metadata)

        await self._redis_client.hset(key, mapping=data)
        await self._redis_client.expire(key, ttl)

    async def is_revoked(
        self,
        jti: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        issued_at: Optional[float] = None,
    ) -> bool:
        """
        Check if a token, user, or session is revoked.

        Args:
            jti: JWT token ID
            user_id: User identifier
            session_id: Session identifier
            issued_at: Token issue timestamp (for user-level revocation)

        Returns:
            True if revoked, False otherwise
        """
        # Check JTI revocation
        if jti:
            if await self._is_jti_revoked(jti):
                return True

        # Check user-level revocation
        if user_id and issued_at:
            if await self._is_user_revoked(user_id, issued_at):
                return True

        # Check session revocation
        if session_id:
            if await self._is_session_revoked(session_id):
                return True

        return False

    async def _is_jti_revoked(self, jti: str) -> bool:
        """Check if a specific JTI is revoked."""
        if self._redis_client:
            exists = await self._redis_client.exists(f"trl:jti:{jti}")
            return bool(exists)

        async with self._lock:
            entry = self._revoked_jtis.get(jti)
            if entry and entry.expires_at > time.time():
                return True
            return False

    async def _is_user_revoked(self, user_id: str, issued_at: float) -> bool:
        """Check if tokens for a user issued before a timestamp are revoked."""
        if self._redis_client:
            timestamp = await self._redis_client.get(f"trl:user_timestamp:{user_id}")
            if timestamp:
                revoke_time = float(timestamp)
                return issued_at < revoke_time
            return False

        async with self._lock:
            revoke_time = self._user_revoke_timestamps.get(user_id)
            if revoke_time and issued_at < revoke_time:
                return True
            return False

    async def _is_session_revoked(self, session_id: str) -> bool:
        """Check if a session is revoked."""
        if self._redis_client:
            exists = await self._redis_client.exists(f"trl:session:{session_id}")
            return bool(exists)

        async with self._lock:
            entry = self._revoked_sessions.get(session_id)
            if entry and entry.expires_at > time.time():
                return True
            return False

    async def get_revocation_reason(
        self,
        jti: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[RevocationReason]:
        """Get the reason a token was revoked."""
        if self._redis_client:
            if jti:
                data = await self._redis_client.hget(f"trl:jti:{jti}", "reason")
                if data:
                    return RevocationReason(data)
            if user_id:
                data = await self._redis_client.hget(f"trl:user:{user_id}", "reason")
                if data:
                    return RevocationReason(data)
            return None

        async with self._lock:
            if jti and jti in self._revoked_jtis:
                return self._revoked_jtis[jti].reason
            if user_id and user_id in self._revoked_users:
                return self._revoked_users[user_id].reason
            return None

    async def get_stats(self) -> Dict[str, int]:
        """Get revocation statistics."""
        if self._redis_client:
            jti_count = 0
            user_count = 0
            session_count = 0

            async for _ in self._redis_client.scan_iter("trl:jti:*"):
                jti_count += 1
            async for _ in self._redis_client.scan_iter("trl:user:*"):
                user_count += 1
            async for _ in self._redis_client.scan_iter("trl:session:*"):
                session_count += 1

            return {
                "revoked_tokens": jti_count,
                "revoked_users": user_count,
                "revoked_sessions": session_count,
            }

        async with self._lock:
            now = time.time()
            return {
                "revoked_tokens": sum(
                    1 for e in self._revoked_jtis.values() if e.expires_at > now
                ),
                "revoked_users": sum(
                    1 for e in self._revoked_users.values() if e.expires_at > now
                ),
                "revoked_sessions": sum(
                    1 for e in self._revoked_sessions.values() if e.expires_at > now
                ),
            }


class TokenRevocationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for checking token revocation.

    Integrates with JWT authentication to reject revoked tokens.
    """

    def __init__(
        self,
        app,
        revocation_list: Optional[TokenRevocationList] = None,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
    ):
        """
        Initialize token revocation middleware.

        Args:
            app: FastAPI application
            revocation_list: Token revocation list instance
            secret_key: JWT secret key for decoding
            algorithm: JWT algorithm
        """
        super().__init__(app)
        self.trl = revocation_list or TokenRevocationList()
        self.secret_key = secret_key or os.getenv("TG_SECRET_KEY", "")
        self.algorithm = algorithm

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check token revocation before processing request."""
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            try:
                # Decode without verification first to get claims
                # (verification happens in auth middleware)
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    options={"verify_signature": False},
                )

                jti = payload.get("jti")
                user_id = payload.get("sub")
                session_id = payload.get("sid")
                issued_at = payload.get("iat")

                # Check revocation
                if await self.trl.is_revoked(
                    jti=jti,
                    user_id=user_id,
                    session_id=session_id,
                    issued_at=issued_at,
                ):
                    reason = await self.trl.get_revocation_reason(jti=jti, user_id=user_id)

                    logger.warning(
                        f"Rejected revoked token: jti={jti[:16] if jti else 'N/A'}...",
                        extra={
                            "jti": jti,
                            "user_id": user_id,
                            "reason": reason.value if reason else "unknown",
                        },
                    )

                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

            except JWTError:
                # Invalid token - let auth middleware handle it
                pass
            except HTTPException:
                # Re-raise our revocation exception
                raise

        return await call_next(request)


# Singleton instance for application-wide use
_default_trl: Optional[TokenRevocationList] = None


def get_revocation_list() -> TokenRevocationList:
    """Get or create the default token revocation list."""
    global _default_trl
    if _default_trl is None:
        _default_trl = TokenRevocationList()
    return _default_trl
