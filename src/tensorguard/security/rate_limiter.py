"""
Rate Limiting Module.

Provides production-grade rate limiting with:
- Token bucket algorithm for smooth traffic shaping
- IP-based, user-based, and API key-based limiting
- Redis backend for distributed deployments
- In-memory fallback for single-instance deployments
- Configurable burst allowance
- Endpoint-specific rate limits
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Requests per window
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Burst allowance (extra requests allowed in short burst)
    burst_allowance: int = 10

    # Window sizes
    minute_window: int = 60  # seconds
    hour_window: int = 3600  # seconds

    # Blocking behavior
    block_duration_minutes: int = 15  # Duration to block after limit exceeded
    warn_threshold: float = 0.8  # Warn when this % of limit reached

    # Endpoint-specific overrides (path pattern -> limits)
    endpoint_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Exemptions
    exempt_paths: List[str] = field(
        default_factory=lambda: ["/health", "/ready", "/live", "/metrics"]
    )
    exempt_ips: List[str] = field(default_factory=list)

    # Redis configuration (optional)
    redis_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create config from environment variables."""
        return cls(
            requests_per_minute=int(os.getenv("TG_RATE_LIMIT_PER_MINUTE", "60")),
            requests_per_hour=int(os.getenv("TG_RATE_LIMIT_PER_HOUR", "1000")),
            burst_allowance=int(os.getenv("TG_RATE_LIMIT_BURST", "10")),
            block_duration_minutes=int(os.getenv("TG_RATE_LIMIT_BLOCK_MINUTES", "15")),
            redis_url=os.getenv("TG_REDIS_URL"),
            exempt_paths=[
                p.strip()
                for p in os.getenv(
                    "TG_RATE_LIMIT_EXEMPT_PATHS", "/health,/ready,/live,/metrics"
                ).split(",")
            ],
        )


@dataclass
class RateLimitEntry:
    """Tracks rate limit state for a single key."""

    tokens: float
    last_update: float
    request_count_minute: int = 0
    request_count_hour: int = 0
    minute_reset: float = 0.0
    hour_reset: float = 0.0
    blocked_until: Optional[float] = None
    warnings_sent: int = 0


class RateLimiter:
    """
    Token bucket rate limiter with distributed support.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Supports both in-memory and Redis backends
    - Per-IP, per-user, and per-API-key limiting
    - Configurable burst allowance
    - Automatic blocking for persistent violators
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig.from_env()
        self._store: Dict[str, RateLimitEntry] = {}
        self._redis_client = None
        self._lock = asyncio.Lock()

        # Try to initialize Redis
        if self.config.redis_url:
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection for distributed rate limiting."""
        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Rate limiter using Redis backend")
        except ImportError:
            logger.warning(
                "Redis not available, using in-memory rate limiting. "
                "Install redis: pip install redis"
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory fallback.")

    def _get_client_key(
        self,
        request: Request,
        key_type: str = "ip",
    ) -> str:
        """
        Generate a unique key for the client.

        Args:
            request: FastAPI request
            key_type: Type of key (ip, user, api_key)

        Returns:
            Unique client identifier
        """
        if key_type == "user":
            # Extract user ID from JWT if available
            user_id = getattr(request.state, "user_id", None)
            if user_id:
                return f"user:{user_id}"

        if key_type == "api_key":
            # Extract API key from header
            api_key = request.headers.get("X-API-Key")
            if api_key:
                # Hash the API key to avoid storing it directly
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                return f"apikey:{key_hash}"

        # Default: IP-based limiting
        # Handle X-Forwarded-For for reverse proxies
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (client IP)
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"

    async def _get_entry(self, key: str) -> RateLimitEntry:
        """Get or create rate limit entry for a key."""
        now = time.time()

        if self._redis_client:
            return await self._get_entry_redis(key, now)

        # In-memory fallback
        async with self._lock:
            if key not in self._store:
                self._store[key] = RateLimitEntry(
                    tokens=float(
                        self.config.requests_per_minute + self.config.burst_allowance
                    ),
                    last_update=now,
                    minute_reset=now + self.config.minute_window,
                    hour_reset=now + self.config.hour_window,
                )
            return self._store[key]

    async def _get_entry_redis(self, key: str, now: float) -> RateLimitEntry:
        """Get rate limit entry from Redis."""
        redis_key = f"ratelimit:{key}"

        try:
            data = await self._redis_client.hgetall(redis_key)

            if not data:
                entry = RateLimitEntry(
                    tokens=float(
                        self.config.requests_per_minute + self.config.burst_allowance
                    ),
                    last_update=now,
                    minute_reset=now + self.config.minute_window,
                    hour_reset=now + self.config.hour_window,
                )
                await self._save_entry_redis(key, entry)
                return entry

            return RateLimitEntry(
                tokens=float(data.get("tokens", self.config.requests_per_minute)),
                last_update=float(data.get("last_update", now)),
                request_count_minute=int(data.get("request_count_minute", 0)),
                request_count_hour=int(data.get("request_count_hour", 0)),
                minute_reset=float(data.get("minute_reset", now)),
                hour_reset=float(data.get("hour_reset", now)),
                blocked_until=float(data["blocked_until"])
                if data.get("blocked_until")
                else None,
                warnings_sent=int(data.get("warnings_sent", 0)),
            )
        except Exception as e:
            logger.warning(f"Redis error, falling back to in-memory: {e}")
            return await self._get_entry(key)

    async def _save_entry_redis(self, key: str, entry: RateLimitEntry) -> None:
        """Save rate limit entry to Redis."""
        redis_key = f"ratelimit:{key}"

        try:
            data = {
                "tokens": str(entry.tokens),
                "last_update": str(entry.last_update),
                "request_count_minute": str(entry.request_count_minute),
                "request_count_hour": str(entry.request_count_hour),
                "minute_reset": str(entry.minute_reset),
                "hour_reset": str(entry.hour_reset),
                "warnings_sent": str(entry.warnings_sent),
            }
            if entry.blocked_until:
                data["blocked_until"] = str(entry.blocked_until)

            await self._redis_client.hset(redis_key, mapping=data)
            # Set TTL to clean up stale entries
            await self._redis_client.expire(redis_key, self.config.hour_window * 2)
        except Exception as e:
            logger.warning(f"Failed to save to Redis: {e}")

    async def _update_entry(self, key: str, entry: RateLimitEntry) -> None:
        """Update rate limit entry."""
        if self._redis_client:
            await self._save_entry_redis(key, entry)
        else:
            async with self._lock:
                self._store[key] = entry

    def _replenish_tokens(self, entry: RateLimitEntry, now: float) -> None:
        """Replenish tokens based on time elapsed."""
        elapsed = now - entry.last_update

        # Token replenishment rate (tokens per second)
        rate = self.config.requests_per_minute / self.config.minute_window

        # Add tokens based on elapsed time
        entry.tokens = min(
            entry.tokens + (elapsed * rate),
            self.config.requests_per_minute + self.config.burst_allowance,
        )
        entry.last_update = now

        # Reset counters if windows have passed
        if now >= entry.minute_reset:
            entry.request_count_minute = 0
            entry.minute_reset = now + self.config.minute_window

        if now >= entry.hour_reset:
            entry.request_count_hour = 0
            entry.hour_reset = now + self.config.hour_window

    async def check_rate_limit(
        self,
        request: Request,
        key_type: str = "ip",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be rate limited.

        Args:
            request: FastAPI request
            key_type: Type of rate limit key

        Returns:
            Tuple of (allowed, metadata)
        """
        # Check exemptions
        path = request.url.path
        if any(path.startswith(exempt) for exempt in self.config.exempt_paths):
            return True, {"exempt": True}

        key = self._get_client_key(request, key_type)

        # Check IP exemption
        if key.startswith("ip:"):
            ip = key[3:]
            if ip in self.config.exempt_ips:
                return True, {"exempt": True}

        entry = await self._get_entry(key)
        now = time.time()

        # Check if blocked
        if entry.blocked_until and now < entry.blocked_until:
            remaining = int(entry.blocked_until - now)
            return False, {
                "blocked": True,
                "retry_after": remaining,
                "reason": "Rate limit exceeded - temporarily blocked",
            }

        # Replenish tokens
        self._replenish_tokens(entry, now)

        # Check endpoint-specific limits
        endpoint_limit = None
        for pattern, limits in self.config.endpoint_overrides.items():
            if path.startswith(pattern):
                endpoint_limit = limits.get("per_minute", self.config.requests_per_minute)
                break

        effective_limit = endpoint_limit or self.config.requests_per_minute

        # Check limits
        if entry.tokens < 1:
            # Rate limit exceeded
            entry.warnings_sent += 1

            # Block persistent violators
            if entry.warnings_sent >= 3:
                entry.blocked_until = now + (self.config.block_duration_minutes * 60)
                await self._update_entry(key, entry)

                logger.warning(
                    f"Rate limit: blocked {key} for {self.config.block_duration_minutes} minutes"
                )

                return False, {
                    "blocked": True,
                    "retry_after": self.config.block_duration_minutes * 60,
                    "reason": "Persistent rate limit violations - temporarily blocked",
                }

            await self._update_entry(key, entry)

            retry_after = int(self.config.minute_window / effective_limit)
            return False, {
                "limited": True,
                "retry_after": retry_after,
                "remaining": 0,
                "limit": effective_limit,
            }

        # Check hourly limit
        if entry.request_count_hour >= self.config.requests_per_hour:
            retry_after = int(entry.hour_reset - now)
            return False, {
                "limited": True,
                "retry_after": retry_after,
                "remaining": 0,
                "limit": self.config.requests_per_hour,
                "window": "hour",
            }

        # Consume a token
        entry.tokens -= 1
        entry.request_count_minute += 1
        entry.request_count_hour += 1

        # Check warning threshold
        warn = False
        if entry.tokens < effective_limit * (1 - self.config.warn_threshold):
            warn = True

        await self._update_entry(key, entry)

        return True, {
            "remaining": int(entry.tokens),
            "limit": effective_limit,
            "reset": int(entry.minute_reset - now),
            "warning": warn,
        }

    async def reset_limit(self, key: str) -> None:
        """Reset rate limit for a key (admin function)."""
        if self._redis_client:
            redis_key = f"ratelimit:{key}"
            await self._redis_client.delete(redis_key)
        else:
            async with self._lock:
                self._store.pop(key, None)

        logger.info(f"Rate limit reset for {key}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Automatically applies rate limits to all requests and adds
    standard rate limit headers to responses.
    """

    def __init__(
        self,
        app,
        rate_limiter: Optional[RateLimiter] = None,
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            rate_limiter: Optional pre-configured rate limiter
            config: Rate limit configuration
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter(config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiter."""
        allowed, metadata = await self.rate_limiter.check_rate_limit(request)

        if not allowed:
            # Return 429 Too Many Requests
            retry_after = metadata.get("retry_after", 60)
            detail = metadata.get("reason", "Rate limit exceeded")

            logger.warning(
                f"Rate limit exceeded: {request.client.host} -> {request.url.path}",
                extra={
                    "client_ip": request.client.host if request.client else "unknown",
                    "path": request.url.path,
                    "retry_after": retry_after,
                },
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=detail,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(metadata.get("limit", 60)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if not metadata.get("exempt"):
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", 60))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", 60))

            if metadata.get("warning"):
                response.headers["X-RateLimit-Warning"] = "Approaching rate limit"

        return response
