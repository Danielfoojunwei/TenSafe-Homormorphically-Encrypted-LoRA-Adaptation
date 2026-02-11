"""
Fallback and Graceful Degradation Utilities.

Provides fallback mechanisms for failure scenarios:
- Default value fallbacks
- Cached result fallbacks
- Fallback chains
- Degraded mode operation

Usage:
    # Simple default fallback
    @default_fallback(default_value=[])
    def get_items():
        return api.get_items()

    # Cached fallback
    @cached_fallback(ttl=300)
    def get_config():
        return config_server.get()

    # Fallback chain
    chain = FallbackChain([
        primary_source,
        secondary_source,
        local_cache,
        default_value,
    ])
    result = chain.execute()
"""

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the fallback cache."""

    value: T
    timestamp: float
    ttl: float
    stale_ttl: float = 0.0

    def is_fresh(self) -> bool:
        """Check if entry is still fresh."""
        return time.time() - self.timestamp < self.ttl

    def is_usable(self) -> bool:
        """Check if entry can be used (fresh or within stale period)."""
        age = time.time() - self.timestamp
        return age < (self.ttl + self.stale_ttl)


class FallbackCache(Generic[T]):
    """
    Cache for fallback values with stale-while-revalidate support.

    Stores successful results for use when the primary source fails.
    """

    def __init__(
        self,
        ttl: float = 300.0,
        stale_ttl: float = 600.0,
        max_entries: int = 1000,
    ):
        """
        Initialize fallback cache.

        Args:
            ttl: Time-to-live for fresh entries (seconds)
            stale_ttl: Additional time stale entries can be used
            max_entries: Maximum cache entries
        """
        self.ttl = ttl
        self.stale_ttl = stale_ttl
        self.max_entries = max_entries

        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get a cache entry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry.is_usable():
                return entry
            return None

    def set(self, key: str, value: T) -> None:
        """Set a cache entry."""
        with self._lock:
            # Evict old entries if at capacity
            if len(self._cache) >= self.max_entries:
                self._evict_oldest()

            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=self.ttl,
                stale_ttl=self.stale_ttl,
            )

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries."""
        if not self._cache:
            return

        # Sort by timestamp and remove oldest 10%
        entries = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
        to_remove = max(1, len(entries) // 10)

        for key, _ in entries[:to_remove]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


class Fallback(Generic[T]):
    """
    Single fallback handler with caching support.

    Wraps a function to provide fallback behavior on failure.
    """

    def __init__(
        self,
        func: Callable[..., T],
        fallback_value: Optional[T] = None,
        fallback_func: Optional[Callable[..., T]] = None,
        cache_ttl: float = 0.0,
        stale_ttl: float = 0.0,
        exceptions: tuple = (Exception,),
        on_fallback: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize fallback handler.

        Args:
            func: Primary function
            fallback_value: Static fallback value
            fallback_func: Dynamic fallback function
            cache_ttl: Cache TTL for successful results
            stale_ttl: Additional stale period for cache
            exceptions: Exceptions to catch
            on_fallback: Callback when fallback is used
        """
        self.func = func
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
        self.exceptions = exceptions
        self.on_fallback = on_fallback

        self._cache = FallbackCache[T](ttl=cache_ttl, stale_ttl=stale_ttl)
        self._use_cache = cache_ttl > 0
        self._lock = threading.RLock()
        self._stats = {
            "total_calls": 0,
            "successes": 0,
            "fallbacks": 0,
            "cache_hits": 0,
        }

    def _get_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key from arguments."""
        try:
            return str((args, sorted(kwargs.items())))
        except Exception:
            return "default"

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Execute with fallback."""
        with self._lock:
            self._stats["total_calls"] += 1

        cache_key = self._get_cache_key(args, kwargs)

        try:
            result = self.func(*args, **kwargs)

            with self._lock:
                self._stats["successes"] += 1

            # Cache successful result
            if self._use_cache:
                self._cache.set(cache_key, result)

            return result

        except self.exceptions as e:
            logger.warning(
                f"Primary function failed, using fallback: {e}",
                extra={"function": self.func.__name__, "error": str(e)},
            )

            # Try cached value first
            if self._use_cache:
                cached = self._cache.get(cache_key)
                if cached:
                    with self._lock:
                        self._stats["cache_hits"] += 1
                        self._stats["fallbacks"] += 1

                    logger.info(
                        f"Using cached fallback for {self.func.__name__}",
                        extra={"fresh": cached.is_fresh()},
                    )
                    return cached.value

            # Use fallback
            with self._lock:
                self._stats["fallbacks"] += 1

            if self.on_fallback:
                try:
                    self.on_fallback(e)
                except Exception as callback_error:
                    logger.warning(f"Fallback callback failed: {callback_error}")

            if self.fallback_func:
                return self.fallback_func(*args, **kwargs)

            if self.fallback_value is not None:
                return self.fallback_value

            # No fallback available, re-raise
            raise

    async def call_async(self, *args: Any, **kwargs: Any) -> T:
        """Execute async function with fallback."""
        with self._lock:
            self._stats["total_calls"] += 1

        cache_key = self._get_cache_key(args, kwargs)

        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*args, **kwargs)
            else:
                result = self.func(*args, **kwargs)

            with self._lock:
                self._stats["successes"] += 1

            if self._use_cache:
                self._cache.set(cache_key, result)

            return result

        except self.exceptions as e:
            logger.warning(
                f"Primary async function failed, using fallback: {e}",
                extra={"function": self.func.__name__, "error": str(e)},
            )

            if self._use_cache:
                cached = self._cache.get(cache_key)
                if cached:
                    with self._lock:
                        self._stats["cache_hits"] += 1
                        self._stats["fallbacks"] += 1
                    return cached.value

            with self._lock:
                self._stats["fallbacks"] += 1

            if self.on_fallback:
                try:
                    self.on_fallback(e)
                except Exception as callback_error:
                    logger.warning(f"Fallback callback failed: {callback_error}")

            if self.fallback_func:
                if asyncio.iscoroutinefunction(self.fallback_func):
                    return await self.fallback_func(*args, **kwargs)
                return self.fallback_func(*args, **kwargs)

            if self.fallback_value is not None:
                return self.fallback_value

            raise

    def get_stats(self) -> dict:
        """Get fallback statistics."""
        with self._lock:
            return dict(self._stats)


class FallbackChain(Generic[T]):
    """
    Chain of fallback sources.

    Tries each source in order until one succeeds.
    """

    def __init__(
        self,
        sources: List[Union[Callable[..., T], T]],
        exceptions: tuple = (Exception,),
        on_source_failure: Optional[Callable[[int, Exception], None]] = None,
    ):
        """
        Initialize fallback chain.

        Args:
            sources: List of fallback sources (functions or values)
            exceptions: Exceptions to catch
            on_source_failure: Callback when a source fails
        """
        self.sources = sources
        self.exceptions = exceptions
        self.on_source_failure = on_source_failure

        self._stats = {
            "total_calls": 0,
            "source_successes": [0] * len(sources),
            "all_failed": 0,
        }
        self._lock = threading.RLock()

    def execute(self, *args: Any, **kwargs: Any) -> T:
        """
        Execute fallback chain.

        Args:
            *args: Arguments passed to callable sources
            **kwargs: Keyword arguments

        Returns:
            Result from first successful source

        Raises:
            Exception: If all sources fail
        """
        with self._lock:
            self._stats["total_calls"] += 1

        last_exception = None

        for i, source in enumerate(self.sources):
            try:
                if callable(source):
                    result = source(*args, **kwargs)
                else:
                    result = source

                with self._lock:
                    self._stats["source_successes"][i] += 1

                logger.debug(f"Fallback chain succeeded at source {i}")
                return result

            except self.exceptions as e:
                last_exception = e

                logger.debug(f"Fallback chain source {i} failed: {e}")

                if self.on_source_failure:
                    try:
                        self.on_source_failure(i, e)
                    except Exception:
                        pass

        with self._lock:
            self._stats["all_failed"] += 1

        logger.error(f"All {len(self.sources)} fallback sources failed")

        if last_exception:
            raise last_exception
        raise RuntimeError("All fallback sources failed")

    async def execute_async(self, *args: Any, **kwargs: Any) -> T:
        """Execute fallback chain asynchronously."""
        with self._lock:
            self._stats["total_calls"] += 1

        last_exception = None

        for i, source in enumerate(self.sources):
            try:
                if callable(source):
                    if asyncio.iscoroutinefunction(source):
                        result = await source(*args, **kwargs)
                    else:
                        result = source(*args, **kwargs)
                else:
                    result = source

                with self._lock:
                    self._stats["source_successes"][i] += 1

                return result

            except self.exceptions as e:
                last_exception = e

                if self.on_source_failure:
                    try:
                        self.on_source_failure(i, e)
                    except Exception:
                        pass

        with self._lock:
            self._stats["all_failed"] += 1

        if last_exception:
            raise last_exception
        raise RuntimeError("All fallback sources failed")

    def get_stats(self) -> dict:
        """Get chain statistics."""
        with self._lock:
            return dict(self._stats)


def default_fallback(
    default_value: T,
    exceptions: tuple = (Exception,),
    log_fallback: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for providing a default value on failure.

    Args:
        default_value: Value to return on failure
        exceptions: Exceptions to catch
        log_fallback: Whether to log fallback usage

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        fallback = Fallback(
            func=func,
            fallback_value=default_value,
            exceptions=exceptions,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await fallback.call_async(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return fallback(*args, **kwargs)

            return sync_wrapper

    return decorator


def cached_fallback(
    ttl: float = 300.0,
    stale_ttl: float = 600.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching results with stale fallback.

    Args:
        ttl: Fresh cache TTL in seconds
        stale_ttl: Additional stale period
        exceptions: Exceptions to catch

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        fallback = Fallback(
            func=func,
            cache_ttl=ttl,
            stale_ttl=stale_ttl,
            exceptions=exceptions,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await fallback.call_async(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return fallback(*args, **kwargs)

            return sync_wrapper

    return decorator


class DegradedModeManager:
    """
    Manages degraded mode operation.

    Tracks system health and enables/disables features
    based on system state.
    """

    def __init__(self):
        """Initialize degraded mode manager."""
        self._degraded = False
        self._disabled_features: set[str] = set()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[bool], None]] = []

    @property
    def is_degraded(self) -> bool:
        """Check if system is in degraded mode."""
        with self._lock:
            return self._degraded

    def enter_degraded_mode(self, reason: str = "") -> None:
        """Enter degraded mode."""
        with self._lock:
            if not self._degraded:
                self._degraded = True
                logger.warning(f"Entering degraded mode: {reason}")

                for listener in self._listeners:
                    try:
                        listener(True)
                    except Exception as e:
                        logger.warning(f"Degraded mode listener failed: {e}")

    def exit_degraded_mode(self, reason: str = "") -> None:
        """Exit degraded mode."""
        with self._lock:
            if self._degraded:
                self._degraded = False
                self._disabled_features.clear()
                logger.info(f"Exiting degraded mode: {reason}")

                for listener in self._listeners:
                    try:
                        listener(False)
                    except Exception as e:
                        logger.warning(f"Degraded mode listener failed: {e}")

    def disable_feature(self, feature: str) -> None:
        """Disable a feature in degraded mode."""
        with self._lock:
            self._disabled_features.add(feature)
            logger.info(f"Disabled feature: {feature}")

    def enable_feature(self, feature: str) -> None:
        """Re-enable a feature."""
        with self._lock:
            self._disabled_features.discard(feature)
            logger.info(f"Enabled feature: {feature}")

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        with self._lock:
            return feature not in self._disabled_features

    def add_listener(self, listener: Callable[[bool], None]) -> None:
        """Add a listener for degraded mode changes."""
        with self._lock:
            self._listeners.append(listener)

    def get_status(self) -> dict:
        """Get degraded mode status."""
        with self._lock:
            return {
                "degraded": self._degraded,
                "disabled_features": list(self._disabled_features),
            }
