"""
Graceful Shutdown Handling.

Provides coordinated shutdown for complex applications:
- Phased shutdown with priorities
- Timeout-protected cleanup
- Resource release coordination
- Signal handling

Usage:
    shutdown = GracefulShutdown()

    # Register cleanup handlers
    shutdown.register("database", db.close, priority=10)
    shutdown.register("worker", worker.stop, priority=5)
    shutdown.register("metrics", metrics.flush, priority=1)

    # Start signal handling
    shutdown.setup_signals()

    # Or manually trigger shutdown
    await shutdown.shutdown()
"""

import asyncio
import atexit
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of graceful shutdown."""

    RUNNING = auto()
    DRAINING = auto()  # Stop accepting new work
    PROCESSING = auto()  # Finish in-flight work
    CLEANUP = auto()  # Release resources
    TERMINATED = auto()


@dataclass
class ShutdownHandler:
    """Configuration for a shutdown handler."""

    name: str
    handler: Union[Callable[[], None], Callable[[], Coroutine[Any, Any, None]]]
    priority: int = 0  # Higher priority runs first
    timeout: float = 30.0
    critical: bool = False
    phase: ShutdownPhase = ShutdownPhase.CLEANUP


class GracefulShutdown:
    """
    Coordinates graceful shutdown of application components.

    Features:
    - Priority-based handler execution
    - Timeout protection for each handler
    - Signal handling (SIGTERM, SIGINT)
    - Async and sync handler support
    - Phased shutdown for complex systems
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        drain_timeout: float = 10.0,
        process_timeout: float = 60.0,
    ):
        """
        Initialize graceful shutdown manager.

        Args:
            default_timeout: Default timeout for handlers
            drain_timeout: Timeout for drain phase
            process_timeout: Timeout for processing phase
        """
        self.default_timeout = default_timeout
        self.drain_timeout = drain_timeout
        self.process_timeout = process_timeout

        self._handlers: List[ShutdownHandler] = []
        self._phase = ShutdownPhase.RUNNING
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._signals_setup = False

        # Register atexit handler
        atexit.register(self._atexit_handler)

    @property
    def phase(self) -> ShutdownPhase:
        """Get current shutdown phase."""
        with self._lock:
            return self._phase

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._phase != ShutdownPhase.RUNNING

    def register(
        self,
        name: str,
        handler: Union[Callable[[], None], Callable[[], Coroutine[Any, Any, None]]],
        priority: int = 0,
        timeout: Optional[float] = None,
        critical: bool = False,
        phase: ShutdownPhase = ShutdownPhase.CLEANUP,
    ) -> None:
        """
        Register a shutdown handler.

        Args:
            name: Handler name for logging
            handler: Cleanup function
            priority: Higher priority runs first
            timeout: Timeout for this handler
            critical: Whether to abort shutdown on failure
            phase: Which phase this handler runs in
        """
        with self._lock:
            self._handlers.append(
                ShutdownHandler(
                    name=name,
                    handler=handler,
                    priority=priority,
                    timeout=timeout or self.default_timeout,
                    critical=critical,
                    phase=phase,
                )
            )
            # Sort by priority (descending)
            self._handlers.sort(key=lambda h: h.priority, reverse=True)

    def unregister(self, name: str) -> bool:
        """
        Unregister a shutdown handler.

        Args:
            name: Handler name

        Returns:
            True if handler was found and removed
        """
        with self._lock:
            for i, handler in enumerate(self._handlers):
                if handler.name == name:
                    del self._handlers[i]
                    return True
            return False

    def setup_signals(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        if self._signals_setup:
            return

        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating graceful shutdown")
            self._shutdown_event.set()

            # Start shutdown in background
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.shutdown())
            except RuntimeError:
                # No running loop, use sync shutdown
                threading.Thread(target=self._sync_shutdown, daemon=True).start()

        # Register signal handlers
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, signal_handler)

        self._signals_setup = True
        logger.info("Signal handlers registered for graceful shutdown")

    async def shutdown(self, reason: str = "requested") -> None:
        """
        Execute graceful shutdown.

        Args:
            reason: Reason for shutdown (for logging)
        """
        with self._lock:
            if self._phase != ShutdownPhase.RUNNING:
                logger.warning("Shutdown already in progress")
                return
            self._phase = ShutdownPhase.DRAINING

        logger.info(f"Starting graceful shutdown: {reason}")
        start_time = time.time()

        try:
            # Phase 1: Draining
            await self._run_phase(ShutdownPhase.DRAINING, self.drain_timeout)

            # Phase 2: Processing
            with self._lock:
                self._phase = ShutdownPhase.PROCESSING
            await self._run_phase(ShutdownPhase.PROCESSING, self.process_timeout)

            # Phase 3: Cleanup
            with self._lock:
                self._phase = ShutdownPhase.CLEANUP
            await self._run_phase(ShutdownPhase.CLEANUP, None)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        finally:
            with self._lock:
                self._phase = ShutdownPhase.TERMINATED

            elapsed = time.time() - start_time
            logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")

    async def _run_phase(
        self,
        phase: ShutdownPhase,
        phase_timeout: Optional[float],
    ) -> None:
        """Run all handlers for a shutdown phase."""
        with self._lock:
            handlers = [h for h in self._handlers if h.phase == phase]

        if not handlers:
            return

        logger.info(f"Shutdown phase: {phase.name} ({len(handlers)} handlers)")

        phase_start = time.time()

        for handler in handlers:
            # Check phase timeout
            if phase_timeout:
                elapsed = time.time() - phase_start
                if elapsed >= phase_timeout:
                    logger.warning(
                        f"Phase {phase.name} timeout exceeded, skipping remaining handlers"
                    )
                    break

            await self._run_handler(handler)

    async def _run_handler(self, handler: ShutdownHandler) -> bool:
        """Run a single shutdown handler with timeout."""
        logger.debug(f"Running shutdown handler: {handler.name}")
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(handler.handler):
                await asyncio.wait_for(
                    handler.handler(),
                    timeout=handler.timeout,
                )
            else:
                # Run sync handler in executor
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, handler.handler),
                    timeout=handler.timeout,
                )

            elapsed = time.time() - start_time
            logger.info(f"Shutdown handler completed: {handler.name} ({elapsed:.2f}s)")
            return True

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(
                f"Shutdown handler timed out: {handler.name} ({elapsed:.2f}s)"
            )
            if handler.critical:
                raise
            return False

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Shutdown handler failed: {handler.name} ({elapsed:.2f}s): {e}"
            )
            if handler.critical:
                raise
            return False

    def _sync_shutdown(self) -> None:
        """Synchronous shutdown for non-async contexts."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.shutdown())
        finally:
            loop.close()

    def _atexit_handler(self) -> None:
        """Handler for atexit to ensure cleanup."""
        if self._phase == ShutdownPhase.RUNNING:
            logger.info("Running atexit shutdown handlers")
            self._sync_shutdown()

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Block until shutdown is initiated.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if shutdown was initiated, False if timeout
        """
        return self._shutdown_event.wait(timeout)

    def request_shutdown(self, reason: str = "requested") -> None:
        """
        Request a shutdown (non-blocking).

        Args:
            reason: Reason for shutdown
        """
        self._shutdown_event.set()

        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.shutdown(reason))
        except RuntimeError:
            threading.Thread(
                target=self._sync_shutdown,
                daemon=True,
            ).start()


class ShutdownCoordinator:
    """
    Coordinates shutdown across multiple services.

    Useful for microservice architectures where services
    need to coordinate shutdown order.
    """

    def __init__(self):
        """Initialize shutdown coordinator."""
        self._managers: Dict[str, GracefulShutdown] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()

    def register_service(
        self,
        name: str,
        manager: GracefulShutdown,
        order: Optional[int] = None,
    ) -> None:
        """
        Register a service's shutdown manager.

        Args:
            name: Service name
            manager: Service's shutdown manager
            order: Shutdown order (lower = earlier)
        """
        with self._lock:
            self._managers[name] = manager

            if name not in self._order:
                if order is not None:
                    self._order.insert(order, name)
                else:
                    self._order.append(name)

    async def shutdown_all(self, reason: str = "coordinated shutdown") -> None:
        """
        Shutdown all registered services in order.

        Args:
            reason: Shutdown reason
        """
        logger.info(f"Coordinating shutdown of {len(self._managers)} services")

        with self._lock:
            order = list(self._order)
            managers = dict(self._managers)

        for name in order:
            manager = managers.get(name)
            if manager:
                logger.info(f"Shutting down service: {name}")
                try:
                    await manager.shutdown(reason=f"{reason} ({name})")
                except Exception as e:
                    logger.error(f"Error shutting down service {name}: {e}")

        logger.info("All services shut down")


# Global shutdown manager
_global_shutdown: Optional[GracefulShutdown] = None


def get_shutdown_manager() -> GracefulShutdown:
    """Get or create the global shutdown manager."""
    global _global_shutdown
    if _global_shutdown is None:
        _global_shutdown = GracefulShutdown()
    return _global_shutdown
