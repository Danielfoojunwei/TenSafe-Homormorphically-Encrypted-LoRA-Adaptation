"""
RLVR Async Rollout Buffer with Staleness Control

Provides async producer/consumer rollout generation for hiding HE-LoRA
inference latency. Decouples trajectory generation from training:
generation workers submit rollouts to a shared buffer, and the training
loop consumes batches as they arrive.

Key features:
- Thread-safe async buffer with staleness bounds
- Staleness manager controlling generation capacity
- Configurable max staleness (steps between generation and consumption)
- Async data tracking to prevent duplicate training on resume
- Generation slot management for backpressure
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from .rollout import Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


@dataclass
class AsyncRolloutConfig:
    """Configuration for async rollout generation."""

    # Maximum staleness: how many training steps old a rollout can be
    max_staleness_steps: int = 5

    # Maximum number of rollouts in the buffer
    max_buffer_size: int = 1000

    # Number of concurrent generation slots
    max_generation_slots: int = 4

    # Minimum batch size to release for training
    min_batch_size: int = 8

    # Timeout for waiting on batch availability (seconds)
    batch_timeout: float = 30.0

    # Whether to track consumed data UIDs for checkpoint resume
    track_consumed_uids: bool = True


@dataclass
class StampedTrajectory:
    """A trajectory with generation metadata for staleness tracking."""

    trajectory: Trajectory
    generation_step: int
    generation_time: float
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))


class StalenessManager:
    """
    Controls generation capacity based on staleness bounds.

    Prevents trajectories from becoming too stale relative to the current
    training step. Dynamically adjusts how many generation workers can
    submit new trajectories.
    """

    def __init__(
        self,
        max_staleness_steps: int = 5,
        max_generation_slots: int = 4,
    ):
        self._max_staleness = max_staleness_steps
        self._max_slots = max_generation_slots
        self._current_training_step = 0
        self._in_flight: int = 0
        self._lock = threading.Lock()
        self._slot_available = threading.Condition(self._lock)

    @property
    def current_step(self) -> int:
        with self._lock:
            return self._current_training_step

    def advance_training_step(self) -> None:
        """Called after each training step to increase capacity."""
        with self._slot_available:
            self._current_training_step += 1
            self._slot_available.notify_all()

    def acquire_slot(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a generation slot. Blocks until capacity is available.

        Returns True if a slot was acquired, False if timed out.
        """
        with self._slot_available:
            deadline = time.monotonic() + timeout if timeout else None

            while self._in_flight >= self._available_capacity():
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False

                self._slot_available.wait(timeout=remaining)

            self._in_flight += 1
            return True

    def release_slot(self) -> None:
        """Release a generation slot after rollout is submitted."""
        with self._slot_available:
            self._in_flight = max(0, self._in_flight - 1)
            self._slot_available.notify_all()

    def _available_capacity(self) -> int:
        """
        Compute available generation capacity.

        Capacity = max_slots - in_flight, bounded by staleness.
        If too many stale items would accumulate, reduce capacity.
        """
        return max(0, self._max_slots - self._in_flight)

    def is_too_stale(self, generation_step: int) -> bool:
        """Check if a trajectory generated at the given step is too stale."""
        with self._lock:
            return (self._current_training_step - generation_step) > self._max_staleness

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "training_step": self._current_training_step,
                "in_flight": self._in_flight,
                "max_slots": self._max_slots,
                "max_staleness": self._max_staleness,
            }


class AsyncRolloutBuffer:
    """
    Thread-safe async rollout buffer with staleness control.

    Generation workers push stamped trajectories into the buffer.
    The training loop pulls batches when enough are available.
    Stale trajectories are automatically evicted.
    """

    def __init__(self, config: Optional[AsyncRolloutConfig] = None):
        self.config = config or AsyncRolloutConfig()

        self._buffer: Deque[StampedTrajectory] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._consumed_uids: Set[str] = set()

        self._staleness_manager = StalenessManager(
            max_staleness_steps=self.config.max_staleness_steps,
            max_generation_slots=self.config.max_generation_slots,
        )

        # Stats
        self._total_submitted = 0
        self._total_consumed = 0
        self._total_evicted = 0

    @property
    def staleness_manager(self) -> StalenessManager:
        return self._staleness_manager

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def submit(
        self,
        trajectories: List[Trajectory],
        generation_step: int,
    ) -> int:
        """
        Submit generated trajectories to the buffer.

        Args:
            trajectories: List of generated trajectories
            generation_step: The training step at which generation started

        Returns:
            Number of trajectories actually added (excludes pre-consumed UIDs)
        """
        now = time.time()
        added = 0

        with self._not_empty:
            for traj in trajectories:
                stamped = StampedTrajectory(
                    trajectory=traj,
                    generation_step=generation_step,
                    generation_time=now,
                )

                # Skip if already consumed (checkpoint resume case)
                if (
                    self.config.track_consumed_uids
                    and stamped.uid in self._consumed_uids
                ):
                    continue

                # Enforce buffer size limit
                if len(self._buffer) >= self.config.max_buffer_size:
                    # Evict oldest
                    self._buffer.popleft()
                    self._total_evicted += 1

                self._buffer.append(stamped)
                added += 1

            self._total_submitted += added
            self._not_empty.notify_all()

        return added

    def pull_batch(
        self,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Optional[TrajectoryBatch]:
        """
        Pull a batch of trajectories for training.

        Blocks until enough trajectories are available or timeout occurs.
        Automatically evicts stale trajectories before pulling.

        Args:
            batch_size: Number of trajectories to pull (default: config.min_batch_size)
            timeout: Maximum time to wait (default: config.batch_timeout)

        Returns:
            TrajectoryBatch or None if timed out
        """
        batch_size = batch_size or self.config.min_batch_size
        timeout = timeout if timeout is not None else self.config.batch_timeout

        with self._not_empty:
            deadline = time.monotonic() + timeout

            while True:
                # Evict stale trajectories
                self._evict_stale()

                if len(self._buffer) >= batch_size:
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Timeout: return what we have if any, else None
                    if len(self._buffer) > 0:
                        break
                    return None

                self._not_empty.wait(timeout=min(remaining, 1.0))

            # Pull trajectories
            actual_size = min(batch_size, len(self._buffer))
            pulled = []
            for _ in range(actual_size):
                stamped = self._buffer.popleft()
                pulled.append(stamped)

                if self.config.track_consumed_uids:
                    self._consumed_uids.add(stamped.uid)

            self._total_consumed += len(pulled)

        # Convert to TrajectoryBatch
        trajectories = [s.trajectory for s in pulled]
        batch = TrajectoryBatch(trajectories=trajectories)

        # Store generation step info in trajectory metadata
        for stamped, traj in zip(pulled, trajectories):
            traj.metadata["generation_step"] = stamped.generation_step
            traj.metadata["generation_time"] = stamped.generation_time
            traj.metadata["uid"] = stamped.uid

        return batch

    def _evict_stale(self) -> int:
        """Evict trajectories that are too stale. Must be called with lock held."""
        evicted = 0
        while self._buffer:
            oldest = self._buffer[0]
            if self._staleness_manager.is_too_stale(oldest.generation_step):
                self._buffer.popleft()
                evicted += 1
                self._total_evicted += 1
            else:
                break
        return evicted

    def advance_step(self) -> None:
        """Notify the buffer that a training step has completed."""
        self._staleness_manager.advance_training_step()
        # Trigger eviction check
        with self._lock:
            self._evict_stale()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "total_submitted": self._total_submitted,
                "total_consumed": self._total_consumed,
                "total_evicted": self._total_evicted,
                "consumed_uids_tracked": len(self._consumed_uids),
                "staleness": self._staleness_manager.get_stats(),
            }

    def clear(self) -> None:
        """Clear all buffered trajectories."""
        with self._lock:
            self._buffer.clear()

    def get_consumed_uids(self) -> Set[str]:
        """Get set of consumed UIDs for checkpoint state."""
        with self._lock:
            return self._consumed_uids.copy()

    def load_consumed_uids(self, uids: Set[str]) -> None:
        """Load consumed UIDs from checkpoint for resume."""
        with self._lock:
            self._consumed_uids = uids.copy()


class AsyncGenerationWorker:
    """
    Worker that generates rollouts asynchronously and submits to a buffer.

    Each worker runs in its own thread, pulling prompts from a data source,
    generating trajectories, and pushing them to the shared buffer.
    """

    def __init__(
        self,
        worker_id: int,
        buffer: AsyncRolloutBuffer,
        generate_fn: Callable[[List[str]], TrajectoryBatch],
        prompt_source: Callable[[], Optional[List[str]]],
    ):
        self.worker_id = worker_id
        self.buffer = buffer
        self.generate_fn = generate_fn
        self.prompt_source = prompt_source

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._total_generated = 0

    def start(self) -> None:
        """Start the generation worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"gen-worker-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Started generation worker {self.worker_id}")

    def stop(self) -> None:
        """Stop the generation worker."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info(f"Stopped generation worker {self.worker_id}")

    def _run_loop(self) -> None:
        """Main generation loop."""
        while self._running:
            try:
                # Get prompts
                prompts = self.prompt_source()
                if prompts is None:
                    # No more data
                    break

                # Acquire generation slot (backpressure)
                acquired = self.buffer.staleness_manager.acquire_slot(timeout=5.0)
                if not acquired:
                    continue

                try:
                    # Get current step for staleness tracking
                    gen_step = self.buffer.staleness_manager.current_step

                    # Generate trajectories
                    batch = self.generate_fn(prompts)

                    # Submit to buffer
                    self.buffer.submit(batch.trajectories, generation_step=gen_step)
                    self._total_generated += len(batch)

                finally:
                    self.buffer.staleness_manager.release_slot()

            except Exception as e:
                logger.error(f"Generation worker {self.worker_id} error: {e}")
                if not self._running:
                    break
                time.sleep(1.0)  # Back off on error

    @property
    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "running": self._running,
            "total_generated": self._total_generated,
        }


class AsyncRolloutOrchestrator:
    """
    Orchestrates multiple async generation workers with a shared buffer.

    Provides a high-level interface for the training loop to:
    1. Start N generation workers
    2. Pull training batches from the buffer
    3. Advance the training step (triggers staleness eviction)
    4. Stop all workers at end of epoch
    """

    def __init__(
        self,
        config: Optional[AsyncRolloutConfig] = None,
        generate_fn: Optional[Callable[[List[str]], TrajectoryBatch]] = None,
        prompt_source: Optional[Callable[[], Optional[List[str]]]] = None,
        num_workers: int = 2,
    ):
        self.config = config or AsyncRolloutConfig()
        self.buffer = AsyncRolloutBuffer(self.config)
        self._generate_fn = generate_fn
        self._prompt_source = prompt_source
        self._num_workers = num_workers
        self._workers: List[AsyncGenerationWorker] = []

    def start(self) -> None:
        """Start all generation workers."""
        if self._generate_fn is None or self._prompt_source is None:
            raise ValueError("generate_fn and prompt_source must be set before start()")

        self._workers = []
        for i in range(self._num_workers):
            worker = AsyncGenerationWorker(
                worker_id=i,
                buffer=self.buffer,
                generate_fn=self._generate_fn,
                prompt_source=self._prompt_source,
            )
            self._workers.append(worker)
            worker.start()

        logger.info(f"Started {self._num_workers} async generation workers")

    def stop(self) -> None:
        """Stop all generation workers."""
        for worker in self._workers:
            worker.stop()
        self._workers = []
        logger.info("Stopped all async generation workers")

    def pull_batch(
        self,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Optional[TrajectoryBatch]:
        """Pull a training batch from the buffer."""
        return self.buffer.pull_batch(batch_size=batch_size, timeout=timeout)

    def advance_step(self) -> None:
        """Notify that a training step has completed."""
        self.buffer.advance_step()

    def get_stats(self) -> Dict[str, Any]:
        buffer_stats = self.buffer.get_stats()
        worker_stats = [w.get_stats() for w in self._workers]
        return {
            "buffer": buffer_stats,
            "workers": worker_stats,
            "num_workers": len(self._workers),
        }

    def save_state(self) -> Dict[str, Any]:
        """Save state for checkpointing."""
        return {
            "consumed_uids": list(self.buffer.get_consumed_uids()),
            "training_step": self.buffer.staleness_manager.current_step,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        uids = set(state.get("consumed_uids", []))
        self.buffer.load_consumed_uids(uids)
