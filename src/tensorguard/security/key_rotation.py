"""
Key Rotation Scheduler Module.

Provides automated key rotation capabilities:
- Scheduled key rotation
- Grace periods for key transitions
- Multi-key support during rotation
- Audit logging of rotations
"""

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RotationReason(str, Enum):
    """Reasons for key rotation."""

    SCHEDULED = "scheduled"
    MANUAL = "manual"
    COMPROMISE = "compromise"
    POLICY = "policy"
    EXPIRATION = "expiration"


@dataclass
class RotationPolicy:
    """
    Key rotation policy configuration.

    Defines when and how keys should be rotated.
    """

    # Rotation interval
    rotation_interval_days: int = 90

    # Grace period (time to keep old key active after rotation)
    grace_period_hours: int = 24

    # Maximum key age (hard limit)
    max_key_age_days: int = 365

    # Minimum rotation interval (prevent too frequent rotations)
    min_rotation_interval_hours: int = 1

    # Enable automatic rotation
    auto_rotate: bool = True

    # Notification settings
    notify_days_before: List[int] = field(default_factory=lambda: [7, 3, 1])

    # Key size requirements
    min_key_size_bits: int = 256

    @classmethod
    def strict(cls) -> "RotationPolicy":
        """Create a strict rotation policy."""
        return cls(
            rotation_interval_days=30,
            grace_period_hours=4,
            max_key_age_days=90,
            min_rotation_interval_hours=4,
            auto_rotate=True,
            notify_days_before=[7, 3, 1],
            min_key_size_bits=256,
        )

    @classmethod
    def standard(cls) -> "RotationPolicy":
        """Create a standard rotation policy."""
        return cls(
            rotation_interval_days=90,
            grace_period_hours=24,
            max_key_age_days=365,
            min_rotation_interval_hours=1,
            auto_rotate=True,
            notify_days_before=[14, 7, 3, 1],
            min_key_size_bits=256,
        )

    @classmethod
    def relaxed(cls) -> "RotationPolicy":
        """Create a relaxed rotation policy (for development)."""
        return cls(
            rotation_interval_days=365,
            grace_period_hours=168,  # 1 week
            max_key_age_days=730,  # 2 years
            min_rotation_interval_hours=0,
            auto_rotate=False,
            notify_days_before=[30, 14, 7],
            min_key_size_bits=128,
        )


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""

    key_id: str
    created_at: datetime
    rotated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    status: str = "active"  # active, rotating, deprecated, revoked
    version: int = 1
    algorithm: str = "AES-256-GCM"
    purpose: str = "encryption"
    rotation_reason: Optional[RotationReason] = None

    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def days_until_expiration(self) -> Optional[int]:
        """Get days until expiration."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)

    def age_days(self) -> int:
        """Get key age in days."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.days


class KeyRotationScheduler:
    """
    Manages automatic key rotation.

    Features:
    - Scheduled automatic rotation
    - Grace period handling
    - Multi-key support during transitions
    - Rotation notifications
    - Audit logging
    """

    def __init__(
        self,
        policy: Optional[RotationPolicy] = None,
        key_generator: Optional[Callable[[], bytes]] = None,
    ):
        """
        Initialize key rotation scheduler.

        Args:
            policy: Rotation policy
            key_generator: Function to generate new keys
        """
        self.policy = policy or RotationPolicy.standard()
        self.key_generator = key_generator or (lambda: secrets.token_bytes(32))

        # Key storage
        self._keys: Dict[str, KeyMetadata] = {}
        self._key_data: Dict[str, bytes] = {}  # Actual key bytes
        self._active_key_id: Optional[str] = None

        # Callbacks
        self._rotation_callbacks: List[Callable[[str, str], None]] = []
        self._notification_callbacks: List[Callable[[str, int], None]] = []

        # Background task
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

    def add_rotation_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """
        Add a callback for key rotation events.

        Callback receives (old_key_id, new_key_id).
        """
        self._rotation_callbacks.append(callback)

    def add_notification_callback(
        self,
        callback: Callable[[str, int], None],
    ) -> None:
        """
        Add a callback for rotation notifications.

        Callback receives (key_id, days_until_rotation).
        """
        self._notification_callbacks.append(callback)

    async def start(self) -> None:
        """Start the rotation scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Key rotation scheduler started")

    async def stop(self) -> None:
        """Stop the rotation scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        logger.info("Key rotation scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Background loop for checking and performing rotations."""
        while self._running:
            try:
                await self._check_rotations()
                await self._check_notifications()
                await self._cleanup_deprecated_keys()

                # Check every hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rotation scheduler: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _check_rotations(self) -> None:
        """Check if any keys need rotation."""
        now = datetime.now(timezone.utc)

        for key_id, metadata in list(self._keys.items()):
            if metadata.status != "active":
                continue

            # Check if rotation is due
            next_rotation = metadata.created_at + timedelta(
                days=self.policy.rotation_interval_days
            )

            if now >= next_rotation:
                if self.policy.auto_rotate:
                    await self.rotate_key(key_id, RotationReason.SCHEDULED)
                else:
                    logger.warning(
                        f"Key {key_id} due for rotation (auto-rotate disabled)"
                    )

            # Check max age
            if metadata.age_days() >= self.policy.max_key_age_days:
                logger.warning(f"Key {key_id} has exceeded max age, forcing rotation")
                await self.rotate_key(key_id, RotationReason.EXPIRATION)

    async def _check_notifications(self) -> None:
        """Check and send rotation notifications."""
        now = datetime.now(timezone.utc)

        for key_id, metadata in self._keys.items():
            if metadata.status != "active":
                continue

            next_rotation = metadata.created_at + timedelta(
                days=self.policy.rotation_interval_days
            )
            days_until = (next_rotation - now).days

            if days_until in self.policy.notify_days_before:
                for callback in self._notification_callbacks:
                    try:
                        callback(key_id, days_until)
                    except Exception as e:
                        logger.error(f"Notification callback error: {e}")

                logger.info(f"Key {key_id} will be rotated in {days_until} days")

    async def _cleanup_deprecated_keys(self) -> None:
        """Remove deprecated keys after grace period."""
        now = datetime.now(timezone.utc)
        grace_period = timedelta(hours=self.policy.grace_period_hours)

        for key_id, metadata in list(self._keys.items()):
            if metadata.status == "deprecated" and metadata.rotated_at:
                if now > metadata.rotated_at + grace_period:
                    await self.revoke_key(key_id)

    def create_key(
        self,
        key_id: Optional[str] = None,
        purpose: str = "encryption",
        algorithm: str = "AES-256-GCM",
        set_active: bool = True,
    ) -> str:
        """
        Create a new managed key.

        Args:
            key_id: Optional key identifier
            purpose: Key purpose
            algorithm: Key algorithm
            set_active: Set as active key

        Returns:
            Key identifier
        """
        if key_id is None:
            key_id = f"key-{secrets.token_hex(8)}"

        now = datetime.now(timezone.utc)

        # Generate key
        key_bytes = self.key_generator()

        # Validate key size
        key_size_bits = len(key_bytes) * 8
        if key_size_bits < self.policy.min_key_size_bits:
            raise ValueError(
                f"Key size {key_size_bits} bits is below minimum "
                f"{self.policy.min_key_size_bits} bits"
            )

        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            created_at=now,
            algorithm=algorithm,
            purpose=purpose,
            status="active",
            version=1,
        )

        self._keys[key_id] = metadata
        self._key_data[key_id] = key_bytes

        if set_active:
            self._active_key_id = key_id

        logger.info(
            f"Created key {key_id}: algorithm={algorithm}, purpose={purpose}",
            extra={"key_id": key_id, "algorithm": algorithm},
        )

        return key_id

    async def rotate_key(
        self,
        key_id: str,
        reason: RotationReason = RotationReason.MANUAL,
    ) -> str:
        """
        Rotate a key.

        Args:
            key_id: Key to rotate
            reason: Reason for rotation

        Returns:
            New key identifier
        """
        old_metadata = self._keys.get(key_id)
        if old_metadata is None:
            raise ValueError(f"Key not found: {key_id}")

        # Check minimum rotation interval
        if old_metadata.rotated_at:
            min_interval = timedelta(hours=self.policy.min_rotation_interval_hours)
            if datetime.now(timezone.utc) - old_metadata.rotated_at < min_interval:
                raise ValueError("Minimum rotation interval not met")

        # Create new key
        new_key_id = self.create_key(
            purpose=old_metadata.purpose,
            algorithm=old_metadata.algorithm,
            set_active=True,
        )

        new_metadata = self._keys[new_key_id]
        new_metadata.version = old_metadata.version + 1

        # Mark old key as deprecated
        old_metadata.status = "deprecated"
        old_metadata.rotated_at = datetime.now(timezone.utc)
        old_metadata.rotation_reason = reason

        # Call rotation callbacks
        for callback in self._rotation_callbacks:
            try:
                callback(key_id, new_key_id)
            except Exception as e:
                logger.error(f"Rotation callback error: {e}")

        logger.info(
            f"Rotated key {key_id} -> {new_key_id}: reason={reason.value}",
            extra={
                "old_key_id": key_id,
                "new_key_id": new_key_id,
                "reason": reason.value,
            },
        )

        return new_key_id

    async def revoke_key(self, key_id: str) -> None:
        """
        Revoke a key (permanent).

        Args:
            key_id: Key to revoke
        """
        metadata = self._keys.get(key_id)
        if metadata is None:
            return

        metadata.status = "revoked"

        # Securely clear key data
        if key_id in self._key_data:
            key_bytes = self._key_data.pop(key_id)
            # Attempt to zero the memory
            try:
                for i in range(len(key_bytes)):
                    key_bytes = key_bytes[:i] + b"\x00" + key_bytes[i + 1 :]
            except TypeError:
                pass  # Bytes are immutable, this is best-effort

        if self._active_key_id == key_id:
            # Find another active key
            for other_id, other_meta in self._keys.items():
                if other_meta.status == "active":
                    self._active_key_id = other_id
                    break
            else:
                self._active_key_id = None

        logger.info(f"Revoked key {key_id}")

    def get_active_key(self) -> Optional[tuple[str, bytes]]:
        """
        Get the current active key.

        Returns:
            Tuple of (key_id, key_bytes) or None
        """
        if self._active_key_id is None:
            return None

        key_bytes = self._key_data.get(self._active_key_id)
        if key_bytes is None:
            return None

        return self._active_key_id, key_bytes

    def get_key(self, key_id: str) -> Optional[bytes]:
        """
        Get a key by ID.

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None
        """
        metadata = self._keys.get(key_id)
        if metadata is None:
            return None

        if metadata.status == "revoked":
            return None

        return self._key_data.get(key_id)

    def get_all_valid_keys(self) -> Dict[str, bytes]:
        """
        Get all valid keys (for decryption during rotation).

        Returns:
            Dictionary of key_id -> key_bytes
        """
        valid_keys = {}

        for key_id, metadata in self._keys.items():
            if metadata.status in ("active", "deprecated"):
                key_bytes = self._key_data.get(key_id)
                if key_bytes:
                    valid_keys[key_id] = key_bytes

        return valid_keys

    def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get metadata for a key."""
        return self._keys.get(key_id)

    def list_keys(self, status: Optional[str] = None) -> List[KeyMetadata]:
        """
        List managed keys.

        Args:
            status: Filter by status

        Returns:
            List of key metadata
        """
        keys = list(self._keys.values())

        if status:
            keys = [k for k in keys if k.status == status]

        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    def get_rotation_status(self) -> Dict[str, Any]:
        """Get rotation status summary."""
        active_keys = [k for k in self._keys.values() if k.status == "active"]
        deprecated_keys = [k for k in self._keys.values() if k.status == "deprecated"]

        next_rotation = None
        if active_keys:
            for key in active_keys:
                rotation_date = key.created_at + timedelta(
                    days=self.policy.rotation_interval_days
                )
                if next_rotation is None or rotation_date < next_rotation:
                    next_rotation = rotation_date

        return {
            "active_keys": len(active_keys),
            "deprecated_keys": len(deprecated_keys),
            "revoked_keys": len([k for k in self._keys.values() if k.status == "revoked"]),
            "auto_rotate_enabled": self.policy.auto_rotate,
            "rotation_interval_days": self.policy.rotation_interval_days,
            "next_rotation": next_rotation.isoformat() if next_rotation else None,
            "scheduler_running": self._running,
        }
