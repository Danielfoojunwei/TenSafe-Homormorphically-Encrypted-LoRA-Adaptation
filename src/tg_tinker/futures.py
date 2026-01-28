"""
TG-Tinker SDK futures module.

This module provides the FutureHandle class for async operation management.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from .exceptions import (
    FutureCancelledError,
    FutureFailedError,
    FutureTimeoutError,
)
from .schemas import FutureStatus, OperationType

if TYPE_CHECKING:
    from .client import ServiceClient


class FutureHandle:
    """
    Handle for an asynchronous operation.

    FutureHandle represents a pending or completed async operation.
    Use `.result()` to wait for and retrieve the result, or `.status()`
    to check the current state without blocking.

    Example:
        >>> future = training_client.forward_backward(batch)
        >>> print(future.status())  # FutureStatus.PENDING
        >>> result = future.result(timeout=300)  # Blocks until complete
        >>> print(result.loss)
    """

    def __init__(
        self,
        future_id: str,
        client: "ServiceClient",
        training_client_id: str,
        operation: OperationType,
        poll_interval: float = 1.0,
    ):
        """
        Initialize a FutureHandle.

        Args:
            future_id: Unique identifier for this future
            client: ServiceClient for polling status
            training_client_id: ID of the associated training client
            operation: Type of operation this future represents
            poll_interval: Interval between status polls in seconds
        """
        self._future_id = future_id
        self._client = client
        self._training_client_id = training_client_id
        self._operation = operation
        self._poll_interval = poll_interval
        self._status: FutureStatus = FutureStatus.PENDING
        self._result: Optional[Any] = None
        self._error: Optional[str] = None
        self._cached = False

    @property
    def future_id(self) -> str:
        """Unique identifier for this future."""
        return self._future_id

    @property
    def training_client_id(self) -> str:
        """ID of the associated training client."""
        return self._training_client_id

    @property
    def operation(self) -> OperationType:
        """Type of operation this future represents."""
        return self._operation

    def status(self, refresh: bool = True) -> FutureStatus:
        """
        Get the current status of the future.

        Args:
            refresh: If True, poll the server for latest status.
                    If False, return cached status.

        Returns:
            Current FutureStatus
        """
        if self._cached and not refresh:
            return self._status

        if self._status in (
            FutureStatus.COMPLETED,
            FutureStatus.FAILED,
            FutureStatus.CANCELLED,
        ):
            return self._status

        # Poll server for status
        response = self._client._get_future_status(self._future_id)
        self._status = response.status

        if self._status in (
            FutureStatus.COMPLETED,
            FutureStatus.FAILED,
            FutureStatus.CANCELLED,
        ):
            self._cached = True

        return self._status

    def done(self) -> bool:
        """
        Check if the future is complete.

        Returns:
            True if the future is completed, failed, or cancelled
        """
        status = self.status(refresh=True)
        return status in (
            FutureStatus.COMPLETED,
            FutureStatus.FAILED,
            FutureStatus.CANCELLED,
        )

    def result(self, timeout: Optional[float] = None) -> Any:
        """
        Wait for and return the result of the operation.

        Blocks until the operation completes or timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            The operation result (type depends on operation)

        Raises:
            FutureTimeoutError: If timeout is reached before completion
            FutureCancelledError: If the future was cancelled
            FutureFailedError: If the operation failed
        """
        start_time = time.time()

        while True:
            status = self.status(refresh=True)

            if status == FutureStatus.COMPLETED:
                # Fetch the result
                if self._result is None:
                    result_response = self._client._get_future_result(self._future_id)
                    self._result = result_response.result
                return self._result

            if status == FutureStatus.FAILED:
                if self._error is None:
                    result_response = self._client._get_future_result(self._future_id)
                    self._error = result_response.error or "Unknown error"
                raise FutureFailedError(
                    future_id=self._future_id,
                    error_message=self._error,
                )

            if status == FutureStatus.CANCELLED:
                raise FutureCancelledError(future_id=self._future_id)

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise FutureTimeoutError(
                        future_id=self._future_id,
                        timeout=timeout,
                    )
                # Adjust poll interval to not exceed remaining time
                remaining = timeout - elapsed
                sleep_time = min(self._poll_interval, remaining)
            else:
                sleep_time = self._poll_interval

            time.sleep(sleep_time)

    def cancel(self) -> bool:
        """
        Attempt to cancel the future.

        Cancellation may not be possible if the operation has already started
        or completed.

        Returns:
            True if cancellation was successful, False otherwise
        """
        if self._status in (
            FutureStatus.COMPLETED,
            FutureStatus.FAILED,
            FutureStatus.CANCELLED,
        ):
            return self._status == FutureStatus.CANCELLED

        success = self._client._cancel_future(self._future_id)
        if success:
            self._status = FutureStatus.CANCELLED
            self._cached = True
        return success

    def exception(self) -> Optional[Exception]:
        """
        Return the exception if the future failed.

        Does not block; returns None if the future is not yet complete
        or completed successfully.

        Returns:
            The exception that caused failure, or None
        """
        status = self.status(refresh=False)

        if status == FutureStatus.FAILED:
            if self._error is None:
                result_response = self._client._get_future_result(self._future_id)
                self._error = result_response.error or "Unknown error"
            return FutureFailedError(
                future_id=self._future_id,
                error_message=self._error,
            )

        if status == FutureStatus.CANCELLED:
            return FutureCancelledError(future_id=self._future_id)

        return None

    def add_done_callback(self, callback: Callable[["FutureHandle"], None]) -> None:
        """
        Add a callback to be called when the future completes.

        Note: In this synchronous implementation, the callback is called
        in a blocking manner after polling confirms completion.

        Args:
            callback: Function to call with this future as argument
        """
        # Simple implementation: spawn a thread to poll and call back
        import threading

        def _poll_and_callback():
            while not self.done():
                time.sleep(self._poll_interval)
            callback(self)

        thread = threading.Thread(target=_poll_and_callback, daemon=True)
        thread.start()

    def __repr__(self) -> str:
        return (
            f"FutureHandle(future_id={self._future_id!r}, "
            f"operation={self._operation.value!r}, "
            f"status={self._status.value!r})"
        )
