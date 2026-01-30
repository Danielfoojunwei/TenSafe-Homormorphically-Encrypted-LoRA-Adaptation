"""
TG-Tinker SDK TrainingClient module.

This module provides the TrainingClient class for training loop control.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .futures import FutureHandle
from .schemas import (
    BatchData,
    DPMetrics,
    ForwardBackwardRequest,
    LoadStateRequest,
    LoadStateResult,
    OperationType,
    OptimStepRequest,
    SampleRequest,
    SampleResult,
    SaveStateRequest,
    SaveStateResult,
    TrainingClientStatus,
    TrainingConfig,
)

if TYPE_CHECKING:
    from .client import ServiceClient


class TrainingClient:
    """
    Client for controlling a training loop.

    TrainingClient exposes primitives for fine-tuning models:
    - forward_backward: Compute loss and gradients
    - optim_step: Apply optimizer update
    - sample: Generate text from current model
    - save_state: Save encrypted checkpoint
    - load_state: Load checkpoint

    forward_backward and optim_step return FutureHandle objects for async
    execution. You can overlap operations before waiting for results.

    Example:
        >>> tc = service.create_training_client(config)
        >>> for batch in dataloader:
        ...     fb_future = tc.forward_backward(batch)
        ...     opt_future = tc.optim_step()
        ...     result = fb_future.result()
        ...     print(f"Loss: {result.loss}")
    """

    def __init__(
        self,
        training_client_id: str,
        client: "ServiceClient",
        config: TrainingConfig,
        step: int = 0,
        dp_metrics: Optional[DPMetrics] = None,
    ):
        """
        Initialize a TrainingClient.

        Args:
            training_client_id: Unique identifier for this training client
            client: ServiceClient for API communication
            config: Training configuration
            step: Current training step
            dp_metrics: Current DP metrics (if DP enabled)
        """
        self._id = training_client_id
        self._client = client
        self._config = config
        self._step = step
        self._status = TrainingClientStatus.READY
        self._dp_metrics = dp_metrics

    @property
    def id(self) -> str:
        """Unique identifier for this training client."""
        return self._id

    @property
    def training_client_id(self) -> str:
        """Alias for id property."""
        return self._id

    @property
    def config(self) -> TrainingConfig:
        """Training configuration."""
        return self._config

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    @property
    def status(self) -> TrainingClientStatus:
        """Current status of the training client."""
        return self._status

    @property
    def dp_enabled(self) -> bool:
        """Whether differential privacy is enabled."""
        return self._config.dp_config is not None and self._config.dp_config.enabled

    @property
    def dp_metrics(self) -> Optional[DPMetrics]:
        """Current differential privacy metrics."""
        return self._dp_metrics

    def forward_backward(
        self,
        batch: Union[BatchData, Dict[str, Any]],
        batch_hash: Optional[str] = None,
    ) -> FutureHandle:
        """
        Queue a forward-backward pass computation.

        Computes the forward pass through the model and backpropagates
        to compute gradients. Gradients are accumulated (not applied).

        This operation is queued and returns immediately with a FutureHandle.
        Use the handle to wait for the result or check status.

        Args:
            batch: Training batch with input_ids, attention_mask, and optionally labels.
                  Can be a BatchData object or a dict.
            batch_hash: Optional client-side hash for verification

        Returns:
            FutureHandle for the async operation

        Example:
            >>> future = tc.forward_backward({
            ...     "input_ids": [[1, 2, 3], [4, 5, 6]],
            ...     "attention_mask": [[1, 1, 1], [1, 1, 1]],
            ...     "labels": [[2, 3, -100], [5, 6, -100]]
            ... })
            >>> result = future.result()
            >>> print(f"Loss: {result.loss}")
        """
        # Convert dict to BatchData if needed
        if isinstance(batch, dict):
            batch = BatchData(**batch)

        request = ForwardBackwardRequest(batch=batch, batch_hash=batch_hash)

        response = self._client._post_forward_backward(self._id, request)

        return FutureHandle(
            future_id=response.future_id,
            client=self._client,
            training_client_id=self._id,
            operation=OperationType.FORWARD_BACKWARD,
            poll_interval=self._client._config.poll_interval,
        )

    def optim_step(self, apply_dp_noise: bool = True) -> FutureHandle:
        """
        Queue an optimizer step.

        Applies the accumulated gradients using the configured optimizer.
        If DP is enabled and apply_dp_noise is True, noise will be added
        before the update.

        This operation is queued and returns immediately with a FutureHandle.

        Args:
            apply_dp_noise: If True and DP is enabled, apply DP noise

        Returns:
            FutureHandle for the async operation

        Example:
            >>> fb_future = tc.forward_backward(batch)
            >>> opt_future = tc.optim_step()  # Can call before waiting
            >>> fb_result = fb_future.result()
            >>> opt_result = opt_future.result()
            >>> print(f"New step: {opt_result.step}")
        """
        request = OptimStepRequest(apply_dp_noise=apply_dp_noise)

        response = self._client._post_optim_step(self._id, request)

        return FutureHandle(
            future_id=response.future_id,
            client=self._client,
            training_client_id=self._id,
            operation=OperationType.OPTIM_STEP,
            poll_interval=self._client._config.poll_interval,
        )

    def sample(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
    ) -> SampleResult:
        """
        Generate samples from the current model state.

        This is a synchronous operation that generates text completions
        for the given prompts.

        Args:
            prompts: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            stop_sequences: Sequences that trigger generation stop

        Returns:
            SampleResult containing completions for all prompts

        Example:
            >>> result = tc.sample("Once upon a time", max_tokens=100)
            >>> print(result.samples[0].completion)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        request = SampleRequest(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences or [],
        )

        return self._client._post_sample(self._id, request)

    def save_state(
        self,
        include_optimizer: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SaveStateResult:
        """
        Save the current training state as an encrypted checkpoint.

        Creates an encrypted artifact containing:
        - Model weights (or LoRA adapter weights)
        - Optionally, optimizer state
        - Training step and configuration
        - DP metrics if applicable

        The checkpoint is encrypted with the tenant's key and stored
        in the artifact store.

        Args:
            include_optimizer: If True, include optimizer state in checkpoint
            metadata: Optional custom metadata to include

        Returns:
            SaveStateResult with artifact information

        Example:
            >>> result = tc.save_state(metadata={"notes": "After epoch 1"})
            >>> print(f"Saved artifact: {result.artifact_id}")
        """
        request = SaveStateRequest(
            include_optimizer=include_optimizer,
            metadata=metadata or {},
        )

        result = self._client._post_save_state(self._id, request)

        return result

    def load_state(self, artifact_id: str) -> LoadStateResult:
        """
        Load training state from an encrypted checkpoint.

        Loads the model weights, optimizer state, and training step
        from a previously saved checkpoint artifact.

        Args:
            artifact_id: ID of the checkpoint artifact to load

        Returns:
            LoadStateResult with updated training client state

        Example:
            >>> result = tc.load_state("art-xxx-yyy")
            >>> print(f"Loaded step: {result.step}")
        """
        request = LoadStateRequest(artifact_id=artifact_id)

        result = self._client._post_load_state(self._id, request)

        # Update local state
        self._step = result.step
        self._status = result.status

        return result

    def refresh(self) -> "TrainingClient":
        """
        Refresh the training client state from the server.

        Updates step, status, and DP metrics from the server.

        Returns:
            self (for method chaining)
        """
        info = self._client.get_training_client(self._id)
        self._step = info.step
        self._status = info.status
        self._dp_metrics = info.dp_metrics
        return self

    def __repr__(self) -> str:
        return (
            f"TrainingClient(id={self._id!r}, "
            f"model={self._config.model_ref!r}, "
            f"step={self._step}, "
            f"status={self._status.value!r})"
        )
