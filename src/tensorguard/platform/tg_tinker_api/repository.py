"""
TG-Tinker Repository Layer.

Database operations for training clients, futures, artifacts, and DP trainers.
Provides a clean abstraction over SQLModel queries.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from .models import (
    TinkerArtifact,
    TinkerFuture,
    TinkerTrainingClient,
    generate_artifact_id,
    generate_future_id,
    generate_tc_id,
)

logger = logging.getLogger(__name__)


class TrainingClientRepository:
    """Repository for TinkerTrainingClient operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        tenant_id: str,
        model_ref: str,
        config_json: Dict[str, Any],
        dp_enabled: bool = False,
    ) -> TinkerTrainingClient:
        """Create a new training client."""
        tc = TinkerTrainingClient(
            id=generate_tc_id(),
            tenant_id=tenant_id,
            model_ref=model_ref,
            status="ready",
            step=0,
            config_json=config_json,
            dp_enabled=dp_enabled,
        )
        self.session.add(tc)
        self.session.commit()
        self.session.refresh(tc)
        return tc

    def get(self, tc_id: str) -> Optional[TinkerTrainingClient]:
        """Get a training client by ID."""
        return self.session.get(TinkerTrainingClient, tc_id)

    def get_for_tenant(self, tc_id: str, tenant_id: str) -> Optional[TinkerTrainingClient]:
        """Get a training client by ID, verifying tenant ownership."""
        tc = self.session.get(TinkerTrainingClient, tc_id)
        if tc and tc.tenant_id == tenant_id:
            return tc
        return None

    def list_for_tenant(self, tenant_id: str) -> List[TinkerTrainingClient]:
        """List all training clients for a tenant."""
        statement = select(TinkerTrainingClient).where(
            TinkerTrainingClient.tenant_id == tenant_id
        )
        return list(self.session.exec(statement))

    def update(self, tc: TinkerTrainingClient) -> TinkerTrainingClient:
        """Update a training client."""
        tc.updated_at = datetime.utcnow()
        self.session.add(tc)
        self.session.commit()
        self.session.refresh(tc)
        return tc

    def delete(self, tc_id: str) -> bool:
        """Delete a training client."""
        tc = self.session.get(TinkerTrainingClient, tc_id)
        if tc:
            self.session.delete(tc)
            self.session.commit()
            return True
        return False


class FutureRepository:
    """Repository for TinkerFuture operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        training_client_id: str,
        tenant_id: str,
        operation: str,
        request_hash: str,
        request_size_bytes: int = 0,
        priority: int = 0,
    ) -> TinkerFuture:
        """Create a new future."""
        future = TinkerFuture(
            id=generate_future_id(),
            training_client_id=training_client_id,
            tenant_id=tenant_id,
            operation=operation,
            status="pending",
            request_hash=request_hash,
            request_size_bytes=request_size_bytes,
            priority=priority,
        )
        self.session.add(future)
        self.session.commit()
        self.session.refresh(future)
        return future

    def get(self, future_id: str) -> Optional[TinkerFuture]:
        """Get a future by ID."""
        return self.session.get(TinkerFuture, future_id)

    def get_for_tenant(self, future_id: str, tenant_id: str) -> Optional[TinkerFuture]:
        """Get a future by ID, verifying tenant ownership."""
        future = self.session.get(TinkerFuture, future_id)
        if future and future.tenant_id == tenant_id:
            return future
        return None

    def list_pending(self, limit: int = 100) -> List[TinkerFuture]:
        """List pending futures for processing."""
        statement = (
            select(TinkerFuture)
            .where(TinkerFuture.status == "pending")
            .order_by(TinkerFuture.priority.desc(), TinkerFuture.created_at)
            .limit(limit)
        )
        return list(self.session.exec(statement))

    def update_status(
        self,
        future_id: str,
        status: str,
        result_json: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TinkerFuture]:
        """Update future status."""
        future = self.session.get(TinkerFuture, future_id)
        if future:
            future.status = status
            if status == "running" and future.started_at is None:
                future.started_at = datetime.utcnow()
            if status in ("completed", "failed", "cancelled"):
                future.completed_at = datetime.utcnow()
            if result_json is not None:
                future.result_json = result_json
            if error_message is not None:
                future.error_message = error_message
            self.session.add(future)
            self.session.commit()
            self.session.refresh(future)
        return future


class ArtifactRepository:
    """Repository for TinkerArtifact operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        training_client_id: str,
        tenant_id: str,
        artifact_type: str,
        storage_key: str,
        size_bytes: int,
        encryption_algorithm: str,
        encryption_key_id: str,
        encryption_nonce: str,
        content_hash: str,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> TinkerArtifact:
        """Create a new artifact record."""
        artifact = TinkerArtifact(
            id=generate_artifact_id(),
            training_client_id=training_client_id,
            tenant_id=tenant_id,
            artifact_type=artifact_type,
            storage_key=storage_key,
            size_bytes=size_bytes,
            encryption_algorithm=encryption_algorithm,
            encryption_key_id=encryption_key_id,
            encryption_nonce=encryption_nonce,
            content_hash=content_hash,
            metadata_json=metadata_json or {},
        )
        self.session.add(artifact)
        self.session.commit()
        self.session.refresh(artifact)
        return artifact

    def get(self, artifact_id: str) -> Optional[TinkerArtifact]:
        """Get an artifact by ID."""
        return self.session.get(TinkerArtifact, artifact_id)

    def get_for_tenant(self, artifact_id: str, tenant_id: str) -> Optional[TinkerArtifact]:
        """Get an artifact by ID, verifying tenant ownership."""
        artifact = self.session.get(TinkerArtifact, artifact_id)
        if artifact and artifact.tenant_id == tenant_id:
            return artifact
        return None

    def list_for_training_client(
        self, training_client_id: str, artifact_type: Optional[str] = None
    ) -> List[TinkerArtifact]:
        """List artifacts for a training client."""
        statement = select(TinkerArtifact).where(
            TinkerArtifact.training_client_id == training_client_id
        )
        if artifact_type:
            statement = statement.where(TinkerArtifact.artifact_type == artifact_type)
        return list(self.session.exec(statement))

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact record (not the blob)."""
        artifact = self.session.get(TinkerArtifact, artifact_id)
        if artifact:
            self.session.delete(artifact)
            self.session.commit()
            return True
        return False


class DPTrainerState:
    """
    Manages DP trainer state.

    Note: The actual DPTrainer computation happens in-process, but we
    persist the state (epsilon spent, steps taken) to the database via
    the training client record.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_dp_metrics(self, training_client_id: str) -> Optional[Dict[str, Any]]:
        """Get DP metrics from training client record."""
        tc = self.session.get(TinkerTrainingClient, training_client_id)
        if tc and tc.dp_enabled:
            return {
                "total_epsilon": tc.dp_total_epsilon,
                "delta": tc.dp_total_delta,
                "num_steps": tc.step,
            }
        return None

    def update_dp_metrics(
        self,
        training_client_id: str,
        total_epsilon: float,
        total_delta: float,
    ) -> bool:
        """Update DP metrics on training client record."""
        tc = self.session.get(TinkerTrainingClient, training_client_id)
        if tc and tc.dp_enabled:
            tc.dp_total_epsilon = total_epsilon
            tc.dp_total_delta = total_delta
            tc.updated_at = datetime.utcnow()
            self.session.add(tc)
            self.session.commit()
            return True
        return False
