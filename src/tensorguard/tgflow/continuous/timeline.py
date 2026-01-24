"""
Continuous Learning Timeline

Tracks the progression of each route through continuous learning stages:
INGEST → NOVELTY_CHECK → PROPOSE → TRAIN → EVAL → PACKAGE → REGISTER → PROMOTE → MONITOR → CONSOLIDATE

Each timeline event captures what happened, when, and why.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


class TimelineStage(str, Enum):
    """Stages in the continuous learning loop."""
    INGEST = "ingest"
    NOVELTY_CHECK = "novelty_check"
    PROPOSE = "propose"
    TRAIN = "train"
    EVAL = "eval"
    PACKAGE = "package"
    REGISTER = "register"
    PROMOTE = "promote"
    MONITOR = "monitor"
    CONSOLIDATE = "consolidate"
    
    # Special stages
    NO_ACTION = "no_action"       # Novelty too low, skipped training
    ROLLBACK = "rollback"         # Rollback event
    ERROR = "error"               # Something failed


class TimelineVerdict(str, Enum):
    """Outcome of a stage."""
    PENDING = "pending"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class TimelineEvent(BaseModel):
    """
    A single event in the route timeline.
    
    Captures what happened at each stage with human-readable explanations.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    route_key: str
    loop_id: str  # Groups events in the same loop execution
    
    # Stage info
    stage: TimelineStage
    verdict: TimelineVerdict = TimelineVerdict.PENDING
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Human-readable explanation
    headline: str  # Short: "Novelty check passed"
    explanation: str  # Longer: "Data changed significantly (novelty=0.65 > threshold=0.3)"
    
    # Metrics (stage-dependent)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # References
    adapter_id: Optional[str] = None
    tgsp_hash: Optional[str] = None
    evidence_id: Optional[str] = None
    
    # Privacy
    privacy_mode: str = "off"
    privacy_encrypted: bool = False
    
    # Error details (if failed)
    error: Optional[str] = None
    remediation: Optional[str] = None
    
    def complete(self, verdict: TimelineVerdict, headline: str = None, explanation: str = None):
        """Mark event as complete."""
        self.completed_at = datetime.utcnow()
        self.verdict = verdict
        if headline:
            self.headline = headline
        if explanation:
            self.explanation = explanation
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)


class LoopExecution(BaseModel):
    """
    A complete loop execution for a route.
    
    Contains all timeline events for one run_once() call.
    """
    loop_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    route_key: str
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Trigger
    trigger: str = "manual"  # "manual" | "scheduled" | "api"
    triggered_by: Optional[str] = None
    
    # Events
    events: List[TimelineEvent] = Field(default_factory=list)
    
    # Outcome
    adapter_produced: Optional[str] = None
    promoted_to_stage: Optional[str] = None
    
    # Summary
    final_verdict: TimelineVerdict = TimelineVerdict.PENDING
    summary: str = ""
    
    def add_event(
        self,
        stage: TimelineStage,
        headline: str,
        explanation: str = "",
        privacy_mode: str = "off",
    ) -> TimelineEvent:
        """Add a new event to the loop."""
        event = TimelineEvent(
            route_key=self.route_key,
            loop_id=self.loop_id,
            stage=stage,
            headline=headline,
            explanation=explanation,
            privacy_mode=privacy_mode,
            privacy_encrypted=privacy_mode == "n2he",
        )
        self.events.append(event)
        return event
    
    def get_current_stage(self) -> Optional[TimelineStage]:
        """Get the most recent stage."""
        if self.events:
            return self.events[-1].stage
        return None
    
    def complete(self, verdict: TimelineVerdict, summary: str):
        """Mark loop as complete."""
        self.completed_at = datetime.utcnow()
        self.final_verdict = verdict
        self.summary = summary


# --- Timeline Explanation Templates ---

class TimelineExplainer:
    """
    Generates human-readable explanations for timeline events.
    
    No JSON by default - plain English that operators can understand.
    """
    
    @staticmethod
    def ingest_started(feed_uri: str) -> tuple:
        return (
            "Ingesting data feed",
            f"Reading data from {feed_uri}"
        )
    
    @staticmethod
    def ingest_completed(ingest_hash: str, record_count: int = None) -> tuple:
        count_str = f" ({record_count} records)" if record_count else ""
        return (
            "Data ingested successfully",
            f"Feed snapshot captured{count_str}. Hash: {ingest_hash[:12]}..."
        )
    
    @staticmethod
    def novelty_low(novelty_score: float, threshold: float) -> tuple:
        return (
            "No update needed",
            f"Data novelty ({novelty_score:.2f}) is below threshold ({threshold:.2f}). "
            "The current stable adapter remains optimal."
        )
    
    @staticmethod
    def novelty_high(novelty_score: float, threshold: float, reasons: List[str]) -> tuple:
        reason_str = "; ".join(reasons[:3]) if reasons else "significant data changes"
        return (
            "Update proposed",
            f"Data novelty ({novelty_score:.2f}) exceeds threshold ({threshold:.2f}). "
            f"Reason: {reason_str}"
        )
    
    @staticmethod
    def training_started(base_model: str, method: str = "LoRA") -> tuple:
        return (
            f"Training {method} adapter",
            f"Fine-tuning adapter on new data using {method} method. Base model: {base_model}"
        )
    
    @staticmethod
    def training_completed(adapter_id: str, epochs: int, final_loss: float) -> tuple:
        return (
            "Training complete",
            f"Adapter {adapter_id[:8]}... trained for {epochs} epochs. Final loss: {final_loss:.4f}"
        )
    
    @staticmethod
    def eval_passed(primary: float, forgetting: float, regression: float) -> tuple:
        return (
            "Evaluation passed",
            f"Quality metric: {primary:.2%}. Forgetting: {forgetting:.2%}. Regression: {regression:.2%}. "
            "All gates passed."
        )
    
    @staticmethod
    def eval_failed_forgetting(forgetting: float, budget: float) -> tuple:
        return (
            "Evaluation failed: forgetting exceeded",
            f"Forgetting score ({forgetting:.2%}) exceeds budget ({budget:.2%}). "
            "The adapter forgot too much previous knowledge. Candidate archived."
        )
    
    @staticmethod
    def eval_failed_regression(regression: float, budget: float) -> tuple:
        return (
            "Evaluation failed: regression detected",
            f"Regression score ({regression:.2%}) exceeds budget ({budget:.2%}). "
            "Performance degraded on held-out tasks. Candidate archived."
        )
    
    @staticmethod
    def promoted_to_canary(adapter_id: str) -> tuple:
        return (
            "Promoted to CANARY",
            f"Adapter {adapter_id[:8]}... is now available for canary testing. "
            "Use X-TGFlow-Canary header to route requests."
        )
    
    @staticmethod
    def promoted_to_stable(adapter_id: str, previous_id: str = None) -> tuple:
        prev_str = f" Replaced {previous_id[:8]}..." if previous_id else ""
        return (
            "Promoted to STABLE",
            f"Adapter {adapter_id[:8]}... is now the production adapter.{prev_str} "
            "Rollback available if needed."
        )
    
    @staticmethod
    def rollback_executed(from_id: str, to_id: str) -> tuple:
        return (
            "Rollback executed",
            f"Rolled back from {from_id[:8]}... to {to_id[:8]}... "
            "Previous stable adapter is now active."
        )
    
    @staticmethod
    def consolidation_started(fast_count: int, slow_count: int) -> tuple:
        return (
            "Consolidation started",
            f"Merging {fast_count} FAST lane adapters into SLOW lane. "
            f"Current SLOW lane: {slow_count} adapters."
        )
    
    @staticmethod
    def privacy_note(mode: str) -> str:
        if mode == "n2he":
            return "Privacy Mode (N2HE): Routing decisions computed on encrypted features."
        return ""
