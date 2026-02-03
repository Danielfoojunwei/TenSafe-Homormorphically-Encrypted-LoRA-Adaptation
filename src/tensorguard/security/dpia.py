"""
Data Protection Impact Assessment (DPIA) Workflow Module.

Implements GDPR Article 35 DPIA requirements:
- Systematic assessment of processing operations
- Risk identification and mitigation
- Necessity and proportionality evaluation
- Automated risk scoring

Compliance Requirements:
- GDPR Article 35: Data Protection Impact Assessment
- ISO 27701 7.2.5: Privacy impact assessment
- ISO 27001 A.8.2: Information classification

Usage:
    from tensorguard.security.dpia import (
        DPIAManager,
        DPIAAssessment,
        RiskLevel,
        create_dpia_for_processing,
    )
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class DPIAStatus(str, Enum):
    """DPIA workflow status."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_CONSULTATION = "requires_consultation"  # Article 36
    ARCHIVED = "archived"


class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingBasis(str, Enum):
    """Legal basis for processing (GDPR Article 6)."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_INTEREST = "public_interest"
    LEGITIMATE_INTEREST = "legitimate_interest"


class DataCategory(str, Enum):
    """Categories of personal data."""

    BASIC_IDENTITY = "basic_identity"  # Name, address
    CONTACT = "contact"  # Email, phone
    IDENTIFICATION = "identification"  # ID numbers, SSN
    FINANCIAL = "financial"  # Payment, bank details
    HEALTH = "health"  # Special category
    BIOMETRIC = "biometric"  # Special category
    GENETIC = "genetic"  # Special category
    RACIAL_ETHNIC = "racial_ethnic"  # Special category
    POLITICAL = "political"  # Special category
    RELIGIOUS = "religious"  # Special category
    SEXUAL_ORIENTATION = "sexual_orientation"  # Special category
    CRIMINAL = "criminal"  # Article 10
    LOCATION = "location"
    BEHAVIORAL = "behavioral"  # Profiling data
    CHILDREN = "children"  # Under 16


# Categories requiring mandatory DPIA (Article 35(3))
MANDATORY_DPIA_CATEGORIES = {
    DataCategory.HEALTH,
    DataCategory.BIOMETRIC,
    DataCategory.GENETIC,
    DataCategory.RACIAL_ETHNIC,
    DataCategory.POLITICAL,
    DataCategory.RELIGIOUS,
    DataCategory.SEXUAL_ORIENTATION,
    DataCategory.CRIMINAL,
    DataCategory.CHILDREN,
}


class ProcessingType(str, Enum):
    """Types of processing operations."""

    COLLECTION = "collection"
    STORAGE = "storage"
    ANALYSIS = "analysis"
    PROFILING = "profiling"
    AUTOMATED_DECISION = "automated_decision"
    LARGE_SCALE = "large_scale"
    SYSTEMATIC_MONITORING = "systematic_monitoring"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"
    INNOVATIVE_TECHNOLOGY = "innovative_technology"
    ENCRYPTION = "encryption"
    ANONYMIZATION = "anonymization"


# Processing types requiring mandatory DPIA
MANDATORY_DPIA_PROCESSING = {
    ProcessingType.PROFILING,
    ProcessingType.AUTOMATED_DECISION,
    ProcessingType.LARGE_SCALE,
    ProcessingType.SYSTEMATIC_MONITORING,
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class RiskAssessment:
    """Individual risk assessment."""

    risk_id: str
    description: str
    likelihood: int  # 1-5
    impact: int  # 1-5
    inherent_risk: RiskLevel = field(init=False)
    mitigations: List[str] = field(default_factory=list)
    residual_likelihood: Optional[int] = None
    residual_impact: Optional[int] = None
    residual_risk: Optional[RiskLevel] = None
    owner: Optional[str] = None
    status: str = "open"

    def __post_init__(self):
        """Calculate inherent risk level."""
        self.inherent_risk = self._calculate_risk_level(self.likelihood, self.impact)
        if self.residual_likelihood and self.residual_impact:
            self.residual_risk = self._calculate_risk_level(
                self.residual_likelihood, self.residual_impact
            )

    @staticmethod
    def _calculate_risk_level(likelihood: int, impact: int) -> RiskLevel:
        """Calculate risk level from likelihood and impact."""
        score = likelihood * impact
        if score <= 4:
            return RiskLevel.LOW
        elif score <= 9:
            return RiskLevel.MEDIUM
        elif score <= 16:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def add_mitigation(
        self,
        mitigation: str,
        new_likelihood: Optional[int] = None,
        new_impact: Optional[int] = None,
    ) -> None:
        """Add mitigation and recalculate residual risk."""
        self.mitigations.append(mitigation)
        if new_likelihood:
            self.residual_likelihood = new_likelihood
        if new_impact:
            self.residual_impact = new_impact
        if self.residual_likelihood and self.residual_impact:
            self.residual_risk = self._calculate_risk_level(
                self.residual_likelihood, self.residual_impact
            )


@dataclass
class DataFlow:
    """Data flow documentation."""

    source: str
    destination: str
    data_categories: List[DataCategory]
    processing_types: List[ProcessingType]
    transfer_mechanism: Optional[str] = None  # SCCs, adequacy, etc.
    encryption_applied: bool = False
    retention_period: Optional[str] = None


@dataclass
class NecessityAssessment:
    """Necessity and proportionality assessment."""

    purpose: str
    legal_basis: ProcessingBasis
    necessity_justification: str
    proportionality_justification: str
    data_minimization: str
    storage_limitation: str
    alternatives_considered: List[str] = field(default_factory=list)
    alternative_rejection_reasons: Dict[str, str] = field(default_factory=dict)


@dataclass
class DPIAAssessment:
    """Complete DPIA assessment."""

    dpia_id: str
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    status: DPIAStatus
    owner: str
    version: int = 1

    # Processing details
    processing_purpose: str = ""
    data_categories: List[DataCategory] = field(default_factory=list)
    processing_types: List[ProcessingType] = field(default_factory=list)
    data_subjects: List[str] = field(default_factory=list)
    estimated_data_subjects: int = 0

    # Legal basis
    legal_basis: Optional[ProcessingBasis] = None
    necessity: Optional[NecessityAssessment] = None

    # Data flows
    data_flows: List[DataFlow] = field(default_factory=list)

    # Risk assessment
    risks: List[RiskAssessment] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.LOW

    # Consultation
    dpo_consulted: bool = False
    dpo_opinion: str = ""
    stakeholders_consulted: List[str] = field(default_factory=list)

    # Approval
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    # Review
    next_review_date: Optional[datetime] = None
    review_frequency_months: int = 12

    def requires_mandatory_dpia(self) -> bool:
        """Check if processing requires mandatory DPIA per Article 35(3)."""
        # Check for special category data
        if any(cat in MANDATORY_DPIA_CATEGORIES for cat in self.data_categories):
            return True

        # Check for high-risk processing types
        if any(pt in MANDATORY_DPIA_PROCESSING for pt in self.processing_types):
            return True

        # Large scale processing
        if self.estimated_data_subjects > 10000:
            return True

        return False

    def calculate_overall_risk(self) -> RiskLevel:
        """Calculate overall risk level from individual risks."""
        if not self.risks:
            return RiskLevel.LOW

        # Use highest residual risk (or inherent if no residual)
        max_risk = RiskLevel.LOW
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

        for risk in self.risks:
            current = risk.residual_risk or risk.inherent_risk
            if risk_order.index(current) > risk_order.index(max_risk):
                max_risk = current

        self.overall_risk = max_risk
        return max_risk

    def requires_supervisory_consultation(self) -> bool:
        """Check if Article 36 prior consultation is required."""
        self.calculate_overall_risk()

        # High residual risk after mitigations
        if self.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # Check if any critical risks remain unmitigated
            for risk in self.risks:
                effective_risk = risk.residual_risk or risk.inherent_risk
                if effective_risk == RiskLevel.CRITICAL and not risk.mitigations:
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dpia_id": self.dpia_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "owner": self.owner,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "processing_purpose": self.processing_purpose,
            "data_categories": [c.value for c in self.data_categories],
            "processing_types": [p.value for p in self.processing_types],
            "data_subjects": self.data_subjects,
            "estimated_data_subjects": self.estimated_data_subjects,
            "legal_basis": self.legal_basis.value if self.legal_basis else None,
            "overall_risk": self.overall_risk.value,
            "risks_count": len(self.risks),
            "requires_mandatory_dpia": self.requires_mandatory_dpia(),
            "requires_consultation": self.requires_supervisory_consultation(),
            "dpo_consulted": self.dpo_consulted,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }


# ============================================================================
# DPIA MANAGER
# ============================================================================


class DPIAManager:
    """
    Data Protection Impact Assessment Manager.

    Manages DPIA lifecycle, risk assessment, and approval workflow.

    Attributes:
        assessments: In-memory assessment storage
        audit_callback: Optional callback for audit logging
    """

    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        audit_callback: Optional[Callable] = None,
    ):
        """
        Initialize DPIA Manager.

        Args:
            storage_backend: Optional external storage
            audit_callback: Optional callback for audit logging
        """
        self._storage = storage_backend
        self._audit_callback = audit_callback
        self._assessments: Dict[str, DPIAAssessment] = {}

    def _audit_log(self, event: str, dpia_id: str, details: Dict[str, Any]) -> None:
        """Log DPIA event for audit trail."""
        if self._audit_callback:
            self._audit_callback(event, dpia_id, details)
        logger.info(f"DPIA: {event} | dpia={dpia_id} | {details}")

    def _generate_dpia_id(self) -> str:
        """Generate unique DPIA ID."""
        return f"DPIA-{uuid.uuid4().hex[:8].upper()}"

    def create_assessment(
        self,
        title: str,
        description: str,
        owner: str,
        processing_purpose: str,
        data_categories: List[DataCategory],
        processing_types: List[ProcessingType],
        data_subjects: List[str],
        estimated_data_subjects: int = 0,
        legal_basis: Optional[ProcessingBasis] = None,
    ) -> DPIAAssessment:
        """
        Create a new DPIA assessment.

        Args:
            title: Assessment title
            description: Description of processing
            owner: Assessment owner (user ID)
            processing_purpose: Purpose of processing
            data_categories: Categories of personal data
            processing_types: Types of processing operations
            data_subjects: Categories of data subjects
            estimated_data_subjects: Estimated number of data subjects
            legal_basis: Legal basis for processing

        Returns:
            New DPIAAssessment
        """
        now = datetime.now(timezone.utc)

        assessment = DPIAAssessment(
            dpia_id=self._generate_dpia_id(),
            title=title,
            description=description,
            created_at=now,
            updated_at=now,
            status=DPIAStatus.DRAFT,
            owner=owner,
            processing_purpose=processing_purpose,
            data_categories=data_categories,
            processing_types=processing_types,
            data_subjects=data_subjects,
            estimated_data_subjects=estimated_data_subjects,
            legal_basis=legal_basis,
        )

        self._assessments[assessment.dpia_id] = assessment

        self._audit_log(
            "dpia.created",
            assessment.dpia_id,
            {
                "title": title,
                "owner": owner,
                "mandatory": assessment.requires_mandatory_dpia(),
            },
        )

        return assessment

    def get_assessment(self, dpia_id: str) -> Optional[DPIAAssessment]:
        """Get DPIA assessment by ID."""
        return self._assessments.get(dpia_id)

    def add_risk(
        self,
        dpia_id: str,
        description: str,
        likelihood: int,
        impact: int,
        owner: Optional[str] = None,
    ) -> RiskAssessment:
        """
        Add a risk to DPIA assessment.

        Args:
            dpia_id: DPIA ID
            description: Risk description
            likelihood: Likelihood (1-5)
            impact: Impact (1-5)
            owner: Risk owner

        Returns:
            New RiskAssessment

        Raises:
            ValueError: If DPIA not found
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        risk = RiskAssessment(
            risk_id=f"RISK-{len(assessment.risks) + 1:03d}",
            description=description,
            likelihood=likelihood,
            impact=impact,
            owner=owner,
        )

        assessment.risks.append(risk)
        assessment.updated_at = datetime.now(timezone.utc)
        assessment.calculate_overall_risk()

        self._audit_log(
            "dpia.risk.added",
            dpia_id,
            {
                "risk_id": risk.risk_id,
                "inherent_risk": risk.inherent_risk.value,
            },
        )

        return risk

    def add_mitigation(
        self,
        dpia_id: str,
        risk_id: str,
        mitigation: str,
        new_likelihood: Optional[int] = None,
        new_impact: Optional[int] = None,
    ) -> None:
        """
        Add mitigation to a risk.

        Args:
            dpia_id: DPIA ID
            risk_id: Risk ID
            mitigation: Mitigation description
            new_likelihood: New likelihood after mitigation
            new_impact: New impact after mitigation

        Raises:
            ValueError: If DPIA or risk not found
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        risk = next((r for r in assessment.risks if r.risk_id == risk_id), None)
        if not risk:
            raise ValueError(f"Risk {risk_id} not found")

        risk.add_mitigation(mitigation, new_likelihood, new_impact)
        assessment.updated_at = datetime.now(timezone.utc)
        assessment.calculate_overall_risk()

        self._audit_log(
            "dpia.risk.mitigated",
            dpia_id,
            {
                "risk_id": risk_id,
                "residual_risk": risk.residual_risk.value if risk.residual_risk else None,
            },
        )

    def add_data_flow(
        self,
        dpia_id: str,
        source: str,
        destination: str,
        data_categories: List[DataCategory],
        processing_types: List[ProcessingType],
        transfer_mechanism: Optional[str] = None,
        encryption_applied: bool = False,
        retention_period: Optional[str] = None,
    ) -> DataFlow:
        """
        Add data flow to DPIA.

        Args:
            dpia_id: DPIA ID
            source: Data source
            destination: Data destination
            data_categories: Categories of data in flow
            processing_types: Processing types applied
            transfer_mechanism: Transfer mechanism (for international transfers)
            encryption_applied: Whether encryption is applied
            retention_period: Data retention period

        Returns:
            New DataFlow

        Raises:
            ValueError: If DPIA not found
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        flow = DataFlow(
            source=source,
            destination=destination,
            data_categories=data_categories,
            processing_types=processing_types,
            transfer_mechanism=transfer_mechanism,
            encryption_applied=encryption_applied,
            retention_period=retention_period,
        )

        assessment.data_flows.append(flow)
        assessment.updated_at = datetime.now(timezone.utc)

        self._audit_log(
            "dpia.dataflow.added",
            dpia_id,
            {"source": source, "destination": destination},
        )

        return flow

    def set_necessity_assessment(
        self,
        dpia_id: str,
        purpose: str,
        legal_basis: ProcessingBasis,
        necessity_justification: str,
        proportionality_justification: str,
        data_minimization: str,
        storage_limitation: str,
        alternatives_considered: Optional[List[str]] = None,
        alternative_rejection_reasons: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set necessity and proportionality assessment.

        Args:
            dpia_id: DPIA ID
            purpose: Processing purpose
            legal_basis: Legal basis for processing
            necessity_justification: Why processing is necessary
            proportionality_justification: Why processing is proportionate
            data_minimization: Data minimization measures
            storage_limitation: Storage limitation measures
            alternatives_considered: Alternative approaches considered
            alternative_rejection_reasons: Reasons for rejecting alternatives

        Raises:
            ValueError: If DPIA not found
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        assessment.necessity = NecessityAssessment(
            purpose=purpose,
            legal_basis=legal_basis,
            necessity_justification=necessity_justification,
            proportionality_justification=proportionality_justification,
            data_minimization=data_minimization,
            storage_limitation=storage_limitation,
            alternatives_considered=alternatives_considered or [],
            alternative_rejection_reasons=alternative_rejection_reasons or {},
        )
        assessment.legal_basis = legal_basis
        assessment.updated_at = datetime.now(timezone.utc)

        self._audit_log(
            "dpia.necessity.set",
            dpia_id,
            {"legal_basis": legal_basis.value},
        )

    def submit_for_review(self, dpia_id: str) -> None:
        """
        Submit DPIA for DPO review.

        Args:
            dpia_id: DPIA ID

        Raises:
            ValueError: If DPIA not found or not in draft status
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        if assessment.status != DPIAStatus.DRAFT:
            raise ValueError(f"DPIA must be in draft status to submit for review")

        # Validate completeness
        errors = self._validate_completeness(assessment)
        if errors:
            raise ValueError(f"DPIA incomplete: {', '.join(errors)}")

        assessment.status = DPIAStatus.IN_REVIEW
        assessment.updated_at = datetime.now(timezone.utc)

        self._audit_log(
            "dpia.submitted",
            dpia_id,
            {"owner": assessment.owner},
        )

    def _validate_completeness(self, assessment: DPIAAssessment) -> List[str]:
        """Validate DPIA completeness."""
        errors = []

        if not assessment.processing_purpose:
            errors.append("Processing purpose required")

        if not assessment.data_categories:
            errors.append("Data categories required")

        if not assessment.legal_basis:
            errors.append("Legal basis required")

        if not assessment.necessity:
            errors.append("Necessity assessment required")

        if not assessment.risks:
            errors.append("At least one risk assessment required")

        if not assessment.data_flows:
            errors.append("At least one data flow required")

        return errors

    def record_dpo_opinion(
        self,
        dpia_id: str,
        dpo_user_id: str,
        opinion: str,
        recommend_approval: bool,
    ) -> None:
        """
        Record DPO opinion on DPIA.

        Args:
            dpia_id: DPIA ID
            dpo_user_id: DPO user ID
            opinion: DPO's written opinion
            recommend_approval: Whether DPO recommends approval

        Raises:
            ValueError: If DPIA not found or not in review
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        if assessment.status != DPIAStatus.IN_REVIEW:
            raise ValueError("DPIA must be in review status for DPO opinion")

        assessment.dpo_consulted = True
        assessment.dpo_opinion = opinion
        assessment.stakeholders_consulted.append(f"DPO:{dpo_user_id}")
        assessment.updated_at = datetime.now(timezone.utc)

        if recommend_approval:
            assessment.status = DPIAStatus.PENDING_APPROVAL
        else:
            if assessment.requires_supervisory_consultation():
                assessment.status = DPIAStatus.REQUIRES_CONSULTATION

        self._audit_log(
            "dpia.dpo.opinion",
            dpia_id,
            {
                "dpo": dpo_user_id,
                "recommend_approval": recommend_approval,
                "new_status": assessment.status.value,
            },
        )

    def approve(self, dpia_id: str, approver_user_id: str) -> None:
        """
        Approve DPIA.

        Args:
            dpia_id: DPIA ID
            approver_user_id: Approver user ID

        Raises:
            ValueError: If DPIA not found or not pending approval
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        if assessment.status != DPIAStatus.PENDING_APPROVAL:
            raise ValueError("DPIA must be pending approval")

        if not assessment.dpo_consulted:
            raise ValueError("DPO consultation required before approval")

        assessment.status = DPIAStatus.APPROVED
        assessment.approved_by = approver_user_id
        assessment.approved_at = datetime.now(timezone.utc)
        assessment.updated_at = assessment.approved_at
        assessment.version += 1

        # Set next review date
        from datetime import timedelta

        assessment.next_review_date = assessment.approved_at + timedelta(
            days=assessment.review_frequency_months * 30
        )

        self._audit_log(
            "dpia.approved",
            dpia_id,
            {
                "approver": approver_user_id,
                "version": assessment.version,
                "next_review": assessment.next_review_date.isoformat(),
            },
        )

    def reject(self, dpia_id: str, rejector_user_id: str, reason: str) -> None:
        """
        Reject DPIA.

        Args:
            dpia_id: DPIA ID
            rejector_user_id: Rejector user ID
            reason: Rejection reason

        Raises:
            ValueError: If DPIA not found
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        assessment.status = DPIAStatus.REJECTED
        assessment.rejection_reason = reason
        assessment.updated_at = datetime.now(timezone.utc)

        self._audit_log(
            "dpia.rejected",
            dpia_id,
            {"rejector": rejector_user_id, "reason": reason},
        )

    def list_assessments(
        self,
        status: Optional[DPIAStatus] = None,
        owner: Optional[str] = None,
    ) -> List[DPIAAssessment]:
        """
        List DPIA assessments with optional filters.

        Args:
            status: Filter by status
            owner: Filter by owner

        Returns:
            List of assessments matching filters
        """
        results = list(self._assessments.values())

        if status:
            results = [a for a in results if a.status == status]

        if owner:
            results = [a for a in results if a.owner == owner]

        return sorted(results, key=lambda a: a.updated_at, reverse=True)

    def get_pending_reviews(self) -> List[DPIAAssessment]:
        """Get DPIAs pending review."""
        return self.list_assessments(status=DPIAStatus.IN_REVIEW)

    def get_due_for_review(self) -> List[DPIAAssessment]:
        """Get approved DPIAs due for periodic review."""
        now = datetime.now(timezone.utc)
        return [
            a
            for a in self._assessments.values()
            if a.status == DPIAStatus.APPROVED
            and a.next_review_date
            and a.next_review_date <= now
        ]

    def generate_report(self, dpia_id: str) -> Dict[str, Any]:
        """
        Generate DPIA report.

        Args:
            dpia_id: DPIA ID

        Returns:
            Report dictionary

        Raises:
            ValueError: If DPIA not found
        """
        assessment = self._assessments.get(dpia_id)
        if not assessment:
            raise ValueError(f"DPIA {dpia_id} not found")

        report = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "assessment": assessment.to_dict(),
            "necessity_assessment": None,
            "data_flows": [],
            "risk_register": [],
            "recommendations": [],
        }

        # Add necessity assessment
        if assessment.necessity:
            report["necessity_assessment"] = {
                "purpose": assessment.necessity.purpose,
                "legal_basis": assessment.necessity.legal_basis.value,
                "necessity_justification": assessment.necessity.necessity_justification,
                "proportionality_justification": assessment.necessity.proportionality_justification,
                "data_minimization": assessment.necessity.data_minimization,
                "storage_limitation": assessment.necessity.storage_limitation,
                "alternatives_considered": assessment.necessity.alternatives_considered,
            }

        # Add data flows
        for flow in assessment.data_flows:
            report["data_flows"].append(
                {
                    "source": flow.source,
                    "destination": flow.destination,
                    "data_categories": [c.value for c in flow.data_categories],
                    "processing_types": [p.value for p in flow.processing_types],
                    "transfer_mechanism": flow.transfer_mechanism,
                    "encryption_applied": flow.encryption_applied,
                    "retention_period": flow.retention_period,
                }
            )

        # Add risk register
        for risk in assessment.risks:
            report["risk_register"].append(
                {
                    "risk_id": risk.risk_id,
                    "description": risk.description,
                    "likelihood": risk.likelihood,
                    "impact": risk.impact,
                    "inherent_risk": risk.inherent_risk.value,
                    "mitigations": risk.mitigations,
                    "residual_likelihood": risk.residual_likelihood,
                    "residual_impact": risk.residual_impact,
                    "residual_risk": risk.residual_risk.value if risk.residual_risk else None,
                    "owner": risk.owner,
                    "status": risk.status,
                }
            )

        # Generate recommendations
        if assessment.requires_supervisory_consultation():
            report["recommendations"].append(
                {
                    "priority": "high",
                    "recommendation": "Supervisory authority consultation required under Article 36",
                }
            )

        unmitigated_high = [
            r
            for r in assessment.risks
            if (r.residual_risk or r.inherent_risk) in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            and not r.mitigations
        ]
        if unmitigated_high:
            report["recommendations"].append(
                {
                    "priority": "high",
                    "recommendation": f"{len(unmitigated_high)} high/critical risks require mitigation",
                }
            )

        return report


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_dpia_for_processing(
    manager: DPIAManager,
    title: str,
    owner: str,
    purpose: str,
    data_categories: List[str],
    processing_types: List[str],
    data_subjects: List[str],
    estimated_count: int = 0,
) -> DPIAAssessment:
    """
    Helper function to create DPIA with string inputs.

    Args:
        manager: DPIAManager instance
        title: Assessment title
        owner: Owner user ID
        purpose: Processing purpose
        data_categories: List of data category strings
        processing_types: List of processing type strings
        data_subjects: List of data subject descriptions
        estimated_count: Estimated number of data subjects

    Returns:
        New DPIAAssessment
    """
    # Convert string inputs to enums
    categories = []
    for cat in data_categories:
        try:
            categories.append(DataCategory(cat.lower()))
        except ValueError:
            logger.warning(f"Unknown data category: {cat}")

    types = []
    for pt in processing_types:
        try:
            types.append(ProcessingType(pt.lower()))
        except ValueError:
            logger.warning(f"Unknown processing type: {pt}")

    return manager.create_assessment(
        title=title,
        description=f"DPIA for: {purpose}",
        owner=owner,
        processing_purpose=purpose,
        data_categories=categories,
        processing_types=types,
        data_subjects=data_subjects,
        estimated_data_subjects=estimated_count,
    )


# Singleton instance
_dpia_manager: Optional[DPIAManager] = None


def get_dpia_manager() -> DPIAManager:
    """Get or create the default DPIA manager."""
    global _dpia_manager
    if _dpia_manager is None:
        _dpia_manager = DPIAManager()
    return _dpia_manager
