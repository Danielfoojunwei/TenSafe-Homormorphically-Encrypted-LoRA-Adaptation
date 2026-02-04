"""Status Page Service for TenSafe.

Core service for component status tracking, incident management,
maintenance scheduling, and uptime calculation.
"""

import asyncio
import json
import logging
import secrets
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    ComponentState,
    ComponentStatus,
    CreateIncidentRequest,
    CreateMaintenanceRequest,
    Incident,
    IncidentSeverity,
    IncidentStatus,
    IncidentUpdate,
    MaintenanceStatus,
    MaintenanceWindow,
    StatusSubscriber,
    SubscribeRequest,
    SystemStatus,
    UpdateIncidentRequest,
    UptimeMetrics,
)

logger = logging.getLogger(__name__)


class StatusService:
    """Status page service for TenSafe.

    Manages component status, incidents, maintenance windows,
    and provides uptime calculations for the public status page.

    Example:
        >>> service = StatusService()
        >>> service.register_component("api", "API Gateway", "Main API endpoint")
        >>> service.update_component_status("api", ComponentState.OPERATIONAL)
        >>> status = service.get_system_status()
    """

    # Default components for TenSafe platform
    DEFAULT_COMPONENTS = {
        "api": ("API Gateway", "Main API endpoint for all TenSafe services"),
        "training": ("Training Service", "ML model training infrastructure"),
        "inference": ("Inference Service", "Model inference and prediction"),
        "database": ("Database", "Primary data storage"),
        "queue": ("Job Queue", "Background job processing"),
        "storage": ("Object Storage", "Model and artifact storage"),
        "auth": ("Authentication", "Identity and access management"),
        "encryption": ("HE Encryption", "Homomorphic encryption service"),
    }

    def __init__(
        self,
        db_session: Optional[Any] = None,
        notification_handler: Optional[Callable] = None,
    ):
        """Initialize status service.

        Args:
            db_session: Optional database session for persistence
            notification_handler: Optional callback for notifications
        """
        self._db = db_session
        self._notification_handler = notification_handler
        self._lock = threading.RLock()

        # In-memory storage (used when no DB is provided)
        self._components: Dict[str, ComponentStatus] = {}
        self._incidents: Dict[str, Incident] = {}
        self._maintenance: Dict[str, MaintenanceWindow] = {}
        self._subscribers: Dict[str, StatusSubscriber] = {}
        self._status_history: Dict[str, List[Tuple[datetime, ComponentState]]] = defaultdict(list)

        # Initialize default components
        self._initialize_default_components()

    def _initialize_default_components(self):
        """Initialize default TenSafe components."""
        for comp_id, (name, description) in self.DEFAULT_COMPONENTS.items():
            self.register_component(comp_id, name, description)

    # ==========================================================================
    # Component Management
    # ==========================================================================

    def register_component(
        self,
        component_id: str,
        name: str,
        description: Optional[str] = None,
        initial_state: ComponentState = ComponentState.OPERATIONAL,
    ) -> ComponentStatus:
        """Register a new component for status tracking.

        Args:
            component_id: Unique identifier for the component
            name: Human-readable component name
            description: Optional description
            initial_state: Initial operational state

        Returns:
            Created ComponentStatus
        """
        with self._lock:
            component = ComponentStatus(
                id=component_id,
                name=name,
                description=description,
                state=initial_state,
                last_check=datetime.utcnow(),
            )
            self._components[component_id] = component
            self._status_history[component_id].append(
                (datetime.utcnow(), initial_state)
            )
            logger.info(f"Registered component: {component_id} ({name})")
            return component

    def update_component_status(
        self,
        component_id: str,
        state: ComponentState,
        response_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ComponentStatus]:
        """Update component status.

        Args:
            component_id: Component identifier
            state: New operational state
            response_time_ms: Optional response time in milliseconds
            metadata: Optional additional metadata

        Returns:
            Updated ComponentStatus or None if not found
        """
        with self._lock:
            if component_id not in self._components:
                logger.warning(f"Unknown component: {component_id}")
                return None

            component = self._components[component_id]
            old_state = component.state

            component.state = state
            component.last_check = datetime.utcnow()

            if response_time_ms is not None:
                component.response_time_ms = response_time_ms

            if metadata:
                component.metadata.update(metadata)

            # Record history
            self._status_history[component_id].append(
                (datetime.utcnow(), state)
            )

            # Trim history (keep last 1000 entries per component)
            if len(self._status_history[component_id]) > 1000:
                self._status_history[component_id] = self._status_history[component_id][-1000:]

            # Notify on state change
            if old_state != state:
                logger.info(f"Component {component_id} state changed: {old_state} -> {state}")
                self._notify_state_change(component_id, old_state, state)

            return component

    def get_component_status(self, component_id: str) -> Optional[ComponentStatus]:
        """Get status for a specific component."""
        with self._lock:
            return self._components.get(component_id)

    def get_all_components(self) -> List[ComponentStatus]:
        """Get status for all registered components."""
        with self._lock:
            return list(self._components.values())

    # ==========================================================================
    # Incident Management
    # ==========================================================================

    def create_incident(
        self,
        request: CreateIncidentRequest,
        created_by: Optional[str] = None,
    ) -> Incident:
        """Create a new incident.

        Args:
            request: Incident creation request
            created_by: Optional user identifier

        Returns:
            Created Incident
        """
        with self._lock:
            incident_id = f"inc_{secrets.token_hex(8)}"

            # Create initial update
            initial_update = IncidentUpdate(
                id=f"upd_{secrets.token_hex(8)}",
                incident_id=incident_id,
                status=IncidentStatus.INVESTIGATING,
                message=request.initial_message,
                created_at=datetime.utcnow(),
                created_by=created_by,
            )

            incident = Incident(
                id=incident_id,
                title=request.title,
                severity=request.severity,
                status=IncidentStatus.INVESTIGATING,
                affected_components=request.affected_components,
                started_at=datetime.utcnow(),
                updates=[initial_update],
            )

            self._incidents[incident_id] = incident

            # Update affected component states
            for comp_id in request.affected_components:
                state = self._severity_to_component_state(request.severity)
                self.update_component_status(comp_id, state)

            logger.warning(f"Incident created: {incident_id} - {request.title}")
            self._notify_incident_created(incident)

            return incident

    def update_incident(
        self,
        incident_id: str,
        request: UpdateIncidentRequest,
        updated_by: Optional[str] = None,
    ) -> Optional[Incident]:
        """Update an existing incident.

        Args:
            incident_id: Incident identifier
            request: Update request
            updated_by: Optional user identifier

        Returns:
            Updated Incident or None if not found
        """
        with self._lock:
            if incident_id not in self._incidents:
                logger.warning(f"Unknown incident: {incident_id}")
                return None

            incident = self._incidents[incident_id]

            # Create update record
            new_status = request.status or incident.status
            update = IncidentUpdate(
                id=f"upd_{secrets.token_hex(8)}",
                incident_id=incident_id,
                status=new_status,
                message=request.message,
                created_at=datetime.utcnow(),
                created_by=updated_by,
            )

            incident.updates.append(update)

            if request.status:
                incident.status = request.status

                # Handle resolution
                if request.status == IncidentStatus.RESOLVED:
                    incident.resolved_at = datetime.utcnow()
                    # Restore affected components to operational
                    for comp_id in incident.affected_components:
                        self.update_component_status(comp_id, ComponentState.OPERATIONAL)

            logger.info(f"Incident {incident_id} updated: {new_status}")
            self._notify_incident_updated(incident, update)

            return incident

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def get_active_incidents(self) -> List[Incident]:
        """Get all active (unresolved) incidents."""
        with self._lock:
            return [i for i in self._incidents.values() if i.is_active]

    def get_incidents(
        self,
        limit: int = 50,
        include_resolved: bool = True,
        severity: Optional[IncidentSeverity] = None,
        component: Optional[str] = None,
    ) -> List[Incident]:
        """Get incidents with optional filters.

        Args:
            limit: Maximum number of incidents to return
            include_resolved: Include resolved incidents
            severity: Filter by severity
            component: Filter by affected component

        Returns:
            List of matching incidents
        """
        with self._lock:
            incidents = list(self._incidents.values())

            if not include_resolved:
                incidents = [i for i in incidents if i.is_active]

            if severity:
                incidents = [i for i in incidents if i.severity == severity]

            if component:
                incidents = [i for i in incidents if component in i.affected_components]

            # Sort by started_at descending
            incidents.sort(key=lambda x: x.started_at, reverse=True)

            return incidents[:limit]

    # ==========================================================================
    # Maintenance Management
    # ==========================================================================

    def schedule_maintenance(
        self,
        request: CreateMaintenanceRequest,
        created_by: Optional[str] = None,
    ) -> MaintenanceWindow:
        """Schedule a maintenance window.

        Args:
            request: Maintenance creation request
            created_by: Optional user identifier

        Returns:
            Created MaintenanceWindow
        """
        with self._lock:
            maintenance_id = f"maint_{secrets.token_hex(8)}"

            maintenance = MaintenanceWindow(
                id=maintenance_id,
                title=request.title,
                description=request.description,
                affected_components=request.affected_components,
                scheduled_start=request.scheduled_start,
                scheduled_end=request.scheduled_end,
                impact=request.impact,
                created_by=created_by,
            )

            self._maintenance[maintenance_id] = maintenance

            logger.info(f"Maintenance scheduled: {maintenance_id} - {request.title}")
            self._notify_maintenance_scheduled(maintenance)

            return maintenance

    def start_maintenance(self, maintenance_id: str) -> Optional[MaintenanceWindow]:
        """Start a scheduled maintenance window."""
        with self._lock:
            if maintenance_id not in self._maintenance:
                return None

            maintenance = self._maintenance[maintenance_id]
            maintenance.status = MaintenanceStatus.IN_PROGRESS
            maintenance.actual_start = datetime.utcnow()

            # Update affected components
            for comp_id in maintenance.affected_components:
                self.update_component_status(comp_id, ComponentState.MAINTENANCE)

            logger.info(f"Maintenance started: {maintenance_id}")
            return maintenance

    def complete_maintenance(self, maintenance_id: str) -> Optional[MaintenanceWindow]:
        """Complete a maintenance window."""
        with self._lock:
            if maintenance_id not in self._maintenance:
                return None

            maintenance = self._maintenance[maintenance_id]
            maintenance.status = MaintenanceStatus.COMPLETED
            maintenance.actual_end = datetime.utcnow()

            # Restore affected components
            for comp_id in maintenance.affected_components:
                self.update_component_status(comp_id, ComponentState.OPERATIONAL)

            logger.info(f"Maintenance completed: {maintenance_id}")
            return maintenance

    def cancel_maintenance(self, maintenance_id: str) -> Optional[MaintenanceWindow]:
        """Cancel a scheduled maintenance window."""
        with self._lock:
            if maintenance_id not in self._maintenance:
                return None

            maintenance = self._maintenance[maintenance_id]
            maintenance.status = MaintenanceStatus.CANCELLED

            logger.info(f"Maintenance cancelled: {maintenance_id}")
            return maintenance

    def get_maintenance(self, maintenance_id: str) -> Optional[MaintenanceWindow]:
        """Get maintenance window by ID."""
        with self._lock:
            return self._maintenance.get(maintenance_id)

    def get_active_maintenance(self) -> List[MaintenanceWindow]:
        """Get all active maintenance windows."""
        with self._lock:
            return [m for m in self._maintenance.values() if m.is_active]

    def get_upcoming_maintenance(self, days: int = 30) -> List[MaintenanceWindow]:
        """Get upcoming scheduled maintenance windows."""
        with self._lock:
            cutoff = datetime.utcnow() + timedelta(days=days)
            upcoming = [
                m for m in self._maintenance.values()
                if m.is_upcoming and m.scheduled_start <= cutoff
            ]
            upcoming.sort(key=lambda x: x.scheduled_start)
            return upcoming

    def get_scheduled_maintenance(
        self,
        include_past: bool = False,
        limit: int = 20,
    ) -> List[MaintenanceWindow]:
        """Get scheduled maintenance windows."""
        with self._lock:
            windows = list(self._maintenance.values())

            if not include_past:
                now = datetime.utcnow()
                windows = [
                    m for m in windows
                    if m.scheduled_end > now or m.status == MaintenanceStatus.IN_PROGRESS
                ]

            windows.sort(key=lambda x: x.scheduled_start)
            return windows[:limit]

    # ==========================================================================
    # Uptime Calculation
    # ==========================================================================

    def calculate_uptime(
        self,
        component_id: str,
        period: str = "monthly",
    ) -> UptimeMetrics:
        """Calculate uptime metrics for a component.

        Args:
            component_id: Component identifier or 'system' for overall
            period: Time period ('daily', 'monthly', 'yearly')

        Returns:
            UptimeMetrics for the specified period
        """
        now = datetime.utcnow()

        # Determine period bounds
        if period == "daily":
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = now
            total_minutes = int((period_end - period_start).total_seconds() / 60)
        elif period == "monthly":
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = now
            total_minutes = int((period_end - period_start).total_seconds() / 60)
        elif period == "yearly":
            period_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = now
            total_minutes = int((period_end - period_start).total_seconds() / 60)
        else:
            # Default to last 30 days
            period_start = now - timedelta(days=30)
            period_end = now
            total_minutes = 30 * 24 * 60

        with self._lock:
            # Calculate downtime from status history
            downtime_minutes = 0.0
            incident_count = 0
            recovery_times = []

            if component_id == "system":
                # For system-wide, check all components
                for comp_id in self._components:
                    comp_downtime, comp_incidents, comp_recovery = self._calculate_component_downtime(
                        comp_id, period_start, period_end
                    )
                    downtime_minutes = max(downtime_minutes, comp_downtime)
                    incident_count += comp_incidents
                    recovery_times.extend(comp_recovery)
            else:
                downtime_minutes, incident_count, recovery_times = self._calculate_component_downtime(
                    component_id, period_start, period_end
                )

            # Ensure we don't exceed total period
            downtime_minutes = min(downtime_minutes, total_minutes)

            # Calculate uptime percentage
            uptime_percentage = ((total_minutes - downtime_minutes) / total_minutes) * 100 if total_minutes > 0 else 100.0

            # Calculate MTTR (Mean Time to Recovery)
            mttr = sum(recovery_times) / len(recovery_times) if recovery_times else None

            # Calculate MTBF (Mean Time Between Failures)
            mtbf = None
            if incident_count > 1:
                mtbf = (total_minutes - downtime_minutes) / (incident_count - 1) / 60  # in hours

            # Get response time metrics from component
            component = self._components.get(component_id) if component_id != "system" else None

            # Determine SLA target based on component
            sla_target = 99.9  # Default Pro tier

            return UptimeMetrics(
                component_id=component_id,
                period=period,
                uptime_percentage=round(uptime_percentage, 4),
                total_minutes=total_minutes,
                downtime_minutes=round(downtime_minutes, 2),
                incident_count=incident_count,
                mttr_minutes=round(mttr, 2) if mttr else None,
                mtbf_hours=round(mtbf, 2) if mtbf else None,
                avg_response_time_ms=component.response_time_ms if component else None,
                sla_target=sla_target,
                sla_met=uptime_percentage >= sla_target,
                sla_breach_minutes=max(0, downtime_minutes - (total_minutes * (100 - sla_target) / 100)),
                period_start=period_start,
                period_end=period_end,
            )

    def _calculate_component_downtime(
        self,
        component_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> Tuple[float, int, List[float]]:
        """Calculate downtime for a component in a given period.

        Returns:
            Tuple of (downtime_minutes, incident_count, recovery_times)
        """
        history = self._status_history.get(component_id, [])

        downtime_minutes = 0.0
        incident_count = 0
        recovery_times = []

        outage_start = None
        degraded_states = {
            ComponentState.DEGRADED,
            ComponentState.PARTIAL_OUTAGE,
            ComponentState.MAJOR_OUTAGE,
        }

        for timestamp, state in history:
            if timestamp < period_start:
                continue
            if timestamp > period_end:
                break

            if state in degraded_states:
                if outage_start is None:
                    outage_start = timestamp
                    incident_count += 1
            elif outage_start is not None:
                # Recovered
                duration = (timestamp - outage_start).total_seconds() / 60
                downtime_minutes += duration
                recovery_times.append(duration)
                outage_start = None

        # Handle ongoing outage
        if outage_start is not None:
            duration = (period_end - outage_start).total_seconds() / 60
            downtime_minutes += duration

        return downtime_minutes, incident_count, recovery_times

    def get_historical_uptime(
        self,
        component_id: str = "system",
        periods: int = 90,
    ) -> List[Dict[str, Any]]:
        """Get historical daily uptime for charting.

        Args:
            component_id: Component identifier or 'system'
            periods: Number of days to include

        Returns:
            List of daily uptime records
        """
        history = []
        now = datetime.utcnow()

        for i in range(periods):
            day = now - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            if day_end > now:
                day_end = now

            with self._lock:
                if component_id == "system":
                    # System-wide uptime
                    max_downtime = 0.0
                    for comp_id in self._components:
                        downtime, _, _ = self._calculate_component_downtime(comp_id, day_start, day_end)
                        max_downtime = max(max_downtime, downtime)
                else:
                    max_downtime, _, _ = self._calculate_component_downtime(component_id, day_start, day_end)

            total_minutes = (day_end - day_start).total_seconds() / 60
            uptime_pct = ((total_minutes - max_downtime) / total_minutes) * 100 if total_minutes > 0 else 100.0

            history.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "uptime_percentage": round(uptime_pct, 4),
                "downtime_minutes": round(max_downtime, 2),
            })

        history.reverse()
        return history

    # ==========================================================================
    # System Status
    # ==========================================================================

    def get_system_status(self) -> SystemStatus:
        """Get overall system status summary.

        Returns:
            SystemStatus with all components, incidents, and maintenance
        """
        with self._lock:
            components = list(self._components.values())
            active_incidents = self.get_active_incidents()
            active_maintenance = self.get_active_maintenance()
            upcoming_maintenance = self.get_upcoming_maintenance(days=7)

            # Determine overall status
            overall_status = self._calculate_overall_status(components, active_incidents, active_maintenance)
            status_message = self._get_status_message(overall_status, active_incidents, active_maintenance)

            return SystemStatus(
                status=overall_status,
                status_message=status_message,
                components=components,
                active_incidents=active_incidents,
                active_maintenance=active_maintenance,
                upcoming_maintenance=upcoming_maintenance,
                last_updated=datetime.utcnow(),
            )

    def _calculate_overall_status(
        self,
        components: List[ComponentStatus],
        incidents: List[Incident],
        maintenance: List[MaintenanceWindow],
    ) -> ComponentState:
        """Calculate overall system status from component states."""
        if not components:
            return ComponentState.UNKNOWN

        # Check for active P1/P2 incidents
        for incident in incidents:
            if incident.severity == IncidentSeverity.P1_CRITICAL:
                return ComponentState.MAJOR_OUTAGE
            if incident.severity == IncidentSeverity.P2_MAJOR:
                return ComponentState.PARTIAL_OUTAGE

        # Check component states
        states = [c.state for c in components]

        if ComponentState.MAJOR_OUTAGE in states:
            return ComponentState.MAJOR_OUTAGE
        if ComponentState.PARTIAL_OUTAGE in states:
            return ComponentState.PARTIAL_OUTAGE
        if ComponentState.DEGRADED in states:
            return ComponentState.DEGRADED
        if ComponentState.MAINTENANCE in states and maintenance:
            return ComponentState.MAINTENANCE

        return ComponentState.OPERATIONAL

    def _get_status_message(
        self,
        status: ComponentState,
        incidents: List[Incident],
        maintenance: List[MaintenanceWindow],
    ) -> str:
        """Generate human-readable status message."""
        if status == ComponentState.OPERATIONAL:
            return "All systems operational"
        if status == ComponentState.DEGRADED:
            return "Some systems experiencing degraded performance"
        if status == ComponentState.PARTIAL_OUTAGE:
            count = len(incidents)
            return f"Partial system outage - {count} active incident{'s' if count > 1 else ''}"
        if status == ComponentState.MAJOR_OUTAGE:
            return "Major system outage - Service disruption in progress"
        if status == ComponentState.MAINTENANCE:
            return f"Scheduled maintenance in progress"
        return "System status unknown"

    # ==========================================================================
    # Subscriptions
    # ==========================================================================

    def subscribe(self, request: SubscribeRequest) -> StatusSubscriber:
        """Subscribe to status updates.

        Args:
            request: Subscription request

        Returns:
            Created StatusSubscriber
        """
        with self._lock:
            subscriber_id = f"sub_{secrets.token_hex(8)}"
            verification_token = secrets.token_urlsafe(32)
            unsubscribe_token = secrets.token_urlsafe(32)

            subscriber = StatusSubscriber(
                id=subscriber_id,
                email=request.email,
                components=request.components,
                notify_on=request.notify_on,
                verification_token=verification_token,
                unsubscribe_token=unsubscribe_token,
            )

            self._subscribers[subscriber_id] = subscriber

            logger.info(f"New subscriber: {request.email}")
            return subscriber

    def verify_subscription(self, token: str) -> bool:
        """Verify a subscription with token."""
        with self._lock:
            for subscriber in self._subscribers.values():
                if subscriber.verification_token == token:
                    subscriber.verified = True
                    subscriber.verification_token = None
                    logger.info(f"Subscription verified: {subscriber.email}")
                    return True
            return False

    def unsubscribe(self, token: str) -> bool:
        """Unsubscribe using token."""
        with self._lock:
            for sub_id, subscriber in list(self._subscribers.items()):
                if subscriber.unsubscribe_token == token:
                    del self._subscribers[sub_id]
                    logger.info(f"Unsubscribed: {subscriber.email}")
                    return True
            return False

    # ==========================================================================
    # Notifications (Internal)
    # ==========================================================================

    def _notify_state_change(
        self,
        component_id: str,
        old_state: ComponentState,
        new_state: ComponentState,
    ):
        """Notify subscribers of component state change."""
        if self._notification_handler:
            try:
                self._notification_handler(
                    "component_state_change",
                    {
                        "component_id": component_id,
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def _notify_incident_created(self, incident: Incident):
        """Notify subscribers of new incident."""
        if self._notification_handler:
            try:
                self._notification_handler(
                    "incident_created",
                    {
                        "incident_id": incident.id,
                        "title": incident.title,
                        "severity": incident.severity.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def _notify_incident_updated(self, incident: Incident, update: IncidentUpdate):
        """Notify subscribers of incident update."""
        if self._notification_handler:
            try:
                self._notification_handler(
                    "incident_updated",
                    {
                        "incident_id": incident.id,
                        "update_id": update.id,
                        "status": update.status.value,
                        "message": update.message,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def _notify_maintenance_scheduled(self, maintenance: MaintenanceWindow):
        """Notify subscribers of scheduled maintenance."""
        if self._notification_handler:
            try:
                self._notification_handler(
                    "maintenance_scheduled",
                    {
                        "maintenance_id": maintenance.id,
                        "title": maintenance.title,
                        "scheduled_start": maintenance.scheduled_start.isoformat(),
                        "scheduled_end": maintenance.scheduled_end.isoformat(),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    @staticmethod
    def _severity_to_component_state(severity: IncidentSeverity) -> ComponentState:
        """Map incident severity to component state."""
        mapping = {
            IncidentSeverity.P1_CRITICAL: ComponentState.MAJOR_OUTAGE,
            IncidentSeverity.P2_MAJOR: ComponentState.PARTIAL_OUTAGE,
            IncidentSeverity.P3_MINOR: ComponentState.DEGRADED,
            IncidentSeverity.P4_LOW: ComponentState.DEGRADED,
        }
        return mapping.get(severity, ComponentState.DEGRADED)


# Singleton instance for global access
_status_service: Optional[StatusService] = None


def get_status_service() -> StatusService:
    """Get global status service instance."""
    global _status_service
    if _status_service is None:
        _status_service = StatusService()
    return _status_service


def init_status_service(
    db_session: Optional[Any] = None,
    notification_handler: Optional[Callable] = None,
) -> StatusService:
    """Initialize global status service.

    Args:
        db_session: Optional database session
        notification_handler: Optional notification callback

    Returns:
        Initialized StatusService
    """
    global _status_service
    _status_service = StatusService(
        db_session=db_session,
        notification_handler=notification_handler,
    )
    return _status_service
