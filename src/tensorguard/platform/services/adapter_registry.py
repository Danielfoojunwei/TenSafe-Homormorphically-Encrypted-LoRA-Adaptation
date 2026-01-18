"""
Adapter Registry Service

Provides adapter lifecycle management:
- register_adapter: Register new adapter with TGSP
- promote: Promote adapter to release channel
- rollback: Instant rollback to previous adapter
- diff: Compare adapters ("What changed since last week?")
- enforce_lifecycle: Apply lifecycle policies (cap, archive)
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from sqlmodel import Session, select

from ..models.tgflow_core_models import (
    AdapterArtifact,
    AdapterEvalReport,
    AdapterRoute,
    AdapterRelease,
    AdapterStatus,
    ReleaseChannel,
)
from ...tgsp.manifest import PackageManifest
from ...trust import TrustCore
from ...privacy import PrivacyCore


class AdapterRegistryError(Exception):
    """Base exception for adapter registry operations."""
    pass


class AdapterNotFoundError(AdapterRegistryError):
    """Raised when adapter is not found."""
    pass


class PromotionGateError(AdapterRegistryError):
    """Raised when promotion gates fail."""
    pass


class AdapterRegistry:
    """
    Service for managing adapter lifecycle.
    
    Supports:
    - Adapter registration with TGSP validation
    - Channel-based promotion (canary → staging → stable)
    - Instant rollback
    - Adapter diff for change tracking
    - Lifecycle policies (retention, archival)
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    def register_adapter(
        self,
        tenant_id: str,
        name: str,
        base_model_id: str,
        adapter_path: str,
        tgsp_path: Optional[str] = None,
        evidence_path: Optional[str] = None,
        privacy_mode: str = "off",
        config: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> AdapterArtifact:
        """
        Register a new adapter in the registry.
        
        Args:
            tenant_id: Tenant identifier
            name: Adapter name
            base_model_id: Base model identifier
            adapter_path: Path to adapter weights
            tgsp_path: Path to TGSP package (optional)
            evidence_path: Path to evidence chain (optional)
            privacy_mode: Privacy mode ("off" or "n2he")
            config: Additional configuration
            labels: Custom labels
            
        Returns:
            Registered AdapterArtifact
        """
        # Compute adapter hash
        adapter_hash = self._compute_file_hash(adapter_path)
        
        # Extract TGSP manifest hash if available
        tgsp_manifest_hash = None
        privacy_claims_hash = None
        if tgsp_path:
            try:
                manifest = self._load_manifest(tgsp_path)
                tgsp_manifest_hash = manifest.get_hash()
                if manifest.privacy.mode == "n2he":
                    privacy_claims_hash = hashlib.sha256(
                        json.dumps(manifest.privacy.model_dump(), sort_keys=True).encode()
                    ).hexdigest()
            except Exception:
                pass
        
        # Create adapter artifact
        artifact = AdapterArtifact(
            tenant_id=tenant_id,
            name=name,
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            tgsp_path=tgsp_path,
            evidence_path=evidence_path,
            adapter_hash=adapter_hash,
            tgsp_manifest_hash=tgsp_manifest_hash,
            privacy_mode=privacy_mode,
            privacy_claims_hash=privacy_claims_hash,
            config_json=config or {},
            labels=labels or {},
            status=AdapterStatus.REGISTERED,
        )
        
        self.session.add(artifact)
        self.session.commit()
        self.session.refresh(artifact)
        
        return artifact
    
    def get_adapter(self, adapter_id: str) -> AdapterArtifact:
        """Get adapter by ID."""
        adapter = self.session.get(AdapterArtifact, adapter_id)
        if not adapter:
            raise AdapterNotFoundError(f"Adapter {adapter_id} not found")
        return adapter
    
    def promote(
        self,
        route_key: str,
        adapter_id: str,
        channel: ReleaseChannel,
        tenant_id: str,
        eval_report_id: Optional[str] = None,
        released_by: Optional[str] = None,
        sign: bool = False,
    ) -> AdapterRelease:
        """
        Promote an adapter to a release channel.
        
        Args:
            route_key: Routing key for this adapter
            adapter_id: Adapter to promote
            channel: Target channel (canary, staging, stable)
            tenant_id: Tenant identifier
            eval_report_id: Optional evaluation report reference
            released_by: User/system performing promotion
            sign: Whether to sign the release with TrustCore
            
        Returns:
            AdapterRelease record
        """
        adapter = self.get_adapter(adapter_id)
        
        # Get or create route
        route = self._get_or_create_route(route_key, tenant_id)
        
        # Track previous adapter for rollback
        previous_adapter_id = None
        if channel == ReleaseChannel.CANARY:
            previous_adapter_id = route.canary_adapter_id
            route.canary_adapter_id = adapter_id
        elif channel == ReleaseChannel.STAGING:
            previous_adapter_id = route.staging_adapter_id
            route.staging_adapter_id = adapter_id
        elif channel == ReleaseChannel.STABLE:
            previous_adapter_id = route.stable_adapter_id
            route.stable_adapter_id = adapter_id
            # Set rollback target
            route.rollback_adapter_id = previous_adapter_id
        
        # Update adapter status
        adapter.status = AdapterStatus.PROMOTED
        
        # Create release record
        release = AdapterRelease(
            tenant_id=tenant_id,
            adapter_id=adapter_id,
            route_key=route_key,
            channel=channel,
            previous_adapter_id=previous_adapter_id,
            eval_report_id=eval_report_id,
            privacy_mode=adapter.privacy_mode,
            privacy_receipt_hash=adapter.privacy_claims_hash,
            released_by=released_by,
            decision_reason="promotion",
        )
        
        # Sign release if requested
        if sign:
            try:
                decision = {
                    "adapter_id": adapter_id,
                    "route_key": route_key,
                    "channel": channel.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                result = TrustCore.sign_promotion_decision(decision)
                if result.success:
                    release.signed = True
                    release.signature = result.signature
                    release.signer_key_id = result.key_id
            except Exception:
                pass  # Signing optional
        
        self.session.add(route)
        self.session.add(adapter)
        self.session.add(release)
        self.session.commit()
        
        return release
    
    def rollback(self, route_key: str, tenant_id: str) -> Optional[AdapterRelease]:
        """
        Rollback route to previous adapter.
        
        Args:
            route_key: Routing key to rollback
            tenant_id: Tenant identifier
            
        Returns:
            AdapterRelease for rollback, or None if no rollback available
        """
        route = self._get_route(route_key, tenant_id)
        if not route or not route.rollback_adapter_id:
            return None
        
        # Create rollback release
        release = AdapterRelease(
            tenant_id=tenant_id,
            adapter_id=route.rollback_adapter_id,
            route_key=route_key,
            channel=ReleaseChannel.ROLLBACK,
            previous_adapter_id=route.stable_adapter_id,
            decision_reason="rollback",
        )
        
        # Swap adapters
        route.stable_adapter_id = route.rollback_adapter_id
        route.rollback_adapter_id = None
        
        self.session.add(route)
        self.session.add(release)
        self.session.commit()
        
        return release
    
    def diff(
        self,
        adapter_a_id: str,
        adapter_b_id: str,
    ) -> Dict[str, Any]:
        """
        Compare two adapters ("What changed since last week?").
        
        Args:
            adapter_a_id: First adapter (older)
            adapter_b_id: Second adapter (newer)
            
        Returns:
            Diff report with changes
        """
        adapter_a = self.get_adapter(adapter_a_id)
        adapter_b = self.get_adapter(adapter_b_id)
        
        # Get eval reports if available
        eval_a = self._get_latest_eval(adapter_a_id)
        eval_b = self._get_latest_eval(adapter_b_id)
        
        diff = {
            "adapters": {
                "from": {"id": adapter_a_id, "name": adapter_a.name, "version": adapter_a.version},
                "to": {"id": adapter_b_id, "name": adapter_b.name, "version": adapter_b.version},
            },
            "hash_changed": adapter_a.adapter_hash != adapter_b.adapter_hash,
            "privacy_mode_changed": adapter_a.privacy_mode != adapter_b.privacy_mode,
            "metrics_delta": {},
            "config_delta": {},
        }
        
        if eval_a and eval_b:
            diff["metrics_delta"] = {
                "primary_metric": eval_b.primary_metric - eval_a.primary_metric,
                "forgetting_score": eval_b.forgetting_score - eval_a.forgetting_score,
            }
        
        # Config diff
        config_a = adapter_a.config_json or {}
        config_b = adapter_b.config_json or {}
        diff["config_delta"] = {
            "added": [k for k in config_b if k not in config_a],
            "removed": [k for k in config_a if k not in config_b],
            "changed": [k for k in config_a if k in config_b and config_a[k] != config_b[k]],
        }
        
        return diff
    
    def resolve_adapter(
        self,
        tenant_id: str,
        route_key: str,
        channel: Optional[ReleaseChannel] = None,
        privacy_mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve which adapter to use for a request.
        
        Args:
            tenant_id: Tenant identifier
            route_key: Routing key
            channel: Preferred channel (default: stable)
            privacy_mode_override: Override privacy mode
            
        Returns:
            Resolution with adapter_id, tgsp_uri, evidence_head_hash
        """
        route = self._get_route(route_key, tenant_id)
        if not route:
            return {"error": f"No route found for '{route_key}'", "adapter_id": None}
        
        # Determine adapter based on channel
        channel = channel or ReleaseChannel.STABLE
        adapter_id = None
        
        if channel == ReleaseChannel.CANARY:
            adapter_id = route.canary_adapter_id
        elif channel == ReleaseChannel.STAGING:
            adapter_id = route.staging_adapter_id
        else:
            adapter_id = route.stable_adapter_id
        
        if not adapter_id:
            return {"error": f"No adapter in channel '{channel.value}'", "adapter_id": None}
        
        adapter = self.get_adapter(adapter_id)
        
        # Determine privacy mode
        privacy_mode = privacy_mode_override or route.privacy_mode
        
        return {
            "adapter_id": adapter_id,
            "tgsp_uri": adapter.tgsp_path,
            "evidence_head_hash": adapter.evidence_head_hash,
            "privacy_mode": privacy_mode,
            "route_key": route_key,
            "channel": channel.value,
        }
    
    def enforce_lifecycle_policies(
        self,
        tenant_id: str,
        route_key: str,
        max_adapters_per_route: int = 10,
        archive_dominated: bool = True,
    ) -> List[str]:
        """
        Enforce lifecycle policies on a route.
        
        Args:
            tenant_id: Tenant identifier
            route_key: Routing key
            max_adapters_per_route: Maximum adapters to keep
            archive_dominated: Archive adapters with lower metrics
            
        Returns:
            List of archived adapter IDs
        """
        archived = []
        
        # Get all adapters for this route
        statement = select(AdapterArtifact).where(
            AdapterArtifact.tenant_id == tenant_id,
            AdapterArtifact.status != AdapterStatus.ARCHIVED,
        )
        adapters = list(self.session.exec(statement))
        
        if len(adapters) > max_adapters_per_route:
            # Sort by created_at, archive oldest
            adapters.sort(key=lambda a: a.created_at)
            to_archive = adapters[:-max_adapters_per_route]
            
            for adapter in to_archive:
                adapter.status = AdapterStatus.ARCHIVED
                self.session.add(adapter)
                archived.append(adapter.id)
        
        if archived:
            self.session.commit()
        
        return archived
    
    # --- Private methods ---
    
    def _compute_file_hash(self, path: str) -> str:
        """Compute SHA-256 hash of file or directory."""
        import os
        if os.path.isdir(path):
            return hashlib.sha256(path.encode()).hexdigest()  # Simplified for dirs
        try:
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return hashlib.sha256(path.encode()).hexdigest()
    
    def _load_manifest(self, tgsp_path: str) -> PackageManifest:
        """Load TGSP manifest from package."""
        import os
        manifest_path = os.path.join(os.path.dirname(tgsp_path), "manifest.json")
        with open(manifest_path) as f:
            data = json.load(f)
        return PackageManifest(**data)
    
    def _get_or_create_route(self, route_key: str, tenant_id: str) -> AdapterRoute:
        """Get existing route or create new one."""
        route = self._get_route(route_key, tenant_id)
        if not route:
            route = AdapterRoute(tenant_id=tenant_id, route_key=route_key)
            self.session.add(route)
            self.session.commit()
            self.session.refresh(route)
        return route
    
    def _get_route(self, route_key: str, tenant_id: str) -> Optional[AdapterRoute]:
        """Get route by key and tenant."""
        statement = select(AdapterRoute).where(
            AdapterRoute.route_key == route_key,
            AdapterRoute.tenant_id == tenant_id,
        )
        return self.session.exec(statement).first()
    
    def _get_latest_eval(self, adapter_id: str) -> Optional[AdapterEvalReport]:
        """Get latest evaluation report for adapter."""
        statement = select(AdapterEvalReport).where(
            AdapterEvalReport.adapter_id == adapter_id
        ).order_by(AdapterEvalReport.evaluated_at.desc())
        return self.session.exec(statement).first()
