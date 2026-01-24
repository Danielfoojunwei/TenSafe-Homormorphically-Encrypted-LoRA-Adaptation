"""
Governance Policy Gates

Enforces safety and privacy constraints before learning or output publication.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PolicyViolation(Exception):
    pass

class GovernanceEngine:
    """
    Evaluates policy gates.
    """
    def __init__(self, policy_config: Dict[str, Any]):
        self.config = policy_config
        self.dp_budget_limit = policy_config.get("dp_budget_limit", 10.0)
        self.allowed_layers = policy_config.get("allowed_layers", [])
        
    def check_dp_budget(self, current_spend: float, cost: float):
        """Gate: DP Budget."""
        if current_spend + cost > self.dp_budget_limit:
            raise PolicyViolation(f"DP Budget exceeded: {current_spend} + {cost} > {self.dp_budget_limit}")
        logger.info(f"DP Gate Passed: {current_spend} + {cost} <= {self.dp_budget_limit}")

    def check_patch_scope(self, patch_metadata: Dict):
        """Gate: Patch Scope."""
        # Check layer names
        layers = patch_metadata.get("layers", [])
        for l in layers:
            if self.allowed_layers and not any(allowed in l for allowed in self.allowed_layers):
                raise PolicyViolation(f"Layer not allowed in patch: {l}")
        logger.info("Patch Scope Gate Passed")

    def check_rollback_contract(self, artifacts: List[str]):
        """Gate: Rollback Contract."""
        # Ensure a rollback manifest exists
        if not any("rollback" in a for a in artifacts):
             raise PolicyViolation("Rollback artifact missing. Cannot publish patch.")
        logger.info("Rollback Contract Gate Passed")
