from typing import Dict, Any, List, Optional

def compute_trust_metrics(tgsp_manifest: Dict[str, Any], signature_verified: bool):
    """
    Computes trust metrics from TGSP manifest and verification status.
    """
    # 1. Evidence Completeness
    # Expected sections: model_lineage, training_evidence, evaluation_evidence, code_provenance
    required_sections = ["model_lineage", "training_evidence", "evaluation_evidence"]
    present_sections = [s for s in required_sections if s in tgsp_manifest]
    completeness = len(present_sections) / len(required_sections)
    
    return {
        "evidence_completeness": float(completeness),
        "signature_verified_rate": 1.0 if signature_verified else 0.0
    }

def compute_privacy_overhead(plaintext_latency_ms: float, privacy_latency_ms: float):
    """
    Computes latency overhead of privacy preservation (N2HE).
    """
    if plaintext_latency_ms <= 0:
        return 0.0
    
    overhead_ratio = (privacy_latency_ms - plaintext_latency_ms) / plaintext_latency_ms
    return float(overhead_ratio)

def aggregate_privacy_metrics(receipt_events: List[Any]):
    """
    Computes success rate and aggregate latency for privacy receipts.
    """
    if not receipt_events:
        return {"privacy_receipt_success_rate": 1.0}
    
    successes = [e for e in receipt_events if e.get("status") == "success"]
    rate = len(successes) / len(receipt_events)
    
    return {
        "privacy_receipt_success_rate": float(rate)
    }
