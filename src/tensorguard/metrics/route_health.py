from typing import Dict, Any, List
from .schemas import MetricName

def compute_route_health_score(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes an explainable health score (0..100) for a route.
    """
    score = 0
    reasons = []
    
    # 1. Accuracy Trend (30 points)
    acc = metrics.get(MetricName.AVG_ACCURACY.value, 0)
    if acc > 0.9:
        score += 30
        reasons.append("Accuracy is high (>90%)")
    elif acc > 0.8:
        score += 20
        reasons.append("Accuracy is acceptable (>80%)")
    else:
        reasons.append("Accuracy is below target")
        
    # 2. Forgetting Budget (30 points)
    forgetting = metrics.get(MetricName.FORGETTING_MEAN.value, 0)
    if forgetting < 0.05:
        score += 30
        reasons.append("Forgetting is minimal (<5%)")
    elif forgetting < 0.1:
        score += 15
        reasons.append("Forgetting is within budget (<10%)")
    else:
        reasons.append("Significant forgetting detected")
        
    # 3. Evidence & Trust (20 points)
    completeness = metrics.get(MetricName.EVIDENCE_COMPLETENESS.value, 0)
    if completeness >= 1.0:
        score += 20
        reasons.append("Trust evidence is complete")
    elif completeness > 0.5:
        score += 10
        reasons.append("Partial trust evidence available")
        
    # 4. Operational Stability (20 points)
    rollback_time = metrics.get(MetricName.ROLLBACK_TIME_MS.value, 500)
    if rollback_time < 1000:
        score += 20
        reasons.append("Rollback is highly responsive")
    else:
        score += 10
        reasons.append("Rollback latency is elevated")

    # Map score to status
    status = "healthy"
    if score < 50:
        status = "critical"
    elif score < 80:
        status = "warning"
        
    return {
        "score": score,
        "status": status,
        "reasons": reasons
    }
