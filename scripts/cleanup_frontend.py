
import os
from pathlib import Path

files = [
    "frontend/src/components/labs/ContinuousDashboard.vue",
    "frontend/src/components/labs/RouteTimeline.vue",
    "frontend/src/components/labs/RouteDetails.vue",
    "frontend/src/components/labs/ReleasesRollback.vue",
    "frontend/src/components/labs/EvidenceDiff.vue",
    "frontend/src/components/labs/RouteWizard.vue"
]

root = Path(".")
for f in files:
    p = root / f
    if p.exists():
        p.unlink()
        print(f"Deleted: {p}")
    else:
        print(f"Not found: {p}")
