"""
Incident Slicer

Creates time-window slices from a continuous stream based on trigger rules.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Slice:
    start_ns: int
    end_ns: int
    trigger_type: str
    metadata: Dict[str, Any]

class IncidentSlicer:
    """
    Monitors stream statistics or external triggers to create slices.
    """
    def __init__(self, pre_window_sec: float = 5.0, post_window_sec: float = 5.0):
        self.pre_msg_ns = int(pre_window_sec * 1e9)
        self.post_msg_ns = int(post_window_sec * 1e9)
        self.slices: List[Slice] = []
        
    def add_manual_trigger(self, timestamp_ns: int, reason: str):
        """Register a manual slice event."""
        start = timestamp_ns - self.pre_msg_ns
        end = timestamp_ns + self.post_msg_ns
        self.slices.append(Slice(start, end, reason, {"manual": True}))
        
    def detect_collisions(self, odometry_stream):
        """Example: Detect abrupt stops."""
        pass
        
    def export_index(self) -> List[Dict]:
        return [
            {
                "start_ns": s.start_ns,
                "end_ns": s.end_ns,
                "trigger": s.trigger_type,
                "meta": s.metadata
            }
            for s in self.slices
        ]
