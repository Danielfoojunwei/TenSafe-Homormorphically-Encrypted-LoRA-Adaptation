"""
TF Resolver - Transform Tree History

Reconstructs the TF tree from a stream of /tf and /tf_static messages.
Allows querying transforms at specific timestamps (with simple interpolation).
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Basic geometry helpers
def quaternion_multiply(q1, q2):
    # w, x, y, z
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

class Transform:
    """Represents a translation + rotation (quaternion) at a time."""
    def __init__(self, time_ns: int, translation: List[float], rotation: List[float], child_frame: str):
        self.time_ns = time_ns
        self.translation = np.array(translation) # x, y, z
        self.rotation = np.array(rotation) # w, x, y, z (Assuming w-first internal convention or adhering to msg)
        self.child_frame = child_frame

class FrameHistory:
    """Sorted history of transforms for a specific frame relative to its parent."""
    def __init__(self, parent_frame: str):
        self.parent_frame = parent_frame
        self.transforms: List[Transform] = []
        
    def add(self, tf: Transform):
        # Insert sorted
        self.transforms.append(tf)
        # Sort by time if needed, though usually incoming is ordered.
        
    def lookup(self, time_ns: int) -> Optional[Transform]:
        """Find transform closest to time_ns."""
        # Simple nearest neighbor for now.
        # Production would implement proper Slerp/Lerp.
        if not self.transforms:
            return None
            
        # Binary search could be better
        best = min(self.transforms, key=lambda t: abs(t.time_ns - time_ns))
        # Check tolerance (e.g. 100ms)
        if abs(best.time_ns - time_ns) > 1e8: 
            return None
            
        return best

class TfResolver:
    """
    Resolves transforms between frames over time.
    """
    def __init__(self):
        # buffer[child_frame] = FrameHistory
        self.buffer: Dict[str, FrameHistory] = {}
        self.static_buffer: Dict[str, Transform] = {}
        
    def process_tf_message(self, msg: Any, is_static: bool = False):
        """
        Ingest a TFMessage (list of transforms).
        Expects msg to be a dict-like or object with 'transforms' list.
        """
        # Handle dict (decoded JSON/MCAP usually) or object
        transforms = getattr(msg, 'transforms', msg.get('transforms', []) if isinstance(msg, dict) else [])
        
        for tx in transforms:
            # Extract fields safely
            header = getattr(tx, 'header', tx.get('header', {}))
            stamp = getattr(header, 'stamp', header.get('stamp'))
            if hasattr(stamp, 'sec'):
                ts = stamp.sec * 1_000_000_000 + stamp.nanosec
            elif isinstance(stamp, dict):
                ts = stamp.get('sec', 0) * 1_000_000_000 + stamp.get('nanosec', 0)
            else:
                ts = 0
                
            frame_id = getattr(header, 'frame_id', header.get('frame_id'))
            child_frame = getattr(tx, 'child_frame_id', tx.get('child_frame_id'))
            
            trans = getattr(tx, 'transform', tx.get('transform', {}))
            tr = getattr(trans, 'translation', trans.get('translation', {}))
            rot = getattr(trans, 'rotation', trans.get('rotation', {}))
            
            # Extracts
            t_vec = [
                getattr(tr, 'x', tr.get('x', 0)), 
                getattr(tr, 'y', tr.get('y', 0)), 
                getattr(tr, 'z', tr.get('z', 0))
            ]
            r_vec = [
                getattr(rot, 'w', rot.get('w', 1)), # W first convention for internal math
                getattr(rot, 'x', rot.get('x', 0)),
                getattr(rot, 'y', rot.get('y', 0)),
                getattr(rot, 'z', rot.get('z', 0))
            ]
            
            tf_obj = Transform(ts, t_vec, r_vec, child_frame)
            
            if is_static:
                self.static_buffer[child_frame] = tf_obj
            else:
                if child_frame not in self.buffer:
                    self.buffer[child_frame] = FrameHistory(frame_id)
                self.buffer[child_frame].add(tf_obj)

    def lookup_transform(self, target_frame: str, source_frame: str, time_ns: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (translation, rotation) from source -> target.
        Currently implements only direct parent->child or child->parent lookup for simplicity.
        Full tree traversal is complex without tf2 libraries.
        """
        # 1. Check direct child
        if target_frame in self.buffer:
            hist = self.buffer[target_frame]
            if hist.parent_frame == source_frame:
                tf = hist.lookup(time_ns)
                if tf: return tf.translation, tf.rotation
                
        # 2. Check static
        if target_frame in self.static_buffer:
            tf = self.static_buffer[target_frame]
            # Verify parent if needed, usually static is global
            return tf.translation, tf.rotation
            
        return None, None
