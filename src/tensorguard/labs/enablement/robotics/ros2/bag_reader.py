"""
Robotics Data Layer - ROS 2 Bag Reader

Provides unified access to rosbag2 (.db3) files.
Relies on 'rosbags' library to avoid full ROS 2 stack requirements.
"""

import logging
from pathlib import Path
from typing import List, Generator, Tuple, Optional, Any, Dict
import json

# Try importing rosbags (lightweight reader)
try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import verify_types, deserialize_cdr, ros1_to_cdr
    from rosbags.typesys import get_types_from_msg, register_types
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False
    Reader = None

logger = logging.getLogger(__name__)

class RosbagReader:
    """
    Reads ROS 2 .db3 bag files using rosbags library.
    Normalizes access to timestamped messages.
    """
    
    def __init__(self, bag_path: str):
        if not HAS_ROSBAGS:
            raise ImportError(
                "rosbags library not found. Install with: pip install rosbags"
            )
        self.path = Path(bag_path)
        self.reader = None
        self.connections = {}
        
    def __enter__(self):
        self.reader = Reader(self.path)
        self.reader.open()
        
        # Index connections by topic
        for conn in self.reader.connections:
            self.connections[conn.topic] = conn
            
        logger.info(f"Opened bag: {self.path} ({self.reader.message_count} messages)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.reader:
            self.reader.close()

    def get_topics(self) -> List[str]:
        """List all topics in the bag."""
        return list(self.connections.keys())

    def get_type_info(self, topic: str) -> str:
        """Get message type for a topic."""
        if topic in self.connections:
            return self.connections[topic].msgtype
        return "unknown"

    def read_messages(
        self, 
        topics: Optional[List[str]] = None, 
        start_ns: Optional[int] = None, 
        end_ns: Optional[int] = None
    ) -> Generator[Tuple[str, int, Any], None, None]:
        """
        Yields (topic, timestamp_ns, raw_bytes) or deserialized object.
        Currently returns (topic, timestamp_ns, raw_bytes, type_name).
        """
        if not self.reader:
            raise RuntimeError("Bag not open. Use 'with' context.")

        connections = [
            self.connections[t] for t in topics if t in self.connections
        ] if topics else list(self.connections.values())

        for conn, timestamp, rawdata in self.reader.messages(connections=connections, start=start_ns, stop=end_ns):
            yield conn.topic, timestamp, rawdata, conn.msgtype

    def get_start_time(self) -> int:
        return self.reader.start_time if self.reader else 0

    def get_end_time(self) -> int:
        return self.reader.end_time if self.reader else 0

    def get_duration(self) -> float:
        return (self.get_end_time() - self.get_start_time()) / 1e9

    def get_message_count(self) -> int:
        return self.reader.message_count

