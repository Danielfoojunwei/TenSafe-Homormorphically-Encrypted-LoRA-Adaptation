"""
Robotics Data Layer - MCAP Reader

Provides unified access to Foxglove MCAP files.
Relies on 'mcap' and 'mcap-ros2-support' libraries.
"""

import logging
from pathlib import Path
from typing import List, Generator, Tuple, Optional, Any
import time

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    HAS_MCAP = True
except ImportError:
    HAS_MCAP = False

logger = logging.getLogger(__name__)

class McapReader:
    """
    Reads MCAP files.
    """
    
    def __init__(self, file_path: str):
        if not HAS_MCAP:
            raise ImportError(
                "MCAP libraries not found. Install with: pip install mcap mcap-ros2-support"
            )
        self.path = Path(file_path)
        self.stream = None
        self.reader = None
        self.summary = None

    def __enter__(self):
        self.stream = open(self.path, "rb")
        self.reader = make_reader(self.stream, decoder_factories=[DecoderFactory()])
        self.summary = self.reader.get_summary()
        logger.info(f"Opened MCAP: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.close()

    def get_topics(self) -> List[str]:
        if not self.summary: return []
        return [c.topic for c in self.summary.channels.values()]

    def read_messages(
        self, 
        topics: Optional[List[str]] = None, 
        start_ns: Optional[int] = None, 
        end_ns: Optional[int] = None
    ) -> Generator[Tuple[str, int, Any, str], None, None]:
        """
        Yields (topic, timestamp_ns, decoded_msg, type_name).
        """
        if not self.reader:
            raise RuntimeError("MCAP not open")

        for schema, channel, message, decoded_msg in self.reader.iter_decoded_messages(
            topics=topics, start_time=start_ns, end_time=end_ns
        ):
            yield channel.topic, message.log_time, decoded_msg, schema.name
