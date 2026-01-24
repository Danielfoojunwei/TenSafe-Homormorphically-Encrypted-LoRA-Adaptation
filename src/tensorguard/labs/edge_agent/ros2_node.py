"""
Edge Agent ROS 2 Node

Subscribes to ROS 2 topics and buffers them into the Spooler.
"""

import sys
import time
import json
import logging
from typing import List

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from cv_bridge import CvBridge # Optional, for Image conversion
    # Import standard messages
    from sensor_msgs.msg import Image, PointCloud2, JointState
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from .spooler import Spooler

logger = logging.getLogger(__name__)

class AgentNode(Node if HAS_ROS2 else object):
    def __init__(self, spooler: Spooler, topics_config: List[dict]):
        if not HAS_ROS2:
            raise ImportError("ROS 2 (rclpy) not found.")
            
        super().__init__('tensorguard_edge_agent')
        self.spooler = spooler
        self.subs = []
        
        # Mapping string types to classes
        type_map = {
            'sensor_msgs/Image': Image,
            'sensor_msgs/PointCloud2': PointCloud2,
            'sensor_msgs/JointState': JointState,
            'nav_msgs/Odometry': Odometry,
            'tf2_msgs/TFMessage': TFMessage
        }
        
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        for t in topics_config:
            topic_name = t['name']
            msg_type_str = t['type']
            msg_class = type_map.get(msg_type_str)
            
            if msg_class:
                self.create_subscription(
                    msg_class, 
                    topic_name, 
                    lambda msg, tn=topic_name: self.listener_callback(msg, tn), 
                    qos
                )
                logger.info(f"Subscribed to {topic_name} ({msg_type_str})")
            else:
                logger.warning(f"Unknown message type: {msg_type_str}")

    def listener_callback(self, msg, topic: str):
        """Serialize and spool."""
        try:
            # Naive serialization: Convert known types to dict
            # In prod, we'd use rosidl_runtime_py.convert.message_to_ordereddict
            # or just pickle the raw bytes if we have a robust deserializer on server.
            # For this MVP, we store a simplified dict or raw bytes.
            
            # Simple timestamp extraction
            ts = time.time_ns()
            if hasattr(msg, 'header'):
                ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                
            # Payload payload
            # Optimization: Just store raw fields for now to verify flow
            payload = {
                "type": str(type(msg)),
                "fields": "redacted_for_mvp" # Actual logic would extract safe fields
            }
            
            self.spooler.enqueue(
                topic=topic,
                timestamp_ns=ts,
                payload=payload
            )
        except Exception as e:
            logger.error(f"Callback error on {topic}: {e}")

def main(args=None):
    if not HAS_ROS2:
        print("Error: ROS 2 environment not sourced.")
        sys.exit(1)
        
    rclpy.init(args=args)
    
    # Config (would come from file)
    topics = [
        {"name": "/cmd_vel", "type": "geometry_msgs/Twist"}, # Not in map, will warn
        {"name": "/odom", "type": "nav_msgs/Odometry"},
        {"name": "/tf", "type": "tf2_msgs/TFMessage"}
    ]
    
    spooler = Spooler("agent_spool.db")
    node = AgentNode(spooler, topics)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
