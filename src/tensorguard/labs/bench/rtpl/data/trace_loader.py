"""
Trace Loader - PCAP Parsing and Flow Reconstruction

Loads encrypted robot traffic traces and extracts metadata features
for traffic analysis attack reproduction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path
import struct
import logging

logger = logging.getLogger(__name__)


@dataclass
class Packet:
    """Single network packet metadata."""
    timestamp: float  # Seconds since epoch
    size: int  # Bytes
    direction: int  # +1 = outgoing (controller→robot), -1 = incoming (robot→controller)
    
    @property
    def is_outgoing(self) -> bool:
        return self.direction == 1


@dataclass
class Flow:
    """Bidirectional flow between controller and robot."""
    packets: List[Packet] = field(default_factory=list)
    
    @property
    def start_time(self) -> float:
        return self.packets[0].timestamp if self.packets else 0.0
    
    @property
    def end_time(self) -> float:
        return self.packets[-1].timestamp if self.packets else 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def get_size_signal(self, direction: Optional[int] = None) -> np.ndarray:
        """Get packet sizes as a signal array, optionally filtered by direction."""
        if direction is None:
            return np.array([p.size for p in self.packets])
        return np.array([p.size for p in self.packets if p.direction == direction])
    
    def get_iat_signal(self) -> np.ndarray:
        """Get inter-arrival times (IAT) as a signal array."""
        if len(self.packets) < 2:
            return np.array([])
        timestamps = np.array([p.timestamp for p in self.packets])
        return np.diff(timestamps)
    
    def get_direction_signal(self) -> np.ndarray:
        """Get packet directions as a signal array (+1/-1)."""
        return np.array([p.direction for p in self.packets])


@dataclass
class TraceFeatures:
    """Extracted features from a traffic trace for classification."""
    # Raw signals
    sizes: np.ndarray
    iats: np.ndarray
    directions: np.ndarray
    
    # Summary statistics
    total_packets: int
    total_incoming: int
    total_outgoing: int
    total_bytes_in: int
    total_bytes_out: int
    duration: float
    
    # Percentile statistics for IAT
    iat_p20: float
    iat_p50: float
    iat_p80: float
    
    # Packet size statistics
    avg_size_in: float
    avg_size_out: float
    
    @classmethod
    def from_flow(cls, flow: Flow) -> "TraceFeatures":
        """Extract features from a Flow object."""
        sizes = flow.get_size_signal()
        iats = flow.get_iat_signal()
        directions = flow.get_direction_signal()
        
        incoming = [p for p in flow.packets if p.direction == -1]
        outgoing = [p for p in flow.packets if p.direction == 1]
        
        return cls(
            sizes=sizes,
            iats=iats,
            directions=directions,
            total_packets=len(flow.packets),
            total_incoming=len(incoming),
            total_outgoing=len(outgoing),
            total_bytes_in=sum(p.size for p in incoming),
            total_bytes_out=sum(p.size for p in outgoing),
            duration=flow.duration,
            iat_p20=float(np.percentile(iats, 20)) if len(iats) > 0 else 0.0,
            iat_p50=float(np.percentile(iats, 50)) if len(iats) > 0 else 0.0,
            iat_p80=float(np.percentile(iats, 80)) if len(iats) > 0 else 0.0,
            avg_size_in=np.mean([p.size for p in incoming]) if incoming else 0.0,
            avg_size_out=np.mean([p.size for p in outgoing]) if outgoing else 0.0,
        )


class TraceLoader:
    """
    Load and parse encrypted robot traffic traces.
    
    Supports:
    - PCAP files (via dpkt)
    - Synthetic trace format (JSON)
    - Raw packet list format
    """
    
    def __init__(self, inactivity_threshold_s: float = 0.5):
        """
        Initialize trace loader.
        
        Args:
            inactivity_threshold_s: Gap threshold for burst segmentation (seconds)
        """
        self.inactivity_threshold = inactivity_threshold_s
    
    def load_pcap(self, path: Path) -> Flow:
        """
        Load a PCAP file and extract packet metadata.
        
        Note: Requires dpkt or scapy to be installed.
        """
        try:
            import dpkt
        except ImportError:
            raise ImportError("dpkt is required for PCAP parsing. Install with: pip install dpkt")
        
        packets = []
        with open(path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                # Try pcapng format
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)
            
            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    
                    # Determine direction based on port or IP
                    # Convention: lower port = server (robot), higher port = client (controller)
                    direction = 1  # Default to outgoing
                    if hasattr(ip.data, 'sport') and hasattr(ip.data, 'dport'):
                        if ip.data.sport < ip.data.dport:
                            direction = -1  # Incoming from robot
                    
                    packets.append(Packet(
                        timestamp=timestamp,
                        size=len(buf),
                        direction=direction
                    ))
                except Exception as e:
                    logger.debug(f"Skipping malformed packet: {e}")
                    continue
        
        return Flow(packets=packets)
    
    def load_synthetic(self, path: Path) -> Flow:
        """Load a synthetic trace from JSON format."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        packets = [
            Packet(timestamp=p['t'], size=p['s'], direction=p['d'])
            for p in data['packets']
        ]
        return Flow(packets=packets)
    
    def load_from_packets(self, packets: List[Tuple[float, int, int]]) -> Flow:
        """
        Load from raw packet list.
        
        Args:
            packets: List of (timestamp, size, direction) tuples
        """
        return Flow(packets=[
            Packet(timestamp=t, size=s, direction=d)
            for t, s, d in packets
        ])
    
    def segment_bursts(self, flow: Flow) -> List[Flow]:
        """
        Segment a flow into bursts based on inactivity threshold.
        
        Returns list of Flow objects, each representing a burst of activity.
        """
        if not flow.packets:
            return []
        
        bursts = []
        current_burst = [flow.packets[0]]
        
        for i in range(1, len(flow.packets)):
            gap = flow.packets[i].timestamp - flow.packets[i-1].timestamp
            if gap > self.inactivity_threshold:
                bursts.append(Flow(packets=current_burst))
                current_burst = [flow.packets[i]]
            else:
                current_burst.append(flow.packets[i])
        
        if current_burst:
            bursts.append(Flow(packets=current_burst))
        
        return bursts
    
    def extract_features(self, flow: Flow) -> TraceFeatures:
        """Extract classification features from a flow."""
        return TraceFeatures.from_flow(flow)
