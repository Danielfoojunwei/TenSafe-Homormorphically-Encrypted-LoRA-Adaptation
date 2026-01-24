"""
Synthetic Dataset Generator - Generate Realistic Robot Traffic Traces

Creates synthetic but structured traffic traces for testing the attack
reproduction pipeline when real Kinova PCAPs are not available.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import logging

from .trace_loader import Packet, Flow

logger = logging.getLogger(__name__)


@dataclass
class ActionProfile:
    """Profile defining traffic characteristics for a robot action."""
    name: str
    
    # Command structure
    num_cartesian_commands: Tuple[int, int]  # (min, max)
    num_gripper_commands: Tuple[int, int]
    has_gripper_speed: bool
    gripper_speed_duration_s: Tuple[float, float]  # (min, max) if present
    
    # Timing characteristics
    action_duration_s: Tuple[float, float]
    inter_command_gap_s: Tuple[float, float]
    
    # Packet sizes
    command_packet_size: Tuple[int, int]
    feedback_packet_size: Tuple[int, int]


# Action profiles based on paper's characterization (Section 4)
ACTION_PROFILES = {
    "pick_and_place": ActionProfile(
        name="pick_and_place",
        num_cartesian_commands=(4, 8),
        num_gripper_commands=(2, 4),
        has_gripper_speed=True,
        gripper_speed_duration_s=(1.0, 3.0),
        action_duration_s=(8.0, 15.0),
        inter_command_gap_s=(0.5, 2.0),
        command_packet_size=(150, 300),
        feedback_packet_size=(200, 400),
    ),
    "pour_water": ActionProfile(
        name="pour_water",
        num_cartesian_commands=(3, 6),
        num_gripper_commands=(1, 3),
        has_gripper_speed=True,
        gripper_speed_duration_s=(2.0, 5.0),
        action_duration_s=(10.0, 20.0),
        inter_command_gap_s=(1.0, 3.0),
        command_packet_size=(150, 300),
        feedback_packet_size=(200, 400),
    ),
    "press_key": ActionProfile(
        name="press_key",
        num_cartesian_commands=(2, 4),
        num_gripper_commands=(0, 1),
        has_gripper_speed=False,
        gripper_speed_duration_s=(0.0, 0.0),
        action_duration_s=(2.0, 5.0),
        inter_command_gap_s=(0.2, 0.8),
        command_packet_size=(100, 200),
        feedback_packet_size=(150, 300),
    ),
    "toggle_switch": ActionProfile(
        name="toggle_switch",
        num_cartesian_commands=(3, 5),
        num_gripper_commands=(1, 2),
        has_gripper_speed=False,
        gripper_speed_duration_s=(0.0, 0.0),
        action_duration_s=(3.0, 7.0),
        inter_command_gap_s=(0.3, 1.0),
        command_packet_size=(120, 250),
        feedback_packet_size=(180, 350),
    ),
}


class SyntheticDatasetGenerator:
    """
    Generate synthetic robot traffic datasets for testing.
    
    Creates realistic traffic patterns based on paper's characterization
    of different robot actions (pick_and_place, pour_water, press_key, toggle_switch).
    """
    
    def __init__(
        self,
        profiles: Optional[Dict[str, ActionProfile]] = None,
        random_seed: int = 42
    ):
        """
        Initialize the generator.
        
        Args:
            profiles: Custom action profiles (uses defaults if None)
            random_seed: Random seed for reproducibility
        """
        self.profiles = profiles or ACTION_PROFILES.copy()
        self.rng = np.random.default_rng(random_seed)
    
    def _sample_range(self, range_tuple: Tuple[float, float]) -> float:
        """Sample uniformly from a (min, max) range."""
        return self.rng.uniform(range_tuple[0], range_tuple[1])
    
    def _sample_int_range(self, range_tuple: Tuple[int, int]) -> int:
        """Sample integer uniformly from a (min, max) range."""
        return self.rng.integers(range_tuple[0], range_tuple[1] + 1)
    
    def generate_action_trace(self, action_name: str) -> Flow:
        """
        Generate a single action trace.
        
        Args:
            action_name: Name of action to generate
            
        Returns:
            Flow object containing synthetic packets
        """
        if action_name not in self.profiles:
            raise ValueError(f"Unknown action: {action_name}. Available: {list(self.profiles.keys())}")
        
        profile = self.profiles[action_name]
        packets = []
        
        current_time = 0.0
        
        # Determine action structure
        num_cartesian = self._sample_int_range(profile.num_cartesian_commands)
        num_gripper = self._sample_int_range(profile.num_gripper_commands)
        total_commands = num_cartesian + num_gripper
        
        # Generate command sequence
        for i in range(total_commands):
            # Command packet (outgoing: controller → robot)
            cmd_size = self._sample_int_range(profile.command_packet_size)
            packets.append(Packet(
                timestamp=current_time,
                size=cmd_size,
                direction=1  # Outgoing
            ))
            
            # Small delay for processing
            current_time += self.rng.uniform(0.001, 0.005)
            
            # Feedback packet (incoming: robot → controller)
            fb_size = self._sample_int_range(profile.feedback_packet_size)
            packets.append(Packet(
                timestamp=current_time,
                size=fb_size,
                direction=-1  # Incoming
            ))
            
            # Inter-command gap
            if i < total_commands - 1:
                gap = self._sample_range(profile.inter_command_gap_s)
                current_time += gap
        
        # Add gripper speed activity if present
        if profile.has_gripper_speed:
            speed_duration = self._sample_range(profile.gripper_speed_duration_s)
            speed_start = current_time + 0.1
            
            # Gripper speed generates rapid, small packets
            num_speed_packets = int(speed_duration * 80)  # ~80 Hz rate
            for j in range(num_speed_packets):
                t = speed_start + j / 80.0
                
                # Small command
                packets.append(Packet(
                    timestamp=t,
                    size=self.rng.integers(50, 100),
                    direction=1
                ))
                # Small feedback
                packets.append(Packet(
                    timestamp=t + 0.005,
                    size=self.rng.integers(80, 150),
                    direction=-1
                ))
        
        # Sort by timestamp
        packets.sort(key=lambda p: p.timestamp)
        
        return Flow(packets=packets)
    
    def generate_dataset(
        self,
        samples_per_action: int = 50,
        actions: Optional[List[str]] = None
    ) -> Tuple[List[Flow], np.ndarray, List[str]]:
        """
        Generate a complete dataset with multiple samples per action.
        
        Args:
            samples_per_action: Number of samples per action class
            actions: List of actions to include (uses all if None)
            
        Returns:
            Tuple of (flows, labels, class_names)
        """
        actions = actions or list(self.profiles.keys())
        
        flows = []
        labels = []
        
        for action_idx, action_name in enumerate(actions):
            logger.info(f"Generating {samples_per_action} samples for {action_name}")
            for _ in range(samples_per_action):
                flow = self.generate_action_trace(action_name)
                flows.append(flow)
                labels.append(action_idx)
        
        # Shuffle
        indices = self.rng.permutation(len(flows))
        flows = [flows[i] for i in indices]
        labels = np.array([labels[i] for i in indices])
        
        return flows, labels, actions
    
    def save_dataset(
        self,
        output_dir: Path,
        samples_per_action: int = 50,
        actions: Optional[List[str]] = None
    ) -> None:
        """
        Generate and save dataset to disk.
        
        Creates JSON files for each trace and a manifest file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        flows, labels, action_names = self.generate_dataset(samples_per_action, actions)
        
        manifest = {
            "class_names": action_names,
            "samples_per_class": samples_per_action,
            "total_samples": len(flows),
            "traces": []
        }
        
        for i, (flow, label) in enumerate(zip(flows, labels)):
            trace_file = output_dir / f"trace_{i:04d}.json"
            
            trace_data = {
                "label": int(label),
                "action": action_names[label],
                "packets": [
                    {"t": p.timestamp, "s": p.size, "d": p.direction}
                    for p in flow.packets
                ]
            }
            
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f)
            
            manifest["traces"].append({
                "file": trace_file.name,
                "label": int(label),
                "action": action_names[label]
            })
        
        with open(output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved dataset to {output_dir}")
    
    def load_dataset(self, input_dir: Path) -> Tuple[List[Flow], np.ndarray, List[str]]:
        """Load a previously saved dataset."""
        input_dir = Path(input_dir)
        
        with open(input_dir / "manifest.json", 'r') as f:
            manifest = json.load(f)
        
        from .trace_loader import TraceLoader
        loader = TraceLoader()
        
        flows = []
        labels = []
        
        for trace_info in manifest["traces"]:
            flow = loader.load_synthetic(input_dir / trace_info["file"])
            flows.append(flow)
            labels.append(trace_info["label"])
        
        return flows, np.array(labels), manifest["class_names"]
