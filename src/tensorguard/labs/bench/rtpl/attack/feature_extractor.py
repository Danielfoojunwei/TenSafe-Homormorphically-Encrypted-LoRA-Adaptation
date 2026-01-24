"""
Feature Extractor - Build Classification Feature Vectors

Combines convolution and correlation statistics with summary statistics
to build feature vectors for robot action classification.

Based on arXiv:2312.06802 Section 6.2 Table 1(a) and 1(b).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

from .convolution_detector import ConvolutionDetector, Detection
from .correlation_detector import CorrelationDetector, Cluster, DEFAULT_GRIPPER_SPEED_PATTERN
from ..data.trace_loader import TraceFeatures, Flow

logger = logging.getLogger(__name__)


@dataclass
class ActionFeatures:
    """
    Complete feature vector for a robot action trace.
    
    Combines:
    - Convolution-based features (Table 1a)
    - Correlation-based features (Table 1b)  
    - Summary statistics (k-FP style)
    """
    # === Convolution-based (Paper Table 1a) ===
    cartesian_total_clusters: int = 0
    cartesian_avg_time_between: float = 0.0
    cartesian_total_length: float = 0.0
    cartesian_avg_length: float = 0.0
    cartesian_max_conv_value: float = 0.0
    cartesian_time_span: float = 0.0
    
    gripper_total_clusters: int = 0
    gripper_avg_time_between: float = 0.0
    gripper_total_length: float = 0.0
    gripper_avg_length: float = 0.0
    
    # === Correlation-based (Paper Table 1b) ===
    gripper_speed_cluster_count: int = 0
    gripper_speed_cumulative_length: float = 0.0
    gripper_speed_avg_length: float = 0.0
    gripper_speed_correlation_median: float = 0.0
    gripper_speed_length_skewness: float = 0.0
    gripper_speed_length_kurtosis: float = 0.0
    
    # === Summary Statistics (k-FP style) ===
    total_packets: int = 0
    total_incoming: int = 0
    total_outgoing: int = 0
    total_bytes: int = 0
    duration: float = 0.0
    
    # IAT percentiles
    iat_p20: float = 0.0
    iat_p50: float = 0.0
    iat_p80: float = 0.0
    
    # Packet size statistics
    avg_size_in: float = 0.0
    avg_size_out: float = 0.0
    size_std: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy feature vector."""
        return np.array([
            # Convolution features
            self.cartesian_total_clusters,
            self.cartesian_avg_time_between,
            self.cartesian_total_length,
            self.cartesian_avg_length,
            self.cartesian_max_conv_value,
            self.cartesian_time_span,
            self.gripper_total_clusters,
            self.gripper_avg_time_between,
            self.gripper_total_length,
            self.gripper_avg_length,
            # Correlation features
            self.gripper_speed_cluster_count,
            self.gripper_speed_cumulative_length,
            self.gripper_speed_avg_length,
            self.gripper_speed_correlation_median,
            self.gripper_speed_length_skewness,
            self.gripper_speed_length_kurtosis,
            # Summary statistics
            self.total_packets,
            self.total_incoming,
            self.total_outgoing,
            self.total_bytes,
            self.duration,
            self.iat_p20,
            self.iat_p50,
            self.iat_p80,
            self.avg_size_in,
            self.avg_size_out,
            self.size_std,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return [
            "cartesian_total_clusters",
            "cartesian_avg_time_between",
            "cartesian_total_length",
            "cartesian_avg_length",
            "cartesian_max_conv_value",
            "cartesian_time_span",
            "gripper_total_clusters",
            "gripper_avg_time_between",
            "gripper_total_length",
            "gripper_avg_length",
            "gripper_speed_cluster_count",
            "gripper_speed_cumulative_length",
            "gripper_speed_avg_length",
            "gripper_speed_correlation_median",
            "gripper_speed_length_skewness",
            "gripper_speed_length_kurtosis",
            "total_packets",
            "total_incoming",
            "total_outgoing",
            "total_bytes",
            "duration",
            "iat_p20",
            "iat_p50",
            "iat_p80",
            "avg_size_in",
            "avg_size_out",
            "size_std",
        ]


class FeatureExtractor:
    """
    Extract classification features from robot traffic traces.
    
    Implements the feature extraction pipeline from arXiv:2312.06802 Section 6.2.
    
    Features include:
    - Convolution-based: cluster counts, timing, max values (Table 1a)
    - Correlation-based: recurring pattern statistics (Table 1b)
    - Summary statistics: packet counts, IAT percentiles, size stats (k-FP style)
    """
    
    def __init__(
        self,
        conv_threshold: float = 0.9,
        corr_threshold: float = 0.5,
        min_cluster_duration_s: float = 1.0
    ):
        """
        Initialize the feature extractor.
        
        Args:
            conv_threshold: Convolution detection threshold
            corr_threshold: Correlation detection threshold
            min_cluster_duration_s: Minimum cluster duration for correlation
        """
        self.conv_detector = ConvolutionDetector(threshold=conv_threshold)
        self.corr_detector = CorrelationDetector(
            min_duration_s=min_cluster_duration_s,
            correlation_threshold=corr_threshold
        )
    
    def extract(self, flow: Flow) -> ActionFeatures:
        """
        Extract all features from a traffic flow.
        
        Args:
            flow: Traffic flow to analyze
            
        Returns:
            ActionFeatures object with all extracted features
        """
        # Get basic trace features
        trace_features = TraceFeatures.from_flow(flow)
        
        # Get signals for analysis
        size_signal = flow.get_size_signal()
        timestamps = np.array([p.timestamp for p in flow.packets])
        
        # Initialize features with summary statistics
        features = ActionFeatures(
            total_packets=trace_features.total_packets,
            total_incoming=trace_features.total_incoming,
            total_outgoing=trace_features.total_outgoing,
            total_bytes=trace_features.total_bytes_in + trace_features.total_bytes_out,
            duration=trace_features.duration,
            iat_p20=trace_features.iat_p20,
            iat_p50=trace_features.iat_p50,
            iat_p80=trace_features.iat_p80,
            avg_size_in=trace_features.avg_size_in,
            avg_size_out=trace_features.avg_size_out,
            size_std=float(np.std(trace_features.sizes)) if len(trace_features.sizes) > 0 else 0.0,
        )
        
        if len(size_signal) == 0:
            return features
        
        # === Convolution-based detection ===
        # Cartesian commands
        try:
            conv_output, cartesian_detections = self.conv_detector.convolve(
                size_signal, "cartesian", timestamps
            )
            features = self._add_convolution_features(
                features, cartesian_detections, conv_output, "cartesian"
            )
        except Exception as e:
            logger.warning(f"Cartesian detection failed: {e}")
        
        # Gripper position commands
        try:
            conv_output, gripper_detections = self.conv_detector.convolve(
                size_signal, "gripper_position", timestamps
            )
            features = self._add_convolution_features(
                features, gripper_detections, conv_output, "gripper"
            )
        except Exception as e:
            logger.warning(f"Gripper detection failed: {e}")
        
        # === Correlation-based detection ===
        # Gripper speed (recurring pattern)
        try:
            clusters = self.corr_detector.detect_clusters(
                size_signal, DEFAULT_GRIPPER_SPEED_PATTERN, timestamps
            )
            cluster_stats = self.corr_detector.extract_cluster_statistics(clusters)
            
            features.gripper_speed_cluster_count = cluster_stats["total_clusters"]
            features.gripper_speed_cumulative_length = cluster_stats["cumulative_length"]
            features.gripper_speed_avg_length = cluster_stats["avg_length"]
            features.gripper_speed_correlation_median = cluster_stats["correlation_median"]
            features.gripper_speed_length_skewness = cluster_stats.get("length_skewness", 0.0)
            features.gripper_speed_length_kurtosis = cluster_stats.get("length_kurtosis", 0.0)
        except Exception as e:
            logger.warning(f"Correlation detection failed: {e}")
        
        return features
    
    def _add_convolution_features(
        self,
        features: ActionFeatures,
        detections: List[Detection],
        conv_output: np.ndarray,
        prefix: str
    ) -> ActionFeatures:
        """Add convolution-based features for a command type."""
        if not detections:
            return features
        
        # Calculate cluster statistics using vectorized numpy operations
        timestamps = np.array([d.timestamp for d in detections])
        confidences = np.array([d.confidence for d in detections])

        # Time gaps between detections using vectorized numpy diff (5-10x faster)
        time_gaps = np.diff(timestamps) if len(timestamps) > 1 else np.array([])

        if prefix == "cartesian":
            features.cartesian_total_clusters = len(detections)
            features.cartesian_avg_time_between = float(np.mean(time_gaps)) if len(time_gaps) > 0 else 0.0
            features.cartesian_total_length = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
            features.cartesian_avg_length = features.cartesian_total_length / len(detections)
            features.cartesian_max_conv_value = float(np.max(confidences))
            features.cartesian_time_span = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        elif prefix == "gripper":
            features.gripper_total_clusters = len(detections)
            features.gripper_avg_time_between = float(np.mean(time_gaps)) if len(time_gaps) > 0 else 0.0
            features.gripper_total_length = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
            features.gripper_avg_length = features.gripper_total_length / len(detections)
        
        return features
    
    def extract_batch(self, flows: List[Flow]) -> np.ndarray:
        """
        Extract features from multiple flows.
        
        Args:
            flows: List of traffic flows
            
        Returns:
            2D array of shape (n_flows, n_features)
        """
        features_list = [self.extract(flow).to_vector() for flow in flows]
        return np.stack(features_list)
