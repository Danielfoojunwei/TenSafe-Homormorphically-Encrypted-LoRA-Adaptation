"""
Correlation Detector - Recurring Pattern Detection via Correlation

Implements the correlation-based pattern detection from arXiv:2312.06802 Section 6.1.
Used for detecting recurring patterns (e.g., gripper speed commands) where correlation
coefficient analysis reduces noise and false positives.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """A cluster of recurring pattern occurrences."""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    correlation_values: np.ndarray
    mean_correlation: float
    
    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


class CorrelationDetector:
    """
    Detect recurring patterns via correlation coefficient (arXiv:2312.06802 Section 6.1).
    
    The paper states: "We employ the correlation coefficient to identify command
    messages that exhibit varying recurring patterns (e.g., the gripper speed command)
    as a means to reduce noise and false positives."
    
    Key insight: "A cluster must consecutively last over 1 second to be considered
    a positive match" - this aligns with inherent gripper speed command activity.
    """
    
    def __init__(
        self,
        min_duration_s: float = 1.0,
        correlation_threshold: float = 0.5,
        window_size: int = 20
    ):
        """
        Initialize the correlation detector.
        
        Args:
            min_duration_s: Minimum cluster duration to be considered valid
            correlation_threshold: Minimum correlation coefficient for match
            window_size: Window size for correlation computation
        """
        self.min_duration_s = min_duration_s
        self.correlation_threshold = correlation_threshold
        self.window_size = window_size
    
    def compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int = 0
    ) -> float:
        """
        Compute correlation coefficient between two signals at given lag.
        
        Paper formula: r[m] = Σ(x[n] - μx)(y[n+m] - μy) / (σx * σy * N)
        
        Args:
            x: First signal (known pattern)
            y: Second signal (observed traffic)
            lag: Lag m for cross-correlation
            
        Returns:
            Correlation coefficient in range [-1, 1]
        """
        if len(x) == 0 or len(y) == 0:
            return 0.0
        
        # Align signals with lag
        if lag >= 0:
            x_aligned = x[:len(x)-lag] if lag > 0 else x
            y_aligned = y[lag:] if lag > 0 else y
        else:
            x_aligned = x[-lag:]
            y_aligned = y[:len(y)+lag]
        
        # Ensure equal length
        min_len = min(len(x_aligned), len(y_aligned))
        if min_len == 0:
            return 0.0
        
        x_aligned = x_aligned[:min_len]
        y_aligned = y_aligned[:min_len]
        
        # Compute correlation coefficient
        x_mean = np.mean(x_aligned)
        y_mean = np.mean(y_aligned)
        x_std = np.std(x_aligned)
        y_std = np.std(y_aligned)
        
        if x_std == 0 or y_std == 0:
            return 0.0
        
        covariance = np.mean((x_aligned - x_mean) * (y_aligned - y_mean))
        return covariance / (x_std * y_std)
    
    def sliding_correlation(
        self,
        signal_data: np.ndarray,
        pattern: np.ndarray
    ) -> np.ndarray:
        """
        Compute sliding window correlation between signal and pattern.
        
        Returns correlation coefficient at each position.
        """
        if len(signal_data) < len(pattern):
            return np.array([])
        
        correlations = []
        window = len(pattern)
        
        for i in range(len(signal_data) - window + 1):
            segment = signal_data[i:i + window]
            r = self.compute_correlation(segment, pattern)
            correlations.append(r)
        
        return np.array(correlations)
    
    def detect_clusters(
        self,
        signal_data: np.ndarray,
        pattern: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[Cluster]:
        """
        Detect clusters of recurring patterns in the signal.
        
        Args:
            signal_data: Input traffic signal
            pattern: Reference pattern to detect
            timestamps: Optional timestamps for each sample
            
        Returns:
            List of detected clusters meeting duration threshold
        """
        if len(signal_data) < self.window_size:
            return []
        
        # Compute sliding correlation
        correlations = self.sliding_correlation(signal_data, pattern)
        
        if len(correlations) == 0:
            return []
        
        # Find regions above threshold
        above_threshold = correlations > self.correlation_threshold
        
        # Find cluster boundaries
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(correlations)]])
        
        # Build clusters
        clusters = []
        for start, end in zip(starts, ends):
            if timestamps is not None and len(timestamps) > end:
                start_time = timestamps[start]
                end_time = timestamps[min(end, len(timestamps) - 1)]
                duration = end_time - start_time
            else:
                # Estimate based on typical 100Hz sample rate
                sample_rate = 100.0
                start_time = start / sample_rate
                end_time = end / sample_rate
                duration = (end - start) / sample_rate
            
            # Filter by minimum duration (paper: 1 second)
            if duration >= self.min_duration_s:
                cluster_corr = correlations[start:end]
                clusters.append(Cluster(
                    start_idx=int(start),
                    end_idx=int(end),
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(duration),
                    correlation_values=cluster_corr,
                    mean_correlation=float(np.mean(cluster_corr))
                ))
        
        logger.debug(f"Detected {len(clusters)} clusters above duration threshold")
        return clusters
    
    def extract_cluster_statistics(self, clusters: List[Cluster]) -> dict:
        """
        Extract summary statistics from detected clusters.
        
        Paper Table 1(b): Key features include cumulative length,
        average length, time gaps between clusters, skewness, kurtosis.
        """
        if not clusters:
            return {
                "total_clusters": 0,
                "cumulative_length": 0.0,
                "avg_length": 0.0,
                "avg_time_gap": 0.0,
                "correlation_median": 0.0,
                "time_span": 0.0,
            }
        
        lengths = [c.duration for c in clusters]
        
        # Time gaps between consecutive clusters
        time_gaps = []
        for i in range(1, len(clusters)):
            gap = clusters[i].start_time - clusters[i-1].end_time
            time_gaps.append(gap)
        
        # All correlation values
        all_correlations = np.concatenate([c.correlation_values for c in clusters])
        
        return {
            "total_clusters": len(clusters),
            "cumulative_length": sum(lengths),
            "avg_length": np.mean(lengths),
            "avg_time_gap": np.mean(time_gaps) if time_gaps else 0.0,
            "correlation_median": float(np.median(all_correlations)),
            "time_span": clusters[-1].end_time - clusters[0].start_time,
            "length_skewness": float(self._skewness(lengths)),
            "length_kurtosis": float(self._kurtosis(lengths)),
        }
    
    def _skewness(self, data: List[float]) -> float:
        """Compute skewness of data."""
        if len(data) < 3:
            return 0.0
        arr = np.array(data)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))
    
    def _kurtosis(self, data: List[float]) -> float:
        """Compute excess kurtosis of data."""
        if len(data) < 4:
            return 0.0
        arr = np.array(data)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3)


# Default patterns for common robot commands
DEFAULT_GRIPPER_SPEED_PATTERN = np.array([
    0.3, 0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 0.5, 0.3
])
