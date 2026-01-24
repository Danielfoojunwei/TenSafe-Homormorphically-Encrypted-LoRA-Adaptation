"""
Convolution Detector - Command Pattern Detection via Convolution

Implements the convolution-based command detection from arXiv:2312.06802 Section 6.1.
Detects command patterns (Cartesian movements, gripper commands) by convolving
traffic signals with pre-defined kernels.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy import signal
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A detected command pattern occurrence."""
    position: int  # Index in signal where detection occurred
    timestamp: float  # Estimated timestamp
    confidence: float  # Convolution value at detection
    pattern_type: str  # Type of pattern detected (e.g., "cartesian", "gripper")


@dataclass
class SweepResult:
    """Result of a threshold sweep."""
    threshold: float
    num_detections: int
    false_positive_rate: float  # Estimated based on detection density


class ConvolutionDetector:
    """
    Detect command patterns via convolution (arXiv:2312.06802 Section 6.1).
    
    The paper shows that different robot commands generate distinct traffic
    sub-patterns that can be detected using convolution. A kernel representing
    the expected pattern is convolved with the observed traffic signal.
    Spikes in the output indicate pattern matches.
    
    Key insight from paper: "The specific choice of convolution kernel does not
    significantly impact accuracy" - the method is robust to kernel variations.
    """
    
    # Default kernels based on paper's characterization
    DEFAULT_KERNELS = {
        "cartesian": np.array([1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0]),
        "gripper_position": np.array([0.5, 1.0, 1.5, 1.0, 0.5]),
        "gripper_speed": np.array([0.3, 0.6, 1.0, 1.0, 1.0, 0.6, 0.3]),
    }
    
    def __init__(
        self,
        kernels: Optional[Dict[str, np.ndarray]] = None,
        threshold: float = 0.9,
        normalize: bool = True
    ):
        """
        Initialize the convolution detector.
        
        Args:
            kernels: Dict mapping pattern names to kernel arrays.
                    If None, uses default kernels.
            threshold: Detection threshold (paper finds 0.9 optimal)
            normalize: Whether to normalize signals and kernels
        """
        self.kernels = kernels or self.DEFAULT_KERNELS.copy()
        self.threshold = threshold
        self.normalize = normalize
    
    def _normalize_signal(self, sig: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        if len(sig) == 0:
            return sig
        std = np.std(sig)
        if std == 0:
            return sig - np.mean(sig)
        return (sig - np.mean(sig)) / std
    
    def convolve(
        self,
        signal_data: np.ndarray,
        kernel_name: str,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Convolve signal with a named kernel and detect patterns.
        
        Args:
            signal_data: Input signal (typically packet sizes over time)
            kernel_name: Name of kernel to use
            timestamps: Optional timestamps for each signal sample
            
        Returns:
            Tuple of (convolution_output, list_of_detections)
        """
        if kernel_name not in self.kernels:
            raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(self.kernels.keys())}")
        
        kernel = self.kernels[kernel_name]
        
        # Normalize if requested
        if self.normalize:
            signal_data = self._normalize_signal(signal_data)
            kernel = self._normalize_signal(kernel)
        
        # Perform convolution
        # Paper: (x * h)[n] = Σ_k x[k] * h[n-k]
        conv_output = signal.convolve(signal_data, kernel, mode='same')
        
        # Apply normalization factor from kernel
        if self.normalize and np.sum(kernel ** 2) > 0:
            conv_output = conv_output / np.sqrt(np.sum(kernel ** 2))
        
        # Detect peaks above threshold
        detections = []
        above_threshold = conv_output > self.threshold
        
        # Find contiguous regions above threshold
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(conv_output)]])
        
        for start, end in zip(starts, ends):
            peak_idx = start + np.argmax(conv_output[start:end])
            peak_value = conv_output[peak_idx]
            
            timestamp = timestamps[peak_idx] if timestamps is not None else float(peak_idx)
            
            detections.append(Detection(
                position=int(peak_idx),
                timestamp=timestamp,
                confidence=float(peak_value),
                pattern_type=kernel_name
            ))
        
        return conv_output, detections
    
    def detect(
        self,
        signal_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, List[Detection]]:
        """
        Detect all pattern types in the signal.
        
        Args:
            signal_data: Input signal
            timestamps: Optional timestamps
            
        Returns:
            Dict mapping pattern types to their detections
        """
        results = {}
        for kernel_name in self.kernels:
            _, detections = self.convolve(signal_data, kernel_name, timestamps)
            results[kernel_name] = detections
            logger.debug(f"Detected {len(detections)} {kernel_name} patterns")
        return results
    
    def sweep_threshold(
        self,
        signal_data: np.ndarray,
        kernel_name: str,
        thresholds: Optional[List[float]] = None
    ) -> List[SweepResult]:
        """
        Sweep over threshold values to analyze detection sensitivity.
        
        Paper Section 6.3: "A threshold value of 0 means every positive
        convolution result is classified as a command message (high false
        positives). A threshold of 1.3 sets a very high bar (high false negatives)."
        
        Args:
            signal_data: Input signal
            kernel_name: Kernel to use
            thresholds: List of thresholds to try (default: 0.0 to 1.5)
            
        Returns:
            List of SweepResults showing detection counts at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.0, 1.6, 0.1).tolist()
        
        results = []
        original_threshold = self.threshold
        
        for t in thresholds:
            self.threshold = t
            _, detections = self.convolve(signal_data, kernel_name)
            
            # Estimate false positive rate based on detection density
            # (More detections than expected = likely false positives)
            detection_density = len(detections) / max(len(signal_data), 1)
            estimated_fpr = min(1.0, detection_density * 10)  # Rough estimate
            
            results.append(SweepResult(
                threshold=t,
                num_detections=len(detections),
                false_positive_rate=estimated_fpr
            ))
        
        self.threshold = original_threshold
        return results
    
    def add_kernel(self, name: str, kernel: np.ndarray) -> None:
        """Add or update a kernel."""
        self.kernels[name] = kernel
    
    def learn_kernel_from_samples(
        self,
        samples: List[np.ndarray],
        name: str,
        kernel_length: int = 9
    ) -> np.ndarray:
        """
        Learn a kernel from sample traffic patterns.
        
        Paper: "We test using 10 different kernels for each type of command
        message, each extracted from 10 distinct samples. Despite variations,
        accuracy remains consistent."
        
        Args:
            samples: List of signal samples representing the pattern
            name: Name for the learned kernel
            kernel_length: Length of kernel to extract
            
        Returns:
            The learned kernel array
        """
        if not samples:
            raise ValueError("Need at least one sample to learn kernel")
        
        # Find the most common pattern shape by averaging aligned samples
        aligned = []
        for sample in samples:
            if len(sample) >= kernel_length:
                # Extract center portion
                start = (len(sample) - kernel_length) // 2
                aligned.append(sample[start:start + kernel_length])
        
        if not aligned:
            # Use first sample's beginning if too short
            kernel = samples[0][:kernel_length] if len(samples[0]) >= kernel_length else samples[0]
        else:
            # Average the aligned samples
            kernel = np.mean(aligned, axis=0)
        
        self.kernels[name] = kernel
        return kernel
