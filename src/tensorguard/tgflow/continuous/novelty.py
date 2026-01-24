"""
Novelty Detector

MVP implementation for detecting data novelty to decide whether training is needed.
NOT research-grade - practical and fast.

Methods:
1. Data hash change detection (fast)
2. Embedding centroid drift (if embeddings available)
3. Keyword distribution drift (simple NLP)
"""

import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import re

logger = logging.getLogger(__name__)


@dataclass
class NoveltyResult:
    """Result of novelty detection."""
    novelty_score: float  # 0.0 = no change, 1.0 = completely new
    should_train: bool
    reasons: List[str]
    
    # Component scores
    hash_changed: bool = False
    hash_novelty: float = 0.0
    centroid_drift: float = 0.0
    keyword_drift: float = 0.0
    
    # Metadata
    current_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "novelty_score": self.novelty_score,
            "should_train": self.should_train,
            "reasons": self.reasons,
            "components": {
                "hash_changed": self.hash_changed,
                "hash_novelty": self.hash_novelty,
                "centroid_drift": self.centroid_drift,
                "keyword_drift": self.keyword_drift,
            },
        }


class NoveltyDetector:
    """
    Detects novelty in data to determine if training is needed.
    
    Uses a weighted combination of:
    - Hash change (40%): Did the data content change at all?
    - Centroid drift (30%): Did the embedding space shift?
    - Keyword drift (30%): Did the topic/vocabulary change?
    """
    
    WEIGHT_HASH = 0.4
    WEIGHT_CENTROID = 0.3
    WEIGHT_KEYWORD = 0.3
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize detector.
        
        Args:
            threshold: Novelty score above which training is triggered
        """
        self.threshold = threshold
    
    def detect(
        self,
        current_data: Dict[str, Any],
        previous_snapshot: Optional[Dict[str, Any]] = None,
    ) -> NoveltyResult:
        """
        Detect novelty between current and previous data snapshots.
        
        Args:
            current_data: Current data snapshot with:
                - content_hash: str
                - sample_texts: List[str] (optional)
                - embedding_centroid: List[float] (optional)
            previous_snapshot: Previous snapshot with same structure
            
        Returns:
            NoveltyResult with score and reasons
        """
        reasons = []
        
        # If no previous snapshot, everything is novel
        if not previous_snapshot:
            return NoveltyResult(
                novelty_score=1.0,
                should_train=True,
                reasons=["First data ingestion - no previous snapshot"],
                hash_changed=True,
                hash_novelty=1.0,
                current_hash=current_data.get("content_hash"),
            )
        
        # 1. Hash change detection (fastest)
        current_hash = current_data.get("content_hash", "")
        previous_hash = previous_snapshot.get("content_hash", "")
        hash_changed = current_hash != previous_hash
        hash_novelty = 1.0 if hash_changed else 0.0
        
        if hash_changed:
            reasons.append("Data content changed (hash mismatch)")
        
        # 2. Centroid drift (if embeddings available)
        centroid_drift = 0.0
        current_centroid = current_data.get("embedding_centroid")
        previous_centroid = previous_snapshot.get("embedding_centroid")
        
        if current_centroid and previous_centroid:
            centroid_drift = self._compute_centroid_drift(
                current_centroid, previous_centroid
            )
            if centroid_drift > 0.2:
                reasons.append(f"Embedding space shifted (drift={centroid_drift:.2f})")
        
        # 3. Keyword drift (if sample texts available)
        keyword_drift = 0.0
        current_texts = current_data.get("sample_texts", [])
        previous_texts = previous_snapshot.get("sample_texts", [])
        
        if current_texts and previous_texts:
            keyword_drift = self._compute_keyword_drift(current_texts, previous_texts)
            if keyword_drift > 0.2:
                reasons.append(f"Topic/vocabulary shifted (drift={keyword_drift:.2f})")
        
        # Compute weighted novelty score
        novelty_score = (
            self.WEIGHT_HASH * hash_novelty +
            self.WEIGHT_CENTROID * centroid_drift +
            self.WEIGHT_KEYWORD * keyword_drift
        )
        
        # Normalize to 0-1
        novelty_score = min(1.0, max(0.0, novelty_score))
        
        should_train = novelty_score >= self.threshold
        
        if not should_train and not reasons:
            reasons.append("No significant changes detected")
        
        return NoveltyResult(
            novelty_score=novelty_score,
            should_train=should_train,
            reasons=reasons,
            hash_changed=hash_changed,
            hash_novelty=hash_novelty,
            centroid_drift=centroid_drift,
            keyword_drift=keyword_drift,
            current_hash=current_hash,
            previous_hash=previous_hash,
        )
    
    def _compute_centroid_drift(
        self,
        current: List[float],
        previous: List[float],
    ) -> float:
        """
        Compute normalized cosine distance between embedding centroids.
        
        Returns: 0.0 = identical, 1.0 = orthogonal/opposite
        """
        if len(current) != len(previous):
            return 0.5  # Incompatible dimensions, moderate novelty
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(current, previous))
        norm_a = sum(a * a for a in current) ** 0.5
        norm_b = sum(b * b for b in previous) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        cosine_sim = dot_product / (norm_a * norm_b)
        
        # Convert to distance (0 = same, 1 = different)
        drift = 1.0 - max(-1.0, min(1.0, cosine_sim))
        return drift / 2.0  # Normalize to 0-1
    
    def _compute_keyword_drift(
        self,
        current_texts: List[str],
        previous_texts: List[str],
        top_k: int = 100,
    ) -> float:
        """
        Compute keyword distribution drift using Jaccard-like distance.
        
        Returns: 0.0 = same keywords, 1.0 = completely different
        """
        current_keywords = self._extract_keywords(current_texts, top_k)
        previous_keywords = self._extract_keywords(previous_texts, top_k)
        
        if not current_keywords and not previous_keywords:
            return 0.0
        
        current_set = set(current_keywords.keys())
        previous_set = set(previous_keywords.keys())
        
        intersection = len(current_set & previous_set)
        union = len(current_set | previous_set)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        return 1.0 - jaccard
    
    def _extract_keywords(
        self,
        texts: List[str],
        top_k: int = 100,
    ) -> Dict[str, int]:
        """Extract top keywords from texts."""
        all_words = []
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend(words)
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'were', 'said', 'this', 'that', 'with',
        }
        words = [w for w in all_words if w not in stopwords]
        
        counter = Counter(words)
        return dict(counter.most_common(top_k))


def compute_content_hash(data: Any) -> str:
    """Compute SHA-256 hash of data content."""
    import json
    if isinstance(data, str):
        content = data
    elif isinstance(data, bytes):
        content = data.decode('utf-8', errors='ignore')
    else:
        content = json.dumps(data, sort_keys=True, default=str)
    
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def create_snapshot(
    content_hash: str,
    sample_texts: Optional[List[str]] = None,
    embedding_centroid: Optional[List[float]] = None,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a data snapshot for novelty comparison."""
    return {
        "content_hash": content_hash,
        "sample_texts": sample_texts or [],
        "embedding_centroid": embedding_centroid,
        "record_count": record_count,
    }
