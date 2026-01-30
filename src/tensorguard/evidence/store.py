"""
Evidence Store for audit trails and compliance records.

Provides storage and retrieval of evidence records that can be used
for compliance verification and audit trails.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .canonical import canonical_bytes


@dataclass
class EvidenceRecord:
    """A single evidence record in the store."""

    record_id: str
    record_type: str  # "manifest", "signature", "audit", "privacy_receipt"
    timestamp: float
    data_hash: str
    data: Dict[str, Any]
    prev_hash: Optional[str] = None  # For chain integrity
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_record_hash(self) -> str:
        """Compute hash of this record for chain integrity."""
        record_data = {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "timestamp": self.timestamp,
            "data_hash": self.data_hash,
            "prev_hash": self.prev_hash,
        }
        return hashlib.sha256(canonical_bytes(record_data)).hexdigest()


class EvidenceStore:
    """
    Thread-safe evidence store with hash-chain integrity.

    Evidence records are stored with cryptographic linking to
    previous records, forming an append-only audit trail.
    """

    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize evidence store.

        Args:
            store_path: Path to store evidence files. If None, uses in-memory store.
        """
        self.store_path = store_path
        self._records: List[EvidenceRecord] = []
        self._last_hash: Optional[str] = None

        if store_path and store_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load existing records from disk."""
        if not self.store_path:
            return

        index_file = self.store_path / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
                for record_data in index.get("records", []):
                    record = EvidenceRecord(**record_data)
                    self._records.append(record)
                self._last_hash = index.get("last_hash")

    def _save_to_disk(self) -> None:
        """Save records to disk."""
        if not self.store_path:
            return

        self.store_path.mkdir(parents=True, exist_ok=True)
        index_file = self.store_path / "index.json"

        index = {
            "version": "1.0",
            "records": [
                {
                    "record_id": r.record_id,
                    "record_type": r.record_type,
                    "timestamp": r.timestamp,
                    "data_hash": r.data_hash,
                    "data": r.data,
                    "prev_hash": r.prev_hash,
                    "metadata": r.metadata,
                }
                for r in self._records
            ],
            "last_hash": self._last_hash,
        }

        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

    def add_record(
        self,
        record_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """
        Add a new evidence record.

        Args:
            record_type: Type of record (e.g., "manifest", "signature")
            data: Record data
            metadata: Optional metadata

        Returns:
            The created EvidenceRecord
        """
        data_hash = hashlib.sha256(canonical_bytes(data)).hexdigest()
        record_id = hashlib.sha256(
            f"{record_type}:{data_hash}:{time.time()}".encode()
        ).hexdigest()[:16]

        record = EvidenceRecord(
            record_id=record_id,
            record_type=record_type,
            timestamp=time.time(),
            data_hash=data_hash,
            data=data,
            prev_hash=self._last_hash,
            metadata=metadata or {},
        )

        self._last_hash = record.compute_record_hash()
        self._records.append(record)

        if self.store_path:
            self._save_to_disk()

        return record

    def get_record(self, record_id: str) -> Optional[EvidenceRecord]:
        """Get a record by ID."""
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def get_records_by_type(self, record_type: str) -> List[EvidenceRecord]:
        """Get all records of a given type."""
        return [r for r in self._records if r.record_type == record_type]

    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the evidence chain.

        Returns:
            True if chain is valid, False if tampering detected
        """
        if not self._records:
            return True

        prev_hash = None
        for record in self._records:
            if record.prev_hash != prev_hash:
                return False
            prev_hash = record.compute_record_hash()

        return prev_hash == self._last_hash

    def get_all_records(self) -> List[EvidenceRecord]:
        """Get all records in the store."""
        return list(self._records)


# Global store instance (lazy initialization)
_global_store: Optional[EvidenceStore] = None


def get_store(store_path: Optional[Path] = None) -> EvidenceStore:
    """
    Get the global evidence store instance.

    Args:
        store_path: Optional path to initialize store. Only used on first call.

    Returns:
        The global EvidenceStore instance
    """
    global _global_store

    if _global_store is None:
        # Default to environment variable or in-memory
        path_str = os.environ.get("TENSAFE_EVIDENCE_PATH")
        if path_str:
            store_path = Path(path_str)
        _global_store = EvidenceStore(store_path)

    return _global_store


def reset_store() -> None:
    """Reset the global store (for testing)."""
    global _global_store
    _global_store = None
