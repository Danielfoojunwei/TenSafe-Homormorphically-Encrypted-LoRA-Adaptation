"""
Edge Agent Spooler

Provides a persistent, disk-backed queue for buffering messages when offline.
Uses SQLite for reliability and simplicity.
"""

import sqlite3
import threading
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Spooler:
    """
    Persistent FIFO queue backed by SQLite.
    Thread-safe.
    """
    def __init__(self, db_path: str = "spool.db", max_size: int = 100_000):
        self.db_path = Path(db_path)
        self.max_size = max_size
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    timestamp_ns INTEGER,
                    payload TEXT, -- JSON serialized message
                    priority INTEGER DEFAULT 0,
                    created_at REAL
                )
            ''')
            conn.commit()
            conn.close()

    def enqueue(self, topic: str, timestamp_ns: int, payload: Dict[str, Any], priority: int = 0):
        """Add message to spool."""
        with self._lock:
            try:
                # Check size limit (naive count)
                if self.size() >= self.max_size:
                    logger.warning("Spooler full, dropping oldest message")
                    self._drop_oldest()

                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute('''
                    INSERT INTO queue (topic, timestamp_ns, payload, priority, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (topic, timestamp_ns, json.dumps(payload), priority, time.time()))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to enqueue: {e}")

    def peek_batch(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Get next batch without removing."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM queue ORDER BY priority DESC, id ASC LIMIT ?', (batch_size,))
            rows = c.fetchall()
            conn.close()

            result = []
            for r in rows:
                try:
                    item = dict(r)
                    item['payload'] = json.loads(item['payload'])
                    result.append(item)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to deserialize spooler message id={r['id']}: {e}")
                    continue
            return result

    def ack_batch(self, ids: List[int]):
        """Remove processed messages."""
        if not ids: return
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            placeholders = ','.join('?' for _ in ids)
            c.execute(f'DELETE FROM queue WHERE id IN ({placeholders})', ids)
            conn.commit()
            conn.close()

    def size(self) -> int:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM queue')
            count = c.fetchone()[0]
            conn.close()
            return count

    def _drop_oldest(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM queue WHERE id = (SELECT min(id) FROM queue)')
        conn.commit()
        conn.close()
