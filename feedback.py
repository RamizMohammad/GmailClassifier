"""
Pluggable feedback storage for classification corrections.

Provides an abstract interface and two initial implementations:
- MemoryFeedbackStore: In-memory (ephemeral, for development/initial deploy)
- JsonFileFeedbackStore: JSON file (ephemeral on Render Free)

Future implementations can add Supabase, PostgreSQL, Google Sheets, etc.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class FeedbackEntry:
    """A single feedback entry."""

    def __init__(
        self,
        email_id: str,
        text: str,
        predicted_category: str,
        correct_category: str,
    ) -> None:
        self.email_id = email_id
        self.text = text
        self.predicted_category = predicted_category
        self.correct_category = correct_category
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_id": self.email_id,
            "text": self.text,
            "predicted_category": self.predicted_category,
            "correct_category": self.correct_category,
            "timestamp": self.timestamp,
        }


class FeedbackStore(ABC):
    """Abstract base class for feedback persistence."""

    @abstractmethod
    def save(self, entry: FeedbackEntry) -> None:
        """Save a feedback entry."""
        ...

    @abstractmethod
    def get_examples(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve feedback entries, optionally filtered by correct_category."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored feedback entries."""
        ...


class MemoryFeedbackStore(FeedbackStore):
    """
    In-memory feedback store.

    WARNING: All data is lost when the service restarts.
    Suitable for development and initial deployment.
    """

    def __init__(self) -> None:
        self._entries: List[FeedbackEntry] = []
        self._lock = Lock()

    def save(self, entry: FeedbackEntry) -> None:
        with self._lock:
            self._entries.append(entry)
        logger.info(
            "Feedback saved (memory): email_id=%s, correct=%s",
            entry.email_id,
            entry.correct_category,
        )

    def get_examples(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if category:
                return [
                    e.to_dict()
                    for e in self._entries
                    if e.correct_category == category
                ]
            return [e.to_dict() for e in self._entries]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


class JsonFileFeedbackStore(FeedbackStore):
    """
    JSON file feedback store.

    WARNING: Render Free has an ephemeral filesystem. Data saved here
    will be lost on every deploy and may be lost on service restart.
    Use only as a stepping stone before migrating to a persistent store.
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        self._file_path = Path(file_path or config.FEEDBACK_FILE_PATH)
        self._lock = Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self._file_path.exists():
            self._file_path.write_text("[]", encoding="utf-8")

    def _read(self) -> List[Dict[str, Any]]:
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, data: List[Dict[str, Any]]) -> None:
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save(self, entry: FeedbackEntry) -> None:
        with self._lock:
            data = self._read()
            data.append(entry.to_dict())
            self._write(data)
        logger.info(
            "Feedback saved (file): email_id=%s, correct=%s",
            entry.email_id,
            entry.correct_category,
        )

    def get_examples(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read()
            if category:
                return [d for d in data if d.get("correct_category") == category]
            return data

    def count(self) -> int:
        with self._lock:
            return len(self._read())


def create_feedback_store() -> FeedbackStore:
    """
    Factory function that creates the appropriate FeedbackStore
    based on the FEEDBACK_STORE_TYPE configuration.
    """
    store_type = config.FEEDBACK_STORE_TYPE.lower()

    if store_type == "memory":
        logger.info("Using in-memory feedback store (ephemeral).")
        return MemoryFeedbackStore()
    elif store_type == "json_file":
        logger.info(
            "Using JSON file feedback store at: %s (WARNING: ephemeral on Render).",
            config.FEEDBACK_FILE_PATH,
        )
        return JsonFileFeedbackStore()
    else:
        logger.warning(
            "Unknown FEEDBACK_STORE_TYPE '%s', falling back to memory store.",
            store_type,
        )
        return MemoryFeedbackStore()
