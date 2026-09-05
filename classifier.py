"""
Semantic email classifier using Sentence Transformers.

Generates embeddings for emails and compares against pre-computed
category prototype embeddings using cosine similarity (via dot product
on normalized vectors).
"""

import gc
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from models import CategoryScore, ClassificationResult, EmailInput

logger = logging.getLogger(__name__)


class EmailClassifier:
    """
    Singleton classifier that loads a Sentence Transformer model and
    pre-computes category embeddings. Supports dynamic unloading to free RAM.
    """

    def __init__(self) -> None:
        self.model = None
        self.model_name: str = config.MODEL_NAME
        self.category_names: List[str] = []
        self.category_embeddings: Optional[np.ndarray] = None  # (num_categories, embed_dim)
        self._is_loaded: bool = False
        
        # Thread safety and dynamic memory management
        self._lock = threading.RLock()
        self.last_used_time: float = 0.0
        self._stop_watcher = threading.Event()
        self._watcher_thread: Optional[threading.Thread] = None

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._is_loaded

    def start_watcher(self) -> None:
        """Starts the background thread that unloads the model when idle."""
        if self._watcher_thread is None or not self._watcher_thread.is_alive():
            self._stop_watcher.clear()
            self._watcher_thread = threading.Thread(
                target=self._idle_watcher_loop, daemon=True
            )
            self._watcher_thread.start()
            logger.info("Memory idle-watcher started (timeout: %ds).", config.IDLE_TIMEOUT_SECONDS)

    def stop_watcher(self) -> None:
        """Stops the background thread."""
        self._stop_watcher.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=2.0)
            self._watcher_thread = None

    def _idle_watcher_loop(self) -> None:
        while not self._stop_watcher.is_set():
            time.sleep(5)
            # If the model is loaded and we've exceeded the idle timeout
            with self._lock:
                if self._is_loaded and (time.time() - self.last_used_time > config.IDLE_TIMEOUT_SECONDS):
                    logger.info("Idle timeout reached. Unloading model to free RAM.")
                    self._unload_unsafe()

    def _unload_unsafe(self) -> None:
        """Unloads the model without acquiring the lock (caller must hold it)."""
        self.model = None
        self.category_embeddings = None
        self._is_loaded = False
        # Force garbage collection to free RAM immediately
        gc.collect()
        logger.info("Model unloaded. RAM freed.")

    def load(self) -> None:
        """Load the model and pre-compute category embeddings. Thread-safe."""
        with self._lock:
            if self._is_loaded:
                self.last_used_time = time.time()
                return

            logger.info("Loading sentence transformer model: %s", self.model_name)

            # Import here to avoid paying import cost if something fails earlier
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")

            # Load and encode categories
            self._load_categories()
            self._is_loaded = True
            self.last_used_time = time.time()
            logger.info(
                "Classifier ready. %d categories loaded.", len(self.category_names)
            )

    def _load_categories(self) -> None:
        """Load categories.json and pre-compute normalized centroid embeddings."""
        categories_path = Path(config.CATEGORIES_FILE)
        if not categories_path.is_absolute():
            # Resolve relative to the app directory
            categories_path = Path(__file__).parent / categories_path

        if not categories_path.exists():
            raise FileNotFoundError(
                f"Categories file not found: {categories_path}"
            )

        with open(categories_path, "r", encoding="utf-8") as f:
            category_data: Dict[str, List[str]] = json.load(f)

        self.category_names = list(category_data.keys())
        logger.info("Computing embeddings for %d categories...", len(self.category_names))

        # Collect all description texts and their category indices
        all_texts: List[str] = []
        category_indices: List[int] = []
        for idx, (category, descriptions) in enumerate(category_data.items()):
            for desc in descriptions:
                all_texts.append(desc)
                category_indices.append(idx)

        # Batch-encode all descriptions at once (normalized)
        all_embeddings = self.model.encode(
            all_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compute centroid for each category
        embed_dim = all_embeddings.shape[1]
        centroids = np.zeros((len(self.category_names), embed_dim), dtype=np.float32)
        counts = np.zeros(len(self.category_names), dtype=np.int32)

        for i, cat_idx in enumerate(category_indices):
            centroids[cat_idx] += all_embeddings[i]
            counts[cat_idx] += 1

        # Average and normalize centroids
        for i in range(len(self.category_names)):
            if counts[i] > 0:
                centroids[i] /= counts[i]

        # Normalize centroids so dot product = cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        self.category_embeddings = centroids / norms

        logger.info("Category embeddings computed and cached.")

    def add_example(self, category: str, text: str, weight: float = 0.1) -> bool:
        """
        Add a real example to a category's centroid embedding.
        """
        with self._lock:
            if not self._is_loaded:
                self.load()
                
            self.last_used_time = time.time()
            if category not in self.category_names:
                return False

            idx = self.category_names.index(category)
            new_embedding = self.model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]

            # Blend: centroid = (1 - weight) * old_centroid + weight * new_example
            blended = (1 - weight) * self.category_embeddings[idx] + weight * new_embedding

            # Re-normalize
            norm = np.linalg.norm(blended)
            if norm > 1e-10:
                blended /= norm

            self.category_embeddings[idx] = blended
            return True

    def classify_batch(
        self, emails: List[EmailInput]
    ) -> List[ClassificationResult]:
        """
        Classify a batch of emails. Thread-safe.
        """
        with self._lock:
            if not self._is_loaded:
                self.load()
            
            self.last_used_time = time.time()

            # Build text representations
            texts = [self._build_text(email) for email in emails]

            # Batch encode (normalized)
            email_embeddings = self.model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Cosine similarity via dot product: (batch, dim) @ (dim, categories) -> (batch, categories)
            similarity_matrix = np.dot(email_embeddings, self.category_embeddings.T)

            # Build results
            results: List[ClassificationResult] = []
            for i, email in enumerate(emails):
                scores = similarity_matrix[i]
                result = self._build_result(email.id, scores)
                results.append(result)

            return results

    def _build_text(self, email: EmailInput) -> str:
        """
        Build normalized text representation for an email.
        """
        parts: List[str] = []

        if email.sender:
            parts.append(f"Sender: {email.sender}")
        if email.subject:
            parts.append(f"Subject: {email.subject}")
        if email.body:
            body = email.body[: config.MAX_BODY_LENGTH]
            parts.append(f"Body: {body}")

        return "\n".join(parts) if parts else "Empty email"

    def _build_result(
        self, email_id: str, scores: np.ndarray
    ) -> ClassificationResult:
        """Build a ClassificationResult from similarity scores."""
        # Get sorted indices (descending)
        sorted_indices = np.argsort(scores)[::-1]

        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else best_idx

        best_score = float(scores[best_idx])
        second_score = float(scores[second_idx])
        margin = best_score - second_score

        # Confidence: weighted combination of absolute similarity and margin
        confidence = self._compute_confidence(best_score, margin)

        # Confidence level based on configurable thresholds
        confidence_level = self._compute_confidence_level(best_score, margin)

        # Top-K categories
        top_k = min(config.TOP_K, len(self.category_names))
        top_categories = [
            CategoryScore(
                category=self.category_names[sorted_indices[j]],
                score=round(float(scores[sorted_indices[j]]), 4),
            )
            for j in range(top_k)
        ]

        return ClassificationResult(
            id=email_id,
            category=self.category_names[best_idx],
            similarity=round(best_score, 4),
            confidence=round(confidence, 4),
            confidence_level=confidence_level,
            margin=round(margin, 4),
            top_categories=top_categories,
        )

    @staticmethod
    def _compute_confidence(similarity: float, margin: float) -> float:
        """
        Compute a numeric confidence score.
        """
        normalized_margin = min(margin / 0.15, 1.0)
        confidence = 0.6 * similarity + 0.4 * normalized_margin
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _compute_confidence_level(similarity: float, margin: float) -> str:
        """
        Compute categorical confidence level based on configurable thresholds.
        """
        if (
            similarity >= config.CONFIDENCE_HIGH_SIMILARITY
            and margin >= config.CONFIDENCE_HIGH_MARGIN
        ):
            return "high"
        elif (
            similarity >= config.CONFIDENCE_MEDIUM_SIMILARITY
            and margin >= config.CONFIDENCE_MEDIUM_MARGIN
        ):
            return "medium"
        else:
            return "low"

    @property
    def num_categories(self) -> int:
        """Number of loaded categories."""
        return len(self.category_names)


# Module-level singleton
classifier = EmailClassifier()
