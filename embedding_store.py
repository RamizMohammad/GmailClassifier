import hashlib
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

import config

logger = logging.getLogger(__name__)


def generate_cache_key(text: str) -> str:
    """
    Generate a deterministic SHA-256 hash for embedding cache key.
    The key incorporates model name, model version, and exact normalized text.
    """
    if not text:
        return ""
    payload = f"{config.MODEL_NAME}|{config.SCHEMA_VERSION}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingStore:
    """
    Abstraction over Qdrant for persistent embedding storage.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.collection_name = config.QDRANT_COLLECTION_NAME
        
        # Use local disk storage
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with dimension {self.dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Connected to Qdrant collection '{self.collection_name}'.")

    def get_batch(self, keys: List[str]) -> Dict[str, List[float]]:
        """
        Retrieve embeddings for a list of cache keys.
        Returns a dictionary mapping key to its embedding vector.
        """
        valid_keys = [k for k in keys if k]
        if not valid_keys:
            return {}

        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=valid_keys,
            with_vectors=True
        )

        return {point.id: point.vector for point in results if point.vector is not None}

    def put_batch(self, keys: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """
        Store a batch of embeddings.
        All lists must have the same length.
        """
        if not keys:
            return

        points = []
        now = datetime.now(timezone.utc).isoformat()
        
        for i, key in enumerate(keys):
            if not key:
                continue
            
            meta = metadatas[i].copy()
            meta.update({
                "model": config.MODEL_NAME,
                "version": config.SCHEMA_VERSION,
                "created_at": now
            })

            points.append(
                PointStruct(
                    id=key,
                    vector=embeddings[i],
                    payload=meta
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vector_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
                "dimension": self.dimension,
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild(self) -> None:
        """Drop and recreate the collection."""
        logger.warning(f"Rebuilding Qdrant collection '{self.collection_name}'. All data will be lost.")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()
