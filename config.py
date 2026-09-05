"""
Configuration module for Gmail Smart Sorter V7.1.

All settings are loaded from environment variables with sensible defaults.
"""

import os
from typing import Optional


# --- Version ---
VERSION: str = "v7.1"
SCHEMA_VERSION: str = "v7.1"

# --- Authentication ---
CLASSIFIER_API_KEY: Optional[str] = os.environ.get("CLASSIFIER_API_KEY")

# --- Model ---
MODEL_NAME: str = os.environ.get("MODEL_NAME", "BAAI/bge-m3")

# --- Limits ---
MAX_BATCH_SIZE: int = int(os.environ.get("MAX_BATCH_SIZE", "100"))
MAX_BODY_LENGTH: int = int(os.environ.get("MAX_BODY_LENGTH", "2000"))
MAX_REQUEST_SIZE: int = int(os.environ.get("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10 MB

# --- Data Files ---
PROTOTYPES_FILE: str = os.environ.get("PROTOTYPES_FILE", "data/prototypes.json")
BENCHMARK_FILE: str = os.environ.get("BENCHMARK_FILE", "data/benchmark.json")

# --- Qdrant Settings ---
QDRANT_PATH: str = os.environ.get("QDRANT_PATH", "qdrant_data")
QDRANT_COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "embeddings_v7_1")

# --- Signal Weights (subject-first) ---
WEIGHT_SUBJECT: float = float(os.environ.get("WEIGHT_SUBJECT", "0.65"))
WEIGHT_BODY: float = float(os.environ.get("WEIGHT_BODY", "0.35"))
WEIGHT_SENDER: float = float(os.environ.get("WEIGHT_SENDER", "0.0")) # Weak signal, typically not used for direct semantic embedding

# --- Prototype Scoring ---
PROTO_TOP1_WEIGHT: float = float(os.environ.get("PROTO_TOP1_WEIGHT", "0.70"))
PROTO_TOPK_WEIGHT: float = float(os.environ.get("PROTO_TOPK_WEIGHT", "0.30"))
PROTO_TOP_K: int = int(os.environ.get("PROTO_TOP_K", "3"))

# --- Decision Thresholds ---
AUTO_SORT_CONFIDENCE: float = float(os.environ.get("AUTO_SORT_CONFIDENCE", "0.82"))
AUTO_SORT_MARGIN: float = float(os.environ.get("AUTO_SORT_MARGIN", "0.08"))
REVIEW_CONFIDENCE: float = float(os.environ.get("REVIEW_CONFIDENCE", "0.60"))

# --- Feedback ---
FEEDBACK_STORE_TYPE: str = os.environ.get("FEEDBACK_STORE_TYPE", "memory")
FEEDBACK_FILE_PATH: str = os.environ.get("FEEDBACK_FILE_PATH", "feedback_data.json")

# --- Server ---
PORT: int = int(os.environ.get("PORT", "8000"))

# --- Top-K categories to return in response ---
TOP_K: int = int(os.environ.get("TOP_K", "3"))

# --- Dynamic Memory Management ---
IDLE_TIMEOUT_SECONDS: int = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "600")) # Increased to 10 mins since load time is longer
ENABLE_KEEP_ALIVE: bool = os.environ.get("ENABLE_KEEP_ALIVE", "true").lower() == "true"
