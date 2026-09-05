"""
Configuration module for Gmail Smart Sorter V6.

All settings are loaded from environment variables with sensible defaults.
"""

import os
from typing import Optional


# --- Authentication ---
CLASSIFIER_API_KEY: Optional[str] = os.environ.get("CLASSIFIER_API_KEY")

# --- Model ---
MODEL_NAME: str = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Limits ---
MAX_BATCH_SIZE: int = int(os.environ.get("MAX_BATCH_SIZE", "100"))
MAX_BODY_LENGTH: int = int(os.environ.get("MAX_BODY_LENGTH", "5000"))
MAX_REQUEST_SIZE: int = int(os.environ.get("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10 MB

# --- Classification Thresholds ---
CONFIDENCE_HIGH_SIMILARITY: float = float(os.environ.get("CONFIDENCE_HIGH_SIMILARITY", "0.78"))
CONFIDENCE_HIGH_MARGIN: float = float(os.environ.get("CONFIDENCE_HIGH_MARGIN", "0.08"))
CONFIDENCE_MEDIUM_SIMILARITY: float = float(os.environ.get("CONFIDENCE_MEDIUM_SIMILARITY", "0.65"))
CONFIDENCE_MEDIUM_MARGIN: float = float(os.environ.get("CONFIDENCE_MEDIUM_MARGIN", "0.04"))

# --- Feedback ---
FEEDBACK_STORE_TYPE: str = os.environ.get("FEEDBACK_STORE_TYPE", "memory")
FEEDBACK_FILE_PATH: str = os.environ.get("FEEDBACK_FILE_PATH", "feedback_data.json")

# --- Categories ---
CATEGORIES_FILE: str = os.environ.get("CATEGORIES_FILE", "categories.json")

# --- Server ---
PORT: int = int(os.environ.get("PORT", "8000"))

# --- Top-K categories to return ---
TOP_K: int = int(os.environ.get("TOP_K", "3"))

# --- Dynamic Memory Management ---
# How long (in seconds) the service must be idle before unloading the model
IDLE_TIMEOUT_SECONDS: int = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "60"))

# Whether to enable the 10-second self-ping keep-alive
ENABLE_KEEP_ALIVE: bool = os.environ.get("ENABLE_KEEP_ALIVE", "true").lower() == "true"
