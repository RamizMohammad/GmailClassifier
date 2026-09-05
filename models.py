"""
Pydantic models for Gmail Smart Sorter V7 request/response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import config


# --- Request Models ---

class EmailInput(BaseModel):
    """A single email to classify."""
    id: str = Field(..., description="Unique identifier for this email (preserved in response)")
    sender: str = Field(default="", description="Email sender address")
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body text")


class ClassifyRequest(BaseModel):
    """Batch classification request."""
    emails: List[EmailInput] = Field(
        ...,
        min_length=1,
        description="List of emails to classify"
    )


class FeedbackRequest(BaseModel):
    """Correction feedback for a classification. Does NOT accept email body (privacy)."""
    id: str = Field(..., description="ID of the email that was misclassified")
    predicted_category: str = Field(..., description="The category the API predicted")
    correct_category: str = Field(..., description="The correct category")


class EvaluateExample(BaseModel):
    """A single labeled example for evaluation."""
    text: str = Field(..., description="The email subject/text to classify")
    expected_category: str = Field(..., description="The correct category label")


class EvaluateRequest(BaseModel):
    """Batch evaluation request with labeled examples."""
    examples: List[EvaluateExample] = Field(
        ...,
        min_length=1,
        description="List of labeled examples to evaluate"
    )


# --- Response Models ---

class CategoryScore(BaseModel):
    """A category and its similarity score."""
    category: str
    score: float


class ClassificationResult(BaseModel):
    """V7 classification result for a single email."""
    id: str
    category: str
    decision: str  # "AUTO_SORT", "REVIEW", "UNMATCHED"
    confidence: float
    similarity: float
    margin: float
    family: str
    family_confidence: float
    family_margin: float
    top3: List[CategoryScore]
    reason: str


class ClassifyResponse(BaseModel):
    """V7 batch classification response."""
    version: str = config.VERSION
    results: List[ClassificationResult]


class HealthResponse(BaseModel):
    """V7 health check response."""
    status: str
    version: str = config.VERSION
    model: str
    model_loaded: bool
    prototype_count: int
    category_count: int


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    status: str
    message: str
    id: str
    correct_category: str


class PerCategoryMetric(BaseModel):
    """Metrics for a single category."""
    category: str
    precision: float
    recall: float
    f1: float
    support: int


class EvaluateResponse(BaseModel):
    """V7 evaluation response with full metrics."""
    version: str = config.VERSION
    total: int
    correct: int
    accuracy: float
    top3_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_category: List[PerCategoryMetric]
    confusion: Dict[str, Dict[str, int]]
    misclassified: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
