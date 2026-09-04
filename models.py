"""
Pydantic models for request/response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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
    """Correction feedback for a classification."""
    email_id: str = Field(..., description="ID of the email that was misclassified")
    text: str = Field(..., description="The email text that was classified")
    predicted_category: str = Field(..., description="The category the API predicted")
    correct_category: str = Field(..., description="The correct category")


# --- Response Models ---

class CategoryScore(BaseModel):
    """A category and its similarity score."""
    category: str
    score: float


class ClassificationResult(BaseModel):
    """Classification result for a single email."""
    id: str
    category: str
    similarity: float
    confidence: float
    confidence_level: str  # "high", "medium", "low"
    margin: float
    top_categories: List[CategoryScore]


class ClassifyResponse(BaseModel):
    """Batch classification response."""
    results: List[ClassificationResult]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    model_loaded: bool
    categories_loaded: int


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    status: str
    message: str
    email_id: str
    correct_category: str


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
