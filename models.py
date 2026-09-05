from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmailInput(BaseModel):
    id: str
    sender: str
    subject: str
    body: str


class CategoryScore(BaseModel):
    category: str
    score: float


class ClassificationResult(BaseModel):
    id: str
    category: str
    decision: str  # AUTO_SORT, REVIEW, UNMATCHED
    confidence: float
    similarity: float
    margin: float
    subject_score: float
    body_score: float
    prototype_support: float
    family: str
    family_confidence: float
    family_margin: float
    top3: List[CategoryScore]
    reason: str


class ClassifyRequest(BaseModel):
    emails: List[EmailInput]


class ClassifyResponse(BaseModel):
    version: str
    results: List[ClassificationResult]


class ErrorResponse(BaseModel):
    detail: str


class FeedbackRequest(BaseModel):
    id: str
    predicted_category: str
    correct_category: str


class FeedbackResponse(BaseModel):
    status: str
    message: str
    id: str
    correct_category: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    model_loaded: bool
    prototype_count: int
    category_count: int
    embedding_dimension: int
    embedding_store: str
    embedding_cache: str


class BenchmarkExample(BaseModel):
    text: str
    expected_category: str
    subject: str = ""
    body: str = ""


class EvaluateRequest(BaseModel):
    examples: List[BenchmarkExample]


class PerCategoryMetric(BaseModel):
    category: str
    precision: float
    recall: float
    f1: float
    support: int


class EvaluateResponse(BaseModel):
    version: str
    total: int
    correct: int
    accuracy: float
    top3_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    auto_sort_precision: float
    auto_sort_coverage: float
    review_rate: float
    unmatched_rate: float
    per_category: List[PerCategoryMetric]
    confusion: Dict[str, Dict[str, int]]
    misclassified: List[Dict[str, Any]]
