"""
Gmail Smart Sorter V7 — Semantic Classifier API

FastAPI application with multi-prototype hierarchical classification,
subject-first weighted signals, and conflict detection.

Endpoints: /health, /classify, /feedback, /evaluate
"""

import asyncio
import json
import logging
import urllib.request
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

import config
from classifier import classifier, ALL_CATEGORIES, CATEGORY_TO_FAMILY
from feedback import FeedbackEntry, FeedbackStore, create_feedback_store
from models import (
    CategoryScore,
    ClassificationResult,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PerCategoryMetric,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Feedback Store ---
feedback_store: FeedbackStore = None  # type: ignore[assignment]


async def keep_alive_task():
    """Background task that pings the health endpoint every 10 seconds."""
    logger.info("Starting 10-second keep-alive pinger...")
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{config.PORT}/health", timeout=5)
        except Exception as e:
            logger.debug("Keep-alive ping failed: %s", e)
        await asyncio.sleep(10)


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup."""
    global feedback_store

    logger.info("Starting Gmail Smart Sorter V7 API...")

    # Start the memory idle watcher (model loads on first request)
    classifier.start_watcher()

    # Initialize feedback store
    feedback_store = create_feedback_store()

    # Start keep-alive if enabled
    keep_alive = None
    if config.ENABLE_KEEP_ALIVE:
        keep_alive = asyncio.create_task(keep_alive_task())

    logger.info("V7 API ready. Model will load on first classification request.")
    yield

    logger.info("Shutting down V7 API...")
    classifier.stop_watcher()
    if keep_alive:
        keep_alive.cancel()


# --- App ---
app = FastAPI(
    title="Gmail Smart Sorter V7",
    description="Multi-prototype hierarchical email classifier using Sentence Transformer embeddings",
    version="7.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# --- Authentication ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify the API key using constant-time comparison."""
    import hmac

    if not config.CLASSIFIER_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API key not set.")
    if not hmac.compare_digest(x_api_key, config.CLASSIFIER_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


# --- Middleware ---
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests that exceed the maximum allowed size."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.MAX_REQUEST_SIZE:
        return JSONResponse(status_code=413, content={"detail": "Request body too large."})
    return await call_next(request)


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format. Check your JSON payload."},
    )


# ─────────────── Endpoints ───────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Public health check. Does NOT trigger model load."""
    return HealthResponse(
        status="ok",
        version=config.VERSION,
        model=config.MODEL_NAME,
        model_loaded=classifier.is_loaded,
        prototype_count=classifier.num_prototypes,
        category_count=classifier.num_categories,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={401: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def classify(request: ClassifyRequest):
    """Batch email classification. Loads model on first request."""
    if len(request.emails) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request.emails)} exceeds maximum of {config.MAX_BATCH_SIZE}.",
        )

    logger.info("Classifying batch of %d emails.", len(request.emails))

    try:
        results = classifier.classify_batch(request.emails)
    except Exception as e:
        logger.error("Classification error: %s", e)
        raise HTTPException(status_code=500, detail="Classification failed.")

    return ClassifyResponse(version=config.VERSION, results=results)


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={401: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def submit_feedback(request: FeedbackRequest):
    """Submit classification correction. Does NOT persist email content."""
    # Validate category
    if request.correct_category not in ALL_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category: '{request.correct_category}'. Valid: {ALL_CATEGORIES}",
        )

    entry = FeedbackEntry(
        email_id=request.id,
        predicted_category=request.predicted_category,
        correct_category=request.correct_category,
    )
    feedback_store.save(entry)

    logger.info(
        "Feedback: id=%s, predicted=%s, correct=%s",
        request.id, request.predicted_category, request.correct_category,
    )

    return FeedbackResponse(
        status="ok",
        message="Feedback recorded.",
        id=request.id,
        correct_category=request.correct_category,
    )


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    responses={401: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def evaluate(request: EvaluateRequest):
    """
    Run benchmark evaluation against labeled examples.
    Returns accuracy, precision, recall, F1, and confusion matrix.
    """
    logger.info("Evaluating %d benchmark examples.", len(request.examples))

    total = len(request.examples)
    correct = 0
    top3_correct = 0
    misclassified = []

    # Per-category tracking
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    support = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for example in request.examples:
        try:
            result = classifier.classify_text(example.text)
        except Exception as e:
            logger.error("Evaluation error for '%s': %s", example.text[:50], e)
            continue

        predicted = result.category
        expected = example.expected_category
        top3_cats = [c.category for c in result.top3]

        confusion[expected][predicted] += 1
        support[expected] += 1

        if predicted == expected:
            correct += 1
            true_positives[expected] += 1
        else:
            false_positives[predicted] += 1
            false_negatives[expected] += 1
            misclassified.append({
                "text": example.text[:100],
                "expected": expected,
                "predicted": predicted,
                "confidence": result.confidence,
                "top3": [{"category": c.category, "score": c.score} for c in result.top3],
            })

        if expected in top3_cats:
            top3_correct += 1

    # Compute per-category metrics
    all_cats = set(list(support.keys()) + list(true_positives.keys()) +
                   list(false_positives.keys()) + list(false_negatives.keys()))
    per_category = []
    for cat in sorted(all_cats):
        tp = true_positives.get(cat, 0)
        fp = false_positives.get(cat, 0)
        fn = false_negatives.get(cat, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_category.append(PerCategoryMetric(
            category=cat,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support.get(cat, 0),
        ))

    # Macro averages
    n_cats = len(per_category) if per_category else 1
    macro_precision = sum(m.precision for m in per_category) / n_cats
    macro_recall = sum(m.recall for m in per_category) / n_cats
    macro_f1 = sum(m.f1 for m in per_category) / n_cats

    accuracy = correct / total if total > 0 else 0.0
    top3_accuracy = top3_correct / total if total > 0 else 0.0

    # Convert confusion defaultdict to regular dict
    confusion_dict = {k: dict(v) for k, v in confusion.items()}

    return EvaluateResponse(
        version=config.VERSION,
        total=total,
        correct=correct,
        accuracy=round(accuracy, 4),
        top3_accuracy=round(top3_accuracy, 4),
        macro_precision=round(macro_precision, 4),
        macro_recall=round(macro_recall, 4),
        macro_f1=round(macro_f1, 4),
        per_category=per_category,
        confusion=confusion_dict,
        misclassified=misclassified,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Gmail Smart Sorter V7 API", "health": "/health", "version": config.VERSION}
