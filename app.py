"""
Gmail Smart Sorter V7.1 — Semantic Classifier API

FastAPI application with Qdrant embedding caching, subject-first weighted signals,
and BAAI/bge-m3 embeddings.
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
    logger.info("Starting 10-second keep-alive pinger...")
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{config.PORT}/health", timeout=5)
        except Exception as e:
            logger.debug("Keep-alive ping failed: %s", e)
        await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feedback_store
    logger.info("Starting Gmail Smart Sorter V7.1 API...")

    classifier.start_watcher()
    feedback_store = create_feedback_store()

    keep_alive = None
    if config.ENABLE_KEEP_ALIVE:
        keep_alive = asyncio.create_task(keep_alive_task())

    logger.info("V7.1 API ready. Model will load on first classification request.")
    yield

    logger.info("Shutting down V7.1 API...")
    classifier.stop_watcher()
    if keep_alive:
        keep_alive.cancel()


app = FastAPI(
    title="Gmail Smart Sorter V7.1",
    description="Multi-prototype hierarchical email classifier with persistent embedding cache",
    version="7.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    import hmac
    if not config.CLASSIFIER_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API key not set.")
    if not hmac.compare_digest(x_api_key, config.CLASSIFIER_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
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
    return HealthResponse(
        status="ok",
        version=config.VERSION,
        model=config.MODEL_NAME,
        model_loaded=classifier.is_loaded,
        prototype_count=classifier.num_prototypes,
        category_count=classifier.num_categories,
        embedding_dimension=classifier.dimension,
        embedding_store="qudrat",
        embedding_cache="ready" if classifier.is_loaded else "waiting",
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={401: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def classify(request: ClassifyRequest):
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


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(verify_api_key)])
async def submit_feedback(request: FeedbackRequest):
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
    return FeedbackResponse(
        status="ok",
        message="Feedback recorded.",
        id=request.id,
        correct_category=request.correct_category,
    )


@app.post("/evaluate", response_model=EvaluateResponse, dependencies=[Depends(verify_api_key)])
async def evaluate(request: EvaluateRequest):
    logger.info("Evaluating %d benchmark examples.", len(request.examples))

    total = len(request.examples)
    correct = 0
    top3_correct = 0
    auto_sort_correct = 0
    auto_sort_total = 0
    review_total = 0
    unmatched_total = 0
    
    misclassified = []
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    support = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    # Convert BenchmarkExample to EmailInput for batching to utilize cache
    emails_to_classify = [
        EmailInput(
            id=f"eval_{i}",
            subject=ex.subject if ex.subject else ex.text, 
            body=ex.body,
            sender=""
        ) for i, ex in enumerate(request.examples)
    ]
    
    try:
        results = classifier.classify_batch(emails_to_classify)
    except Exception as e:
        logger.error("Evaluation batch classification error: %s", e)
        raise HTTPException(status_code=500, detail="Evaluation failed.")

    for i, result in enumerate(results):
        example = request.examples[i]
        predicted = result.category
        expected = example.expected_category
        top3_cats = [c.category for c in result.top3]

        confusion[expected][predicted] += 1
        support[expected] += 1

        if result.decision == "AUTO_SORT":
            auto_sort_total += 1
            if predicted == expected:
                auto_sort_correct += 1
        elif result.decision == "REVIEW":
            review_total += 1
        else:
            unmatched_total += 1

        if predicted == expected:
            correct += 1
            true_positives[expected] += 1
        else:
            false_positives[predicted] += 1
            false_negatives[expected] += 1
            misclassified.append({
                "text": (example.subject or example.text)[:100],
                "expected": expected,
                "predicted": predicted,
                "decision": result.decision,
                "confidence": result.confidence,
                "top3": [{"category": c.category, "score": c.score} for c in result.top3],
            })

        if expected in top3_cats:
            top3_correct += 1

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

    n_cats = len(per_category) if per_category else 1
    macro_precision = sum(m.precision for m in per_category) / n_cats
    macro_recall = sum(m.recall for m in per_category) / n_cats
    macro_f1 = sum(m.f1 for m in per_category) / n_cats

    accuracy = correct / total if total > 0 else 0.0
    top3_accuracy = top3_correct / total if total > 0 else 0.0
    auto_sort_precision = auto_sort_correct / auto_sort_total if auto_sort_total > 0 else 0.0
    auto_sort_coverage = auto_sort_total / total if total > 0 else 0.0
    review_rate = review_total / total if total > 0 else 0.0
    unmatched_rate = unmatched_total / total if total > 0 else 0.0

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
        auto_sort_precision=round(auto_sort_precision, 4),
        auto_sort_coverage=round(auto_sort_coverage, 4),
        review_rate=round(review_rate, 4),
        unmatched_rate=round(unmatched_rate, 4),
        per_category=per_category,
        confusion=confusion_dict,
        misclassified=misclassified,
    )


# --- Admin Endpoints ---

@app.get("/embedding-cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Return Qdrant cache statistics."""
    if not classifier.is_loaded:
        classifier.load()
    return classifier.store.stats()


@app.post("/embedding-cache/rebuild", dependencies=[Depends(verify_api_key)])
async def cache_rebuild():
    """Drop the Qdrant collection and re-initialize it."""
    classifier.store.rebuild()
    
    # Force classifier to reload and re-encode prototypes
    classifier._is_loaded = False
    classifier.load()
    
    return {"status": "ok", "message": "Embedding cache rebuilt successfully."}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Gmail Smart Sorter V7.1 API", "health": "/health", "version": config.VERSION}
