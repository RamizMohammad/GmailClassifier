"""
Gmail Smart Sorter V6 — Semantic Classifier API

FastAPI application that classifies emails using Sentence Transformer
embeddings and cosine similarity against pre-defined category prototypes.
"""

import asyncio
import logging
import urllib.request
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

import config
from classifier import classifier
from feedback import FeedbackEntry, FeedbackStore, create_feedback_store
from models import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Feedback Store (module-level, initialized in lifespan) ---
feedback_store: FeedbackStore = None  # type: ignore[assignment]


async def keep_alive_task():
    """Background task that pings the health endpoint every 10 seconds."""
    logger.info("Starting 10-second keep-alive pinger...")
    while True:
        try:
            # Simple HTTP GET to self
            urllib.request.urlopen(f"http://127.0.0.1:{config.PORT}/health", timeout=5)
        except Exception as e:
            logger.debug(f"Keep-alive ping failed: {e}")
        await asyncio.sleep(10)


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup."""
    global feedback_store

    logger.info("Starting Gmail Smart Sorter V6 API...")

    # We do NOT load the model immediately. It will load on the first /classify request.
    # Start the memory idle watcher
    classifier.start_watcher()

    # Initialize feedback store
    feedback_store = create_feedback_store()

    # Start the keep-alive task if enabled
    keep_alive = None
    if config.ENABLE_KEEP_ALIVE:
        keep_alive = asyncio.create_task(keep_alive_task())

    logger.info("API ready. Waiting for first classification request to load model.")
    yield

    logger.info("Shutting down API...")
    classifier.stop_watcher()
    if keep_alive:
        keep_alive.cancel()


# --- App ---
app = FastAPI(
    title="Gmail Smart Sorter V6",
    description="Semantic email classifier using Sentence Transformer embeddings",
    version="6.0.0",
    docs_url=None,  # Disable Swagger UI in production
    redoc_url=None,  # Disable ReDoc in production
    lifespan=lifespan,
)


# --- Authentication Dependency ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify the API key from the X-API-Key header."""
    if not config.CLASSIFIER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API key not set.",
        )
    if x_api_key != config.CLASSIFIER_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


# --- Request Size Middleware ---
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests that exceed the maximum allowed size."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large."},
        )
    return await call_next(request)


# --- Exception Handlers ---
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Return cleaner validation error messages."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format. Check your JSON payload."},
    )


# --- Health Endpoint (Public) ---
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint. Does not require authentication.
    Does NOT trigger a model load.
    """
    return HealthResponse(
        status="ok",
        model=config.MODEL_NAME,
        model_loaded=classifier.is_loaded,
        categories_loaded=classifier.num_categories,
    )


# --- Classification Endpoint ---
@app.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={401: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def classify(request: ClassifyRequest):
    """
    Batch email classification endpoint.

    Loads the AI model into RAM if it's currently unloaded.
    """
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

    return ClassifyResponse(results=results)


# --- Feedback Endpoint ---
@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={401: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit classification correction feedback.
    """
    # Force a load if not loaded to ensure categories are populated
    if not classifier.is_loaded:
        try:
            classifier.load()
        except Exception as e:
            logger.error("Failed to load classifier for feedback: %s", e)
            raise HTTPException(status_code=500, detail="Failed to initialize classifier.")

    # Validate that the correct category exists
    if request.correct_category not in classifier.category_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category: '{request.correct_category}'. "
            f"Valid categories: {classifier.category_names}",
        )

    # Save feedback
    entry = FeedbackEntry(
        email_id=request.email_id,
        text=request.text,
        predicted_category=request.predicted_category,
        correct_category=request.correct_category,
    )
    feedback_store.save(entry)

    # Update the category centroid with this real example
    classifier.add_example(
        category=request.correct_category,
        text=request.text,
        weight=0.1,
    )

    logger.info(
        "Feedback processed: email_id=%s, predicted=%s, correct=%s",
        request.email_id,
        request.predicted_category,
        request.correct_category,
    )

    return FeedbackResponse(
        status="ok",
        message="Feedback recorded and classifier updated.",
        email_id=request.email_id,
        correct_category=request.correct_category,
    )


# --- Root ---
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to health check."""
    return {"message": "Gmail Smart Sorter V6 API", "health": "/health"}
