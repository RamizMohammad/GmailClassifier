# Gmail Smart Sorter V6 — Semantic Classifier API

A lightweight, production-quality REST API that classifies emails using **Sentence Transformer embeddings** and **cosine similarity**. Designed as the AI layer of Gmail Smart Sorter V6, deployed on **Render Free** (≤512 MB RAM, CPU-only).

> **This is NOT a generative AI service.** No LLMs, no text generation. Pure embedding-based semantic similarity classification.

## Architecture

```
Google Apps Script
        │
        │ HTTPS POST /classify
        ▼
┌─────────────────────────┐
│   Render FastAPI API     │
│                          │
│  ┌────────────────────┐  │
│  │ Sentence Transformer│  │
│  │ (all-MiniLM-L6-v2) │  │
│  └────────┬───────────┘  │
│           │               │
│  ┌────────▼───────────┐  │
│  │ Email → Embedding  │  │
│  └────────┬───────────┘  │
│           │               │
│  ┌────────▼───────────┐  │
│  │ Cosine Similarity  │  │
│  │ vs Category Protos │  │
│  └────────┬───────────┘  │
│           │               │
│  ┌────────▼───────────┐  │
│  │ Classification     │  │
│  │ Result + Confidence│  │
│  └────────────────────┘  │
└─────────────┬────────────┘
              │
              ▼
Google Apps Script
   → Gmail Labels / Archive / Review
```

## Project Structure

```
gmail-sorter-v6/
├── app.py              # FastAPI application (endpoints, auth, middleware)
├── classifier.py       # Sentence Transformer classifier (model, embeddings, similarity)
├── models.py           # Pydantic request/response models
├── config.py           # Environment-based configuration
├── feedback.py         # Pluggable feedback storage
├── categories.json     # Category prototypes (semantic descriptions)
├── requirements.txt    # Python dependencies (CPU-only PyTorch)
├── render.yaml         # Render deployment blueprint
├── .gitignore
└── tests/
    ├── test_health.py
    └── test_classifier.py
```

## Local Development

### Prerequisites

- Python 3.11+
- ~2 GB disk space (for model download on first run)

### Setup

```bash
cd gmail-sorter-v6

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest httpx

# Set required environment variable
export CLASSIFIER_API_KEY="your-dev-api-key"
```

### Run Locally

```bash
uvicorn app:app --reload --port 8000
```

The first startup will download the model (~80 MB). Subsequent starts use the cached model.

### Run Tests

```bash
export CLASSIFIER_API_KEY="test-api-key-12345"
pytest tests/ -v
```

## API Reference

### `GET /health` — Health Check (Public)

No authentication required.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "model_loaded": true,
  "categories_loaded": 24
}
```

### `POST /classify` — Batch Classification (Authenticated)

Requires `X-API-Key` header.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-dev-api-key" \
  -d '{
    "emails": [
      {
        "id": "msg-001",
        "sender": "jobs@linkedin.com",
        "subject": "Software Engineering Lead - Python",
        "body": "We are hiring a Software Engineering Lead with Python experience."
      },
      {
        "id": "msg-002",
        "sender": "newsletter@deeplearning.ai",
        "subject": "Master In-Demand AI & ML Skills",
        "body": "Enroll in our comprehensive AI and Machine Learning course."
      }
    ]
  }'
```

Response:
```json
{
  "results": [
    {
      "id": "msg-001",
      "category": "Jobs/Job Alerts",
      "similarity": 0.8723,
      "confidence": 0.8892,
      "confidence_level": "high",
      "margin": 0.1245,
      "top_categories": [
        {"category": "Jobs/Job Alerts", "score": 0.8723},
        {"category": "Jobs/Recruiters", "score": 0.7478},
        {"category": "Development/Programming", "score": 0.5912}
      ]
    },
    {
      "id": "msg-002",
      "category": "Learning/Courses",
      "similarity": 0.8156,
      "confidence": 0.8401,
      "confidence_level": "high",
      "margin": 0.0987,
      "top_categories": [
        {"category": "Learning/Courses", "score": 0.8156},
        {"category": "Learning/AI-Tech", "score": 0.7169},
        {"category": "Promotions", "score": 0.4523}
      ]
    }
  ]
}
```

### `POST /feedback` — Classification Correction (Authenticated)

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-dev-api-key" \
  -d '{
    "email_id": "msg-001",
    "text": "Software Engineering Lead - Python. We are hiring...",
    "predicted_category": "Development/Programming",
    "correct_category": "Jobs/Job Alerts"
  }'
```

Response:
```json
{
  "status": "ok",
  "message": "Feedback recorded and classifier updated.",
  "email_id": "msg-001",
  "correct_category": "Jobs/Job Alerts"
}
```

## Confidence Levels

The API returns a `confidence_level` field to help Apps Script decide how to handle each classification:

| Level | Criteria | Recommended Action |
|-------|----------|-------------------|
| `high` | similarity ≥ 0.78 AND margin ≥ 0.08 | Auto-sort (apply label, archive) |
| `medium` | similarity ≥ 0.65 AND margin ≥ 0.04 | Send to Review label |
| `low` | Below medium thresholds | Leave in inbox, don't auto-sort |

### Understanding the Metrics

- **`similarity`**: Cosine similarity between the email embedding and the best-matching category centroid (range: -1 to 1, higher = better match)
- **`confidence`**: Weighted combination of similarity (60%) and normalized margin (40%)
- **`margin`**: Difference between the best and second-best category scores. A high similarity with a low margin means the email is ambiguous between categories
- **`confidence_level`**: Categorical label derived from the thresholds above

## How Apps Script V6 Should Consume the API

### Recommended Flow

```javascript
function classifyEmails(emails) {
  const API_URL = "https://your-service.onrender.com/classify";
  const API_KEY = PropertiesService.getScriptProperties().getProperty("CLASSIFIER_API_KEY");

  // Build the request payload
  const payload = {
    emails: emails.map(email => ({
      id: email.getId(),
      sender: email.getFrom(),
      subject: email.getSubject(),
      body: email.getPlainBody().substring(0, 5000)  // Truncate
    }))
  };

  const options = {
    method: "post",
    contentType: "application/json",
    headers: { "X-API-Key": API_KEY },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  const response = UrlFetchApp.fetch(API_URL, options);
  const data = JSON.parse(response.getContentText());

  // Process results
  data.results.forEach(result => {
    // Combine with rule-based scores if available
    const ruleScore = getRuleScore(result.id);  // Your existing rule engine

    if (result.confidence_level === "high" || ruleScore > 0.9) {
      applyLabel(result.id, result.category);
      archiveThread(result.id);
    } else if (result.confidence_level === "medium") {
      applyLabel(result.id, "Review");
    }
    // Low confidence: leave in inbox
  });
}
```

### Batch Size Recommendations

- **Optimal**: 10-50 emails per batch
- **Maximum**: 100 emails per batch
- Process in batches if you have more than 100 emails

### Handling Cold Starts

Render Free services sleep after ~15 minutes of inactivity. The first request after sleep triggers a cold start (~15-30 seconds). To handle this:

```javascript
// Pre-warm the service before classification
function warmUp() {
  const response = UrlFetchApp.fetch("https://your-service.onrender.com/health", {
    muteHttpExceptions: true
  });
  return JSON.parse(response.getContentText()).model_loaded;
}
```

## Render Deployment

### Option 1: Blueprint (Recommended)

1. Push this repository to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New** → **Blueprint**
4. Connect your GitHub repo
5. Render will auto-detect `render.yaml`
6. Set `CLASSIFIER_API_KEY` in the Render dashboard (Environment → Add Environment Variable)

### Option 2: Manual Setup

1. **New** → **Web Service** → Connect GitHub repo
2. **Runtime**: Python 3
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. **Plan**: Free
6. **Environment Variables**:
   - `CLASSIFIER_API_KEY` = (generate a strong random key)
   - `PYTHON_VERSION` = `3.11.9`

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLASSIFIER_API_KEY` | **Yes** | — | API key for authentication |
| `MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformer model |
| `MAX_BATCH_SIZE` | No | `100` | Maximum emails per batch |
| `MAX_BODY_LENGTH` | No | `5000` | Max characters of email body to process |
| `CONFIDENCE_HIGH_SIMILARITY` | No | `0.78` | High confidence similarity threshold |
| `CONFIDENCE_HIGH_MARGIN` | No | `0.08` | High confidence margin threshold |
| `CONFIDENCE_MEDIUM_SIMILARITY` | No | `0.65` | Medium confidence similarity threshold |
| `CONFIDENCE_MEDIUM_MARGIN` | No | `0.04` | Medium confidence margin threshold |
| `FEEDBACK_STORE_TYPE` | No | `memory` | Feedback store: `memory` or `json_file` |
| `TOP_K` | No | `3` | Number of top categories to return |

## Memory & Performance

### Memory Budget (~512 MB on Render Free)

| Component | Estimated Memory |
|-----------|-----------------|
| Python runtime | ~30 MB |
| FastAPI + Uvicorn | ~20 MB |
| PyTorch (CPU) | ~80 MB |
| Model weights | ~80 MB |
| Category embeddings | <1 MB |
| Request processing headroom | ~100 MB |
| **Total estimated** | **~310 MB** |

### Performance Optimizations

1. **Model loaded once** at startup — never reloaded per request
2. **Category embeddings cached** at startup — not recomputed per request
3. **Normalized embeddings** — cosine similarity via fast dot product
4. **Batch encoding** — all emails in one `model.encode()` call
5. **Body truncation** — emails limited to 5,000 characters
6. **CPU-only PyTorch** — ~1 GB smaller than full PyTorch
7. **No unnecessary dependencies** — minimal `requirements.txt`

### Cold Start Timeline

| Phase | Time |
|-------|------|
| Python + import | ~2s |
| Model load (cached) | ~3-5s |
| Category embedding | ~1-2s |
| **Total cold start** | **~6-9s** |

First-ever deploy (model download): ~30-60s additional.

## Categories (24 Total)

Security • Finance/UPI • Finance/Credit Cards • Finance/Investments • Finance/Banks • Finance/Bills • Jobs/Job Alerts • Jobs/Interviews • Jobs/Recruiters • Development/GitHub • Development/Server Alerts • Development/Programming • Learning/Courses • Learning/Events • Learning/AI-Tech • Shopping/Amazon • Shopping/Flipkart • Shopping/Other • Travel • Social • Newsletters • Promotions • Work • Work/Reports • Work/HR • Review

## Security

- **API key authentication** on all classification endpoints
- **Request size limits** (10 MB max)
- **Batch size limits** (100 emails max)
- **Body length limits** (5,000 chars)
- **No email content logging** by default
- **No Gmail credentials** stored
- **No persistent email storage**
- **CORS disabled** by default
- **Swagger UI disabled** in production

## License

Private — Gmail Smart Sorter V6.
