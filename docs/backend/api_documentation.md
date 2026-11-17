# SENTINEL AI – Backend API Specification & Developer Guide
**Course**: CSE276 – Artificial Intelligence Foundations  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document Version**: 4.0 (Phase 4 Backend & Database Integration)

---

## 1. Quickstart & Execution Guide

### 1.1 Prerequisites
* Python 3.10+ (tested on Python 3.14.6)
* SQLite 3

### 1.2 Installation
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
# Windows:
..\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 1.3 Running the Server
```bash
# Start Uvicorn ASGI server on port 8000
python main.py
# Or directly via Uvicorn CLI:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

* **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
* **Interactive ReDoc UI**: `http://127.0.0.1:8000/redoc`
* **OpenAPI JSON Spec**: `http://127.0.0.1:8000/openapi.json`

---

## 2. API Endpoints Reference

Base URL: `http://127.0.0.1:8000/api/v1`

---

### 2.1 Single Text Sentiment Prediction

#### `POST /api/v1/sentiment/predict`
Analyzes a single input text using the trained champion Logistic Regression NLP model.

* **Request Headers**: `Content-Type: application/json`
* **Request Body**:
```json
{
  "text": "I absolutely loved this flight experience, fantastic customer service and comfortable seating!"
}
```
* **Validation Rules**: `text` must be between 1 and 5,000 characters, non-empty, and non-whitespace.
* **Response (200 OK)**:
```json
{
  "text": "I absolutely loved this flight experience, fantastic customer service and comfortable seating!",
  "cleaned_text": "absolutely love flight experience fantastic customer service comfortable seat",
  "sentiment": "positive",
  "confidence": 0.8085,
  "probabilities": {
    "positive": 0.8085,
    "negative": 0.1544,
    "neutral": 0.0371
  },
  "processing_time_ms": 1.24,
  "id": 142,
  "created_at": "2026-11-17T15:39:00.123456Z"
}
```

---

### 2.2 Bulk CSV File Analysis

#### `POST /api/v1/sentiment/batch-csv`
Uploads and classifies an entire CSV dataset of texts in bulk.

* **Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `file`: UploadFile (required, `.csv` format, $\le 10$ MB).
  * `text_column`: String (optional, target column name if auto-detection is not desired).
* **Response (200 OK)**:
```json
{
  "job_id": "job_94a2f8c14e02",
  "filename": "customer_feedback.csv",
  "total_records": 500,
  "processed_records": 500,
  "distribution": {
    "positive": 310,
    "neutral": 115,
    "negative": 75
  },
  "percentages": {
    "positive": 62.0,
    "neutral": 23.0,
    "negative": 15.0
  },
  "average_confidence": 0.8872,
  "processing_duration_seconds": 0.42,
  "preview_results": [
    {
      "row_index": 0,
      "text": "Excellent support from the engineering team.",
      "sentiment": "positive",
      "confidence": 0.9612
    }
  ],
  "download_url": "/api/v1/sentiment/batch-csv/download/job_94a2f8c14e02"
}
```

#### `GET /api/v1/sentiment/batch-csv/download/{job_id}`
Downloads the resulting CSV file annotated with:
* `sentiment_predicted`
* `confidence_score`
* `prob_positive`
* `prob_negative`
* `prob_neutral`

---

### 2.3 Analytics & Reporting

#### `GET /api/v1/analytics/summary`
Returns global sentiment distributions and statistics across all single queries and batch uploads.

* **Response (200 OK)**:
```json
{
  "total_analyses": 508,
  "positive_count": 314,
  "negative_count": 78,
  "neutral_count": 116,
  "percentages": {
    "positive": 61.81,
    "negative": 15.35,
    "neutral": 22.83
  },
  "average_confidence": 0.8854
}
```

#### `GET /api/v1/analytics/trends`
Returns daily sentiment frequencies formatted for Recharts timeline area charts.

* **Response (200 OK)**:
```json
{
  "timeline": [
    {
      "date": "2026-11-17",
      "positive": 45,
      "negative": 12,
      "neutral": 8,
      "total": 65
    }
  ]
}
```

#### `GET /api/v1/analytics/distribution`
Returns categorized polarities formatted for Recharts Pie charts with theme color mappings.

* **Response (200 OK)**:
```json
{
  "distribution": [
    { "name": "Positive", "value": 314, "percentage": 61.81, "color": "#10b981" },
    { "name": "Neutral", "value": 116, "percentage": 22.83, "color": "#f59e0b" },
    { "name": "Negative", "value": 78, "percentage": 15.35, "color": "#ef4444" }
  ],
  "total_samples": 508
}
```

---

### 2.4 Analysis History & Audit Trail

#### `GET /api/v1/sentiment/history`
Retrieves paginated, searchable, filterable history records.

* **Query Parameters**:
  * `page`: Integer (default: 1)
  * `page_size`: Integer (default: 20, max: 100)
  * `sentiment`: String (`positive`, `negative`, `neutral`)
  * `search`: String (matches substring of `input_text`)
  * `start_date`: ISO Date string
  * `end_date`: ISO Date string
* **Response (200 OK)**:
```json
{
  "items": [
    {
      "id": 142,
      "input_text": "I absolutely loved this flight experience...",
      "cleaned_text": "absolutely love flight experience...",
      "predicted_sentiment": "positive",
      "confidence": 0.8085,
      "probabilities": { "positive": 0.8085, "negative": 0.1544, "neutral": 0.0371 },
      "processing_time_ms": 1.24,
      "created_at": "2026-11-17T15:39:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

#### `DELETE /api/v1/sentiment/history/{record_id}`
Deletes an individual history record from the database.

* **Response (200 OK)**:
```json
{
  "success": true,
  "message": "Analysis record 142 deleted successfully.",
  "id": 142
}
```

---

### 2.5 Model Intelligence & Benchmarks

#### `GET /api/v1/model/info`
Exposes active classifier architecture, algorithm, and training configuration.

* **Response (200 OK)**:
```json
{
  "model_name": "Multinomial Logistic Regression (L2, Balanced)",
  "algorithm": "logistic_regression",
  "supported_classes": ["negative", "neutral", "positive"],
  "vocabulary_size": 7235,
  "training_dataset_size": 11524,
  "testing_dataset_size": 2881,
  "hyperparameters": {
    "C": 0.5,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs"
  },
  "training_date": "2026-11-17T20:55:57.123456"
}
```

#### `GET /api/v1/model/metrics`
Returns verified statistical evaluation metrics on the untouched holdout test partition.

* **Response (200 OK)**:
```json
{
  "model_name": "Multinomial Logistic Regression (L2, Balanced)",
  "accuracy": 0.7643,
  "precision_macro": 0.7017,
  "recall_macro": 0.7320,
  "f1_macro": 0.7138,
  "f1_weighted": 0.7712,
  "class_metrics": {
    "negative": { "precision": 0.8934, "recall": 0.8083, "f1_score": 0.8487 },
    "neutral": { "precision": 0.5430, "recall": 0.6634, "f1_score": 0.5972 },
    "positive": { "precision": 0.6687, "recall": 0.7243, "f1_score": 0.6954 }
  },
  "confusion_matrix": [
    [1467, 257, 91],
    [132, 404, 73],
    [43, 83, 331]
  ]
}
```

---

### 2.6 Health Diagnostics

#### `GET /api/v1/health`
Validates full stack service health and runtime readiness.

* **Response (200 OK)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "active_model": "Multinomial Logistic Regression (L2, Balanced)",
  "timestamp": "2026-11-17T15:39:18.123456Z"
}
```

---

## 3. Automated Testing Instructions

Run the 14-test automated suite using `pytest`:

```bash
# Run tests with verbose output
python -m pytest tests/test_backend.py -v

# Or execute the test module directly:
python tests/test_backend.py
```
Tests automatically spin up an isolated temporary database in `tmp` and drop it upon completion to prevent any corruption of the production SQLite database.
