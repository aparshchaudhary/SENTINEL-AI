# SENTINEL AI — API Architecture & Specification
**Course**: CSE276 – Artificial Intelligence Foundations  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document Version**: 1.0 (Phase 1 Planning)

---

## 1. API Design Principles

* **Protocol**: RESTful HTTP over JSON.
* **Base Path**: `/api/v1`
* **Status Codes**: Standard HTTP codes (200 OK, 201 Created, 400 Bad Request, 422 Unprocessable Entity, 500 Internal Server Error).
* **Validation**: Enforced via Pydantic schemas on all incoming payloads and outgoing responses.
* **Error Envelope**: Unified error response schema `{ "detail": "Human readable error message", "error_code": "ERR_TYPE" }`.

---

## 2. Planned API Endpoints

### 2.1 Health & Diagnostics
#### `GET /api/v1/health`
* **Description**: Verifies API availability, database connectivity, and ML model loaded status.
* **Response (200 OK)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-11-17T20:30:00Z",
  "components": {
    "database": "connected",
    "model_loaded": true,
    "model_name": "LogisticRegression_v1",
    "vectorizer_vocab_size": 25000
  }
}
```

---

### 2.2 Sentiment Inference
#### `POST /api/v1/sentiment/predict`
* **Description**: Analyzes a single string of text, returning the classified sentiment polarity and calibrated confidence probabilities.
* **Request Body**:
```json
{
  "text": "The platform architecture is remarkably robust, clean, and intuitive to use!",
  "model_type": "logistic_regression"
}
```
* **Validation Rules**:
  * `text`: String, required, min_length: 2, max_length: 5000 characters.
  * `model_type`: String, optional (default: "logistic_regression", enum: ["logistic_regression", "naive_bayes"]).
* **Response (200 OK)**:
```json
{
  "id": 1042,
  "original_text": "The platform architecture is remarkably robust, clean, and intuitive to use!",
  "cleaned_text": "platform architecture remarkably robust clean intuitive use",
  "sentiment": "positive",
  "confidence": 0.942,
  "probabilities": {
    "positive": 0.942,
    "neutral": 0.043,
    "negative": 0.015
  },
  "model_used": "LogisticRegression",
  "processing_time_ms": 14.8,
  "created_at": "2026-11-17T20:30:15Z"
}
```

---

### 2.3 Batch CSV Analysis
#### `POST /api/v1/sentiment/batch-csv`
* **Description**: Accepts a CSV file upload, identifies the target text column, classifies every record in bulk, and yields aggregate metrics alongside annotated rows.
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `file`: UploadFile (required, `.csv` format, max size 10MB).
  * `text_column`: String (optional, defaults to automatic detection or "text").
* **Response (200 OK)**:
```json
{
  "job_id": "job_94a2f8c1",
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
  "average_confidence": 0.887,
  "processing_duration_seconds": 1.42,
  "preview_results": [
    {
      "row_index": 0,
      "text": "Excellent support from the engineering team.",
      "sentiment": "positive",
      "confidence": 0.961
    }
  ]
}
```

---

### 2.4 Analytics & Aggregations
#### `GET /api/v1/analytics/summary`
* **Description**: Retrieves aggregate sentiment distribution, total analyses executed, and global average confidence metrics.
* **Query Parameters**:
  * `timeframe`: String, optional (`"all"`, `"today"`, `"7d"`, `"30d"`).
* **Response (200 OK)**:
```json
{
  "total_analyzed": 1420,
  "distribution": {
    "positive": 820,
    "neutral": 360,
    "negative": 240
  },
  "percentages": {
    "positive": 57.75,
    "neutral": 25.35,
    "negative": 16.90
  },
  "average_confidence": 0.894,
  "last_analysis_at": "2026-11-17T20:28:40Z"
}
```

#### `GET /api/v1/analytics/trends`
* **Description**: Time-series grouping of sentiment counts over time for Recharts trend lines.
* **Query Parameters**:
  * `interval`: String, optional (`"hour"`, `"day"`, `"week"`).
* **Response (200 OK)**:
```json
{
  "timeline": [
    { "timestamp": "2026-11-17T12:00:00Z", "positive": 45, "neutral": 12, "negative": 8 },
    { "timestamp": "2026-11-17T13:00:00Z", "positive": 60, "neutral": 18, "negative": 14 }
  ]
}
```

---

### 2.5 Analysis History
#### `GET /api/v1/sentiment/history`
* **Description**: Paginated list of historical sentiment analysis records.
* **Query Parameters**:
  * `page`: Integer (default: 1, min: 1)
  * `page_size`: Integer (default: 20, max: 100)
  * `sentiment`: String, optional (`"positive"`, `"negative"`, `"neutral"`)
  * `search`: String, optional (keyword filtering on text content)
* **Response (200 OK)**:
```json
{
  "items": [
    {
      "id": 1042,
      "text": "The platform architecture is remarkably robust...",
      "sentiment": "positive",
      "confidence": 0.942,
      "created_at": "2026-11-17T20:30:15Z"
    }
  ],
  "total": 1042,
  "page": 1,
  "page_size": 20,
  "total_pages": 53
}
```

#### `DELETE /api/v1/sentiment/history/{record_id}`
* **Description**: Deletes an analysis entry from the history log.
* **Response (200 OK)**:
```json
{
  "success": true,
  "message": "Record 1042 successfully deleted"
}
```

---

### 2.6 Model Performance & Metadata
#### `GET /api/v1/model/info`
* **Description**: Exposes metadata about the active ML model, vectorizer parameters, and feature dimensions.
* **Response (200 OK)**:
```json
{
  "active_model": "Multinomial Logistic Regression (L2)",
  "vectorizer": "TfidfVectorizer",
  "vocabulary_size": 25000,
  "ngram_range": [1, 2],
  "classes": ["negative", "neutral", "positive"],
  "dataset_trained_on": "Sentiment140 + Academic Multi-Domain Corpus",
  "training_sample_size": 50000,
  "last_trained": "2026-11-15T18:00:00Z"
}
```

#### `GET /api/v1/model/metrics`
* **Description**: Provides comprehensive evaluation statistics (accuracy, macro precision, recall, F1-score, and confusion matrix).
* **Response (200 OK)**:
```json
{
  "accuracy": 0.864,
  "macro_f1": 0.858,
  "classes": {
    "negative": { "precision": 0.87, "recall": 0.85, "f1_score": 0.86 },
    "neutral": { "precision": 0.82, "recall": 0.81, "f1_score": 0.815 },
    "positive": { "precision": 0.89, "recall": 0.91, "f1_score": 0.90 }
  },
  "confusion_matrix": [
    [850, 95, 55],
    [110, 810, 80],
    [40, 50, 910]
  ]
}
```
