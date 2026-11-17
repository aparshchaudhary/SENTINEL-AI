# SENTINEL AI – Backend Architecture & System Blueprint
**Course**: CSE276 – Artificial Intelligence Foundations  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document Version**: 4.0 (Phase 4 Backend & Database Integration)

---

## 1. System Architecture Overview

The **SENTINEL AI** backend is built on **FastAPI** (Python 3.14 / ASGI) using a strict layered architecture:

```
                                  [ USER CLIENT / FRONTEND ]
                                              │
                                     RESTful JSON (HTTP)
                                              ▼
                             +─────────────────────────────────+
                             |       Uvicorn ASGI Server       |
                             +─────────────────────────────────+
                                              │
                             +─────────────────────────────────+
                             |         FastAPI Gateway         |
                             |  - CORS & Security Middleware   |
                             |  - Global Exception Interceptor |
                             |  - API v1 Aggregator Router     |
                             +─────────────────────────────────+
                                              │
              ┌───────────────────────────────┴──────────────────────────────┐
              ▼                                                              ▼
+───────────────────────────+                                  +───────────────────────────+
|     CONTROLLER LAYER      |                                  |     CONTROLLER LAYER      |
|  (Endpoints & Schemas)    |                                  |  (Endpoints & Schemas)    |
| - sentiment.py, batch.py  |                                  | - analytics.py, history.py|
+───────────────────────────+                                  +───────────────────────────+
              │                                                              │
              └───────────────────────────────┬──────────────────────────────┘
                                              ▼
                             +─────────────────────────────────+
                             |          SERVICE LAYER          |
                             |     (SentimentService Singl.)   |
                             +─────────────────────────────────+
                                      │               │
                     ┌────────────────┘               └────────────────┐
                     ▼                                                 ▼
+─────────────────────────────────────────+       +─────────────────────────────────────────+
|       ML INFERENCE ENGINE (OFFLINE)     |       |          PERSISTENCE LAYER (ACID)       |
| - NLPPreprocessor (Negation-Preserving) |       | - SQLite Relational Database            |
| - Fitted TfidfVectorizer (7,235 N-grams)|       | - SQLAlchemy 2.0 ORM Engine             |
| - Champion Logistic Regression Model    |       | - Tables: analyses, batch_jobs, metrics |
+─────────────────────────────────────────+       +─────────────────────────────────────────+
```

---

## 2. Key Architectural Modules

### 2.1 Lifespan Management (`backend/main.py`)
* During FastAPI initialization, the `@asynccontextmanager lifespan` executes:
  1. **SQLite Database Schema Initialization**: Verifies and provisions all required tables via `Base.metadata.create_all(bind=engine)`.
  2. **Model Memory Warm-Up**: Loads the champion Logistic Regression model and fitted TF-IDF vectorizer into memory **once** on startup, avoiding costly disk deserialization overhead per request.
  3. **Initial Metric Seeding**: Seeds test holdout benchmarks into the `model_metrics` table if unpopulated.

### 2.2 Reusable Service Layer (`backend/app/services/sentiment_service.py`)
* Implements a **Thread-Safe Singleton** coordinating:
  * Single-text inference and transaction recording.
  * Streaming batch CSV processing with column heuristic detection.
  * Analytical trend, summary, and distribution queries.
  * Full-text search and paginated history filtering.

### 2.3 Persistence Layer (`backend/app/database/`, `backend/app/models/`)
* **Engine**: SQLite with SQLAlchemy 2.0.
* **Connection Pooling**: `connect_args={"check_same_thread": False}` enabling asynchronous multi-worker request handling.
* **Scoped Session Dependency (`get_db`)**: Yields an isolated database session per request, guaranteeing that transactions commit or rollback cleanly without leaking connections.

---

## 3. Database Schema Specification

### Table 1: `sentiment_analyses`
Stores individual real-time analysis requests.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique transaction identifier |
| `input_text` | TEXT | NOT NULL | Raw submitted text |
| `cleaned_text` | TEXT | NULLABLE | Preprocessed tokens without stopwords |
| `predicted_sentiment` | VARCHAR(20) | NOT NULL, INDEXED | Predicted polarity (`positive`, `negative`, `neutral`) |
| `confidence` | FLOAT | NOT NULL | Maximum estimated posterior probability ($[0.0, 1.0]$) |
| `positive_probability`| FLOAT | NOT NULL | Estimated probability of positive polarity |
| `negative_probability`| FLOAT | NOT NULL | Estimated probability of negative polarity |
| `neutral_probability` | FLOAT | NOT NULL | Estimated probability of neutral polarity |
| `processing_time_ms`  | FLOAT | NOT NULL | Execution latency in milliseconds |
| `created_at` | DATETIME | NOT NULL, INDEXED | UTC creation timestamp |

### Table 2: `batch_jobs`
Stores metadata and distribution statistics for bulk CSV upload sessions.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR(50) | PRIMARY KEY, INDEXED | UUID job identifier (e.g. `job_a4f8b912c3de`) |
| `filename` | VARCHAR(255)| NOT NULL | Uploaded file name |
| `total_records` | INTEGER | NOT NULL | Total rows classified |
| `positive_count`| INTEGER | DEFAULT 0 | Total positive classifications |
| `negative_count`| INTEGER | DEFAULT 0 | Total negative classifications |
| `neutral_count` | INTEGER | DEFAULT 0 | Total neutral classifications |
| `status` | VARCHAR(30) | DEFAULT 'completed' | Job processing status |
| `annotated_filename`| VARCHAR(255)| NULLABLE | Path/name of downloadable annotated CSV |
| `created_at` | DATETIME | NOT NULL, INDEXED | UTC timestamp |

### Table 3: `model_metrics`
Stores holdout test benchmarks for transparency and auditing.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique ID |
| `model_name` | VARCHAR(100)| NOT NULL | Champion algorithm name |
| `accuracy` | FLOAT | NOT NULL | Holdout test accuracy |
| `precision` | FLOAT | NOT NULL | Macro precision |
| `recall` | FLOAT | NOT NULL | Macro recall |
| `macro_f1` | FLOAT | NOT NULL | Macro F1-score |
| `weighted_f1` | FLOAT | NOT NULL | Weighted F1-score |
| `created_at` | DATETIME | DEFAULT UTC | Record creation date |

---

## 4. Security & Robustness Measures

1. **Input Validation**: Enforced at the gateway via Pydantic schemas. String inputs must be non-empty, non-whitespace, and $\le 5,000$ characters.
2. **Safe CSV Handling**: CSV files are checked for MIME type, file extension, and restricted to a maximum size of 10MB to prevent Denial of Service (DoS) attacks.
3. **Internal Error Masking**: Unhandled exceptions are intercepted by a centralized error handler returning a standard error code without leaking server file paths or Python stack traces.
4. **CORS Governance**: Controlled via `settings.CORS_ORIGINS` to allow communication only with authorized frontend origins (e.g., Vite development port 5173).
