# SENTINEL AI — System Architecture Specification
**Course**: CSE276 – Artificial Intelligence Foundations  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document Version**: 1.0 (Phase 1)

---

## 1. System Overview

**SENTINEL AI** is a production-grade, full-stack sentiment intelligence system engineered to analyze text streams and tabular datasets in real-time. Designed with modularity and academic rigor, it classifies sentiment into three discrete polarity states: **Positive**, **Negative**, and **Neutral**, backed by real statistical Machine Learning (NLP) classifiers rather than simulated outputs or arbitrary heuristics.

```
+-----------------------------------------------------------------------+
|                             USER CLIENT                               |
|               (React.js + Tailwind CSS + Lucide + Recharts)           |
+-----------------------------------------------------------------------+
                                   |
                         HTTPS / RESTful JSON
                                   v
+-----------------------------------------------------------------------+
|                           FASTAPI GATEWAY                             |
|  - CORS & Security Middleware       - Request Validation (Pydantic)  |
|  - API Routing (v1 endpoints)       - Exception Handlers             |
+-----------------------------------------------------------------------+
           |                                             |
           v                                             v
+-----------------------+                    +--------------------------+
|  SENTIMENT INFERENCE  |                    |    PERSISTENCE LAYER     |
|       SERVICE         |                    |      (SQLite / ORM)      |
| - Text Preprocessor   |                    | - Analysis Records       |
| - TF-IDF Vectorizer   |                    | - Batch Upload Logs      |
| - ML Model (LR / MNB) |                    | - Performance Metrics    |
+-----------------------+                    +--------------------------+
```

---

## 2. Architectural Pillars

### 2.1 Frontend Architecture
* **Framework**: React 18 with Vite for ultra-fast compilation and Hot Module Replacement (HMR).
* **Styling & UI**: Tailwind CSS coupled with utility helpers (`clsx`, `tailwind-merge`) creating an enterprise SaaS aesthetic (dark/light neutral mode, glassmorphic accents, high-contrast visual cues).
* **Component Architecture**:
  * **Presentational Components** (`components/common/`): Pure, reusable UI atoms (Buttons, Badges, Cards, Modals, FileDropzone).
  * **Domain Components** (`components/sentiment/`): Sentiment input box, polarity indicator meter, confidence progress bars, keyword highlighters.
  * **Dashboard & Visualizations** (`components/dashboard/`): Recharts-powered pie charts for sentiment distribution, bar charts for confidence distribution, and timeline charts.
  * **Pages & Layouts** (`pages/`, `layouts/`): App Shell with Sidebar/Header navigation, Real-time Analyzer page, Batch CSV Analysis page, History Explorer page, and Model Benchmarks page.
* **State Management & Service Integration**:
  * Custom hooks (`hooks/useSentiment.ts`, `hooks/useBatchAnalysis.ts`, `hooks/useHistory.ts`) managing async request states, loading states, and error propagation.
  * Axios-based API client layer (`services/api.ts`) with standardized response interceptors.

### 2.2 Backend Architecture
* **Web Framework**: FastAPI with Python 3.
* **ASGI Server**: Uvicorn running asynchronous workers.
* **Design Pattern**: Layered Service-Repository Architecture:
  1. **Routing Layer (`app/api/`)**: Declares REST endpoints, handles HTTP status codes, and routes requests to appropriate service handlers.
  2. **Schema Layer (`app/schemas/`)**: Pydantic models for strict input validation, type coercion, and response serialization.
  3. **Service Layer (`app/services/`)**: Orchestrates business logic, model inference calls, and transaction orchestration.
  4. **Data Access Layer (`app/database/`, `app/models/`)**: SQLAlchemy models and scoped session management for SQLite persistence.
  5. **Core Layer (`app/core/`)**: Application configuration, logging configurations, constants, and custom exception classes.

### 2.3 Machine Learning Architecture
* **Subsystem**: Dedicated `ml/` package completely decoupled from the web runtime to ensure clean testing and reproducibility.
* **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) converting cleaned text tokens into high-dimensional sparse feature vectors with n-gram extraction (unigrams + bigrams).
* **Classifiers**:
  * **Multinomial Naive Bayes (MNB)**: Probabilistic baseline classifier leveraging Bayes' theorem with laplace smoothing.
  * **Logistic Regression (LR)**: Multinomial logistic regression with L2 regularization, delivering calibrated posterior probabilities via softmax.
* **Model Serialization**: Serialized artifact bundles (`model.joblib`, `vectorizer.joblib`, `metadata.json`) loaded into memory at backend startup during FastAPI lifespan events, eliminating per-request deserialization overhead.

### 2.4 Database Architecture
* **Engine**: SQLite 3 (zero-configuration, embedded ACID-compliant relational storage).
* **Tables**:
  * `sentiment_analyses`: Stores individual predictions (input text, cleaned text, predicted sentiment, confidence score, probabilities breakdown, timestamp, processing latency).
  * `batch_jobs`: Tracks CSV upload sessions (file name, total rows, processed count, distribution summary, timestamp).
  * `batch_items`: Stores row-level results linked to specific `batch_jobs` via foreign keys.
  * `model_metrics`: Stores evaluation benchmarks (accuracy, precision, recall, macro F1, test split size, evaluation date).

### 2.5 Data Processing Workflow
1. **Single Text Analysis Flow**:
   * User enters text in React UI -> Dispatched via Axios POST request.
   * FastAPI validates input via `SentimentRequest` Pydantic schema (length checks, empty string validation).
   * Request forwarded to `SentimentService`.
   * Raw text passed to `TextPreprocessor` (regex cleaning, lowercasing, tokenization, stopword removal).
   * Preprocessed tokens transformed via `TfidfVectorizer.transform()`.
   * Loaded ML model executes `.predict()` and `.predict_proba()`.
   * Result enriched with confidence score, processing latency, and dominant sentiment.
   * Analysis asynchronously saved to SQLite database via SQLAlchemy.
   * Structured JSON response returned to React client -> Visualized in UI with color-coded polarity badges and Recharts confidence gauges.

2. **Bulk CSV Analysis Flow**:
   * User uploads `.csv` file via drag-and-drop zone.
   * Backend validates file MIME type and column structure.
   * Background task / streaming batch processor chunks text rows, executes batch vectorization and vector inference.
   * Aggregates summary statistics (total positive, negative, neutral percentages).
   * Persists batch run and individual records.
   * Returns batch completion report with downloadable annotated results.
