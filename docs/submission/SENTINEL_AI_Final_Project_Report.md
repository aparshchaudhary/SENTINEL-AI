# SENTINEL AI: Intelligent Sentiment Analysis Platform
## Academic Project Report for CSE276 – Artificial Intelligence Foundations
**Department of Computer Science & Engineering**  
**Academic Year**: 2026–2027  

---

# 1. Cover Page

* **Project Title**: SENTINEL AI – Intelligent Sentiment Analysis Platform
* **Academic Course**: CSE276 – Artificial Intelligence Foundations
* **Domain**: Artificial Intelligence, Natural Language Processing (NLP), Full-Stack Systems
* **Target Audience**: Academic Evaluators, Software Architects, NLP Researchers
* **Software Version**: 1.0.0
* **Date of Submission**: November 2026
* **Status**: Complete & Verified (Phases 1–8)

---

# 2. Abstract

Sentiment analysis—the automated computational categorization of textual sentiment polarity—is a foundational pillar of modern Artificial Intelligence and Natural Language Processing (NLP). In commercial ecosystems, processing unstructured customer feedback across millions of remarks requires scalable, low-latency, and mathematically grounded classification architectures. 

**SENTINEL AI** is an enterprise-grade, end-to-end intelligent sentiment analysis platform engineered to ingest unstructured text and classify it into three discrete sentiment polarities: **Positive**, **Negative**, and **Neutral**. The platform is powered by a genuine supervised machine learning pipeline trained and evaluated on 14,405 verified customer feedback records (CrowdFlower US Airline Sentiment dataset). 

Textual preprocessing features deterministic regular expression cleaning, mention/URL elimination, NLTK tokenization, WordNet lemmatization, and **strict negation preservation** to preserve semantic inversions (e.g., transforming `"not happy"` into `"not_happy"`). Feature extraction utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization across unigrams and bigrams with sublinear term frequency scaling, yielding a curated vocabulary of 7,235 mathematical features. 

Through 5-fold stratified cross-validation and rigorous hyperparameter tuning, a **Multinomial Logistic Regression** model ($C=0.5$, balanced class weighting, `lbfgs` solver) was established as champion, achieving a holdout test accuracy of **76.43%**, a macro F1-score of **0.7138**, and a weighted F1-score of **0.7712**, with single-sample inference latency under **20 milliseconds**.

The platform is operationalized through a decoupled, asynchronous **FastAPI** ASGI backend with SQLite persistence via SQLAlchemy 2.0 ORM, and a responsive **React 18** SaaS dashboard powered by Vite, Tailwind CSS, and Recharts. The system supports single-text real-time inference, high-throughput batch CSV processing with annotated export, paginated search and audit trails, and interactive model telemetry. The entire platform was validated through 39 automated integration and security test cases with 100% pass rates.

---

# 3. Introduction

The proliferation of online digital communication channels—social networks, review portals, support ticket desks, and customer forums—has created immense volumes of unstructured textual data. Organizations, academic institutions, and enterprises can no longer rely on manual reading to gauge public perception, service satisfaction, or user sentiment. 

Traditional rule-based or dictionary-based approaches (such as keyword matching or lexicon lookups like VADER) suffer severe degradation when confronted with linguistic nuances, contextual sarcasm, domain-specific vernacular, and syntactic negations. Machine Learning (ML) approaches formulate sentiment classification as a supervised statistical pattern recognition problem, mapping high-dimensional numerical feature representations of text to probability distributions over discrete semantic classes.

**SENTINEL AI** was designed and implemented for the **CSE276 – Artificial Intelligence Foundations** course to demonstrate the practical application of foundational AI concepts:
1. Converting human language into numerical vector spaces using information-theoretic metrics (TF-IDF).
2. Optimizing convex loss functions using numerical optimization solvers (`lbfgs`).
3. Managing class imbalances via balanced cost-sensitive learning.
4. Structuring production software architectures capable of serving low-latency machine learning inferences to modern web clients.

---

# 4. Problem Statement

Automated sentiment analysis over unstructured customer feedback presents four critical technical challenges:

1. **Syntactic Negation Inversion**: Standard stopword removal blindly eliminates words like `"not"`, `"no"`, and `"never"`, inverting negative statements (`"The flight was not good"`) into false positive representations (`"flight good"`).
2. **Three-Class Ambiguity & Imbalance**: Real-world sentiment corpora exhibit heavy skew (negative remarks frequently outnumber positive remarks by 3:1) while neutral remarks present high lexical overlap with both positive and negative classes.
3. **Inference Latency vs. Throughput**: Production platforms must provide immediate real-time feedback for single queries (<50 ms) while efficiently processing batch datasets containing thousands of customer remarks without server memory exhaustion.
4. **Architectural Decoupling**: Academic projects often merge ML scripts into user interface code, creating brittle monoliths. A production-grade system requires clean separation between presentation, API routing, persistence, and statistical inference.

---

# 5. Project Objectives

The core objectives of the SENTINEL AI project are:

1. **Genuine ML/NLP Implementation**: Construct an actual supervised learning pipeline using Scikit-learn, NLTK, NumPy, and Pandas without relying on simulated outputs, hardcoded dictionaries, or external closed-source AI APIs.
2. **Robust Negation Preservation**: Implement a deterministic preprocessing engine that protects negations and binds them to adjacent predicates.
3. **Three-Class Polarity Classification**: Accurately predict and output probabilities across Positive, Negative, and Neutral classes.
4. **High Holdout Generalizability**: Achieve a macro F1-score exceeding 0.70 and accuracy exceeding 75% on an unseen holdout test split.
5. **Decoupled SaaS Architecture**: Build an asynchronous FastAPI backend paired with an enterprise React 18 SPA.
6. **Dual Execution Modes**: Support real-time single-text analysis and bulk CSV batch processing with downloadable annotated datasets.
7. **Complete Auditability & Telemetry**: Maintain persistent SQLite audit logs and real-time dashboard analytics tracking temporal volume and distribution.
8. **Security & Production Hardening**: Defend against path traversal, CORS vulnerabilities, SQL injection, and payload overflow with 100% automated test verification.

---

# 6. Scope of the Project

### In-Scope Capabilities
- Ingestion of English customer feedback, reviews, and tweets up to 5,000 characters per single query.
- Automated NLP preprocessing: regex cleaning, mention removal, URL stripping, tokenization, lemmatization, and negation prefixing.
- Feature extraction using unigrams and bigrams via Sublinear TF-IDF vectorization (7,235 vocabulary features).
- In-memory inference using serialized champion Multinomial Logistic Regression model ($C=0.5$).
- Bulk CSV file processing with automated column detection, progress tracking, and annotated CSV generation.
- Relational persistence in SQLite with SQLAlchemy 2.0 ORM for single analyses, batch job metadata, and evaluation metrics.
- Interactive React 18 web application featuring 6 modular pages (Dashboard, Live Analyzer, Batch Analyzer, History, Model Performance, Settings).
- Full REST API with OpenAPI 3.1.0 specifications and Swagger UI documentation.

### Out-of-Scope (Academic Context)
- User authentication and role-based access control (omitted to enable frictionless evaluation by course instructors).
- Multi-lingual text analysis (pipeline optimized for English corpora).
- GPU-accelerated deep transformer training (focused on foundational statistical machine learning algorithms).

---

# 7. Existing System vs. Proposed System

| Technical Dimension | Existing Conventional Systems | Proposed SENTINEL AI Platform |
| :--- | :--- | :--- |
| **Methodology** | Keyword matching, static regex lookups, or rule-based lexicons (VADER) | Statistical NLP with Supervised Machine Learning (Multinomial Logistic Regression with Softmax) |
| **Negation Handling** | Stripped by standard stopword filters, reversing polarities | Dedicated regex negation binding (`"not good"` &rarr; `"not_good"`) |
| **Feature Representation** | Raw word counts (Bag-of-Words) | Sublinear TF-IDF with Unigram + Bigram feature extraction |
| **Class Imbalance** | Ignored, causing models to collapse onto majority negative class | Cost-sensitive balanced loss weighting inverse to class frequencies |
| **System Architecture** | Monolithic Jupyter Notebooks or CLI scripts | Decoupled client-server architecture (React 18 + FastAPI + SQLite) |
| **Batch Capabilities** | Manual loop execution in terminal | High-throughput streaming CSV processor with annotated file generation |
| **Persistence & Audit** | Ephemeral outputs in console memory | Relational SQLite database with pagination, search, and audit history |
| **Verification & Testing** | Manual spot-checking | 39 automated integration, E2E, and security test cases (100% pass rate) |

---

# 8. Technology Stack

### A. Frontend Layer
* **React.js (v18.3.1)**: Modern component architecture, virtual DOM reconciliation, declarative hooks (`useState`, `useEffect`, `useCallback`).
* **Vite (v5.4.21)**: High-performance ESM build tool with Hot Module Replacement (HMR) and optimized Rollup code-splitting.
* **Tailwind CSS (v3.4.1)**: Utility-first design system with dark navy (`#0b0f19`) and charcoal panel styling.
* **Recharts (v2.12.7)**: Declarative SVG charting library for temporal area charts, donut distribution diagrams, and bar charts.
* **Lucide React (v0.379.0)**: Consistent SVG iconography.
* **Axios (v1.7.2)**: Centralized REST HTTP client with response interceptors and error extraction.
* **React Router DOM (v6.23.1)**: Client-side routing across 6 application views.

### B. Backend & API Gateway
* **Python (v3.14.6 / 3.10+)**: Core runtime language.
* **FastAPI (v0.115.0)**: Asynchronous web framework with automatic OpenAPI documentation and high throughput.
* **Uvicorn (v0.30.6)**: Lightning-fast ASGI web server implementation.
* **Pydantic (v2.9.2)**: Schema definition, request data validation, and response serialization.
* **Python-Multipart (v0.0.12)**: Streaming multipart form-data parser for CSV file uploads.
* **Python-Dotenv (v1.0.1)**: Environment configuration management.

### C. Machine Learning & Natural Language Processing
* **Scikit-learn (v1.5.2)**: TF-IDF vectorization, Logistic Regression, Naive Bayes, Stratified K-Fold CV, and classification metrics.
* **NLTK (v3.9.1)**: Natural Language Toolkit for Punkt tokenization, WordNet lemmatization, and stopword filtering.
* **Pandas (v2.2.3)**: Tabular data processing and batch CSV manipulation.
* **NumPy (v2.1.2)**: High-speed matrix operations and probability array calculations.
* **Joblib (v1.4.2)**: High-efficiency pipeline serialization and disk deserialization.

### D. Relational Persistence & Testing
* **SQLite 3**: Embedded ACID-compliant relational database.
* **SQLAlchemy (v2.0.35)**: Object-Relational Mapping (ORM) and connection pooling.
* **Pytest (v8.3.3)** & **HTTPX (v0.27.2)**: Automated testing harness for API, integration, and security verification.

---

# 9. System Architecture

SENTINEL AI follows an enterprise 4-tier decoupled architecture:

```
┌────────────────────────────────────────────────────────┐
│                   React 18 SPA Client                  │
│    Dashboard • Live Analyzer • Batch • History • Model  │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON REST APIs
                            ▼
┌────────────────────────────────────────────────────────┐
│               FastAPI ASGI Backend Gateway             │
│    CORS Middleware • Exception Handlers • Pydantic     │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐┌───────────────────────────┐
│     Sentiment Service     ││   SQLAlchemy 2.0 ORM      │
│  Orchestration & Latency  ││  SQLite: sentinel.db     │
└─────────────┬─────────────┘└───────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────┐
│               Machine Learning Runtime                 │
│  NLP Preprocessor ──► TF-IDF Vectorizer ──► Logistic   │
│  (Negation Binder)    (7,235 N-Grams)       Regression │
└────────────────────────────────────────────────────────┘
```

### Architectural Principles:
1. **Single Responsibility**: Frontend manages client state and visual presentation; backend handles validation, business logic, and database persistence; ML engine executes deterministic statistical inference.
2. **In-Memory Model Serving**: Model artifacts (`model.joblib`, `vectorizer.joblib`) are deserialized once during FastAPI lifespan startup, preventing per-request disk I/O overhead.
3. **Stateless REST Communication**: All API endpoints communicate through standardized JSON payloads and status codes.

---

# 10. Dataset Description

The machine learning pipeline was developed using the verified **CrowdFlower Twitter US Airline Sentiment Dataset**:
- **Source**: Publicly published CrowdFlower / Kaggle benchmark corpus.
- **Total Valid Records**: **14,405 records** (after deduplication and null elimination).
- **Sentiment Polarity Classes**:
  1. **Negative**: 9,077 samples (63.01%)
  2. **Neutral**: 3,047 samples (21.15%)
  3. **Positive**: 2,281 samples (15.84%)
- **Data Split**: Stratified 80/20 train/test split preserving exact class ratios:
  - **Training Set (`train_sentiment.csv`)**: **11,524 records**
  - **Testing Set (`test_sentiment.csv`)**: **2,881 records**

---

# 11. Data Preprocessing & NLP Pipeline

Raw natural language text is subjected to a 6-stage deterministic preprocessing pipeline implemented in [text_cleaner.py](file:///C:/Users/hp/.gemini/antigravity/scratch/SENTINEL-AI/backend/ml/preprocessing/text_cleaner.py) and [tokenizer.py](file:///C:/Users/hp/.gemini/antigravity/scratch/SENTINEL-AI/backend/ml/preprocessing/tokenizer.py):

```
Raw Text
   │
   ▼
[Step 1] Lowercase Conversion & HTML Entity Decoding
   │
   ▼
[Step 2] Regex Removal of User Handles (@user), URLs (http/https), and Special Characters
   │
   ▼
[Step 3] Negation Detection & Binding ("not happy" ──► "not_happy")
   │
   ▼
[Step 4] NLTK Tokenization into Discrete Word Tokens
   │
   ▼
[Step 5] Filter Stopwords (Preserving Negation Tokens: not, no, never, nor, n't)
   │
   ▼
[Step 6] WordNet Lemmatization (Base Morphological Forms)
   │
   ▼
Cleaned Token Sequence
```

### Negation Handling Example:
- **Input Text**: `"The flight was not comfortable and service was never good."`
- **Standard Preprocessing (Flawed)**: `"flight comfortable service good"` &rarr; **Misclassified as Positive**
- **SENTINEL AI Preprocessing**: `"flight not_comfortable service never_good"` &rarr; **Correctly Classified as Negative**

---

# 12. Machine Learning Methodology

### Feature Extraction: Sublinear TF-IDF Vectorization
Text sequences are converted into real-valued feature vectors using **Term Frequency-Inverse Document Frequency**:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where sublinear term frequency scaling is applied:

$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for } \text{count} > 0$$

And inverse document frequency scales inversely with corpus prevalence:

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

- **N-Gram Range**: $(1, 2)$ — extracts both single words and two-word phrases (e.g., `"customer service"`, `"delayed flight"`).
- **Document Frequency Cutoffs**: `min_df=3` (removes typos), `max_df=0.95` (removes corpus-wide boilerplate).
- **Resulting Vocabulary**: **7,235 n-gram features**.

### Classification Algorithms Evaluated:
1. **Multinomial Naive Bayes (MNB)**: Generative probabilistic classifier applying Bayes' Theorem with Laplace smoothing ($\alpha=0.5, 1.0$).
2. **Multinomial Logistic Regression (MLR)**: Discriminative classifier fitting a linear decision boundary transformed by the Softmax function:

$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

Cost-sensitive learning balances the loss function inversely proportional to class frequencies:

$$w_k = \frac{N}{K \cdot N_k}$$

---

# 13. Model Training, Evaluation & Benchmarks

Hyperparameters were tuned across 5 Stratified Cross-Validation folds:

| Model Architecture | Hyperparameters | Holdout Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | $\alpha = 0.5$ | 73.12% | 0.6920 | 0.6480 | 0.6672 | 0.7285 |
| **Multinomial Naive Bayes** | $\alpha = 1.0$ | 71.85% | 0.6954 | 0.6120 | 0.6421 | 0.7098 |
| **Logistic Regression** | $C = 0.1$, balanced | 74.20% | 0.6840 | 0.7210 | 0.6980 | 0.7510 |
| **Logistic Regression (CHAMPION)** | **$C = 0.5$, balanced, lbfgs** | **76.43%** | **0.7017** | **0.7320** | **0.7138** | **0.7712** |
| **Logistic Regression** | $C = 1.0$, balanced | 76.10% | 0.7005 | 0.7250 | 0.7102 | 0.7680 |

### Holdout Confusion Matrix (2,881 Unseen Samples):
```
                 Predicted Negative   Predicted Neutral   Predicted Positive    Total Actual
Actual Negative         1,467                257                  91               1,815
Actual Neutral           132                 404                  73                 609
Actual Positive           43                  83                 331                 457
Total Predicted         1,642                744                 495               2,881
```

### Class-Wise Evaluation Metrics:
* **Negative**: Precision: **89.34%** | Recall: **80.83%** | F1-Score: **0.8487** | Support: 1,815
* **Neutral**: Precision: **54.30%** | Recall: **66.34%** | F1-Score: **0.5972** | Support: 609
* **Positive**: Precision: **66.87%** | Recall: **72.43%** | F1-Score: **0.6954** | Support: 457

---

# 14. Backend Implementation

The backend is built as an asynchronous service in [backend/main.py](file:///C:/Users/hp/.gemini/antigravity/scratch/SENTINEL-AI/backend/main.py):
- **Application Lifespan**: Executes `init_db()` to create SQLite tables and warms up `SentimentService` so the ML model is pre-loaded before accepting network traffic.
- **Service Layer ([sentiment_service.py](file:///C:/Users/hp/.gemini/antigravity/scratch/SENTINEL-AI/backend/app/services/sentiment_service.py))**: Coordinates ML predictions, records inference duration in milliseconds, and persists transactions with `db.rollback()` safety.
- **Batch CSV Streamer**: Streams uploaded files, detects target text columns, executes inference row-by-row, computes summary distribution percentages, writes annotated CSV files to `uploads/`, and creates `BatchJob` tracking records.
- **Error Handlers**: Traps `RequestValidationError`, `StarletteHTTPException`, and generic `Exception` to return uniform JSON contracts with standardized `error_code` fields.

---

# 15. Frontend Implementation

The frontend is a single-page application built with React 18, Vite, and Tailwind CSS:
- **Design Tokens**: Tailored dark theme featuring `#0b0f19` background, `#111827` panels, and `#1e293b` cards with purple (`#8b5cf6`) and electric blue (`#3b82f6`) brand gradients.
- **Component Modularity**:
  - `ResultCard.jsx`: Displays polarity badge, confidence likelihood, processing latency, expandable preprocessed tokens, and animated Softmax probability bars.
  - `MetricsCard.jsx`: Telemetry cards displaying lifetime counts, rates, and trend icons.
  - `TrendsChart.jsx` & `SentimentPieChart.jsx`: Recharts visualizations mapped directly to `/analytics/trends` and `/analytics/distribution`.
- **Theme Switcher**: Supported via `ThemeContext.jsx` with persistent dark/light mode state saved in `localStorage`.
- **Code Splitting**: Rollup chunking separates vendor dependencies into `vendor-react` (163.9 kB), `vendor-charts` (424.7 kB), and `vendor-ui` (17.3 kB), ensuring sub-second cold starts.

---

# 16. Database Design

Embedded persistence is provided by SQLite with SQLAlchemy 2.0 ORM schemas defined in [domain.py](file:///C:/Users/hp/.gemini/antigravity/scratch/SENTINEL-AI/backend/app/models/domain.py):

### 1. `sentiment_analyses` Table
- `id` (INTEGER, Primary Key, Autoincrement)
- `input_text` (TEXT, Not Null) — Raw input text (indexed for fast search)
- `cleaned_text` (TEXT, Nullable) — Tokens produced by NLP cleaner
- `predicted_sentiment` (VARCHAR(20), Not Null, Indexed) — `positive`, `negative`, `neutral`
- `confidence` (FLOAT, Not Null) — Maximum estimated Softmax probability
- `positive_probability` (FLOAT, Not Null)
- `negative_probability` (FLOAT, Not Null)
- `neutral_probability` (FLOAT, Not Null)
- `processing_time_ms` (FLOAT, Not Null) — Wall-clock model execution time
- `created_at` (DATETIME, Default UTC Now, Indexed)

### 2. `batch_jobs` Table
- `id` (VARCHAR(50), Primary Key) — Unique job identifier (e.g. `job_da05af5f874e`)
- `filename` (VARCHAR(255), Not Null) — Uploaded file name
- `total_records` (INTEGER, Not Null)
- `positive_count` (INTEGER, Not Null)
- `negative_count` (INTEGER, Not Null)
- `neutral_count` (INTEGER, Not Null)
- `status` (VARCHAR(50), Default "completed")
- `annotated_filename` (VARCHAR(255), Not Null) — On-disk path to annotated CSV
- `created_at` (DATETIME, Default UTC Now)

### 3. `model_metrics` Table
- `id` (INTEGER, Primary Key)
- `model_name` (VARCHAR(100), Not Null)
- `accuracy` (FLOAT, Not Null)
- `precision` (FLOAT, Not Null)
- `recall` (FLOAT, Not Null)
- `macro_f1` (FLOAT, Not Null)
- `weighted_f1` (FLOAT, Not Null)
- `recorded_at` (DATETIME, Default UTC Now)

---

# 17. API Documentation

All endpoints are versioned under `/api/v1` and accessible via Swagger UI at `/docs`:

| Method | Endpoint | Description | Request Body / Params | Response Contract |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Diagnostics & dependency health | None | `HealthResponse` (status, DB, model status) |
| `POST` | `/api/v1/sentiment/predict` | Single text classification | `{"text": "..."}` | `SentimentResponse` (sentiment, probs, latency) |
| `POST` | `/api/v1/sentiment/batch-csv` | Bulk CSV analysis | Multipart `file`, optional `text_column` | `BatchAnalysisResponse` (summary, preview, download URL) |
| `GET` | `/api/v1/sentiment/batch-csv/download/{id}` | Export classified CSV | Path `id` | `FileResponse` (`text/csv`) |
| `GET` | `/api/v1/analytics/summary` | Global counts & percentages | None | `AnalyticsSummaryResponse` |
| `GET` | `/api/v1/analytics/trends` | Historical sentiment volume | None | `TrendsResponse` (timeline array) |
| `GET` | `/api/v1/analytics/distribution` | Category distribution | None | `DistributionResponse` (items with theme colors) |
| `GET` | `/api/v1/sentiment/history` | Paginated audit log | `page`, `page_size`, `search`, `sentiment` | `PaginatedHistoryResponse` |
| `DELETE` | `/api/v1/sentiment/history/{id}` | Delete history record | Path `id` | `DeleteResponse` |
| `GET` | `/api/v1/model/info` | Champion model metadata | None | `ModelInfoResponse` (vocab size, hyperparameters) |
| `GET` | `/api/v1/model/metrics` | Holdout evaluation metrics | None | `ModelMetricsResponse` (accuracy, confusion matrix) |

---

# 18. Testing & Validation

The platform features an automated test harness comprising 39 distinct test cases across three test modules:

1. **`test_backend.py` (14 Tests)**: Verifies health check, positive/negative/neutral predictions, payload validation, invalid CSV rejections, analytics aggregations, pagination, deletion, and DB transactions.
2. **`test_phase6_e2e.py` (12 Tests)**: End-to-end testing of exact test sentences, live SQLite persistence verification, keyword searches, batch upload & download parsing, model metadata matching, and 50-cycle latency benchmarking.
3. **`test_phase7_security_reliability.py` (13 Tests)**: Verifies path traversal prevention, CORS lockdown, boundary text limits (5,000 vs 5,001 chars), empty/whitespace inputs, CSV file security, upload limit enforcement (HTTP 413), and SQL injection resilience.

**Test Results**: **39 passed out of 39 tests (100% success rate)** in 15.33 seconds.

---

# 19. Results & Discussion

### Verification of Prescribed Test Sentences:
1. **Positive Input**: *"I absolutely love this product. The quality and customer service are excellent."*
   - Predicted Class: **Positive** | Confidence: **89.11%** (Pos: 0.8911, Neg: 0.0532, Neu: 0.0557) | Latency: **15.2 ms**
2. **Negative Input**: *"This was a terrible experience. The service was extremely disappointing."*
   - Predicted Class: **Negative** | Confidence: **80.25%** (Neg: 0.8025, Neu: 0.1164, Pos: 0.0811) | Latency: **14.8 ms**
3. **Neutral Input**: *"The package was delivered yesterday at 10 AM."*
   - Predicted Class: **Neutral** | Confidence: **44.47%** (Neu: 0.4447, Neg: 0.2911, Pos: 0.2642) | Latency: **14.6 ms**

### Key Findings:
- The combination of unigrams and bigrams allowed the model to capture key phrases (`"customer service"`, `"terrible experience"`) that single-word approaches miss.
- Balanced class weighting proved essential: without it, naive models collapsed toward negative predictions due to the 63% negative class prevalence.
- Sublinear term frequency scaling ($1 + \log(\text{TF})$) successfully dampened the influence of highly repetitive words in long reviews.

---

# 20. Limitations

1. **Language Scope**: Preprocessing, stopwords, and lemmatization are customized specifically for English text.
2. **Contextual Polysemy**: As a bag-of-ngrams model, TF-IDF cannot capture deep syntactic dependencies or long-range coreference resolution that modern transformer models (e.g., BERT, RoBERTa) encode.
3. **Single-Node Persistence**: SQLite is embedded and optimized for single-process concurrency. Multi-instance scaling requires migration to PostgreSQL.
4. **Estimated Probabilities**: Softmax probabilities reflect class likelihoods under the logistic regression loss function; they are calibrated estimates rather than empirical Bayesian posteriors.

---

# 21. Future Scope

1. **Transformer Ensembling**: Integrate fine-tuned lightweight transformer backbones (DistilBERT or RoBERTa) via ONNX Runtime to compare classical linear models with attention-based embeddings.
2. **Aspect-Based Sentiment Analysis (ABSA)**: Deconstruct reviews into specific aspect polarities (e.g., separate sentiment for *Flight Attendants*, *Punctuality*, and *Baggage Handling*).
3. **Enterprise PostgreSQL & Redis**: Migrate database storage to managed PostgreSQL and integrate Redis for rate-limiting and asynchronous task queues (Celery/RQ).
4. **Active Learning Feedback Loop**: Implement a user feedback widget allowing domain experts to flag misclassifications for continuous retraining.

---

# 22. Conclusion

**SENTINEL AI** successfully demonstrates the complete lifecycle of an applied Artificial Intelligence engineering project—from raw dataset cleaning and linguistic negation preservation, through statistical feature extraction and cost-sensitive hyperparameter optimization, to production ASGI REST serving and interactive SaaS dashboard design. 

The champion Multinomial Logistic Regression model delivers reliable, real-time sentiment classification with **76.43% accuracy**, **0.7138 macro F1**, and sub-20 ms inference latency on unseen data. The decoupled architecture, thorough error handling, and 100% test pass rate satisfy all rigorous criteria established for the **CSE276 – Artificial Intelligence Foundations** academic submission.

---

# 23. References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
2. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft). Pearson.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
5. Tiangolo, S. (2018). *FastAPI: Modern, High-Performance Web Framework for Python*. https://fastapi.tiangolo.com
6. CrowdFlower. (2015). *Twitter US Airline Sentiment Dataset*. Published on Kaggle Datasets.
