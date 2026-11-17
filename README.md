# SENTINEL AI – Intelligent Sentiment Analysis Platform
**Academic Project for CSE276 – Artificial Intelligence Foundations**  
*A production-grade, full-stack Natural Language Processing (NLP) platform for real-time and batch sentiment intelligence.*

---

## 📌 1. Project Overview

**SENTINEL AI** is an advanced sentiment analysis platform designed to process, analyze, and categorize unstructured text data into three discrete polarity classes:
* 🟢 **Positive**
* 🔴 **Negative**
* 🟡 **Neutral**

Unlike simplistic keyword-matching tools or simulated dummy outputs, **SENTINEL AI** is founded upon genuine Machine Learning and Natural Language Processing (NLP) methodologies. It employs statistical text preprocessing, **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization with unigram and bigram extraction, and supervised classifiers (**Multinomial Naive Bayes** and **Multinomial Logistic Regression**) trained on verified sentiment corpora.

The system is delivered via an enterprise-grade SaaS interface featuring interactive analytics, single-input real-time analysis, batch CSV processing, historical inspection, and model diagnostic benchmarks.

---

## 🎯 2. Academic & Technical Objectives

1. **Demonstrate AI/ML Foundations**:
   * Rigorous implementation of text preprocessing (regex cleaning, lowercasing, stopword elimination, tokenization, lemmatization).
   * Feature engineering via mathematical TF-IDF vectorization with sublinear scaling.
   * Supervised classification algorithms with cross-validation and hyperparameter optimization.
   * Comprehensive model evaluation (Accuracy, Precision, Recall, Macro F1-Score, and Confusion Matrix analysis).

2. **Full-Stack Modular Architecture**:
   * Complete separation of concerns between client presentation, API orchestration, ML inference, and relational persistence.
   * Asynchronous, high-throughput REST API gateway built with FastAPI and Pydantic.
   * Modern, responsive SaaS interface built with React.js, Vite, Tailwind CSS, Lucide Icons, and Recharts.

3. **Enterprise Usability**:
   * Real-time single text classification with calibrated confidence scores and probability breakdowns.
   * Scalable bulk CSV file processing for enterprise customer feedback and social data.
   * Persistent SQLite storage for audit trails, historical exploration, and temporal trend visualization.

---

## 🛠️ 3. Technology Stack

### Frontend
* **React.js (v18)**: Modern component-based declarative user interface.
* **Vite**: Next-generation frontend tooling and rapid HMR build engine.
* **Tailwind CSS**: Utility-first CSS framework for a sleek, dark/light SaaS aesthetic.
* **Lucide React Icons**: Consistent, clean iconography.
* **Recharts**: Composable charting library for sentiment distributions and confidence gauges.
* **Axios**: Standardized HTTP client for REST communication.

### Backend & API
* **Python (3.10+)**: Primary runtime language for ML and server engineering.
* **FastAPI**: Modern, high-performance web framework for building APIs.
* **Uvicorn**: Asynchronous ASGI web server implementation.
* **Pydantic (v2)**: Data validation and schema enforcement using Python type annotations.
* **SQLAlchemy**: Python SQL toolkit and Object Relational Mapper (ORM).

### Machine Learning & Data Science
* **Scikit-learn**: Core ML framework for TF-IDF vectorization, Logistic Regression, Naive Bayes, and evaluation metrics.
* **NLTK (Natural Language Toolkit)**: Linguistic text processing (tokenization, stopwords, lemmatization).
* **Pandas**: Tabular data manipulation and batch CSV ingestion.
* **NumPy**: Numerical vector operations and probability matrix operations.
* **Joblib**: Efficient serialization of trained model pipelines.

### Database & Storage
* **SQLite 3**: Embedded relational database for zero-config persistence of analyses, batch jobs, and evaluation logs.

---

## 📂 4. Project Directory Structure

```
SENTINEL-AI/
│
├── frontend/
│   ├── src/
│   │   ├── assets/               # Brand logos, icons, static illustrations
│   │   ├── components/           # Modular UI components
│   │   │   ├── common/           # Reusable UI primitives (Button, Card, Badge, Modal, Dropzone)
│   │   │   ├── dashboard/        # Data visualization charts (Pie, Bar, Area charts)
│   │   │   └── sentiment/        # Domain components (SentimentCard, PolarityMeter, ProbGauge)
│   │   ├── hooks/                # Custom React state & network hooks
│   │   ├── layouts/              # Main application shell, Sidebar, Navbar
│   │   ├── pages/                # LiveAnalyzer, BatchAnalyzer, Dashboard, History, ModelInfo
│   │   ├── services/             # Axios API client and route services
│   │   ├── App.jsx               # Main React application tree
│   │   └── main.jsx              # DOM entry point
│   ├── package.json              # Frontend dependencies and npm scripts
│   ├── tailwind.config.js        # Design system styles and colors
│   └── vite.config.js            # Vite build configuration
│
├── backend/
│   ├── app/
│   │   ├── api/                  # API routers and versioned endpoints
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/    # Controllers: sentiment, batch, analytics, health
│   │   │   │   └── router.py     # Aggregated v1 API router
│   │   ├── core/                 # Config (Pydantic BaseSettings), logging, constants
│   │   ├── database/             # SQLite connection, SessionLocal, Base model
│   │   ├── models/               # SQLAlchemy ORM database models
│   │   ├── schemas/              # Pydantic schemas for request/response validation
│   │   └── services/             # Business logic and ML orchestration services
│   │
│   ├── ml/
│   │   ├── datasets/             # Training datasets and corpus files
│   │   ├── preprocessing/        # Text cleaners, tokenizers, stopword filters
│   │   ├── training/             # Offline training scripts, grid search, evaluators
│   │   └── saved_models/         # Serialized artifacts (model.joblib, vectorizer.joblib)
│   │
│   ├── requirements.txt          # Backend Python dependencies
│   └── main.py                   # FastAPI initialization & Uvicorn entry point
│
├── docs/                         # Engineering & Academic Documentation
│   ├── architecture/             # System diagrams & component responsibilities
│   ├── api/                      # OpenAPI specifications & endpoint docs
│   └── ml/                       # NLP pipeline details, formulas & evaluation criteria
│
├── .gitignore                    # Version control ignore definitions
└── README.md                     # Master project documentation
```

---

## 🗺️ 5. Planned Development Phases

| Phase | Title | Scope & Objectives | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **System Architecture & Foundation** | Requirements breakdown, architectural blueprints, directory scaffolding, ML pipeline design, API contract specifications, environment & dependency audit. | **COMPLETED** |
| **Phase 2** | **Dataset & NLP Preprocessing** | Public 3-class dataset acquisition (14,405 records), data cleaning engine, negation preservation, NLTK tokenization & WordNet lemmatization, stratified 80/20 splitting. | **COMPLETED** |
| **Phase 3** | **Model Training & Evaluation** | TF-IDF feature extraction (7,235 n-grams), hyperparameter tuning, training Logistic Regression & Multinomial Naive Bayes, evaluation & visualization, model serialization to `.joblib`. Champion: Logistic Regression (Macro F1: 0.7138, Accuracy: 76.43%). | **COMPLETED** |
| **Phase 4** | **Backend REST API & Database** | Production FastAPI application with lifespan model loading, SQLite database with SQLAlchemy 2.0 ORM, Pydantic v2 schemas, single text prediction, streaming batch CSV processing & download, analytics aggregation, paginated history with search/filter, model metadata endpoints, health check. All 14 integration tests passing. | **COMPLETED** |
| **Phase 5** | **Frontend Modern SaaS Interface** | React 18 + Vite setup, Tailwind CSS design system, Live Text Sentiment Analyzer, Batch CSV upload & annotated export, Recharts executive dashboard, paginated audit history with search/filter & deletion modal, model performance inspection view, system settings & theme toggle. Zero warnings, optimized chunking. | **COMPLETED** |
| **Phase 6** | **System Integration, Testing & Polish** | End-to-end integration test suite (26/26 tests passing, 100%), verified live predictions for test sentences, history search/filter/deletion, batch CSV upload/download, performance latency benchmarking (<17ms avg), responsive viewport verification, and comprehensive test documentation. | **COMPLETED** |
| **Phase 7** | **Production Readiness & Hardening** | Security hardening (CORS lockdown, path traversal protection, upload limits, error trace sanitization, DB rollback guards), environment externalization (.env.example templates), 39 automated tests passing (100%), production deployment guides, and readiness assessment. | **COMPLETED** |
| **Phase 8** | **Final Validation & Academic Submission** | Complete end-to-end application verification, live health & inference checks, 39/39 passing pytest suites, zero-error production frontend build, 23-section Academic Project Report, 12-slide presentation outline, 45-question viva preparation guide, and comprehensive demonstration walkthrough. | **COMPLETED** |

---

## 🔬 6. Machine Learning Workflow Summary

```
Raw Text Input
      │
      ▼
Text Preprocessor (Regex Cleaning, Lowercase, URL/Mention Strip, Negation Preservation)
      │
      ▼
NLTK Tokenizer & WordNet Lemmatizer
      │
      ▼
TF-IDF Vectorizer (Unigrams + Bigrams, Sublinear TF, 7,235 Features)
      │
      ▼
Classifiers Benchmark: Multinomial Naive Bayes vs. Multinomial Logistic Regression
      │
      ▼
Evaluation Suite (Accuracy: 76.43%, Macro F1: 0.7138, Weighted F1: 0.7712, 3x3 Confusion Matrix)
      │
      ▼
Model Selection (Champion: Multinomial Logistic Regression, C=0.5, balanced)
      │
      ▼
Serialization (`joblib.dump`) -> In-Memory Runtime FastAPI Serving -> React 18 SaaS UI
```

---

## 📡 7. REST API Endpoints Overview

* `POST /api/v1/sentiment/predict` — Analyze single text string with confidence probabilities.
* `POST /api/v1/sentiment/batch-csv` — Bulk CSV file upload and classification with target column detection.
* `GET /api/v1/sentiment/batch-csv/download/{job_id}` — Export classified CSV dataset with path traversal security.
* `GET /api/v1/analytics/summary` — Global sentiment distribution and metrics.
* `GET /api/v1/analytics/trends` — Temporal sentiment trend data for charts.
* `GET /api/v1/analytics/distribution` — Categorical distribution and percentages.
* `GET /api/v1/sentiment/history` — Paginated analysis history with search & filter.
* `DELETE /api/v1/sentiment/history/{id}` — Remove an entry from history.
* `GET /api/v1/model/info` — Active model metadata and vocabulary size.
* `GET /api/v1/model/metrics` — Accuracy, F1-scores, and confusion matrix.
* `GET /api/v1/health` — System status, DB connection, and model readiness.

Interactive Swagger UI documentation is available at `http://127.0.0.1:8000/docs`.

---

## 🚀 8. Quickstart & Installation Instructions

### Backend Startup
```bash
# 1. Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # On Windows PowerShell
# source .venv/bin/activate    # On Linux/macOS

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Configure environment
cp backend/.env.example backend/.env

# 4. Start FastAPI server
python backend/main.py
```
Backend runs on `http://127.0.0.1:8000`.

### Frontend Startup
```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env

# 3. Run development server
npm run dev

# 4. Or build production assets
npm run build
npm run preview
```
Frontend runs on `http://localhost:5173`.

### Running Automated Test Suites (39 Tests Total)
```bash
# Run all test suites
python -m pytest backend/tests/test_backend.py -v
python -m pytest backend/tests/test_phase6_e2e.py -v
python -m pytest backend/tests/test_phase7_security_reliability.py -v
```

---

## 📚 9. Documentation Index

### Academic Submission & Evaluation
- **[Final Project Report](file:///docs/submission/SENTINEL_AI_Final_Project_Report.md)**: 23-section master academic thesis report formatted for CSE276 university submission.
- **[Demonstration Guide](file:///docs/submission/demo_guide.md)**: Step-by-step evaluator demonstration script, terminal commands, test sentences, and troubleshooting.
- **[Viva Voce Questions & Answers](file:///docs/submission/viva_questions.md)**: 45 in-depth technical questions spanning AI/ML, NLP, Backend, Frontend, and critical defense scenarios.
- **[Presentation Outline](file:///docs/submission/presentation_outline.md)**: 12-slide structured slide deck outline with visual recommendations and complete speaker notes.

### Engineering & System Documentation
- **[System Architecture](file:///docs/architecture/system_architecture.md)**: Architectural blueprints & data flow diagrams.
- **[ML Pipeline & Metrics](file:///docs/ml/ml_pipeline.md)**: NLP algorithms, TF-IDF formulas, and benchmark comparisons.
- **[Backend Architecture](file:///docs/backend/backend_architecture.md)**: Service layer, ORM models, and database contracts.
- **[Frontend Architecture](file:///docs/frontend/frontend_architecture.md)**: UI design tokens, component tree, and chunk splitting.
- **[Deployment Guide](file:///docs/deployment/deployment_guide.md)**: Production setup, Nginx config, and troubleshooting.
- **[Security Checklist](file:///docs/deployment/security_checklist.md)**: Hardening checklist, limitations, and recommendations.
- **[Phase 7 Readiness Report](file:///docs/testing/phase7_readiness_report.md)**: Full audit findings and test execution metrics.
