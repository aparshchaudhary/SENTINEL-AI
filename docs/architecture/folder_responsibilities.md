# SENTINEL AI — Folder & Component Responsibilities Guide
**Course**: CSE276 – Artificial Intelligence Foundations  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document Version**: 1.0 (Phase 1)

---

## 1. Directory Tree & Component Responsibilities

```
SENTINEL-AI/
├── frontend/                     # React + Vite Client Application
│   ├── src/
│   │   ├── assets/               # Static assets (SVGs, icons, logos, brand imagery)
│   │   ├── components/           # Modular React components
│   │   │   ├── common/           # Generic atomic UI elements (Button, Card, Badge, Modal, Dropzone)
│   │   │   ├── dashboard/        # Visual analytical widgets (DistributionPie, TimelineChart, MetricsCard)
│   │   │   └── sentiment/        # Domain sentiment widgets (TextInputArea, PolarityMeter, ProbabilityGauge)
│   │   ├── hooks/                # Custom React hooks (useSentiment, useBatchAnalysis, useHistory)
│   │   ├── layouts/              # Top-level shell layouts (Sidebar, Header, MainLayout, Footer)
│   │   ├── pages/                # Top-level route pages (LiveAnalyzer, BatchAnalyzer, Dashboard, History, ModelInfo)
│   │   ├── services/             # Axios REST API client & endpoint service definitions
│   │   ├── App.jsx               # Application root component with routing & layout wrappers
│   │   └── main.jsx              # DOM entry point mounting React root
│   ├── index.html                # HTML entry template
│   ├── package.json              # NPM package dependencies and scripts
│   ├── tailwind.config.js        # Tailwind CSS styling tokens, theme colors & font families
│   └── vite.config.js            # Vite build configuration, alias paths & dev proxy setup
│
├── backend/                      # FastAPI Python Web & Service Layer
│   ├── app/
│   │   ├── api/                  # API routing declarations
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/    # Route controllers (sentiment.py, batch.py, analytics.py, health.py)
│   │   │   │   └── router.py     # Central v1 APIRouter aggregating all sub-endpoint modules
│   │   ├── core/                 # Global settings, environment variables (Pydantic BaseSettings), logging
│   │   ├── database/             # SQLite connection engine, SessionLocal generator, Base declarative class
│   │   ├── models/               # SQLAlchemy ORM database models (SentimentRecord, BatchJob, ModelMetric)
│   │   ├── schemas/              # Pydantic schemas for request validation & response formatting
│   │   └── services/             # Core business logic: SentimentService, BatchService, AnalyticsService
│   │
│   ├── ml/                       # Machine Learning Engineering Subsystem
│   │   ├── datasets/             # Raw & processed CSV dataset files for model training
│   │   ├── preprocessing/        # Text cleaners, regex normalizers, tokenizers, stopword filters
│   │   ├── training/             # Offline training scripts, hyperparameter tuning, model evaluators
│   │   └── saved_models/         # Serialized model artifacts (.joblib), vectorizers, and metadata.json
│   │
│   ├── requirements.txt          # Python package requirements
│   └── main.py                   # FastAPI application initialization & Uvicorn startup
│
├── docs/                         # Engineering, Architecture & Course Documentation
│   ├── architecture/             # System blueprints & folder responsibilities
│   ├── api/                      # OpenAPI contracts, endpoint specifications & error formats
│   └── ml/                       # Machine Learning mathematical formulations & evaluation benchmarks
│
├── .gitignore                    # Version control ignore rules (virtualenvs, node_modules, temp DBs)
└── README.md                     # Master project documentation
```

---

## 2. Responsibilities by Major Subsystem

### 2.1 Frontend Subsystem (`frontend/`)
* **`components/common`**: Reusable design system atoms. Must remain completely agnostic of sentiment business logic.
* **`components/sentiment`**: Domain-specific UI displays. Renders input controls, polarity indicators (Positive: emerald, Neutral: amber, Negative: rose), and raw/cleaned text diffs.
* **`components/dashboard`**: Data visualization layer built on Recharts. Renders pie, bar, and area charts for historical metrics.
* **`services/`**: Encapsulates all network traffic. No raw `fetch` or `axios` calls should ever occur directly inside visual components.
* **`hooks/`**: Manages React UI state, optimistic updates, pagination logic, and error boundaries.

### 2.2 Backend Subsystem (`backend/app/`)
* **`api/`**: Thin controller layer. Parses incoming HTTP requests, validates them against Pydantic schemas, delegates business operations to services, and returns formatted HTTP responses.
* **`schemas/`**: Pydantic models acting as strict contracts. Guarantees type safety for requests and responses.
* **`services/`**: The core application logic. Invokes the ML model for inference, formats output, and coordinates database writes.
* **`models/` & `database/`**: Relational persistence layer using SQLAlchemy ORM to manage SQLite tables and connections cleanly.
* **`core/`**: Application configuration management using `pydantic-settings`. Loads `.env` parameters (CORS origins, database URLs, model artifact paths).

### 2.3 Machine Learning Subsystem (`backend/ml/`)
* **`preprocessing/`**: Deterministic text normalization pipeline. Crucially, the exact same preprocessing logic used in training must be executed at runtime for incoming user requests.
* **`training/`**: Standalone training modules. Ingests raw datasets, runs cross-validation, tunes hyperparameters, computes classification metrics, and outputs serialized artifacts.
* **`saved_models/`**: Isolated directory storing fitted `vectorizer.joblib`, `model.joblib`, and `metadata.json`. Never committed to git if exceeding standard file sizes.
