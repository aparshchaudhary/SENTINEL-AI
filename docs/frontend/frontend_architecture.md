# SENTINEL AI – Frontend Architecture & UI Documentation

**Academic Course**: CSE276 – Artificial Intelligence Foundations  
**Project**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Phase**: Phase 5 – Professional React Frontend Development  

---

## 1. Executive Summary

Phase 5 delivers a modern, high-performance, dark-themed SaaS web application built with **React 18**, **Vite 5**, and **Tailwind CSS**. It connects directly to the FastAPI backend (`http://127.0.0.1:8000/api/v1`) and the serialized Multinomial Logistic Regression machine learning model.

### Key Capabilities:
- **Zero Simulation / Zero Hardcoding**: Every visualization, metric card, confusion matrix row, and classification probability is retrieved directly from live REST API contracts or model serialization metadata.
- **Responsive SaaS Aesthetics**: Engineered around a tailored dark palette (`#0b0f19` background, `#111827` surface panels, `#1e293b` cards) with purple and electric blue brand accents, emerald/rose/amber polarity tags, and subtle glassmorphic elevation.
- **Modular Component Tree**: Composable, reusable UI primitives (`Button`, `Card`, `Badge`, `Modal`, `EmptyState`, `Spinner`), sentiment-specific widgets (`SentimentBadge`, `ProbabilityBar`, `ResultCard`), and responsive Recharts visualizers (`SentimentPieChart`, `TrendsChart`, `ComparisonBarChart`).
- **Production Asset Chunking**: Split vendor bundles (`vendor-react`, `vendor-charts`, `vendor-ui`, application code) yielding sub-450 kB maximum bundle sizes and sub-second cold starts.

---

## 2. Directory Hierarchy

```
frontend/
├── dist/                              # Production build artifacts
│   ├── assets/
│   │   ├── index-CXo15M0G.js          # Main application bundle (126.2 kB)
│   │   ├── index-CNTweHkW.css         # Purged Tailwind stylesheet (34.6 kB)
│   │   ├── vendor-charts-Z6shZHfH.js  # Recharts vendor bundle (424.7 kB)
│   │   ├── vendor-react-Dko1Q2pP.js   # React & React-Router bundle (163.9 kB)
│   │   └── vendor-ui-DeCfQgid.js      # Lucide icons bundle (17.3 kB)
│   └── index.html
├── src/
│   ├── components/
│   │   ├── common/                    # Core UI Primitives
│   │   │   ├── Badge.jsx              # Status & category pills
│   │   │   ├── Button.jsx             # Accessible button with spinner & variants
│   │   │   ├── Card.jsx               # SaaS card with header and actions
│   │   │   ├── EmptyState.jsx         # Graceful zero-data illustration
│   │   │   ├── Modal.jsx              # Accessible backdrop modal dialog
│   │   │   └── Spinner.jsx            # Animated SVG loading spinner
│   │   ├── dashboard/                 # Analytics & Data Visualizations
│   │   │   ├── ComparisonBarChart.jsx # Recharts categorical breakdown
│   │   │   ├── MetricsCard.jsx        # Telemetry KPI cards with trends
│   │   │   ├── SentimentPieChart.jsx  # Recharts donut polarity distribution
│   │   │   └── TrendsChart.jsx        # Recharts multi-area historical timeline
│   │   ├── layout/                    # Layout Shell Elements
│   │   │   ├── Navbar.jsx             # Sticky topbar with live health ping & theme toggle
│   │   │   └── Sidebar.jsx            # Responsive navigation drawer & champion model badge
│   │   └── sentiment/                 # Domain-Specific Sentiment Primitives
│   │       ├── ProbabilityBar.jsx     # Softmax class distribution progress bars
│   │       ├── ResultCard.jsx         # Full prediction card with NLP token drawer
│   │       └── SentimentBadge.jsx     # Positive/Negative/Neutral badges with icons
│   ├── context/
│   │   └── ThemeContext.jsx           # Dark/Light theme state persisted to localStorage
│   ├── layouts/
│   │   └── AppLayout.jsx              # Outer application shell with dynamic Outlet
│   ├── pages/                         # Core Route Views
│   │   ├── BatchAnalyzer.jsx          # Drag-and-drop CSV upload, progress, preview & download
│   │   ├── Dashboard.jsx              # Platform executive overview & metrics grid
│   │   ├── History.jsx                # Paginated audit log with search, filters & deletion modal
│   │   ├── LiveAnalyzer.jsx           # Single-text interactive inference with latency display
│   │   ├── ModelPerformance.jsx       # Validation KPIs, class metrics & Confusion Matrix
│   │   └── Settings.jsx               # Backend connectivity diagnostics & academic documentation
│   ├── services/
│   │   └── api.js                     # Centralized Axios REST client with interceptors
│   ├── App.jsx                        # React Router DOM configuration
│   ├── index.css                      # Tailwind base, utilities & animation keyframes
│   └── main.jsx                       # React 18 DOM mount entry point
├── .env                               # Runtime environment variables
├── .env.example                       # Documentation for environment variables
├── index.html                         # SPA HTML entry point with Inter & JetBrains Mono fonts
├── package.json                       # Dependencies & build scripts
├── postcss.config.js                  # Tailwind & Autoprefixer plugin configuration
├── tailwind.config.js                 # Theme extensions, palettes, and typography
└── vite.config.js                     # Vite build configuration with proxy & chunking
```

---

## 3. Route & View Mapping

| Path | Component | View Purpose | Associated Backend Endpoints |
| :--- | :--- | :--- | :--- |
| `/` | `Dashboard.jsx` | Executive KPIs, volume timeline, and polarity distribution | `GET /analytics/summary`<br>`GET /analytics/trends`<br>`GET /analytics/distribution` |
| `/analyzer` | `LiveAnalyzer.jsx` | Real-time text inference, tokenization view, Softmax probabilities | `POST /sentiment/predict` |
| `/batch` | `BatchAnalyzer.jsx` | Bulk CSV dataset processing, distribution preview, and CSV export | `POST /sentiment/batch-csv`<br>`GET /sentiment/batch-csv/download/{job_id}` |
| `/history` | `History.jsx` | Paginated search, filter by polarity, and record deletion | `GET /sentiment/history`<br>`DELETE /sentiment/history/{id}` |
| `/model` | `ModelPerformance.jsx` | Holdout accuracy, macro F1, confusion matrix, hyperparameters | `GET /model/info`<br>`GET /model/metrics` |
| `/settings` | `Settings.jsx` | Theme switcher, live API health diagnostics, system metadata | `GET /health` |

---

## 4. API Client Integration (`services/api.js`)

All client communication is centralized through an Axios instance with automatic response extraction and error unwrapping:

```javascript
// Base configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';

// Interceptor extracts server error messages or connection drop diagnostics
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    let errorMessage = 'An unexpected network error occurred.';
    if (error.response?.data?.detail) {
      errorMessage = typeof error.response.data.detail === 'string' 
        ? error.response.data.detail 
        : JSON.stringify(error.response.data.detail);
    } else if (error.request) {
      errorMessage = 'Backend server is unreachable. Please ensure FastAPI is running on port 8000.';
    }
    return Promise.reject(new Error(errorMessage));
  }
);
```

---

## 5. Build & Performance Benchmarks

Execution of `npm run build` verifies clean compilation and optimal bundling:
- **Build Duration**: ~12.0 seconds
- **Modules Transformed**: 2,177
- **CSS Bundle**: 34.64 kB (6.32 kB gzip)
- **Vendor React**: 163.90 kB (53.46 kB gzip)
- **Vendor Recharts**: 424.71 kB (114.12 kB gzip)
- **Application Logic**: 126.21 kB (39.10 kB gzip)
- **UI Icons**: 17.27 kB (3.66 kB gzip)
- **Total Gzipped Payload**: ~217 kB
