# SENTINEL AI – Phase 6 Integration & End-to-End Testing Report

**Academic Course**: CSE276 – Artificial Intelligence Foundations  
**Project**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Stage**: Phase 6 – Full-Stack Integration & End-to-End Verification  
**Evaluation Date**: November 17, 2026  
**Status**: **ALL TESTS PASSED (100% Pass Rate)**

---

## 1. Backend Integration Results

The FastAPI ASGI backend (`backend/main.py`) integrates with all system components:
- **FastAPI Lifespan**: Successfully executes database initialization, pre-loads the champion Multinomial Logistic Regression model and TF-IDF vectorizer into memory, and verifies readiness before accepting HTTP traffic.
- **Health Diagnostic (`GET /api/v1/health`)**: Verified returning HTTP 200 with `status: "healthy"`, `database: "connected"`, `model_loaded: true`, `vectorizer_loaded: true`, and active model name `Multinomial Logistic Regression (L2, Balanced)`.
- **API Documentation (`GET /docs`, `GET /openapi.json`)**: Verified rendering OpenAPI 3.1.0 specifications and Swagger UI with all versioned `/api/v1` endpoints registered.
- **CORS Architecture**: Standardized in `backend/app/core/config.py` with explicit allowed origins (`http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:4173`, `http://127.0.0.1:4173`, `http://localhost:3000`, `http://127.0.0.1:3000`) and regex pattern `^https?://(localhost|127\.0\.0\.1)(:\d+)?$` with `allow_credentials=True`. Disallowed wildcard `"*"` with credentials to eliminate browser CORS rejection vulnerabilities.

---

## 2. Frontend Integration Results

The React 18 / Vite / Tailwind SPA frontend (`frontend/`) was audited and built:
- **Compilation & Bundling**: Executed `npm run build` using Vite 5.4.21. Transformed 2,177 modules in 12.06 seconds with **0 errors and 0 warnings**.
- **Bundle Chunk Optimization**:
  - `dist/assets/index-Biain0MF.js`: 126.22 kB (39.12 kB gzip)
  - `dist/assets/index-CNTweHkW.css`: 34.64 kB (6.32 kB gzip)
  - `dist/assets/vendor-react-Dko1Q2pP.js`: 163.90 kB (53.46 kB gzip)
  - `dist/assets/vendor-charts-Z6shZHfH.js`: 424.71 kB (114.12 kB gzip)
  - `dist/assets/vendor-ui-DeCfQgid.js`: 17.27 kB (3.66 kB gzip)
- **All 6 Route Pages Verified**:
  1. `Dashboard.jsx` (`/`): Executive metrics, temporal trends area chart, polarity distribution donut, and comparative volume breakdown.
  2. `LiveAnalyzer.jsx` (`/analyzer`): Interactive single text classification with live tokenization display, confidence gauge, and latency pill.
  3. `BatchAnalyzer.jsx` (`/batch`): Drag-and-drop CSV dataset ingestion, validation, real-time progress bar, distribution breakdown, and annotated CSV export.
  4. `History.jsx` (`/history`): Full audit log table with keyword search, sentiment filter, pagination, and deletion modal.
  5. `ModelPerformance.jsx` (`/model`): Holdout validation metrics, hyperparameter inventory, class metrics breakdown, and 3x3 confusion matrix heatmap.
  6. `Settings.jsx` (`/settings`): Dark/Light theme switcher, backend connectivity diagnostic ping, and academic architecture overview.

---

## 3. End-to-End Prediction Results

Tested using the exact required academic test sentences:

| Sentence ID & Text | Ground Truth Polarity | Model Predicted Polarity | Confidence | Softmax Probabilities | Processing Latency | SQLite DB Persistence |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **P-1**: *"I absolutely love this product. The quality and customer service are excellent."* | Positive | **positive** | **89.11%** | Pos: 0.8911<br>Neg: 0.0532<br>Neu: 0.0557 | 15.2 ms | Verified (ID #54) |
| **N-1**: *"This was a terrible experience. The service was extremely disappointing."* | Negative | **negative** | **80.25%** | Pos: 0.0811<br>Neg: 0.8025<br>Neu: 0.1164 | 14.8 ms | Verified (ID #55) |
| **U-1**: *"The package was delivered yesterday at 10 AM."* | Neutral | **neutral** | **44.47%** | Pos: 0.2642<br>Neg: 0.2911<br>Neu: 0.4447 | 14.6 ms | Verified (ID #56) |

**Observations**:
- Every classification matched the expected class.
- The Softmax distribution correctly summed to 1.0.
- All transactions were persisted with UTC timestamps and full probability vectors.

---

## 4. History Testing Results

- **Pagination & Querying**: `GET /api/v1/sentiment/history` successfully retrieved records sorted by `created_at DESC`.
- **Search Verification**: Querying `search=customer+service` filtered and matched sentence `P-1` with 100% precision.
- **Filter Verification**: Filtering by `sentiment=positive` and `sentiment=negative` isolated corresponding records.
- **Record Deletion**: Created temporary test record (ID #57) and issued `DELETE /api/v1/sentiment/history/57`. Verified `200 OK` response with `success: true`. Direct database inspection confirmed the record was removed from the SQLite database. Subsequent `GET /api/v1/sentiment/history/57` returned `404 Not Found`.

---

## 5. Dashboard Analytics Results

- **Telemetry Aggregation**:
  - Total lifetime records dynamically aggregated across single predictions and bulk batch jobs.
  - Class percentages computed accurately without division-by-zero risks on zero analyses.
- **Timeline Trends (`/api/v1/analytics/trends`)**: Grouped by ISO date, returning counts for positive, negative, and neutral items.
- **Distribution (`/api/v1/analytics/distribution`)**: Formatted for Recharts pie rendering with standardized colors (`#10b981` positive, `#ef4444` negative, `#f59e0b` neutral).

---

## 6. CSV Upload & Download Results

Tested batch pipeline using a 5-row CSV dataset (`test_feedback.csv`) with positive, negative, and neutral remarks:
- **Validation**: Handled multipart upload and auto-detected the `feedback_text` column.
- **Processing**: Successfully classified all 5 rows in under 0.15s, generated summary statistics, and created a `BatchJob` record in SQLite (`job_da05af5f874e`).
- **Download**: Successfully downloaded the annotated CSV from `GET /api/v1/sentiment/batch-csv/download/job_da05af5f874e`.
- **Data Integrity**: Parsed the downloaded CSV and confirmed the existence of:
  - `sentiment_predicted`
  - `confidence_score`
  - `prob_positive`, `prob_negative`, `prob_neutral`
  Every row aligned with ground-truth sentiment labels.

---

## 7. Model Performance Verification

Verified `/api/v1/model/info` and `/api/v1/model/metrics` against the serialized `metadata.json`:
- **Model Name**: Multinomial Logistic Regression (L2, Balanced)
- **Holdout Test Set Size**: 2,881 samples
- **Training Dataset Size**: 11,524 samples
- **Holdout Accuracy**: **76.43%**
- **Macro F1-Score**: **0.7138**
- **Weighted F1-Score**: **0.7712**
- **Confusion Matrix (3x3)**:
  - Row 0 (Actual Negative): `[1467, 257, 91]` (Support: 1,815)
  - Row 1 (Actual Neutral): `[132, 404, 73]` (Support: 609)
  - Row 2 (Actual Positive): `[43, 83, 331]` (Support: 457)

---

## 8. Responsive Testing Results

Audited component styles and breakpoints against standard display viewports:
- **Desktop (1920x1080)**: Full 4-column KPI cards, side-by-side charts (2:1 ratio for trends to distribution), and fixed 64px left sidebar navigation.
- **Laptop (1366x768)**: Fluid layout with no horizontal scrollbars; max-width container keeps readable line lengths.
- **Tablet (768x1024)**: KPI cards collapse to 2 columns (`sm:grid-cols-2`), charts stack vertically, sidebar switches to slide-over drawer triggered by header hamburger menu.
- **Mobile (390x844)**: KPI cards collapse to 1 column, navigation drawer features dark backdrop overlay with touch dismissal, textareas and tables wrapped in `overflow-x-auto` to prevent horizontal viewport clipping.

---

## 9. Error Handling Results

Tested edge cases and exception boundaries:
1. **Empty Text (`""`)**: Rejected with HTTP 422 and structured message: `"Input text cannot be empty or purely whitespace."`
2. **Whitespace-Only Text (`"   \n\t "`): Rejected with HTTP 422.
3. **Oversized Text (> 5,000 characters)**: Rejected with HTTP 422 with validation error citing max length constraint.
4. **Empty CSV Upload**: Rejected with HTTP 400 and structured error detail: `"The uploaded CSV file is empty and contains no data rows."`
5. **Non-Existent History ID**: `DELETE /api/v1/sentiment/history/99999999` returned HTTP 404 with structured error message.
6. **No Stack Trace Leaks**: Verified that all 4xx/5xx responses return structured JSON with an `error_code` and user-friendly `detail` string, preventing internal traceback or file path leakage.

---

## 10. Performance Measurements

Conducted latency benchmarking over 50 consecutive inference cycles:
- **Minimum Latency**: 14.68 ms
- **Mean Latency**: 16.82 ms
- **Median Latency (P50)**: 16.20 ms
- **95th Percentile (P95)**: 18.50 ms
- **99th Percentile (P99)**: 24.15 ms
- **Maximum Latency**: 24.15 ms
- **Batch Processing Throughput**: ~40 samples per second for un-batched synchronous CPU inference.

---

## 11. Bugs Discovered & Fixed

### Bug 1: CORS Wildcard with Credentials Conflict
- **Discovery**: `settings.CORS_ORIGINS` contained `"*"` while `allow_credentials=True` was configured in `CORSMiddleware`.
- **Root Cause**: Modern browsers reject cross-origin requests when `Access-Control-Allow-Origin: *` is paired with credentials.
- **Fix**: Removed `"*"` from `CORS_ORIGINS`, added explicit local ports (`5173`, `4173`, `3000`), and added `allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"`.

### Bug 2: Missing Class Metrics Field Mapping (`f1_score` vs `f1-score`)
- **Discovery**: In `test_10_model_performance_endpoints`, the schema expected `f1-score` while `metadata.json` contained `f1_score`.
- **Root Cause**: Scikit-learn's `classification_report` output format variations across versions.
- **Fix**: Updated `sentiment_service.py` to enrich `class_metrics` with both `f1_score` and `f1-score` aliases, and automatically compute class `support` counts from confusion matrix row sums. Updated `ModelPerformance.jsx` with safe nullish coalescing: `(vals.f1_score ?? vals['f1-score'] ?? 0).toFixed(4)`.

---

## 12. Automated Test Suite Summary

- **Backend Integration Suite (`test_backend.py`)**: 14/14 tests passed (100%)
- **Phase 6 Full-Stack E2E Suite (`test_phase6_e2e.py`)**: 12/12 tests passed (100%)
- **Total Automated Test Cases**: **26/26 passed (100%)**

---

## 13. Remaining Issues

- **None**: All 13 audit areas and test suites pass with zero defects, zero compilation errors, and complete database persistence integrity.
