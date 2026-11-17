# SENTINEL AI – Production Deployment & Operations Guide

**Academic Course**: CSE276 – Artificial Intelligence Foundations  
**Project**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document**: Production Deployment, Environment Setup & Operational Runbook  

---

## 1. System Requirements

### Hardware Prerequisites
- **CPU**: Dual-core x86_64 or ARM64 processor (2.0+ GHz recommended).
- **RAM**: Minimum 2 GB (4 GB recommended for high-volume batch CSV processing).
- **Disk Storage**: 1 GB available disk space for dependencies, dataset corpora, and SQLite database storage.

### Software Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+), macOS (12+), or Windows (10/11 / Server 2019+).
- **Python**: Version 3.10 to 3.14 (Verified on Python 3.14.6).
- **Node.js**: Version 18.0+ or 20.0+ LTS (Verified on Node v24.16.0 with npm 11.13.0).

---

## 2. Environment Architecture & Configuration

SENTINEL AI separates client presentation from backend inference and persistence via standard REST API contracts.

### Environment Variable Contracts

#### Backend (`backend/.env`)
Create a `.env` file inside `backend/` copied from `backend/.env.example`:

```env
# Server Environment Mode: development | production | test
ENVIRONMENT=production

# Server Host and Port Binding
HOST=127.0.0.1
PORT=8000

# API Versioning Prefix
API_V1_STR=/api/v1

# Relational Database Connection (Default: embedded SQLite)
# For enterprise production: postgresql://sentinel_user:secret@localhost:5432/sentinel_db
DATABASE_URL=sqlite:///sentinel.db

# CORS Whitelist (Comma-separated origins, no wildcards in production)
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173

# Security & Upload Constraints
MAX_UPLOAD_SIZE_MB=10
MAX_TEXT_LENGTH=5000
```

#### Frontend (`frontend/.env`)
Create a `.env` file inside `frontend/` copied from `frontend/.env.example`:

```env
# Backend REST API Gateway URL
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

---

## 3. Step-by-Step Installation & Startup

### Step 1: Clone & Navigate
```bash
git clone https://github.com/organization/SENTINEL-AI.git
cd SENTINEL-AI
```

### Step 2: Backend Virtual Environment & Dependencies
```bash
# 1. Create and activate Python virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Step 3: Verify Machine Learning Model Artifacts
Verify that the pre-trained champion artifacts exist in `backend/ml/saved_models/`:
- `model.joblib` (Champion Multinomial Logistic Regression model)
- `vectorizer.joblib` (Fitted TF-IDF Vectorizer with 7,235 n-gram features)
- `metadata.json` (Evaluation benchmarks & confusion matrix)

### Step 4: Run Automated Verification Tests
```bash
# Execute backend unit and integration test suites
python -m pytest backend/tests/test_backend.py -v
python -m pytest backend/tests/test_phase6_e2e.py -v
python -m pytest backend/tests/test_phase7_security_reliability.py -v
```
All 39 tests must pass with 100% success.

### Step 5: Start Backend Service
```bash
# Start FastAPI ASGI server with Uvicorn
python backend/main.py

# Alternatively, for production multi-worker deployment:
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 4
```
Access interactive API documentation at: `http://127.0.0.1:8000/docs`

---

## 4. Frontend Installation & Production Build

### Development Mode
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### Production Static Build
```bash
cd frontend
npm run build
```
Build outputs are generated in `frontend/dist/`:
- `index.html` (SPA entry point)
- `assets/index-*.js` (Application bundle)
- `assets/vendor-react-*.js` (React, Router vendor chunk)
- `assets/vendor-charts-*.js` (Recharts vendor chunk)
- `assets/vendor-ui-*.js` (Lucide React icons chunk)
- `assets/index-*.css` (Tailwind CSS stylesheet)

### Preview Production Build Locally
```bash
npm run preview -- --port 4173
```
Access the production build at: `http://localhost:4173`

---

## 5. Web Server Configuration (Nginx / Caddy SPA Deployment)

When serving `frontend/dist/` behind an Nginx reverse proxy:

```nginx
server {
    listen 80;
    server_name sentinel.example.com;

    # Static Frontend SPA
    root /var/www/sentinel-ai/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API Requests to FastAPI Backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 10M;
    }
}
```

---

## 6. Troubleshooting & Operational Runbook

| Symptom / Log Error | Root Cause | Remediation Step |
| :--- | :--- | :--- |
| `GET /api/v1/health` reports `status: "degraded"` | Missing ML model files or database unreachable | Verify `backend/ml/saved_models/model.joblib` exists and database path in `DATABASE_URL` is writable. |
| Browser console shows CORS error on API fetch | Client origin not included in backend whitelist | Add frontend domain/port to `CORS_ORIGINS` in `backend/.env`. |
| HTTP 413 `File exceeds maximum permissible size` | Uploaded CSV exceeds configured limit | Adjust `MAX_UPLOAD_SIZE_MB` in `backend/.env` or reduce file size. |
| HTTP 422 `Request payload validation failed` | Input text exceeds 5,000 characters or is whitespace | Verify user input passes client-side validation before dispatch. |
| HTTP 404 on page reload in production | Missing SPA fallback configuration on web server | Configure web server rewrite rule (e.g., `try_files $uri $uri/ /index.html;`). |
