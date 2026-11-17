# SENTINEL AI – Security Assessment & Production Hardening Checklist

**Academic Course**: CSE276 – Artificial Intelligence Foundations  
**Project**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Stage**: Phase 7 – Security Audit & Production Hardening  

---

## 1. Security Controls Implemented & Verified

### A. Environment & Secrets Management
- [x] **No Secrets in Source Code**: No API keys, database credentials, or secret tokens hardcoded in application repositories.
- [x] **Environment Separation**: Settings externalized using `backend/.env.example` and `frontend/.env.example`.
- [x] **Version Control Exclusions**: `.gitignore` comprehensively configured to exclude local `.env`, virtual environments (`.venv`), database dumps (`*.db`), temporary uploads (`backend/uploads/*.csv`), and build outputs (`frontend/dist/`).

### B. API Gateway & Network Security
- [x] **CORS Origin Lockdown**: Wildcard `"*"` removed from CORS when `allow_credentials=True` is enabled. Strict origin whitelist (`localhost:5173`, `127.0.0.1:5173`, `localhost:4173`, `127.0.0.1:4173`, `localhost:3000`, `127.0.0.1:3000`) and loopback regex enforced.
- [x] **Information Leakage Prevention**: Centralized exception handlers catch unhandled exceptions and return structured JSON (`VALIDATION_ERROR`, `INTERNAL_SERVER_ERROR`) without leaking Python tracebacks, file system paths, or SQL queries.
- [x] **Health Check Sanitization**: Database disconnection errors are logged internally to server logs while the public `/health` endpoint reports sanitized status (`"disconnected"`).

### C. Input Validation & Request Sanitization
- [x] **Single Text Boundaries**: Pydantic v2 schema enforces `min_length=1`, `max_length=5000`, and rejects whitespace-only inputs with HTTP 422.
- [x] **Query Parameter Bounds**: Pagination parameters strictly bounded (`page >= 1`, `1 <= page_size <= 100`).
- [x] **SQL Injection Immunity**: All queries use SQLAlchemy 2.0 parameterized queries and ORM objects; search strings are passed via parameterized `.ilike()` bindings.

### D. File Upload & Download Security
- [x] **File Extension Verification**: Batch upload strictly restricts input files to `.csv` extension.
- [x] **Payload Size Enforcement**: Enforces `MAX_UPLOAD_SIZE_MB` (default: 10 MB) returning HTTP 413 for oversized files.
- [x] **Empty & Corrupt File Rejection**: Rejects 0-byte files, header-only CSVs, and unparseable binary content with HTTP 400.
- [x] **Target Column Strictness**: Rejects requests specifying non-existent `text_column` names rather than silently falling back.
- [x] **Path Traversal Protection**: Download endpoint verifies that resolved file paths strictly reside inside the designated `settings.UPLOADS_DIR` directory using `os.path.abspath` prefix checks.
- [x] **Filename Sanitization**: Output filenames sanitized with regex `re.sub(r"[^a-zA-Z0-9_.-]", "_", ...)` to prevent shell injection or directory traversal.

### E. Database Transaction Safety
- [x] **Transaction Rollbacks**: All database write operations (`add`, `delete`, `commit`) wrapped in `try...except` blocks that issue `db.rollback()` upon failure to prevent corrupted session states.

---

## 2. Identified Limitations (Academic Scope)

| Scope Area | Current Implementation | Production Implication |
| :--- | :--- | :--- |
| **User Authentication & Authorization** | Open REST API (No JWT / OAuth2 / API keys) | Designed for university evaluation and academic grading. Multi-tenant commercial SaaS requires OAuth2/JWT and role-based access control (RBAC). |
| **Rate Limiting** | Unlimited client requests | A denial-of-service (DoS) attack could consume server compute resources. Production requires Redis-backed rate limiting (e.g. `slowapi` or Cloudflare WAF). |
| **Storage Architecture** | Local disk storage for uploads (`backend/uploads/`) | Suitable for single-node deployment. Multi-node cloud deployments require object storage (e.g., AWS S3, Google Cloud Storage). |
| **Database Engine** | Embedded SQLite (`sentinel.db`) | Excellent for zero-config demonstration and moderate workloads. High concurrency writes require PostgreSQL. |

---

## 3. Recommended Production Precautions for Commercial Deployment

1. **Deploy Reverse Proxy with TLS/SSL**: Terminate HTTPS at an Nginx, Traefik, or Cloudflare gateway with TLS 1.3.
2. **Implement API Rate Limiting**: Limit unauthenticated endpoints to 60 requests/minute per IP address.
3. **Database Migration to PostgreSQL**: Point `DATABASE_URL` to a managed PostgreSQL cluster with connection pooling (e.g., PgBouncer).
4. **Automated Upload Cleanup Job**: Schedule a cron or background task to purge temporary annotated CSV files older than 24 hours.
5. **Security Scanning**: Integrate `bandit` for Python AST security analysis and `npm audit` in CI/CD pipelines.
