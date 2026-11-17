"""
SENTINEL AI – Intelligent Sentiment Analysis Platform
Academic Project: CSE276 – Artificial Intelligence Foundations

Main Application Entry Point & ASGI Server
"""

import os
import sys
from contextlib import asynccontextmanager

# Enable NLTK downloads through egress proxies (prevents CWE-918 blocking on cloud hosts like Render)
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Ensure backend root is in Python module path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.core.config import settings
from app.database.session import init_db, SessionLocal
from app.models.domain import ModelMetrics
from app.services.sentiment_service import sentiment_service
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes database tables, loads ML model into memory once during startup,
    and seeds initial evaluation metrics.
    """
    print("=" * 70)
    print("STARTING SENTINEL AI BACKEND SERVICE...")
    print(f"Project: {settings.PROJECT_NAME} (v{settings.VERSION})")
    print("=" * 70)

    # 0. Ensure NLTK resources are available in production
    try:
        import nltk
        for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        print("[NLTK] Verified linguistic corpora and tokenizers.")
    except Exception as e:
        print(f"[NLTK WARNING] Resource check: {e}")

    # 1. Initialize Database Tables
    try:
        init_db()
        print("[DATABASE] Schema tables initialized successfully.")
    except Exception as e:
        print(f"[DATABASE ERROR] Table initialization failed: {e}")

    # 2. Warm up ML Predictor
    try:
        if sentiment_service.is_model_loaded():
            print(f"[ML ENGINE] Model pre-loaded: {sentiment_service.get_model_name()}")
        else:
            print("[ML ENGINE WARNING] Model artifacts not pre-loaded. Will load lazily.")
    except Exception as e:
        print(f"[ML ENGINE ERROR] Model warm-up failed: {e}")

    # 3. Seed initial evaluation metrics into database if empty
    try:
        db = SessionLocal()
        metrics_count = db.query(ModelMetrics).count()
        if metrics_count == 0 and sentiment_service.is_model_loaded() and sentiment_service.predictor.metadata:
            meta = sentiment_service.predictor.metadata
            seed_metrics = ModelMetrics(
                model_name=meta.get("model_name", "Logistic Regression"),
                accuracy=meta.get("accuracy", 0.7643),
                precision=meta.get("precision_macro", 0.7017),
                recall=meta.get("recall_macro", 0.7320),
                macro_f1=meta.get("f1_macro", 0.7138),
                weighted_f1=meta.get("f1_weighted", 0.7712),
            )
            db.add(seed_metrics)
            db.commit()
            print("[DATABASE] Evaluation metrics seeded successfully.")
        db.close()
    except Exception as e:
        print(f"[DATABASE WARNING] Failed to seed evaluation metrics: {e}")

    print("[SENTINEL AI] Backend is ready to accept requests.")
    yield
    print("[SENTINEL AI] Shutting down backend service.")


# Initialize FastAPI Application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|.*\.onrender\.com)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Centralized Exception Handlers
# ==========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors cleanly."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        errors.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request payload validation failed.",
            "errors": errors,
            "error_code": "VALIDATION_ERROR",
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles standard HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Centralized fallback handler preventing raw stacktrace leakage."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# ==========================================
# Register API Routers
# ==========================================

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
def health_check():
    """Top-level health check endpoint for cloud load balancers and Render."""
    return {
        "status": "healthy",
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": sentiment_service.is_model_loaded(),
    }


@app.get("/", tags=["Root"])
def root_endpoint():
    """Root platform entry point directing to Swagger documentation."""
    return {
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "documentation_url": "/docs",
        "api_v1_prefix": settings.API_V1_STR,
        "active_model": sentiment_service.get_model_name(),
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("ENVIRONMENT", "development").lower() == "development"
    uvicorn.run("main:app", host=host, port=port, reload=reload_flag)
