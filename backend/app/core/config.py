"""
SENTINEL AI – Application Configuration Settings
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()


class Settings:
    PROJECT_NAME: str = "SENTINEL AI – Intelligent Sentiment Analysis Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    DESCRIPTION: str = (
        "Enterprise-grade intelligent sentiment analysis platform powered by genuine "
        "machine learning (NLP) classifiers for CSE276 Artificial Intelligence Foundations."
    )

    # Base Directories
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ML_DIR: str = os.path.join(BASE_DIR, "ml")
    SAVED_MODELS_DIR: str = os.path.join(ML_DIR, "saved_models")
    UPLOADS_DIR: str = os.path.join(BASE_DIR, "uploads")

    # ML Artifacts
    MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(SAVED_MODELS_DIR, "model.joblib"))
    VECTORIZER_PATH: str = os.getenv("VECTORIZER_PATH", os.path.join(SAVED_MODELS_DIR, "vectorizer.joblib"))
    METADATA_PATH: str = os.getenv("METADATA_PATH", os.path.join(SAVED_MODELS_DIR, "metadata.json"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'sentinel.db')}"
    )

    # CORS Configuration
    DEFAULT_CORS: List[str] = [
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:3000",
    ]

    @property
    def CORS_ORIGINS(self) -> List[str]:
        raw = os.getenv("CORS_ORIGINS")
        if raw:
            origins = [o.strip() for o in raw.split(",") if o.strip()]
            # Filter out wildcard "*" when allow_credentials=True is active
            return [o for o in origins if o != "*"]
        return self.DEFAULT_CORS

    # Security & File Upload Limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))


settings = Settings()

# Ensure uploads directory exists
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
