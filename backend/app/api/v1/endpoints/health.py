"""
SENTINEL AI – System Health & Dependency Diagnostics
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import get_db
from app.schemas.sentiment import HealthResponse
from app.services.sentiment_service import sentiment_service

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health & System Diagnostics",
    description="Validates API runtime availability, database connection integrity, "
                "and ML model artifact loading status in memory.",
)
def check_health(db: Session = Depends(get_db)):
    # Test database connectivity
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("Database health check probe failed: %s", e)
        db_status = "disconnected"

    model_loaded = sentiment_service.is_model_loaded()
    vec_loaded = sentiment_service.is_vectorizer_loaded()
    active_model = sentiment_service.get_model_name()

    overall_status = "healthy" if (db_status == "connected" and model_loaded and vec_loaded) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.VERSION,
        database=db_status,
        model_loaded=model_loaded,
        vectorizer_loaded=vec_loaded,
        active_model=active_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
