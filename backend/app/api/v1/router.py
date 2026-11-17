"""
SENTINEL AI – Aggregated v1 API Router
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    sentiment,
    batch,
    analytics,
    history,
    model,
    health,
)

api_router = APIRouter()

# 1. Single prediction & batch processing & history (grouped under /sentiment)
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment Analysis"])
api_router.include_router(batch.router, prefix="/sentiment", tags=["Batch CSV Analysis"])
api_router.include_router(history.router, prefix="/sentiment", tags=["Analysis History"])

# 2. Analytics & Visualizations
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics & Reporting"])

# 3. Model Information & Metrics
api_router.include_router(model.router, prefix="/model", tags=["Model Intelligence"])

# 4. Health Diagnostics
api_router.include_router(health.router, tags=["System Diagnostics"])
