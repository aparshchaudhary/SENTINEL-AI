"""
SENTINEL AI – Analytics & Aggregation Endpoints
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.sentiment import (
    AnalyticsSummaryResponse,
    TrendsResponse,
    DistributionResponse,
)
from app.services.sentiment_service import sentiment_service

router = APIRouter()


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Analytics Summary",
    description="Returns aggregate sentiment distribution counts, percentages, and global average confidence.",
)
def get_analytics_summary(db: Session = Depends(get_db)):
    return sentiment_service.get_analytics_summary(db=db)


@router.get(
    "/trends",
    response_model=TrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Sentiment Timeline Trends",
    description="Returns time-series sentiment classifications grouped by date for Recharts area/line charts.",
)
def get_analytics_trends(db: Session = Depends(get_db)):
    return sentiment_service.get_analytics_trends(db=db)


@router.get(
    "/distribution",
    response_model=DistributionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Visual Sentiment Distribution",
    description="Returns sentiment distribution formatted with theme colors for Recharts pie and donut charts.",
)
def get_analytics_distribution(db: Session = Depends(get_db)):
    return sentiment_service.get_analytics_distribution(db=db)
