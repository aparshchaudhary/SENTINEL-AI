"""
SENTINEL AI – Analysis History & Audit Trail Endpoints
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.sentiment import (
    PaginatedHistoryResponse,
    HistoryRecord,
    DeleteResponse,
)
from app.services.sentiment_service import sentiment_service

router = APIRouter()


@router.get(
    "/history",
    response_model=PaginatedHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Paginated Analysis History",
    description="Retrieves chronological sentiment analysis history with keyword search, "
                "polarity filtering, date interval filtering, and pagination.",
)
def get_history(
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    sentiment: Optional[str] = Query(None, description="Filter by polarity (positive, negative, neutral)"),
    search: Optional[str] = Query(None, description="Search query string matching input text"),
    start_date: Optional[str] = Query(None, description="ISO format start date filter (e.g. 2026-11-01)"),
    end_date: Optional[str] = Query(None, description="ISO format end date filter (e.g. 2026-11-30)"),
    db: Session = Depends(get_db),
):
    return sentiment_service.get_history(
        db=db,
        page=page,
        page_size=page_size,
        sentiment=sentiment,
        search=search,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/history/{record_id}",
    response_model=HistoryRecord,
    status_code=status.HTTP_200_OK,
    summary="Get Single History Record",
    description="Fetches comprehensive details for a specific analysis query by record ID.",
)
def get_history_by_id(
    record_id: int,
    db: Session = Depends(get_db),
):
    record = sentiment_service.get_history_by_id(record_id=record_id, db=db)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with ID {record_id} does not exist.",
        )
    return record


@router.delete(
    "/history/{record_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete History Record",
    description="Permanently removes an analysis record from SQLite storage.",
)
def delete_history_record(
    record_id: int,
    db: Session = Depends(get_db),
):
    deleted = sentiment_service.delete_history_record(record_id=record_id, db=db)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with ID {record_id} does not exist.",
        )
    return DeleteResponse(
        success=True,
        message=f"Analysis record {record_id} deleted successfully.",
        id=record_id,
    )
