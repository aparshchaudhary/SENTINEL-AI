"""
SENTINEL AI – Batch CSV Sentiment Analysis & Download Endpoints
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

import logging
import os
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import get_db
from app.models.domain import BatchJob
from app.schemas.sentiment import BatchAnalysisResponse
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/batch-csv",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Bulk CSV Dataset",
    description="Accepts a CSV file, detects the text column, classifies every record in bulk "
                "with the trained model, aggregates distribution metrics, and generates an annotated CSV.",
)
async def analyze_batch_csv(
    file: UploadFile = File(..., description="Target CSV file to analyze"),
    text_column: Optional[str] = Form(None, description="Explicit name of the text column (optional)"),
    db: Session = Depends(get_db),
):
    if not sentiment_service.is_model_loaded():
        logger.error("Batch CSV analysis called while ML model is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment classification model is currently unavailable. Please check system health.",
        )

    # Validate extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .csv files are supported.",
        )

    # Read bytes with size protection
    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file content.",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum permissible size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is completely empty.",
        )

    try:
        result = sentiment_service.process_batch_csv(
            file_bytes=content,
            filename=file.filename,
            text_column=text_column,
            db=db,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Internal batch processing error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during batch CSV processing.",
        )


@router.get(
    "/batch-csv/download/{job_id}",
    summary="Download Annotated Batch CSV",
    description="Downloads the resulting CSV file containing original rows plus predicted sentiments and probabilities.",
)
def download_batch_csv(
    job_id: str,
    db: Session = Depends(get_db),
):
    job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
    if not job or not job.annotated_filename:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No batch job found with ID: {job_id}",
        )

    file_path = os.path.join(settings.UPLOADS_DIR, job.annotated_filename)
    resolved_path = os.path.abspath(file_path)
    uploads_dir_abs = os.path.abspath(settings.UPLOADS_DIR)

    # Path traversal protection
    if not resolved_path.startswith(uploads_dir_abs):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path access denied.",
        )

    if not os.path.exists(resolved_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated result file has expired or was removed from server.",
        )

    return FileResponse(
        path=resolved_path,
        media_type="text/csv",
        filename=os.path.basename(resolved_path),
    )
