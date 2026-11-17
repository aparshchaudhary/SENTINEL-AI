"""
SENTINEL AI – Real-time Sentiment Prediction Endpoint
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.sentiment import SentimentRequest, SentimentResponse
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Single Text Sentiment",
    description="Analyzes input text using the trained champion Logistic Regression NLP model, "
                "yielding sentiment polarity, class probabilities, and processing latency.",
)
def predict_sentiment(
    payload: SentimentRequest,
    db: Session = Depends(get_db),
):
    if not sentiment_service.is_model_loaded():
        logger.error("Predict called while ML model is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment classification model is currently unavailable. Please check system health.",
        )

    try:
        response = sentiment_service.analyze_single_text(text=payload.text, db=db)
        return response
    except Exception as e:
        logger.error("Inference processing error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the sentiment prediction request.",
        )
