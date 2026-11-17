"""
SENTINEL AI – Model Metadata & Evaluation Benchmarks Endpoints
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from fastapi import APIRouter, status

from app.schemas.sentiment import ModelInfoResponse, ModelMetricsResponse
from app.services.sentiment_service import sentiment_service

router = APIRouter()


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Active Model Information",
    description="Returns metadata about the deployed champion classifier, algorithm configuration, "
                "vocabulary size, and hyperparameters.",
)
def get_model_info():
    return sentiment_service.get_model_info()


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Holdout Evaluation Metrics",
    description="Returns actual statistical metrics evaluated on the untouched holdout test partition "
                "(accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix).",
)
def get_model_metrics():
    return sentiment_service.get_model_metrics()
