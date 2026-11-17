"""
SENTINEL AI – Pydantic Data Contracts & Schemas
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ==========================================
# 1. Single Text Prediction Schemas
# ==========================================

class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        description="Raw input text to analyze for sentiment polarity",
        min_length=1,
        max_length=5000,
        examples=["I absolutely love this airline! The flight crew was outstanding and seats were comfortable."]
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty or purely whitespace.")
        return v.strip()


class ClassProbabilities(BaseModel):
    positive: float = Field(..., ge=0.0, le=1.0)
    negative: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)


class SentimentResponse(BaseModel):
    text: str
    cleaned_text: Optional[str] = None
    sentiment: str = Field(..., description="Predicted polarity: positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated maximum class likelihood")
    probabilities: ClassProbabilities
    processing_time_ms: float
    id: Optional[int] = None
    created_at: Optional[datetime] = None


# ==========================================
# 2. Batch CSV Analysis Schemas
# ==========================================

class BatchItemPreview(BaseModel):
    row_index: int
    text: str
    sentiment: str
    confidence: float


class BatchAnalysisResponse(BaseModel):
    job_id: str
    filename: str
    total_records: int
    processed_records: int
    distribution: Dict[str, int]
    percentages: Dict[str, float]
    average_confidence: float
    processing_duration_seconds: float
    preview_results: List[BatchItemPreview]
    download_url: Optional[str] = None


# ==========================================
# 3. Analytics Schemas
# ==========================================

class AnalyticsSummaryResponse(BaseModel):
    total_analyses: int
    positive_count: int
    negative_count: int
    neutral_count: int
    percentages: Dict[str, float]
    average_confidence: float


class TrendItem(BaseModel):
    date: str
    positive: int
    negative: int
    neutral: int
    total: int


class TrendsResponse(BaseModel):
    timeline: List[TrendItem]


class DistributionItem(BaseModel):
    name: str
    value: int
    percentage: float
    color: str


class DistributionResponse(BaseModel):
    distribution: List[DistributionItem]
    total_samples: int


# ==========================================
# 4. History Schemas
# ==========================================

class HistoryRecord(BaseModel):
    id: int
    input_text: str
    cleaned_text: Optional[str] = None
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    created_at: datetime


class PaginatedHistoryResponse(BaseModel):
    items: List[HistoryRecord]
    total: int
    page: int
    page_size: int
    total_pages: int


class DeleteResponse(BaseModel):
    success: bool
    message: str
    id: int


# ==========================================
# 5. Model Information & Metrics Schemas
# ==========================================

class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    supported_classes: List[str]
    vocabulary_size: int
    training_dataset_size: int
    testing_dataset_size: int
    hyperparameters: Dict[str, Any]
    training_date: str


class ModelMetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]


# ==========================================
# 6. Health Diagnostic Schema
# ==========================================

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    model_loaded: bool
    vectorizer_loaded: bool
    active_model: str
    timestamp: str
