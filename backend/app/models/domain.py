"""
SENTINEL AI – SQLAlchemy Database Models
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from app.database.session import Base


def utc_now():
    return datetime.now(timezone.utc)


class SentimentAnalysis(Base):
    """
    Persists individual real-time sentiment analysis queries,
    predicted polarity, estimated class probabilities, and response latency.
    """
    __tablename__ = "sentiment_analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    input_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    predicted_sentiment = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    positive_probability = Column(Float, nullable=False)
    negative_probability = Column(Float, nullable=False)
    neutral_probability = Column(Float, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, index=True)


class BatchJob(Base):
    """
    Records metadata and aggregate distribution statistics
    for bulk CSV upload analysis sessions.
    """
    __tablename__ = "batch_jobs"

    id = Column(String(50), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    total_records = Column(Integer, nullable=False)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    status = Column(String(30), default="completed")
    annotated_filename = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True)


class ModelMetrics(Base):
    """
    Stores historical offline evaluation benchmarks and performance metrics.
    """
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    macro_f1 = Column(Float, nullable=False)
    weighted_f1 = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now)
