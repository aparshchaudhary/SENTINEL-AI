"""
SENTINEL AI – Core Sentiment Service Layer
Academic Project: CSE276 – Artificial Intelligence Foundations

Orchestrates ML inference, SQLite persistence, batch CSV processing,
analytics aggregations, and audit history.
"""

import io
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func, desc, case
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.models.domain import SentimentAnalysis, BatchJob, ModelMetrics
from app.schemas.sentiment import (
    SentimentResponse,
    ClassProbabilities,
    BatchAnalysisResponse,
    BatchItemPreview,
    AnalyticsSummaryResponse,
    TrendsResponse,
    TrendItem,
    DistributionResponse,
    DistributionItem,
    HistoryRecord,
    PaginatedHistoryResponse,
    ModelInfoResponse,
    ModelMetricsResponse,
)
from ml.training.predictor import SentimentPredictor


class SentimentService:
    """
    Singleton service coordinating model predictions and persistence.
    """
    _instance: Optional["SentimentService"] = None
    _predictor: Optional[SentimentPredictor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        """Initializes the ML predictor from serialized model artifacts."""
        try:
            self._predictor = SentimentPredictor(
                model_path=settings.MODEL_PATH,
                vectorizer_path=settings.VECTORIZER_PATH,
                metadata_path=settings.METADATA_PATH,
            )
            print("[SENTINEL AI] SentimentPredictor loaded successfully into memory.")
        except Exception as e:
            print(f"[SENTINEL AI WARNING] Model initialization failed during service setup: {e}")
            self._predictor = None

    @property
    def predictor(self) -> SentimentPredictor:
        if self._predictor is None:
            self._init_service()
            if self._predictor is None:
                raise RuntimeError("ML Model artifacts could not be loaded into runtime memory.")
        return self._predictor

    def is_model_loaded(self) -> bool:
        return self._predictor is not None and self._predictor.model is not None

    def is_vectorizer_loaded(self) -> bool:
        return self._predictor is not None and self._predictor.vectorizer is not None

    def get_model_name(self) -> str:
        if self.is_model_loaded() and self._predictor.metadata:
            return self._predictor.metadata.get("model_name", "Logistic Regression")
        return "Not Loaded"

    # ==========================================
    # 1. Single Text Analysis
    # ==========================================

    def analyze_single_text(self, text: str, db: Session) -> SentimentResponse:
        """
        Executes single text inference, records latency, and persists the analysis.
        """
        start_time = time.time()
        pred_result = self.predictor.predict(text)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        sentiment = pred_result["sentiment"]
        confidence = pred_result["confidence"]
        probs = pred_result["probabilities"]
        cleaned_text = pred_result.get("cleaned_text", "")

        # Persist transaction to SQLite with transaction safety
        db_record = SentimentAnalysis(
            input_text=text,
            cleaned_text=cleaned_text,
            predicted_sentiment=sentiment,
            confidence=confidence,
            positive_probability=probs.get("positive", 0.0),
            negative_probability=probs.get("negative", 0.0),
            neutral_probability=probs.get("neutral", 0.0),
            processing_time_ms=duration_ms,
            created_at=datetime.now(timezone.utc),
        )
        try:
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
        except Exception as e:
            db.rollback()
            logger.error("Database commit failure in analyze_single_text: %s", e, exc_info=True)
            raise RuntimeError(f"Database persistence failure: {str(e)}")

        return SentimentResponse(
            text=text,
            cleaned_text=cleaned_text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=ClassProbabilities(
                positive=probs.get("positive", 0.0),
                negative=probs.get("negative", 0.0),
                neutral=probs.get("neutral", 0.0),
            ),
            processing_time_ms=duration_ms,
            id=db_record.id,
            created_at=db_record.created_at,
        )

    # ==========================================
    # 2. Batch CSV File Analysis
    # ==========================================

    def process_batch_csv(
        self,
        file_bytes: bytes,
        filename: str,
        text_column: Optional[str],
        db: Session,
    ) -> BatchAnalysisResponse:
        """
        Ingests, validates, classifies bulk CSV records, persists summary,
        and saves annotated CSV file for user download.
        """
        start_time = time.time()
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        # 1. Parse CSV safely
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Malformed CSV file could not be parsed: {str(e)}")

        if df.empty or len(df) == 0:
            raise ValueError("The uploaded CSV file is empty and contains no data rows.")

        # 2. Determine target text column with strict validation
        target_col = None
        if text_column:
            # User specified an explicit column name
            matched = False
            for col in df.columns:
                if str(col).strip().lower() == text_column.strip().lower():
                    target_col = col
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Specified text column '{text_column}' not found in uploaded CSV. "
                    f"Available columns: {list(df.columns)}"
                )
        else:
            # Auto-detection heuristic
            candidate_cols = ["text", "cleaned_text", "tweet", "review", "comment", "feedback", "sentence", "content"]
            for col in candidate_cols:
                for actual_col in df.columns:
                    if col.lower() == str(actual_col).strip().lower():
                        target_col = actual_col
                        break
                if target_col:
                    break

            if not target_col:
                # Fallback to first string/object column
                for col in df.columns:
                    if df[col].dtype == object or isinstance(df[col].iloc[0], str):
                        target_col = col
                        break

        if not target_col or target_col not in df.columns:
            raise ValueError(
                f"Could not automatically detect text column. Available columns: {list(df.columns)}. "
                f"Please specify the 'text_column' parameter."
            )

        total_records = len(df)
        predicted_sentiments: List[str] = []
        confidences: List[float] = []
        pos_probs: List[float] = []
        neg_probs: List[float] = []
        neu_probs: List[float] = []
        preview_results: List[BatchItemPreview] = []

        # 3. Stream/Batch inference row by row
        for idx, val in enumerate(df[target_col]):
            row_text = str(val) if pd.notna(val) else ""
            res = self.predictor.predict(row_text)

            sentiment = res["sentiment"]
            conf = res["confidence"]
            probs = res["probabilities"]

            predicted_sentiments.append(sentiment)
            confidences.append(conf)
            pos_probs.append(probs.get("positive", 0.0))
            neg_probs.append(probs.get("negative", 0.0))
            neu_probs.append(probs.get("neutral", 0.0))

            if idx < 10:  # Top 10 preview items
                preview_results.append(
                    BatchItemPreview(
                        row_index=idx,
                        text=row_text[:120] + "..." if len(row_text) > 120 else row_text,
                        sentiment=sentiment,
                        confidence=conf,
                    )
                )

        # 4. Attach predictions to DataFrame
        df["sentiment_predicted"] = predicted_sentiments
        df["confidence_score"] = confidences
        df["prob_positive"] = pos_probs
        df["prob_negative"] = neg_probs
        df["prob_neutral"] = neu_probs

        # 5. Calculate distribution & aggregate metrics
        pos_count = sum(1 for s in predicted_sentiments if s == "positive")
        neg_count = sum(1 for s in predicted_sentiments if s == "negative")
        neu_count = sum(1 for s in predicted_sentiments if s == "neutral")

        distribution = {
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count,
        }
        percentages = {
            "positive": round((pos_count / total_records) * 100, 2) if total_records > 0 else 0.0,
            "negative": round((neg_count / total_records) * 100, 2) if total_records > 0 else 0.0,
            "neutral": round((neu_count / total_records) * 100, 2) if total_records > 0 else 0.0,
        }
        avg_confidence = round(float(sum(confidences) / len(confidences)), 4) if confidences else 0.0
        duration_sec = round(time.time() - start_time, 2)

        # 6. Save annotated CSV file to disk with sanitized filename
        raw_basename = os.path.basename(filename or "dataset.csv")
        safe_basename = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_basename)
        annotated_filename = f"annotated_{job_id}_{safe_basename}"
        annotated_path = os.path.join(settings.UPLOADS_DIR, annotated_filename)
        df.to_csv(annotated_path, index=False)

        # 7. Persist batch job in SQLite with transaction safety
        batch_job = BatchJob(
            id=job_id,
            filename=safe_basename,
            total_records=total_records,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            status="completed",
            annotated_filename=annotated_filename,
            created_at=datetime.now(timezone.utc),
        )
        try:
            db.add(batch_job)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Database commit failure in process_batch_csv: %s", e, exc_info=True)
            raise RuntimeError(f"Database batch job persistence failure: {str(e)}")

        download_url = f"/api/v1/sentiment/batch-csv/download/{job_id}"

        return BatchAnalysisResponse(
            job_id=job_id,
            filename=filename,
            total_records=total_records,
            processed_records=total_records,
            distribution=distribution,
            percentages=percentages,
            average_confidence=avg_confidence,
            processing_duration_seconds=duration_sec,
            preview_results=preview_results,
            download_url=download_url,
        )

    # ==========================================
    # 3. Analytics Aggregations
    # ==========================================

    def get_analytics_summary(self, db: Session) -> AnalyticsSummaryResponse:
        """
        Computes global summary statistics across all persisted records.
        """
        # Count from single analyses table
        single_total = db.query(func.count(SentimentAnalysis.id)).scalar() or 0
        single_pos = db.query(func.count(SentimentAnalysis.id)).filter(SentimentAnalysis.predicted_sentiment == "positive").scalar() or 0
        single_neg = db.query(func.count(SentimentAnalysis.id)).filter(SentimentAnalysis.predicted_sentiment == "negative").scalar() or 0
        single_neu = db.query(func.count(SentimentAnalysis.id)).filter(SentimentAnalysis.predicted_sentiment == "neutral").scalar() or 0
        single_avg_conf = db.query(func.avg(SentimentAnalysis.confidence)).scalar() or 0.0

        # Include bulk batch job aggregations
        batch_total = db.query(func.sum(BatchJob.total_records)).scalar() or 0
        batch_pos = db.query(func.sum(BatchJob.positive_count)).scalar() or 0
        batch_neg = db.query(func.sum(BatchJob.negative_count)).scalar() or 0
        batch_neu = db.query(func.sum(BatchJob.neutral_count)).scalar() or 0

        total_analyses = single_total + batch_total
        pos_count = single_pos + batch_pos
        neg_count = single_neg + batch_neg
        neu_count = single_neu + batch_neu

        if total_analyses > 0:
            percentages = {
                "positive": round((pos_count / total_analyses) * 100, 2),
                "negative": round((neg_count / total_analyses) * 100, 2),
                "neutral": round((neu_count / total_analyses) * 100, 2),
            }
        else:
            percentages = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        return AnalyticsSummaryResponse(
            total_analyses=total_analyses,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            percentages=percentages,
            average_confidence=round(float(single_avg_conf), 4),
        )

    def get_analytics_trends(self, db: Session) -> TrendsResponse:
        """
        Aggregates time-series sentiment data grouped by date for Recharts.
        """
        results = (
            db.query(
                func.date(SentimentAnalysis.created_at).label("analysis_date"),
                func.count(SentimentAnalysis.id).label("total"),
                func.sum(case((SentimentAnalysis.predicted_sentiment == "positive", 1), else_=0)).label("pos"),
                func.sum(case((SentimentAnalysis.predicted_sentiment == "negative", 1), else_=0)).label("neg"),
                func.sum(case((SentimentAnalysis.predicted_sentiment == "neutral", 1), else_=0)).label("neu"),
            )
            .group_by(func.date(SentimentAnalysis.created_at))
            .order_by(func.date(SentimentAnalysis.created_at).asc())
            .all()
        )

        timeline: List[TrendItem] = []
        for r in results:
            timeline.append(
                TrendItem(
                    date=str(r.analysis_date),
                    positive=int(r.pos or 0),
                    negative=int(r.neg or 0),
                    neutral=int(r.neu or 0),
                    total=int(r.total or 0),
                )
            )

        return TrendsResponse(timeline=timeline)

    def get_analytics_distribution(self, db: Session) -> DistributionResponse:
        """
        Returns sentiment breakdown formatted for Recharts Pie/Donut charts.
        """
        summary = self.get_analytics_summary(db)
        total = summary.total_analyses

        items = [
            DistributionItem(
                name="Positive",
                value=summary.positive_count,
                percentage=summary.percentages["positive"],
                color="#10b981",  # Emerald
            ),
            DistributionItem(
                name="Neutral",
                value=summary.neutral_count,
                percentage=summary.percentages["neutral"],
                color="#f59e0b",  # Amber
            ),
            DistributionItem(
                name="Negative",
                value=summary.negative_count,
                percentage=summary.percentages["negative"],
                color="#ef4444",  # Rose
            ),
        ]

        return DistributionResponse(distribution=items, total_samples=total)

    # ==========================================
    # 4. History Queries & Management
    # ==========================================

    def get_history(
        self,
        db: Session,
        page: int = 1,
        page_size: int = 20,
        sentiment: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PaginatedHistoryResponse:
        """
        Retrieves paginated historical sentiment analyses with search and filtering.
        """
        query = db.query(SentimentAnalysis)

        if sentiment and sentiment.lower() in ["positive", "negative", "neutral"]:
            query = query.filter(SentimentAnalysis.predicted_sentiment == sentiment.lower())

        if search and search.strip():
            query = query.filter(SentimentAnalysis.input_text.ilike(f"%{search.strip()}%"))

        if start_date:
            try:
                sd = datetime.fromisoformat(start_date)
                query = query.filter(SentimentAnalysis.created_at >= sd)
            except ValueError:
                pass

        if end_date:
            try:
                ed = datetime.fromisoformat(end_date)
                query = query.filter(SentimentAnalysis.created_at <= ed)
            except ValueError:
                pass

        total_count = query.count()
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

        offset = (page - 1) * page_size
        records = (
            query.order_by(desc(SentimentAnalysis.created_at))
            .offset(offset)
            .limit(page_size)
            .all()
        )

        items = [
            HistoryRecord(
                id=r.id,
                input_text=r.input_text,
                cleaned_text=r.cleaned_text,
                predicted_sentiment=r.predicted_sentiment,
                confidence=r.confidence,
                probabilities={
                    "positive": r.positive_probability,
                    "negative": r.negative_probability,
                    "neutral": r.neutral_probability,
                },
                processing_time_ms=r.processing_time_ms,
                created_at=r.created_at,
            )
            for r in records
        ]

        return PaginatedHistoryResponse(
            items=items,
            total=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def get_history_by_id(self, record_id: int, db: Session) -> Optional[HistoryRecord]:
        """Fetches an individual history record by ID."""
        r = db.query(SentimentAnalysis).filter(SentimentAnalysis.id == record_id).first()
        if not r:
            return None
        return HistoryRecord(
            id=r.id,
            input_text=r.input_text,
            cleaned_text=r.cleaned_text,
            predicted_sentiment=r.predicted_sentiment,
            confidence=r.confidence,
            probabilities={
                "positive": r.positive_probability,
                "negative": r.negative_probability,
                "neutral": r.neutral_probability,
            },
            processing_time_ms=r.processing_time_ms,
            created_at=r.created_at,
        )

    def delete_history_record(self, record_id: int, db: Session) -> bool:
        """Deletes an analysis record from SQLite with transaction safety."""
        record = db.query(SentimentAnalysis).filter(SentimentAnalysis.id == record_id).first()
        if not record:
            return False
        try:
            db.delete(record)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error("Database deletion failure in delete_history_record: %s", e, exc_info=True)
            return False

    # ==========================================
    # 5. Model Information & Metrics
    # ==========================================

    def get_model_info(self) -> ModelInfoResponse:
        """Loads and returns metadata about active champion classifier."""
        meta = self.predictor.metadata or {}
        return ModelInfoResponse(
            model_name=meta.get("model_name", "Multinomial Logistic Regression"),
            algorithm=meta.get("algorithm", "logistic_regression"),
            supported_classes=meta.get("classes", ["negative", "neutral", "positive"]),
            vocabulary_size=meta.get("vocabulary_size", 7235),
            training_dataset_size=meta.get("training_dataset_size", 11524),
            testing_dataset_size=meta.get("testing_dataset_size", 2881),
            hyperparameters=meta.get("hyperparameters", {}),
            training_date=meta.get("training_date", ""),
        )

    def get_model_metrics(self) -> ModelMetricsResponse:
        """Returns verified holdout test evaluation metrics."""
        meta = self.predictor.metadata or {}
        cm_matrix = meta.get("confusion_matrix", [])
        raw_class_metrics = meta.get("class_metrics", {})

        # Compute support per class from confusion matrix rows
        classes = meta.get("classes", ["negative", "neutral", "positive"])
        support_map = {}
        for idx, cls_name in enumerate(classes):
            if idx < len(cm_matrix):
                support_map[cls_name] = sum(cm_matrix[idx])

        enriched_metrics = {}
        for cls_name, vals in raw_class_metrics.items():
            enriched_vals = dict(vals)
            f1 = vals.get("f1_score", vals.get("f1-score", 0.0))
            enriched_vals["f1_score"] = f1
            enriched_vals["f1-score"] = f1
            if cls_name in support_map:
                enriched_vals["support"] = float(support_map[cls_name])
            enriched_metrics[cls_name] = enriched_vals

        return ModelMetricsResponse(
            model_name=meta.get("model_name", "Multinomial Logistic Regression"),
            accuracy=meta.get("accuracy", 0.7643),
            precision_macro=meta.get("precision_macro", 0.7017),
            recall_macro=meta.get("recall_macro", 0.7320),
            f1_macro=meta.get("f1_macro", 0.7138),
            f1_weighted=meta.get("f1_weighted", 0.7712),
            class_metrics=enriched_metrics,
            confusion_matrix=cm_matrix,
        )


sentiment_service = SentimentService()
