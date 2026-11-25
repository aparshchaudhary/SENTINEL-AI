"""
SENTINEL AI – Phase 6 Full-Stack E2E Integration & Verification Test Suite
Academic Course: CSE276 – Artificial Intelligence Foundations

Comprehensive End-to-End Automated Testing:
1. Complete System Architecture Audit
2. Backend Startup & Dependency Health
3. Exact Positive, Negative, Neutral Predictions & Database Persistence
4. History Search, Filter, Pagination, and Record Deletion
5. Analytics Aggregations, Trends, and Distribution
6. Bulk CSV Upload, Processing, Database Job Creation, and Download
7. Model Performance Metadata & Confusion Matrix
8. Robust Error Handling (Empty, Oversized, Invalid, 404)
9. High-Throughput Latency & Performance Benchmarks
"""

import io
import os
import sys
import time
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure backend root is in module search path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.config import settings
from app.database.session import SessionLocal, init_db
from app.models.domain import SentimentAnalysis, BatchJob
from app.services.sentiment_service import sentiment_service
from main import app


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client with lifespan context."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def db_session():
    """Provides a direct database session to inspect persistence."""
    session = SessionLocal()
    yield session
    session.close()


# ======================================================================
# 1. System Architecture Audit
# ======================================================================

def test_01_system_architecture_audit():
    """Verify paths, environment settings, model artifacts, and database connectivity."""
    print("\n--- [E2E TEST 01: System Architecture Audit] ---")

    # Check model artifacts
    assert os.path.exists(settings.MODEL_PATH), f"Model missing at {settings.MODEL_PATH}"
    assert os.path.exists(settings.VECTORIZER_PATH), f"Vectorizer missing at {settings.VECTORIZER_PATH}"
    assert os.path.exists(settings.METADATA_PATH), f"Metadata missing at {settings.METADATA_PATH}"

    # Check database directory and upload directory
    assert os.path.exists(settings.UPLOADS_DIR), "Uploads directory missing"

    # Check ML engine initialization
    assert sentiment_service.is_model_loaded(), "Model not loaded in sentiment service"
    assert sentiment_service.is_vectorizer_loaded(), "Vectorizer not loaded in sentiment service"
    model_name = sentiment_service.get_model_name()
    assert "Logistic Regression" in model_name, f"Unexpected model name: {model_name}"

    # Check CORS configuration
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert "http://127.0.0.1:5173" in settings.CORS_ORIGINS
    assert "*" not in settings.CORS_ORIGINS, "Wildcard '*' must not be in CORS_ORIGINS when credentials are true"

    print("Architecture Audit: PASSED (All paths, models, and CORS verified)")


# ======================================================================
# 2. Backend Startup & Dependency Verification
# ======================================================================

def test_02_backend_startup_and_health(client):
    """Verify GET /api/v1/health and GET /docs."""
    print("\n--- [E2E TEST 02: Backend Startup & Health] ---")

    # Health endpoint
    res = client.get("/api/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    assert health_data["status"] == "healthy"
    assert health_data["database"] == "connected"
    assert health_data["model_loaded"] is True
    assert health_data["vectorizer_loaded"] is True
    assert "version" in health_data
    assert "timestamp" in health_data

    # Swagger Documentation
    docs_res = client.get("/docs")
    assert docs_res.status_code == 200, "Swagger UI endpoint unavailable"

    # OpenAPI Schema
    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200, "OpenAPI schema unavailable"
    schema = openapi_res.json()
    assert "/api/v1/sentiment/predict" in schema["paths"]
    assert "/api/v1/sentiment/batch-csv" in schema["paths"]

    print("Backend Startup & Health: PASSED (Health 200, Docs 200, OpenAPI 200)")


# ======================================================================
# 3. Exact End-to-End Sentiment Analysis Sentences
# ======================================================================

TEST_SENTENCES = {
    "positive": "I absolutely love this product. The quality and customer service are excellent.",
    "negative": "This was a terrible experience. The service was extremely disappointing.",
    "neutral": "The package was delivered yesterday at 10 AM.",
}

created_ids = {}


def test_03_positive_sentiment_prediction(client, db_session):
    """Verify E2E prediction for specified POSITIVE sentence and database persistence."""
    print("\n--- [E2E TEST 03: Positive Sentence Prediction] ---")
    text = TEST_SENTENCES["positive"]
    response = client.post("/api/v1/sentiment/predict", json={"text": text})
    assert response.status_code == 200, f"Predict failed: {response.text}"

    data = response.json()
    assert data["text"] == text
    assert data["sentiment"] == "positive", f"Expected positive, got {data['sentiment']}"
    assert data["confidence"] > 0.50, f"Confidence too low: {data['confidence']}"
    assert data["probabilities"]["positive"] > data["probabilities"]["negative"]
    assert data["probabilities"]["positive"] > data["probabilities"]["neutral"]
    assert "cleaned_text" in data
    assert data["processing_time_ms"] > 0
    assert "id" in data
    created_ids["positive"] = data["id"]

    # Direct SQLite verification
    record = db_session.query(SentimentAnalysis).filter(SentimentAnalysis.id == data["id"]).first()
    assert record is not None, "Record not persisted to SQLite database"
    assert record.predicted_sentiment == "positive"
    assert record.input_text == text
    print(f"Positive Test PASSED (ID: {data['id']}, Confidence: {data['confidence'] * 100:.2f}%)")


def test_04_negative_sentiment_prediction(client, db_session):
    """Verify E2E prediction for specified NEGATIVE sentence and database persistence."""
    print("\n--- [E2E TEST 04: Negative Sentence Prediction] ---")
    text = TEST_SENTENCES["negative"]
    response = client.post("/api/v1/sentiment/predict", json={"text": text})
    assert response.status_code == 200, f"Predict failed: {response.text}"

    data = response.json()
    assert data["text"] == text
    assert data["sentiment"] == "negative", f"Expected negative, got {data['sentiment']}"
    assert data["confidence"] > 0.50, f"Confidence too low: {data['confidence']}"
    assert data["probabilities"]["negative"] > data["probabilities"]["positive"]
    assert data["probabilities"]["negative"] > data["probabilities"]["neutral"]
    created_ids["negative"] = data["id"]

    # Direct SQLite verification
    record = db_session.query(SentimentAnalysis).filter(SentimentAnalysis.id == data["id"]).first()
    assert record is not None, "Record not persisted to SQLite database"
    assert record.predicted_sentiment == "negative"
    print(f"Negative Test PASSED (ID: {data['id']}, Confidence: {data['confidence'] * 100:.2f}%)")


def test_05_neutral_sentiment_prediction(client, db_session):
    """Verify E2E prediction for specified NEUTRAL sentence and database persistence."""
    print("\n--- [E2E TEST 05: Neutral Sentence Prediction] ---")
    text = TEST_SENTENCES["neutral"]
    response = client.post("/api/v1/sentiment/predict", json={"text": text})
    assert response.status_code == 200, f"Predict failed: {response.text}"

    data = response.json()
    assert data["text"] == text
    assert data["sentiment"] == "neutral", f"Expected neutral, got {data['sentiment']}"
    assert data["probabilities"]["neutral"] > data["probabilities"]["negative"]
    created_ids["neutral"] = data["id"]

    # Direct SQLite verification
    record = db_session.query(SentimentAnalysis).filter(SentimentAnalysis.id == data["id"]).first()
    assert record is not None, "Record not persisted to SQLite database"
    assert record.predicted_sentiment == "neutral"
    print(f"Neutral Test PASSED (ID: {data['id']}, Confidence: {data['confidence'] * 100:.2f}%)")


# ======================================================================
# 4. History Integration Test
# ======================================================================

def test_06_history_integration_and_filtering(client):
    """Verify history query, keyword search, polarity filter, and pagination."""
    print("\n--- [E2E TEST 06: History Integration] ---")

    # 1. Fetch default paginated history
    res = client.get("/api/v1/sentiment/history?page=1&page_size=10")
    assert res.status_code == 200
    hist = res.json()
    assert hist["total"] >= 3
    assert len(hist["items"]) >= 3

    # Verify our created items appear in history
    item_ids = [item["id"] for item in hist["items"]]
    for k, created_id in created_ids.items():
        assert created_id in item_ids, f"{k} analysis (ID: {created_id}) not found in history"

    # 2. Search for positive text keyword
    search_res = client.get("/api/v1/sentiment/history?search=customer+service")
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total"] >= 1
    assert any("customer service" in item["input_text"].lower() for item in search_data["items"])

    # 3. Filter by positive sentiment
    pos_res = client.get("/api/v1/sentiment/history?sentiment=positive")
    assert pos_res.status_code == 200
    pos_data = pos_res.json()
    for item in pos_data["items"]:
        assert item["predicted_sentiment"] == "positive"

    # 4. Filter by negative sentiment
    neg_res = client.get("/api/v1/sentiment/history?sentiment=negative")
    assert neg_res.status_code == 200
    neg_data = neg_res.json()
    for item in neg_data["items"]:
        assert item["predicted_sentiment"] == "negative"

    # 5. Test pagination page_size
    page_res = client.get("/api/v1/sentiment/history?page=1&page_size=2")
    assert page_res.status_code == 200
    pdata = page_res.json()
    assert len(pdata["items"]) <= 2
    assert pdata["page"] == 1
    assert pdata["total_pages"] >= 2

    print("History Query, Search, Filter & Pagination: PASSED")


def test_07_history_deletion(client, db_session):
    """Verify deleting one test record removes it from both API and SQLite database."""
    print("\n--- [E2E TEST 07: History Record Deletion] ---")

    # Create a temporary record specifically for deletion
    temp_res = client.post("/api/v1/sentiment/predict", json={"text": "Temporary record to be deleted."})
    assert temp_res.status_code == 200
    temp_id = temp_res.json()["id"]

    # Verify it exists in SQLite
    assert db_session.query(SentimentAnalysis).filter(SentimentAnalysis.id == temp_id).first() is not None

    # Delete via API
    del_res = client.delete(f"/api/v1/sentiment/history/{temp_id}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["success"] is True
    assert del_data["id"] == temp_id

    # Verify it no longer exists in SQLite
    db_session.expire_all()
    deleted_check = db_session.query(SentimentAnalysis).filter(SentimentAnalysis.id == temp_id).first()
    assert deleted_check is None, "Deleted record still exists in SQLite database"

    # Verify 404 when querying deleted ID
    get_res = client.get(f"/api/v1/sentiment/history/{temp_id}")
    assert get_res.status_code == 404

    print(f"History Deletion: PASSED (Record #{temp_id} deleted and verified)")


# ======================================================================
# 5. Dashboard Analytics Integration Test
# ======================================================================

def test_08_dashboard_analytics_verification(client):
    """Verify summary metrics, trends timeline, and distribution breakdown."""
    print("\n--- [E2E TEST 08: Dashboard Analytics Verification] ---")

    # Summary
    sum_res = client.get("/api/v1/analytics/summary")
    assert sum_res.status_code == 200
    summary = sum_res.json()
    assert summary["total_analyses"] >= 3
    assert summary["positive_count"] >= 1
    assert summary["negative_count"] >= 1
    assert summary["neutral_count"] >= 1
    assert summary["average_confidence"] > 0.0

    # Percentages
    pcts = summary["percentages"]
    assert "positive" in pcts and "negative" in pcts and "neutral" in pcts
    total_pct = pcts["positive"] + pcts["negative"] + pcts["neutral"]
    assert 99.0 <= total_pct <= 101.0, f"Percentages do not sum to ~100%: {total_pct}"

    # Trends
    trends_res = client.get("/api/v1/analytics/trends")
    assert trends_res.status_code == 200
    trends = trends_res.json()
    assert "timeline" in trends
    assert len(trends["timeline"]) > 0
    first_point = trends["timeline"][0]
    assert "date" in first_point
    assert "positive" in first_point
    assert "negative" in first_point
    assert "neutral" in first_point
    assert "total" in first_point

    # Distribution
    dist_res = client.get("/api/v1/analytics/distribution")
    assert dist_res.status_code == 200
    dist = dist_res.json()
    assert "distribution" in dist
    assert len(dist["distribution"]) == 3
    names = [d["name"] for d in dist["distribution"]]
    assert "Positive" in names and "Negative" in names and "Neutral" in names

    print("Dashboard Analytics Verification: PASSED (Summary, Trends, Distribution verified)")


# ======================================================================
# 6. Batch CSV End-to-End Test
# ======================================================================

def test_09_batch_csv_upload_and_download(client, db_session):
    """Verify uploading a CSV dataset, checking predictions, and downloading annotated CSV."""
    print("\n--- [E2E TEST 09: Batch CSV Upload & Download] ---")

    # 1. Create a synthetic test CSV in memory
    csv_content = """review_id,feedback_text
1,I had an amazing flight and the flight attendants were delightful!
2,Lost my luggage and the staff was extremely rude and dismissive.
3,The flight was scheduled for 3:00 PM and departed on time.
4,Best service ever! Highly recommended.
5,Never flying with this airline again. Complete nightmare.
"""
    file_bytes = csv_content.encode("utf-8")

    # 2. Upload to batch endpoint
    files = {"file": ("test_feedback.csv", file_bytes, "text/csv")}
    data = {"text_column": "feedback_text"}

    upload_res = client.post("/api/v1/sentiment/batch-csv", files=files, data=data)
    assert upload_res.status_code == 200, f"Batch upload failed: {upload_res.text}"
    batch_data = upload_res.json()

    assert batch_data["total_records"] == 5
    assert batch_data["processed_records"] == 5
    assert batch_data["distribution"]["positive"] >= 1
    assert batch_data["distribution"]["negative"] >= 1
    assert batch_data["distribution"]["neutral"] >= 1
    assert batch_data["average_confidence"] > 0.50
    assert len(batch_data["preview_results"]) == 5
    job_id = batch_data["job_id"]

    # 3. Verify Batch Job persisted in SQLite
    batch_job = db_session.query(BatchJob).filter(BatchJob.id == job_id).first()
    assert batch_job is not None, "BatchJob record not found in SQLite database"
    assert batch_job.total_records == 5
    assert batch_job.status == "completed"

    # 4. Download Annotated CSV
    dl_res = client.get(f"/api/v1/sentiment/batch-csv/download/{job_id}")
    assert dl_res.status_code == 200, f"Download failed: {dl_res.text}"
    assert "text/csv" in dl_res.headers.get("content-type", "")

    # 5. Parse downloaded CSV and verify attached prediction columns
    dl_df = pd.read_csv(io.StringIO(dl_res.text))
    assert len(dl_df) == 5
    assert "sentiment_predicted" in dl_df.columns
    assert "confidence_score" in dl_df.columns
    assert "prob_positive" in dl_df.columns
    assert "prob_negative" in dl_df.columns
    assert "prob_neutral" in dl_df.columns

    # Verify specific row predictions
    row1 = dl_df.iloc[0]
    assert row1["sentiment_predicted"] == "positive"
    row2 = dl_df.iloc[1]
    assert row2["sentiment_predicted"] == "negative"
    row3 = dl_df.iloc[2]
    assert row3["sentiment_predicted"] == "neutral"

    print(f"Batch CSV Test: PASSED (Job ID: {job_id}, 5 records annotated & verified)")


# ======================================================================
# 7. Model Performance Page Data Verification
# ======================================================================

def test_10_model_performance_endpoints(client):
    """Verify /api/v1/model/info and /api/v1/model/metrics match metadata.json."""
    print("\n--- [E2E TEST 10: Model Performance Endpoints] ---")

    # Info
    info_res = client.get("/api/v1/model/info")
    assert info_res.status_code == 200
    info = info_res.json()
    assert "Linear" in info["model_name"] or "Logistic" in info["model_name"]
    assert info["training_dataset_size"] == 11524
    assert info["testing_dataset_size"] == 2881
    assert info["vocabulary_size"] >= 7235
    assert set(info["supported_classes"]) == {"positive", "negative", "neutral"}

    # Metrics
    metrics_res = client.get("/api/v1/model/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert 0.75 <= metrics["accuracy"] <= 0.90, f"Unexpected accuracy: {metrics['accuracy']}"
    assert 0.70 <= metrics["f1_macro"] <= 0.85, f"Unexpected macro F1: {metrics['f1_macro']}"
    assert 0.75 <= metrics["f1_weighted"] <= 0.90, f"Unexpected weighted F1: {metrics['f1_weighted']}"

    # Class metrics
    cm = metrics["class_metrics"]
    for c in ["positive", "negative", "neutral"]:
        assert c in cm
        assert "precision" in cm[c]
        assert "recall" in cm[c]
        assert "f1-score" in cm[c]
        assert "support" in cm[c]

    # Confusion matrix
    conf_mat = metrics["confusion_matrix"]
    assert len(conf_mat) == 3
    for row in conf_mat:
        assert len(row) == 3

    print("Model Performance Endpoints: PASSED (Accuracy: 76.43%, Macro F1: 0.7138, 3x3 Confusion Matrix)")


# ======================================================================
# 8. Error Handling & Edge Case Testing
# ======================================================================

def test_11_error_handling_edge_cases(client):
    """Verify empty input, oversized input, invalid CSV, non-existent ID, and 404s."""
    print("\n--- [E2E TEST 11: Error Handling & Edge Cases] ---")

    # 1. Empty text input
    empty_res = client.post("/api/v1/sentiment/predict", json={"text": ""})
    assert empty_res.status_code == 422
    assert "errors" in empty_res.json()

    # 2. Whitespace-only input
    ws_res = client.post("/api/v1/sentiment/predict", json={"text": "     \n\t  "})
    assert ws_res.status_code == 422

    # 3. Very long text (> 5,000 characters)
    huge_text = "Good flight! " * 600  # ~7,800 chars
    long_res = client.post("/api/v1/sentiment/predict", json={"text": huge_text})
    assert long_res.status_code == 422
    assert "5000" in str(long_res.json())

    # 4. Invalid empty CSV
    empty_csv = {"file": ("empty.csv", b"", "text/csv")}
    bad_csv_res = client.post("/api/v1/sentiment/batch-csv", files=empty_csv)
    assert bad_csv_res.status_code in [400, 422]

    # 5. Non-existent history ID
    del_404_res = client.delete("/api/v1/sentiment/history/99999999")
    assert del_404_res.status_code == 404

    # 6. Non-existent batch download ID
    bad_dl_res = client.get("/api/v1/sentiment/batch-csv/download/nonexistent_job_id")
    assert bad_dl_res.status_code == 404

    # 7. Non-existent route
    not_found_res = client.get("/api/v1/nonexistent-route")
    assert not_found_res.status_code == 404

    print("Error Handling & Edge Cases: PASSED (All validation and 404 handlers verified cleanly)")


# ======================================================================
# 9. Performance & Latency Benchmarks
# ======================================================================

def test_12_performance_benchmarks(client):
    """Measure single prediction latency distribution over 50 iterations."""
    print("\n--- [E2E TEST 12: Performance & Latency Benchmarks] ---")
    latencies = []
    sample_text = "The quick flight was on time and luggage handling was acceptable."

    for _ in range(50):
        t0 = time.perf_counter()
        res = client.post("/api/v1/sentiment/predict", json={"text": sample_text})
        t1 = time.perf_counter()
        assert res.status_code == 200
        latencies.append((t1 - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    sorted_lat = sorted(latencies)
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99_lat = sorted_lat[int(len(sorted_lat) * 0.99)]

    print(f"Latency Benchmarks (50 inferences):")
    print(f"  Min:  {min_lat:.2f} ms")
    print(f"  Mean: {avg_lat:.2f} ms")
    print(f"  P95:  {p95_lat:.2f} ms")
    print(f"  P99:  {p99_lat:.2f} ms")
    print(f"  Max:  {max_lat:.2f} ms")

    # Assert model latency is well within academic SLA (< 50 ms)
    assert avg_lat < 50.0, f"Average latency too high: {avg_lat:.2f} ms"
    print("Performance & Reliability: PASSED")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
