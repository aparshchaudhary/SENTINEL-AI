"""
SENTINEL AI – Backend Integration & API Verification Suite
Academic Project: CSE276 – Artificial Intelligence Foundations

Comprehensive automated testing covering:
1. System Health & Diagnostics
2. Positive Sentiment Prediction
3. Negative Sentiment Prediction
4. Neutral Sentiment Prediction
5. Empty Input Validation
6. Invalid Payload Handling
7. Bulk CSV Upload & Analysis
8. Malformed / Invalid CSV Handling
9. Analytics Summary Aggregation
10. History Querying & Pagination
11. History Record Deletion
12. Model Metadata & Evaluation Benchmarks
13. Database Transaction & Persistence Verification
"""

import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.database.session import Base, get_db
from app.models.domain import SentimentAnalysis, BatchJob, ModelMetrics
from main import app

# Create isolated test database
TEST_DB_PATH = os.path.join(tempfile.gettempdir(), f"sentinel_test_{os.getpid()}.db")
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_test_database():
    """Initializes tables in test database and cleans up after test execution."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ==========================================
# 1. Health & Diagnostics Test
# ==========================================

def test_01_health_endpoint(client):
    """Test 1: Health endpoint returns healthy and dependencies are connected."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True
    assert "version" in data


# ==========================================
# 2-4. Single Text Sentiment Predictions
# ==========================================

def test_02_positive_sentiment_prediction(client):
    """Test 2: Positive text is classified as positive with valid probabilities and DB record."""
    payload = {
        "text": "I absolutely loved this flight experience, fantastic customer service and comfortable seating!"
    }
    response = client.post("/api/v1/sentiment/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5
    assert data["probabilities"]["positive"] > 0.5
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.05
    assert data["id"] is not None
    assert data["processing_time_ms"] >= 0.0


def test_03_negative_sentiment_prediction(client):
    """Test 3: Negative complaint text is classified as negative with valid probabilities."""
    payload = {
        "text": "Horrible experience, delayed for four hours, lost all my luggage, and extremely rude staff."
    }
    response = client.post("/api/v1/sentiment/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.6
    assert data["probabilities"]["negative"] > 0.6
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.05


def test_04_neutral_sentiment_prediction(client):
    """Test 4: Neutral factual question is classified as neutral."""
    payload = {
        "text": "What is the baggage weight allowance for domestic flight 402 departing tomorrow morning?"
    }
    response = client.post("/api/v1/sentiment/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sentiment"] == "neutral"
    assert data["confidence"] > 0.5
    assert data["probabilities"]["neutral"] > 0.5


# ==========================================
# 5-6. Input Validation & Error Handling
# ==========================================

def test_05_empty_text_validation(client):
    """Test 5: Empty and whitespace-only text inputs are rejected with status 422."""
    response = client.post("/api/v1/sentiment/predict", json={"text": "   "})
    assert response.status_code == 422
    data = response.json()
    assert "error_code" in data
    assert data["error_code"] == "VALIDATION_ERROR"


def test_06_invalid_request_handling(client):
    """Test 6: Missing required field or malformed JSON yields 422."""
    response = client.post("/api/v1/sentiment/predict", json={"wrong_field": "Sample text"})
    assert response.status_code == 422


# ==========================================
# 7-8. Batch CSV Analysis & Errors
# ==========================================

def test_07_batch_csv_upload_and_analysis(client):
    """Test 7: Valid CSV is analyzed in bulk, returning distribution and download URL."""
    csv_content = (
        "id,text\n"
        "1,I loved the wonderful flight crew and smooth landing!\n"
        "2,Terrible delay and lost luggage at the terminal.\n"
        "3,Flight 204 departs at 10 AM.\n"
        "4,Outstanding food and great entertainment.\n"
        "5,Horrible customer support and bad seats.\n"
    )

    files = {"file": ("test_feedback.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/api/v1/sentiment/batch-csv", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["job_id"].startswith("job_")
    assert data["total_records"] == 5
    assert data["processed_records"] == 5
    assert "distribution" in data
    assert data["distribution"]["positive"] >= 1
    assert data["distribution"]["negative"] >= 1
    assert "download_url" in data
    assert len(data["preview_results"]) == 5

    # Test download of annotated CSV
    job_id = data["job_id"]
    dl_response = client.get(f"/api/v1/sentiment/batch-csv/download/{job_id}")
    assert dl_response.status_code == 200
    assert "sentiment_predicted" in dl_response.text


def test_08_invalid_csv_handling(client):
    """Test 8: Non-CSV file or empty CSV is rejected with 400 Bad Request."""
    # Test non-csv extension
    files = {"file": ("notes.txt", b"Some random text", "text/plain")}
    response = client.post("/api/v1/sentiment/batch-csv", files=files)
    assert response.status_code == 400

    # Test empty CSV
    empty_file = {"file": ("empty.csv", b"", "text/csv")}
    response = client.post("/api/v1/sentiment/batch-csv", files=empty_file)
    assert response.status_code == 400


# ==========================================
# 9. Analytics API
# ==========================================

def test_09_analytics_summary(client):
    """Test 9: Analytics summary aggregates counts from actual database records."""
    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["total_analyses"] >= 8  # 3 single predictions + 5 from batch
    assert data["positive_count"] >= 1
    assert data["negative_count"] >= 1
    assert "percentages" in data
    assert sum(data["percentages"].values()) == pytest.approx(100.0, abs=0.5)


def test_09b_analytics_trends_and_distribution(client):
    """Test 9b: Trends and distribution endpoints return valid chart data."""
    trends_resp = client.get("/api/v1/analytics/trends")
    assert trends_resp.status_code == 200
    trends_data = trends_resp.json()
    assert len(trends_data["timeline"]) >= 1

    dist_resp = client.get("/api/v1/analytics/distribution")
    assert dist_resp.status_code == 200
    dist_data = dist_resp.json()
    assert len(dist_data["distribution"]) == 3


# ==========================================
# 10-11. History API & Deletion
# ==========================================

def test_10_sentiment_history_pagination(client):
    """Test 10: Sentiment history returns paginated records with newest-first order."""
    # Ensure at least one known query is recorded for isolated test execution
    client.post(
        "/api/v1/sentiment/predict",
        json={"text": "Lost my luggage during the connecting transfer."}
    )

    response = client.get("/api/v1/sentiment/history?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] >= 1
    assert len(data["items"]) >= 1
    assert data["page"] == 1

    # Test search filter
    search_resp = client.get("/api/v1/sentiment/history?search=luggage")
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["total"] >= 1
    assert "luggage" in search_data["items"][0]["input_text"].lower()


def test_11_history_deletion(client):
    """Test 11: Deleting a record removes it from the database safely."""
    # Create a transient record
    pred_resp = client.post(
        "/api/v1/sentiment/predict",
        json={"text": "Temporary record to be deleted."}
    )
    rec_id = pred_resp.json()["id"]

    # Verify existence
    get_resp = client.get(f"/api/v1/sentiment/history/{rec_id}")
    assert get_resp.status_code == 200

    # Delete record
    del_resp = client.delete(f"/api/v1/sentiment/history/{rec_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Verify 404 on subsequent get
    assert client.get(f"/api/v1/sentiment/history/{rec_id}").status_code == 404


# ==========================================
# 12. Model Metadata & Metrics
# ==========================================

def test_12_model_information_and_metrics(client):
    """Test 12: Model info and metrics return genuine metadata and test benchmarks."""
    info_resp = client.get("/api/v1/model/info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert "Linear" in info_data["model_name"] or "Logistic" in info_data["model_name"]
    assert info_data["supported_classes"] == ["negative", "neutral", "positive"]
    assert info_data["vocabulary_size"] >= 7235

    metrics_resp = client.get("/api/v1/model/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert metrics_data["accuracy"] >= 0.76
    assert metrics_data["f1_macro"] >= 0.71
    assert "confusion_matrix" in metrics_data


# ==========================================
# 13. Database Connectivity & Transactions
# ==========================================

def test_13_database_connectivity_and_persistence():
    """Test 13: Direct database session writes, reads, and verifies data integrity."""
    db = TestingSessionLocal()
    try:
        sample_record = SentimentAnalysis(
            input_text="Direct session persistence test",
            cleaned_text="direct session persistence test",
            predicted_sentiment="neutral",
            confidence=0.88,
            positive_probability=0.06,
            negative_probability=0.06,
            neutral_probability=0.88,
            processing_time_ms=1.2,
        )
        db.add(sample_record)
        db.commit()
        db.refresh(sample_record)

        # Retrieve and verify
        retrieved = db.query(SentimentAnalysis).filter(SentimentAnalysis.id == sample_record.id).first()
        assert retrieved is not None
        assert retrieved.input_text == "Direct session persistence test"
        assert retrieved.predicted_sentiment == "neutral"
    finally:
        db.close()


def run_all_tests():
    """Standalone runner for terminal verification."""
    print("=" * 80)
    print("RUNNING SENTINEL AI BACKEND INTEGRATION TEST SUITE")
    print("=" * 80)
    pytest_args = [__file__, "-v", "--tb=short"]
    exit_code = pytest.main(pytest_args)
    return exit_code == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
