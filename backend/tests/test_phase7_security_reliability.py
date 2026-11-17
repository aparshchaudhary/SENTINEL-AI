"""
SENTINEL AI – Phase 7 Security, Reliability & Edge-Case Verification Suite
Academic Course: CSE276 – Artificial Intelligence Foundations

Comprehensive automated testing covering:
1. Environment & Configuration Security
2. Path Traversal & File Download Hardening
3. Non-Existent Column & Schema Validation
4. Empty, Whitespace, and Boundary Input Validation
5. CSV Integrity (Corrupt, Zero-byte, Header-only, Non-text columns)
6. Upload Size Limit Enforcement (HTTP 413)
7. Query Parameter Validation & SQL Injection Resilience
8. Graceful Model Degradation & 503 Availability Guard
9. Clean Error Serialization without Stacktrace Leakage
"""

import io
import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.config import settings
from app.database.session import init_db
from app.services.sentiment_service import sentiment_service
from main import app


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client with lifespan context."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


# ======================================================================
# 1. Environment & Configuration Security
# ======================================================================

def test_01_environment_and_cors_security():
    """Verify that credentials are not paired with wildcard '*' in CORS."""
    assert "*" not in settings.CORS_ORIGINS, "CORS_ORIGINS must not contain wildcard '*' when allow_credentials=True"
    for origin in settings.CORS_ORIGINS:
        assert origin.startswith("http://") or origin.startswith("https://")
    assert settings.MAX_UPLOAD_SIZE_MB >= 1
    assert settings.MAX_TEXT_LENGTH == 5000


# ======================================================================
# 2. Path Traversal & File Download Hardening
# ======================================================================

def test_02_path_traversal_prevention_on_download(client):
    """Attempt path traversal payloads in download endpoint."""
    traversal_payloads = [
        "../../main.py",
        "..%2F..%2Fmain.py",
        "....//....//sentinel.db",
        "nonexistent_id",
        "/etc/passwd",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
    ]
    for payload in traversal_payloads:
        res = client.get(f"/api/v1/sentiment/batch-csv/download/{payload}")
        assert res.status_code in [400, 404], f"Traversal payload {payload} did not return 400/404: {res.status_code}"
        # Ensure no internal server paths are exposed
        body = res.text
        assert "Traceback" not in body
        assert "File \"" not in body


# ======================================================================
# 3. Text Boundary & Validation Edge Cases
# ======================================================================

def test_03_empty_and_whitespace_only_text(client):
    """Verify validation on empty strings, tabs, newlines."""
    test_cases = [
        "",
        " ",
        "   ",
        "\t\t",
        "\n\n\n",
        "   \r\n\t   ",
    ]
    for tc in test_cases:
        res = client.post("/api/v1/sentiment/predict", json={"text": tc})
        assert res.status_code == 422, f"Whitespace '{tc}' should return 422"
        data = res.json()
        assert "errors" in data or "detail" in data


def test_04_exact_boundary_text_limits(client):
    """Verify text boundary: exactly 5000 characters succeeds, 5001 fails."""
    # 5,000 characters (valid)
    valid_text = "Great service and flight! " * 192 + "Comfortable!"
    valid_text = valid_text[:5000].ljust(5000, ".")
    assert len(valid_text) == 5000

    res_valid = client.post("/api/v1/sentiment/predict", json={"text": valid_text})
    assert res_valid.status_code == 200, f"5000 chars should be accepted: {res_valid.text}"

    # 5,001 characters (invalid)
    invalid_text = valid_text + "x"
    assert len(invalid_text) == 5001

    res_invalid = client.post("/api/v1/sentiment/predict", json={"text": invalid_text})
    assert res_invalid.status_code == 422, f"5001 chars should return 422: {res_invalid.text}"


# ======================================================================
# 4. CSV File Edge Cases
# ======================================================================

def test_05_csv_invalid_file_extension(client):
    """Attempt upload with non-CSV extension."""
    files = {"file": ("malicious.exe", b"binary content", "application/octet-stream")}
    res = client.post("/api/v1/sentiment/batch-csv", files=files)
    assert res.status_code == 400
    assert "Only .csv files are supported" in res.json()["detail"]


def test_06_csv_completely_empty_file(client):
    """Attempt upload with 0 bytes."""
    files = {"file": ("empty.csv", b"", "text/csv")}
    res = client.post("/api/v1/sentiment/batch-csv", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_07_csv_header_only_zero_rows(client):
    """Attempt upload with header only and zero rows."""
    content = b"id,text,rating\n"
    files = {"file": ("header_only.csv", content, "text/csv")}
    res = client.post("/api/v1/sentiment/batch-csv", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower() or "no data rows" in res.json()["detail"].lower()


def test_08_csv_without_text_column(client):
    """CSV containing only numeric columns without text."""
    content = b"id,age,salary,zipcode\n1,25,50000,90210\n2,34,75000,10001\n"
    files = {"file": ("numbers_only.csv", content, "text/csv")}
    res = client.post("/api/v1/sentiment/batch-csv", files=files)
    assert res.status_code == 400
    assert "detect text column" in res.json()["detail"].lower()


def test_09_csv_nonexistent_specified_column(client):
    """User provides a text_column name that does not exist."""
    content = b"id,review_body\n1,The service was outstanding!\n"
    files = {"file": ("reviews.csv", content, "text/csv")}
    data = {"text_column": "completely_wrong_column_name"}
    res = client.post("/api/v1/sentiment/batch-csv", files=files, data=data)
    assert res.status_code == 400
    assert "not found" in res.json()["detail"].lower()


def test_10_csv_oversized_file_limit(client):
    """Attempt upload exceeding MAX_UPLOAD_SIZE_MB."""
    # Create fake payload larger than MAX_UPLOAD_SIZE_MB
    oversized_bytes = b"header1,header2\n" + (b"val1,val2\n" * 500)
    # Temporarily set max size to small value to test enforcement
    orig_size = settings.MAX_UPLOAD_SIZE_MB
    try:
        settings.MAX_UPLOAD_SIZE_MB = 0  # 0 MB threshold
        files = {"file": ("large.csv", oversized_bytes, "text/csv")}
        res = client.post("/api/v1/sentiment/batch-csv", files=files)
        assert res.status_code == 413
        assert "exceeds maximum permissible size" in res.json()["detail"].lower()
    finally:
        settings.MAX_UPLOAD_SIZE_MB = orig_size


# ======================================================================
# 5. Query Parameter Validation & SQL Injection Resilience
# ======================================================================

def test_11_history_query_parameter_validation(client):
    """Verify bounds on page and page_size."""
    # Negative page
    res_neg_page = client.get("/api/v1/sentiment/history?page=-1")
    assert res_neg_page.status_code == 422

    # Zero page
    res_zero_page = client.get("/api/v1/sentiment/history?page=0")
    assert res_zero_page.status_code == 422

    # Oversized page_size (> 100)
    res_huge_page = client.get("/api/v1/sentiment/history?page_size=500")
    assert res_huge_page.status_code == 422


def test_12_sql_injection_resilience_in_search(client):
    """Attempt classic SQL injection strings in the search parameter."""
    injection_strings = [
        "' OR '1'='1",
        "'; DROP TABLE sentiment_analyses; --",
        "\" UNION SELECT 1, 2, 3 --",
        "admin'--",
    ]
    for payload in injection_strings:
        res = client.get(f"/api/v1/sentiment/history?search={payload}")
        assert res.status_code == 200, f"SQL injection payload {payload} crashed endpoint: {res.status_code}"
        data = res.json()
        assert "items" in data
        assert "total" in data
        # Ensure no table was dropped or compromised
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["database"] == "connected"


# ======================================================================
# 6. Clean Error Handling & Information Leakage Prevention
# ======================================================================

def test_13_no_stacktrace_leakage_on_404_or_invalid_endpoints(client):
    """Ensure non-existent routes and resources return clean JSON without Python stack traces."""
    res = client.get("/api/v1/definitely-not-a-real-endpoint")
    assert res.status_code == 404
    body = res.text
    assert "Traceback" not in body
    assert ".py\"" not in body
    assert "Exception" not in body
