"""
SENTINEL AI – Root ASGI Proxy Entrypoint for Cloud Deployment
Academic Project: CSE276 – Artificial Intelligence Foundations

Enables deployment on platforms like Render, Railway, or Heroku where
the start command `uvicorn main:app` may be executed from the repository root
without specifying `backend` as the working directory.
"""

import os
import sys

# Ensure both project root and backend directory are in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the actual FastAPI application from backend.main
from backend.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("ENVIRONMENT", "development").lower() == "development"
    uvicorn.run("main:app", host=host, port=port, reload=reload_flag)
