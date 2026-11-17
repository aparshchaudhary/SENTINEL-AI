"""
SENTINEL AI – Database Engine & Session Management
Academic Project: CSE276 – Artificial Intelligence Foundations
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.core.config import settings

# Normalize PostgreSQL URL for SQLAlchemy 2.0 (Render provides postgres:// by default)
db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# SQLite connection configuration with multithread support
engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request.
    Automatically closes session upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Creates database tables if they do not exist.
    """
    import app.models.domain  # noqa: F401 - Register models with Base
    Base.metadata.create_all(bind=engine)
