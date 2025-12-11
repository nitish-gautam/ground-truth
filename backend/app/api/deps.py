"""
API Dependencies
================

Shared dependencies for API endpoints including database sessions,
authentication, and service providers.
"""

from typing import Generator
from sqlalchemy.orm import Session
from app.core.database import db_manager


def get_sync_db() -> Generator[Session, None, None]:
    """
    Dependency to get synchronous database session.

    Use this for endpoints that need sync database access
    (e.g., when using service classes that require sync sessions).

    Yields:
        SQLAlchemy Session object

    Example:
        @router.get("/example")
        def get_example(db: Session = Depends(get_sync_db)):
            # Use db session
            pass
    """
    db = db_manager.get_sync_session()
    try:
        yield db
    finally:
        db.close()
