"""
Database Table Creation Script
===============================

Creates all database tables from SQLAlchemy models.
"""
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.base import Base
from app.core.database import db_manager

# Import all models to register them with Base.metadata
from app.models import (
    gpr_data,
    environmental,
    validation,
    utilities,
    ml_analytics
)


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")

    # Initialize sync engine
    print("Initializing database connection...")
    db_manager.init_sync_engine()

    print(f"Found {len(Base.metadata.tables)} tables to create:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")

    # Create all tables
    print("\nCreating tables...")
    Base.metadata.create_all(bind=db_manager.sync_engine)
    print("\n✅ All tables created successfully!")


if __name__ == "__main__":
    create_tables()
