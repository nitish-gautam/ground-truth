"""
Database Table Dropping Script
===============================

Drops all database tables. USE WITH CAUTION!
"""
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def drop_tables():
    """Drop all database tables."""
    print("⚠️  WARNING: This will drop all database tables!")
    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() != 'yes':
        print("❌ Operation cancelled.")
        return

    print("\nDropping database tables...")

    # Initialize sync engine
    print("Initializing database connection...")
    db_manager.init_sync_engine()

    print(f"Found {len(Base.metadata.tables)} tables to drop:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")

    # Drop all tables
    print("\nDropping tables...")
    Base.metadata.drop_all(bind=db_manager.sync_engine)
    print("\n✅ All tables dropped successfully!")


if __name__ == "__main__":
    drop_tables()
