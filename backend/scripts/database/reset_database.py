"""
Database Reset Script
=====================

Drops all tables and recreates them. USE WITH CAUTION!
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


def reset_database():
    """Drop and recreate all database tables."""
    print("⚠️  WARNING: This will DROP and RECREATE all database tables!")
    print("⚠️  All data will be LOST!")
    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() != 'yes':
        print("❌ Operation cancelled.")
        return

    # Initialize sync engine
    print("\nInitializing database connection...")
    db_manager.init_sync_engine()

    # Drop tables
    print(f"\n1. Dropping {len(Base.metadata.tables)} existing tables...")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")
    Base.metadata.drop_all(bind=db_manager.sync_engine)
    print("✅ Tables dropped")

    # Create tables
    print(f"\n2. Creating {len(Base.metadata.tables)} tables...")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")
    Base.metadata.create_all(bind=db_manager.sync_engine)
    print("✅ Tables created")

    print("\n✅ Database reset complete!")


if __name__ == "__main__":
    reset_database()
