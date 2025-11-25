"""
Database Table Listing Script
==============================

Lists all tables, columns, and indexes in the database.
"""
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import inspect
from app.core.database import db_manager


def list_tables():
    """List all database tables with details."""
    print("Listing database tables...")

    # Initialize sync engine
    db_manager.init_sync_engine()

    # Get inspector
    inspector = inspect(db_manager.sync_engine)

    # Get all table names
    table_names = inspector.get_table_names()

    print(f"\n📊 Found {len(table_names)} tables in database:\n")

    for table_name in sorted(table_names):
        print(f"📋 Table: {table_name}")

        # Get columns
        columns = inspector.get_columns(table_name)
        print(f"   Columns ({len(columns)}):")
        for col in columns:
            nullable = "NULL" if col['nullable'] else "NOT NULL"
            col_type = str(col['type'])
            print(f"     • {col['name']}: {col_type} {nullable}")

        # Get indexes
        indexes = inspector.get_indexes(table_name)
        if indexes:
            print(f"   Indexes ({len(indexes)}):")
            for idx in indexes:
                unique = "UNIQUE" if idx['unique'] else ""
                print(f"     • {idx['name']}: {idx['column_names']} {unique}")

        # Get foreign keys
        fks = inspector.get_foreign_keys(table_name)
        if fks:
            print(f"   Foreign Keys ({len(fks)}):")
            for fk in fks:
                print(f"     • {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")

        print()


if __name__ == "__main__":
    list_tables()
