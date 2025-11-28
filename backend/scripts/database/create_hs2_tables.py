"""
Create HS2 Tables Script
========================

Creates HS2 Assurance Intelligence database tables.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.models.base import Base
from app.core.database import db_manager

# Import HS2 models to register them with Base.metadata
from app.models.hs2 import (
    HS2Asset,
    HS2Deliverable,
    HS2Cost,
    HS2Certificate,
    HS2Rule,
    HS2Evaluation,
)


def create_hs2_tables():
    """Create HS2 database tables."""
    logger.info("=" * 80)
    logger.info("Creating HS2 Assurance Intelligence Tables")
    logger.info("=" * 80)
    
    # Initialize sync engine
    logger.info("Initializing database connection...")
    db_manager.init_sync_engine()
    
    # Get HS2 table names
    hs2_tables = [
        "hs2_assets",
        "hs2_deliverables",
        "hs2_costs",
        "hs2_certificates",
        "hs2_taem_rules",
        "hs2_evaluations",
    ]
    
    logger.info(f"\nTables to create:")
    for table_name in hs2_tables:
        logger.info(f"  - {table_name}")
    
    # Create all tables
    logger.info("\nCreating tables...")
    try:
        Base.metadata.create_all(bind=db_manager.sync_engine, checkfirst=True)
        logger.info("✅ All HS2 tables created successfully!")
    except Exception as e:
        logger.error(f"Failed to create tables: {str(e)}")
        raise
    
    logger.info("=" * 80)


if __name__ == "__main__":
    create_hs2_tables()
