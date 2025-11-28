"""
Create HS2 Progress Verification Tables Script
===============================================

Creates progress tracking, point cloud comparison, and schedule milestone tables
using SQLAlchemy models.

Run with: python backend/scripts/database/create_progress_tables_v2.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.models.base import Base
from app.core.database import db_manager

# Import HS2 models first (needed for foreign key references)
from app.models.hs2 import HS2Asset

# Import progress models to register them with Base.metadata
from app.models.progress import (
    HS2ProgressSnapshot,
    HS2PointCloudComparison,
    HS2ScheduleMilestone,
)


def create_progress_tables():
    """Create HS2 Progress Verification database tables."""
    logger.info("=" * 80)
    logger.info("HS2 PROGRESS VERIFICATION - DATABASE MIGRATION")
    logger.info("=" * 80)

    # Initialize sync engine
    logger.info("\n🔧 Initializing database connection...")
    db_manager.init_sync_engine()

    # Get progress table names
    progress_tables = [
        "hs2_progress_snapshots",
        "hs2_point_cloud_comparisons",
        "hs2_schedule_milestones",
    ]

    logger.info(f"\n📋 Tables to create:")
    for table_name in progress_tables:
        logger.info(f"   - {table_name}")

    # Create all tables
    logger.info("\n🔨 Creating tables...")
    try:
        Base.metadata.create_all(
            bind=db_manager.sync_engine,
            checkfirst=True,
            tables=[
                Base.metadata.tables["hs2_progress_snapshots"],
                Base.metadata.tables["hs2_point_cloud_comparisons"],
                Base.metadata.tables["hs2_schedule_milestones"],
            ]
        )
        logger.info("✅ All progress verification tables created successfully!")

        logger.info("\n📊 Table Details:")
        logger.info("   - hs2_progress_snapshots: Timeline progress tracking")
        logger.info("   - hs2_point_cloud_comparisons: BIM vs reality analysis")
        logger.info("   - hs2_schedule_milestones: Gantt chart & critical path")

        logger.info("\n🔗 Relationships:")
        logger.info("   - All tables linked to hs2_assets via asset_id (CASCADE DELETE)")

        logger.info("\n📈 Key Features:")
        logger.info("   - Earned Value Management (EVM) metrics")
        logger.info("   - Point cloud deviation analysis")
        logger.info("   - Critical path calculation")
        logger.info("   - Anomaly detection (JSONB)")
        logger.info("   - Heatmap visualization data")

    except Exception as e:
        logger.error(f"❌ Failed to create tables: {str(e)}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    logger.info("\n📝 Next Steps:")
    logger.info("   1. Create Pydantic schemas (backend/app/schemas/progress.py)")
    logger.info("   2. Implement API endpoints (backend/app/api/v1/endpoints/progress_verification.py)")
    logger.info("   3. Create progress analyzer service (backend/app/services/progress_analyzer.py)")
    logger.info("   4. Build frontend components (frontend/src/components/hs2/progress/)")


if __name__ == "__main__":
    create_progress_tables()
