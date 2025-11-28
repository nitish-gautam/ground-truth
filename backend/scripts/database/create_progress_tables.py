"""
Database migration script for HS2 Progress Verification tables.

This script creates the necessary tables for progress tracking, point cloud comparison,
and schedule management in the HS2 Infrastructure Intelligence Platform.

Tables created:
- hs2_progress_snapshots: Timeline tracking of physical, cost, and schedule progress
- hs2_point_cloud_comparisons: Results of BIM vs reality capture comparisons
- hs2_schedule_milestones: Milestone tracking with dependency management

Run with: python backend/scripts/database/create_progress_tables.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from app.core.database import db_manager


async def create_progress_tables():
    """Create all progress verification tables."""

    # Initialize database connection
    if db_manager.async_engine is None:
        await db_manager.init_async_engine()

    # Split into individual statements for asyncpg compatibility
    statements = [
        """
    -- ============================================================================
    -- HS2 PROGRESS SNAPSHOTS TABLE
    -- ============================================================================
    -- Timeline tracking of progress measurements for each asset
    -- Links: hs2_assets (asset_id)
    -- Purpose: Store periodic progress snapshots for trending and analysis

    CREATE TABLE IF NOT EXISTS hs2_progress_snapshots (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        asset_id UUID NOT NULL REFERENCES hs2_assets(id) ON DELETE CASCADE,
        snapshot_date TIMESTAMP NOT NULL,

        -- Progress Metrics (0-100%)
        physical_progress_pct DECIMAL(5,2) CHECK (physical_progress_pct >= 0 AND physical_progress_pct <= 100),
        cost_progress_pct DECIMAL(5,2) CHECK (cost_progress_pct >= 0 AND cost_progress_pct <= 100),
        schedule_progress_pct DECIMAL(5,2) CHECK (schedule_progress_pct >= 0 AND schedule_progress_pct <= 100),

        -- Earned Value Management Metrics
        planned_value DECIMAL(15,2),           -- PV = Budgeted cost of work scheduled
        earned_value DECIMAL(15,2),            -- EV = Budgeted cost of work performed
        actual_cost DECIMAL(15,2),             -- AC = Actual cost of work performed
        cost_variance DECIMAL(15,2),           -- CV = EV - AC
        schedule_variance DECIMAL(15,2),       -- SV = EV - PV
        cost_performance_index DECIMAL(5,3),   -- CPI = EV / AC
        schedule_performance_index DECIMAL(5,3), -- SPI = EV / PV

        -- Point Cloud Data
        point_cloud_file_path TEXT,
        deviation_score DECIMAL(10,2),         -- Average deviation in mm

        -- Anomalies and Issues
        anomalies JSONB DEFAULT '[]'::jsonb,   -- [{"type": "cost_overrun", "severity": "high", "description": "..."}]

        -- Metadata
        data_source VARCHAR(50) DEFAULT 'manual',  -- 'manual', 'point_cloud', 'survey', 'bim_comparison'
        confidence_score DECIMAL(5,2) DEFAULT 100.00,
        notes TEXT,
        created_by UUID,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_progress_asset_date ON hs2_progress_snapshots(asset_id, snapshot_date DESC);
    CREATE INDEX IF NOT EXISTS idx_progress_snapshot_date ON hs2_progress_snapshots(snapshot_date DESC);
    CREATE INDEX IF NOT EXISTS idx_progress_anomalies ON hs2_progress_snapshots USING GIN(anomalies);


    -- ============================================================================
    -- HS2 POINT CLOUD COMPARISONS TABLE
    -- ============================================================================
    -- Stores results of BIM model vs reality capture (LiDAR/drone) comparisons
    -- Links: hs2_assets (asset_id)
    -- Purpose: Track automated progress verification from site scans

    CREATE TABLE IF NOT EXISTS hs2_point_cloud_comparisons (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        asset_id UUID NOT NULL REFERENCES hs2_assets(id) ON DELETE CASCADE,

        -- File References (stored in MinIO)
        baseline_file_path TEXT NOT NULL,      -- BIM model or baseline scan
        current_file_path TEXT NOT NULL,       -- Current site scan (LAS/LAZ)
        comparison_date TIMESTAMP NOT NULL,

        -- Volume Analysis
        volume_difference_m3 DECIMAL(15,3),    -- Actual vs planned volume
        volume_planned_m3 DECIMAL(15,3),
        volume_actual_m3 DECIMAL(15,3),

        -- Surface Deviation Analysis
        surface_deviation_avg DECIMAL(10,3),   -- Average deviation in mm
        surface_deviation_max DECIMAL(10,3),   -- Maximum deviation in mm
        surface_deviation_min DECIMAL(10,3),   -- Minimum deviation in mm
        surface_deviation_std DECIMAL(10,3),   -- Standard deviation in mm

        -- Completion Metrics
        completion_percentage DECIMAL(5,2) CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
        points_within_tolerance_pct DECIMAL(5,2),  -- % of points within tolerance (e.g., ±50mm)
        tolerance_threshold_mm DECIMAL(10,2) DEFAULT 50.00,

        -- Visualization Data
        heatmap_data JSONB,                    -- {"points": [...], "colors": [...], "bounds": {...}}
        hotspots JSONB DEFAULT '[]'::jsonb,    -- [{"location": [x,y,z], "deviation": 150.5, "severity": "high"}]

        -- Processing Metadata
        processing_time_seconds INTEGER,
        point_count_baseline INTEGER,
        point_count_current INTEGER,
        algorithm_version VARCHAR(20) DEFAULT 'v1.0',

        -- Quality Metrics
        confidence_score DECIMAL(5,2) DEFAULT 95.00,
        quality_flags JSONB DEFAULT '[]'::jsonb,  -- [{"flag": "incomplete_scan", "description": "..."}]

        -- Metadata
        processed_by VARCHAR(100),             -- Service/user that ran comparison
        notes TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_pointcloud_asset_date ON hs2_point_cloud_comparisons(asset_id, comparison_date DESC);
    CREATE INDEX IF NOT EXISTS idx_pointcloud_comparison_date ON hs2_point_cloud_comparisons(comparison_date DESC);
    CREATE INDEX IF NOT EXISTS idx_pointcloud_completion ON hs2_point_cloud_comparisons(completion_percentage);


    -- ============================================================================
    -- HS2 SCHEDULE MILESTONES TABLE
    -- ============================================================================
    -- Milestone tracking with dependency management and critical path analysis
    -- Links: hs2_assets (asset_id)
    -- Purpose: Schedule management, Gantt charts, and delay prediction

    CREATE TABLE IF NOT EXISTS hs2_schedule_milestones (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        asset_id UUID NOT NULL REFERENCES hs2_assets(id) ON DELETE CASCADE,

        -- Milestone Details
        milestone_name VARCHAR(255) NOT NULL,
        milestone_code VARCHAR(50),            -- e.g., "VIA-001-FOUND"
        milestone_type VARCHAR(50),            -- 'foundation', 'structure', 'completion', 'inspection'
        description TEXT,

        -- Schedule Dates
        planned_date TIMESTAMP,                -- Original planned date
        baseline_date TIMESTAMP,               -- Baseline after approved changes
        forecast_date TIMESTAMP,               -- ML-predicted completion date
        actual_date TIMESTAMP,                 -- Actual completion date
        early_start TIMESTAMP,                 -- Earliest possible start
        late_start TIMESTAMP,                  -- Latest allowable start
        early_finish TIMESTAMP,                -- Earliest possible finish
        late_finish TIMESTAMP,                 -- Latest allowable finish

        -- Duration
        planned_duration_days INTEGER,
        actual_duration_days INTEGER,

        -- Status
        status VARCHAR(50) DEFAULT 'not_started',  -- 'not_started', 'in_progress', 'completed', 'delayed', 'cancelled'
        completion_percentage DECIMAL(5,2) DEFAULT 0.00 CHECK (completion_percentage >= 0 AND completion_percentage <= 100),

        -- Dependencies (JSONB for flexibility)
        predecessors JSONB DEFAULT '[]'::jsonb,  -- [{"milestone_id": "uuid", "type": "finish_to_start", "lag_days": 0}]
        successors JSONB DEFAULT '[]'::jsonb,    -- [{"milestone_id": "uuid", "type": "finish_to_start"}]

        -- Critical Path Analysis
        is_critical_path BOOLEAN DEFAULT false,
        float_days INTEGER DEFAULT 0,          -- Total float (slack)

        -- Variance Analysis
        schedule_variance_days INTEGER,        -- Actual vs planned (negative = delay)
        variance_reason TEXT,

        -- Risk and Issues
        risk_level VARCHAR(20) DEFAULT 'low',  -- 'low', 'medium', 'high', 'critical'
        risk_factors JSONB DEFAULT '[]'::jsonb,  -- [{"factor": "weather", "impact": "2 days delay"}]

        -- Resource Allocation
        assigned_contractor VARCHAR(255),
        estimated_cost DECIMAL(15,2),
        actual_cost DECIMAL(15,2),

        -- Metadata
        created_by UUID,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_milestone_asset ON hs2_schedule_milestones(asset_id);
    CREATE INDEX IF NOT EXISTS idx_milestone_dates ON hs2_schedule_milestones(planned_date, actual_date);
    CREATE INDEX IF NOT EXISTS idx_milestone_status ON hs2_schedule_milestones(status);
    CREATE INDEX IF NOT EXISTS idx_milestone_critical_path ON hs2_schedule_milestones(is_critical_path) WHERE is_critical_path = true;
    CREATE INDEX IF NOT EXISTS idx_milestone_type ON hs2_schedule_milestones(milestone_type);


    -- ============================================================================
    -- TRIGGER FOR UPDATED_AT TIMESTAMPS
    -- ============================================================================

    CREATE OR REPLACE FUNCTION update_progress_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS update_hs2_progress_snapshots_updated_at ON hs2_progress_snapshots;
    CREATE TRIGGER update_hs2_progress_snapshots_updated_at
        BEFORE UPDATE ON hs2_progress_snapshots
        FOR EACH ROW
        EXECUTE FUNCTION update_progress_updated_at();

    DROP TRIGGER IF EXISTS update_hs2_point_cloud_comparisons_updated_at ON hs2_point_cloud_comparisons;
    CREATE TRIGGER update_hs2_point_cloud_comparisons_updated_at
        BEFORE UPDATE ON hs2_point_cloud_comparisons
        FOR EACH ROW
        EXECUTE FUNCTION update_progress_updated_at();

    DROP TRIGGER IF EXISTS update_hs2_schedule_milestones_updated_at ON hs2_schedule_milestones;
    CREATE TRIGGER update_hs2_schedule_milestones_updated_at
        BEFORE UPDATE ON hs2_schedule_milestones
        FOR EACH ROW
        EXECUTE FUNCTION update_progress_updated_at();


    -- ============================================================================
    -- GRANT PERMISSIONS (adjust as needed for your setup)
    -- ============================================================================

    -- GRANT ALL PRIVILEGES ON hs2_progress_snapshots TO your_app_user;
    -- GRANT ALL PRIVILEGES ON hs2_point_cloud_comparisons TO your_app_user;
    -- GRANT ALL PRIVILEGES ON hs2_schedule_milestones TO your_app_user;
    """

    async with db_manager.async_engine.begin() as conn:
        print("🔧 Creating HS2 Progress Verification tables...")
        await conn.execute(text(create_tables_sql))
        print("✅ Tables created successfully:")
        print("   - hs2_progress_snapshots")
        print("   - hs2_point_cloud_comparisons")
        print("   - hs2_schedule_milestones")
        print("\n📊 Indexes created:")
        print("   - Progress: asset_date, snapshot_date, anomalies (GIN)")
        print("   - Point Cloud: asset_date, comparison_date, completion")
        print("   - Milestones: asset, dates, status, critical_path, type")
        print("\n🔄 Triggers created:")
        print("   - Auto-update updated_at timestamps on all tables")


async def verify_tables():
    """Verify that tables were created successfully."""
    verify_sql = """
    SELECT
        table_name,
        (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
    FROM information_schema.tables t
    WHERE table_schema = 'public'
    AND table_name IN ('hs2_progress_snapshots', 'hs2_point_cloud_comparisons', 'hs2_schedule_milestones')
    ORDER BY table_name;
    """

    async for session in db_manager.get_async_session():
        result = await session.execute(text(verify_sql))
        tables = result.fetchall()

        print("\n🔍 Verification:")
        if len(tables) == 3:
            print("✅ All 3 tables created successfully:")
            for table_name, column_count in tables:
                print(f"   - {table_name}: {column_count} columns")
        else:
            print(f"⚠️  Only {len(tables)}/3 tables found")
            for table_name, column_count in tables:
                print(f"   - {table_name}: {column_count} columns")


async def main():
    """Main execution function."""
    print("=" * 80)
    print("HS2 PROGRESS VERIFICATION - DATABASE MIGRATION")
    print("=" * 80)
    print("\nThis script will create the following tables:")
    print("  1. hs2_progress_snapshots")
    print("  2. hs2_point_cloud_comparisons")
    print("  3. hs2_schedule_milestones")
    print("\n" + "=" * 80 + "\n")

    try:
        await create_progress_tables()
        await verify_tables()
        print("\n" + "=" * 80)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Verify tables in your database client")
        print("  2. Create SQLAlchemy models (backend/app/models/progress.py)")
        print("  3. Create Pydantic schemas (backend/app/schemas/progress.py)")
        print("  4. Implement API endpoints (backend/app/api/v1/endpoints/progress_verification.py)")

    except Exception as e:
        print(f"\n❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
