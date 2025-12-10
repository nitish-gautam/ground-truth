"""
HS2 Progress Assurance - Progress Tracking Endpoints

Handles progress snapshots, time-series tracking, and dashboard data.
"""

from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from uuid import UUID

from app.core.database import get_async_session
from app.schemas.progress import (
    ProgressSnapshotCreate,
    ProgressSnapshotResponse,
    ProgressSnapshotListResponse,
    ProgressDashboardResponse,
    ProgressTrendResponse
)

router = APIRouter(prefix="/progress", tags=["Progress Tracking"])


@router.post("/snapshots", response_model=ProgressSnapshotResponse, status_code=201)
async def create_progress_snapshot(
    snapshot: ProgressSnapshotCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Create a new progress snapshot

    This captures the state of a project at a specific point in time,
    including LiDAR scans, hyperspectral analysis, and BIM comparison results.
    """
    try:
        query = text("""
            INSERT INTO progress_snapshots (
                id, project_id, snapshot_date, snapshot_name,
                lidar_scan_id, bim_model_id, hyperspectral_scan_id,
                percent_complete, completed_volume_m3, planned_volume_m3,
                schedule_variance_days, quality_score, defects_detected,
                critical_issues, created_at
            ) VALUES (
                gen_random_uuid(), :project_id, :snapshot_date, :snapshot_name,
                :lidar_scan_id, :bim_model_id, :hyperspectral_scan_id,
                :percent_complete, :completed_volume_m3, :planned_volume_m3,
                :schedule_variance_days, :quality_score, :defects_detected,
                :critical_issues, NOW()
            )
            RETURNING *
        """)

        result = await session.execute(query, snapshot.dict())
        await session.commit()

        row = result.fetchone()
        return ProgressSnapshotResponse.from_orm(row)

    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {str(e)}")


@router.get("/snapshots", response_model=ProgressSnapshotListResponse)
async def list_progress_snapshots(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    List progress snapshots with optional filtering

    Returns snapshots ordered by date (most recent first).
    Supports filtering by project and date range.
    """
    try:
        # Build dynamic query based on filters
        where_clauses = []
        params = {"limit": limit, "offset": offset}

        if project_id:
            where_clauses.append("project_id = :project_id")
            params["project_id"] = project_id
        if start_date:
            where_clauses.append("snapshot_date >= :start_date")
            params["start_date"] = start_date
        if end_date:
            where_clauses.append("snapshot_date <= :end_date")
            params["end_date"] = end_date

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get total count
        count_query = text(f"SELECT COUNT(*) FROM progress_snapshots WHERE {where_sql}")
        count_result = await session.execute(count_query, params)
        total = count_result.scalar()

        # Get snapshots
        query = text(f"""
            SELECT
                ps.*,
                p.project_name,
                bm.model_name as bim_model_name,
                pls.point_count as lidar_point_count,
                hs.band_count as hyperspectral_band_count
            FROM progress_snapshots ps
            LEFT JOIN projects p ON ps.project_id = p.id
            LEFT JOIN bim_models bm ON ps.bim_model_id = bm.id
            LEFT JOIN progress_lidar_scans pls ON ps.lidar_scan_id = pls.id
            LEFT JOIN hyperspectral_scans hs ON ps.hyperspectral_scan_id = hs.id
            WHERE {where_sql}
            ORDER BY ps.snapshot_date DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await session.execute(query, params)
        rows = result.fetchall()

        snapshots = [ProgressSnapshotResponse.from_orm(row) for row in rows]

        return ProgressSnapshotListResponse(
            snapshots=snapshots,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {str(e)}")


@router.get("/snapshots/{snapshot_id}", response_model=ProgressSnapshotResponse)
async def get_progress_snapshot(
    snapshot_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get a specific progress snapshot by ID

    Returns detailed information including related scans and models.
    """
    try:
        query = text("""
            SELECT
                ps.*,
                p.project_name,
                p.project_code,
                bm.model_name as bim_model_name,
                bm.element_count as bim_element_count,
                pls.point_count as lidar_point_count,
                pls.scan_quality as lidar_quality,
                hs.band_count as hyperspectral_band_count,
                hs.image_quality_score as hyperspectral_quality
            FROM progress_snapshots ps
            LEFT JOIN projects p ON ps.project_id = p.id
            LEFT JOIN bim_models bm ON ps.bim_model_id = bm.id
            LEFT JOIN progress_lidar_scans pls ON ps.lidar_scan_id = pls.id
            LEFT JOIN hyperspectral_scans hs ON ps.hyperspectral_scan_id = hs.id
            WHERE ps.id = :snapshot_id
        """)

        result = await session.execute(query, {"snapshot_id": snapshot_id})
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        return ProgressSnapshotResponse.from_orm(row)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot: {str(e)}")


@router.get("/dashboard", response_model=ProgressDashboardResponse)
async def get_progress_dashboard(
    project_id: str = Query(..., description="Project ID"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get unified progress dashboard data

    Returns aggregated metrics for the latest snapshot plus trend data.
    Perfect for real-time dashboard displays.
    """
    try:
        # Get latest snapshot
        latest_query = text("""
            SELECT
                ps.*,
                p.project_name,
                p.location,
                COUNT(mqa.id) as total_material_assessments,
                COUNT(CASE WHEN mqa.meets_specification = true THEN 1 END) as passed_assessments,
                AVG(mqa.quality_score) as avg_material_quality,
                COUNT(pda.id) as total_elements_analyzed,
                COUNT(CASE WHEN pda.within_tolerance = true THEN 1 END) as elements_within_tolerance,
                AVG(pda.mean_deviation_mm) as avg_deviation_mm
            FROM progress_snapshots ps
            LEFT JOIN projects p ON ps.project_id = p.id
            LEFT JOIN material_quality_assessments mqa ON ps.hyperspectral_scan_id = mqa.scan_id
            LEFT JOIN bim_lidar_alignments bla ON ps.lidar_scan_id = bla.lidar_scan_id
            LEFT JOIN progress_deviation_analysis pda ON bla.id = pda.alignment_id
            WHERE ps.project_id = :project_id
            GROUP BY ps.id, p.project_name, p.location
            ORDER BY ps.snapshot_date DESC
            LIMIT 1
        """)

        result = await session.execute(latest_query, {"project_id": project_id})
        latest = result.fetchone()

        if not latest:
            raise HTTPException(status_code=404, detail="No snapshots found for project")

        # Get historical trend (last 6 months)
        trend_query = text("""
            SELECT
                snapshot_date,
                percent_complete,
                quality_score,
                schedule_variance_days
            FROM progress_snapshots
            WHERE project_id = :project_id
              AND snapshot_date >= CURRENT_DATE - INTERVAL '6 months'
            ORDER BY snapshot_date ASC
        """)

        trend_result = await session.execute(trend_query, {"project_id": project_id})
        trend_data = trend_result.fetchall()

        return ProgressDashboardResponse(
            project_id=project_id,
            project_name=latest.project_name,
            latest_snapshot=ProgressSnapshotResponse.from_orm(latest),
            material_quality_summary={
                "total_assessments": latest.total_material_assessments,
                "passed_assessments": latest.passed_assessments,
                "pass_rate": (latest.passed_assessments / latest.total_material_assessments * 100) if latest.total_material_assessments > 0 else 0,
                "avg_quality_score": float(latest.avg_material_quality) if latest.avg_material_quality else 0
            },
            deviation_summary={
                "total_elements": latest.total_elements_analyzed,
                "within_tolerance": latest.elements_within_tolerance,
                "tolerance_rate": (latest.elements_within_tolerance / latest.total_elements_analyzed * 100) if latest.total_elements_analyzed > 0 else 0,
                "avg_deviation_mm": float(latest.avg_deviation_mm) if latest.avg_deviation_mm else 0
            },
            trend_data=[
                ProgressTrendResponse(
                    date=row.snapshot_date,
                    percent_complete=row.percent_complete,
                    quality_score=row.quality_score,
                    schedule_variance_days=row.schedule_variance_days
                )
                for row in trend_data
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@router.get("/snapshots/{snapshot_id}/trends", response_model=List[ProgressTrendResponse])
async def get_snapshot_trends(
    snapshot_id: str,
    months: int = Query(6, ge=1, le=24, description="Number of months to include"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get progress trend data for comparison

    Returns historical snapshots for the same project, useful for
    showing progress over time in charts.
    """
    try:
        query = text("""
            SELECT
                ps2.snapshot_date as date,
                ps2.percent_complete,
                ps2.quality_score,
                ps2.schedule_variance_days,
                ps2.completed_volume_m3,
                ps2.defects_detected
            FROM progress_snapshots ps1
            JOIN progress_snapshots ps2 ON ps1.project_id = ps2.project_id
            WHERE ps1.id = :snapshot_id
              AND ps2.snapshot_date >= ps1.snapshot_date - INTERVAL ':months months'
              AND ps2.snapshot_date <= ps1.snapshot_date
            ORDER BY ps2.snapshot_date ASC
        """)

        result = await session.execute(query, {"snapshot_id": snapshot_id, "months": months})
        rows = result.fetchall()

        return [ProgressTrendResponse.from_orm(row) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.delete("/snapshots/{snapshot_id}", status_code=204)
async def delete_progress_snapshot(
    snapshot_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Delete a progress snapshot

    WARNING: This will also delete associated reports.
    """
    try:
        query = text("DELETE FROM progress_snapshots WHERE id = :snapshot_id")
        result = await session.execute(query, {"snapshot_id": snapshot_id})
        await session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        return None

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete snapshot: {str(e)}")
