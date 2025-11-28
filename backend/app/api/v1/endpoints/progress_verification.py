"""
HS2 Progress Verification Endpoints
===================================

API endpoints for progress tracking, point cloud comparison, schedule management,
and earned value analysis.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
import shutil
from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.core.database import get_db
from app.services.point_cloud_processor import get_point_cloud_processor
from app.models.progress import (
    HS2ProgressSnapshot,
    HS2PointCloudComparison,
    HS2ScheduleMilestone,
)
from app.models.hs2 import HS2Asset
from app.schemas.progress import (
    ProgressSnapshotCreate,
    ProgressSnapshotResponse,
    ProgressSnapshotUpdate,
    ProgressTimelineResponse,
    PointCloudComparisonCreate,
    PointCloudComparisonResponse,
    PointCloudComparisonListResponse,
    ScheduleMilestoneCreate,
    ScheduleMilestoneResponse,
    ScheduleMilestoneUpdate,
    GanttChartResponse,
    EarnedValueAnalysis,
    CostProgressAlignment,
    ProgressVerificationSummary,
)

router = APIRouter()


# ==================== Progress Snapshot Endpoints ====================

@router.post("/snapshot", response_model=ProgressSnapshotResponse, status_code=status.HTTP_201_CREATED)
async def create_progress_snapshot(
    snapshot_data: ProgressSnapshotCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new progress snapshot for an asset.

    **Request Body:**
    - **asset_id**: UUID of the asset
    - **snapshot_date**: Timestamp of the snapshot
    - **physical_progress_pct**: Physical progress percentage (0-100)
    - **cost_progress_pct**: Cost progress percentage (0-100)
    - **earned_value**: Earned Value (EV) amount
    - **actual_cost**: Actual Cost (AC) amount
    - **anomalies**: List of detected anomalies

    **Returns:**
    Created progress snapshot with calculated EVM metrics (CPI, SPI, variances).
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == snapshot_data.asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {snapshot_data.asset_id} not found"
            )

        # Create snapshot
        snapshot = HS2ProgressSnapshot(**snapshot_data.model_dump())

        # Calculate derived EVM metrics
        if snapshot.earned_value and snapshot.actual_cost and snapshot.actual_cost > 0:
            snapshot.cost_variance = snapshot.earned_value - snapshot.actual_cost
            snapshot.cost_performance_index = snapshot.earned_value / snapshot.actual_cost

        if snapshot.earned_value and snapshot.planned_value and snapshot.planned_value > 0:
            snapshot.schedule_variance = snapshot.earned_value - snapshot.planned_value
            snapshot.schedule_performance_index = snapshot.earned_value / snapshot.planned_value

        db.add(snapshot)
        await db.commit()
        await db.refresh(snapshot)

        logger.info(f"Created progress snapshot for asset {snapshot_data.asset_id}")
        return snapshot

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating progress snapshot: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create progress snapshot: {str(e)}"
        )


@router.get("/snapshot/{asset_id}", response_model=ProgressTimelineResponse)
async def get_progress_timeline(
    asset_id: UUID = Path(..., description="Asset UUID"),
    start_date: Optional[datetime] = Query(None, description="Start date for timeline"),
    end_date: Optional[datetime] = Query(None, description="End date for timeline"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum snapshots to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get progress timeline for an asset with all snapshots.

    **Path Parameters:**
    - **asset_id**: UUID of the asset

    **Query Parameters:**
    - **start_date**: Filter snapshots from this date
    - **end_date**: Filter snapshots until this date
    - **limit**: Maximum number of snapshots (default: 100)

    **Returns:**
    Timeline with snapshots and aggregated metrics (trends, averages, performance indices).
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found"
            )

        # Build query
        query = select(HS2ProgressSnapshot).where(
            HS2ProgressSnapshot.asset_id == asset_id
        )

        if start_date:
            query = query.where(HS2ProgressSnapshot.snapshot_date >= start_date)
        if end_date:
            query = query.where(HS2ProgressSnapshot.snapshot_date <= end_date)

        query = query.order_by(desc(HS2ProgressSnapshot.snapshot_date)).limit(limit)

        result = await db.execute(query)
        snapshots = result.scalars().all()

        # Calculate aggregated metrics
        metrics = {}
        if snapshots:
            physical_values = [s.physical_progress_pct for s in snapshots if s.physical_progress_pct]
            cpi_values = [s.cost_performance_index for s in snapshots if s.cost_performance_index]

            if physical_values:
                metrics["avg_physical_progress"] = float(sum(physical_values) / len(physical_values))
                metrics["latest_physical_progress"] = float(physical_values[0])

            if cpi_values:
                metrics["avg_cost_performance_index"] = float(sum(cpi_values) / len(cpi_values))
                metrics["latest_cpi"] = float(cpi_values[0])

            # Simple trend calculation (positive if improving)
            if len(physical_values) >= 2:
                metrics["trend"] = "improving" if physical_values[0] > physical_values[-1] else "declining"

        return ProgressTimelineResponse(
            asset_id=asset_id,
            snapshots=snapshots,
            metrics=metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching progress timeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch progress timeline: {str(e)}"
        )


@router.patch("/snapshot/{snapshot_id}", response_model=ProgressSnapshotResponse)
async def update_progress_snapshot(
    snapshot_id: UUID = Path(..., description="Snapshot UUID"),
    update_data: ProgressSnapshotUpdate = ...,
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing progress snapshot.

    **Path Parameters:**
    - **snapshot_id**: UUID of the snapshot to update

    **Request Body:**
    Any fields from ProgressSnapshotUpdate (all optional)

    **Returns:**
    Updated snapshot with recalculated EVM metrics.
    """
    try:
        result = await db.execute(
            select(HS2ProgressSnapshot).where(HS2ProgressSnapshot.id == snapshot_id)
        )
        snapshot = result.scalar_one_or_none()

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot {snapshot_id} not found"
            )

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(snapshot, key, value)

        # Recalculate derived metrics
        if snapshot.earned_value and snapshot.actual_cost and snapshot.actual_cost > 0:
            snapshot.cost_variance = snapshot.earned_value - snapshot.actual_cost
            snapshot.cost_performance_index = snapshot.earned_value / snapshot.actual_cost

        if snapshot.earned_value and snapshot.planned_value and snapshot.planned_value > 0:
            snapshot.schedule_variance = snapshot.earned_value - snapshot.planned_value
            snapshot.schedule_performance_index = snapshot.earned_value / snapshot.planned_value

        await db.commit()
        await db.refresh(snapshot)

        logger.info(f"Updated progress snapshot {snapshot_id}")
        return snapshot

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating progress snapshot: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update progress snapshot: {str(e)}"
        )


# ==================== Point Cloud Comparison Endpoints ====================

@router.post("/point-cloud/compare", response_model=PointCloudComparisonResponse, status_code=status.HTTP_201_CREATED)
async def create_point_cloud_comparison(
    comparison_data: PointCloudComparisonCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new point cloud comparison (BIM vs reality capture).

    **Request Body:**
    - **asset_id**: UUID of the asset
    - **baseline_file_path**: Path to BIM model or baseline scan
    - **current_file_path**: Path to current site scan (LAS/LAZ)
    - **volume_difference_m3**: Calculated volume difference
    - **surface_deviation_avg**: Average surface deviation in mm
    - **completion_percentage**: Calculated completion percentage
    - **heatmap_data**: Visualization data for 3D heatmap

    **Returns:**
    Created comparison with all analysis results.
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == comparison_data.asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {comparison_data.asset_id} not found"
            )

        # Create comparison
        comparison = HS2PointCloudComparison(**comparison_data.model_dump())
        comparison.algorithm_version = "v1.0"  # Default
        comparison.processed_by = "api_endpoint"  # Could be enhanced with user info

        db.add(comparison)
        await db.commit()
        await db.refresh(comparison)

        logger.info(f"Created point cloud comparison for asset {comparison_data.asset_id}")
        return comparison

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating point cloud comparison: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create point cloud comparison: {str(e)}"
        )


@router.post("/point-cloud/upload-and-compare", response_model=PointCloudComparisonResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_compare_point_clouds(
    asset_id: UUID = Form(..., description="Asset UUID"),
    baseline_file: UploadFile = File(..., description="Baseline point cloud file (LAS/LAZ/PLY)"),
    current_file: UploadFile = File(..., description="Current scan file (LAS/LAZ/PLY)"),
    tolerance_mm: float = Form(50.0, description="Tolerance threshold in millimeters"),
    downsample_voxel_size: float = Form(0.05, description="Voxel size for downsampling (meters)"),
    remove_outliers: bool = Form(True, description="Remove statistical outliers"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process point cloud files for comparison (automated workflow).

    **Form Data:**
    - **asset_id**: UUID of the asset being analyzed
    - **baseline_file**: Baseline point cloud file (BIM export or initial scan)
    - **current_file**: Current site scan (LiDAR/photogrammetry)
    - **tolerance_mm**: Tolerance threshold in mm (default: 50mm)
    - **downsample_voxel_size**: Downsampling resolution in meters (default: 0.05m = 5cm)
    - **remove_outliers**: Whether to remove outliers (default: true)

    **Supported Formats:**
    - LAS (LiDAR data)
    - LAZ (compressed LAS)
    - PLY (mesh/point cloud)

    **Processing Steps:**
    1. Validate file formats
    2. Save files temporarily
    3. Load and preprocess point clouds
    4. Align using ICP algorithm
    5. Compute surface deviations
    6. Calculate volume differences
    7. Generate heatmap visualization
    8. Assess quality
    9. Save results to database
    10. Clean up temporary files

    **Returns:**
    Complete comparison analysis with all metrics, heatmaps, and quality flags.
    """
    try:
        logger.info(f"Starting point cloud upload and comparison for asset {asset_id}")

        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found"
            )

        # Validate file extensions
        allowed_extensions = {".las", ".laz", ".ply"}
        baseline_ext = FilePath(baseline_file.filename).suffix.lower()
        current_ext = FilePath(current_file.filename).suffix.lower()

        if baseline_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Baseline file must be LAS, LAZ, or PLY (got {baseline_ext})"
            )

        if current_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Current file must be LAS, LAZ, or PLY (got {current_ext})"
            )

        # Create temporary directory for processing
        temp_dir = FilePath("/tmp/point_cloud_processing") / str(asset_id)
        temp_dir.mkdir(parents=True, exist_ok=True)

        baseline_path = temp_dir / f"baseline_{datetime.now().timestamp()}{baseline_ext}"
        current_path = temp_dir / f"current_{datetime.now().timestamp()}{current_ext}"

        try:
            # Save uploaded files
            logger.info(f"Saving baseline file to {baseline_path}")
            with open(baseline_path, "wb") as f:
                shutil.copyfileobj(baseline_file.file, f)

            logger.info(f"Saving current file to {current_path}")
            with open(current_path, "wb") as f:
                shutil.copyfileobj(current_file.file, f)

            # Process point clouds
            processor = get_point_cloud_processor(tolerance_mm=tolerance_mm)

            logger.info("Starting point cloud comparison processing")
            comparison_results = await processor.process_comparison(
                baseline_file_path=str(baseline_path),
                current_file_path=str(current_path),
                downsample_voxel_size=downsample_voxel_size,
                remove_outliers_flag=remove_outliers
            )

            # Prepare permanent file paths (would integrate with MinIO/S3 in production)
            permanent_baseline_path = f"point-clouds/{asset_id}/baseline/{baseline_file.filename}"
            permanent_current_path = f"point-clouds/{asset_id}/scans/{datetime.now().isoformat()}/{current_file.filename}"

            # Create database record
            comparison = HS2PointCloudComparison(
                asset_id=asset_id,
                baseline_file_path=permanent_baseline_path,
                current_file_path=permanent_current_path,
                comparison_date=datetime.now(),
                **comparison_results
            )

            db.add(comparison)
            await db.commit()
            await db.refresh(comparison)

            logger.info(f"Point cloud comparison completed successfully for asset {asset_id}")
            return comparison

        finally:
            # Clean up temporary files
            try:
                if baseline_path.exists():
                    baseline_path.unlink()
                if current_path.exists():
                    current_path.unlink()
                logger.info("Temporary files cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary files: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in point cloud upload and comparison: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process point cloud comparison: {str(e)}"
        )


@router.get("/point-cloud/{asset_id}", response_model=PointCloudComparisonListResponse)
async def get_point_cloud_comparisons(
    asset_id: UUID = Path(..., description="Asset UUID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum comparisons to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all point cloud comparisons for an asset.

    **Path Parameters:**
    - **asset_id**: UUID of the asset

    **Query Parameters:**
    - **limit**: Maximum number of comparisons (default: 50)

    **Returns:**
    List of all comparisons with analysis results, ordered by date (newest first).
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found"
            )

        # Get comparisons
        query = select(HS2PointCloudComparison).where(
            HS2PointCloudComparison.asset_id == asset_id
        ).order_by(desc(HS2PointCloudComparison.comparison_date)).limit(limit)

        result = await db.execute(query)
        comparisons = result.scalars().all()

        # Get total count
        count_result = await db.execute(
            select(func.count()).select_from(HS2PointCloudComparison).where(
                HS2PointCloudComparison.asset_id == asset_id
            )
        )
        total_count = count_result.scalar()

        return PointCloudComparisonListResponse(
            asset_id=asset_id,
            comparisons=comparisons,
            total_count=total_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching point cloud comparisons: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch point cloud comparisons: {str(e)}"
        )


# ==================== Schedule Milestone Endpoints ====================

@router.post("/milestone", response_model=ScheduleMilestoneResponse, status_code=status.HTTP_201_CREATED)
async def create_schedule_milestone(
    milestone_data: ScheduleMilestoneCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new schedule milestone for an asset.

    **Request Body:**
    - **asset_id**: UUID of the asset
    - **milestone_name**: Name of the milestone
    - **planned_date**: Planned completion date
    - **status**: Current status (not_started, in_progress, completed, delayed)
    - **predecessors**: List of predecessor milestone dependencies
    - **is_critical_path**: Whether this milestone is on the critical path

    **Returns:**
    Created milestone with all schedule details.
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == milestone_data.asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {milestone_data.asset_id} not found"
            )

        # Create milestone
        milestone = HS2ScheduleMilestone(**milestone_data.model_dump())

        # Calculate schedule variance if actual date is set
        if milestone.actual_date and milestone.planned_date:
            variance_delta = milestone.actual_date - milestone.planned_date
            milestone.schedule_variance_days = variance_delta.days

        db.add(milestone)
        await db.commit()
        await db.refresh(milestone)

        logger.info(f"Created schedule milestone for asset {milestone_data.asset_id}")
        return milestone

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating schedule milestone: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create schedule milestone: {str(e)}"
        )


@router.get("/milestone/{asset_id}", response_model=List[ScheduleMilestoneResponse])
async def get_schedule_milestones(
    asset_id: UUID = Path(..., description="Asset UUID"),
    milestone_type: Optional[str] = Query(None, description="Filter by milestone type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all schedule milestones for an asset.

    **Path Parameters:**
    - **asset_id**: UUID of the asset

    **Query Parameters:**
    - **milestone_type**: Filter by type (foundation, structure, completion, inspection)
    - **status**: Filter by status (not_started, in_progress, completed, delayed)

    **Returns:**
    List of all milestones for the asset, ordered by planned date.
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found"
            )

        # Build query
        query = select(HS2ScheduleMilestone).where(
            HS2ScheduleMilestone.asset_id == asset_id
        )

        if milestone_type:
            query = query.where(HS2ScheduleMilestone.milestone_type == milestone_type)
        if status:
            query = query.where(HS2ScheduleMilestone.status == status)

        query = query.order_by(HS2ScheduleMilestone.planned_date)

        result = await db.execute(query)
        milestones = result.scalars().all()

        return milestones

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching schedule milestones: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch schedule milestones: {str(e)}"
        )


@router.patch("/milestone/{milestone_id}", response_model=ScheduleMilestoneResponse)
async def update_schedule_milestone(
    milestone_id: UUID = Path(..., description="Milestone UUID"),
    update_data: ScheduleMilestoneUpdate = ...,
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing schedule milestone.

    **Path Parameters:**
    - **milestone_id**: UUID of the milestone to update

    **Request Body:**
    Any fields from ScheduleMilestoneUpdate (all optional)

    **Returns:**
    Updated milestone with recalculated variance metrics.
    """
    try:
        result = await db.execute(
            select(HS2ScheduleMilestone).where(HS2ScheduleMilestone.id == milestone_id)
        )
        milestone = result.scalar_one_or_none()

        if not milestone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Milestone {milestone_id} not found"
            )

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(milestone, key, value)

        # Recalculate schedule variance
        if milestone.actual_date and milestone.planned_date:
            variance_delta = milestone.actual_date - milestone.planned_date
            milestone.schedule_variance_days = variance_delta.days

        await db.commit()
        await db.refresh(milestone)

        logger.info(f"Updated schedule milestone {milestone_id}")
        return milestone

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schedule milestone: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update schedule milestone: {str(e)}"
        )


@router.get("/gantt/{asset_id}", response_model=GanttChartResponse)
async def get_gantt_chart_data(
    asset_id: UUID = Path(..., description="Asset UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get Gantt chart data for an asset with all milestones and critical path.

    **Path Parameters:**
    - **asset_id**: UUID of the asset

    **Returns:**
    Complete Gantt chart data including:
    - All milestones
    - Critical path milestone IDs
    - Project start/end dates
    - Schedule metrics (on-time, delayed, average delay)
    """
    try:
        # Verify asset exists
        asset_result = await db.execute(
            select(HS2Asset).where(HS2Asset.id == asset_id)
        )
        asset = asset_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found"
            )

        # Get all milestones
        result = await db.execute(
            select(HS2ScheduleMilestone).where(
                HS2ScheduleMilestone.asset_id == asset_id
            ).order_by(HS2ScheduleMilestone.planned_date)
        )
        milestones = result.scalars().all()

        if not milestones:
            return GanttChartResponse(
                asset_id=asset_id,
                milestones=[],
                critical_path=[],
                project_start=None,
                project_end=None,
                total_duration_days=None,
                metrics={"on_time_milestones": 0, "delayed_milestones": 0, "avg_delay_days": 0}
            )

        # Get critical path milestones
        critical_path = [m.id for m in milestones if m.is_critical_path]

        # Calculate project dates
        planned_dates = [m.planned_date for m in milestones if m.planned_date]
        project_start = min(planned_dates) if planned_dates else None
        project_end = max(planned_dates) if planned_dates else None
        total_duration = (project_end - project_start).days if (project_start and project_end) else None

        # Calculate metrics
        on_time = sum(1 for m in milestones if m.status == "completed" and (
            not m.schedule_variance_days or m.schedule_variance_days <= 0
        ))
        delayed = sum(1 for m in milestones if m.schedule_variance_days and m.schedule_variance_days > 0)

        delays = [m.schedule_variance_days for m in milestones if m.schedule_variance_days and m.schedule_variance_days > 0]
        avg_delay = sum(delays) / len(delays) if delays else 0

        metrics = {
            "on_time_milestones": on_time,
            "delayed_milestones": delayed,
            "avg_delay_days": avg_delay,
            "total_milestones": len(milestones),
            "completed_milestones": sum(1 for m in milestones if m.status == "completed"),
            "critical_path_count": len(critical_path)
        }

        return GanttChartResponse(
            asset_id=asset_id,
            milestones=milestones,
            critical_path=critical_path,
            project_start=project_start,
            project_end=project_end,
            total_duration_days=total_duration,
            metrics=metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating Gantt chart data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate Gantt chart data: {str(e)}"
        )


# ==================== Health Check ====================

@router.get("/health")
async def progress_health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check for progress verification service.

    Returns counts of all progress-related records.
    """
    try:
        snapshots_count = await db.execute(select(func.count()).select_from(HS2ProgressSnapshot))
        comparisons_count = await db.execute(select(func.count()).select_from(HS2PointCloudComparison))
        milestones_count = await db.execute(select(func.count()).select_from(HS2ScheduleMilestone))

        return {
            "status": "healthy",
            "tables": {
                "progress_snapshots": snapshots_count.scalar(),
                "point_cloud_comparisons": comparisons_count.scalar(),
                "schedule_milestones": milestones_count.scalar()
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )
