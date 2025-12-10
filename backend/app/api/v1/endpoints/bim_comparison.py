"""
HS2 Progress Assurance - BIM Comparison Endpoints

Handles BIM model uploads, LiDAR alignment, and deviation analysis.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_async_session

router = APIRouter(prefix="/progress/bim", tags=["BIM Comparison"])


@router.post("/models", status_code=201)
async def upload_bim_model(
    project_id: str,
    model_name: str,
    model_version: str,
    discipline: str,
    file: UploadFile = File(..., description="IFC file"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Upload BIM model (IFC format)

    Accepts IFC4, IFC4.3, or IFC2x3 files representing design intent.
    Used as baseline for progress verification.

    **DEMO NOTE**: In production, this would:
    1. Upload IFC to S3/MinIO
    2. Parse with ifcopenshell
    3. Extract geometry and elements
    4. Build spatial index (R-tree)
    """
    return {
        "message": "BIM model upload initiated",
        "model_id": "mock-bim-id-123",
        "status": "processing",
        "note": "Stub endpoint. Full IFC parsing in Week 3."
    }


@router.get("/models")
async def list_bim_models(
    project_id: Optional[str] = None,
    is_baseline: Optional[bool] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """List BIM models for a project"""
    try:
        where_clauses = ["is_active = true"]
        params = {}

        if project_id:
            where_clauses.append("project_id = :project_id")
            params["project_id"] = project_id
        if is_baseline is not None:
            where_clauses.append("is_baseline = :is_baseline")
            params["is_baseline"] = is_baseline

        where_sql = " AND ".join(where_clauses)

        query = text(f"""
            SELECT
                id, project_id, model_name, model_version,
                element_count, discipline, lod_level,
                is_baseline, processing_status, created_at
            FROM bim_models
            WHERE {where_sql}
            ORDER BY created_at DESC
        """)

        result = await session.execute(query, params)
        models = [dict(row._mapping) for row in result.fetchall()]

        return {"models": models, "total": len(models)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/align")
async def align_bim_to_lidar(
    bim_model_id: str,
    lidar_scan_id: str,
    alignment_method: str = "ICP",
    max_iterations: int = 50,
    convergence_threshold_m: float = 0.001,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Align BIM model to LiDAR scan using ICP

    Iterative Closest Point algorithm finds optimal transformation
    to align designed geometry with reality capture.

    **DEMO NOTE**: In production uses Open3D:
    ```python
    icp_result = o3d.pipelines.registration.registration_icp(
        source=lidar_cloud,
        target=bim_cloud,
        max_correspondence_distance=convergence_threshold_m,
        ...
    )
    ```

    For demo, returns pre-computed alignment.
    """
    try:
        # Check if alignment already exists
        check_query = text("""
            SELECT id FROM bim_lidar_alignments
            WHERE bim_model_id = :bim_model_id
              AND lidar_scan_id = :lidar_scan_id
        """)
        result = await session.execute(check_query, {
            "bim_model_id": bim_model_id,
            "lidar_scan_id": lidar_scan_id
        })
        existing = result.fetchone()

        if existing:
            return {
                "message": "Alignment already exists",
                "alignment_id": str(existing[0]),
                "status": "completed"
            }

        # In demo, return mock alignment
        # In production, this would queue Celery task for ICP processing
        return {
            "message": "Alignment task queued",
            "alignment_id": "mock-alignment-id",
            "status": "processing",
            "estimated_duration_minutes": 5,
            "note": "Stub endpoint. Full ICP implementation in Week 3-4."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alignments/{alignment_id}")
async def get_alignment_result(
    alignment_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Get BIM-LiDAR alignment results"""
    try:
        query = text("""
            SELECT
                bla.*,
                bm.model_name,
                pls.scanner_model,
                pls.point_count
            FROM bim_lidar_alignments bla
            LEFT JOIN bim_models bm ON bla.bim_model_id = bm.id
            LEFT JOIN progress_lidar_scans pls ON bla.lidar_scan_id = pls.id
            WHERE bla.id = :alignment_id
        """)

        result = await session.execute(query, {"alignment_id": alignment_id})
        alignment = result.fetchone()

        if not alignment:
            raise HTTPException(status_code=404, detail="Alignment not found")

        return dict(alignment._mapping)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deviations")
async def get_deviation_analysis(
    alignment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    severity: Optional[str] = Query(None, description="Filter by severity"),
    within_tolerance: Optional[bool] = None,
    limit: int = Query(1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get deviation analysis results

    Returns element-level comparison of BIM vs reality:
    - Mean/max deviation in mm
    - Severity classification
    - Color codes for visualization
    """
    try:
        where_clauses = []
        params = {"limit": limit}

        if alignment_id:
            where_clauses.append("pda.alignment_id = :alignment_id")
            params["alignment_id"] = alignment_id

        if project_id:
            where_clauses.append("""
                pda.alignment_id IN (
                    SELECT bla.id FROM bim_lidar_alignments bla
                    JOIN bim_models bm ON bla.bim_model_id = bm.id
                    WHERE bm.project_id = :project_id
                )
            """)
            params["project_id"] = project_id

        if severity:
            where_clauses.append("pda.severity = :severity")
            params["severity"] = severity

        if within_tolerance is not None:
            where_clauses.append("pda.within_tolerance = :within_tolerance")
            params["within_tolerance"] = within_tolerance

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = text(f"""
            SELECT
                pda.id, pda.alignment_id,
                pda.bim_element_id, pda.element_type, pda.element_name,
                pda.mean_deviation_mm, pda.max_deviation_mm,
                pda.severity, pda.within_tolerance, pda.color_code,
                pda.confidence_score, pda.created_at
            FROM progress_deviation_analysis pda
            WHERE {where_sql}
            ORDER BY pda.mean_deviation_mm DESC
            LIMIT :limit
        """)

        result = await session.execute(query, params)
        deviations = [dict(row._mapping) for row in result.fetchall()]

        # Calculate summary stats
        summary = {
            "total_elements": len(deviations),
            "within_tolerance": sum(1 for d in deviations if d['within_tolerance']),
            "avg_deviation_mm": sum(d['mean_deviation_mm'] for d in deviations) / len(deviations) if deviations else 0,
            "max_deviation_mm": max((d['max_deviation_mm'] for d in deviations), default=0),
            "severity_breakdown": {
                "None": sum(1 for d in deviations if d['severity'] == 'None'),
                "Minor": sum(1 for d in deviations if d['severity'] == 'Minor'),
                "Moderate": sum(1 for d in deviations if d['severity'] == 'Moderate'),
                "Major": sum(1 for d in deviations if d['severity'] == 'Major'),
                "Critical": sum(1 for d in deviations if d['severity'] == 'Critical')
            }
        }

        return {
            "deviations": deviations,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deviations/heatmap")
async def get_deviation_heatmap_data(
    alignment_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get deviation data formatted for 3D heatmap visualization

    Returns color-coded mesh data for rendering in frontend.
    """
    try:
        query = text("""
            SELECT
                bim_element_id,
                element_type,
                mean_deviation_mm,
                color_code,
                ST_AsGeoJSON(location) as location_geojson,
                ST_AsGeoJSON(deviation_geometry) as geometry_geojson
            FROM progress_deviation_analysis
            WHERE alignment_id = :alignment_id
            ORDER BY mean_deviation_mm DESC
        """)

        result = await session.execute(query, {"alignment_id": alignment_id})
        elements = [dict(row._mapping) for row in result.fetchall()]

        return {
            "alignment_id": alignment_id,
            "elements": elements,
            "color_scale": {
                "#00FF00": "Within tolerance (<10mm)",
                "#FFFF00": "Minor deviation (10-20mm)",
                "#FFA500": "Moderate deviation (20-30mm)",
                "#FF0000": "Critical deviation (>30mm)"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
