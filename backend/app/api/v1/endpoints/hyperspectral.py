"""
HS2 Progress Assurance - Hyperspectral Imaging Endpoints

Handles hyperspectral scan uploads, material quality assessments,
and spectral library management.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_async_session

router = APIRouter(prefix="/progress/hyperspectral", tags=["Hyperspectral Imaging"])


@router.post("/scans", status_code=201)
async def upload_hyperspectral_scan(
    project_id: str,
    site_location: str,
    camera_model: str,
    wavelength_range: str,
    band_count: int,
    file: UploadFile = File(..., description="Hyperspectral data file (ENVI/HDF5)"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Upload hyperspectral scan data

    Accepts ENVI format (.hdr + .dat) or HDF5 files containing
    hyperspectral data. Recommended camera: Specim IQ (204 spectral
    bands, 400-1000nm) for material quality analysis.

    **DEMO NOTE**: In production, this would:
    1. Upload file to S3/MinIO
    2. Queue Celery task for processing
    3. Extract spectral signatures
    4. Run ML models for material prediction
    """
    try:
        # TODO: Implement file upload to MinIO
        # TODO: Queue Celery task for processing
        # For now, return mock response

        return {
            "message": "Hyperspectral scan upload initiated",
            "scan_id": "mock-scan-id-123",
            "status": "processing",
            "note": "This is a stub endpoint. Full implementation coming in Week 3."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/scans")
async def list_hyperspectral_scans(
    project_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_async_session)
):
    """
    List hyperspectral scans

    Returns scans with processing status and metadata.
    """
    try:
        where_clause = "WHERE project_id = :project_id" if project_id else "WHERE 1=1"
        params = {"limit": limit, "offset": offset}
        if project_id:
            params["project_id"] = project_id

        query = text(f"""
            SELECT
                id, project_id, site_location, scan_date,
                camera_model, wavelength_range, band_count,
                processing_status, image_quality_score,
                created_at
            FROM hyperspectral_scans
            {where_clause}
            ORDER BY scan_date DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await session.execute(query, params)
        scans = [dict(row._mapping) for row in result.fetchall()]

        return {"scans": scans, "total": len(scans), "limit": limit, "offset": offset}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scans: {str(e)}")


@router.get("/scans/{scan_id}")
async def get_hyperspectral_scan(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Get detailed hyperspectral scan information"""
    try:
        query = text("""
            SELECT * FROM hyperspectral_scans WHERE id = :scan_id
        """)

        result = await session.execute(query, {"scan_id": scan_id})
        scan = result.fetchone()

        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")

        return dict(scan._mapping)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scan: {str(e)}")


@router.get("/scans/{scan_id}/quality")
async def get_material_quality_assessments(
    scan_id: str,
    material_type: Optional[str] = Query(None, description="Filter by material type"),
    meets_spec: Optional[bool] = Query(None, description="Filter by specification compliance"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get material quality assessments for a scan

    Returns AI-predicted material properties including:
    - Concrete strength (MPa)
    - Defect detection
    - Quality scores
    - Spectral evidence
    """
    try:
        where_clauses = ["scan_id = :scan_id"]
        params = {"scan_id": scan_id}

        if material_type:
            where_clauses.append("material_type = :material_type")
            params["material_type"] = material_type
        if meets_spec is not None:
            where_clauses.append("meets_specification = :meets_spec")
            params["meets_spec"] = meets_spec

        where_sql = " AND ".join(where_clauses)

        query = text(f"""
            SELECT
                id, material_type, material_subtype,
                predicted_strength_mpa, specification_strength_mpa,
                meets_specification, quality_score, quality_grade,
                defect_count, defects_detected,
                spectral_match_score, model_confidence,
                created_at
            FROM material_quality_assessments
            WHERE {where_sql}
            ORDER BY quality_score DESC
        """)

        result = await session.execute(query, params)
        assessments = [dict(row._mapping) for row in result.fetchall()]

        # Calculate summary statistics
        summary = {
            "total_assessments": len(assessments),
            "passed_assessments": sum(1 for a in assessments if a['meets_specification']),
            "avg_quality_score": sum(a['quality_score'] for a in assessments) / len(assessments) if assessments else 0,
            "avg_strength_mpa": sum(a['predicted_strength_mpa'] for a in assessments if a['predicted_strength_mpa']) / len([a for a in assessments if a['predicted_strength_mpa']]) if any(a['predicted_strength_mpa'] for a in assessments) else 0
        }

        return {
            "scan_id": scan_id,
            "assessments": assessments,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assessments: {str(e)}")


@router.post("/analyze-material")
async def analyze_material_quality(
    scan_id: str,
    region_coords: dict,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Analyze material quality for a specific region

    **DEMO NOTE**: In production, this would:
    1. Extract spectral signature from region
    2. Run CNN model for strength prediction
    3. Detect defects using anomaly detection
    4. Return quality assessment with confidence

    For demo, returns mock analysis.
    """
    return {
        "status": "completed",
        "analysis": {
            "material_type": "concrete",
            "predicted_strength_mpa": 42.5,
            "specification_strength_mpa": 40.0,
            "meets_specification": True,
            "quality_score": 88.5,
            "quality_grade": "B",
            "defects_detected": [],
            "confidence": 92.3
        },
        "note": "This is a stub endpoint. Full ML model implementation coming in Week 5-6."
    }


@router.get("/spectral-library")
async def get_spectral_library(
    material_category: Optional[str] = Query(None, description="Filter by category"),
    is_validated: Optional[bool] = Query(None, description="Filter by validation status"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get reference spectral signatures from library

    The spectral library contains validated reference materials with
    known properties for comparison during analysis.
    """
    try:
        where_clauses = ["is_active = true"]
        params = {}

        if material_category:
            where_clauses.append("material_category = :material_category")
            params["material_category"] = material_category
        if is_validated is not None:
            where_clauses.append("is_validated = :is_validated")
            params["is_validated"] = is_validated

        where_sql = " AND ".join(where_clauses)

        query = text(f"""
            SELECT
                id, material_name, material_category, material_grade,
                material_properties, validation_confidence,
                is_validated, usage_count, last_used_at,
                created_at
            FROM spectral_library
            WHERE {where_sql}
            ORDER BY usage_count DESC, material_name ASC
        """)

        result = await session.execute(query, params)
        library_items = [dict(row._mapping) for row in result.fetchall()]

        return {"library": library_items, "total": len(library_items)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get library: {str(e)}")


@router.post("/spectral-library")
async def add_spectral_reference(
    material_name: str,
    material_category: str,
    wavelengths: List[float],
    reflectance_values: List[float],
    material_properties: dict,
    lab_test_results: Optional[dict] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Add new reference material to spectral library

    This requires:
    1. Hyperspectral scan of known material
    2. Destructive lab test results for validation
    3. Quality control approval

    **Used for building training data library.**
    """
    try:
        # Validate inputs
        if len(wavelengths) != len(reflectance_values):
            raise HTTPException(status_code=400, detail="Wavelengths and reflectance arrays must match")

        query = text("""
            INSERT INTO spectral_library (
                id, material_name, material_category,
                wavelengths, reflectance_values, material_properties,
                lab_test_results, is_validated, created_at
            ) VALUES (
                gen_random_uuid(), :material_name, :material_category,
                :wavelengths, :reflectance_values, :material_properties::jsonb,
                :lab_test_results::jsonb, :is_validated, NOW()
            )
            RETURNING id
        """)

        result = await session.execute(query, {
            "material_name": material_name,
            "material_category": material_category,
            "wavelengths": wavelengths,
            "reflectance_values": reflectance_values,
            "material_properties": material_properties,
            "lab_test_results": lab_test_results,
            "is_validated": lab_test_results is not None
        })
        await session.commit()

        library_id = result.scalar()

        return {
            "message": "Reference material added to library",
            "library_id": str(library_id),
            "is_validated": lab_test_results is not None
        }

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add reference: {str(e)}")
