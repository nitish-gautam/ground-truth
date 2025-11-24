"""
GPR data management endpoints
============================

API endpoints for managing GPR surveys, scans, and signal data.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.logging_config import log_api_request, log_api_response
from app.models.gpr_data import GPRSurvey, GPRScan, GPRSignalData
from app.schemas.gpr import (
    GPRSurveyResponse,
    GPRScanResponse,
    GPRSurveyCreate,
    GPRScanStatistics
)

router = APIRouter()


@router.get("/surveys", response_model=List[GPRSurveyResponse])
async def get_surveys(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get list of GPR surveys."""
    start_time = datetime.now()
    log_api_request("/gpr/surveys", "GET", skip=skip, limit=limit, status=status)

    try:
        query = select(GPRSurvey).options(selectinload(GPRSurvey.scans))

        if status:
            query = query.where(GPRSurvey.status == status)

        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        surveys = result.scalars().all()

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_api_response("/gpr/surveys", 200, duration_ms, surveys_found=len(surveys))

        return surveys

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get surveys: {str(e)}")


@router.get("/surveys/{survey_id}", response_model=GPRSurveyResponse)
async def get_survey(survey_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get specific GPR survey."""
    start_time = datetime.now()
    log_api_request(f"/gpr/surveys/{survey_id}", "GET")

    try:
        query = select(GPRSurvey).options(
            selectinload(GPRSurvey.scans),
            selectinload(GPRSurvey.environmental_data)
        ).where(GPRSurvey.id == survey_id)

        result = await db.execute(query)
        survey = result.scalar_one_or_none()

        if not survey:
            raise HTTPException(status_code=404, detail="Survey not found")

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_api_response(f"/gpr/surveys/{survey_id}", 200, duration_ms)

        return survey

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get survey: {str(e)}")


@router.post("/surveys", response_model=GPRSurveyResponse)
async def create_survey(
    survey_data: GPRSurveyCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new GPR survey."""
    start_time = datetime.now()
    log_api_request("/gpr/surveys", "POST")

    try:
        survey = GPRSurvey(**survey_data.model_dump())
        db.add(survey)
        await db.commit()
        await db.refresh(survey)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_api_response("/gpr/surveys", 201, duration_ms)

        return survey

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create survey: {str(e)}")


@router.get("/scans", response_model=List[GPRScanResponse])
async def get_scans(
    survey_id: Optional[UUID] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """Get list of GPR scans."""
    start_time = datetime.now()
    log_api_request("/gpr/scans", "GET", survey_id=str(survey_id) if survey_id else None)

    try:
        query = select(GPRScan)

        if survey_id:
            query = query.where(GPRScan.survey_id == survey_id)

        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        scans = result.scalars().all()

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_api_response("/gpr/scans", 200, duration_ms, scans_found=len(scans))

        return scans

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scans: {str(e)}")


@router.get("/statistics", response_model=GPRScanStatistics)
async def get_gpr_statistics(db: AsyncSession = Depends(get_db)):
    """Get GPR data statistics."""
    start_time = datetime.now()
    log_api_request("/gpr/statistics", "GET")

    try:
        # Count surveys
        surveys_result = await db.execute(select(func.count(GPRSurvey.id)))
        total_surveys = surveys_result.scalar() or 0

        # Count scans
        scans_result = await db.execute(select(func.count(GPRScan.id)))
        total_scans = scans_result.scalar() or 0

        # Count signal data records
        signals_result = await db.execute(select(func.count(GPRSignalData.id)))
        total_signals = signals_result.scalar() or 0

        # Get processing status distribution
        status_result = await db.execute(
            select(GPRScan.processing_status, func.count(GPRScan.id))
            .group_by(GPRScan.processing_status)
        )
        status_distribution = dict(status_result.all())

        statistics = GPRScanStatistics(
            total_surveys=total_surveys,
            total_scans=total_scans,
            total_signal_records=total_signals,
            processing_status_distribution=status_distribution,
            last_updated=datetime.now()
        )

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_api_response("/gpr/statistics", 200, duration_ms)

        return statistics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")