"""
Validation and ground truth endpoints
====================================

API endpoints for ground truth data and validation results.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()


@router.get("/ground-truth")
async def get_ground_truth_data(db: AsyncSession = Depends(get_db)):
    """Get ground truth utility data."""
    return {"message": "Ground truth data endpoint - implementation pending"}


@router.get("/accuracy-metrics")
async def get_accuracy_metrics(db: AsyncSession = Depends(get_db)):
    """Get accuracy assessment metrics."""
    return {"message": "Accuracy metrics endpoint - implementation pending"}