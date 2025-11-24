"""
Main API router for version 1
=============================

Combines all API endpoint routers into a single router for the application.
"""

from fastapi import APIRouter

from .endpoints import (
    gpr_data,
    environmental,
    validation,
    analytics,
    processing,
    datasets,
    material_classification,
    pas128_compliance
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["datasets"]
)

api_router.include_router(
    gpr_data.router,
    prefix="/gpr",
    tags=["gpr-data"]
)

api_router.include_router(
    environmental.router,
    prefix="/environmental",
    tags=["environmental"]
)

api_router.include_router(
    processing.router,
    prefix="/processing",
    tags=["signal-processing"]
)

api_router.include_router(
    validation.router,
    prefix="/validation",
    tags=["validation"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)

api_router.include_router(
    material_classification.router,
    tags=["material-classification"]
)

api_router.include_router(
    pas128_compliance.router,
    prefix="/compliance",
    tags=["pas128-compliance"]
)