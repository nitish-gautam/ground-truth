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
    pas128_compliance,
    hs2_assets,
    hs2_deliverables,
    hs2_rules,
    hs2_dashboard,
    monitoring,
    gis,
    bim,
    progress_verification,
    graph,
    lidar,
    hyperspectral
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

# HS2 Assurance Intelligence Demonstrator Endpoints
api_router.include_router(
    hs2_assets.router,
    prefix="/hs2",
    tags=["hs2-assets"]
)

api_router.include_router(
    hs2_deliverables.router,
    prefix="/hs2",
    tags=["hs2-deliverables"]
)

api_router.include_router(
    hs2_rules.router,
    prefix="/hs2",
    tags=["hs2-taem-rules"]
)

api_router.include_router(
    hs2_dashboard.router,
    prefix="/hs2",
    tags=["hs2-dashboard"]
)

# Monitoring endpoints (noise, vibration, environmental)
api_router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["monitoring"]
)

# GIS endpoints (shapefiles, route data)
api_router.include_router(
    gis.router,
    prefix="/gis",
    tags=["gis-data"]
)

# BIM endpoints (IFC models)
api_router.include_router(
    bim.router,
    prefix="/bim",
    tags=["bim-models"]
)

# Progress Verification endpoints (snapshots, point cloud, schedule, EVM)
api_router.include_router(
    progress_verification.router,
    prefix="/progress",
    tags=["progress-verification"]
)

# Graph Database endpoints (Neo4j - explainability, visualization, impact analysis)
api_router.include_router(
    graph.router,
    tags=["graph-database"]
)

# LiDAR Data endpoints (DTM tiles, elevation profiles, coverage)
api_router.include_router(
    lidar.router,
    prefix="/lidar",
    tags=["lidar-data"]
)

# Hyperspectral Imaging endpoints (material analysis, concrete quality, defect detection)
# Note: Router already includes /progress/hyperspectral prefix
api_router.include_router(
    hyperspectral.router,
    tags=["hyperspectral-imaging"]
)
