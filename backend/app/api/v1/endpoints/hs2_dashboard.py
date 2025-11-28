"""
HS2 Dashboard Endpoints
======================

API endpoints for dashboard statistics, analytics, and aggregate reporting.
"""

from datetime import datetime
from typing import Optional
from pathlib import Path
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from loguru import logger

from app.core.database import get_db
from app.schemas.hs2 import (
    DashboardSummary,
    AssetTypeBreakdown,
    ContractorBreakdown,
    RouteBreakdown,
)

router = APIRouter()


def get_injunctions_count_from_geojson(organized_path: Path) -> int:
    """Helper function to read legal injunctions count from GeoJSON file"""
    injunctions_path = organized_path / "gis-shapefiles/legal/injunctions/HS2_Injunctions.geojson"
    if not injunctions_path.exists():
        return 0

    try:
        with open(injunctions_path, 'r') as f:
            geojson_data = json.load(f)
            return len(geojson_data.get('features', []))
    except Exception as e:
        logger.warning(f"Failed to read injunctions count from GeoJSON: {e}")
        return 0


@router.get("/dashboard/dataset-stats")
async def get_dataset_stats():
    """
    Get real statistics about available datasets

    Returns actual counts of BIM models, GIS shapefiles, noise monitoring files, etc.
    """
    import os

    # BIM Models
    bim_path = Path("/datasets/hs2/rawdata/IFC4.3.x-sample-models-main/models")
    bim_count = len(list(bim_path.rglob("*.ifc"))) if bim_path.exists() else 0
    bim_size_mb = sum(f.stat().st_size for f in bim_path.rglob("*.ifc") if f.exists()) / (1024 * 1024) if bim_path.exists() else 0

    # GIS Shapefiles (rawdata)
    gis_path = Path("/datasets/hs2/rawdata/Phase_2a_WDEIAR_September_2016/Shapefiles")
    gis_count = 0
    if gis_path.exists():
        for category in gis_path.iterdir():
            if category.is_dir():
                gis_count += len(list(category.glob("*.shp")))

    # GIS Shapefiles from organized data (NEW)
    organized_path = Path("/datasets/hs2/organized")
    ecology_shp_count = 0
    property_shp_count = 0

    # Count ecology shapefiles
    ecology_path = organized_path / "ecology/surveys/november-2024/shapefiles"
    if ecology_path.exists():
        ecology_shp_count = len(list(ecology_path.glob("*.shp")))

    # Count property compensation shapefiles
    property_path = organized_path / "property/compensation/july-2014/shapefiles"
    if property_path.exists():
        property_shp_count = len(list(property_path.glob("*.shp")))

    # Count injunctions zones from GeoJSON (actual feature count, not file count)
    injunctions_count = get_injunctions_count_from_geojson(organized_path)

    # For file count in GIS total, count it as 1 file
    injunctions_file_count = 1 if injunctions_count > 0 else 0

    # Total GIS count (files)
    total_gis_count = gis_count + ecology_shp_count + property_shp_count + injunctions_file_count

    # Count ecology CSV files
    ecology_csv_count = 0
    ecology_csv_path = organized_path / "ecology/surveys/november-2024/csv"
    if ecology_csv_path.exists():
        ecology_csv_count = len(list(ecology_csv_path.glob("*.csv")))

    # Noise Monitoring Files
    monitoring_path = Path("/datasets/hs2/rawdata")
    noise_files = list(monitoring_path.glob("hs2_monthly_monitoring_data_*/"))
    noise_months = len(noise_files)
    noise_files_count = 0
    for month_dir in noise_files:
        noise_files_count += len(list(month_dir.glob("*.xlsx")))

    # Get noise measurement count from database
    try:
        from app.models.noise_monitoring import NoiseMonitoringMeasurement
        from app.core.database import get_sync_db

        db = get_sync_db()
        noise_measurements = db.query(func.count(NoiseMonitoringMeasurement.id)).scalar() or 0
        db.close()
    except Exception as e:
        logger.warning(f"Failed to get noise measurements count: {e}")
        noise_measurements = 0

    # Total dataset size
    total_size_gb = 0
    if Path("/datasets/hs2").exists():
        total_size_bytes = sum(f.stat().st_size for f in Path("/datasets/hs2").rglob("*") if f.is_file())
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)

    return {
        "bim_models": {
            "count": bim_count,
            "size_mb": round(bim_size_mb, 2),
            "format": "IFC 4.3.x"
        },
        "gis_shapefiles": {
            "count": total_gis_count,
            "rawdata_count": gis_count,
            "organized_count": ecology_shp_count + property_shp_count + injunctions_file_count,
            "categories": ["Construction", "Environmental", "Parliamentary", "Ecology", "Property", "Legal"],
            "coverage": "Phase 2a Route + Ecology Surveys + Property Compensation"
        },
        "ecology_surveys": {
            "shapefiles": ecology_shp_count,
            "csv_files": ecology_csv_count,
            "survey_date": "November 2024",
            "total_files": ecology_shp_count + ecology_csv_count
        },
        "property_compensation": {
            "shapefiles": property_shp_count,
            "consultation_date": "July 2014"
        },
        "legal_injunctions": {
            "geojson_files": injunctions_file_count,
            "zones": injunctions_count,
            "description": "Court-ordered restriction zones"
        },
        "noise_monitoring": {
            "months": noise_months,
            "files": noise_files_count,
            "measurements": noise_measurements,
            "date_range": "December 2024 - September 2025"
        },
        "total_files": bim_count + total_gis_count + noise_files_count + ecology_csv_count,
        "total_size_gb": round(total_size_gb, 2)
    }


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    db: AsyncSession = Depends(get_db)
):
    """
    Get overall dashboard summary with statistics across all dimensions.
    
    **Returns:**
    - **Overall Metrics**: Total assets, readiness status distribution, average TAEM score
    - **Asset Type Breakdown**: Statistics grouped by asset type (Viaduct, Bridge, Tunnel, etc.)
    - **Contractor Breakdown**: Statistics grouped by contractor
    - **Route Breakdown**: Statistics grouped by route section (Phase 1, Phase 2a, Phase 2b)
    
    **Use Cases:**
    - Executive dashboard overview
    - Portfolio-level readiness assessment
    - Risk identification across contractors/routes
    """
    start_time = datetime.now()
    logger.info("Generating dashboard summary")
    
    try:
        from app.models.hs2 import HS2Asset
        
        # Get overall statistics
        overall_query = select(
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score')
        )
        overall_result = await db.execute(overall_query)
        overall_stats = overall_result.first()
        
        total_assets = overall_stats.total or 0
        ready = overall_stats.ready or 0
        not_ready = overall_stats.not_ready or 0
        at_risk = overall_stats.at_risk or 0
        avg_score = float(overall_stats.avg_score or 0.0)
        
        # Calculate percentages
        ready_pct = (ready / total_assets * 100) if total_assets > 0 else 0.0
        not_ready_pct = (not_ready / total_assets * 100) if total_assets > 0 else 0.0
        at_risk_pct = (at_risk / total_assets * 100) if total_assets > 0 else 0.0
        
        # Get breakdown by asset type
        asset_type_query = select(
            HS2Asset.asset_type,
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk')
        ).group_by(HS2Asset.asset_type)
        
        asset_type_result = await db.execute(asset_type_query)
        asset_type_rows = asset_type_result.all()
        
        by_asset_type = []
        for row in asset_type_rows:
            total = row.total or 0
            ready_count = row.ready or 0
            ready_percentage = (ready_count / total * 100) if total > 0 else 0.0
            
            by_asset_type.append(AssetTypeBreakdown(
                asset_type=row.asset_type,
                total=total,
                ready=ready_count,
                not_ready=row.not_ready or 0,
                at_risk=row.at_risk or 0,
                ready_pct=round(ready_percentage, 2)
            ))
        
        # Get breakdown by contractor
        contractor_query = select(
            HS2Asset.contractor,
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score')
        ).group_by(HS2Asset.contractor)
        
        contractor_result = await db.execute(contractor_query)
        contractor_rows = contractor_result.all()
        
        by_contractor = []
        for row in contractor_rows:
            total = row.total or 0
            ready_count = row.ready or 0
            ready_percentage = (ready_count / total * 100) if total > 0 else 0.0
            
            by_contractor.append(ContractorBreakdown(
                contractor=row.contractor,
                total=total,
                ready=ready_count,
                not_ready=row.not_ready or 0,
                at_risk=row.at_risk or 0,
                ready_pct=round(ready_percentage, 2),
                avg_taem_score=round(float(row.avg_score or 0.0), 2)
            ))
        
        # Get breakdown by route section
        route_query = select(
            HS2Asset.route_section,
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk')
        ).group_by(HS2Asset.route_section)
        
        route_result = await db.execute(route_query)
        route_rows = route_result.all()
        
        by_route = []
        for row in route_rows:
            total = row.total or 0
            ready_count = row.ready or 0
            ready_percentage = (ready_count / total * 100) if total > 0 else 0.0
            
            by_route.append(RouteBreakdown(
                route_section=row.route_section,
                total=total,
                ready=ready_count,
                not_ready=row.not_ready or 0,
                at_risk=row.at_risk or 0,
                ready_pct=round(ready_percentage, 2)
            ))
        
        # Build dashboard summary
        summary = DashboardSummary(
            total_assets=total_assets,
            ready=ready,
            not_ready=not_ready,
            at_risk=at_risk,
            ready_pct=round(ready_pct, 2),
            not_ready_pct=round(not_ready_pct, 2),
            at_risk_pct=round(at_risk_pct, 2),
            avg_taem_score=round(avg_score, 2),
            last_updated=datetime.utcnow(),
            by_asset_type=by_asset_type,
            by_contractor=by_contractor,
            by_route=by_route
        )
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Dashboard summary generated in {duration_ms:.2f}ms - {total_assets} assets")

        return summary

    except Exception as e:
        logger.error(f"Failed to generate dashboard summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dashboard summary: {str(e)}"
        )


@router.get("/dashboard/unified-summary")
async def get_unified_dashboard_summary(db: AsyncSession = Depends(get_db)):
    """
    Unified Dashboard Summary Endpoint
    ===================================

    Comprehensive dashboard combining:
    - Asset summary (500 synthetic assets)
    - Deliverable status (2,000 synthetic deliverables)
    - Dataset statistics (real HS2 data)
    - TAEM readiness scores
    - Priority breakdowns

    This endpoint provides all data needed for the customer showcase dashboard.
    """
    from app.models.hs2 import HS2Asset, HS2Deliverable
    from pathlib import Path

    start_time = datetime.now()
    logger.info("Generating unified dashboard summary...")

    try:
        # ==================== ASSETS SUMMARY ====================
        assets_query = select(
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score')
        )

        result = await db.execute(assets_query)
        row = result.one()

        assets_summary = {
            "total": row.total or 0,
            "ready": row.ready or 0,
            "not_ready": row.not_ready or 0,
            "at_risk": row.at_risk or 0,
            "avg_taem_score": round(float(row.avg_score or 0.0), 2),
            "is_synthetic": True
        }

        # Assets by type
        type_query = select(
            HS2Asset.asset_type,
            func.count(HS2Asset.id).label('count')
        ).group_by(HS2Asset.asset_type).order_by(func.count(HS2Asset.id).desc())

        type_result = await db.execute(type_query)
        assets_by_type = [
            {"asset_type": row.asset_type, "count": row.count}
            for row in type_result.all()
        ]

        # Assets by contractor
        contractor_query = select(
            HS2Asset.contractor,
            func.count(HS2Asset.id).label('count'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score')
        ).group_by(HS2Asset.contractor).order_by(func.count(HS2Asset.id).desc())

        contractor_result = await db.execute(contractor_query)
        assets_by_contractor = [
            {
                "contractor": row.contractor,
                "count": row.count,
                "avg_taem_score": round(float(row.avg_score or 0.0), 2)
            }
            for row in contractor_result.all()
        ]

        # ==================== DELIVERABLES SUMMARY ====================
        deliverables_query = select(
            func.count(HS2Deliverable.id).label('total'),
            func.count(case((HS2Deliverable.status == 'Approved', 1))).label('approved'),
            func.count(case((HS2Deliverable.status == 'Pending', 1))).label('pending'),
            func.count(case((HS2Deliverable.status == 'Not Started', 1))).label('not_started'),
            func.count(case((HS2Deliverable.status == 'Overdue', 1))).label('overdue')
        )

        deliv_result = await db.execute(deliverables_query)
        deliv_row = deliv_result.one()

        deliverables_summary = {
            "total": deliv_row.total or 0,
            "approved": deliv_row.approved or 0,
            "pending": deliv_row.pending or 0,
            "not_started": deliv_row.not_started or 0,
            "overdue": deliv_row.overdue or 0,
            "is_synthetic": True
        }

        # Deliverables by priority
        priority_query = select(
            HS2Deliverable.priority,
            func.count(HS2Deliverable.id).label('count')
        ).group_by(HS2Deliverable.priority).order_by(
            case(
                (HS2Deliverable.priority == 'Critical', 1),
                (HS2Deliverable.priority == 'Major', 2),
                (HS2Deliverable.priority == 'Minor', 3)
            )
        )

        priority_result = await db.execute(priority_query)
        deliverables_by_priority = [
            {"priority": row.priority, "count": row.count}
            for row in priority_result.all()
        ]

        # Overdue deliverables detail
        overdue_query = select(
            HS2Deliverable.deliverable_name,
            HS2Deliverable.due_date,
            HS2Deliverable.days_overdue,
            HS2Deliverable.priority,
            HS2Deliverable.responsible_party
        ).where(HS2Deliverable.status == 'Overdue').order_by(
            HS2Deliverable.days_overdue.desc()
        ).limit(10)

        overdue_result = await db.execute(overdue_query)
        overdue_deliverables = [
            {
                "name": row.deliverable_name,
                "due_date": row.due_date.isoformat() if row.due_date else None,
                "days_overdue": row.days_overdue,
                "priority": row.priority,
                "responsible_party": row.responsible_party
            }
            for row in overdue_result.all()
        ]

        # ==================== DATASET STATISTICS (REAL DATA) ====================
        organized_path = Path("/datasets/hs2/organized")

        # GIS shapefiles count
        gis_path = Path("/datasets/hs2/rawdata/Phase_2a_WDEIAR_September_2016/Shapefiles")
        gis_count = 0
        if gis_path.exists():
            for category in gis_path.iterdir():
                if category.is_dir():
                    gis_count += len(list(category.glob("*.shp")))

        # Ecology surveys
        ecology_path = organized_path / "ecology/surveys/november-2024/shapefiles"
        ecology_shp = len(list(ecology_path.glob("*.shp"))) if ecology_path.exists() else 0

        ecology_csv_path = organized_path / "ecology/surveys/november-2024/csv"
        ecology_csv = len(list(ecology_csv_path.glob("*.csv"))) if ecology_csv_path.exists() else 0

        # Legal injunctions - Read actual count from GeoJSON
        injunctions_count = get_injunctions_count_from_geojson(organized_path)

        # Property compensation
        property_path = organized_path / "property/compensation/july-2014/shapefiles"
        property_shp = len(list(property_path.glob("*.shp"))) if property_path.exists() else 0

        # BIM models
        bim_path = Path("/datasets/hs2/rawdata/IFC4.3.x-sample-models-main/models")
        bim_count = len(list(bim_path.rglob("*.ifc"))) if bim_path.exists() else 0

        # Noise monitoring
        monitoring_path = Path("/datasets/hs2/rawdata")
        noise_months = len(list(monitoring_path.glob("hs2_monthly_monitoring_data_*/")))

        # Get noise measurement count from database
        try:
            from app.models.noise_monitoring import NoiseMonitoringMeasurement
            noise_query = select(func.count(NoiseMonitoringMeasurement.id))
            noise_result = await db.execute(noise_query)
            noise_measurements = noise_result.scalar() or 0
        except Exception as e:
            logger.warning(f"Failed to get noise measurements: {e}")
            noise_measurements = 0

        datasets_summary = {
            "bim_models": {"count": bim_count, "is_real": True},
            "gis_shapefiles": {"count": gis_count + ecology_shp + property_shp, "is_real": True},
            "ecology_surveys": {"shapefiles": ecology_shp, "csv_files": ecology_csv, "is_real": True},
            "legal_injunctions": {"zones": injunctions_count, "is_real": True},
            "property_compensation": {"shapefiles": property_shp, "is_real": True},
            "noise_monitoring": {
                "months": noise_months,
                "measurements": noise_measurements,
                "is_real": True
            }
        }

        # ==================== BUILD UNIFIED RESPONSE ====================
        unified_summary = {
            "assets": assets_summary,
            "assets_by_type": assets_by_type,
            "assets_by_contractor": assets_by_contractor,
            "deliverables": deliverables_summary,
            "deliverables_by_priority": deliverables_by_priority,
            "overdue_deliverables": overdue_deliverables,
            "datasets": datasets_summary,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "generation_time_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2),
                "data_sources": {
                    "synthetic": ["assets", "deliverables"],
                    "real": ["datasets", "gis_layers", "noise_monitoring", "bim_models"]
                }
            }
        }

        logger.success(f"Unified dashboard summary generated in {unified_summary['metadata']['generation_time_ms']}ms")

        return unified_summary

    except Exception as e:
        logger.error(f"Failed to generate unified dashboard summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate unified dashboard summary: {str(e)}"
        )


@router.post("/dashboard/taem-compliance-check")
async def run_taem_compliance_check(db: AsyncSession = Depends(get_db)):
    """
    TAEM Compliance Check Endpoint
    ===============================

    Simulates a TAEM (Technical Assurance Evidence Model) compliance check
    across all 500 assets, analyzing:
    - Readiness status vs TAEM scores
    - Deliverable completion rates
    - Quality level compliance (PAS 128 QL-A through QL-D)
    - Risk assessment

    Returns compliance report with pass/fail criteria and recommendations.
    """
    from app.models.hs2 import HS2Asset, HS2Deliverable

    start_time = datetime.now()
    logger.info("Running TAEM compliance check...")

    try:
        # Get all assets
        assets_query = select(HS2Asset)
        result = await db.execute(assets_query)
        assets = result.scalars().all()

        # Get all deliverables
        deliverables_query = select(HS2Deliverable)
        deliv_result = await db.execute(deliverables_query)
        deliverables = deliv_result.scalars().all()

        # Analyze compliance by asset
        compliant_assets = []
        non_compliant_assets = []
        at_risk_assets = []

        for asset in assets:
            # Get deliverables for this asset
            asset_deliverables = [d for d in deliverables if d.asset_id == asset.id]
            approved_count = sum(1 for d in asset_deliverables if d.status == 'Approved')
            overdue_count = sum(1 for d in asset_deliverables if d.status == 'Overdue')

            total_deliverables = len(asset_deliverables)
            completion_rate = (approved_count / total_deliverables * 100) if total_deliverables > 0 else 0

            # Compliance criteria
            is_compliant = (
                asset.readiness_status == 'Ready' and
                asset.taem_evaluation_score >= 85 and
                completion_rate >= 90 and
                overdue_count == 0
            )

            is_at_risk = (
                asset.readiness_status == 'At Risk' or
                asset.taem_evaluation_score < 50 or
                overdue_count > 0
            )

            if is_compliant:
                compliant_assets.append({
                    "asset_id": asset.asset_id,
                    "asset_name": asset.asset_name,
                    "contractor": asset.contractor,
                    "taem_score": asset.taem_evaluation_score,
                    "completion_rate": round(completion_rate, 1)
                })
            elif is_at_risk:
                at_risk_assets.append({
                    "asset_id": asset.asset_id,
                    "asset_name": asset.asset_name,
                    "contractor": asset.contractor,
                    "taem_score": asset.taem_evaluation_score,
                    "completion_rate": round(completion_rate, 1),
                    "issues": [
                        f"Readiness: {asset.readiness_status}",
                        f"TAEM Score: {asset.taem_evaluation_score}",
                        f"Overdue Deliverables: {overdue_count}"
                    ]
                })
            else:
                non_compliant_assets.append({
                    "asset_id": asset.asset_id,
                    "asset_name": asset.asset_name,
                    "contractor": asset.contractor,
                    "taem_score": asset.taem_evaluation_score,
                    "completion_rate": round(completion_rate, 1)
                })

        # Calculate overall compliance metrics
        total_assets = len(assets)
        compliant_count = len(compliant_assets)
        non_compliant_count = len(non_compliant_assets)
        at_risk_count = len(at_risk_assets)

        overall_compliance_rate = (compliant_count / total_assets * 100) if total_assets > 0 else 0

        # Determine overall status
        if overall_compliance_rate >= 90:
            overall_status = "PASS"
            status_color = "success"
        elif overall_compliance_rate >= 70:
            overall_status = "CONDITIONAL PASS"
            status_color = "warning"
        else:
            overall_status = "FAIL"
            status_color = "error"

        # Generate recommendations
        recommendations = []

        if at_risk_count > 0:
            recommendations.append({
                "priority": "Critical",
                "issue": f"{at_risk_count} assets at risk require immediate intervention",
                "action": "Escalate to project leadership for resource allocation"
            })

        # Check Align JV performance
        align_assets = [a for a in assets if a.contractor == "Align Joint Venture"]
        align_avg_score = sum(a.taem_evaluation_score for a in align_assets) / len(align_assets) if align_assets else 0

        if align_avg_score < 60:
            recommendations.append({
                "priority": "Critical",
                "issue": f"Align Joint Venture average TAEM score ({align_avg_score:.2f}) below threshold",
                "action": "Conduct contractor performance review and implement improvement plan"
            })

        # Check overdue deliverables
        total_overdue = sum(1 for d in deliverables if d.status == 'Overdue')
        if total_overdue > 20:
            recommendations.append({
                "priority": "High",
                "issue": f"{total_overdue} deliverables overdue",
                "action": "Implement deliverable tracking dashboard and weekly review meetings"
            })

        if non_compliant_count > total_assets * 0.2:
            recommendations.append({
                "priority": "Medium",
                "issue": f"{non_compliant_count} assets non-compliant but not at risk",
                "action": "Schedule compliance workshops and provide technical support"
            })

        # Build response
        compliance_report = {
            "overall_status": overall_status,
            "status_color": status_color,
            "overall_compliance_rate": round(overall_compliance_rate, 2),
            "summary": {
                "total_assets": total_assets,
                "compliant": compliant_count,
                "non_compliant": non_compliant_count,
                "at_risk": at_risk_count
            },
            "compliant_assets": compliant_assets[:10],
            "at_risk_assets": at_risk_assets[:10],
            "recommendations": recommendations,
            "compliance_criteria": {
                "min_taem_score": 85,
                "min_completion_rate": 90,
                "max_overdue_deliverables": 0,
                "required_readiness_status": "Ready"
            },
            "metadata": {
                "check_timestamp": datetime.utcnow().isoformat(),
                "execution_time_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2),
                "is_simulated": True
            }
        }

        logger.success(f"TAEM compliance check completed in {compliance_report['metadata']['execution_time_ms']}ms - Status: {overall_status}")

        return compliance_report

    except Exception as e:
        logger.error(f"Failed to run TAEM compliance check: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run TAEM compliance check: {str(e)}"
        )


@router.get("/dashboard/by-contractor")
async def get_contractor_breakdown(
    contractor: Optional[str] = Query(None, description="Filter by specific contractor"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed breakdown by contractor.
    
    **Query Parameters:**
    - **contractor**: Optional filter for specific contractor
    
    **Returns:**
    Contractor-level statistics including asset counts, readiness distribution,
    and average TAEM scores.
    """
    logger.info(f"Generating contractor breakdown - contractor={contractor}")
    
    try:
        from app.models.hs2 import HS2Asset
        
        # Build query
        query = select(
            HS2Asset.contractor,
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score'),
            func.min(HS2Asset.taem_evaluation_score).label('min_score'),
            func.max(HS2Asset.taem_evaluation_score).label('max_score')
        ).group_by(HS2Asset.contractor)
        
        if contractor:
            query = query.where(HS2Asset.contractor == contractor)
        
        result = await db.execute(query)
        rows = result.all()
        
        breakdown = []
        for row in rows:
            total = row.total or 0
            ready_count = row.ready or 0
            ready_pct = (ready_count / total * 100) if total > 0 else 0.0
            
            breakdown.append({
                "contractor": row.contractor,
                "total_assets": total,
                "ready": ready_count,
                "not_ready": row.not_ready or 0,
                "at_risk": row.at_risk or 0,
                "ready_pct": round(ready_pct, 2),
                "avg_taem_score": round(float(row.avg_score or 0.0), 2),
                "min_taem_score": round(float(row.min_score or 0.0), 2),
                "max_taem_score": round(float(row.max_score or 0.0), 2)
            })
        
        logger.info(f"Generated contractor breakdown - {len(breakdown)} contractors")
        
        return {
            "contractors": breakdown,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate contractor breakdown: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate contractor breakdown: {str(e)}"
        )


@router.get("/dashboard/by-asset-type")
async def get_asset_type_breakdown(
    asset_type: Optional[str] = Query(None, description="Filter by specific asset type"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed breakdown by asset type.
    
    **Query Parameters:**
    - **asset_type**: Optional filter for specific asset type (e.g., "Viaduct", "Bridge", "Tunnel")
    
    **Returns:**
    Asset type statistics including counts, readiness distribution, and average scores.
    """
    logger.info(f"Generating asset type breakdown - asset_type={asset_type}")
    
    try:
        from app.models.hs2 import HS2Asset
        
        # Build query
        query = select(
            HS2Asset.asset_type,
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score')
        ).group_by(HS2Asset.asset_type)
        
        if asset_type:
            query = query.where(HS2Asset.asset_type == asset_type)
        
        result = await db.execute(query)
        rows = result.all()
        
        breakdown = []
        for row in rows:
            total = row.total or 0
            ready_count = row.ready or 0
            ready_pct = (ready_count / total * 100) if total > 0 else 0.0
            
            breakdown.append({
                "asset_type": row.asset_type,
                "total_assets": total,
                "ready": ready_count,
                "not_ready": row.not_ready or 0,
                "at_risk": row.at_risk or 0,
                "ready_pct": round(ready_pct, 2),
                "avg_taem_score": round(float(row.avg_score or 0.0), 2)
            })
        
        logger.info(f"Generated asset type breakdown - {len(breakdown)} types")
        
        return {
            "asset_types": breakdown,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate asset type breakdown: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate asset type breakdown: {str(e)}"
        )


@router.get("/dashboard/by-route")
async def get_route_breakdown(
    route_section: Optional[str] = Query(None, description="Filter by specific route section"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed breakdown by route section.
    
    **Query Parameters:**
    - **route_section**: Optional filter for specific route section (e.g., "Phase 1", "Phase 2a", "Phase 2b")
    
    **Returns:**
    Route-level statistics including asset counts, readiness distribution, and progress metrics.
    """
    logger.info(f"Generating route breakdown - route_section={route_section}")
    
    try:
        from app.models.hs2 import HS2Asset
        
        # Build query
        query = select(
            HS2Asset.route_section,
            func.count(HS2Asset.id).label('total'),
            func.count(case((HS2Asset.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Asset.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Asset.readiness_status == 'At Risk', 1))).label('at_risk'),
            func.avg(HS2Asset.taem_evaluation_score).label('avg_score')
        ).group_by(HS2Asset.route_section)
        
        if route_section:
            query = query.where(HS2Asset.route_section == route_section)
        
        result = await db.execute(query)
        rows = result.all()
        
        breakdown = []
        for row in rows:
            total = row.total or 0
            ready_count = row.ready or 0
            ready_pct = (ready_count / total * 100) if total > 0 else 0.0
            
            breakdown.append({
                "route_section": row.route_section,
                "total_assets": total,
                "ready": ready_count,
                "not_ready": row.not_ready or 0,
                "at_risk": row.at_risk or 0,
                "ready_pct": round(ready_pct, 2),
                "avg_taem_score": round(float(row.avg_score or 0.0), 2)
            })
        
        logger.info(f"Generated route breakdown - {len(breakdown)} routes")
        
        return {
            "routes": breakdown,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate route breakdown: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate route breakdown: {str(e)}"
        )


@router.get("/dashboard/trends")
async def get_readiness_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get readiness trends over time.
    
    **Query Parameters:**
    - **days**: Number of days to analyze (default: 30, max: 365)
    
    **Returns:**
    Time-series data showing readiness status changes and TAEM score trends.
    
    **Note:** Requires evaluation history to be populated.
    """
    logger.info(f"Generating readiness trends for {days} days")
    
    try:
        from app.models.hs2 import HS2Evaluation
        from datetime import timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get evaluation trends
        query = select(
            func.date_trunc('day', HS2Evaluation.evaluation_date).label('date'),
            func.count(HS2Evaluation.id).label('evaluations'),
            func.avg(HS2Evaluation.overall_score).label('avg_score'),
            func.count(case((HS2Evaluation.readiness_status == 'Ready', 1))).label('ready'),
            func.count(case((HS2Evaluation.readiness_status == 'Not Ready', 1))).label('not_ready'),
            func.count(case((HS2Evaluation.readiness_status == 'At Risk', 1))).label('at_risk')
        ).where(
            HS2Evaluation.evaluation_date >= start_date
        ).group_by(
            func.date_trunc('day', HS2Evaluation.evaluation_date)
        ).order_by(
            func.date_trunc('day', HS2Evaluation.evaluation_date)
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        trends = []
        for row in rows:
            trends.append({
                "date": row.date.isoformat() if row.date else None,
                "evaluations_count": row.evaluations or 0,
                "avg_taem_score": round(float(row.avg_score or 0.0), 2),
                "ready": row.ready or 0,
                "not_ready": row.not_ready or 0,
                "at_risk": row.at_risk or 0
            })
        
        logger.info(f"Generated trends for {len(trends)} days")
        
        return {
            "trends": trends,
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate readiness trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate readiness trends: {str(e)}"
        )
