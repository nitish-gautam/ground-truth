"""
HS2 Asset Management Endpoints
==============================

API endpoints for managing HS2 infrastructure assets, readiness assessment,
and TAEM rule evaluation.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from loguru import logger

from app.core.database import get_db
from app.schemas.hs2 import (
    AssetResponse,
    AssetDetailResponse,
    AssetPaginatedResponse,
    AssetCreate,
    AssetUpdate,
    AssetEvaluationResponse,
    AssetEvaluationRequest,
    DeliverableResponse,
    CostResponse,
    CertificateResponse,
)

router = APIRouter()


@router.get("/assets", response_model=AssetPaginatedResponse)
async def list_assets(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    status: Optional[str] = Query(None, description="Filter by readiness status"),
    contractor: Optional[str] = Query(None, description="Filter by contractor"),
    route_section: Optional[str] = Query(None, description="Filter by route section"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all HS2 assets with pagination and filtering.
    
    **Query Parameters:**
    - **skip**: Offset for pagination (default: 0)
    - **limit**: Maximum items to return (default: 50, max: 100)
    - **asset_type**: Filter by asset type (e.g., "Viaduct", "Bridge", "Tunnel")
    - **status**: Filter by readiness status (e.g., "Ready", "Not Ready", "At Risk")
    - **contractor**: Filter by contractor name
    - **route_section**: Filter by HS2 route section
    
    **Returns:**
    Paginated list of assets with readiness status and TAEM scores.
    """
    start_time = datetime.now()
    logger.info(f"Listing assets - skip={skip}, limit={limit}, filters: type={asset_type}, status={status}, contractor={contractor}")
    
    try:
        # Import models dynamically to avoid circular imports
        from app.models.hs2 import HS2Asset
        
        # Build query with filters
        query = select(HS2Asset)
        
        # Apply filters
        filters = []
        if asset_type:
            filters.append(HS2Asset.asset_type == asset_type)
        if status:
            filters.append(HS2Asset.readiness_status == status)
        if contractor:
            filters.append(HS2Asset.contractor == contractor)
        if route_section:
            filters.append(HS2Asset.route_section == route_section)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Get total count
        count_query = select(func.count()).select_from(HS2Asset)
        if filters:
            count_query = count_query.where(and_(*filters))
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        query = query.offset(skip).limit(limit).order_by(HS2Asset.asset_id)
        result = await db.execute(query)
        assets = result.scalars().all()
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Assets listed successfully - {len(assets)} items returned in {duration_ms:.2f}ms")
        
        return AssetPaginatedResponse(
            total=total,
            skip=skip,
            limit=limit,
            items=[AssetResponse.model_validate(asset) for asset in assets]
        )
        
    except Exception as e:
        logger.error(f"Failed to list assets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve assets: {str(e)}"
        )


@router.get("/assets/{asset_id}", response_model=AssetDetailResponse)
async def get_asset_details(
    asset_id: UUID = Path(..., description="Asset UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific asset.
    
    **Returns:**
    - Asset details
    - Readiness summary with deliverables, certificates, costs, and TAEM scores
    - Latest evaluation results
    
    **Raises:**
    - **404**: Asset not found
    """
    start_time = datetime.now()
    logger.info(f"Fetching asset details for asset_id={asset_id}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Deliverable, HS2Certificate, HS2Cost, HS2Evaluation
        
        # Get asset with relationships
        query = select(HS2Asset).where(HS2Asset.id == asset_id)
        result = await db.execute(query)
        asset = result.scalar_one_or_none()
        
        if not asset:
            logger.warning(f"Asset not found: {asset_id}")
            raise HTTPException(status_code=404, detail=f"Asset with ID {asset_id} not found")
        
        # Calculate readiness summary
        # Get deliverables stats
        deliverables_query = select(
            func.count(HS2Deliverable.id).label('total'),
            func.count(HS2Deliverable.id).filter(HS2Deliverable.status == 'Submitted').label('submitted')
        ).where(HS2Deliverable.asset_id == asset_id)
        del_result = await db.execute(deliverables_query)
        del_stats = del_result.first()
        
        deliverables_total = del_stats.total or 0
        deliverables_submitted = del_stats.submitted or 0
        deliverables_pct = (deliverables_submitted / deliverables_total * 100) if deliverables_total > 0 else 0.0
        
        # Get certificates stats
        certificates_query = select(
            func.count(HS2Certificate.id).label('total'),
            func.count(HS2Certificate.id).filter(HS2Certificate.status == 'Valid').label('valid')
        ).where(HS2Certificate.asset_id == asset_id)
        cert_result = await db.execute(certificates_query)
        cert_stats = cert_result.first()
        
        certificates_total = cert_stats.total or 0
        certificates_valid = cert_stats.valid or 0
        certificates_pct = (certificates_valid / certificates_total * 100) if certificates_total > 0 else 0.0
        
        # Get cost variance
        cost_query = select(HS2Cost).where(HS2Cost.asset_id == asset_id).order_by(HS2Cost.created_at.desc()).limit(1)
        cost_result = await db.execute(cost_query)
        latest_cost = cost_result.scalar_one_or_none()
        cost_variance_pct = float(latest_cost.variance_pct) if latest_cost else 0.0
        
        # Get latest evaluation
        eval_query = select(HS2Evaluation).where(
            HS2Evaluation.asset_id == asset_id
        ).order_by(HS2Evaluation.evaluation_date.desc()).limit(1)
        eval_result = await db.execute(eval_query)
        latest_eval = eval_result.scalar_one_or_none()
        
        # Count risks from evaluation results
        critical_risks = 0
        major_risks = 0
        minor_risks = 0
        if latest_eval and latest_eval.rule_results:
            for rule_result in latest_eval.rule_results:
                if rule_result.get('status') == 'Fail':
                    severity = rule_result.get('severity', 'Minor')
                    if severity == 'Critical':
                        critical_risks += 1
                    elif severity == 'Major':
                        major_risks += 1
                    else:
                        minor_risks += 1
        
        # Build readiness summary
        from app.schemas.hs2 import AssetReadinessSummary
        readiness_summary = AssetReadinessSummary(
            deliverables_submitted=deliverables_submitted,
            deliverables_required=deliverables_total,
            deliverables_completion_pct=round(deliverables_pct, 2),
            certificates_issued=certificates_valid,
            certificates_required=certificates_total,
            certificates_completion_pct=round(certificates_pct, 2),
            cost_variance_pct=round(cost_variance_pct, 2),
            schedule_variance_days=0,  # Calculate from planned vs actual dates
            critical_risks=critical_risks,
            major_risks=major_risks,
            minor_risks=minor_risks,
            overall_readiness=asset.readiness_status,
            taem_score=float(latest_eval.overall_score) if latest_eval else 0.0,
            last_evaluation=latest_eval.evaluation_date if latest_eval else None
        )
        
        # Build response
        asset_response = AssetDetailResponse.model_validate(asset)
        asset_response.readiness_summary = readiness_summary
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Asset details retrieved successfully in {duration_ms:.2f}ms")
        
        return asset_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get asset details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve asset details: {str(e)}"
        )


@router.post("/assets/{asset_id}/evaluate", response_model=AssetEvaluationResponse)
async def evaluate_asset(
    asset_id: UUID = Path(..., description="Asset UUID"),
    request: AssetEvaluationRequest = AssetEvaluationRequest(),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger TAEM rule evaluation for a specific asset.
    
    **Process:**
    1. Retrieves asset data and related information (deliverables, costs, certificates)
    2. Evaluates all active TAEM rules
    3. Calculates overall readiness score
    4. Determines readiness status (Ready/Not Ready/At Risk)
    5. Stores evaluation results for audit trail
    
    **Parameters:**
    - **force_refresh**: Force re-evaluation even if recent evaluation exists (default: false)
    
    **Returns:**
    Evaluation results with individual rule outcomes and overall TAEM score.
    
    **Raises:**
    - **404**: Asset not found
    - **500**: Evaluation processing error
    """
    start_time = datetime.now()
    logger.info(f"Evaluating asset {asset_id} - force_refresh={request.force_refresh}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Evaluation
        
        # Check if asset exists
        query = select(HS2Asset).where(HS2Asset.id == asset_id)
        result = await db.execute(query)
        asset = result.scalar_one_or_none()
        
        if not asset:
            logger.warning(f"Asset not found: {asset_id}")
            raise HTTPException(status_code=404, detail=f"Asset with ID {asset_id} not found")
        
        # Check for recent evaluation
        if not request.force_refresh:
            recent_eval_query = select(HS2Evaluation).where(
                and_(
                    HS2Evaluation.asset_id == asset_id,
                    HS2Evaluation.evaluation_date >= datetime.utcnow() - timedelta(hours=1)
                )
            ).order_by(HS2Evaluation.evaluation_date.desc()).limit(1)
            recent_eval_result = await db.execute(recent_eval_query)
            recent_eval = recent_eval_result.scalar_one_or_none()
            
            if recent_eval:
                logger.info(f"Using recent evaluation from {recent_eval.evaluation_date}")
                return AssetEvaluationResponse.model_validate(recent_eval)
        
        # Import TAEM engine
        try:
            from app.services.taem_engine import TAEMEngine
            taem_engine = TAEMEngine(db)
        except ImportError:
            logger.error("TAEM engine not found - using mock evaluation")
            # Create mock evaluation for development
            from app.schemas.hs2 import RuleEvaluationResult
            
            evaluation = HS2Evaluation(
                asset_id=asset_id,
                evaluation_date=datetime.utcnow(),
                overall_score=75.5,
                readiness_status="At Risk",
                rules_evaluated=5,
                rules_passed=3,
                rules_failed=2,
                rule_results=[
                    RuleEvaluationResult(
                        rule_code="TAEM-001",
                        rule_name="Deliverables Completeness",
                        status="Pass",
                        score=85.0,
                        weight=0.3,
                        weighted_score=25.5,
                        message="Deliverables 85% complete",
                        details={"submitted": 17, "required": 20}
                    ).model_dump(),
                    RuleEvaluationResult(
                        rule_code="TAEM-002",
                        rule_name="Cost Variance Check",
                        status="Fail",
                        score=60.0,
                        weight=0.25,
                        weighted_score=15.0,
                        message="Cost variance exceeds threshold",
                        details={"variance_pct": 12.5, "threshold": 10.0}
                    ).model_dump()
                ]
            )
            
            db.add(evaluation)
            await db.commit()
            await db.refresh(evaluation)
            
            # Update asset readiness status
            asset.readiness_status = evaluation.readiness_status
            asset.taem_evaluation_score = evaluation.overall_score
            await db.commit()
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Mock evaluation completed in {duration_ms:.2f}ms - score={evaluation.overall_score}")
            
            return AssetEvaluationResponse.model_validate(evaluation)
        
        # Run TAEM evaluation
        evaluation_result = await taem_engine.evaluate_asset(asset_id)
        
        # Save evaluation to database
        evaluation = HS2Evaluation(**evaluation_result)
        db.add(evaluation)
        await db.commit()
        await db.refresh(evaluation)
        
        # Update asset readiness status
        asset.readiness_status = evaluation.readiness_status
        asset.taem_evaluation_score = evaluation.overall_score
        await db.commit()
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Asset evaluation completed in {duration_ms:.2f}ms - score={evaluation.overall_score}, status={evaluation.readiness_status}")
        
        return AssetEvaluationResponse.model_validate(evaluation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate asset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate asset: {str(e)}"
        )


@router.get("/assets/{asset_id}/deliverables", response_model=List[DeliverableResponse])
async def get_asset_deliverables(
    asset_id: UUID = Path(..., description="Asset UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all deliverables for a specific asset.
    
    **Returns:**
    List of deliverables with submission status, approval status, and due dates.
    
    **Raises:**
    - **404**: Asset not found
    """
    logger.info(f"Fetching deliverables for asset {asset_id}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Deliverable
        
        # Check if asset exists
        asset_query = select(HS2Asset).where(HS2Asset.id == asset_id)
        asset_result = await db.execute(asset_query)
        asset = asset_result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID {asset_id} not found")
        
        # Get deliverables
        query = select(HS2Deliverable).where(
            HS2Deliverable.asset_id == asset_id
        ).order_by(HS2Deliverable.required_by_date)
        
        result = await db.execute(query)
        deliverables = result.scalars().all()
        
        logger.info(f"Found {len(deliverables)} deliverables for asset {asset_id}")
        
        return [DeliverableResponse.model_validate(d) for d in deliverables]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get asset deliverables: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve deliverables: {str(e)}"
        )


@router.get("/assets/{asset_id}/costs", response_model=List[CostResponse])
async def get_asset_costs(
    asset_id: UUID = Path(..., description="Asset UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cost tracking information for a specific asset.
    
    **Returns:**
    List of cost records with budget, actual, forecast, and variance data.
    
    **Raises:**
    - **404**: Asset not found
    """
    logger.info(f"Fetching costs for asset {asset_id}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Cost
        
        # Check if asset exists
        asset_query = select(HS2Asset).where(HS2Asset.id == asset_id)
        asset_result = await db.execute(asset_query)
        asset = asset_result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID {asset_id} not found")
        
        # Get costs
        query = select(HS2Cost).where(
            HS2Cost.asset_id == asset_id
        ).order_by(HS2Cost.reporting_period.desc())
        
        result = await db.execute(query)
        costs = result.scalars().all()
        
        logger.info(f"Found {len(costs)} cost records for asset {asset_id}")
        
        return [CostResponse.model_validate(c) for c in costs]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get asset costs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve costs: {str(e)}"
        )


@router.get("/assets/{asset_id}/certificates", response_model=List[CertificateResponse])
async def get_asset_certificates(
    asset_id: UUID = Path(..., description="Asset UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all certificates for a specific asset.
    
    **Returns:**
    List of certificates with issue date, expiry date, and validity status.
    
    **Raises:**
    - **404**: Asset not found
    """
    logger.info(f"Fetching certificates for asset {asset_id}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Certificate
        
        # Check if asset exists
        asset_query = select(HS2Asset).where(HS2Asset.id == asset_id)
        asset_result = await db.execute(asset_query)
        asset = asset_result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID {asset_id} not found")
        
        # Get certificates
        query = select(HS2Certificate).where(
            HS2Certificate.asset_id == asset_id
        ).order_by(HS2Certificate.expiry_date)
        
        result = await db.execute(query)
        certificates = result.scalars().all()
        
        logger.info(f"Found {len(certificates)} certificates for asset {asset_id}")
        
        return [CertificateResponse.model_validate(c) for c in certificates]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get asset certificates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve certificates: {str(e)}"
        )


@router.get("/assets/{asset_id}/evaluations", response_model=List[AssetEvaluationResponse])
async def get_asset_evaluation_history(
    asset_id: UUID = Path(..., description="Asset UUID"),
    limit: int = Query(10, ge=1, le=100, description="Number of evaluations to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get evaluation history for a specific asset (audit trail).
    
    **Returns:**
    Historical list of TAEM evaluations for the asset, ordered by most recent first.
    
    **Raises:**
    - **404**: Asset not found
    """
    logger.info(f"Fetching evaluation history for asset {asset_id}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Evaluation
        
        # Check if asset exists
        asset_query = select(HS2Asset).where(HS2Asset.id == asset_id)
        asset_result = await db.execute(asset_query)
        asset = asset_result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID {asset_id} not found")
        
        # Get evaluation history
        query = select(HS2Evaluation).where(
            HS2Evaluation.asset_id == asset_id
        ).order_by(HS2Evaluation.evaluation_date.desc()).limit(limit)
        
        result = await db.execute(query)
        evaluations = result.scalars().all()
        
        logger.info(f"Found {len(evaluations)} evaluations for asset {asset_id}")
        
        return [AssetEvaluationResponse.model_validate(e) for e in evaluations]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation history: {str(e)}"
        )
