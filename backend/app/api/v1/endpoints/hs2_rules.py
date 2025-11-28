"""
HS2 TAEM Rules Endpoints
========================

API endpoints for managing TAEM (Technical Assurance and Evaluation Model) rules,
rule evaluation, and audit trails.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from loguru import logger

from app.core.database import get_db
from app.schemas.hs2 import (
    TAEMRuleResponse,
    TAEMRuleUpdate,
    AssetEvaluationResponse,
    EvaluationPaginatedResponse,
)

router = APIRouter()


@router.get("/taem/rules", response_model=List[TAEMRuleResponse])
async def list_taem_rules(
    category: Optional[str] = Query(None, description="Filter by category"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    active_only: bool = Query(True, description="Show only active rules"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all TAEM rules with optional filtering.
    
    **Query Parameters:**
    - **category**: Filter by category (e.g., "Deliverables", "Costs", "Certificates", "Schedule")
    - **severity**: Filter by severity (e.g., "Critical", "Major", "Minor")
    - **active_only**: Show only active rules (default: true)
    
    **Returns:**
    List of TAEM rules with configuration details.
    """
    logger.info(f"Listing TAEM rules - category={category}, severity={severity}, active_only={active_only}")
    
    try:
        from app.models.hs2 import HS2Rule
        
        # Build query with filters
        query = select(HS2Rule)
        
        filters = []
        if category:
            filters.append(HS2Rule.rule_category == category)
        if severity:
            filters.append(HS2Rule.severity == severity)
        if active_only:
            filters.append(HS2Rule.is_active == True)
        
        if filters:
            query = query.where(and_(*filters))
        
        query = query.order_by(HS2Rule.rule_code)
        result = await db.execute(query)
        rules = result.scalars().all()
        
        logger.info(f"Found {len(rules)} TAEM rules")
        
        return [TAEMRuleResponse.model_validate(r) for r in rules]
        
    except Exception as e:
        logger.error(f"Failed to list TAEM rules: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve TAEM rules: {str(e)}"
        )


@router.get("/taem/rules/{rule_id}", response_model=TAEMRuleResponse)
async def get_taem_rule(
    rule_id: UUID = Path(..., description="Rule UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific TAEM rule.
    
    **Returns:**
    Rule configuration including code, name, description, category, severity, and weight.
    
    **Raises:**
    - **404**: Rule not found
    """
    logger.info(f"Fetching TAEM rule {rule_id}")
    
    try:
        from app.models.hs2 import HS2Rule
        
        query = select(HS2Rule).where(HS2Rule.id == rule_id)
        result = await db.execute(query)
        rule = result.scalar_one_or_none()
        
        if not rule:
            logger.warning(f"Rule not found: {rule_id}")
            raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found")
        
        logger.info(f"Rule retrieved successfully: {rule.rule_code}")
        return TAEMRuleResponse.model_validate(rule)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get TAEM rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve TAEM rule: {str(e)}"
        )


@router.patch("/taem/rules/{rule_id}", response_model=TAEMRuleResponse)
async def update_taem_rule(
    rule_id: UUID = Path(..., description="Rule UUID"),
    rule_data: TAEMRuleUpdate = TAEMRuleUpdate(),
    db: AsyncSession = Depends(get_db)
):
    """
    Update TAEM rule configuration (tinkerability).
    
    **Use Cases:**
    - Adjust rule weight in overall score calculation
    - Modify threshold values for pass/fail criteria
    - Enable/disable specific rules
    - Update severity classification
    
    **Request Body:**
    All fields are optional - only provided fields will be updated.
    
    **Returns:**
    Updated rule configuration.
    
    **Raises:**
    - **404**: Rule not found
    - **422**: Validation error (e.g., invalid weight value)
    """
    logger.info(f"Updating TAEM rule {rule_id}")
    
    try:
        from app.models.hs2 import HS2Rule
        
        # Get existing rule
        query = select(HS2Rule).where(HS2Rule.id == rule_id)
        result = await db.execute(query)
        rule = result.scalar_one_or_none()
        
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found")
        
        # Update fields
        update_data = rule_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(rule, field, value)
        
        await db.commit()
        await db.refresh(rule)
        
        logger.info(f"Rule updated successfully: {rule.rule_code}")
        return TAEMRuleResponse.model_validate(rule)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update TAEM rule: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update TAEM rule: {str(e)}"
        )


@router.post("/taem/evaluate-all", status_code=202)
async def evaluate_all_assets(
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(False, description="Force re-evaluation of all assets"),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger TAEM evaluation for all assets (background task).
    
    **Process:**
    1. Queues evaluation jobs for all assets
    2. Runs evaluations in the background
    3. Returns immediately with job status
    
    **Query Parameters:**
    - **force_refresh**: Force re-evaluation even for recently evaluated assets (default: false)
    
    **Returns:**
    Job status and estimated completion time.
    
    **Status Code:** 202 Accepted (processing in background)
    """
    logger.info(f"Triggering evaluation for all assets - force_refresh={force_refresh}")
    
    try:
        from app.models.hs2 import HS2Asset
        
        # Get all assets
        query = select(func.count(HS2Asset.id))
        result = await db.execute(query)
        total_assets = result.scalar() or 0
        
        if total_assets == 0:
            raise HTTPException(status_code=404, detail="No assets found to evaluate")
        
        # Queue background evaluation task
        # Note: This is a simplified version. In production, use Celery or similar
        async def evaluate_all_assets_task():
            """Background task to evaluate all assets."""
            from app.models.hs2 import HS2Asset
            
            assets_query = select(HS2Asset.id)
            assets_result = await db.execute(assets_query)
            asset_ids = [row[0] for row in assets_result.all()]
            
            logger.info(f"Starting evaluation of {len(asset_ids)} assets")
            
            for asset_id in asset_ids:
                try:
                    # Import TAEM engine
                    from app.services.taem_engine import TAEMEngine
                    taem_engine = TAEMEngine(db)
                    
                    # Run evaluation
                    await taem_engine.evaluate_asset(asset_id, force_refresh=force_refresh)
                    logger.debug(f"Evaluated asset {asset_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate asset {asset_id}: {str(e)}")
                    continue
            
            logger.info(f"Completed evaluation of all assets")
        
        # Add task to background
        background_tasks.add_task(evaluate_all_assets_task)
        
        # Estimate completion time (rough estimate: 2 seconds per asset)
        estimated_duration_seconds = total_assets * 2
        estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_duration_seconds)
        
        logger.info(f"Queued evaluation for {total_assets} assets")
        
        return {
            "status": "accepted",
            "message": f"Evaluation queued for {total_assets} assets",
            "total_assets": total_assets,
            "estimated_completion": estimated_completion,
            "job_id": None  # In production, return actual job ID
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue evaluation: {str(e)}"
        )


@router.get("/taem/evaluations", response_model=EvaluationPaginatedResponse)
async def get_evaluation_history(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    asset_id: Optional[UUID] = Query(None, description="Filter by asset ID"),
    from_date: Optional[datetime] = Query(None, description="Filter evaluations from date"),
    to_date: Optional[datetime] = Query(None, description="Filter evaluations to date"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get evaluation history (audit trail) across all assets.
    
    **Query Parameters:**
    - **skip**: Offset for pagination (default: 0)
    - **limit**: Maximum items to return (default: 50, max: 100)
    - **asset_id**: Filter by specific asset
    - **from_date**: Filter evaluations from date (ISO 8601 format)
    - **to_date**: Filter evaluations to date (ISO 8601 format)
    
    **Returns:**
    Paginated list of historical evaluations for audit purposes.
    """
    logger.info(f"Fetching evaluation history - skip={skip}, limit={limit}, asset_id={asset_id}")
    
    try:
        from app.models.hs2 import HS2Evaluation
        
        # Build query with filters
        query = select(HS2Evaluation)
        
        filters = []
        if asset_id:
            filters.append(HS2Evaluation.asset_id == asset_id)
        if from_date:
            filters.append(HS2Evaluation.evaluation_date >= from_date)
        if to_date:
            filters.append(HS2Evaluation.evaluation_date <= to_date)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Get total count
        count_query = select(func.count()).select_from(HS2Evaluation)
        if filters:
            count_query = count_query.where(and_(*filters))
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        query = query.offset(skip).limit(limit).order_by(HS2Evaluation.evaluation_date.desc())
        result = await db.execute(query)
        evaluations = result.scalars().all()
        
        logger.info(f"Found {len(evaluations)} evaluations")
        
        return EvaluationPaginatedResponse(
            total=total,
            skip=skip,
            limit=limit,
            items=[AssetEvaluationResponse.model_validate(e) for e in evaluations]
        )
        
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation history: {str(e)}"
        )


@router.get("/taem/rules/statistics/summary")
async def get_rules_statistics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get TAEM rules statistics.
    
    **Returns:**
    - Total rules count
    - Active/inactive distribution
    - Category distribution
    - Severity distribution
    """
    logger.info("Fetching TAEM rules statistics")
    
    try:
        from app.models.hs2 import HS2Rule
        
        # Total count
        total_query = select(func.count(HS2Rule.id))
        total_result = await db.execute(total_query)
        total = total_result.scalar() or 0
        
        # Active count
        active_query = select(func.count(HS2Rule.id)).where(HS2Rule.is_active == True)
        active_result = await db.execute(active_query)
        active = active_result.scalar() or 0
        
        # Category distribution
        category_query = select(
            HS2Rule.rule_category,
            func.count(HS2Rule.id)
        ).group_by(HS2Rule.rule_category)
        category_result = await db.execute(category_query)
        category_distribution = dict(category_result.all())
        
        # Severity distribution
        severity_query = select(
            HS2Rule.severity,
            func.count(HS2Rule.id)
        ).group_by(HS2Rule.severity)
        severity_result = await db.execute(severity_query)
        severity_distribution = dict(severity_result.all())
        
        logger.info(f"Statistics retrieved: total={total}, active={active}")
        
        return {
            "total_rules": total,
            "active_rules": active,
            "inactive_rules": total - active,
            "category_distribution": category_distribution,
            "severity_distribution": severity_distribution,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get rules statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )
