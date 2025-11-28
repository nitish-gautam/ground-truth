"""
HS2 Deliverables Management Endpoints
====================================

API endpoints for managing project deliverables, submission tracking,
and approval workflows.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from loguru import logger

from app.core.database import get_db
from app.schemas.hs2 import (
    DeliverableResponse,
    DeliverablePaginatedResponse,
    DeliverableCreate,
    DeliverableUpdate,
)

router = APIRouter()


@router.get("/deliverables", response_model=DeliverablePaginatedResponse)
async def list_deliverables(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    deliverable_type: Optional[str] = Query(None, description="Filter by deliverable type"),
    asset_id: Optional[UUID] = Query(None, description="Filter by asset ID"),
    overdue_only: bool = Query(False, description="Show only overdue deliverables"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all deliverables with pagination and filtering.
    
    **Query Parameters:**
    - **skip**: Offset for pagination (default: 0)
    - **limit**: Maximum items to return (default: 50, max: 100)
    - **status**: Filter by status (e.g., "Submitted", "Pending", "Approved", "Rejected")
    - **deliverable_type**: Filter by type (e.g., "Design Report", "As-Built Drawing")
    - **asset_id**: Filter by specific asset
    - **overdue_only**: Show only overdue deliverables (default: false)
    
    **Returns:**
    Paginated list of deliverables with submission and approval status.
    """
    start_time = datetime.now()
    logger.info(f"Listing deliverables - skip={skip}, limit={limit}, filters: status={status}, type={deliverable_type}")
    
    try:
        from app.models.hs2 import HS2Deliverable
        
        # Build query with filters
        query = select(HS2Deliverable)
        
        # Apply filters
        filters = []
        if status:
            filters.append(HS2Deliverable.status == status)
        if deliverable_type:
            filters.append(HS2Deliverable.deliverable_type == deliverable_type)
        if asset_id:
            filters.append(HS2Deliverable.asset_id == asset_id)
        if overdue_only:
            filters.append(
                and_(
                    HS2Deliverable.required_by_date < datetime.utcnow(),
                    HS2Deliverable.status != 'Submitted'
                )
            )
        
        if filters:
            query = query.where(and_(*filters))
        
        # Get total count
        count_query = select(func.count()).select_from(HS2Deliverable)
        if filters:
            count_query = count_query.where(and_(*filters))
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        query = query.offset(skip).limit(limit).order_by(HS2Deliverable.required_by_date)
        result = await db.execute(query)
        deliverables = result.scalars().all()
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Deliverables listed successfully - {len(deliverables)} items returned in {duration_ms:.2f}ms")
        
        return DeliverablePaginatedResponse(
            total=total,
            skip=skip,
            limit=limit,
            items=[DeliverableResponse.model_validate(d) for d in deliverables]
        )
        
    except Exception as e:
        logger.error(f"Failed to list deliverables: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve deliverables: {str(e)}"
        )


@router.get("/deliverables/{deliverable_id}", response_model=DeliverableResponse)
async def get_deliverable(
    deliverable_id: UUID = Path(..., description="Deliverable UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific deliverable.
    
    **Returns:**
    Deliverable details including submission status, approval status, and timeline.
    
    **Raises:**
    - **404**: Deliverable not found
    """
    logger.info(f"Fetching deliverable {deliverable_id}")
    
    try:
        from app.models.hs2 import HS2Deliverable
        
        query = select(HS2Deliverable).where(HS2Deliverable.id == deliverable_id)
        result = await db.execute(query)
        deliverable = result.scalar_one_or_none()
        
        if not deliverable:
            logger.warning(f"Deliverable not found: {deliverable_id}")
            raise HTTPException(status_code=404, detail=f"Deliverable with ID {deliverable_id} not found")
        
        logger.info(f"Deliverable retrieved successfully")
        return DeliverableResponse.model_validate(deliverable)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deliverable: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve deliverable: {str(e)}"
        )


@router.post("/deliverables", response_model=DeliverableResponse, status_code=201)
async def create_deliverable(
    deliverable_data: DeliverableCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new deliverable.
    
    **Request Body:**
    - **asset_id**: Associated asset UUID
    - **deliverable_type**: Type of deliverable
    - **deliverable_name**: Name/description
    - **required_by_date**: Required submission date
    - **responsible_party**: Party responsible for submission
    
    **Returns:**
    Created deliverable with assigned UUID.
    
    **Raises:**
    - **404**: Asset not found
    - **422**: Validation error
    """
    logger.info(f"Creating deliverable for asset {deliverable_data.asset_id}")
    
    try:
        from app.models.hs2 import HS2Asset, HS2Deliverable
        
        # Verify asset exists
        asset_query = select(HS2Asset).where(HS2Asset.id == deliverable_data.asset_id)
        asset_result = await db.execute(asset_query)
        asset = asset_result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID {deliverable_data.asset_id} not found")
        
        # Create deliverable
        deliverable = HS2Deliverable(
            **deliverable_data.model_dump(),
            status="Pending",  # Default status
            submission_date=None,
            approval_status=None,
            days_overdue=None
        )
        
        db.add(deliverable)
        await db.commit()
        await db.refresh(deliverable)
        
        logger.info(f"Deliverable created successfully with ID {deliverable.id}")
        return DeliverableResponse.model_validate(deliverable)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create deliverable: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create deliverable: {str(e)}"
        )


@router.put("/deliverables/{deliverable_id}", response_model=DeliverableResponse)
async def update_deliverable(
    deliverable_id: UUID = Path(..., description="Deliverable UUID"),
    deliverable_data: DeliverableUpdate = DeliverableUpdate(),
    db: AsyncSession = Depends(get_db)
):
    """
    Update deliverable information.
    
    **Common Use Cases:**
    - Mark as submitted: Set status="Submitted", submission_date=now
    - Update approval status: Set approval_status="Approved" or "Rejected"
    - Extend deadline: Update required_by_date
    
    **Request Body:**
    All fields are optional - only provided fields will be updated.
    
    **Returns:**
    Updated deliverable.
    
    **Raises:**
    - **404**: Deliverable not found
    - **422**: Validation error
    """
    logger.info(f"Updating deliverable {deliverable_id}")
    
    try:
        from app.models.hs2 import HS2Deliverable
        
        # Get existing deliverable
        query = select(HS2Deliverable).where(HS2Deliverable.id == deliverable_id)
        result = await db.execute(query)
        deliverable = result.scalar_one_or_none()
        
        if not deliverable:
            raise HTTPException(status_code=404, detail=f"Deliverable with ID {deliverable_id} not found")
        
        # Update fields
        update_data = deliverable_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(deliverable, field, value)
        
        # Calculate days overdue if status changes
        if 'status' in update_data and deliverable.required_by_date:
            if deliverable.status != 'Submitted' and deliverable.required_by_date < datetime.utcnow():
                deliverable.days_overdue = (datetime.utcnow() - deliverable.required_by_date).days
            else:
                deliverable.days_overdue = None
        
        await db.commit()
        await db.refresh(deliverable)
        
        logger.info(f"Deliverable updated successfully")
        return DeliverableResponse.model_validate(deliverable)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update deliverable: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update deliverable: {str(e)}"
        )


@router.delete("/deliverables/{deliverable_id}", status_code=204)
async def delete_deliverable(
    deliverable_id: UUID = Path(..., description="Deliverable UUID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a deliverable.
    
    **Warning:** This action cannot be undone.
    
    **Raises:**
    - **404**: Deliverable not found
    """
    logger.info(f"Deleting deliverable {deliverable_id}")
    
    try:
        from app.models.hs2 import HS2Deliverable
        
        query = select(HS2Deliverable).where(HS2Deliverable.id == deliverable_id)
        result = await db.execute(query)
        deliverable = result.scalar_one_or_none()
        
        if not deliverable:
            raise HTTPException(status_code=404, detail=f"Deliverable with ID {deliverable_id} not found")
        
        await db.delete(deliverable)
        await db.commit()
        
        logger.info(f"Deliverable deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deliverable: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete deliverable: {str(e)}"
        )


@router.get("/deliverables/statistics/summary")
async def get_deliverables_statistics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get overall deliverables statistics.
    
    **Returns:**
    - Total deliverables count
    - Status distribution
    - Overdue count
    - Average completion rate
    """
    logger.info("Fetching deliverables statistics")
    
    try:
        from app.models.hs2 import HS2Deliverable
        
        # Total count
        total_query = select(func.count(HS2Deliverable.id))
        total_result = await db.execute(total_query)
        total = total_result.scalar() or 0
        
        # Status distribution
        status_query = select(
            HS2Deliverable.status,
            func.count(HS2Deliverable.id)
        ).group_by(HS2Deliverable.status)
        status_result = await db.execute(status_query)
        status_distribution = dict(status_result.all())
        
        # Overdue count
        overdue_query = select(func.count(HS2Deliverable.id)).where(
            and_(
                HS2Deliverable.required_by_date < datetime.utcnow(),
                HS2Deliverable.status != 'Submitted'
            )
        )
        overdue_result = await db.execute(overdue_query)
        overdue = overdue_result.scalar() or 0
        
        # Completion rate
        submitted = status_distribution.get('Submitted', 0)
        completion_rate = (submitted / total * 100) if total > 0 else 0.0
        
        logger.info(f"Statistics retrieved: total={total}, overdue={overdue}")
        
        return {
            "total_deliverables": total,
            "status_distribution": status_distribution,
            "overdue_count": overdue,
            "completion_rate": round(completion_rate, 2),
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get deliverables statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )
