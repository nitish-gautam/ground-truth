"""
HS2 Graph API Endpoints
=======================

Neo4j graph database endpoints for:
- Asset explainability ("Why Not Ready?")
- Dependency analysis
- Impact analysis
- Graph visualization data
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from app.services.graph_db import GraphDBService, get_graph_db

router = APIRouter(prefix="/graph", tags=["Graph Database"])


# ==================== Response Models ====================

class BlockerInfo(BaseModel):
    """Information about a single blocker"""
    id: str
    type: str
    name: Optional[str]
    status: Optional[str]
    reason: str


class ExplainabilityResponse(BaseModel):
    """Why an asset is not ready - explainability data"""
    asset_id: str
    asset_name: str
    status: str
    taem_score: float
    blockers: List[BlockerInfo]
    blocker_count: int
    ready: bool


class GraphNode(BaseModel):
    """Node in graph visualization"""
    id: str
    label: str
    type: str
    status: Optional[str] = None
    taem_score: Optional[float] = None


class GraphLink(BaseModel):
    """Edge/relationship in graph"""
    source: str
    target: str
    type: str
    label: str


class GraphVisualizationResponse(BaseModel):
    """Graph data for D3.js visualization"""
    nodes: List[GraphNode]
    links: List[GraphLink]
    center_node_id: str


class ImpactAnalysisResponse(BaseModel):
    """Impact analysis result"""
    asset_id: str
    asset_name: str
    impacted_asset_count: int
    impacted_assets: List[Dict[str, Any]]


class GraphStatsResponse(BaseModel):
    """Graph database statistics"""
    assets: int
    deliverables: int
    certificates: int
    relationships: int
    last_updated: datetime


# ==================== Endpoints ====================

@router.get(
    "/explainability/{asset_id}",
    response_model=ExplainabilityResponse,
    summary="Get asset explainability",
    description="Explains why an asset is not ready by showing all blockers"
)
async def get_asset_explainability(
    asset_id: str,
    graph_db: GraphDBService = Depends(get_graph_db)
) -> ExplainabilityResponse:
    """
    Get explainability data for an asset - answers "Why Not Ready?"

    Shows:
    - Missing deliverables
    - Expired certificates
    - Cost overruns
    - Other blockers

    Args:
        asset_id: HS2 asset ID (e.g., "VA-007")

    Returns:
        Explainability data with list of blockers
    """
    try:
        result = await graph_db.get_why_not_ready(asset_id)

        if not result["asset"]:
            raise HTTPException(
                status_code=404,
                detail=f"Asset {asset_id} not found in graph database"
            )

        asset = result["asset"]

        return ExplainabilityResponse(
            asset_id=asset["id"],
            asset_name=asset["name"],
            status=asset["status"],
            taem_score=asset.get("taem_score", 0.0),
            blockers=[
                BlockerInfo(
                    id=b["id"],
                    type=b["type"],
                    name=b.get("name"),
                    status=b.get("status"),
                    reason=b["reason"]
                )
                for b in result["blockers"]
            ],
            blocker_count=result["blocker_count"],
            ready=(asset["status"] == "Ready")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get explainability data: {str(e)}"
        )


@router.get(
    "/visualization/{asset_id}",
    response_model=GraphVisualizationResponse,
    summary="Get graph visualization data",
    description="Get nodes and edges for D3.js/React Flow visualization"
)
async def get_graph_visualization(
    asset_id: str,
    depth: int = Query(2, ge=1, le=5, description="Relationship depth to traverse"),
    graph_db: GraphDBService = Depends(get_graph_db)
) -> GraphVisualizationResponse:
    """
    Get graph data formatted for frontend visualization.

    Args:
        asset_id: Center asset ID
        depth: How many relationship levels to include (1-5)

    Returns:
        Nodes and links for graph visualization
    """
    try:
        data = await graph_db.get_graph_for_visualization(asset_id, depth)

        if not data["nodes"]:
            raise HTTPException(
                status_code=404,
                detail=f"Asset {asset_id} not found or has no relationships"
            )

        return GraphVisualizationResponse(
            nodes=[GraphNode(**node) for node in data["nodes"]],
            links=[GraphLink(**link) for link in data["links"]],
            center_node_id=asset_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get visualization data: {str(e)}"
        )


@router.get(
    "/dependencies/{asset_id}",
    response_model=Dict[str, Any],
    summary="Get dependency chain",
    description="Get full dependency chain for an asset"
)
async def get_dependency_chain(
    asset_id: str,
    max_depth: int = Query(5, ge=1, le=10),
    graph_db: GraphDBService = Depends(get_graph_db)
) -> Dict[str, Any]:
    """
    Get complete dependency chain showing what this asset depends on.

    Args:
        asset_id: Asset to analyze
        max_depth: Maximum depth to traverse

    Returns:
        Dependency chains
    """
    try:
        chains = await graph_db.get_dependency_chain(asset_id, max_depth)

        return {
            "asset_id": asset_id,
            "chains": chains,
            "chain_count": len(chains)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dependency chain: {str(e)}"
        )


@router.get(
    "/impact/{asset_id}",
    response_model=ImpactAnalysisResponse,
    summary="Get impact analysis",
    description="Analyze what would be impacted if this asset changes"
)
async def get_impact_analysis(
    asset_id: str,
    graph_db: GraphDBService = Depends(get_graph_db)
) -> ImpactAnalysisResponse:
    """
    Analyze downstream impact of changes to this asset.

    Shows all assets that depend on this one and would be affected.

    Args:
        asset_id: Asset to analyze

    Returns:
        List of impacted assets
    """
    try:
        result = await graph_db.get_impact_analysis(asset_id)

        if not result["asset"]:
            raise HTTPException(
                status_code=404,
                detail=f"Asset {asset_id} not found"
            )

        return ImpactAnalysisResponse(
            asset_id=asset_id,
            asset_name=result["asset"]["name"],
            impacted_asset_count=result["impact_count"],
            impacted_assets=result["impacted_assets"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get impact analysis: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    summary="Get graph database statistics",
    description="Get counts of nodes and relationships in graph"
)
async def get_graph_stats(
    graph_db: GraphDBService = Depends(get_graph_db)
) -> GraphStatsResponse:
    """
    Get statistics about the graph database.

    Returns:
        Node and relationship counts
    """
    try:
        stats = await graph_db.get_stats()

        return GraphStatsResponse(
            assets=stats["assets"],
            deliverables=stats["deliverables"],
            certificates=stats["certificates"],
            relationships=stats["relationships"],
            last_updated=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph stats: {str(e)}"
        )


@router.get(
    "/health",
    summary="Check graph database health",
    description="Verify Neo4j connectivity"
)
async def check_graph_health(
    graph_db: GraphDBService = Depends(get_graph_db)
) -> Dict[str, Any]:
    """
    Health check for Neo4j connection.

    Returns:
        Health status and basic info
    """
    try:
        is_healthy = await graph_db.health_check()

        if is_healthy:
            stats = await graph_db.get_stats()
            return {
                "status": "healthy",
                "connected": True,
                "node_count": stats["assets"] + stats["deliverables"] + stats["certificates"],
                "relationship_count": stats["relationships"]
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False
            }

    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }
