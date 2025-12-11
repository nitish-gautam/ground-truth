"""
LiDAR Data API Endpoints
========================

Endpoints for LiDAR DTM tile access and elevation profile generation.

Main capabilities:
- Query elevation at specific points
- Generate elevation profiles along lines (e.g., railway alignments)
- Find DTM tiles covering specific areas
- Query historical LiDAR point cloud coverage
"""

import logging
from typing import List, Optional, Tuple
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session

from app.api.deps import get_sync_db
from app.services.lidar_processor import LidarProcessor
from app.models.lidar import LidarDTMTile, LidarPointCloudCoverage
from app.schemas import lidar as schemas

logger = logging.getLogger(__name__)

router = APIRouter()


def get_lidar_processor(db: Session = Depends(get_sync_db)) -> LidarProcessor:
    """Dependency to get LiDAR processor instance."""
    return LidarProcessor(db)


@router.get("/tiles", response_model=List[schemas.LidarDTMTileResponse])
async def list_dtm_tiles(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    year: Optional[int] = Query(None, description="Filter by capture year"),
    db: Session = Depends(get_sync_db)
):
    """
    List available LiDAR DTM tiles.

    **Query Parameters:**
    - `skip`: Pagination offset
    - `limit`: Maximum results (1-1000)
    - `year`: Filter by capture year

    **Returns:**
    - List of DTM tile metadata with elevation statistics
    """
    logger.info(f"Listing DTM tiles (skip={skip}, limit={limit}, year={year})")

    query = db.query(LidarDTMTile)

    if year:
        query = query.filter(LidarDTMTile.capture_year == year)

    tiles = query.offset(skip).limit(limit).all()

    logger.info(f"Found {len(tiles)} DTM tiles")

    return [
        schemas.LidarDTMTileResponse(
            id=tile.id,
            tile_name=tile.tile_name,
            grid_reference=tile.grid_reference,
            file_path=tile.file_path,
            file_size_mb=tile.file_size_mb,
            resolution_meters=tile.resolution_meters,
            elevation_stats=schemas.ElevationStats(
                min_elevation=tile.min_elevation,
                max_elevation=tile.max_elevation,
                mean_elevation=tile.mean_elevation,
                std_elevation=tile.std_elevation
            ) if tile.min_elevation is not None else None,
            capture_year=tile.capture_year,
            capture_date=tile.capture_date,
            source=tile.source,
            dataset_name=tile.dataset_name,
            is_accessible=tile.is_accessible,
            metadata=tile.tile_metadata
        )
        for tile in tiles
    ]


@router.get("/tiles/{tile_name}", response_model=schemas.LidarDTMTileResponse)
async def get_dtm_tile(
    tile_name: str,
    processor: LidarProcessor = Depends(get_lidar_processor)
):
    """
    Get detailed information about a specific DTM tile.

    **Parameters:**
    - `tile_name`: Tile name (e.g., 'SK23ne', 'SK24ne')

    **Returns:**
    - Full tile metadata and elevation statistics
    """
    logger.info(f"Getting DTM tile: {tile_name}")

    try:
        result = await processor.read_dtm_tile(tile_name)
        return schemas.LidarDTMTileResponse(**result)

    except ValueError as e:
        logger.error(f"Tile not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get tile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get tile: {str(e)}")


@router.post("/elevation/point", response_model=schemas.ElevationPointResponse)
async def get_elevation_at_point(
    request: schemas.ElevationPointRequest = Body(...),
    processor: LidarProcessor = Depends(get_lidar_processor)
):
    """
    Get elevation at a specific point (British National Grid coordinates).

    **Input:**
    - `easting`: Easting coordinate (meters)
    - `northing`: Northing coordinate (meters)

    **Returns:**
    - Elevation value (meters above ODN)
    - Source tile information
    - Resolution and capture year

    **Coordinate System:** British National Grid (EPSG:27700)
    **Vertical Datum:** ODN (Ordnance Datum Newlyn)
    """
    logger.info(f"Getting elevation at ({request.easting}, {request.northing})")

    try:
        result = await processor.get_elevation_at_point(
            request.easting,
            request.northing
        )
        return schemas.ElevationPointResponse(**result)

    except ValueError as e:
        logger.error(f"No tile covers point: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get elevation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get elevation: {str(e)}")


@router.post("/elevation/profile", response_model=schemas.ElevationProfileResponse)
async def generate_elevation_profile(
    request: schemas.ElevationProfileRequest = Body(...),
    processor: LidarProcessor = Depends(get_lidar_processor)
):
    """
    Generate elevation profile along a line between two points.

    **Use Case:** Railway alignment analysis, road profiles, terrain cross-sections

    **Input:**
    - `start_point`: [easting, northing] start coordinates
    - `end_point`: [easting, northing] end coordinates
    - `num_samples`: Number of elevation samples (10-1000, default 100)
    - `save_profile`: Whether to save profile to database
    - `profile_name`: Name for saved profile (optional)

    **Returns:**
    - Elevation data array with distance/elevation/coordinates
    - Profile statistics (min/max elevation, elevation gain/loss)
    - Profile length and sample count

    **Example:**
    ```json
    {
        "start_point": [423000, 338000],
        "end_point": [424000, 339000],
        "num_samples": 100
    }
    ```
    """
    logger.info(
        f"Generating elevation profile: "
        f"{request.start_point} to {request.end_point}, "
        f"{request.num_samples} samples"
    )

    try:
        result = await processor.generate_elevation_profile(
            start_point=tuple(request.start_point),
            end_point=tuple(request.end_point),
            num_samples=request.num_samples,
            save_profile=request.save_profile,
            profile_name=request.profile_name
        )

        return schemas.ElevationProfileResponse(**result)

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate profile: {str(e)}")


@router.post("/tiles/find-in-bounds", response_model=List[schemas.LidarDTMTileResponse])
async def find_tiles_in_bounds(
    request: schemas.TileBoundsQuery = Body(...),
    processor: LidarProcessor = Depends(get_lidar_processor)
):
    """
    Find all DTM tiles that intersect with a bounding box.

    **Use Case:** Find tiles covering a specific area (e.g., HS2 route corridor)

    **Input:**
    - `min_easting`: Minimum easting (west edge)
    - `min_northing`: Minimum northing (south edge)
    - `max_easting`: Maximum easting (east edge)
    - `max_northing`: Maximum northing (north edge)
    - `year`: Optional filter by capture year

    **Returns:**
    - List of tiles that intersect the bounding box
    - Tile metadata and elevation statistics

    **Example:**
    ```json
    {
        "min_easting": 423000,
        "min_northing": 338000,
        "max_easting": 425000,
        "max_northing": 340000,
        "year": 2022
    }
    ```
    """
    logger.info(
        f"Finding tiles in bounds: ({request.min_easting}, {request.min_northing}) to "
        f"({request.max_easting}, {request.max_northing}), year={request.year}"
    )

    try:
        results = await processor.find_tiles_in_bounds(
            min_easting=request.min_easting,
            min_northing=request.min_northing,
            max_easting=request.max_easting,
            max_northing=request.max_northing,
            year=request.year
        )

        return [schemas.LidarDTMTileResponse(**tile) for tile in results]

    except Exception as e:
        logger.error(f"Failed to find tiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to find tiles: {str(e)}")


@router.get("/coverage", response_model=List[schemas.LidarCoverageAvailability])
async def query_point_cloud_coverage(
    easting: float = Query(..., description="Easting coordinate"),
    northing: float = Query(..., description="Northing coordinate"),
    year_start: int = Query(2005, ge=2005, description="Start year"),
    year_end: int = Query(2022, le=2030, description="End year"),
    processor: LidarProcessor = Depends(get_lidar_processor)
):
    """
    Query historical LiDAR point cloud coverage at a specific location.

    **Use Case:** Check data availability for time-series analysis

    **Query Parameters:**
    - `easting`: Easting coordinate (BNG)
    - `northing`: Northing coordinate (BNG)
    - `year_start`: Start year for coverage query (default: 2005)
    - `year_end`: End year for coverage query (default: 2022)

    **Returns:**
    - List of available point cloud data by year
    - Data quality, point density, and capture dates
    - RGB/intensity/classification availability

    **Example:**
    ```
    GET /api/v1/lidar/coverage?easting=423500&northing=338500&year_start=2010&year_end=2022
    ```
    """
    logger.info(
        f"Querying point cloud coverage at ({easting}, {northing}), "
        f"years {year_start}-{year_end}"
    )

    try:
        results = await processor.query_point_cloud_coverage(
            easting=easting,
            northing=northing,
            year_start=year_start,
            year_end=year_end
        )

        return [schemas.LidarCoverageAvailability(**coverage) for coverage in results]

    except Exception as e:
        logger.error(f"Failed to query coverage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to query coverage: {str(e)}")


@router.post("/tiles/{tile_name}/calculate-stats", response_model=schemas.ElevationStats)
async def calculate_tile_statistics(
    tile_name: str,
    force_recalculate: bool = Query(False, description="Force recalculation"),
    processor: LidarProcessor = Depends(get_lidar_processor)
):
    """
    Calculate elevation statistics for a DTM tile.

    **Parameters:**
    - `tile_name`: Tile name (e.g., 'SK23ne')
    - `force_recalculate`: Force recalculation even if stats exist

    **Returns:**
    - Elevation statistics (min, max, mean, std)

    **Note:** This operation can be slow for large tiles (~1 minute per tile)
    """
    logger.info(f"Calculating statistics for tile: {tile_name}")

    try:
        result = await processor.calculate_tile_statistics(
            tile_name=tile_name,
            force_recalculate=force_recalculate
        )

        return schemas.ElevationStats(**result)

    except ValueError as e:
        logger.error(f"Tile not found or inaccessible: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Statistics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
