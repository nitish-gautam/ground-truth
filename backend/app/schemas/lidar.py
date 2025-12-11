"""
LiDAR Pydantic Schemas
=====================

Request/response schemas for LiDAR DTM and point cloud API endpoints.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


# ============================================================================
# LiDAR DTM Tile Schemas
# ============================================================================

class LidarDTMTileBase(BaseModel):
    """Base schema for LiDAR DTM tile"""
    tile_name: str = Field(..., description="UK National Grid tile name (e.g., 'SK23ne')")
    grid_reference: str = Field(..., description="UK National Grid reference")
    resolution_meters: float = Field(1.0, description="Grid resolution in meters")
    capture_year: Optional[int] = Field(None, description="Year of LiDAR capture")
    source: Optional[str] = Field(None, description="Data source organization")
    dataset_name: Optional[str] = Field(None, description="Dataset name")


class LidarDTMTileCreate(LidarDTMTileBase):
    """Schema for creating a new DTM tile record"""
    file_path: str = Field(..., description="Path to GeoTIFF file")
    file_size_mb: Optional[float] = Field(None, description="File size in MB")
    min_elevation: Optional[float] = Field(None, description="Minimum elevation")
    max_elevation: Optional[float] = Field(None, description="Maximum elevation")
    mean_elevation: Optional[float] = Field(None, description="Mean elevation")
    bounds_geojson: Dict[str, Any] = Field(..., description="Bounding box as GeoJSON")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LidarDTMTileUpdate(BaseModel):
    """Schema for updating DTM tile record"""
    is_processed: Optional[bool] = None
    is_accessible: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class LidarDTMTileResponse(LidarDTMTileBase):
    """Schema for DTM tile response"""
    id: UUID
    file_path: str
    file_size_mb: Optional[float]
    elevation_stats: Optional[ElevationStats] = None  # Forward reference (defined below)
    capture_date: Optional[datetime]
    is_processed: bool
    is_accessible: bool
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LidarDTMTileList(BaseModel):
    """Schema for list of DTM tiles"""
    tiles: List[LidarDTMTileResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Elevation Statistics Schema
# ============================================================================

class ElevationStats(BaseModel):
    """Schema for elevation statistics"""
    min_elevation: float = Field(..., description="Minimum elevation in meters")
    max_elevation: float = Field(..., description="Maximum elevation in meters")
    mean_elevation: float = Field(..., description="Mean elevation in meters")
    std_elevation: float = Field(..., description="Standard deviation of elevation")


# ============================================================================
# Elevation Query Schemas
# ============================================================================

class ElevationPointRequest(BaseModel):
    """Schema for querying elevation at a specific point"""
    easting: float = Field(..., description="Easting coordinate (British National Grid)")
    northing: float = Field(..., description="Northing coordinate (British National Grid)")
    tile_name: Optional[str] = Field(None, description="Specific tile to query (optional)")

# Alias for backward compatibility
ElevationPointQuery = ElevationPointRequest


class ElevationPointResponse(BaseModel):
    """Schema for elevation query response"""
    easting: float
    northing: float
    elevation: Optional[float] = Field(None, description="Elevation in meters (ODN)")
    tile_name: str = Field(..., description="Source DTM tile")
    interpolation_method: str = Field("bilinear", description="Interpolation method used")


class ElevationProfileRequest(BaseModel):
    """Schema for generating elevation profile"""
    start_point: List[float] = Field(..., description="[easting, northing] start point")
    end_point: List[float] = Field(..., description="[easting, northing] end point")
    num_samples: int = Field(100, ge=10, le=1000, description="Number of samples along profile")
    profile_name: Optional[str] = Field(None, description="Name for saved profile")
    save_profile: bool = Field(False, description="Save profile to database")

    @field_validator('start_point', 'end_point')
    @classmethod
    def validate_point(cls, v):
        if len(v) != 2:
            raise ValueError("Point must have exactly 2 coordinates [easting, northing]")
        return v


class ElevationProfilePoint(BaseModel):
    """Single point in elevation profile"""
    distance: float = Field(..., description="Distance from start (meters)")
    elevation: float = Field(..., description="Elevation (meters ODN)")
    easting: float = Field(..., description="Easting coordinate")
    northing: float = Field(..., description="Northing coordinate")


class ElevationProfileResponse(BaseModel):
    """Schema for elevation profile response"""
    profile_id: Optional[UUID] = None
    profile_name: Optional[str] = None
    start_point: List[float]
    end_point: List[float]
    profile_length_m: float
    num_samples: int
    min_elevation: float
    max_elevation: float
    elevation_gain: float
    elevation_loss: float
    profile_data: List[ElevationProfilePoint]
    source_tiles: List[str] = Field(..., description="DTM tiles used")
    created_at: datetime


# ============================================================================
# Point Cloud Coverage Schemas
# ============================================================================

class LidarCoverageBase(BaseModel):
    """Base schema for point cloud coverage"""
    year: int = Field(..., ge=2005, le=2030, description="Coverage year")
    tile_reference: str = Field(..., description="Tile reference identifier")
    data_available: bool = Field(True, description="Is data available")


class LidarCoverageCreate(LidarCoverageBase):
    """Schema for creating coverage record"""
    tile_name: Optional[str] = None
    coverage_area_geojson: Dict[str, Any] = Field(..., description="Coverage polygon as GeoJSON")
    source_format: str = Field(..., description="Source format (geojson, shapefile, etc.)")
    index_file_path: Optional[str] = None
    data_quality: Optional[str] = None
    point_density: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LidarCoverageResponse(LidarCoverageBase):
    """Schema for coverage response"""
    id: UUID
    tile_name: Optional[str]
    source_format: str
    data_quality: Optional[str]
    point_density: Optional[float]
    has_rgb: bool
    has_intensity: bool
    has_classification: bool
    provider: Optional[str]
    license_type: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class LidarCoverageQuery(BaseModel):
    """Schema for querying coverage availability"""
    location: List[float] = Field(..., description="[easting, northing] query point")
    year_start: Optional[int] = Field(2005, ge=2005, description="Start year")
    year_end: Optional[int] = Field(2022, le=2030, description="End year")

    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if len(v) != 2:
            raise ValueError("Location must have exactly 2 coordinates [easting, northing]")
        return v


class LidarCoverageAvailability(BaseModel):
    """Schema for coverage availability response"""
    location: List[float]
    available_years: List[int]
    coverage_details: List[LidarCoverageResponse]


# ============================================================================
# Spatial Query Schemas
# ============================================================================

class TileBoundsQuery(BaseModel):
    """Schema for DTM tile bounding box spatial query"""
    min_easting: float = Field(..., description="Minimum easting (west edge)")
    min_northing: float = Field(..., description="Minimum northing (south edge)")
    max_easting: float = Field(..., description="Maximum easting (east edge)")
    max_northing: float = Field(..., description="Maximum northing (north edge)")
    year: Optional[int] = Field(None, description="Filter by capture year")

    @field_validator('max_easting')
    @classmethod
    def validate_easting(cls, v, info):
        if 'min_easting' in info.data and v <= info.data['min_easting']:
            raise ValueError("max_easting must be greater than min_easting")
        return v

    @field_validator('max_northing')
    @classmethod
    def validate_northing(cls, v, info):
        if 'min_northing' in info.data and v <= info.data['min_northing']:
            raise ValueError("max_northing must be greater than min_northing")
        return v

# Alias for simpler bounding box queries without year filter
BoundingBoxQuery = TileBoundsQuery


class DTMTilesInBoundsResponse(BaseModel):
    """Schema for DTM tiles within bounding box"""
    query_bounds: BoundingBoxQuery
    tiles_found: int
    tiles: List[LidarDTMTileResponse]


# ============================================================================
# Statistics and Analysis Schemas
# ============================================================================

class DTMStatistics(BaseModel):
    """Schema for DTM tile statistics"""
    tile_name: str
    resolution_meters: float
    area_km2: float
    min_elevation: float
    max_elevation: float
    mean_elevation: float
    std_elevation: float
    elevation_range: float


class LidarDatasetSummary(BaseModel):
    """Schema for overall dataset summary"""
    total_dtm_tiles: int
    total_coverage_records: int
    dtm_total_area_km2: float
    coverage_years: List[int]
    earliest_capture: Optional[int]
    latest_capture: Optional[int]
    average_resolution_m: float
    tiles_by_grid_reference: Dict[str, int]
