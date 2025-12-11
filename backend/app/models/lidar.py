"""
LiDAR Data Models
================

SQLAlchemy models for LiDAR DTM tiles and point cloud coverage data.
Supports spatial queries with PostGIS.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
from datetime import datetime
import uuid

from .base import BaseModel


class LidarDTMTile(BaseModel):
    """
    Digital Terrain Model (DTM) tile data.

    Represents a single LiDAR DTM tile with elevation raster data.
    UK National Grid tiles (e.g., SK23ne, SK24ne) with 1m resolution.
    """

    __tablename__ = "lidar_dtm_tiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Tile identification
    tile_name = Column(String(50), unique=True, nullable=False, index=True)  # e.g., 'SK23ne'
    grid_reference = Column(String(10), nullable=False, index=True)  # UK National Grid ref

    # File information
    file_path = Column(Text, nullable=False)  # Path to GeoTIFF file
    file_size_mb = Column(Float)  # File size in megabytes

    # Technical specifications
    resolution_meters = Column(Float, default=1.0)  # Grid resolution (usually 1m)

    # Spatial extent (British National Grid EPSG:27700)
    bounds = Column(Geometry('POLYGON', srid=27700), nullable=False)

    # Elevation statistics
    min_elevation = Column(Float)  # Minimum elevation in meters
    max_elevation = Column(Float)  # Maximum elevation in meters
    mean_elevation = Column(Float)  # Mean elevation
    std_elevation = Column(Float)  # Standard deviation of elevation

    # Temporal information
    capture_year = Column(Integer, index=True)  # Year of LiDAR capture
    capture_date = Column(DateTime)  # Specific capture date if available

    # Data source
    source = Column(String(100))  # e.g., 'UK Environment Agency'
    dataset_name = Column(String(200))  # e.g., 'lidar_composite_dtm-2022-1'

    # Metadata (JSONB for flexibility)
    tile_metadata = Column(JSON, default=dict)  # Additional metadata from .tif.xml

    # Processing status
    is_processed = Column(Boolean, default=False)  # Has been indexed/processed
    is_accessible = Column(Boolean, default=True)  # File is accessible

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<LidarDTMTile(tile_name='{self.tile_name}', resolution={self.resolution_meters}m)>"


class LidarPointCloudCoverage(BaseModel):
    """
    Historical LiDAR point cloud coverage indices.

    Represents spatial coverage of available LiDAR point clouds
    for different years (2005-2022). This is metadata/index data,
    not the actual point clouds.
    """

    __tablename__ = "lidar_point_cloud_coverage"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Temporal information
    year = Column(Integer, nullable=False, index=True)  # Year of coverage (2005-2022)

    # Tile identification
    tile_reference = Column(String(50), nullable=False, index=True)  # Tile identifier
    tile_name = Column(String(100))  # Human-readable tile name

    # Spatial coverage (British National Grid EPSG:27700)
    coverage_area = Column(Geometry('POLYGON', srid=27700), nullable=False)

    # Data availability
    data_available = Column(Boolean, default=True)  # Is point cloud data available
    data_quality = Column(String(20))  # 'high', 'medium', 'low'
    point_density = Column(Float)  # Points per square meter

    # Source format information
    source_format = Column(String(20))  # 'geojson', 'shapefile', 'gdb', 'geopackage'
    index_file_path = Column(Text)  # Path to source index file

    # Point cloud specifications
    has_rgb = Column(Boolean, default=False)  # Has RGB color data
    has_intensity = Column(Boolean, default=False)  # Has intensity values
    has_classification = Column(Boolean, default=False)  # Has point classification

    # Capture details
    capture_date_start = Column(DateTime)  # Start of capture period
    capture_date_end = Column(DateTime)  # End of capture period

    # Data provider
    provider = Column(String(100))  # Data provider organization
    license_type = Column(String(100))  # License (e.g., 'OGL v3.0')

    # Metadata (JSONB for flexibility)
    coverage_metadata = Column(JSON, default=dict)  # Additional attributes from source

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<LidarPointCloudCoverage(year={self.year}, tile='{self.tile_reference}')>"


class LidarElevationProfile(BaseModel):
    """
    Stored elevation profiles extracted from DTM tiles.

    Represents elevation cross-sections along a line,
    useful for railway alignment, road profiles, etc.
    """

    __tablename__ = "lidar_elevation_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Profile identification
    profile_name = Column(String(200))  # User-defined name
    description = Column(Text)  # Profile description

    # Line geometry (British National Grid EPSG:27700)
    line_geometry = Column(Geometry('LINESTRING', srid=27700), nullable=False)

    # Source DTM tiles used
    source_tiles = Column(JSON)  # List of tile names used

    # Profile data
    num_samples = Column(Integer)  # Number of elevation samples
    profile_length_m = Column(Float)  # Total profile length in meters

    # Elevation statistics
    min_elevation = Column(Float)
    max_elevation = Column(Float)
    elevation_gain = Column(Float)  # Total upward elevation change
    elevation_loss = Column(Float)  # Total downward elevation change

    # Profile data (stored as JSON array)
    elevation_data = Column(JSON)  # [{distance: 0, elevation: 45.2, lat: x, lon: y}, ...]

    # Associated project/survey
    project_id = Column(UUID(as_uuid=True), index=True)  # Reference to HS2 project
    survey_id = Column(UUID(as_uuid=True), index=True)  # Reference to survey

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100))  # User who created the profile

    def __repr__(self):
        return f"<LidarElevationProfile(name='{self.profile_name}', length={self.profile_length_m}m)>"
