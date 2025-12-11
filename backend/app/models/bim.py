"""
BIM (Building Information Modeling) Data Models
==============================================

SQLAlchemy models for BIM test files, IFC models, and architectural scans.
Supports IFC 4.0 and IFC 4.3 (with infrastructure extensions for HS2).
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from geoalchemy2 import Geometry
from datetime import datetime
import uuid
import enum

from .base import BaseModel


class IFCVersion(str, enum.Enum):
    """IFC schema versions"""
    IFC_2X3 = "2x3"
    IFC_4 = "4.0.2.1"
    IFC_4_3 = "4.3.2.0"
    UNKNOWN = "unknown"


class BIMPurpose(str, enum.Enum):
    """Purpose of BIM model"""
    IMPORT_TEST = "import_test"
    VALIDATION = "validation"
    CERTIFICATION = "certification"
    PRODUCTION = "production"
    REFERENCE = "reference"


class ComplexityLevel(str, enum.Enum):
    """Model complexity"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class BIMTestModel(BaseModel):
    """
    BIM test models for import validation and certification.

    Stores IFC test files (4.0 and 4.3) for validating
    BIM import functionality and ensuring IFC conformance.
    """

    __tablename__ = "bim_test_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Model identification
    model_name = Column(String(200), nullable=False, index=True)
    description = Column(Text)

    # IFC version
    ifc_version = Column(
        SQLEnum(IFCVersion, name='ifc_version_enum'),
        nullable=False,
        index=True
    )
    ifc_schema = Column(String(50))  # Detailed schema info

    # File information
    file_path = Column(Text, nullable=False)  # Path to IFC file
    file_size_mb = Column(Float)  # File size in megabytes
    file_format = Column(String(20), default='IFC')  # 'IFC', 'IFCXML', 'IFCZIP'

    # Purpose and usage
    purpose = Column(
        SQLEnum(BIMPurpose, name='bim_purpose_enum'),
        default=BIMPurpose.IMPORT_TEST,
        index=True
    )
    complexity_level = Column(
        SQLEnum(ComplexityLevel, name='complexity_level_enum'),
        default=ComplexityLevel.MEDIUM
    )

    # Model statistics
    element_count = Column(Integer)  # Total number of IFC elements
    unique_element_types = Column(Integer)  # Number of unique IfcProduct types
    has_geometry = Column(Boolean, default=True)  # Contains geometric representations
    has_properties = Column(Boolean, default=True)  # Contains property sets
    has_relationships = Column(Boolean, default=True)  # Contains relationships

    # IFC 4.3 infrastructure features (critical for HS2)
    has_alignment = Column(Boolean, default=False)  # IfcAlignment (railway)
    has_bridge = Column(Boolean, default=False)  # IfcBridge
    has_tunnel = Column(Boolean, default=False)  # IfcTunnel
    has_earthworks = Column(Boolean, default=False)  # IfcEarthworksCut/Fill
    has_rail = Column(Boolean, default=False)  # IfcRail, IfcTrackElement

    # Spatial extent (if georeferenced)
    bounding_box = Column(Geometry('POLYGON', srid=27700))  # British National Grid
    coordinate_system = Column(String(100))  # CRS information

    # Test/validation results
    import_tested = Column(Boolean, default=False)  # Has been tested for import
    import_success = Column(Boolean)  # Import successful
    import_error_message = Column(Text)  # Error message if import failed
    validation_passed = Column(Boolean)  # Passed validation tests
    validation_report = Column(JSON)  # Detailed validation results

    # buildingSMART certification
    is_certified = Column(Boolean, default=False)  # buildingSMART certified
    certification_date = Column(DateTime)
    certification_body = Column(String(100))

    # Data source
    source = Column(String(100))  # 'buildingSMART', 'HS2', 'Contractor'
    source_url = Column(Text)  # URL to source if available

    # Metadata (JSONB for flexibility)
    model_metadata = Column(JSON, default=dict)  # Additional IFC metadata

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_tested_at = Column(DateTime)

    def __repr__(self):
        return f"<BIMTestModel(name='{self.model_name}', version={self.ifc_version.value})>"


class ArchitecturalScan(BaseModel):
    """
    Architectural and archaeological 3D scans.

    Point clouds and meshes from terrestrial laser scanning
    or photogrammetry. From ArchScanLib dataset.
    """

    __tablename__ = "architectural_scans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Scan identification
    scan_name = Column(String(200), nullable=False, index=True)
    description = Column(Text)

    # Scan type
    scan_type = Column(String(50), index=True)  # 'point_cloud', 'mesh', 'hybrid'
    scan_method = Column(String(50))  # 'terrestrial_laser', 'mobile_laser', 'photogrammetry'

    # File information
    file_path = Column(Text, nullable=False)
    file_format = Column(String(20))  # 'e57', 'las', 'laz', 'ply', 'obj', 'xyz'
    file_size_mb = Column(Float)

    # Point cloud statistics
    point_count = Column(Integer)  # Number of points
    resolution_mm = Column(Float)  # Scanning resolution in millimeters
    point_density = Column(Float)  # Points per square meter

    # Color and intensity
    has_rgb = Column(Boolean, default=False)  # Has RGB color data
    has_intensity = Column(Boolean, default=False)  # Has intensity values
    has_normals = Column(Boolean, default=False)  # Has normal vectors

    # Spatial extent (British National Grid EPSG:27700)
    bounds = Column(Geometry('POLYGON', srid=27700))
    centroid = Column(Geometry('POINT', srid=27700))

    # Physical dimensions
    width_m = Column(Float)  # Width in meters
    height_m = Column(Float)  # Height in meters
    depth_m = Column(Float)  # Depth in meters

    # Capture details
    capture_date = Column(DateTime)
    scanner_model = Column(String(100))  # e.g., 'Leica ScanStation P50'
    scan_positions = Column(Integer)  # Number of scanner positions
    registration_error_mm = Column(Float)  # Registration accuracy

    # Building/structure information
    building_name = Column(String(200))
    building_type = Column(String(100))  # 'heritage', 'commercial', 'residential'
    heritage_status = Column(String(100))  # 'listed', 'scheduled_monument', etc.

    # Data source
    source = Column(String(100), default='ArchScanLib')
    source_organization = Column(String(200))
    license_type = Column(String(100))

    # Processing status
    is_processed = Column(Boolean, default=False)
    is_georeferenced = Column(Boolean, default=False)
    is_classified = Column(Boolean, default=False)

    # Metadata
    scan_metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ArchitecturalScan(name='{self.scan_name}', points={self.point_count})>"


class BIMElement(BaseModel):
    """
    Extracted IFC elements from BIM models.

    Stores individual building elements (walls, slabs, beams, etc.)
    extracted from IFC files for detailed analysis.
    """

    __tablename__ = "bim_elements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Parent model reference
    bim_model_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # IFC identification
    ifc_guid = Column(String(50), nullable=False, index=True)  # IFC GlobalId
    ifc_type = Column(String(100), nullable=False, index=True)  # e.g., 'IfcWall', 'IfcSlab'
    ifc_name = Column(String(200))  # Element name
    ifc_description = Column(Text)  # Element description

    # Element hierarchy
    parent_element_id = Column(UUID(as_uuid=True))  # Parent element (if nested)
    building_storey = Column(String(100))  # Building level
    spatial_structure = Column(ARRAY(String))  # [Project, Site, Building, Storey]

    # Geometric properties
    has_geometry = Column(Boolean, default=False)
    geometry_type = Column(String(50))  # 'SweptSolid', 'Brep', 'CSG', etc.
    volume_m3 = Column(Float)  # Volume in cubic meters
    area_m2 = Column(Float)  # Area in square meters

    # Location (if georeferenced)
    location = Column(Geometry('POINT', srid=27700))  # Element centroid
    bounding_box = Column(Geometry('POLYGON', srid=27700))

    # Material properties
    materials = Column(JSON)  # List of materials used
    material_layers = Column(JSON)  # Layer structure (walls, slabs)

    # Quantities
    quantities = Column(JSON)  # Quantity takeoff data

    # Properties
    properties = Column(JSON)  # Property sets and values

    # Classification
    classification_code = Column(String(100))  # Uniclass, Omniclass, etc.
    classification_name = Column(String(200))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<BIMElement(type='{self.ifc_type}', guid='{self.ifc_guid[:8]}...')>"
