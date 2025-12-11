"""
BIM (Building Information Modeling) Pydantic Schemas
====================================================

Request/response schemas for BIM/IFC and architectural scan API endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class IFCVersionEnum(str, Enum):
    """IFC schema versions"""
    IFC_2X3 = "2x3"
    IFC_4 = "4.0.2.1"
    IFC_4_3 = "4.3.2.0"
    UNKNOWN = "unknown"


class BIMPurposeEnum(str, Enum):
    """Purpose of BIM model"""
    IMPORT_TEST = "import_test"
    VALIDATION = "validation"
    CERTIFICATION = "certification"
    PRODUCTION = "production"
    REFERENCE = "reference"


class ComplexityLevelEnum(str, Enum):
    """Model complexity"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ScanTypeEnum(str, Enum):
    """Scan type"""
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    HYBRID = "hybrid"


# ============================================================================
# BIM Test Model Schemas
# ============================================================================

class BIMTestModelBase(BaseModel):
    """Base schema for BIM test model"""
    model_name: str = Field(..., description="Model identifier")
    ifc_version: IFCVersionEnum = Field(..., description="IFC schema version")
    purpose: BIMPurposeEnum = Field(BIMPurposeEnum.IMPORT_TEST)
    complexity_level: ComplexityLevelEnum = Field(ComplexityLevelEnum.MEDIUM)


class BIMTestModelCreate(BIMTestModelBase):
    """Schema for creating BIM test model record"""
    description: Optional[str] = None
    file_path: str = Field(..., description="Path to IFC file")
    file_size_mb: Optional[float] = None
    file_format: str = Field("IFC", description="File format")
    source: Optional[str] = Field(None, description="Data source")
    source_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BIMTestModelUpdate(BaseModel):
    """Schema for updating BIM test model"""
    import_tested: Optional[bool] = None
    import_success: Optional[bool] = None
    import_error_message: Optional[str] = None
    validation_passed: Optional[bool] = None
    validation_report: Optional[Dict[str, Any]] = None
    is_certified: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class BIMTestModelResponse(BIMTestModelBase):
    """Schema for BIM test model response"""
    id: UUID
    description: Optional[str]
    ifc_schema: Optional[str]
    file_path: str
    file_size_mb: Optional[float]
    file_format: str

    # Model statistics
    element_count: Optional[int]
    unique_element_types: Optional[int]
    has_geometry: bool
    has_properties: bool
    has_relationships: bool

    # IFC 4.3 infrastructure features (critical for HS2)
    has_alignment: bool
    has_bridge: bool
    has_tunnel: bool
    has_earthworks: bool
    has_rail: bool

    # Test results
    import_tested: bool
    import_success: Optional[bool]
    import_error_message: Optional[str]
    validation_passed: Optional[bool]
    validation_report: Optional[Dict[str, Any]]

    # Certification
    is_certified: bool
    certification_date: Optional[datetime]
    certification_body: Optional[str]

    # Source
    source: Optional[str]
    source_url: Optional[str]

    # Metadata
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_tested_at: Optional[datetime]

    class Config:
        from_attributes = True


class BIMTestModelList(BaseModel):
    """Schema for list of BIM test models"""
    models: List[BIMTestModelResponse]
    total: int
    page: int
    page_size: int
    ifc_version_counts: Dict[str, int]


# ============================================================================
# IFC Import/Validation Schemas
# ============================================================================

class IFCImportRequest(BaseModel):
    """Schema for IFC import request"""
    file_path: Optional[str] = Field(None, description="Path to IFC file on server")
    file_data: Optional[str] = Field(None, description="Base64 encoded IFC file data")
    extract_geometry: bool = Field(True, description="Extract geometric representations")
    extract_properties: bool = Field(True, description="Extract property sets")
    extract_relationships: bool = Field(True, description="Extract element relationships")
    validate_schema: bool = Field(True, description="Validate against IFC schema")

    @field_validator('file_data', 'file_path')
    @classmethod
    def validate_file_source(cls, v, info):
        # Pydantic V2: info.data contains the field values
        if info.field_name == 'file_path' and not info.data.get('file_data') and not v:
            raise ValueError("Must provide either file_path or file_data")
        return v


class IFCElementSummary(BaseModel):
    """Schema for IFC element summary"""
    ifc_type: str = Field(..., description="IFC element type (e.g., 'IfcWall')")
    count: int = Field(..., description="Number of elements of this type")
    has_geometry_count: int = Field(..., description="Elements with geometry")
    sample_guids: List[str] = Field(..., description="Sample GUIDs (max 5)")


class IFCImportResponse(BaseModel):
    """Schema for IFC import response"""
    import_id: UUID
    ifc_version: IFCVersionEnum
    schema_valid: bool
    import_success: bool
    import_errors: List[str] = Field(default_factory=list)

    # Statistics
    total_elements: int
    element_types_summary: List[IFCElementSummary]
    has_geometry: bool
    has_properties: bool
    has_relationships: bool

    # IFC 4.3 features
    has_alignment: bool
    has_bridge: bool
    has_tunnel: bool
    has_earthworks: bool

    # Processing info
    processing_time_ms: int
    imported_at: datetime


class IFCValidationRequest(BaseModel):
    """Schema for IFC validation request"""
    model_id: Optional[UUID] = Field(None, description="Existing BIM model ID")
    file_path: Optional[str] = Field(None, description="Path to IFC file")
    validation_rules: List[str] = Field(
        default_factory=lambda: ["schema", "geometry", "relationships"],
        description="Validation rules to apply"
    )


class IFCValidationResult(BaseModel):
    """Schema for single validation result"""
    rule: str = Field(..., description="Validation rule name")
    passed: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    details: Optional[Dict[str, Any]] = None


class IFCValidationResponse(BaseModel):
    """Schema for IFC validation response"""
    model_id: Optional[UUID]
    validation_date: datetime
    overall_passed: bool
    results: List[IFCValidationResult]
    summary: str


# ============================================================================
# BIM Element Schemas
# ============================================================================

class BIMElementBase(BaseModel):
    """Base schema for BIM element"""
    ifc_guid: str = Field(..., description="IFC GlobalId")
    ifc_type: str = Field(..., description="IFC element type")
    ifc_name: Optional[str] = Field(None, description="Element name")


class BIMElementCreate(BIMElementBase):
    """Schema for creating BIM element record"""
    bim_model_id: UUID = Field(..., description="Parent BIM model ID")
    ifc_description: Optional[str] = None
    parent_element_id: Optional[UUID] = None
    building_storey: Optional[str] = None
    spatial_structure: Optional[List[str]] = None
    has_geometry: bool = Field(False)
    geometry_type: Optional[str] = None
    volume_m3: Optional[float] = None
    area_m2: Optional[float] = None
    materials: Optional[List[Dict[str, Any]]] = None
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    quantities: Optional[Dict[str, Any]] = None


class BIMElementResponse(BIMElementBase):
    """Schema for BIM element response"""
    id: UUID
    bim_model_id: UUID
    ifc_description: Optional[str]
    parent_element_id: Optional[UUID]
    building_storey: Optional[str]
    spatial_structure: Optional[List[str]]
    has_geometry: bool
    geometry_type: Optional[str]
    volume_m3: Optional[float]
    area_m2: Optional[float]
    materials: Optional[List[Dict[str, Any]]]
    properties: Dict[str, Any]
    quantities: Optional[Dict[str, Any]]
    classification_code: Optional[str]
    classification_name: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class BIMElementQuery(BaseModel):
    """Schema for querying BIM elements"""
    bim_model_id: UUID
    ifc_types: Optional[List[str]] = Field(None, description="Filter by IFC types")
    building_storey: Optional[str] = Field(None, description="Filter by storey")
    has_geometry: Optional[bool] = None
    search_properties: Optional[Dict[str, str]] = Field(None, description="Property key-value filters")


class BIMElementList(BaseModel):
    """Schema for list of BIM elements"""
    elements: List[BIMElementResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Architectural Scan Schemas
# ============================================================================

class ArchitecturalScanBase(BaseModel):
    """Base schema for architectural scan"""
    scan_name: str = Field(..., description="Scan identifier")
    scan_type: ScanTypeEnum = Field(..., description="Scan type")
    scan_method: Optional[str] = Field(None, description="Scanning method")


class ArchitecturalScanCreate(ArchitecturalScanBase):
    """Schema for creating architectural scan record"""
    description: Optional[str] = None
    file_path: str = Field(..., description="Path to scan file")
    file_format: str = Field(..., description="File format (e.g., 'e57', 'las', 'ply')")
    file_size_mb: Optional[float] = None
    point_count: Optional[int] = None
    resolution_mm: Optional[float] = None
    capture_date: Optional[datetime] = None
    scanner_model: Optional[str] = None
    building_name: Optional[str] = None
    building_type: Optional[str] = None
    source: str = Field("ArchScanLib", description="Data source")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ArchitecturalScanResponse(ArchitecturalScanBase):
    """Schema for architectural scan response"""
    id: UUID
    description: Optional[str]
    file_path: str
    file_format: str
    file_size_mb: Optional[float]
    point_count: Optional[int]
    resolution_mm: Optional[float]
    point_density: Optional[float]
    has_rgb: bool
    has_intensity: bool
    has_normals: bool
    width_m: Optional[float]
    height_m: Optional[float]
    depth_m: Optional[float]
    capture_date: Optional[datetime]
    scanner_model: Optional[str]
    scan_positions: Optional[int]
    registration_error_mm: Optional[float]
    building_name: Optional[str]
    building_type: Optional[str]
    heritage_status: Optional[str]
    source: Optional[str]
    source_organization: Optional[str]
    license_type: Optional[str]
    is_processed: bool
    is_georeferenced: bool
    is_classified: bool
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ArchitecturalScanList(BaseModel):
    """Schema for list of architectural scans"""
    scans: List[ArchitecturalScanResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Statistics and Summary Schemas
# ============================================================================

class BIMDatasetStatistics(BaseModel):
    """Schema for BIM dataset statistics"""
    total_models: int
    total_elements: int
    total_scans: int
    ifc_version_distribution: Dict[str, int]
    models_by_purpose: Dict[str, int]
    models_by_complexity: Dict[str, int]
    certified_models: int
    models_with_alignment: int
    models_with_bridge: int
    models_with_tunnel: int
    total_scan_points: int
    avg_scan_resolution_mm: float


class IFCFeatureSupport(BaseModel):
    """Schema for IFC feature support summary"""
    ifc_version: IFCVersionEnum
    supports_alignment: bool
    supports_bridge: bool
    supports_tunnel: bool
    supports_earthworks: bool
    supports_rail: bool
    description: str


# ============================================================================
# Bulk Operations
# ============================================================================

class BulkIFCImportRequest(BaseModel):
    """Schema for bulk IFC import"""
    file_paths: List[str] = Field(..., min_items=1, max_items=50)
    purpose: BIMPurposeEnum = Field(BIMPurposeEnum.VALIDATION)
    extract_elements: bool = Field(True, description="Extract all elements")


class BulkIFCImportResponse(BaseModel):
    """Schema for bulk import response"""
    batch_id: UUID
    total_files: int
    successful_imports: int
    failed_imports: int
    results: List[IFCImportResponse]
    started_at: datetime
    completed_at: Optional[datetime]
