"""
Hyperspectral Imaging Pydantic Schemas
======================================

Request/response schemas for hyperspectral material analysis API endpoints.
Critical for HS2 concrete quality assessment feature.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class MaterialTypeEnum(str, Enum):
    """Material types for classification"""
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    STEEL = "steel"
    WOOD = "wood"
    SOIL = "soil"
    VEGETATION = "vegetation"
    WATER = "water"
    UNKNOWN = "unknown"


class QualityLabelEnum(str, Enum):
    """Dataset split labels"""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class DefectSeverity(str, Enum):
    """Defect severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


# ============================================================================
# Material Sample Schemas
# ============================================================================

class HyperspectralSampleBase(BaseModel):
    """Base schema for hyperspectral material sample"""
    sample_id: str = Field(..., description="Unique sample identifier (e.g., 'Auto128')")
    material_type: MaterialTypeEnum = Field(..., description="Material classification")
    material_subtype: Optional[str] = Field(None, description="Detailed material subtype")
    surface_condition: Optional[str] = Field(None, description="Surface condition")
    source: str = Field("UMKC", description="Data source")


class HyperspectralSampleCreate(HyperspectralSampleBase):
    """Schema for creating material sample record"""
    sample_name: Optional[str] = None
    surface_age: Optional[str] = None
    moisture_level: Optional[str] = None
    image_path: str = Field(..., description="Path to sample image")
    image_format: str = Field("JPEG", description="Image format")
    resolution: str = Field("1000x1000", description="Image resolution")
    spectral_signature: Optional[List[float]] = Field(None, description="Array of 204 band values")
    quality_label: QualityLabelEnum = Field(QualityLabelEnum.TRAINING)
    is_augmented: bool = Field(False)
    parent_sample_id: Optional[UUID] = None
    ground_truth_strength_mpa: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class HyperspectralSampleResponse(HyperspectralSampleBase):
    """Schema for material sample response"""
    id: UUID
    sample_name: Optional[str]
    surface_age: Optional[str]
    moisture_level: Optional[str]
    image_path: str
    image_format: str
    resolution: str
    file_size_kb: Optional[float]
    spectral_signature: Optional[List[float]]
    num_bands: Optional[int]
    wavelength_range_nm: Optional[str]
    is_specim_compatible: bool
    quality_label: QualityLabelEnum
    is_augmented: bool
    ground_truth_strength_mpa: Optional[float]
    ground_truth_moisture_pct: Optional[float]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class HyperspectralSampleList(BaseModel):
    """Schema for list of material samples"""
    samples: List[HyperspectralSampleResponse]
    total: int
    page: int
    page_size: int
    material_type_counts: Dict[str, int]


# ============================================================================
# Analysis Request/Response Schemas
# ============================================================================

class HyperspectralAnalysisRequest(BaseModel):
    """Schema for requesting hyperspectral image analysis"""
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image file")
    analysis_name: Optional[str] = Field(None, description="Name for this analysis")
    project_id: Optional[UUID] = Field(None, description="Associated HS2 project ID")
    location_easting: Optional[float] = Field(None, description="Location easting (if georeferenced)")
    location_northing: Optional[float] = Field(None, description="Location northing (if georeferenced)")

    # Analysis options
    predict_material: bool = Field(True, description="Perform material classification")
    predict_strength: bool = Field(True, description="Predict concrete strength (if material is concrete)")
    detect_defects: bool = Field(True, description="Detect surface defects")
    extract_signature: bool = Field(True, description="Extract spectral signature")

    @field_validator('image_data', 'image_url')
    @classmethod
    def validate_image_source(cls, v, info):
        # Pydantic V2: info.data contains the field values
        if info.field_name == 'image_url' and not info.data.get('image_data') and not v:
            raise ValueError("Must provide either image_data or image_url")
        return v


class DefectInfo(BaseModel):
    """Schema for detected defect information"""
    defect_type: str = Field(..., description="Type of defect (void, crack, spalling, etc.)")
    location_x: int = Field(..., description="X pixel coordinate")
    location_y: int = Field(..., description="Y pixel coordinate")
    size_pixels: int = Field(..., description="Defect size in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    severity: DefectSeverity = Field(..., description="Defect severity assessment")


class SpectralSignature(BaseModel):
    """Schema for spectral signature data"""
    wavelengths_nm: List[float] = Field(..., description="Wavelength values (nm)")
    reflectance_values: List[float] = Field(..., description="Reflectance values")
    num_bands: int = Field(..., description="Number of spectral bands")

    # Key wavelength analysis
    cement_hydration_500_600: Optional[float] = Field(None, description="Avg reflectance 500-600nm")
    moisture_700_850: Optional[float] = Field(None, description="Avg reflectance 700-850nm")
    aggregate_900_1000: Optional[float] = Field(None, description="Avg reflectance 900-1000nm")


class HyperspectralAnalysisResponse(BaseModel):
    """Schema for hyperspectral analysis results"""
    id: UUID
    analysis_name: Optional[str]

    # Material classification
    predicted_material: MaterialTypeEnum
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    # Concrete-specific predictions (if material is concrete)
    predicted_strength_mpa: Optional[float] = Field(None, description="Predicted concrete strength")
    strength_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    strength_range_min: Optional[float] = Field(None, description="Lower bound")
    strength_range_max: Optional[float] = Field(None, description="Upper bound")

    # Quality assessment
    curing_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    moisture_content_pct: Optional[float] = Field(None, description="Estimated moisture %")
    aggregate_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Defects
    defects_detected: List[DefectInfo] = Field(default_factory=list)
    defect_severity: DefectSeverity = Field(DefectSeverity.NONE)

    # Spectral data
    spectral_signature: Optional[SpectralSignature] = None

    # Model information
    model_version: str
    model_accuracy: Optional[float] = None
    processing_time_ms: int

    # Timestamps
    created_at: datetime
    analyzed_at: datetime

    class Config:
        from_attributes = True


class ConcreteStrengthPrediction(BaseModel):
    """Simplified schema for concrete strength prediction only"""
    predicted_strength_mpa: float = Field(..., description="Predicted strength")
    confidence: float = Field(..., ge=0.0, le=1.0)
    strength_range_min: float
    strength_range_max: float
    prediction_quality: str = Field(..., description="'high', 'medium', or 'low'")
    model_r_squared: float = Field(..., description="Model R² value")


# ============================================================================
# Calibration Schemas
# ============================================================================

class ConcreteStrengthCalibrationBase(BaseModel):
    """Base schema for calibration data"""
    calibration_name: str = Field(..., description="Calibration identifier")
    test_location: str = Field(..., description="'lab', 'field', or site name")
    concrete_grade: str = Field(..., description="Concrete grade (e.g., 'C40/50')")
    num_samples: int = Field(..., ge=1, description="Number of calibration samples")


class ConcreteStrengthCalibrationCreate(ConcreteStrengthCalibrationBase):
    """Schema for creating calibration record"""
    description: Optional[str] = None
    test_date: Optional[datetime] = None
    strength_range_min: float = Field(..., description="Minimum strength tested (MPa)")
    strength_range_max: float = Field(..., description="Maximum strength tested (MPa)")
    target_strength: Optional[float] = None
    key_wavelengths: List[float] = Field(..., description="Critical wavelengths")
    spectral_coefficients: Dict[str, Any] = Field(..., description="Model coefficients")
    r_squared: float = Field(..., ge=0.0, le=1.0, description="R² value")
    mae: float = Field(..., description="Mean Absolute Error (MPa)")
    rmse: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ConcreteStrengthCalibrationResponse(ConcreteStrengthCalibrationBase):
    """Schema for calibration response"""
    id: UUID
    description: Optional[str]
    test_date: Optional[datetime]
    strength_range_min: float
    strength_range_max: float
    target_strength: Optional[float]
    key_wavelengths: List[float]
    r_squared: float
    mae: float
    rmse: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    is_validated: bool
    validation_date: Optional[datetime]
    validation_sample_count: Optional[int]
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: Optional[str]

    class Config:
        from_attributes = True


# ============================================================================
# Batch Processing Schemas
# ============================================================================

class BatchAnalysisRequest(BaseModel):
    """Schema for batch processing multiple images"""
    image_urls: List[str] = Field(..., min_items=1, max_items=100)
    project_id: Optional[UUID] = None
    analysis_options: HyperspectralAnalysisRequest = Field(
        default_factory=lambda: HyperspectralAnalysisRequest(
            predict_material=True,
            predict_strength=True,
            detect_defects=True,
            extract_signature=False  # Disable for batch to save processing time
        )
    )


class BatchAnalysisStatus(BaseModel):
    """Schema for batch processing status"""
    batch_id: UUID
    total_images: int
    processed: int
    failed: int
    in_progress: int
    status: str = Field(..., description="'pending', 'processing', 'completed', 'failed'")
    results: List[HyperspectralAnalysisResponse] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None


# ============================================================================
# Statistics and Reporting Schemas
# ============================================================================

class MaterialClassificationStats(BaseModel):
    """Schema for material classification statistics"""
    material_type: MaterialTypeEnum
    count: int
    percentage: float
    avg_confidence: float
    avg_strength_mpa: Optional[float] = None  # For concrete only


class HyperspectralDatasetStats(BaseModel):
    """Schema for overall dataset statistics"""
    total_samples: int
    training_samples: int
    validation_samples: int
    test_samples: int
    material_distribution: List[MaterialClassificationStats]
    avg_spectral_bands: float
    sources: List[str]
    augmented_samples: int
    with_ground_truth: int


class AnalysisPerformanceMetrics(BaseModel):
    """Schema for analysis performance metrics"""
    total_analyses: int
    avg_processing_time_ms: float
    material_accuracy: float = Field(..., description="Overall classification accuracy")
    concrete_strength_r_squared: float = Field(..., description="Concrete prediction R²")
    concrete_strength_mae: float = Field(..., description="MAE for concrete (MPa)")
    defect_detection_rate: float = Field(..., description="Defect detection success rate")
    timestamp: datetime


# ============================================================================
# Training Data Management
# ============================================================================

class DataAugmentationRequest(BaseModel):
    """Schema for data augmentation request"""
    source_sample_id: UUID = Field(..., description="Original sample to augment")
    augmentation_methods: List[str] = Field(
        ...,
        description="Methods to apply: 'rotation', 'brightness', 'contrast', 'noise', 'flip'"
    )
    num_variations: int = Field(10, ge=1, le=50, description="Number of augmented samples to generate")


class DataAugmentationResponse(BaseModel):
    """Schema for augmentation response"""
    source_sample_id: UUID
    generated_samples: List[HyperspectralSampleResponse]
    total_generated: int
    augmentation_methods_applied: List[str]
