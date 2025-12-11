"""
Hyperspectral Imaging Data Models
=================================

SQLAlchemy models for hyperspectral material samples and analysis results.
Critical for HS2 concrete quality assessment feature.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from geoalchemy2 import Geometry
from datetime import datetime
import uuid
import enum

from .base import BaseModel


class MaterialType(str, enum.Enum):
    """Material types for hyperspectral classification"""
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    STEEL = "steel"
    WOOD = "wood"
    SOIL = "soil"
    VEGETATION = "vegetation"
    WATER = "water"
    UNKNOWN = "unknown"


class QualityLabel(str, enum.Enum):
    """Dataset split labels"""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class HyperspectralMaterialSample(BaseModel):
    """
    Hyperspectral training/validation samples for material classification.

    Based on UMKC Material Surfaces dataset:
    - 150 samples (Concrete: 75, Asphalt: 75)
    - Can be expanded to 1,500+ with augmentation
    - Used to train concrete quality AI models
    """

    __tablename__ = "hyperspectral_material_samples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Sample identification
    sample_id = Column(String(50), unique=True, nullable=False, index=True)  # e.g., 'Auto128'
    sample_name = Column(String(200))  # Human-readable name

    # Material classification
    material_type = Column(
        SQLEnum(MaterialType, name='material_type_enum'),
        nullable=False,
        index=True
    )
    material_subtype = Column(String(50))  # e.g., 'gray_concrete', 'fresh', 'weathered'

    # Surface characteristics
    surface_condition = Column(String(50))  # 'smooth', 'rough', 'cracked', 'stained'
    surface_age = Column(String(50))  # 'new', 'aged', 'weathered'
    moisture_level = Column(String(50))  # 'dry', 'damp', 'wet'

    # Image information
    image_path = Column(Text, nullable=False)  # Path to image file
    image_format = Column(String(20), default='JPEG')  # 'JPEG', 'PNG', 'TIFF'
    resolution = Column(String(20))  # e.g., '1000x1000'
    file_size_kb = Column(Float)  # File size in kilobytes

    # Spectral signature (if available from full HSI cube)
    spectral_signature = Column(JSON)  # Array of reflectance values per band
    num_bands = Column(Integer)  # Number of spectral bands (204 for Specim IQ)
    wavelength_range_nm = Column(String(20))  # e.g., '400-1000'

    # Specim IQ compatibility
    is_specim_compatible = Column(Boolean, default=True)  # Compatible with Specim IQ
    spectral_resolution_nm = Column(Float, default=3.0)  # ~3nm for Specim IQ

    # Data source
    source = Column(String(100), default='UMKC')  # 'UMKC', 'Field', 'Lab'
    dataset_name = Column(String(200))  # e.g., 'UMKC_Material_Surfaces'

    # Quality labels for ML
    quality_label = Column(
        SQLEnum(QualityLabel, name='quality_label_enum'),
        default=QualityLabel.TRAINING,
        index=True
    )

    # Augmentation tracking
    is_augmented = Column(Boolean, default=False)  # Is this an augmented sample
    parent_sample_id = Column(UUID(as_uuid=True))  # Reference to original sample
    augmentation_method = Column(String(100))  # 'rotation', 'brightness', 'noise', etc.

    # Ground truth (for validation)
    ground_truth_strength_mpa = Column(Float)  # Concrete strength (if measured)
    ground_truth_moisture_pct = Column(Float)  # Moisture content percentage
    ground_truth_defects = Column(JSON)  # List of known defects

    # Metadata (JSONB for flexibility)
    sample_metadata = Column(JSON, default=dict)  # Additional metadata

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<HyperspectralMaterialSample(id='{self.sample_id}', type={self.material_type.value})>"


class HyperspectralAnalysis(BaseModel):
    """
    Results from hyperspectral image analysis.

    Stores predictions and analysis results from uploaded HSI images.
    Used in production for HS2 concrete quality assessment.
    """

    __tablename__ = "hyperspectral_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Analysis identification
    analysis_name = Column(String(200))
    description = Column(Text)

    # Source image
    image_path = Column(Text, nullable=False)  # Path to analyzed image
    image_metadata = Column(JSON)  # Image metadata

    # Location information (if georeferenced)
    location = Column(Geometry('POINT', srid=27700))  # British National Grid
    project_id = Column(UUID(as_uuid=True), index=True)  # Reference to HS2 project
    survey_id = Column(UUID(as_uuid=True), index=True)  # Reference to survey

    # Material classification results
    predicted_material = Column(
        SQLEnum(MaterialType, name='material_type_enum'),
        nullable=False
    )
    confidence_score = Column(Float)  # 0.0 to 1.0

    # Concrete-specific predictions (if material is concrete)
    predicted_strength_mpa = Column(Float)  # Predicted concrete strength
    strength_confidence = Column(Float)  # Confidence in strength prediction
    strength_range_min = Column(Float)  # Lower bound of prediction range
    strength_range_max = Column(Float)  # Upper bound of prediction range

    # Quality assessment
    curing_quality_score = Column(Float)  # 0.0 to 1.0 (500-600nm analysis)
    moisture_content_pct = Column(Float)  # Estimated moisture percentage (700-850nm)
    aggregate_quality_score = Column(Float)  # 0.0 to 1.0 (900-1000nm analysis)

    # Defect detection
    defects_detected = Column(JSON)  # List of detected defects
    defect_locations = Column(JSON)  # Pixel coordinates of defects
    defect_severity = Column(String(20))  # 'none', 'minor', 'moderate', 'severe'

    # Spectral analysis
    spectral_signature = Column(JSON)  # Extracted spectral signature
    key_wavelengths = Column(JSON)  # Important wavelengths and their values

    # ML model used
    model_id = Column(UUID(as_uuid=True))  # Reference to ML model
    model_version = Column(String(50))  # Model version used
    model_accuracy = Column(Float)  # Known accuracy of the model

    # Processing information
    processing_time_ms = Column(Integer)  # Processing time in milliseconds
    processed_by = Column(String(100))  # User or system that ran analysis

    # Validation (if ground truth available)
    is_validated = Column(Boolean, default=False)
    actual_strength_mpa = Column(Float)  # Actual measured strength (if known)
    validation_error = Column(Float)  # Absolute error vs ground truth

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<HyperspectralAnalysis(material={self.predicted_material.value}, confidence={self.confidence_score:.2f})>"


class ConcreteStrengthCalibration(BaseModel):
    """
    Calibration data for concrete strength prediction models.

    Stores correlation between spectral signatures and measured
    concrete strength. Critical for achieving R²=0.89 (lab), R²=0.82 (field).
    """

    __tablename__ = "concrete_strength_calibrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Calibration identification
    calibration_name = Column(String(200), nullable=False)
    description = Column(Text)

    # Test conditions
    test_location = Column(String(100))  # 'lab', 'field', 'site_name'
    test_date = Column(DateTime)
    num_samples = Column(Integer)  # Number of calibration samples

    # Concrete specifications
    concrete_grade = Column(String(50))  # e.g., 'C40/50', 'C30/37'
    strength_range_min = Column(Float)  # Minimum strength tested (MPa)
    strength_range_max = Column(Float)  # Maximum strength tested (MPa)
    target_strength = Column(Float)  # Design strength (MPa)

    # Spectral correlation
    key_wavelengths = Column(ARRAY(Float))  # Critical wavelengths for this concrete
    spectral_coefficients = Column(JSON)  # Model coefficients

    # Model performance
    r_squared = Column(Float)  # R² value (target: 0.89 lab, 0.82 field)
    mae = Column(Float)  # Mean Absolute Error (target: 3.2 MPa lab, 4.2 MPa field)
    rmse = Column(Float)  # Root Mean Square Error
    precision = Column(Float)  # Precision percentage
    recall = Column(Float)  # Recall percentage

    # Validation status
    is_validated = Column(Boolean, default=False)
    validation_date = Column(DateTime)
    validation_sample_count = Column(Integer)

    # Metadata
    calibration_metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100))

    def __repr__(self):
        return f"<ConcreteStrengthCalibration(name='{self.calibration_name}', R²={self.r_squared:.3f})>"
