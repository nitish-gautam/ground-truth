"""
Database models for the Underground Utility Detection Platform
============================================================

SQLAlchemy ORM models for all data entities including GPR data,
environmental factors, ground truth validation, ML analytics,
LiDAR data, hyperspectral imaging, and BIM models.
"""

from .base import BaseModel
from .gpr_data import GPRSurvey, GPRScan, GPRSignalData, GPRProcessingResult
from .environmental import EnvironmentalData, WeatherCondition, GroundCondition
from .validation import GroundTruthData, ValidationResult, AccuracyMetrics
from .utilities import UtilityRecord, UtilityMaterial, UtilityDiscipline
from .ml_analytics import MLModel, FeatureVector, ModelPerformance, TrainingSession
from .progress import HS2ProgressSnapshot, HS2PointCloudComparison, HS2ScheduleMilestone

# New models for LiDAR, Hyperspectral, and BIM data
from .lidar import LidarDTMTile, LidarPointCloudCoverage, LidarElevationProfile
from .hyperspectral import (
    HyperspectralMaterialSample,
    HyperspectralAnalysis,
    ConcreteStrengthCalibration,
    MaterialType,
    QualityLabel
)
from .bim import (
    BIMTestModel,
    ArchitecturalScan,
    BIMElement,
    IFCVersion,
    BIMPurpose,
    ComplexityLevel
)

__all__ = [
    # Base
    "BaseModel",
    # GPR
    "GPRSurvey",
    "GPRScan",
    "GPRSignalData",
    "GPRProcessingResult",
    # Environmental
    "EnvironmentalData",
    "WeatherCondition",
    "GroundCondition",
    # Validation
    "GroundTruthData",
    "ValidationResult",
    "AccuracyMetrics",
    # Utilities
    "UtilityRecord",
    "UtilityMaterial",
    "UtilityDiscipline",
    # ML Analytics
    "MLModel",
    "FeatureVector",
    "ModelPerformance",
    "TrainingSession",
    # HS2 Progress
    "HS2ProgressSnapshot",
    "HS2PointCloudComparison",
    "HS2ScheduleMilestone",
    # LiDAR
    "LidarDTMTile",
    "LidarPointCloudCoverage",
    "LidarElevationProfile",
    # Hyperspectral
    "HyperspectralMaterialSample",
    "HyperspectralAnalysis",
    "ConcreteStrengthCalibration",
    "MaterialType",
    "QualityLabel",
    # BIM
    "BIMTestModel",
    "ArchitecturalScan",
    "BIMElement",
    "IFCVersion",
    "BIMPurpose",
    "ComplexityLevel"
]