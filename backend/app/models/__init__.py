"""
Database models for the Underground Utility Detection Platform
============================================================

SQLAlchemy ORM models for all data entities including GPR data,
environmental factors, ground truth validation, and ML analytics.
"""

from .base import BaseModel
from .gpr_data import GPRSurvey, GPRScan, GPRSignalData, GPRProcessingResult
from .environmental import EnvironmentalData, WeatherCondition, GroundCondition
from .validation import GroundTruthData, ValidationResult, AccuracyMetrics
from .utilities import UtilityRecord, UtilityMaterial, UtilityDiscipline
from .ml_analytics import MLModel, FeatureVector, ModelPerformance, TrainingSession

__all__ = [
    "BaseModel",
    "GPRSurvey",
    "GPRScan",
    "GPRSignalData",
    "GPRProcessingResult",
    "EnvironmentalData",
    "WeatherCondition",
    "GroundCondition",
    "GroundTruthData",
    "ValidationResult",
    "AccuracyMetrics",
    "UtilityRecord",
    "UtilityMaterial",
    "UtilityDiscipline",
    "MLModel",
    "FeatureVector",
    "ModelPerformance",
    "TrainingSession"
]