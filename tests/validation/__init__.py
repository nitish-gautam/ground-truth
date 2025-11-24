"""
GPR Validation Framework.

Comprehensive validation capabilities for Ground Penetrating Radar (GPR) signal
processing accuracy using real ground truth data from the University of Twente.
"""

from .accuracy import (
    AccuracyAssessor,
    DetectedUtility,
    GroundTruthUtility,
    create_accuracy_assessor
)

from .pas128 import (
    PAS128ComplianceValidator,
    QualityLevel,
    DetectionMethod,
    create_pas128_validator
)

from .statistical import (
    StatisticalValidator,
    StatisticalTest,
    create_statistical_validator
)

from .environmental import (
    EnvironmentalValidator,
    EnvironmentalFactor,
    PerformanceMetric,
    create_environmental_validator
)

__all__ = [
    # Accuracy Assessment
    'AccuracyAssessor',
    'DetectedUtility',
    'GroundTruthUtility',
    'create_accuracy_assessor',

    # PAS 128 Compliance
    'PAS128ComplianceValidator',
    'QualityLevel',
    'DetectionMethod',
    'create_pas128_validator',

    # Statistical Validation
    'StatisticalValidator',
    'StatisticalTest',
    'create_statistical_validator',

    # Environmental Validation
    'EnvironmentalValidator',
    'EnvironmentalFactor',
    'PerformanceMetric',
    'create_environmental_validator'
]