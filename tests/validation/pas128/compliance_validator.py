"""
PAS 128 Compliance Validation Framework.

This module provides comprehensive testing for PAS 128:2014 specification compliance,
including quality level achievement validation, detection method effectiveness,
and completeness assessment for survey deliverables.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import math


logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """PAS 128 Quality Levels."""
    QL_D = "QL-D"
    QL_C = "QL-C"
    QL_B = "QL-B"
    QL_A = "QL-A"


class DetectionMethod(Enum):
    """Detection methods as per PAS 128."""
    ELECTROMAGNETIC = "electromagnetic"
    GROUND_PENETRATING_RADAR = "ground_penetrating_radar"
    RADIO_DETECTION = "radio_detection"
    INTRUSIVE_INVESTIGATION = "intrusive_investigation"


@dataclass
class QualityLevelSpecification:
    """Quality level specification from PAS 128."""
    level: QualityLevel
    description: str
    horizontal_accuracy: float  # in mm
    depth_accuracy: Optional[float]  # in mm, None if unspecified
    required_methods: List[DetectionMethod]
    required_deliverables: List[str]


@dataclass
class DetectionResult:
    """Individual utility detection result."""
    x_position: float
    y_position: float
    depth: Optional[float]
    material: Optional[str]
    diameter: Optional[float]
    discipline: Optional[str]
    confidence: float
    detection_method: DetectionMethod
    verified: bool = False


@dataclass
class SurveyDeliverables:
    """Survey deliverables for compliance checking."""
    survey_report: bool = False
    utility_location_plans: bool = False
    risk_assessment: bool = False
    detection_survey_results: bool = False
    intrusive_investigation_results: bool = False
    verification_photos: bool = False
    metadata_complete: bool = False


class ComplianceResult(NamedTuple):
    """Result of PAS 128 compliance validation."""
    compliant: bool
    achieved_quality_level: Optional[QualityLevel]
    horizontal_accuracy_mm: float
    depth_accuracy_mm: Optional[float]
    missing_deliverables: List[str]
    method_coverage: Dict[DetectionMethod, bool]
    compliance_score: float  # 0-1 scale


@dataclass
class ValidationMetrics:
    """Metrics for validation against ground truth."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    position_errors: List[float] = field(default_factory=list)
    depth_errors: List[float] = field(default_factory=list)
    material_matches: int = 0
    material_mismatches: int = 0


class PAS128ComplianceValidator:
    """Validator for PAS 128:2014 specification compliance."""

    def __init__(self, specification_path: Optional[Path] = None):
        """
        Initialize the PAS 128 compliance validator.

        Args:
            specification_path: Path to PAS 128 specification JSON file
        """
        self.specification_path = specification_path
        self.quality_levels: Dict[QualityLevel, QualityLevelSpecification] = {}
        self._load_specifications()

    def _load_specifications(self) -> None:
        """Load PAS 128 specifications from file or use defaults."""
        if self.specification_path and self.specification_path.exists():
            try:
                with open(self.specification_path, 'r', encoding='utf-8') as f:
                    spec_data = json.load(f)
                self._parse_specifications(spec_data)
                logger.info(f"Loaded PAS 128 specifications from {self.specification_path}")
            except Exception as e:
                logger.warning(f"Failed to load specifications from file: {e}")
                self._load_default_specifications()
        else:
            self._load_default_specifications()

    def _parse_specifications(self, spec_data: Dict[str, Any]) -> None:
        """Parse specifications from JSON data."""
        quality_levels_data = spec_data.get('quality_levels', {})

        for level_key, level_data in quality_levels_data.items():
            try:
                quality_level = QualityLevel(level_key)

                # Parse accuracy requirements
                accuracy_str = level_data.get('accuracy', '')
                horizontal_accuracy = self._parse_accuracy(accuracy_str, 'horizontal')
                depth_accuracy = self._parse_accuracy(accuracy_str, 'depth')

                # Parse required methods
                methods = level_data.get('methods', [])
                required_methods = self._parse_detection_methods(methods)

                # Parse deliverables
                deliverables = level_data.get('deliverables', [])

                self.quality_levels[quality_level] = QualityLevelSpecification(
                    level=quality_level,
                    description=level_data.get('description', ''),
                    horizontal_accuracy=horizontal_accuracy,
                    depth_accuracy=depth_accuracy,
                    required_methods=required_methods,
                    required_deliverables=deliverables
                )

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse quality level {level_key}: {e}")

    def _load_default_specifications(self) -> None:
        """Load default PAS 128 specifications."""
        logger.info("Loading default PAS 128 specifications")

        self.quality_levels = {
            QualityLevel.QL_D: QualityLevelSpecification(
                level=QualityLevel.QL_D,
                description="Desk study only",
                horizontal_accuracy=2000.0,
                depth_accuracy=None,
                required_methods=[],
                required_deliverables=["Survey report", "Utility location plans"]
            ),
            QualityLevel.QL_C: QualityLevelSpecification(
                level=QualityLevel.QL_C,
                description="Comprehensive records search with site reconnaissance",
                horizontal_accuracy=1000.0,
                depth_accuracy=None,
                required_methods=[],
                required_deliverables=["Survey report", "Utility location plans", "Risk assessment"]
            ),
            QualityLevel.QL_B: QualityLevelSpecification(
                level=QualityLevel.QL_B,
                description="QL-C plus detection using appropriate equipment",
                horizontal_accuracy=500.0,
                depth_accuracy=None,
                required_methods=[DetectionMethod.ELECTROMAGNETIC, DetectionMethod.GROUND_PENETRATING_RADAR],
                required_deliverables=["Survey report", "Utility location plans", "Risk assessment", "Detection survey results"]
            ),
            QualityLevel.QL_A: QualityLevelSpecification(
                level=QualityLevel.QL_A,
                description="QL-B plus verification by intrusive investigation",
                horizontal_accuracy=300.0,
                depth_accuracy=300.0,
                required_methods=[DetectionMethod.ELECTROMAGNETIC, DetectionMethod.GROUND_PENETRATING_RADAR, DetectionMethod.INTRUSIVE_INVESTIGATION],
                required_deliverables=["Survey report", "Utility location plans", "Risk assessment", "Detection survey results", "Intrusive investigation results", "Verification photos"]
            )
        }

    def _parse_accuracy(self, accuracy_str: str, accuracy_type: str) -> Optional[float]:
        """Parse accuracy value from string."""
        if not accuracy_str:
            return None

        # Extract numeric values (assuming format like "±300mm horizontally, ±300mm depth")
        if accuracy_type == 'horizontal':
            # Look for horizontal accuracy
            import re
            horizontal_match = re.search(r'±(\d+)mm.*horizontal', accuracy_str, re.IGNORECASE)
            if horizontal_match:
                return float(horizontal_match.group(1))
            # Fallback to first number if no specific horizontal mention
            number_match = re.search(r'±(\d+)mm', accuracy_str)
            if number_match:
                return float(number_match.group(1))

        elif accuracy_type == 'depth':
            # Look for depth accuracy
            import re
            depth_match = re.search(r'±(\d+)mm.*depth', accuracy_str, re.IGNORECASE)
            if depth_match:
                return float(depth_match.group(1))

        return None

    def _parse_detection_methods(self, methods: List[str]) -> List[DetectionMethod]:
        """Parse detection methods from specification."""
        parsed_methods = []
        method_mapping = {
            'electromagnetic detection': DetectionMethod.ELECTROMAGNETIC,
            'ground penetrating radar': DetectionMethod.GROUND_PENETRATING_RADAR,
            'radio detection': DetectionMethod.RADIO_DETECTION,
            'trial holes': DetectionMethod.INTRUSIVE_INVESTIGATION,
            'vacuum excavation': DetectionMethod.INTRUSIVE_INVESTIGATION,
            'hand digging': DetectionMethod.INTRUSIVE_INVESTIGATION
        }

        for method in methods:
            method_lower = method.lower()
            for key, detection_method in method_mapping.items():
                if key in method_lower:
                    if detection_method not in parsed_methods:
                        parsed_methods.append(detection_method)
                    break

        return parsed_methods

    def validate_compliance(
        self,
        detection_results: List[DetectionResult],
        ground_truth_utilities: List[Dict[str, Any]],
        deliverables: SurveyDeliverables,
        target_quality_level: QualityLevel = QualityLevel.QL_B
    ) -> ComplianceResult:
        """
        Validate PAS 128 compliance for a survey.

        Args:
            detection_results: List of detection results from survey
            ground_truth_utilities: Ground truth utility data for validation
            deliverables: Survey deliverables provided
            target_quality_level: Target quality level to validate against

        Returns:
            ComplianceResult with compliance status and metrics
        """
        logger.info(f"Validating PAS 128 compliance for quality level {target_quality_level.value}")

        if target_quality_level not in self.quality_levels:
            raise ValueError(f"Unknown quality level: {target_quality_level}")

        specification = self.quality_levels[target_quality_level]

        # Calculate accuracy metrics
        validation_metrics = self._calculate_validation_metrics(detection_results, ground_truth_utilities)
        horizontal_accuracy = self._calculate_horizontal_accuracy(validation_metrics.position_errors)
        depth_accuracy = self._calculate_depth_accuracy(validation_metrics.depth_errors)

        # Check method coverage
        method_coverage = self._check_method_coverage(detection_results, specification.required_methods)

        # Check deliverables completeness
        missing_deliverables = self._check_deliverables(deliverables, specification.required_deliverables)

        # Determine compliance
        accuracy_compliant = horizontal_accuracy <= specification.horizontal_accuracy
        if specification.depth_accuracy is not None:
            accuracy_compliant = accuracy_compliant and (depth_accuracy <= specification.depth_accuracy)

        methods_compliant = all(method_coverage.values())
        deliverables_compliant = len(missing_deliverables) == 0

        compliant = accuracy_compliant and methods_compliant and deliverables_compliant

        # Calculate achieved quality level
        achieved_quality_level = self._determine_achieved_quality_level(
            horizontal_accuracy, depth_accuracy, method_coverage, missing_deliverables
        )

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            horizontal_accuracy, depth_accuracy, specification, method_coverage, missing_deliverables
        )

        return ComplianceResult(
            compliant=compliant,
            achieved_quality_level=achieved_quality_level,
            horizontal_accuracy_mm=horizontal_accuracy,
            depth_accuracy_mm=depth_accuracy,
            missing_deliverables=missing_deliverables,
            method_coverage=method_coverage,
            compliance_score=compliance_score
        )

    def _calculate_validation_metrics(
        self,
        detection_results: List[DetectionResult],
        ground_truth_utilities: List[Dict[str, Any]]
    ) -> ValidationMetrics:
        """Calculate validation metrics against ground truth."""
        metrics = ValidationMetrics()

        # Match detections to ground truth utilities
        matches = self._match_detections_to_ground_truth(detection_results, ground_truth_utilities)

        for detection, ground_truth in matches:
            if ground_truth is not None:
                # True positive - calculate position error
                position_error = self._calculate_position_error(detection, ground_truth)
                metrics.position_errors.append(position_error)

                # Depth error (if both have depth)
                if detection.depth is not None and 'depth' in ground_truth:
                    depth_error = abs(detection.depth - ground_truth['depth'])
                    metrics.depth_errors.append(depth_error)

                # Material match
                if detection.material and 'material' in ground_truth:
                    if detection.material == ground_truth['material']:
                        metrics.material_matches += 1
                    else:
                        metrics.material_mismatches += 1

                metrics.true_positives += 1
            else:
                # False positive
                metrics.false_positives += 1

        # Count unmatched ground truth utilities (false negatives)
        matched_ground_truth = [gt for _, gt in matches if gt is not None]
        metrics.false_negatives = len(ground_truth_utilities) - len(matched_ground_truth)

        return metrics

    def _match_detections_to_ground_truth(
        self,
        detections: List[DetectionResult],
        ground_truth: List[Dict[str, Any]],
        distance_threshold: float = 1.0  # meters
    ) -> List[Tuple[DetectionResult, Optional[Dict[str, Any]]]]:
        """Match detection results to ground truth utilities."""
        matches = []
        used_ground_truth = set()

        for detection in detections:
            best_match = None
            best_distance = float('inf')

            for i, gt_utility in enumerate(ground_truth):
                if i in used_ground_truth:
                    continue

                distance = self._calculate_position_error(detection, gt_utility)
                if distance < best_distance and distance <= distance_threshold:
                    best_distance = distance
                    best_match = (i, gt_utility)

            if best_match:
                used_ground_truth.add(best_match[0])
                matches.append((detection, best_match[1]))
            else:
                matches.append((detection, None))

        return matches

    def _calculate_position_error(
        self,
        detection: DetectionResult,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate horizontal position error between detection and ground truth."""
        dx = detection.x_position - ground_truth.get('x_position', 0)
        dy = detection.y_position - ground_truth.get('y_position', 0)
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_horizontal_accuracy(self, position_errors: List[float]) -> float:
        """Calculate horizontal accuracy metric (95th percentile error in mm)."""
        if not position_errors:
            return 0.0

        # Convert to mm and calculate 95th percentile
        errors_mm = [error * 1000 for error in position_errors]  # Convert m to mm
        return np.percentile(errors_mm, 95)

    def _calculate_depth_accuracy(self, depth_errors: List[float]) -> Optional[float]:
        """Calculate depth accuracy metric (95th percentile error in mm)."""
        if not depth_errors:
            return None

        # Convert to mm and calculate 95th percentile
        errors_mm = [error * 1000 for error in depth_errors]  # Convert m to mm
        return np.percentile(errors_mm, 95)

    def _check_method_coverage(
        self,
        detection_results: List[DetectionResult],
        required_methods: List[DetectionMethod]
    ) -> Dict[DetectionMethod, bool]:
        """Check if all required detection methods were used."""
        used_methods = set(result.detection_method for result in detection_results)
        return {method: method in used_methods for method in required_methods}

    def _check_deliverables(
        self,
        provided_deliverables: SurveyDeliverables,
        required_deliverables: List[str]
    ) -> List[str]:
        """Check completeness of survey deliverables."""
        deliverable_mapping = {
            "Survey report": provided_deliverables.survey_report,
            "Utility location plans": provided_deliverables.utility_location_plans,
            "Risk assessment": provided_deliverables.risk_assessment,
            "Detection survey results": provided_deliverables.detection_survey_results,
            "Intrusive investigation results": provided_deliverables.intrusive_investigation_results,
            "Verification photos": provided_deliverables.verification_photos
        }

        missing = []
        for required in required_deliverables:
            if required in deliverable_mapping and not deliverable_mapping[required]:
                missing.append(required)
            elif required not in deliverable_mapping:
                missing.append(required)

        return missing

    def _determine_achieved_quality_level(
        self,
        horizontal_accuracy: float,
        depth_accuracy: Optional[float],
        method_coverage: Dict[DetectionMethod, bool],
        missing_deliverables: List[str]
    ) -> Optional[QualityLevel]:
        """Determine the highest quality level achieved."""
        # Check from highest to lowest quality level
        for quality_level in [QualityLevel.QL_A, QualityLevel.QL_B, QualityLevel.QL_C, QualityLevel.QL_D]:
            spec = self.quality_levels[quality_level]

            # Check accuracy
            accuracy_ok = horizontal_accuracy <= spec.horizontal_accuracy
            if spec.depth_accuracy is not None:
                accuracy_ok = accuracy_ok and (depth_accuracy is not None and depth_accuracy <= spec.depth_accuracy)

            # Check methods
            methods_ok = all(method_coverage.get(method, False) for method in spec.required_methods)

            # Check deliverables
            deliverables_ok = all(
                deliverable not in missing_deliverables
                for deliverable in spec.required_deliverables
            )

            if accuracy_ok and methods_ok and deliverables_ok:
                return quality_level

        return None

    def _calculate_compliance_score(
        self,
        horizontal_accuracy: float,
        depth_accuracy: Optional[float],
        specification: QualityLevelSpecification,
        method_coverage: Dict[DetectionMethod, bool],
        missing_deliverables: List[str]
    ) -> float:
        """Calculate overall compliance score (0-1)."""
        scores = []

        # Accuracy score
        accuracy_score = min(1.0, specification.horizontal_accuracy / max(horizontal_accuracy, 1.0))
        scores.append(accuracy_score)

        if specification.depth_accuracy is not None and depth_accuracy is not None:
            depth_score = min(1.0, specification.depth_accuracy / max(depth_accuracy, 1.0))
            scores.append(depth_score)

        # Method coverage score
        if specification.required_methods:
            method_score = sum(method_coverage.values()) / len(specification.required_methods)
            scores.append(method_score)

        # Deliverables score
        if specification.required_deliverables:
            deliverables_score = 1.0 - (len(missing_deliverables) / len(specification.required_deliverables))
            scores.append(deliverables_score)

        return np.mean(scores) if scores else 0.0

    def generate_compliance_report(
        self,
        compliance_result: ComplianceResult,
        validation_metrics: ValidationMetrics,
        target_quality_level: QualityLevel
    ) -> Dict[str, Any]:
        """Generate detailed compliance report."""
        report = {
            "compliance_summary": {
                "target_quality_level": target_quality_level.value,
                "achieved_quality_level": compliance_result.achieved_quality_level.value if compliance_result.achieved_quality_level else None,
                "compliant": compliance_result.compliant,
                "compliance_score": compliance_result.compliance_score
            },
            "accuracy_assessment": {
                "horizontal_accuracy_mm": compliance_result.horizontal_accuracy_mm,
                "depth_accuracy_mm": compliance_result.depth_accuracy_mm,
                "horizontal_accuracy_target_mm": self.quality_levels[target_quality_level].horizontal_accuracy,
                "depth_accuracy_target_mm": self.quality_levels[target_quality_level].depth_accuracy
            },
            "detection_performance": {
                "true_positives": validation_metrics.true_positives,
                "false_positives": validation_metrics.false_positives,
                "false_negatives": validation_metrics.false_negatives,
                "precision": validation_metrics.true_positives / max(validation_metrics.true_positives + validation_metrics.false_positives, 1),
                "recall": validation_metrics.true_positives / max(validation_metrics.true_positives + validation_metrics.false_negatives, 1)
            },
            "method_coverage": {method.value: covered for method, covered in compliance_result.method_coverage.items()},
            "deliverables_assessment": {
                "missing_deliverables": compliance_result.missing_deliverables,
                "completeness_percentage": max(0, 100 - len(compliance_result.missing_deliverables) * 100 / len(self.quality_levels[target_quality_level].required_deliverables))
            }
        }

        return report


def create_pas128_validator(specification_path: Optional[str] = None) -> PAS128ComplianceValidator:
    """
    Factory function to create a PAS 128 compliance validator.

    Args:
        specification_path: Optional path to PAS 128 specification file

    Returns:
        Configured PAS128ComplianceValidator instance
    """
    spec_path = Path(specification_path) if specification_path else None
    return PAS128ComplianceValidator(spec_path)