"""
PAS 128 Deliverables Assessment System

This service provides comprehensive assessment of survey deliverables
against PAS 128 requirements, including quality evaluation, completeness
checking, and format compliance validation.
"""
import logging
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from ..schemas.pas128 import (
    QualityLevel, DeliverableType, SurveyData, DeliverableItem,
    ComplianceCheck, UtilityDetection
)

logger = logging.getLogger(__name__)


class DeliverableFormat(Enum):
    """Supported deliverable formats"""
    PDF = "pdf"
    DRAWING = "dwg"
    CAD = "cad"
    SHAPEFILE = "shp"
    IMAGE = "jpg"
    DOCUMENT = "doc"
    EXCEL = "xlsx"
    CSV = "csv"


class DeliverableQuality(Enum):
    """Deliverable quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"
    INADEQUATE = "inadequate"


@dataclass
class DeliverableAssessment:
    """Detailed assessment of a deliverable"""
    deliverable_type: DeliverableType
    provided: bool
    quality_score: float  # 0.0 to 1.0
    completeness_score: float  # 0.0 to 1.0
    format_compliance: bool
    content_adequacy: float  # 0.0 to 1.0
    accuracy_assessment: float  # 0.0 to 1.0
    technical_compliance: float  # 0.0 to 1.0
    issues_identified: List[str]
    recommendations: List[str]
    file_analysis: Dict[str, Any]


@dataclass
class ReportSection:
    """Report section analysis"""
    section_name: str
    present: bool
    completeness: float
    quality: float
    content_summary: str
    issues: List[str]


@dataclass
class UtilityPlanAssessment:
    """Utility location plan assessment"""
    plan_type: str
    scale_appropriate: bool
    legend_present: bool
    utilities_marked: int
    accuracy_indicators: bool
    coordinate_system: Optional[str]
    completeness_score: float
    technical_quality: float


class PAS128DeliverablesAssessor:
    """
    Comprehensive deliverables assessment system for PAS 128 compliance.

    This service assesses:
    - Survey report completeness and quality
    - Utility location plan accuracy and format compliance
    - Risk assessment comprehensiveness
    - Detection survey results documentation
    - Intrusive investigation documentation
    - Verification photos quality and relevance
    """

    def __init__(self):
        """Initialize the deliverables assessment system."""
        self.logger = logging.getLogger(__name__)

        # Initialize deliverable requirements
        self.deliverable_requirements = self._initialize_deliverable_requirements()

        # Initialize quality assessment criteria
        self.quality_criteria = self._initialize_quality_criteria()

        # Initialize format specifications
        self.format_specs = self._initialize_format_specifications()

        # Initialize content requirements
        self.content_requirements = self._initialize_content_requirements()

        self.logger.info("PAS 128 Deliverables Assessor initialized")

    def _initialize_deliverable_requirements(self) -> Dict[QualityLevel, Dict[DeliverableType, Dict[str, Any]]]:
        """Initialize deliverable requirements for each quality level."""
        return {
            QualityLevel.QL_D: {
                DeliverableType.SURVEY_REPORT: {
                    "required": True,
                    "minimum_sections": ["executive_summary", "methodology", "findings", "recommendations"],
                    "minimum_pages": 5,
                    "technical_detail_level": "basic"
                },
                DeliverableType.UTILITY_LOCATION_PLANS: {
                    "required": True,
                    "minimum_scale": "1:500",
                    "accuracy_indication": True,
                    "legend_required": True,
                    "format_options": ["pdf", "dwg", "shp"]
                }
            },
            QualityLevel.QL_C: {
                DeliverableType.SURVEY_REPORT: {
                    "required": True,
                    "minimum_sections": ["executive_summary", "methodology", "findings", "risk_assessment", "recommendations", "limitations"],
                    "minimum_pages": 8,
                    "technical_detail_level": "intermediate"
                },
                DeliverableType.UTILITY_LOCATION_PLANS: {
                    "required": True,
                    "minimum_scale": "1:250",
                    "accuracy_indication": True,
                    "legend_required": True,
                    "coordinate_system_required": True,
                    "format_options": ["pdf", "dwg", "shp"]
                },
                DeliverableType.RISK_ASSESSMENT: {
                    "required": True,
                    "minimum_sections": ["risk_identification", "risk_evaluation", "mitigation_measures"],
                    "risk_matrix_required": True,
                    "quantitative_assessment": False
                }
            },
            QualityLevel.QL_B: {
                DeliverableType.DETECTION_SURVEY_RESULTS: {
                    "required": True,
                    "equipment_details": True,
                    "detection_logs": True,
                    "confidence_levels": True,
                    "method_limitations": True,
                    "raw_data_inclusion": "optional"
                }
            },
            QualityLevel.QL_A: {
                DeliverableType.INTRUSIVE_INVESTIGATION_RESULTS: {
                    "required": True,
                    "excavation_logs": True,
                    "verification_data": True,
                    "utility_confirmation": True,
                    "depth_measurements": True,
                    "material_identification": True
                },
                DeliverableType.VERIFICATION_PHOTOS: {
                    "required": True,
                    "minimum_photos_per_excavation": 3,
                    "photo_quality_requirements": "high_resolution",
                    "metadata_required": True,
                    "before_during_after": True
                }
            }
        }

    def _initialize_quality_criteria(self) -> Dict[str, Dict[str, float]]:
        """Initialize quality assessment criteria."""
        return {
            "report_quality": {
                "excellent": 0.95,
                "good": 0.85,
                "satisfactory": 0.70,
                "poor": 0.50,
                "inadequate": 0.30
            },
            "completeness_thresholds": {
                "complete": 0.95,
                "mostly_complete": 0.80,
                "partially_complete": 0.60,
                "incomplete": 0.40,
                "severely_incomplete": 0.20
            },
            "technical_quality": {
                "professional": 0.90,
                "adequate": 0.75,
                "basic": 0.60,
                "poor": 0.40,
                "unacceptable": 0.20
            }
        }

    def _initialize_format_specifications(self) -> Dict[DeliverableType, Dict[str, Any]]:
        """Initialize format specifications for deliverables."""
        return {
            DeliverableType.SURVEY_REPORT: {
                "preferred_formats": ["pdf", "doc", "docx"],
                "minimum_resolution": None,
                "file_size_limits": {"min": 1, "max": 50},  # MB
                "content_requirements": ["text", "diagrams", "tables"]
            },
            DeliverableType.UTILITY_LOCATION_PLANS: {
                "preferred_formats": ["dwg", "pdf", "shp"],
                "minimum_resolution": 300,  # DPI for PDF
                "scale_requirements": ["1:250", "1:500", "1:1000"],
                "content_requirements": ["legend", "scale", "coordinates", "utility_symbols"]
            },
            DeliverableType.RISK_ASSESSMENT: {
                "preferred_formats": ["pdf", "doc", "xlsx"],
                "content_requirements": ["risk_matrix", "mitigation_measures", "probability_impact"]
            },
            DeliverableType.DETECTION_SURVEY_RESULTS: {
                "preferred_formats": ["pdf", "xlsx", "csv"],
                "content_requirements": ["equipment_logs", "detection_data", "confidence_levels"]
            },
            DeliverableType.INTRUSIVE_INVESTIGATION_RESULTS: {
                "preferred_formats": ["pdf", "doc", "xlsx"],
                "content_requirements": ["excavation_logs", "utility_verification", "measurements"]
            },
            DeliverableType.VERIFICATION_PHOTOS: {
                "preferred_formats": ["jpg", "png", "tiff"],
                "minimum_resolution": 1920,  # pixels width
                "metadata_requirements": ["timestamp", "location", "equipment"]
            }
        }

    def _initialize_content_requirements(self) -> Dict[DeliverableType, List[str]]:
        """Initialize content requirements for each deliverable type."""
        return {
            DeliverableType.SURVEY_REPORT: [
                "project_overview",
                "survey_objectives",
                "methodology_description",
                "equipment_specifications",
                "environmental_conditions",
                "findings_summary",
                "utility_inventory",
                "accuracy_statement",
                "limitations_constraints",
                "recommendations",
                "appendices"
            ],
            DeliverableType.UTILITY_LOCATION_PLANS: [
                "site_boundaries",
                "utility_locations",
                "utility_types",
                "depth_information",
                "accuracy_indicators",
                "legend",
                "scale_bar",
                "north_arrow",
                "coordinate_grid",
                "revision_information"
            ],
            DeliverableType.RISK_ASSESSMENT: [
                "risk_identification",
                "likelihood_assessment",
                "impact_assessment",
                "risk_matrix",
                "mitigation_measures",
                "residual_risk",
                "monitoring_requirements"
            ],
            DeliverableType.DETECTION_SURVEY_RESULTS: [
                "survey_parameters",
                "equipment_settings",
                "detection_logs",
                "signal_responses",
                "confidence_levels",
                "verification_status",
                "method_limitations"
            ],
            DeliverableType.INTRUSIVE_INVESTIGATION_RESULTS: [
                "excavation_locations",
                "excavation_methods",
                "utility_confirmations",
                "depth_measurements",
                "material_descriptions",
                "condition_assessments",
                "safety_observations"
            ],
            DeliverableType.VERIFICATION_PHOTOS: [
                "pre_excavation_photos",
                "during_excavation_photos",
                "utility_exposure_photos",
                "measurement_photos",
                "restoration_photos"
            ]
        }

    def assess_deliverable(self, deliverable: DeliverableItem,
                          quality_level: QualityLevel,
                          survey_data: Optional[SurveyData] = None) -> DeliverableAssessment:
        """
        Assess a single deliverable against PAS 128 requirements.

        Args:
            deliverable: Deliverable item to assess
            quality_level: Target quality level for assessment
            survey_data: Optional survey data for context

        Returns:
            Detailed deliverable assessment
        """
        self.logger.debug(f"Assessing deliverable: {deliverable.deliverable_type}")

        deliverable_type = deliverable.deliverable_type
        issues = []
        recommendations = []

        # Check if deliverable is provided
        if not deliverable.provided:
            return DeliverableAssessment(
                deliverable_type=deliverable_type,
                provided=False,
                quality_score=0.0,
                completeness_score=0.0,
                format_compliance=False,
                content_adequacy=0.0,
                accuracy_assessment=0.0,
                technical_compliance=0.0,
                issues_identified=["Deliverable not provided"],
                recommendations=[f"Provide required {deliverable_type.value.replace('_', ' ')}"],
                file_analysis={"provided": False}
            )

        # Assess file format compliance
        format_compliance, format_issues = self._assess_format_compliance(deliverable)
        issues.extend(format_issues)

        # Assess content adequacy
        content_adequacy, content_issues = self._assess_content_adequacy(
            deliverable, quality_level, survey_data
        )
        issues.extend(content_issues)

        # Assess technical compliance
        technical_compliance, technical_issues = self._assess_technical_compliance(
            deliverable, quality_level
        )
        issues.extend(technical_issues)

        # Assess accuracy (for location-based deliverables)
        accuracy_assessment = self._assess_accuracy_compliance(deliverable, survey_data)

        # Use provided scores or calculate defaults
        quality_score = deliverable.quality_score if deliverable.quality_score is not None else content_adequacy
        completeness_score = deliverable.completeness_score if deliverable.completeness_score is not None else content_adequacy

        # Generate recommendations
        recommendations = self._generate_deliverable_recommendations(
            deliverable_type, issues, quality_score, quality_level
        )

        # Perform file analysis
        file_analysis = self._analyze_deliverable_file(deliverable)

        return DeliverableAssessment(
            deliverable_type=deliverable_type,
            provided=True,
            quality_score=quality_score,
            completeness_score=completeness_score,
            format_compliance=format_compliance,
            content_adequacy=content_adequacy,
            accuracy_assessment=accuracy_assessment,
            technical_compliance=technical_compliance,
            issues_identified=issues,
            recommendations=recommendations,
            file_analysis=file_analysis
        )

    def _assess_format_compliance(self, deliverable: DeliverableItem) -> Tuple[bool, List[str]]:
        """Assess format compliance of deliverable."""
        issues = []
        deliverable_type = deliverable.deliverable_type

        if not deliverable.file_path:
            issues.append("No file path provided for deliverable")
            return False, issues

        # Extract file extension
        file_extension = Path(deliverable.file_path).suffix.lower().lstrip('.')

        # Check format specifications
        format_specs = self.format_specs.get(deliverable_type, {})
        preferred_formats = format_specs.get("preferred_formats", [])

        if preferred_formats and file_extension not in preferred_formats:
            issues.append(f"File format '{file_extension}' not in preferred formats: {preferred_formats}")
            return False, issues

        # Check if file exists (if it's a real path)
        if os.path.exists(deliverable.file_path):
            # Check file size
            file_size_mb = os.path.getsize(deliverable.file_path) / (1024 * 1024)
            size_limits = format_specs.get("file_size_limits", {})

            if "min" in size_limits and file_size_mb < size_limits["min"]:
                issues.append(f"File size too small: {file_size_mb:.1f}MB < {size_limits['min']}MB")

            if "max" in size_limits and file_size_mb > size_limits["max"]:
                issues.append(f"File size too large: {file_size_mb:.1f}MB > {size_limits['max']}MB")

        return len(issues) == 0, issues

    def _assess_content_adequacy(self, deliverable: DeliverableItem,
                               quality_level: QualityLevel,
                               survey_data: Optional[SurveyData]) -> Tuple[float, List[str]]:
        """Assess content adequacy of deliverable."""
        issues = []
        deliverable_type = deliverable.deliverable_type

        # Get content requirements
        content_requirements = self.content_requirements.get(deliverable_type, [])
        if not content_requirements:
            return 0.8, []  # Default score if no specific requirements

        # Simulate content analysis (in real implementation, this would parse the actual file)
        content_adequacy_score = self._simulate_content_analysis(
            deliverable, quality_level, content_requirements
        )

        # Check specific deliverable requirements
        deliverable_reqs = self.deliverable_requirements.get(quality_level, {}).get(deliverable_type, {})

        # Assess specific content based on deliverable type
        if deliverable_type == DeliverableType.SURVEY_REPORT:
            report_issues = self._assess_survey_report_content(deliverable, deliverable_reqs, survey_data)
            issues.extend(report_issues)

        elif deliverable_type == DeliverableType.UTILITY_LOCATION_PLANS:
            plan_issues = self._assess_utility_plans_content(deliverable, deliverable_reqs, survey_data)
            issues.extend(plan_issues)

        elif deliverable_type == DeliverableType.RISK_ASSESSMENT:
            risk_issues = self._assess_risk_assessment_content(deliverable, deliverable_reqs)
            issues.extend(risk_issues)

        elif deliverable_type == DeliverableType.VERIFICATION_PHOTOS:
            photo_issues = self._assess_verification_photos_content(deliverable, deliverable_reqs, survey_data)
            issues.extend(photo_issues)

        # Adjust score based on issues
        if issues:
            content_adequacy_score *= (1 - len(issues) * 0.1)
            content_adequacy_score = max(0.0, content_adequacy_score)

        return content_adequacy_score, issues

    def _simulate_content_analysis(self, deliverable: DeliverableItem,
                                 quality_level: QualityLevel,
                                 content_requirements: List[str]) -> float:
        """Simulate content analysis (placeholder for actual file parsing)."""
        # In a real implementation, this would parse the actual file content
        # For now, we simulate based on deliverable quality score or reasonable defaults

        if deliverable.quality_score is not None:
            return deliverable.quality_score

        # Simulate based on deliverable type and quality level
        base_scores = {
            QualityLevel.QL_D: 0.7,
            QualityLevel.QL_C: 0.75,
            QualityLevel.QL_B: 0.8,
            QualityLevel.QL_A: 0.85
        }

        return base_scores.get(quality_level, 0.7)

    def _assess_survey_report_content(self, deliverable: DeliverableItem,
                                    requirements: Dict[str, Any],
                                    survey_data: Optional[SurveyData]) -> List[str]:
        """Assess survey report content adequacy."""
        issues = []

        # Check minimum sections requirement
        min_sections = requirements.get("minimum_sections", [])
        if min_sections and not deliverable.notes:
            issues.append("Cannot verify required sections - no content summary provided")

        # Check technical detail level
        detail_level = requirements.get("technical_detail_level", "basic")
        if detail_level == "advanced" and deliverable.quality_score and deliverable.quality_score < 0.8:
            issues.append(f"Technical detail level may be insufficient for {detail_level} requirements")

        # Cross-reference with survey data
        if survey_data:
            # Check if report addresses all executed methods
            if len(survey_data.methods_executed) > 2 and not deliverable.notes:
                issues.append("Report should address all executed survey methods")

            # Check if environmental conditions are documented
            if survey_data.environmental_conditions.soil_type and not deliverable.notes:
                issues.append("Environmental conditions should be documented in report")

        return issues

    def _assess_utility_plans_content(self, deliverable: DeliverableItem,
                                    requirements: Dict[str, Any],
                                    survey_data: Optional[SurveyData]) -> List[str]:
        """Assess utility location plans content."""
        issues = []

        # Check scale requirements
        min_scale = requirements.get("minimum_scale")
        if min_scale and not deliverable.notes:
            issues.append(f"Cannot verify scale meets minimum requirement: {min_scale}")

        # Check legend requirement
        if requirements.get("legend_required", False) and not deliverable.notes:
            issues.append("Legend presence cannot be verified")

        # Check coordinate system requirement
        if requirements.get("coordinate_system_required", False) and not deliverable.notes:
            issues.append("Coordinate system specification cannot be verified")

        # Cross-reference with detected utilities
        if survey_data and survey_data.utility_detections:
            detected_count = len(survey_data.utility_detections)
            if detected_count > 5 and not deliverable.notes:
                issues.append(f"Plans should show all {detected_count} detected utilities")

        return issues

    def _assess_risk_assessment_content(self, deliverable: DeliverableItem,
                                      requirements: Dict[str, Any]) -> List[str]:
        """Assess risk assessment content."""
        issues = []

        # Check risk matrix requirement
        if requirements.get("risk_matrix_required", False) and not deliverable.notes:
            issues.append("Risk matrix presence cannot be verified")

        # Check minimum sections
        min_sections = requirements.get("minimum_sections", [])
        if len(min_sections) > 2 and not deliverable.notes:
            issues.append("Cannot verify all required risk assessment sections")

        # Check quantitative assessment requirement
        if requirements.get("quantitative_assessment", False) and deliverable.quality_score and deliverable.quality_score < 0.8:
            issues.append("Quantitative risk assessment may be inadequate")

        return issues

    def _assess_verification_photos_content(self, deliverable: DeliverableItem,
                                          requirements: Dict[str, Any],
                                          survey_data: Optional[SurveyData]) -> List[str]:
        """Assess verification photos content."""
        issues = []

        # Check minimum photos per excavation
        min_photos = requirements.get("minimum_photos_per_excavation", 3)
        if survey_data:
            # Count intrusive methods executed
            intrusive_methods = sum(1 for method in survey_data.methods_executed
                                  if method.method.value in ["trial_holes", "vacuum_excavation", "hand_digging"])
            expected_photos = intrusive_methods * min_photos

            if expected_photos > 0 and not deliverable.notes:
                issues.append(f"Expected at least {expected_photos} verification photos")

        # Check photo quality requirements
        quality_req = requirements.get("photo_quality_requirements")
        if quality_req == "high_resolution" and not deliverable.notes:
            issues.append("Cannot verify high resolution photo requirement")

        # Check metadata requirement
        if requirements.get("metadata_required", False) and not deliverable.notes:
            issues.append("Photo metadata presence cannot be verified")

        return issues

    def _assess_technical_compliance(self, deliverable: DeliverableItem,
                                   quality_level: QualityLevel) -> Tuple[float, List[str]]:
        """Assess technical compliance of deliverable."""
        issues = []

        # Get deliverable requirements for quality level
        deliverable_reqs = self.deliverable_requirements.get(quality_level, {}).get(deliverable.deliverable_type, {})

        # Check if deliverable is required
        if not deliverable_reqs.get("required", False):
            return 1.0, []  # Not required, so fully compliant

        # Assess technical quality based on available information
        technical_score = 0.8  # Default reasonable score

        # Adjust based on quality score if available
        if deliverable.quality_score is not None:
            technical_score = deliverable.quality_score

        # Check specific technical requirements
        if deliverable.deliverable_type == DeliverableType.UTILITY_LOCATION_PLANS:
            if not deliverable_reqs.get("accuracy_indication", False):
                issues.append("Accuracy indication may be missing from utility plans")

        elif deliverable.deliverable_type == DeliverableType.VERIFICATION_PHOTOS:
            if deliverable_reqs.get("metadata_required", False) and not deliverable.notes:
                issues.append("Photo metadata requirements may not be met")

        # Adjust score based on issues
        if issues:
            technical_score *= (1 - len(issues) * 0.15)

        return max(0.0, technical_score), issues

    def _assess_accuracy_compliance(self, deliverable: DeliverableItem,
                                  survey_data: Optional[SurveyData]) -> float:
        """Assess accuracy compliance for location-based deliverables."""
        if deliverable.deliverable_type not in [DeliverableType.UTILITY_LOCATION_PLANS]:
            return 1.0  # Not applicable

        if not survey_data or not survey_data.utility_detections:
            return 0.5  # Cannot assess without detection data

        # Calculate average detection accuracy
        accuracies = [det.accuracy.horizontal_accuracy for det in survey_data.utility_detections]
        avg_accuracy = np.mean(accuracies)

        # Assess against quality level requirements
        quality_level = survey_data.target_quality_level
        accuracy_thresholds = {
            QualityLevel.QL_D: 2000.0,
            QualityLevel.QL_C: 1000.0,
            QualityLevel.QL_B: 500.0,
            QualityLevel.QL_A: 300.0
        }

        threshold = accuracy_thresholds.get(quality_level, 1000.0)
        if avg_accuracy <= threshold:
            return 1.0
        else:
            return threshold / avg_accuracy  # Proportional score

    def _generate_deliverable_recommendations(self, deliverable_type: DeliverableType,
                                            issues: List[str],
                                            quality_score: float,
                                            quality_level: QualityLevel) -> List[str]:
        """Generate recommendations for deliverable improvement."""
        recommendations = []

        # Format recommendations
        if any("format" in issue.lower() for issue in issues):
            format_specs = self.format_specs.get(deliverable_type, {})
            preferred_formats = format_specs.get("preferred_formats", [])
            if preferred_formats:
                recommendations.append(f"Use preferred formats: {', '.join(preferred_formats)}")

        # Content recommendations
        if any("section" in issue.lower() or "content" in issue.lower() for issue in issues):
            if deliverable_type == DeliverableType.SURVEY_REPORT:
                recommendations.append("Ensure all required report sections are included and adequately detailed")
                recommendations.append("Include comprehensive methodology description and limitations")

            elif deliverable_type == DeliverableType.UTILITY_LOCATION_PLANS:
                recommendations.append("Include comprehensive legend, scale, and coordinate system information")
                recommendations.append("Ensure all detected utilities are accurately plotted with confidence indicators")

            elif deliverable_type == DeliverableType.RISK_ASSESSMENT:
                recommendations.append("Include quantitative risk matrix and detailed mitigation measures")

        # Quality improvement recommendations
        if quality_score < 0.7:
            recommendations.append("Improve overall deliverable quality through better documentation and presentation")

        # Technical recommendations
        if any("technical" in issue.lower() or "compliance" in issue.lower() for issue in issues):
            recommendations.append("Ensure compliance with relevant technical standards and PAS 128 requirements")

        # Quality level specific recommendations
        if quality_level in [QualityLevel.QL_B, QualityLevel.QL_A]:
            if deliverable_type == DeliverableType.DETECTION_SURVEY_RESULTS:
                recommendations.append("Include detailed equipment logs and confidence level assessments")

        if quality_level == QualityLevel.QL_A:
            if deliverable_type == DeliverableType.VERIFICATION_PHOTOS:
                recommendations.append("Provide high-resolution photos with metadata for all excavation stages")

        return recommendations

    def _analyze_deliverable_file(self, deliverable: DeliverableItem) -> Dict[str, Any]:
        """Analyze deliverable file properties."""
        analysis = {
            "file_path": deliverable.file_path,
            "file_exists": False,
            "file_size_mb": 0.0,
            "file_extension": None,
            "analysis_timestamp": datetime.now().isoformat()
        }

        if deliverable.file_path:
            file_path = Path(deliverable.file_path)
            analysis["file_extension"] = file_path.suffix.lower().lstrip('.')

            if file_path.exists():
                analysis["file_exists"] = True
                analysis["file_size_mb"] = file_path.stat().st_size / (1024 * 1024)
                analysis["modification_time"] = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

        return analysis

    def assess_all_deliverables(self, survey_data: SurveyData) -> Dict[DeliverableType, DeliverableAssessment]:
        """
        Assess all deliverables in survey data.

        Args:
            survey_data: Complete survey data to assess

        Returns:
            Dictionary mapping deliverable types to their assessments
        """
        self.logger.info(f"Assessing all deliverables for survey {survey_data.survey_id}")

        assessments = {}

        for deliverable in survey_data.deliverables:
            assessment = self.assess_deliverable(
                deliverable,
                survey_data.target_quality_level,
                survey_data
            )
            assessments[deliverable.deliverable_type] = assessment

        return assessments

    def generate_deliverables_compliance_report(self, survey_data: SurveyData) -> List[ComplianceCheck]:
        """
        Generate compliance checks for all deliverables.

        Args:
            survey_data: Survey data to generate compliance report for

        Returns:
            List of compliance checks for deliverables
        """
        self.logger.info(f"Generating deliverables compliance report for survey {survey_data.survey_id}")

        assessments = self.assess_all_deliverables(survey_data)
        compliance_checks = []

        for deliverable_type, assessment in assessments.items():
            check = ComplianceCheck(
                check_name=f"Deliverable: {deliverable_type.value.replace('_', ' ').title()}",
                passed=assessment.provided and assessment.quality_score >= 0.7,
                score=assessment.quality_score if assessment.provided else 0.0,
                details=f"Provided: {assessment.provided}, Quality: {assessment.quality_score:.2f}, "
                       f"Completeness: {assessment.completeness_score:.2f}",
                requirements=[
                    "Deliverable provided",
                    "Format compliance",
                    "Content adequacy",
                    "Technical compliance"
                ],
                gaps=assessment.issues_identified
            )
            compliance_checks.append(check)

        # Overall deliverables compliance check
        provided_count = sum(1 for assessment in assessments.values() if assessment.provided)
        total_count = len(assessments)
        overall_quality = np.mean([
            assessment.quality_score for assessment in assessments.values()
            if assessment.provided
        ]) if any(assessment.provided for assessment in assessments.values()) else 0.0

        overall_check = ComplianceCheck(
            check_name="Overall Deliverables Compliance",
            passed=provided_count == total_count and overall_quality >= 0.7,
            score=overall_quality,
            details=f"Provided: {provided_count}/{total_count}, Average quality: {overall_quality:.2f}",
            requirements=["All required deliverables provided and meet quality standards"],
            gaps=[
                f"Deliverable {deliv_type.value} not provided or below quality threshold"
                for deliv_type, assessment in assessments.items()
                if not assessment.provided or assessment.quality_score < 0.7
            ]
        )
        compliance_checks.append(overall_check)

        return compliance_checks

    def get_deliverable_requirements_summary(self, quality_level: QualityLevel) -> Dict[str, Any]:
        """
        Get summary of deliverable requirements for a quality level.

        Args:
            quality_level: Quality level to get requirements for

        Returns:
            Summary of deliverable requirements
        """
        requirements = self.deliverable_requirements.get(quality_level, {})

        summary = {
            "quality_level": quality_level.value,
            "total_deliverables": len(requirements),
            "required_deliverables": [],
            "optional_deliverables": [],
            "key_requirements": {}
        }

        for deliverable_type, req_details in requirements.items():
            if req_details.get("required", False):
                summary["required_deliverables"].append(deliverable_type.value)
            else:
                summary["optional_deliverables"].append(deliverable_type.value)

            summary["key_requirements"][deliverable_type.value] = {
                "format_options": req_details.get("format_options", []),
                "minimum_sections": req_details.get("minimum_sections", []),
                "special_requirements": [
                    key for key, value in req_details.items()
                    if key not in ["required", "format_options", "minimum_sections"] and value
                ]
            }

        return summary