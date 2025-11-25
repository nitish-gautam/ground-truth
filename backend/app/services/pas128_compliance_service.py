"""
PAS 128:2022 Compliance Checking Service

This service implements comprehensive PAS 128 compliance checking,
automated quality level determination, and compliance reporting.
"""
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..schemas.pas128 import (
    QualityLevel, SurveyMethod, DeliverableType, DetectionMethod,
    SurveyData, ComplianceReport, QualityLevelAssessment,
    EnvironmentalImpactAssessment, ComplianceCheck, QualityLevelRequirements,
    AccuracyMeasurement, MethodExecution, DeliverableItem,
    EnvironmentalCondition, UtilityDetection
)

logger = logging.getLogger(__name__)


class PAS128ComplianceService:
    """
    Comprehensive PAS 128:2022 compliance checking and quality level determination service.

    This service provides:
    - Automated quality level determination
    - Method requirement validation
    - Deliverables assessment
    - Environmental factor integration
    - Compliance scoring and reporting
    """

    def __init__(self, pas128_specs_path: Optional[str] = None):
        """Initialize the PAS 128 compliance service."""
        self.logger = logging.getLogger(__name__)

        # Load PAS 128 specifications
        if pas128_specs_path is None:
            # Default path to PAS 128 specifications
            pas128_specs_path = "/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/pas128_docs/quality_levels_specification.json"

        self.pas128_specs = self._load_pas128_specifications(pas128_specs_path)
        self.quality_level_requirements = self._build_quality_level_requirements()

        self.logger.info("PAS 128 Compliance Service initialized successfully")

    def _load_pas128_specifications(self, specs_path: str) -> Dict[str, Any]:
        """Load PAS 128 specifications from JSON file."""
        try:
            with open(specs_path, 'r') as f:
                specs = json.load(f)
            self.logger.info(f"Loaded PAS 128 specifications from {specs_path}")
            return specs
        except FileNotFoundError:
            self.logger.warning(f"PAS 128 specifications file not found at {specs_path}, using defaults")
            return {}  # Return empty dict, will use hardcoded defaults
        except Exception as e:
            self.logger.error(f"Failed to load PAS 128 specifications: {e}")
            return {}  # Return empty dict instead of raising

    def _build_quality_level_requirements(self) -> Dict[QualityLevel, QualityLevelRequirements]:
        """Build structured quality level requirements from specifications."""
        requirements = {}

        # QL-D Requirements
        requirements[QualityLevel.QL_D] = QualityLevelRequirements(
            quality_level=QualityLevel.QL_D,
            required_methods=[
                SurveyMethod.RECORDS_SEARCH,
                SurveyMethod.SITE_RECONNAISSANCE
            ],
            required_deliverables=[
                DeliverableType.SURVEY_REPORT,
                DeliverableType.UTILITY_LOCATION_PLANS
            ],
            accuracy_requirements={"horizontal": 2000.0}  # ±2000mm
        )

        # QL-C Requirements
        requirements[QualityLevel.QL_C] = QualityLevelRequirements(
            quality_level=QualityLevel.QL_C,
            required_methods=[
                SurveyMethod.COMPREHENSIVE_RECORDS,
                SurveyMethod.SITE_RECONNAISSANCE,
                SurveyMethod.TOPOGRAPHICAL_SURVEY
            ],
            required_deliverables=[
                DeliverableType.SURVEY_REPORT,
                DeliverableType.UTILITY_LOCATION_PLANS,
                DeliverableType.RISK_ASSESSMENT
            ],
            accuracy_requirements={"horizontal": 1000.0}  # ±1000mm
        )

        # QL-B Requirements
        requirements[QualityLevel.QL_B] = QualityLevelRequirements(
            quality_level=QualityLevel.QL_B,
            required_methods=[
                SurveyMethod.COMPREHENSIVE_RECORDS,
                SurveyMethod.SITE_RECONNAISSANCE,
                SurveyMethod.TOPOGRAPHICAL_SURVEY,
                SurveyMethod.ELECTROMAGNETIC_DETECTION,
                SurveyMethod.GROUND_PENETRATING_RADAR
            ],
            required_deliverables=[
                DeliverableType.SURVEY_REPORT,
                DeliverableType.UTILITY_LOCATION_PLANS,
                DeliverableType.RISK_ASSESSMENT,
                DeliverableType.DETECTION_SURVEY_RESULTS
            ],
            accuracy_requirements={"horizontal": 500.0}  # ±500mm horizontally
        )

        # QL-A Requirements
        requirements[QualityLevel.QL_A] = QualityLevelRequirements(
            quality_level=QualityLevel.QL_A,
            required_methods=[
                SurveyMethod.COMPREHENSIVE_RECORDS,
                SurveyMethod.SITE_RECONNAISSANCE,
                SurveyMethod.TOPOGRAPHICAL_SURVEY,
                SurveyMethod.ELECTROMAGNETIC_DETECTION,
                SurveyMethod.GROUND_PENETRATING_RADAR,
                SurveyMethod.TRIAL_HOLES,
                SurveyMethod.VACUUM_EXCAVATION,
                SurveyMethod.HAND_DIGGING
            ],
            required_deliverables=[
                DeliverableType.SURVEY_REPORT,
                DeliverableType.UTILITY_LOCATION_PLANS,
                DeliverableType.RISK_ASSESSMENT,
                DeliverableType.DETECTION_SURVEY_RESULTS,
                DeliverableType.INTRUSIVE_INVESTIGATION_RESULTS,
                DeliverableType.VERIFICATION_PHOTOS
            ],
            accuracy_requirements={"horizontal": 300.0, "vertical": 300.0}  # ±300mm both
        )

        return requirements

    def assess_environmental_impact(self, environmental_conditions: EnvironmentalCondition) -> EnvironmentalImpactAssessment:
        """
        Assess the impact of environmental conditions on survey method effectiveness.

        Args:
            environmental_conditions: Environmental conditions during survey

        Returns:
            Environmental impact assessment
        """
        self.logger.info("Assessing environmental impact on survey methods")

        # Assess soil impact on GPR
        soil_gpr_impact = self._assess_soil_gpr_impact(environmental_conditions.soil_type)

        # Assess weather impact
        weather_impact = self._assess_weather_impact(environmental_conditions.weather_conditions)

        # Assess site constraints impact
        site_impact = self._assess_site_constraints_impact(environmental_conditions.site_constraints)

        # Calculate overall environmental score
        overall_score = np.mean([soil_gpr_impact, weather_impact, site_impact])

        # Assess method effectiveness
        method_effectiveness = {
            DetectionMethod.GROUND_PENETRATING_RADAR: soil_gpr_impact * weather_impact,
            DetectionMethod.ELECTROMAGNETIC: weather_impact * site_impact,
            DetectionMethod.RADIO_DETECTION: weather_impact * 0.9,  # Less affected by soil
            DetectionMethod.VISUAL_INSPECTION: weather_impact * 0.8
        }

        # Generate recommendations
        recommendations = self._generate_environmental_recommendations(
            environmental_conditions, soil_gpr_impact, weather_impact, site_impact
        )

        return EnvironmentalImpactAssessment(
            soil_impact_on_gpr=soil_gpr_impact,
            weather_impact=weather_impact,
            site_constraints_impact=site_impact,
            overall_environmental_score=overall_score,
            method_effectiveness=method_effectiveness,
            recommended_adjustments=recommendations
        )

    def _assess_soil_gpr_impact(self, soil_type: str) -> float:
        """Assess impact of soil type on GPR effectiveness."""
        soil_impacts = {
            "clay": 0.3,      # Poor GPR performance
            "wet_clay": 0.2,  # Very poor GPR performance
            "sand": 0.8,      # Good GPR performance
            "gravel": 0.7,    # Moderate GPR performance
            "silt": 0.6,      # Moderate GPR performance
            "peat": 0.4,      # Poor GPR performance
            "bedrock": 0.9,   # Excellent GPR performance
            "mixed": 0.6      # Average performance
        }

        return soil_impacts.get(soil_type.lower(), 0.6)  # Default moderate impact

    def _assess_weather_impact(self, weather_conditions: Optional[str]) -> float:
        """Assess impact of weather on survey quality."""
        if not weather_conditions:
            return 0.8  # Assume reasonable conditions

        weather = weather_conditions.lower()

        if any(condition in weather for condition in ["heavy rain", "storm", "severe"]):
            return 0.3
        elif any(condition in weather for condition in ["rain", "wet", "snow"]):
            return 0.6
        elif any(condition in weather for condition in ["fog", "mist"]):
            return 0.7
        elif any(condition in weather for condition in ["clear", "dry", "sunny"]):
            return 1.0
        else:
            return 0.8  # Default reasonable conditions

    def _assess_site_constraints_impact(self, site_constraints: List[str]) -> float:
        """Assess impact of site constraints on survey quality."""
        if not site_constraints:
            return 1.0  # No constraints

        high_impact_constraints = ["traffic", "active construction", "limited access", "hazardous area"]
        medium_impact_constraints = ["utilities present", "soft ground", "vegetation"]

        high_impact_count = sum(1 for constraint in site_constraints
                               if any(impact in constraint.lower() for impact in high_impact_constraints))
        medium_impact_count = sum(1 for constraint in site_constraints
                                 if any(impact in constraint.lower() for impact in medium_impact_constraints))

        # Calculate impact score
        impact_score = 1.0 - (high_impact_count * 0.3 + medium_impact_count * 0.15)
        return max(0.2, impact_score)  # Minimum 20% effectiveness

    def _generate_environmental_recommendations(self, conditions: EnvironmentalCondition,
                                              soil_impact: float, weather_impact: float,
                                              site_impact: float) -> List[str]:
        """Generate environmental-based recommendations."""
        recommendations = []

        # Soil-based recommendations
        if soil_impact < 0.5:
            if "clay" in conditions.soil_type.lower():
                recommendations.append("Consider higher frequency GPR antennas for clay soils")
                recommendations.append("Increase survey line density to compensate for reduced GPR penetration")
                recommendations.append("Consider electromagnetic methods as primary detection method")

        # Weather-based recommendations
        if weather_impact < 0.7:
            recommendations.append("Consider rescheduling survey for better weather conditions")
            recommendations.append("Use weather-resistant equipment and protective measures")

        # Site constraint recommendations
        if site_impact < 0.7:
            recommendations.append("Develop site-specific safety and access protocols")
            recommendations.append("Consider alternative survey methods for constrained areas")

        return recommendations

    def validate_method_requirements(self, survey_data: SurveyData,
                                   target_quality_level: QualityLevel) -> Dict[SurveyMethod, bool]:
        """
        Validate that required methods for target quality level were executed.

        Args:
            survey_data: Survey data to validate
            target_quality_level: Target quality level to validate against

        Returns:
            Dictionary mapping methods to compliance status
        """
        self.logger.info(f"Validating method requirements for {target_quality_level}")

        requirements = self.quality_level_requirements[target_quality_level]
        executed_methods = {method.method for method in survey_data.methods_executed}

        compliance_status = {}
        for required_method in requirements.required_methods:
            compliance_status[required_method] = required_method in executed_methods

        return compliance_status

    def validate_deliverables_requirements(self, survey_data: SurveyData,
                                         target_quality_level: QualityLevel) -> Dict[DeliverableType, bool]:
        """
        Validate that required deliverables for target quality level were provided.

        Args:
            survey_data: Survey data to validate
            target_quality_level: Target quality level to validate against

        Returns:
            Dictionary mapping deliverables to compliance status
        """
        self.logger.info(f"Validating deliverable requirements for {target_quality_level}")

        requirements = self.quality_level_requirements[target_quality_level]
        provided_deliverables = {
            deliverable.deliverable_type for deliverable in survey_data.deliverables
            if deliverable.provided
        }

        compliance_status = {}
        for required_deliverable in requirements.required_deliverables:
            compliance_status[required_deliverable] = required_deliverable in provided_deliverables

        return compliance_status

    def assess_accuracy_compliance(self, survey_data: SurveyData,
                                 target_quality_level: QualityLevel) -> Dict[str, bool]:
        """
        Assess accuracy compliance for target quality level.

        Args:
            survey_data: Survey data to assess
            target_quality_level: Target quality level to assess against

        Returns:
            Dictionary mapping accuracy types to compliance status
        """
        self.logger.info(f"Assessing accuracy compliance for {target_quality_level}")

        requirements = self.quality_level_requirements[target_quality_level]
        accuracy_compliance = {}

        if not survey_data.utility_detections:
            # No detections to assess
            return {key: False for key in requirements.accuracy_requirements.keys()}

        # Calculate average accuracy from detections
        horizontal_accuracies = [
            detection.accuracy.horizontal_accuracy
            for detection in survey_data.utility_detections
        ]
        vertical_accuracies = [
            detection.accuracy.vertical_accuracy
            for detection in survey_data.utility_detections
            if detection.accuracy.vertical_accuracy is not None
        ]

        avg_horizontal_accuracy = np.mean(horizontal_accuracies) if horizontal_accuracies else float('inf')
        avg_vertical_accuracy = np.mean(vertical_accuracies) if vertical_accuracies else float('inf')

        # Check horizontal accuracy compliance
        if "horizontal" in requirements.accuracy_requirements:
            required_horizontal = requirements.accuracy_requirements["horizontal"]
            accuracy_compliance["horizontal"] = avg_horizontal_accuracy <= required_horizontal

        # Check vertical accuracy compliance
        if "vertical" in requirements.accuracy_requirements:
            required_vertical = requirements.accuracy_requirements["vertical"]
            accuracy_compliance["vertical"] = avg_vertical_accuracy <= required_vertical

        return accuracy_compliance

    def determine_achievable_quality_level(self, survey_data: SurveyData,
                                         consider_environmental: bool = True) -> QualityLevelAssessment:
        """
        Determine the highest achievable quality level based on survey data.

        Args:
            survey_data: Survey data to assess
            consider_environmental: Whether to consider environmental factors

        Returns:
            Quality level assessment with recommendations
        """
        self.logger.info("Determining achievable quality level")

        # Assess environmental impact if requested
        environmental_impact = None
        if consider_environmental:
            environmental_impact = self.assess_environmental_impact(survey_data.environmental_conditions)

        # Check compliance for each quality level (from highest to lowest)
        quality_levels = [QualityLevel.QL_A, QualityLevel.QL_B, QualityLevel.QL_C, QualityLevel.QL_D]

        achieved_level = QualityLevel.QL_D  # Start with lowest level
        limiting_factors = []
        recommendations = []

        for ql in reversed(quality_levels):  # Check from lowest to highest
            methods_compliance = self.validate_method_requirements(survey_data, ql)
            deliverables_compliance = self.validate_deliverables_requirements(survey_data, ql)
            accuracy_compliance = self.assess_accuracy_compliance(survey_data, ql)

            # Check if all requirements are met
            methods_passed = all(methods_compliance.values())
            deliverables_passed = all(deliverables_compliance.values())
            accuracy_passed = all(accuracy_compliance.values())

            # Consider environmental factors
            environmental_suitable = True
            if consider_environmental and environmental_impact:
                # Lower quality level requirements if environmental conditions are poor
                env_threshold = 0.6
                if environmental_impact.overall_environmental_score < env_threshold:
                    environmental_suitable = ql in [QualityLevel.QL_D, QualityLevel.QL_C]

            if methods_passed and deliverables_passed and accuracy_passed and environmental_suitable:
                achieved_level = ql
            else:
                # Identify limiting factors for next level up
                if not methods_passed:
                    missing_methods = [method for method, passed in methods_compliance.items() if not passed]
                    limiting_factors.extend([f"Missing method: {method}" for method in missing_methods])

                if not deliverables_passed:
                    missing_deliverables = [deliv for deliv, passed in deliverables_compliance.items() if not passed]
                    limiting_factors.extend([f"Missing deliverable: {deliv}" for deliv in missing_deliverables])

                if not accuracy_passed:
                    poor_accuracy = [acc_type for acc_type, passed in accuracy_compliance.items() if not passed]
                    limiting_factors.extend([f"Insufficient accuracy: {acc_type}" for acc_type in poor_accuracy])

                if not environmental_suitable:
                    limiting_factors.append("Environmental conditions unsuitable for higher quality levels")

        # Generate recommendations for achieving higher quality levels
        if achieved_level != QualityLevel.QL_A:
            next_level_index = quality_levels.index(achieved_level) - 1
            if next_level_index >= 0:
                next_level = quality_levels[next_level_index]
                recommendations = self._generate_improvement_recommendations(
                    survey_data, next_level, environmental_impact
                )

        # Calculate confidence based on compliance scores
        confidence = self._calculate_assessment_confidence(
            survey_data, achieved_level, environmental_impact
        )

        return QualityLevelAssessment(
            assessed_quality_level=achieved_level,
            confidence=confidence,
            methods_compliance=self.validate_method_requirements(survey_data, achieved_level),
            deliverables_compliance=self.validate_deliverables_requirements(survey_data, achieved_level),
            accuracy_compliance=self.assess_accuracy_compliance(survey_data, achieved_level),
            limiting_factors=limiting_factors,
            recommendations=recommendations
        )

    def _generate_improvement_recommendations(self, survey_data: SurveyData,
                                            target_level: QualityLevel,
                                            environmental_impact: Optional[EnvironmentalImpactAssessment]) -> List[str]:
        """Generate recommendations for achieving target quality level."""
        recommendations = []

        # Method-based recommendations
        required_methods = self.quality_level_requirements[target_level].required_methods
        executed_methods = {method.method for method in survey_data.methods_executed}

        missing_methods = set(required_methods) - executed_methods
        for method in missing_methods:
            if method == SurveyMethod.GROUND_PENETRATING_RADAR:
                recommendations.append("Deploy GPR systems with appropriate antenna frequencies for site conditions")
            elif method == SurveyMethod.ELECTROMAGNETIC_DETECTION:
                recommendations.append("Conduct electromagnetic detection using cable avoidance tools and precision locators")
            elif method == SurveyMethod.TRIAL_HOLES:
                recommendations.append("Perform intrusive investigation using trial holes at key utility crossings")
            elif method == SurveyMethod.VACUUM_EXCAVATION:
                recommendations.append("Use vacuum excavation for safe utility exposure and verification")
            else:
                recommendations.append(f"Implement {method.value.replace('_', ' ')} methodology")

        # Deliverable-based recommendations
        required_deliverables = self.quality_level_requirements[target_level].required_deliverables
        provided_deliverables = {
            deliv.deliverable_type for deliv in survey_data.deliverables if deliv.provided
        }

        missing_deliverables = set(required_deliverables) - provided_deliverables
        for deliverable in missing_deliverables:
            recommendations.append(f"Provide {deliverable.value.replace('_', ' ')}")

        # Environmental recommendations
        if environmental_impact and environmental_impact.overall_environmental_score < 0.7:
            recommendations.extend(environmental_impact.recommended_adjustments)

        return recommendations

    def _calculate_assessment_confidence(self, survey_data: SurveyData,
                                       assessed_level: QualityLevel,
                                       environmental_impact: Optional[EnvironmentalImpactAssessment]) -> float:
        """Calculate confidence in quality level assessment."""
        confidence_factors = []

        # Data completeness factor
        if survey_data.utility_detections:
            avg_detection_confidence = np.mean([det.confidence for det in survey_data.utility_detections])
            confidence_factors.append(avg_detection_confidence)
        else:
            confidence_factors.append(0.5)  # Lower confidence without detections

        # Method execution quality factor
        if survey_data.methods_executed:
            avg_execution_quality = np.mean([method.execution_quality for method in survey_data.methods_executed])
            confidence_factors.append(avg_execution_quality)
        else:
            confidence_factors.append(0.3)  # Very low confidence without method execution data

        # Environmental factor
        if environmental_impact:
            confidence_factors.append(environmental_impact.overall_environmental_score)
        else:
            confidence_factors.append(0.8)  # Assume reasonable conditions

        # Deliverable quality factor
        provided_deliverables = [deliv for deliv in survey_data.deliverables if deliv.provided]
        if provided_deliverables:
            quality_scores = [deliv.quality_score for deliv in provided_deliverables if deliv.quality_score is not None]
            if quality_scores:
                avg_deliverable_quality = np.mean(quality_scores)
                confidence_factors.append(avg_deliverable_quality)
            else:
                confidence_factors.append(0.7)  # Moderate confidence if no quality scores
        else:
            confidence_factors.append(0.4)  # Low confidence without deliverables

        return np.mean(confidence_factors)

    def perform_comprehensive_compliance_check(self, survey_data: SurveyData) -> ComplianceReport:
        """
        Perform comprehensive PAS 128 compliance check and generate detailed report.

        Args:
            survey_data: Survey data to check for compliance

        Returns:
            Comprehensive compliance report
        """
        self.logger.info(f"Performing comprehensive compliance check for survey {survey_data.survey_id}")

        start_time = datetime.now()

        # Assess environmental impact
        environmental_impact = self.assess_environmental_impact(survey_data.environmental_conditions)

        # Determine achievable quality level
        quality_level_assessment = self.determine_achievable_quality_level(survey_data, consider_environmental=True)

        # Perform individual compliance checks
        compliance_checks = self._perform_individual_compliance_checks(survey_data, environmental_impact)

        # Analyze methods and deliverables
        methods_analysis = self._analyze_methods_execution(survey_data)
        deliverables_analysis = self._analyze_deliverables_quality(survey_data)
        accuracy_analysis = self._analyze_accuracy_performance(survey_data)

        # Calculate overall compliance score
        overall_score = self._calculate_overall_compliance_score(
            quality_level_assessment, compliance_checks, environmental_impact
        )

        # Identify critical gaps and recommendations
        critical_gaps, recommendations, next_steps = self._identify_gaps_and_recommendations(
            survey_data, quality_level_assessment, compliance_checks
        )

        # Generate compliance summary
        compliance_summary = self._generate_compliance_summary(
            survey_data, quality_level_assessment, overall_score
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Compliance check completed in {processing_time:.2f} seconds")

        return ComplianceReport(
            survey_id=survey_data.survey_id,
            assessment_date=datetime.now(),
            target_quality_level=survey_data.target_quality_level,
            achieved_quality_level=quality_level_assessment.assessed_quality_level,
            overall_compliance_score=overall_score,
            quality_level_assessment=quality_level_assessment,
            environmental_impact=environmental_impact,
            compliance_checks=compliance_checks,
            methods_analysis=methods_analysis,
            deliverables_analysis=deliverables_analysis,
            accuracy_analysis=accuracy_analysis,
            critical_gaps=critical_gaps,
            improvement_recommendations=recommendations,
            next_steps=next_steps,
            compliance_summary=compliance_summary
        )

    def _perform_individual_compliance_checks(self, survey_data: SurveyData,
                                            environmental_impact: EnvironmentalImpactAssessment) -> List[ComplianceCheck]:
        """Perform individual compliance checks."""
        checks = []

        # Method compliance check
        target_level = survey_data.target_quality_level
        methods_compliance = self.validate_method_requirements(survey_data, target_level)
        methods_passed = sum(methods_compliance.values())
        methods_total = len(methods_compliance)
        methods_score = methods_passed / methods_total if methods_total > 0 else 0

        checks.append(ComplianceCheck(
            check_name="Method Requirements",
            passed=methods_score == 1.0,
            score=methods_score,
            details=f"Passed {methods_passed}/{methods_total} required methods",
            requirements=[method.value for method in methods_compliance.keys()],
            gaps=[method.value for method, passed in methods_compliance.items() if not passed]
        ))

        # Deliverable compliance check
        deliverables_compliance = self.validate_deliverables_requirements(survey_data, target_level)
        deliverables_passed = sum(deliverables_compliance.values())
        deliverables_total = len(deliverables_compliance)
        deliverables_score = deliverables_passed / deliverables_total if deliverables_total > 0 else 0

        checks.append(ComplianceCheck(
            check_name="Deliverable Requirements",
            passed=deliverables_score == 1.0,
            score=deliverables_score,
            details=f"Provided {deliverables_passed}/{deliverables_total} required deliverables",
            requirements=[deliv.value for deliv in deliverables_compliance.keys()],
            gaps=[deliv.value for deliv, passed in deliverables_compliance.items() if not passed]
        ))

        # Accuracy compliance check
        accuracy_compliance = self.assess_accuracy_compliance(survey_data, target_level)
        accuracy_passed = sum(accuracy_compliance.values())
        accuracy_total = len(accuracy_compliance)
        accuracy_score = accuracy_passed / accuracy_total if accuracy_total > 0 else 0

        checks.append(ComplianceCheck(
            check_name="Accuracy Requirements",
            passed=accuracy_score == 1.0,
            score=accuracy_score,
            details=f"Met {accuracy_passed}/{accuracy_total} accuracy requirements",
            requirements=list(accuracy_compliance.keys()),
            gaps=[acc_type for acc_type, passed in accuracy_compliance.items() if not passed]
        ))

        # Environmental suitability check
        env_suitable = environmental_impact.overall_environmental_score >= 0.6
        checks.append(ComplianceCheck(
            check_name="Environmental Suitability",
            passed=env_suitable,
            score=environmental_impact.overall_environmental_score,
            details=f"Environmental score: {environmental_impact.overall_environmental_score:.2f}",
            requirements=["Suitable environmental conditions for survey methods"],
            gaps=["Environmental conditions may impact survey quality"] if not env_suitable else []
        ))

        return checks

    def _analyze_methods_execution(self, survey_data: SurveyData) -> Dict[SurveyMethod, Dict[str, Any]]:
        """Analyze the execution quality of survey methods."""
        analysis = {}

        for method_execution in survey_data.methods_executed:
            method = method_execution.method
            analysis[method] = {
                "executed": True,
                "execution_quality": method_execution.execution_quality,
                "coverage_area": method_execution.coverage_area,
                "equipment_count": len(method_execution.equipment_used),
                "limitations": method_execution.limitations_encountered,
                "execution_date": method_execution.execution_date.isoformat(),
                "summary": method_execution.results_summary
            }

        return analysis

    def _analyze_deliverables_quality(self, survey_data: SurveyData) -> Dict[DeliverableType, Dict[str, Any]]:
        """Analyze the quality of provided deliverables."""
        analysis = {}

        for deliverable in survey_data.deliverables:
            deliv_type = deliverable.deliverable_type
            analysis[deliv_type] = {
                "provided": deliverable.provided,
                "quality_score": deliverable.quality_score,
                "completeness_score": deliverable.completeness_score,
                "file_available": deliverable.file_path is not None,
                "notes": deliverable.notes
            }

        return analysis

    def _analyze_accuracy_performance(self, survey_data: SurveyData) -> Dict[str, Any]:
        """Analyze accuracy performance of utility detections."""
        if not survey_data.utility_detections:
            return {"no_detections": True}

        horizontal_accuracies = [det.accuracy.horizontal_accuracy for det in survey_data.utility_detections]
        vertical_accuracies = [
            det.accuracy.vertical_accuracy for det in survey_data.utility_detections
            if det.accuracy.vertical_accuracy is not None
        ]

        analysis = {
            "total_detections": len(survey_data.utility_detections),
            "horizontal_accuracy": {
                "mean": float(np.mean(horizontal_accuracies)),
                "std": float(np.std(horizontal_accuracies)),
                "min": float(np.min(horizontal_accuracies)),
                "max": float(np.max(horizontal_accuracies))
            },
            "average_confidence": float(np.mean([det.confidence for det in survey_data.utility_detections])),
            "verified_detections": sum(1 for det in survey_data.utility_detections if det.verified),
            "utility_types": list(set(det.utility_type for det in survey_data.utility_detections))
        }

        if vertical_accuracies:
            analysis["vertical_accuracy"] = {
                "mean": float(np.mean(vertical_accuracies)),
                "std": float(np.std(vertical_accuracies)),
                "min": float(np.min(vertical_accuracies)),
                "max": float(np.max(vertical_accuracies))
            }

        return analysis

    def _calculate_overall_compliance_score(self, quality_assessment: QualityLevelAssessment,
                                          compliance_checks: List[ComplianceCheck],
                                          environmental_impact: EnvironmentalImpactAssessment) -> float:
        """Calculate overall compliance score."""
        # Weight the different compliance aspects
        weights = {
            "quality_level": 0.4,
            "compliance_checks": 0.4,
            "environmental": 0.2
        }

        # Quality level score (based on achievement vs. target)
        quality_levels = [QualityLevel.QL_D, QualityLevel.QL_C, QualityLevel.QL_B, QualityLevel.QL_A]
        achieved_index = quality_levels.index(quality_assessment.assessed_quality_level)
        quality_score = (achieved_index + 1) / len(quality_levels)

        # Compliance checks average score
        checks_score = np.mean([check.score for check in compliance_checks])

        # Environmental score
        env_score = environmental_impact.overall_environmental_score

        # Calculate weighted average
        overall_score = (
            weights["quality_level"] * quality_score +
            weights["compliance_checks"] * checks_score +
            weights["environmental"] * env_score
        )

        return overall_score

    def _identify_gaps_and_recommendations(self, survey_data: SurveyData,
                                         quality_assessment: QualityLevelAssessment,
                                         compliance_checks: List[ComplianceCheck]) -> Tuple[List[str], List[str], List[str]]:
        """Identify critical gaps and generate recommendations."""
        critical_gaps = []
        recommendations = []
        next_steps = []

        # Collect gaps from compliance checks
        for check in compliance_checks:
            if not check.passed:
                critical_gaps.extend(check.gaps)

        # Add limiting factors from quality assessment
        critical_gaps.extend(quality_assessment.limiting_factors)

        # Generate recommendations
        recommendations.extend(quality_assessment.recommendations)

        # Add specific recommendations based on gaps
        if any("method" in gap.lower() for gap in critical_gaps):
            recommendations.append("Review and implement missing survey methods according to PAS 128 requirements")

        if any("deliverable" in gap.lower() for gap in critical_gaps):
            recommendations.append("Complete missing deliverables to meet quality level requirements")

        if any("accuracy" in gap.lower() for gap in critical_gaps):
            recommendations.append("Improve survey accuracy through better equipment calibration and methodology")

        # Define next steps
        if quality_assessment.assessed_quality_level != survey_data.target_quality_level:
            next_steps.append(f"Implement missing requirements to achieve {survey_data.target_quality_level}")

        next_steps.append("Review compliance report and prioritize improvement actions")
        next_steps.append("Schedule follow-up assessment after implementing recommendations")

        return critical_gaps, recommendations, next_steps

    def _generate_compliance_summary(self, survey_data: SurveyData,
                                   quality_assessment: QualityLevelAssessment,
                                   overall_score: float) -> Dict[str, Any]:
        """Generate compliance summary statistics."""
        return {
            "survey_id": survey_data.survey_id,
            "target_vs_achieved": {
                "target": survey_data.target_quality_level,
                "achieved": quality_assessment.assessed_quality_level,
                "gap": survey_data.target_quality_level != quality_assessment.assessed_quality_level
            },
            "overall_score": overall_score,
            "score_category": self._categorize_compliance_score(overall_score),
            "confidence": quality_assessment.confidence,
            "total_detections": len(survey_data.utility_detections),
            "methods_executed": len(survey_data.methods_executed),
            "deliverables_provided": sum(1 for d in survey_data.deliverables if d.provided),
            "environmental_score": survey_data.environmental_conditions.soil_type
        }

    def _categorize_compliance_score(self, score: float) -> str:
        """Categorize compliance score into descriptive categories."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Satisfactory"
        elif score >= 0.6:
            return "Needs Improvement"
        else:
            return "Poor"