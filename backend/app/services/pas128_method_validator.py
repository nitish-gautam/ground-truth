"""
PAS 128 Method Requirement Validation Framework

This service provides comprehensive validation of survey methods
against PAS 128 requirements, including method effectiveness assessment,
equipment validation, and execution quality evaluation.
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from dataclasses import dataclass

from ..schemas.pas128 import (
    QualityLevel, SurveyMethod, DetectionMethod, DeliverableType,
    SurveyData, MethodExecution, DetectionEquipment, EnvironmentalCondition,
    UtilityDetection, AccuracyMeasurement, ComplianceCheck
)

logger = logging.getLogger(__name__)


class MethodValidationResult(Enum):
    """Method validation result types"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class MethodValidationDetails:
    """Detailed validation results for a method"""
    method: SurveyMethod
    validation_result: MethodValidationResult
    compliance_score: float  # 0.0 to 1.0
    execution_quality: float  # 0.0 to 1.0
    equipment_adequacy: float  # 0.0 to 1.0
    coverage_adequacy: float  # 0.0 to 1.0
    environmental_suitability: float  # 0.0 to 1.0
    issues_identified: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any]


@dataclass
class EquipmentValidation:
    """Equipment validation details"""
    equipment: DetectionEquipment
    is_suitable: bool
    calibration_valid: bool
    operator_qualified: bool
    frequency_appropriate: bool
    condition_assessment: str
    limitations: List[str]


class PAS128MethodValidator:
    """
    Comprehensive method requirement validation framework for PAS 128 compliance.

    This service validates:
    - Method execution against PAS 128 requirements
    - Equipment suitability and calibration
    - Operator qualifications and certifications
    - Method effectiveness in given environmental conditions
    - Coverage adequacy and execution quality
    """

    def __init__(self):
        """Initialize the method validation framework."""
        self.logger = logging.getLogger(__name__)

        # Initialize method requirements
        self.method_requirements = self._initialize_method_requirements()

        # Initialize equipment specifications
        self.equipment_specs = self._initialize_equipment_specifications()

        # Initialize environmental limitations
        self.environmental_limitations = self._initialize_environmental_limitations()

        # Initialize validation thresholds
        self.validation_thresholds = self._initialize_validation_thresholds()

        self.logger.info("PAS 128 Method Validator initialized")

    def _initialize_method_requirements(self) -> Dict[QualityLevel, Dict[SurveyMethod, Dict[str, Any]]]:
        """Initialize method requirements for each quality level."""
        return {
            QualityLevel.QL_D: {
                SurveyMethod.RECORDS_SEARCH: {
                    "required": True,
                    "minimum_scope": "basic_utility_records",
                    "coverage_requirement": 0.8,
                    "execution_standards": ["desk_study_methodology"]
                },
                SurveyMethod.SITE_RECONNAISSANCE: {
                    "required": True,
                    "minimum_scope": "visual_inspection",
                    "coverage_requirement": 0.9,
                    "execution_standards": ["visual_survey_protocol"]
                }
            },
            QualityLevel.QL_C: {
                SurveyMethod.COMPREHENSIVE_RECORDS: {
                    "required": True,
                    "minimum_scope": "all_available_records",
                    "coverage_requirement": 0.95,
                    "execution_standards": ["comprehensive_search_protocol"]
                },
                SurveyMethod.SITE_RECONNAISSANCE: {
                    "required": True,
                    "minimum_scope": "detailed_inspection",
                    "coverage_requirement": 0.95,
                    "execution_standards": ["detailed_survey_protocol"]
                },
                SurveyMethod.TOPOGRAPHICAL_SURVEY: {
                    "required": True,
                    "minimum_scope": "site_boundaries_features",
                    "coverage_requirement": 1.0,
                    "execution_standards": ["topographical_survey_standards"]
                }
            },
            QualityLevel.QL_B: {
                SurveyMethod.ELECTROMAGNETIC_DETECTION: {
                    "required": True,
                    "minimum_scope": "comprehensive_em_survey",
                    "coverage_requirement": 0.9,
                    "execution_standards": ["electromagnetic_detection_protocol"],
                    "equipment_requirements": ["cable_avoidance_tools", "precision_locators"]
                },
                SurveyMethod.GROUND_PENETRATING_RADAR: {
                    "required": True,
                    "minimum_scope": "systematic_gpr_survey",
                    "coverage_requirement": 0.85,
                    "execution_standards": ["gpr_survey_protocol"],
                    "equipment_requirements": ["multi_frequency_gpr"]
                }
            },
            QualityLevel.QL_A: {
                SurveyMethod.TRIAL_HOLES: {
                    "required": True,
                    "minimum_scope": "verification_excavation",
                    "coverage_requirement": 0.2,  # Selective intrusive investigation
                    "execution_standards": ["safe_excavation_protocol"],
                    "safety_requirements": ["permit_to_dig", "safe_digging_practices"]
                },
                SurveyMethod.VACUUM_EXCAVATION: {
                    "required": True,
                    "minimum_scope": "utility_exposure",
                    "coverage_requirement": 0.15,
                    "execution_standards": ["vacuum_excavation_protocol"],
                    "equipment_requirements": ["vacuum_excavation_unit"]
                }
            }
        }

    def _initialize_equipment_specifications(self) -> Dict[DetectionMethod, Dict[str, Any]]:
        """Initialize equipment specifications and requirements."""
        return {
            DetectionMethod.ELECTROMAGNETIC: {
                "required_equipment": ["cable_avoidance_tool", "precision_locator"],
                "frequency_ranges": {
                    "cable_avoidance": "8kHz-33kHz",
                    "precision_location": "512Hz-33kHz"
                },
                "calibration_interval": 365,  # days
                "operator_certification": "electromagnetic_detection_certified",
                "environmental_limitations": ["wet_conditions", "electromagnetic_interference"]
            },
            DetectionMethod.GROUND_PENETRATING_RADAR: {
                "required_equipment": ["gpr_system"],
                "frequency_ranges": {
                    "shallow_utilities": "1000MHz-2000MHz",
                    "medium_depth": "400MHz-900MHz",
                    "deep_utilities": "100MHz-400MHz"
                },
                "calibration_interval": 180,  # days
                "operator_certification": "gpr_operator_certified",
                "environmental_limitations": ["clay_soils", "high_conductivity", "frozen_ground"]
            },
            DetectionMethod.RADIO_DETECTION: {
                "required_equipment": ["signal_generator", "radio_receiver"],
                "frequency_ranges": {
                    "standard_detection": "8kHz-33kHz",
                    "long_range": "512Hz-8kHz"
                },
                "calibration_interval": 365,  # days
                "operator_certification": "radio_detection_certified",
                "environmental_limitations": ["signal_attenuation", "interference"]
            }
        }

    def _initialize_environmental_limitations(self) -> Dict[str, Dict[str, float]]:
        """Initialize environmental limitations for different methods."""
        return {
            "soil_limitations": {
                "clay_gpr_impact": 0.3,
                "wet_clay_gpr_impact": 0.2,
                "sand_gpr_impact": 0.9,
                "gravel_electromagnetic_impact": 0.8,
                "organic_soil_impact": 0.4
            },
            "weather_limitations": {
                "heavy_rain_impact": 0.3,
                "frozen_ground_impact": 0.4,
                "high_temperature_impact": 0.8,
                "humidity_impact": 0.7
            },
            "site_limitations": {
                "electromagnetic_interference": 0.5,
                "traffic_impact": 0.6,
                "restricted_access": 0.4,
                "underground_services": 0.7
            }
        }

    def _initialize_validation_thresholds(self) -> Dict[str, float]:
        """Initialize validation thresholds for compliance assessment."""
        return {
            "minimum_compliance_score": 0.8,
            "minimum_execution_quality": 0.7,
            "minimum_equipment_adequacy": 0.8,
            "minimum_coverage": 0.8,
            "minimum_environmental_suitability": 0.6,
            "calibration_validity_threshold": 30,  # days before expiry
            "operator_certification_validity": 1095  # days (3 years)
        }

    def validate_method_execution(self, method_execution: MethodExecution,
                                quality_level: QualityLevel,
                                environmental_conditions: EnvironmentalCondition) -> MethodValidationDetails:
        """
        Validate a single method execution against PAS 128 requirements.

        Args:
            method_execution: Method execution details to validate
            quality_level: Target quality level for validation
            environmental_conditions: Environmental conditions during execution

        Returns:
            Detailed validation results
        """
        self.logger.debug(f"Validating method execution: {method_execution.method}")

        method = method_execution.method
        issues = []
        recommendations = []
        evidence = {}

        # Check if method is required for quality level
        if not self._is_method_required(method, quality_level):
            return MethodValidationDetails(
                method=method,
                validation_result=MethodValidationResult.NOT_APPLICABLE,
                compliance_score=1.0,
                execution_quality=method_execution.execution_quality,
                equipment_adequacy=1.0,
                coverage_adequacy=1.0,
                environmental_suitability=1.0,
                issues_identified=[],
                recommendations=[],
                evidence={"not_required": True}
            )

        # Validate equipment adequacy
        equipment_score, equipment_issues = self._validate_equipment_adequacy(
            method_execution.equipment_used, method, environmental_conditions
        )
        issues.extend(equipment_issues)
        evidence["equipment_validation"] = equipment_score

        # Validate coverage adequacy
        coverage_score, coverage_issues = self._validate_coverage_adequacy(
            method_execution, quality_level
        )
        issues.extend(coverage_issues)
        evidence["coverage_validation"] = coverage_score

        # Validate execution quality
        execution_score = method_execution.execution_quality
        if execution_score < self.validation_thresholds["minimum_execution_quality"]:
            issues.append(f"Execution quality below threshold: {execution_score:.2f}")

        # Validate environmental suitability
        env_score, env_issues = self._validate_environmental_suitability(
            method, environmental_conditions
        )
        issues.extend(env_issues)
        evidence["environmental_validation"] = env_score

        # Validate execution standards compliance
        standards_score, standards_issues = self._validate_execution_standards(
            method_execution, quality_level
        )
        issues.extend(standards_issues)
        evidence["standards_validation"] = standards_score

        # Calculate overall compliance score
        compliance_score = np.mean([
            equipment_score,
            coverage_score,
            execution_score,
            env_score,
            standards_score
        ])

        # Determine validation result
        if compliance_score >= self.validation_thresholds["minimum_compliance_score"]:
            validation_result = MethodValidationResult.COMPLIANT
        elif compliance_score >= 0.6:
            validation_result = MethodValidationResult.PARTIALLY_COMPLIANT
        else:
            validation_result = MethodValidationResult.NON_COMPLIANT

        # Generate recommendations
        recommendations = self._generate_method_recommendations(
            method, issues, compliance_score, environmental_conditions
        )

        return MethodValidationDetails(
            method=method,
            validation_result=validation_result,
            compliance_score=compliance_score,
            execution_quality=execution_score,
            equipment_adequacy=equipment_score,
            coverage_adequacy=coverage_score,
            environmental_suitability=env_score,
            issues_identified=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _is_method_required(self, method: SurveyMethod, quality_level: QualityLevel) -> bool:
        """Check if method is required for the specified quality level."""
        if quality_level not in self.method_requirements:
            return False

        return method in self.method_requirements[quality_level]

    def _validate_equipment_adequacy(self, equipment_list: List[DetectionEquipment],
                                   method: SurveyMethod,
                                   environmental_conditions: EnvironmentalCondition) -> Tuple[float, List[str]]:
        """Validate equipment adequacy for the method."""
        issues = []
        equipment_scores = []

        if not equipment_list:
            issues.append("No equipment specified for method execution")
            return 0.0, issues

        for equipment in equipment_list:
            equipment_validation = self._validate_individual_equipment(
                equipment, environmental_conditions
            )

            score = 0.0
            if equipment_validation.is_suitable:
                score += 0.3
            if equipment_validation.calibration_valid:
                score += 0.3
            if equipment_validation.operator_qualified:
                score += 0.2
            if equipment_validation.frequency_appropriate:
                score += 0.2

            equipment_scores.append(score)

            # Collect issues
            if not equipment_validation.is_suitable:
                issues.append(f"Equipment {equipment.model} not suitable for method {method}")
            if not equipment_validation.calibration_valid:
                issues.append(f"Equipment {equipment.model} calibration expired or invalid")
            if not equipment_validation.operator_qualified:
                issues.append(f"Operator not qualified for equipment {equipment.model}")

        overall_score = np.mean(equipment_scores) if equipment_scores else 0.0
        return overall_score, issues

    def _validate_individual_equipment(self, equipment: DetectionEquipment,
                                     environmental_conditions: EnvironmentalCondition) -> EquipmentValidation:
        """Validate individual equipment piece."""
        detection_method = equipment.equipment_type

        # Check equipment specifications
        specs = self.equipment_specs.get(detection_method, {})

        # Validate calibration
        calibration_valid = True
        if equipment.calibration_date:
            days_since_calibration = (datetime.now() - equipment.calibration_date).days
            max_interval = specs.get("calibration_interval", 365)
            calibration_valid = days_since_calibration <= max_interval

        # Validate operator qualifications
        operator_qualified = bool(equipment.operator_certification)

        # Validate frequency appropriateness
        frequency_appropriate = True
        if equipment.frequency_range and "frequency_ranges" in specs:
            # This would need more sophisticated frequency range validation
            frequency_appropriate = True

        # Check equipment suitability for conditions
        is_suitable = self._assess_equipment_environmental_suitability(
            equipment, environmental_conditions
        )

        limitations = []
        if detection_method in self.equipment_specs:
            env_limitations = self.equipment_specs[detection_method].get("environmental_limitations", [])
            for limitation in env_limitations:
                if self._check_environmental_limitation(limitation, environmental_conditions):
                    limitations.append(limitation)

        return EquipmentValidation(
            equipment=equipment,
            is_suitable=is_suitable,
            calibration_valid=calibration_valid,
            operator_qualified=operator_qualified,
            frequency_appropriate=frequency_appropriate,
            condition_assessment="suitable" if is_suitable else "limited",
            limitations=limitations
        )

    def _assess_equipment_environmental_suitability(self, equipment: DetectionEquipment,
                                                  environmental_conditions: EnvironmentalCondition) -> bool:
        """Assess equipment suitability for environmental conditions."""
        detection_method = equipment.equipment_type

        # GPR suitability assessment
        if detection_method == DetectionMethod.GROUND_PENETRATING_RADAR:
            soil_type = environmental_conditions.soil_type.lower()
            if "clay" in soil_type:
                return False  # Poor performance in clay
            if environmental_conditions.moisture_content and environmental_conditions.moisture_content > 40:
                return False  # Poor performance in high moisture

        # Electromagnetic suitability assessment
        elif detection_method == DetectionMethod.ELECTROMAGNETIC:
            if "interference" in str(environmental_conditions.site_constraints).lower():
                return False
            if environmental_conditions.weather_conditions and "rain" in environmental_conditions.weather_conditions.lower():
                return False

        return True

    def _check_environmental_limitation(self, limitation: str,
                                      environmental_conditions: EnvironmentalCondition) -> bool:
        """Check if environmental limitation applies to current conditions."""
        limitation_lower = limitation.lower()

        if "clay" in limitation_lower and "clay" in environmental_conditions.soil_type.lower():
            return True
        if "wet" in limitation_lower and environmental_conditions.weather_conditions:
            if "rain" in environmental_conditions.weather_conditions.lower():
                return True
        if "interference" in limitation_lower:
            if any("interference" in constraint.lower() for constraint in environmental_conditions.site_constraints):
                return True

        return False

    def _validate_coverage_adequacy(self, method_execution: MethodExecution,
                                  quality_level: QualityLevel) -> Tuple[float, List[str]]:
        """Validate coverage adequacy for the method."""
        issues = []

        method = method_execution.method
        coverage_area = method_execution.coverage_area

        if coverage_area is None:
            issues.append("Coverage area not specified")
            return 0.5, issues

        # Get required coverage for quality level and method
        method_reqs = self.method_requirements.get(quality_level, {}).get(method, {})
        required_coverage = method_reqs.get("coverage_requirement", 0.8)

        # For this validation, we assume total survey area is 1000 m² (this should come from survey data)
        assumed_survey_area = 1000.0
        coverage_ratio = coverage_area / assumed_survey_area

        if coverage_ratio < required_coverage:
            issues.append(f"Coverage inadequate: {coverage_ratio:.2f} < {required_coverage:.2f}")
            score = coverage_ratio / required_coverage
        else:
            score = 1.0

        return min(1.0, score), issues

    def _validate_environmental_suitability(self, method: SurveyMethod,
                                          environmental_conditions: EnvironmentalCondition) -> Tuple[float, List[str]]:
        """Validate environmental suitability for the method."""
        issues = []
        suitability_factors = []

        # Soil impact assessment
        if method == SurveyMethod.GROUND_PENETRATING_RADAR:
            soil_impact = self._assess_soil_impact_on_gpr(environmental_conditions.soil_type)
            suitability_factors.append(soil_impact)

            if soil_impact < 0.5:
                issues.append(f"Soil type '{environmental_conditions.soil_type}' limits GPR effectiveness")

        # Weather impact assessment
        weather_impact = self._assess_weather_impact(environmental_conditions.weather_conditions)
        suitability_factors.append(weather_impact)

        if weather_impact < 0.6:
            issues.append("Weather conditions limit method effectiveness")

        # Site constraints impact
        site_impact = self._assess_site_constraints_impact(environmental_conditions.site_constraints)
        suitability_factors.append(site_impact)

        if site_impact < 0.6:
            issues.append("Site constraints limit method execution")

        overall_suitability = np.mean(suitability_factors) if suitability_factors else 0.8

        return overall_suitability, issues

    def _assess_soil_impact_on_gpr(self, soil_type: str) -> float:
        """Assess impact of soil type on GPR effectiveness."""
        soil_impacts = {
            "clay": 0.3,
            "wet_clay": 0.2,
            "sand": 0.9,
            "dry_sand": 0.95,
            "gravel": 0.8,
            "silt": 0.6,
            "peat": 0.4,
            "organic": 0.4,
            "bedrock": 0.9,
            "mixed": 0.6
        }

        return soil_impacts.get(soil_type.lower(), 0.6)

    def _assess_weather_impact(self, weather_conditions: Optional[str]) -> float:
        """Assess weather impact on method execution."""
        if not weather_conditions:
            return 0.8

        weather = weather_conditions.lower()
        if any(condition in weather for condition in ["heavy rain", "storm", "severe"]):
            return 0.3
        elif any(condition in weather for condition in ["rain", "wet"]):
            return 0.6
        elif any(condition in weather for condition in ["clear", "dry", "sunny"]):
            return 1.0
        else:
            return 0.8

    def _assess_site_constraints_impact(self, site_constraints: List[str]) -> float:
        """Assess site constraints impact on method execution."""
        if not site_constraints:
            return 1.0

        high_impact = ["traffic", "active construction", "limited access", "hazardous"]
        medium_impact = ["utilities present", "vegetation", "restricted hours"]

        high_count = sum(1 for constraint in site_constraints
                        if any(impact in constraint.lower() for impact in high_impact))
        medium_count = sum(1 for constraint in site_constraints
                          if any(impact in constraint.lower() for impact in medium_impact))

        impact_score = 1.0 - (high_count * 0.3 + medium_count * 0.15)
        return max(0.2, impact_score)

    def _validate_execution_standards(self, method_execution: MethodExecution,
                                    quality_level: QualityLevel) -> Tuple[float, List[str]]:
        """Validate execution against standards."""
        issues = []

        method = method_execution.method
        method_reqs = self.method_requirements.get(quality_level, {}).get(method, {})

        # Check if execution standards are documented
        if not method_execution.results_summary:
            issues.append("Method execution results not documented")
            return 0.5, issues

        # Check execution against standards (simplified validation)
        execution_standards = method_reqs.get("execution_standards", [])
        if execution_standards and not method_execution.results_summary:
            issues.append("Execution standards compliance not documented")
            return 0.6, issues

        # Check safety requirements for intrusive methods
        if method in [SurveyMethod.TRIAL_HOLES, SurveyMethod.VACUUM_EXCAVATION]:
            safety_reqs = method_reqs.get("safety_requirements", [])
            if safety_reqs:
                # Check if safety measures are documented in limitations or results
                safety_documented = any(
                    "safety" in limitation.lower() or "permit" in limitation.lower()
                    for limitation in method_execution.limitations_encountered
                )
                if not safety_documented:
                    issues.append("Safety requirements compliance not documented")
                    return 0.7, issues

        return 1.0, issues

    def _generate_method_recommendations(self, method: SurveyMethod,
                                       issues: List[str],
                                       compliance_score: float,
                                       environmental_conditions: EnvironmentalCondition) -> List[str]:
        """Generate recommendations for method improvement."""
        recommendations = []

        # Equipment recommendations
        if any("equipment" in issue.lower() for issue in issues):
            if method == SurveyMethod.GROUND_PENETRATING_RADAR:
                recommendations.append("Ensure GPR equipment is properly calibrated and suitable for soil conditions")
                if "clay" in environmental_conditions.soil_type.lower():
                    recommendations.append("Consider higher frequency antennas for clay soil conditions")

            elif method == SurveyMethod.ELECTROMAGNETIC_DETECTION:
                recommendations.append("Use both cable avoidance tools and precision locators for comprehensive detection")

        # Coverage recommendations
        if any("coverage" in issue.lower() for issue in issues):
            recommendations.append("Increase survey coverage to meet PAS 128 requirements")
            recommendations.append("Ensure systematic survey pattern with appropriate line spacing")

        # Environmental recommendations
        if any("soil" in issue.lower() for issue in issues):
            recommendations.append("Adjust survey methodology for soil conditions")

        if any("weather" in issue.lower() for issue in issues):
            recommendations.append("Consider rescheduling survey for better weather conditions")

        # Quality improvement recommendations
        if compliance_score < 0.8:
            recommendations.append("Improve method execution quality through better planning and procedures")
            recommendations.append("Provide additional operator training if needed")

        # Safety recommendations for intrusive methods
        if method in [SurveyMethod.TRIAL_HOLES, SurveyMethod.VACUUM_EXCAVATION]:
            recommendations.append("Ensure all safety protocols and permits are properly documented")
            recommendations.append("Follow safe digging practices and utility location procedures")

        return recommendations

    def validate_all_methods(self, survey_data: SurveyData) -> Dict[SurveyMethod, MethodValidationDetails]:
        """
        Validate all executed methods in survey data.

        Args:
            survey_data: Complete survey data to validate

        Returns:
            Dictionary mapping methods to their validation details
        """
        self.logger.info(f"Validating all methods for survey {survey_data.survey_id}")

        validation_results = {}

        for method_execution in survey_data.methods_executed:
            validation_result = self.validate_method_execution(
                method_execution,
                survey_data.target_quality_level,
                survey_data.environmental_conditions
            )
            validation_results[method_execution.method] = validation_result

        return validation_results

    def generate_method_compliance_report(self, survey_data: SurveyData) -> List[ComplianceCheck]:
        """
        Generate compliance checks for all methods.

        Args:
            survey_data: Survey data to generate compliance report for

        Returns:
            List of compliance checks for methods
        """
        self.logger.info(f"Generating method compliance report for survey {survey_data.survey_id}")

        validation_results = self.validate_all_methods(survey_data)
        compliance_checks = []

        for method, validation_details in validation_results.items():
            check = ComplianceCheck(
                check_name=f"Method Validation: {method.value.replace('_', ' ').title()}",
                passed=validation_details.validation_result == MethodValidationResult.COMPLIANT,
                score=validation_details.compliance_score,
                details=f"Validation result: {validation_details.validation_result.value}, "
                       f"Issues: {len(validation_details.issues_identified)}",
                requirements=[
                    "Equipment adequacy",
                    "Coverage requirements",
                    "Execution quality",
                    "Environmental suitability",
                    "Standards compliance"
                ],
                gaps=validation_details.issues_identified
            )
            compliance_checks.append(check)

        # Overall method compliance check
        overall_scores = [details.compliance_score for details in validation_results.values()]
        overall_score = np.mean(overall_scores) if overall_scores else 0.0

        overall_check = ComplianceCheck(
            check_name="Overall Method Compliance",
            passed=overall_score >= self.validation_thresholds["minimum_compliance_score"],
            score=overall_score,
            details=f"Average method compliance score: {overall_score:.2f}",
            requirements=["All required methods executed to standard"],
            gaps=[
                f"Method {method.value} below compliance threshold"
                for method, details in validation_results.items()
                if details.compliance_score < self.validation_thresholds["minimum_compliance_score"]
            ]
        )
        compliance_checks.append(overall_check)

        return compliance_checks