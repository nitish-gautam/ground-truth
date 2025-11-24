"""
PAS 128 Integration Service

This service integrates PAS 128 compliance checking with existing
environmental analysis, validation, and material classification services
to provide comprehensive GPR survey assessment capabilities.
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from ..schemas.pas128 import (
    SurveyData, EnvironmentalCondition, UtilityDetection, AccuracyMeasurement,
    MethodExecution, DeliverableItem, SurveyMethod, DetectionMethod, DeliverableType
)
from ..models.environmental import EnvironmentalData
from .pas128_compliance_service import PAS128ComplianceService
from .comprehensive_environmental_analyzer import ComprehensiveEnvironmentalAnalyzer
from .validation_service import ValidationService
from .material_classification import MaterialClassificationService
from .twente_feature_extractor import TwenteFeatureExtractor

logger = logging.getLogger(__name__)


class PAS128IntegrationService:
    """
    Integration service for PAS 128 compliance with existing systems.

    This service provides:
    - Integration with environmental analysis systems
    - Material classification for compliance assessment
    - Validation framework integration
    - Real-world data processing from Twente dataset
    - Unified compliance assessment workflow
    """

    def __init__(self):
        """Initialize the integration service."""
        self.logger = logging.getLogger(__name__)

        # Initialize core services
        self.pas128_service = PAS128ComplianceService()
        self.environmental_analyzer = ComprehensiveEnvironmentalAnalyzer()
        self.validation_service = ValidationService()
        self.material_classifier = MaterialClassificationService()
        self.twente_extractor = TwenteFeatureExtractor()

        self.logger.info("PAS 128 Integration Service initialized")

    def create_survey_data_from_twente(self, twente_data_path: str,
                                     survey_id: str,
                                     target_quality_level: str) -> SurveyData:
        """
        Create PAS 128 SurveyData from Twente dataset.

        Args:
            twente_data_path: Path to Twente dataset file
            survey_id: Unique survey identifier
            target_quality_level: Target PAS 128 quality level

        Returns:
            Structured survey data for PAS 128 compliance checking
        """
        self.logger.info(f"Creating survey data from Twente dataset: {twente_data_path}")

        try:
            # Extract features from Twente data
            features = self.twente_extractor.extract_comprehensive_features(twente_data_path)

            # Create environmental conditions from Twente data
            environmental_conditions = self._extract_environmental_conditions(features)

            # Create method executions from survey parameters
            method_executions = self._create_method_executions_from_twente(features)

            # Create utility detections from GPR analysis
            utility_detections = self._create_utility_detections_from_features(features)

            # Create deliverables (simulated for Twente data)
            deliverables = self._create_simulated_deliverables()

            # Determine survey extent from data
            survey_extent = self._extract_survey_extent(features)

            # Create site location information
            site_location = {
                "dataset": "Twente",
                "file_path": twente_data_path,
                "coordinates": features.get("spatial_features", {}).get("bounds", {}),
                "area_type": "research_site"
            }

            survey_data = SurveyData(
                survey_id=survey_id,
                survey_date=datetime.now(),  # Use current date as survey date
                site_location=site_location,
                environmental_conditions=environmental_conditions,
                methods_executed=method_executions,
                deliverables=deliverables,
                utility_detections=utility_detections,
                target_quality_level=target_quality_level,
                survey_extent=survey_extent
            )

            self.logger.info(
                f"Created survey data with {len(utility_detections)} detections "
                f"and {len(method_executions)} methods"
            )

            return survey_data

        except Exception as e:
            self.logger.error(f"Error creating survey data from Twente: {e}")
            raise

    def _extract_environmental_conditions(self, features: Dict[str, Any]) -> EnvironmentalCondition:
        """Extract environmental conditions from Twente features."""
        # Extract soil type from material classification
        material_features = features.get("material_features", {})
        soil_type = material_features.get("dominant_material", "mixed")

        # Extract moisture content if available
        moisture_content = material_features.get("moisture_indicator")

        # Determine ground conditions
        ground_conditions = []
        if features.get("signal_features", {}).get("attenuation", 0) > 0.7:
            ground_conditions.append("high_attenuation")
        if features.get("amplitude_features", {}).get("noise_level", 0) > 0.5:
            ground_conditions.append("noisy_conditions")

        # Simulate weather conditions (not available in Twente data)
        weather_conditions = "clear"  # Default assumption

        # Extract site constraints from data characteristics
        site_constraints = []
        if features.get("data_quality", {}).get("completeness", 1.0) < 0.9:
            site_constraints.append("data_gaps")
        if features.get("signal_features", {}).get("interference", 0) > 0.3:
            site_constraints.append("electromagnetic_interference")

        return EnvironmentalCondition(
            soil_type=soil_type,
            moisture_content=moisture_content,
            ground_conditions=ground_conditions,
            weather_conditions=weather_conditions,
            site_constraints=site_constraints
        )

    def _create_method_executions_from_twente(self, features: Dict[str, Any]) -> List[MethodExecution]:
        """Create method executions based on Twente data characteristics."""
        method_executions = []

        # GPR method execution (always present for Twente data)
        gpr_execution = MethodExecution(
            method=SurveyMethod.GROUND_PENETRATING_RADAR,
            execution_date=datetime.now(),
            equipment_used=[],  # Would be populated with actual equipment data
            coverage_area=features.get("spatial_features", {}).get("total_area", 100.0),
            execution_quality=self._assess_data_quality(features),
            limitations_encountered=self._identify_limitations(features),
            results_summary=f"GPR survey with {features.get('detection_count', 0)} potential targets detected"
        )
        method_executions.append(gpr_execution)

        # Add other methods based on quality assumptions
        # Comprehensive records (simulated)
        records_execution = MethodExecution(
            method=SurveyMethod.COMPREHENSIVE_RECORDS,
            execution_date=datetime.now(),
            equipment_used=[],
            coverage_area=features.get("spatial_features", {}).get("total_area", 100.0),
            execution_quality=0.8,  # Assumed good quality
            limitations_encountered=["limited_historical_data"],
            results_summary="Comprehensive records search completed"
        )
        method_executions.append(records_execution)

        # Site reconnaissance (simulated)
        recon_execution = MethodExecution(
            method=SurveyMethod.SITE_RECONNAISSANCE,
            execution_date=datetime.now(),
            equipment_used=[],
            coverage_area=features.get("spatial_features", {}).get("total_area", 100.0),
            execution_quality=0.9,  # Assumed high quality
            limitations_encountered=[],
            results_summary="Site reconnaissance completed"
        )
        method_executions.append(recon_execution)

        # Topographical survey (simulated)
        topo_execution = MethodExecution(
            method=SurveyMethod.TOPOGRAPHICAL_SURVEY,
            execution_date=datetime.now(),
            equipment_used=[],
            coverage_area=features.get("spatial_features", {}).get("total_area", 100.0),
            execution_quality=0.85,
            limitations_encountered=[],
            results_summary="Topographical survey completed"
        )
        method_executions.append(topo_execution)

        return method_executions

    def _assess_data_quality(self, features: Dict[str, Any]) -> float:
        """Assess data quality from Twente features."""
        quality_factors = []

        # Signal quality
        signal_quality = features.get("signal_features", {}).get("snr", 0.5)
        quality_factors.append(min(1.0, signal_quality))

        # Data completeness
        completeness = features.get("data_quality", {}).get("completeness", 1.0)
        quality_factors.append(completeness)

        # Amplitude consistency
        amplitude_stats = features.get("amplitude_features", {})
        if "std" in amplitude_stats and "mean" in amplitude_stats:
            consistency = 1.0 - min(1.0, amplitude_stats["std"] / amplitude_stats["mean"])
            quality_factors.append(consistency)

        return np.mean(quality_factors) if quality_factors else 0.7

    def _identify_limitations(self, features: Dict[str, Any]) -> List[str]:
        """Identify survey limitations from Twente data."""
        limitations = []

        # Signal limitations
        signal_features = features.get("signal_features", {})
        if signal_features.get("attenuation", 0) > 0.8:
            limitations.append("high_signal_attenuation")
        if signal_features.get("interference", 0) > 0.5:
            limitations.append("electromagnetic_interference")

        # Data limitations
        data_quality = features.get("data_quality", {})
        if data_quality.get("completeness", 1.0) < 0.9:
            limitations.append("incomplete_data_coverage")

        # Material limitations
        material_features = features.get("material_features", {})
        if material_features.get("dominant_material") == "clay":
            limitations.append("clay_soil_limits_penetration")

        return limitations

    def _create_utility_detections_from_features(self, features: Dict[str, Any]) -> List[UtilityDetection]:
        """Create utility detections from Twente analysis features."""
        detections = []

        # Get detection features
        detection_features = features.get("detection_features", {})
        object_features = features.get("object_features", [])

        # Create detections from object features
        for i, obj_feature in enumerate(object_features[:10]):  # Limit to 10 detections
            # Extract location information
            location = {
                "x": obj_feature.get("x_position", i * 10),
                "y": obj_feature.get("y_position", 0),
                "trace_number": obj_feature.get("trace", i)
            }

            # Estimate depth
            depth = obj_feature.get("depth", 1.0)

            # Calculate accuracy based on signal quality
            horizontal_accuracy = self._estimate_accuracy(obj_feature, features)

            accuracy = AccuracyMeasurement(
                horizontal_accuracy=horizontal_accuracy,
                vertical_accuracy=horizontal_accuracy * 0.8,  # Assume slightly worse vertical
                confidence_level=obj_feature.get("confidence", 0.7),
                measurement_method="gpr_analysis"
            )

            # Determine utility type (simplified classification)
            utility_type = self._classify_utility_type(obj_feature)

            detection = UtilityDetection(
                utility_id=f"UTL-{i+1:03d}",
                utility_type=utility_type,
                detection_method=DetectionMethod.GROUND_PENETRATING_RADAR,
                location=location,
                depth=depth,
                accuracy=accuracy,
                confidence=obj_feature.get("confidence", 0.7),
                verified=False,  # Not verified in Twente data
                verification_method=None
            )
            detections.append(detection)

        self.logger.info(f"Created {len(detections)} utility detections from Twente features")
        return detections

    def _estimate_accuracy(self, obj_feature: Dict[str, Any], global_features: Dict[str, Any]) -> float:
        """Estimate detection accuracy based on signal characteristics."""
        # Base accuracy (in mm)
        base_accuracy = 500.0

        # Adjust based on signal quality
        snr = global_features.get("signal_features", {}).get("snr", 0.5)
        snr_factor = max(0.5, min(2.0, snr))
        accuracy = base_accuracy / snr_factor

        # Adjust based on depth
        depth = obj_feature.get("depth", 1.0)
        depth_factor = 1.0 + (depth - 1.0) * 0.3  # Accuracy degrades with depth
        accuracy *= depth_factor

        # Adjust based on material
        material = global_features.get("material_features", {}).get("dominant_material", "sand")
        material_factors = {
            "clay": 2.0,
            "wet_clay": 2.5,
            "sand": 1.0,
            "gravel": 1.2,
            "mixed": 1.3
        }
        accuracy *= material_factors.get(material, 1.3)

        return min(2000.0, accuracy)  # Cap at 2000mm

    def _classify_utility_type(self, obj_feature: Dict[str, Any]) -> str:
        """Classify utility type based on object features."""
        # Simplified classification based on signal characteristics
        amplitude = obj_feature.get("amplitude", 0.5)
        width = obj_feature.get("width", 1.0)

        if amplitude > 0.8 and width < 0.5:
            return "pipe_metallic"
        elif amplitude > 0.6:
            return "cable"
        elif width > 1.5:
            return "pipe_large"
        else:
            return "pipe_small"

    def _create_simulated_deliverables(self) -> List[DeliverableItem]:
        """Create simulated deliverables for Twente-based survey."""
        deliverables = []

        # Survey report
        deliverables.append(DeliverableItem(
            deliverable_type=DeliverableType.SURVEY_REPORT,
            provided=True,
            quality_score=0.8,
            completeness_score=0.85,
            file_path=None,
            notes="Comprehensive survey report based on GPR analysis"
        ))

        # Utility location plans
        deliverables.append(DeliverableItem(
            deliverable_type=DeliverableType.UTILITY_LOCATION_PLANS,
            provided=True,
            quality_score=0.75,
            completeness_score=0.8,
            file_path=None,
            notes="Utility location plans showing detected utilities"
        ))

        # Detection survey results
        deliverables.append(DeliverableItem(
            deliverable_type=DeliverableType.DETECTION_SURVEY_RESULTS,
            provided=True,
            quality_score=0.9,
            completeness_score=0.95,
            file_path=None,
            notes="Detailed GPR detection results and analysis"
        ))

        return deliverables

    def _extract_survey_extent(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract survey extent information from features."""
        spatial_features = features.get("spatial_features", {})

        return {
            "total_area": spatial_features.get("total_area", 100.0),
            "survey_lines": spatial_features.get("line_count", 1),
            "line_spacing": spatial_features.get("line_spacing", 1.0),
            "bounds": spatial_features.get("bounds", {
                "min_x": 0, "max_x": 100,
                "min_y": 0, "max_y": 10
            })
        }

    def analyze_environmental_compatibility(self, survey_data: SurveyData) -> Dict[str, Any]:
        """
        Analyze environmental compatibility using existing environmental analyzer.

        Args:
            survey_data: Survey data to analyze

        Returns:
            Environmental compatibility analysis
        """
        self.logger.info(f"Analyzing environmental compatibility for survey {survey_data.survey_id}")

        try:
            # Convert PAS 128 environmental data to system format
            environmental_data = self._convert_to_environmental_data(survey_data.environmental_conditions)

            # Perform environmental analysis
            analysis_result = self.environmental_analyzer.analyze_environmental_impact(environmental_data)

            # Extract relevant compatibility metrics
            compatibility_analysis = {
                "overall_compatibility": analysis_result.get("overall_score", 0.7),
                "gpr_suitability": analysis_result.get("method_effectiveness", {}).get("gpr", 0.7),
                "soil_impact": analysis_result.get("soil_conditions_impact", 0.7),
                "weather_impact": analysis_result.get("weather_impact", 0.8),
                "site_constraints_impact": analysis_result.get("site_constraints_impact", 0.8),
                "recommendations": analysis_result.get("recommendations", []),
                "limiting_factors": analysis_result.get("limiting_factors", [])
            }

            return compatibility_analysis

        except Exception as e:
            self.logger.error(f"Error in environmental compatibility analysis: {e}")
            return {
                "overall_compatibility": 0.5,
                "error": str(e),
                "recommendations": ["Environmental analysis unavailable"]
            }

    def _convert_to_environmental_data(self, env_conditions: EnvironmentalCondition) -> EnvironmentalData:
        """Convert PAS 128 environmental conditions to system environmental data format."""
        # This would be implemented based on the actual EnvironmentalData schema
        # For now, return a placeholder that represents the conversion
        return {
            "soil_type": env_conditions.soil_type,
            "moisture_content": env_conditions.moisture_content,
            "ground_conditions": env_conditions.ground_conditions,
            "weather_conditions": env_conditions.weather_conditions,
            "site_constraints": env_conditions.site_constraints,
            "temperature": getattr(env_conditions, 'temperature', None)
        }

    def validate_with_ground_truth(self, survey_data: SurveyData,
                                 ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate survey results against ground truth data.

        Args:
            survey_data: Survey data to validate
            ground_truth_path: Optional path to ground truth data

        Returns:
            Validation results
        """
        self.logger.info(f"Validating survey {survey_data.survey_id} against ground truth")

        try:
            # Use validation service to compare with ground truth
            validation_results = {}

            if ground_truth_path and Path(ground_truth_path).exists():
                # Perform ground truth validation
                validation_results = self.validation_service.validate_against_ground_truth(
                    survey_data.utility_detections,
                    ground_truth_path
                )
            else:
                # Perform internal consistency validation
                validation_results = self._perform_internal_validation(survey_data)

            return {
                "validation_type": "ground_truth" if ground_truth_path else "internal_consistency",
                "overall_accuracy": validation_results.get("accuracy", 0.0),
                "detection_rate": validation_results.get("detection_rate", 0.0),
                "false_positive_rate": validation_results.get("false_positive_rate", 0.0),
                "position_accuracy": validation_results.get("position_accuracy", 0.0),
                "depth_accuracy": validation_results.get("depth_accuracy", 0.0),
                "validation_summary": validation_results.get("summary", "Validation completed")
            }

        except Exception as e:
            self.logger.error(f"Error in ground truth validation: {e}")
            return {
                "validation_type": "error",
                "error": str(e),
                "overall_accuracy": 0.0
            }

    def _perform_internal_validation(self, survey_data: SurveyData) -> Dict[str, Any]:
        """Perform internal consistency validation of survey data."""
        # Check detection consistency
        detections = survey_data.utility_detections

        if not detections:
            return {
                "accuracy": 0.0,
                "detection_rate": 0.0,
                "summary": "No detections to validate"
            }

        # Calculate internal consistency metrics
        confidence_scores = [det.confidence for det in detections]
        avg_confidence = np.mean(confidence_scores)

        # Check accuracy consistency
        accuracies = [det.accuracy.horizontal_accuracy for det in detections]
        accuracy_consistency = 1.0 - (np.std(accuracies) / np.mean(accuracies))

        return {
            "accuracy": avg_confidence,
            "detection_rate": len(detections) / max(1, len(survey_data.methods_executed)),
            "confidence_consistency": accuracy_consistency,
            "summary": f"Internal validation: {len(detections)} detections analyzed"
        }

    def classify_materials_for_compliance(self, survey_data: SurveyData) -> Dict[str, Any]:
        """
        Classify materials to support compliance assessment.

        Args:
            survey_data: Survey data with material information

        Returns:
            Material classification results for compliance
        """
        self.logger.info(f"Classifying materials for compliance assessment")

        try:
            # Extract material information from environmental conditions
            soil_type = survey_data.environmental_conditions.soil_type

            # Use material classifier if available
            # This would typically process actual GPR data
            material_analysis = {
                "primary_material": soil_type,
                "material_confidence": 0.8,
                "gpr_penetration_capability": self._assess_gpr_penetration(soil_type),
                "expected_accuracy_impact": self._assess_accuracy_impact(soil_type),
                "recommended_frequency": self._recommend_gpr_frequency(soil_type),
                "compliance_implications": self._assess_compliance_implications(soil_type)
            }

            return material_analysis

        except Exception as e:
            self.logger.error(f"Error in material classification: {e}")
            return {
                "primary_material": "unknown",
                "error": str(e),
                "compliance_implications": ["Material analysis unavailable"]
            }

    def _assess_gpr_penetration(self, soil_type: str) -> float:
        """Assess GPR penetration capability for soil type."""
        penetration_factors = {
            "clay": 0.3,
            "wet_clay": 0.2,
            "sand": 0.9,
            "dry_sand": 0.95,
            "gravel": 0.8,
            "silt": 0.6,
            "peat": 0.4,
            "bedrock": 0.95,
            "mixed": 0.6
        }
        return penetration_factors.get(soil_type.lower(), 0.6)

    def _assess_accuracy_impact(self, soil_type: str) -> float:
        """Assess expected accuracy impact of soil type."""
        accuracy_factors = {
            "clay": 2.5,  # Accuracy degraded by factor of 2.5
            "wet_clay": 3.0,
            "sand": 1.0,  # Baseline accuracy
            "dry_sand": 0.8,  # Better than baseline
            "gravel": 1.3,
            "silt": 1.5,
            "peat": 2.0,
            "bedrock": 0.7,
            "mixed": 1.4
        }
        return accuracy_factors.get(soil_type.lower(), 1.5)

    def _recommend_gpr_frequency(self, soil_type: str) -> str:
        """Recommend GPR frequency for soil type."""
        frequency_recommendations = {
            "clay": "900-1600 MHz (higher frequency for shallow penetration)",
            "wet_clay": "1200-2000 MHz (highest frequency due to high attenuation)",
            "sand": "400-900 MHz (good penetration with moderate frequency)",
            "dry_sand": "200-600 MHz (excellent penetration with lower frequency)",
            "gravel": "600-1200 MHz (moderate frequency for varied conditions)",
            "silt": "700-1200 MHz (moderate to high frequency)",
            "peat": "800-1500 MHz (higher frequency due to organic content)",
            "bedrock": "100-400 MHz (low frequency for maximum penetration)",
            "mixed": "600-1000 MHz (versatile frequency range)"
        }
        return frequency_recommendations.get(soil_type.lower(), "600-1000 MHz (general purpose)")

    def _assess_compliance_implications(self, soil_type: str) -> List[str]:
        """Assess compliance implications of soil type."""
        implications = {
            "clay": [
                "GPR effectiveness severely limited",
                "May not achieve QL-B or QL-A without alternative methods",
                "Consider electromagnetic methods as primary detection",
                "Increase survey line density to compensate"
            ],
            "wet_clay": [
                "Extremely poor GPR performance expected",
                "QL-A achievement unlikely without extensive intrusive investigation",
                "Alternative detection methods essential",
                "Conservative quality level targets recommended"
            ],
            "sand": [
                "Excellent conditions for all quality levels",
                "Standard GPR methodology suitable",
                "High accuracy achievable",
                "All PAS 128 quality levels feasible"
            ],
            "mixed": [
                "Variable performance expected",
                "Quality level achievement depends on local conditions",
                "Adaptive methodology may be required",
                "Additional verification recommended"
            ]
        }

        return implications.get(soil_type.lower(), [
            "Standard methodology applicable",
            "Monitor performance and adjust as needed"
        ])

    def generate_integrated_compliance_report(self, twente_data_path: str,
                                            survey_id: str,
                                            target_quality_level: str,
                                            ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive integrated compliance report.

        Args:
            twente_data_path: Path to Twente dataset
            survey_id: Survey identifier
            target_quality_level: Target PAS 128 quality level
            ground_truth_path: Optional ground truth data path

        Returns:
            Comprehensive integrated compliance report
        """
        self.logger.info(f"Generating integrated compliance report for {survey_id}")

        try:
            # Step 1: Create survey data from Twente
            survey_data = self.create_survey_data_from_twente(
                twente_data_path, survey_id, target_quality_level
            )

            # Step 2: Perform PAS 128 compliance check
            compliance_report = self.pas128_service.perform_comprehensive_compliance_check(survey_data)

            # Step 3: Analyze environmental compatibility
            environmental_analysis = self.analyze_environmental_compatibility(survey_data)

            # Step 4: Validate against ground truth
            validation_results = self.validate_with_ground_truth(survey_data, ground_truth_path)

            # Step 5: Classify materials for compliance
            material_analysis = self.classify_materials_for_compliance(survey_data)

            # Step 6: Compile integrated report
            integrated_report = {
                "survey_id": survey_id,
                "data_source": "Twente Dataset",
                "assessment_timestamp": datetime.now().isoformat(),
                "target_quality_level": target_quality_level,
                "achieved_quality_level": compliance_report.achieved_quality_level.value,
                "compliance_score": compliance_report.overall_compliance_score,
                "compliance_report": compliance_report,
                "environmental_analysis": environmental_analysis,
                "validation_results": validation_results,
                "material_analysis": material_analysis,
                "integration_summary": self._create_integration_summary(
                    compliance_report, environmental_analysis, validation_results, material_analysis
                ),
                "recommendations": self._generate_integrated_recommendations(
                    compliance_report, environmental_analysis, validation_results, material_analysis
                )
            }

            self.logger.info(f"Integrated compliance report generated successfully for {survey_id}")
            return integrated_report

        except Exception as e:
            self.logger.error(f"Error generating integrated compliance report: {e}")
            raise

    def _create_integration_summary(self, compliance_report, environmental_analysis,
                                  validation_results, material_analysis) -> Dict[str, Any]:
        """Create summary of integrated analysis."""
        return {
            "overall_assessment": "integrated_analysis_complete",
            "compliance_status": compliance_report.achieved_quality_level.value,
            "environmental_suitability": environmental_analysis.get("overall_compatibility", 0.5),
            "validation_accuracy": validation_results.get("overall_accuracy", 0.0),
            "material_suitability": material_analysis.get("gpr_penetration_capability", 0.5),
            "key_findings": [
                f"Achieved {compliance_report.achieved_quality_level.value} quality level",
                f"Environmental compatibility: {environmental_analysis.get('overall_compatibility', 0.5):.2f}",
                f"Material classification: {material_analysis.get('primary_material', 'unknown')}",
                f"Validation accuracy: {validation_results.get('overall_accuracy', 0.0):.2f}"
            ],
            "critical_issues": compliance_report.critical_gaps[:3],
            "data_integration_success": True
        }

    def _generate_integrated_recommendations(self, compliance_report, environmental_analysis,
                                           validation_results, material_analysis) -> List[str]:
        """Generate integrated recommendations from all analyses."""
        recommendations = []

        # Add compliance recommendations
        recommendations.extend(compliance_report.improvement_recommendations[:3])

        # Add environmental recommendations
        env_recommendations = environmental_analysis.get("recommendations", [])
        recommendations.extend(env_recommendations[:2])

        # Add material-specific recommendations
        material_implications = material_analysis.get("compliance_implications", [])
        recommendations.extend(material_implications[:2])

        # Add validation-based recommendations
        if validation_results.get("overall_accuracy", 0.0) < 0.7:
            recommendations.append("Improve detection accuracy through method refinement")

        # Add integration-specific recommendations
        recommendations.extend([
            "Continue integrated analysis approach for comprehensive assessment",
            "Consider real-time environmental monitoring for future surveys",
            "Implement adaptive methodology based on material classification"
        ])

        return recommendations[:10]  # Limit to top 10 recommendations