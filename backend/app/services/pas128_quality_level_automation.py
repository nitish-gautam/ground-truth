"""
PAS 128 Automated Quality Level Determination System

This service provides automated quality level determination using machine learning
and rule-based approaches to assess the highest achievable PAS 128 quality level
based on survey data, environmental conditions, and method effectiveness.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from ..schemas.pas128 import (
    QualityLevel, SurveyMethod, DetectionMethod, DeliverableType,
    SurveyData, QualityLevelAssessment, EnvironmentalCondition,
    MethodExecution, UtilityDetection, AccuracyMeasurement
)

logger = logging.getLogger(__name__)


@dataclass
class QualityLevelFeatures:
    """Features used for quality level determination"""
    # Method features
    has_electromagnetic: bool
    has_gpr: bool
    has_intrusive: bool
    has_comprehensive_records: bool
    has_topographical: bool
    method_execution_quality: float

    # Accuracy features
    avg_horizontal_accuracy: float
    avg_vertical_accuracy: float
    accuracy_confidence: float

    # Environmental features
    soil_suitability_score: float
    weather_impact_score: float
    site_constraints_score: float

    # Detection features
    detection_count: int
    detection_confidence: float
    detection_verification_rate: float

    # Deliverable features
    deliverable_completeness: float
    deliverable_quality: float

    # Coverage features
    survey_coverage: float
    method_coverage_adequacy: float


class PAS128QualityLevelAutomation:
    """
    Automated quality level determination system for PAS 128 compliance.

    This system uses both rule-based and machine learning approaches to:
    - Automatically determine achievable quality levels
    - Predict quality level based on survey characteristics
    - Provide confidence scores and recommendations
    - Account for environmental limitations and method effectiveness
    """

    def __init__(self):
        """Initialize the quality level automation system."""
        self.logger = logging.getLogger(__name__)

        # Quality level hierarchy (ordered from lowest to highest)
        self.quality_levels = [QualityLevel.QL_D, QualityLevel.QL_C, QualityLevel.QL_B, QualityLevel.QL_A]

        # Initialize method effectiveness mappings
        self.method_effectiveness = self._initialize_method_effectiveness()

        # Initialize environmental impact models
        self.environmental_models = self._initialize_environmental_models()

        # Initialize quality level requirements
        self.quality_requirements = self._initialize_quality_requirements()

        # Machine learning model for quality level prediction
        self.ml_model = None
        self.feature_encoders = {}

        self.logger.info("PAS 128 Quality Level Automation system initialized")

    def _initialize_method_effectiveness(self) -> Dict[DetectionMethod, Dict[str, float]]:
        """Initialize method effectiveness mappings based on conditions."""
        return {
            DetectionMethod.GROUND_PENETRATING_RADAR: {
                "clay": 0.3,
                "wet_clay": 0.2,
                "sand": 0.9,
                "gravel": 0.8,
                "silt": 0.6,
                "peat": 0.4,
                "bedrock": 0.95,
                "base_effectiveness": 0.75
            },
            DetectionMethod.ELECTROMAGNETIC: {
                "base_effectiveness": 0.85,
                "metallic_utilities": 0.95,
                "non_metallic": 0.3,
                "deep_utilities": 0.5
            },
            DetectionMethod.RADIO_DETECTION: {
                "base_effectiveness": 0.8,
                "signal_attenuation": 0.6,
                "access_limited": 0.4
            }
        }

    def _initialize_environmental_models(self) -> Dict[str, Any]:
        """Initialize environmental impact models."""
        return {
            "soil_gpr_impact": {
                "clay": 0.25,
                "wet_clay": 0.15,
                "sand": 0.85,
                "dry_sand": 0.95,
                "gravel": 0.75,
                "silt": 0.55,
                "peat": 0.35,
                "organic": 0.4,
                "bedrock": 0.9,
                "mixed": 0.6
            },
            "weather_impact": {
                "excellent": 1.0,
                "good": 0.9,
                "fair": 0.7,
                "poor": 0.5,
                "severe": 0.3
            },
            "site_constraints": {
                "none": 1.0,
                "minor": 0.85,
                "moderate": 0.65,
                "significant": 0.45,
                "severe": 0.25
            }
        }

    def _initialize_quality_requirements(self) -> Dict[QualityLevel, Dict[str, Any]]:
        """Initialize quality level requirements and thresholds."""
        return {
            QualityLevel.QL_D: {
                "required_methods": [SurveyMethod.RECORDS_SEARCH, SurveyMethod.SITE_RECONNAISSANCE],
                "accuracy_threshold": {"horizontal": 2000.0},
                "min_deliverables": 2,
                "environmental_threshold": 0.3
            },
            QualityLevel.QL_C: {
                "required_methods": [
                    SurveyMethod.COMPREHENSIVE_RECORDS,
                    SurveyMethod.SITE_RECONNAISSANCE,
                    SurveyMethod.TOPOGRAPHICAL_SURVEY
                ],
                "accuracy_threshold": {"horizontal": 1000.0},
                "min_deliverables": 3,
                "environmental_threshold": 0.4
            },
            QualityLevel.QL_B: {
                "required_methods": [
                    SurveyMethod.COMPREHENSIVE_RECORDS,
                    SurveyMethod.SITE_RECONNAISSANCE,
                    SurveyMethod.TOPOGRAPHICAL_SURVEY,
                    SurveyMethod.ELECTROMAGNETIC_DETECTION,
                    SurveyMethod.GROUND_PENETRATING_RADAR
                ],
                "accuracy_threshold": {"horizontal": 500.0},
                "min_deliverables": 4,
                "environmental_threshold": 0.5,
                "min_detection_methods": 2
            },
            QualityLevel.QL_A: {
                "required_methods": [
                    SurveyMethod.COMPREHENSIVE_RECORDS,
                    SurveyMethod.SITE_RECONNAISSANCE,
                    SurveyMethod.TOPOGRAPHICAL_SURVEY,
                    SurveyMethod.ELECTROMAGNETIC_DETECTION,
                    SurveyMethod.GROUND_PENETRATING_RADAR,
                    SurveyMethod.TRIAL_HOLES,
                    SurveyMethod.VACUUM_EXCAVATION
                ],
                "accuracy_threshold": {"horizontal": 300.0, "vertical": 300.0},
                "min_deliverables": 6,
                "environmental_threshold": 0.6,
                "min_detection_methods": 2,
                "min_intrusive_methods": 1,
                "min_verification_rate": 0.8
            }
        }

    def extract_features(self, survey_data: SurveyData) -> QualityLevelFeatures:
        """
        Extract features from survey data for quality level determination.

        Args:
            survey_data: Survey data to extract features from

        Returns:
            Extracted features for quality level determination
        """
        self.logger.debug(f"Extracting features for survey {survey_data.survey_id}")

        # Method features
        executed_methods = {method.method for method in survey_data.methods_executed}

        has_electromagnetic = SurveyMethod.ELECTROMAGNETIC_DETECTION in executed_methods
        has_gpr = SurveyMethod.GROUND_PENETRATING_RADAR in executed_methods
        has_intrusive = any(method in executed_methods for method in [
            SurveyMethod.TRIAL_HOLES, SurveyMethod.VACUUM_EXCAVATION, SurveyMethod.HAND_DIGGING
        ])
        has_comprehensive_records = SurveyMethod.COMPREHENSIVE_RECORDS in executed_methods
        has_topographical = SurveyMethod.TOPOGRAPHICAL_SURVEY in executed_methods

        # Calculate average method execution quality
        method_execution_quality = np.mean([
            method.execution_quality for method in survey_data.methods_executed
        ]) if survey_data.methods_executed else 0.0

        # Accuracy features
        if survey_data.utility_detections:
            horizontal_accuracies = [det.accuracy.horizontal_accuracy for det in survey_data.utility_detections]
            vertical_accuracies = [
                det.accuracy.vertical_accuracy for det in survey_data.utility_detections
                if det.accuracy.vertical_accuracy is not None
            ]

            avg_horizontal_accuracy = np.mean(horizontal_accuracies)
            avg_vertical_accuracy = np.mean(vertical_accuracies) if vertical_accuracies else 1000.0
            accuracy_confidence = np.mean([det.accuracy.confidence_level for det in survey_data.utility_detections])
        else:
            avg_horizontal_accuracy = 2000.0  # Default poor accuracy
            avg_vertical_accuracy = 1000.0
            accuracy_confidence = 0.5

        # Environmental features
        soil_suitability = self._calculate_soil_suitability(survey_data.environmental_conditions)
        weather_impact = self._calculate_weather_impact(survey_data.environmental_conditions)
        site_constraints = self._calculate_site_constraints_impact(survey_data.environmental_conditions)

        # Detection features
        detection_count = len(survey_data.utility_detections)
        detection_confidence = np.mean([
            det.confidence for det in survey_data.utility_detections
        ]) if survey_data.utility_detections else 0.0

        verified_detections = sum(1 for det in survey_data.utility_detections if det.verified)
        detection_verification_rate = (
            verified_detections / detection_count if detection_count > 0 else 0.0
        )

        # Deliverable features
        provided_deliverables = [deliv for deliv in survey_data.deliverables if deliv.provided]
        deliverable_completeness = len(provided_deliverables) / len(survey_data.deliverables) if survey_data.deliverables else 0.0

        quality_scores = [
            deliv.quality_score for deliv in provided_deliverables
            if deliv.quality_score is not None
        ]
        deliverable_quality = np.mean(quality_scores) if quality_scores else 0.5

        # Coverage features
        total_coverage = sum(
            method.coverage_area for method in survey_data.methods_executed
            if method.coverage_area is not None
        )
        survey_coverage = min(1.0, total_coverage / 1000.0) if total_coverage > 0 else 0.5  # Normalize to 1000 m²

        # Method coverage adequacy
        detection_methods = [
            method for method in survey_data.methods_executed
            if method.method in [SurveyMethod.ELECTROMAGNETIC_DETECTION, SurveyMethod.GROUND_PENETRATING_RADAR]
        ]
        method_coverage_adequacy = len(detection_methods) / 2.0 if detection_methods else 0.0

        return QualityLevelFeatures(
            has_electromagnetic=has_electromagnetic,
            has_gpr=has_gpr,
            has_intrusive=has_intrusive,
            has_comprehensive_records=has_comprehensive_records,
            has_topographical=has_topographical,
            method_execution_quality=method_execution_quality,
            avg_horizontal_accuracy=avg_horizontal_accuracy,
            avg_vertical_accuracy=avg_vertical_accuracy,
            accuracy_confidence=accuracy_confidence,
            soil_suitability_score=soil_suitability,
            weather_impact_score=weather_impact,
            site_constraints_score=site_constraints,
            detection_count=detection_count,
            detection_confidence=detection_confidence,
            detection_verification_rate=detection_verification_rate,
            deliverable_completeness=deliverable_completeness,
            deliverable_quality=deliverable_quality,
            survey_coverage=survey_coverage,
            method_coverage_adequacy=method_coverage_adequacy
        )

    def _calculate_soil_suitability(self, env_conditions: EnvironmentalCondition) -> float:
        """Calculate soil suitability score for GPR and other methods."""
        soil_type = env_conditions.soil_type.lower()

        # Base soil impact on GPR
        base_score = self.environmental_models["soil_gpr_impact"].get(soil_type, 0.6)

        # Adjust for moisture content if available
        if env_conditions.moisture_content is not None:
            if env_conditions.moisture_content > 30:  # High moisture reduces effectiveness
                base_score *= 0.7
            elif env_conditions.moisture_content < 10:  # Very dry conditions may improve effectiveness
                base_score *= 1.1
                base_score = min(1.0, base_score)

        return base_score

    def _calculate_weather_impact(self, env_conditions: EnvironmentalCondition) -> float:
        """Calculate weather impact on survey quality."""
        if not env_conditions.weather_conditions:
            return 0.8  # Assume reasonable conditions

        weather = env_conditions.weather_conditions.lower()

        # Categorize weather conditions
        if any(condition in weather for condition in ["excellent", "clear", "dry", "sunny"]):
            return 1.0
        elif any(condition in weather for condition in ["good", "partly cloudy"]):
            return 0.9
        elif any(condition in weather for condition in ["fair", "overcast", "light rain"]):
            return 0.7
        elif any(condition in weather for condition in ["poor", "rain", "wet"]):
            return 0.5
        elif any(condition in weather for condition in ["severe", "storm", "heavy rain"]):
            return 0.3
        else:
            return 0.8  # Default

    def _calculate_site_constraints_impact(self, env_conditions: EnvironmentalCondition) -> float:
        """Calculate site constraints impact on survey quality."""
        if not env_conditions.site_constraints:
            return 1.0

        # Count constraint severity
        high_impact = ["traffic", "active construction", "hazardous", "limited access", "underground services"]
        medium_impact = ["utilities present", "soft ground", "vegetation", "restricted hours"]
        low_impact = ["minor obstacles", "aesthetic concerns"]

        high_count = sum(1 for constraint in env_conditions.site_constraints
                        if any(impact in constraint.lower() for impact in high_impact))
        medium_count = sum(1 for constraint in env_conditions.site_constraints
                          if any(impact in constraint.lower() for impact in medium_impact))
        low_count = sum(1 for constraint in env_conditions.site_constraints
                       if any(impact in constraint.lower() for impact in low_impact))

        # Calculate impact score
        impact_score = 1.0 - (high_count * 0.4 + medium_count * 0.2 + low_count * 0.1)
        return max(0.1, impact_score)

    def determine_quality_level_rule_based(self, survey_data: SurveyData,
                                         conservative: bool = True) -> QualityLevelAssessment:
        """
        Determine quality level using rule-based approach.

        Args:
            survey_data: Survey data to assess
            conservative: Whether to use conservative assessment

        Returns:
            Quality level assessment
        """
        self.logger.info(f"Determining quality level using rule-based approach for survey {survey_data.survey_id}")

        features = self.extract_features(survey_data)
        executed_methods = {method.method for method in survey_data.methods_executed}

        # Start from highest quality level and work down
        assessed_level = QualityLevel.QL_D  # Default to lowest
        limiting_factors = []
        confidence = 0.5

        for quality_level in reversed(self.quality_levels):
            requirements = self.quality_requirements[quality_level]
            can_achieve = True
            level_factors = []

            # Check method requirements
            required_methods = set(requirements["required_methods"])
            missing_methods = required_methods - executed_methods

            if missing_methods:
                can_achieve = False
                level_factors.extend([f"Missing method: {method.value}" for method in missing_methods])

            # Check accuracy requirements
            accuracy_req = requirements["accuracy_threshold"]
            if "horizontal" in accuracy_req:
                if features.avg_horizontal_accuracy > accuracy_req["horizontal"]:
                    can_achieve = False
                    level_factors.append(f"Horizontal accuracy insufficient: {features.avg_horizontal_accuracy:.1f}mm > {accuracy_req['horizontal']:.1f}mm")

            if "vertical" in accuracy_req:
                if features.avg_vertical_accuracy > accuracy_req["vertical"]:
                    can_achieve = False
                    level_factors.append(f"Vertical accuracy insufficient: {features.avg_vertical_accuracy:.1f}mm > {accuracy_req['vertical']:.1f}mm")

            # Check environmental threshold
            env_score = min(features.soil_suitability_score, features.weather_impact_score, features.site_constraints_score)
            if env_score < requirements["environmental_threshold"]:
                if conservative:
                    can_achieve = False
                    level_factors.append(f"Environmental conditions unsuitable for {quality_level}: {env_score:.2f} < {requirements['environmental_threshold']:.2f}")

            # Check deliverable requirements
            if features.deliverable_completeness < (requirements["min_deliverables"] / 6.0):  # Assuming max 6 deliverables
                can_achieve = False
                level_factors.append("Insufficient deliverables provided")

            # Check specific requirements for higher levels
            if quality_level in [QualityLevel.QL_B, QualityLevel.QL_A]:
                detection_methods_count = sum([features.has_electromagnetic, features.has_gpr])
                if detection_methods_count < requirements.get("min_detection_methods", 0):
                    can_achieve = False
                    level_factors.append("Insufficient detection methods")

            if quality_level == QualityLevel.QL_A:
                if not features.has_intrusive:
                    can_achieve = False
                    level_factors.append("No intrusive investigation methods")

                if features.detection_verification_rate < requirements.get("min_verification_rate", 0.8):
                    can_achieve = False
                    level_factors.append(f"Insufficient verification rate: {features.detection_verification_rate:.2f}")

            if can_achieve:
                assessed_level = quality_level
                confidence = self._calculate_rule_based_confidence(features, quality_level)
                break
            else:
                limiting_factors.extend(level_factors)

        # Generate methods and deliverables compliance
        target_requirements = self.quality_requirements[assessed_level]
        methods_compliance = {
            method: method in executed_methods
            for method in target_requirements["required_methods"]
        }

        provided_deliverables = {deliv.deliverable_type for deliv in survey_data.deliverables if deliv.provided}
        deliverables_compliance = {}  # This would need actual deliverable mapping

        accuracy_compliance = self._assess_accuracy_compliance_automated(features, assessed_level)

        # Generate recommendations
        recommendations = self._generate_automated_recommendations(
            features, assessed_level, limiting_factors, survey_data.target_quality_level
        )

        return QualityLevelAssessment(
            assessed_quality_level=assessed_level,
            confidence=confidence,
            methods_compliance=methods_compliance,
            deliverables_compliance=deliverables_compliance,
            accuracy_compliance=accuracy_compliance,
            limiting_factors=limiting_factors,
            recommendations=recommendations
        )

    def _calculate_rule_based_confidence(self, features: QualityLevelFeatures,
                                       assessed_level: QualityLevel) -> float:
        """Calculate confidence in rule-based assessment."""
        confidence_factors = []

        # Method execution quality
        confidence_factors.append(features.method_execution_quality)

        # Detection confidence
        confidence_factors.append(features.detection_confidence)

        # Accuracy confidence
        confidence_factors.append(features.accuracy_confidence)

        # Environmental suitability
        env_score = min(
            features.soil_suitability_score,
            features.weather_impact_score,
            features.site_constraints_score
        )
        confidence_factors.append(env_score)

        # Deliverable quality
        confidence_factors.append(features.deliverable_quality)

        # Coverage adequacy
        confidence_factors.append(features.survey_coverage)

        return np.mean(confidence_factors)

    def _assess_accuracy_compliance_automated(self, features: QualityLevelFeatures,
                                            assessed_level: QualityLevel) -> Dict[str, bool]:
        """Assess accuracy compliance for automated assessment."""
        requirements = self.quality_requirements[assessed_level]["accuracy_threshold"]

        compliance = {}
        if "horizontal" in requirements:
            compliance["horizontal"] = features.avg_horizontal_accuracy <= requirements["horizontal"]

        if "vertical" in requirements:
            compliance["vertical"] = features.avg_vertical_accuracy <= requirements["vertical"]

        return compliance

    def _generate_automated_recommendations(self, features: QualityLevelFeatures,
                                          assessed_level: QualityLevel,
                                          limiting_factors: List[str],
                                          target_level: QualityLevel) -> List[str]:
        """Generate automated recommendations for improvement."""
        recommendations = []

        # If not achieving target level, provide specific recommendations
        if assessed_level != target_level:
            target_index = self.quality_levels.index(target_level)
            current_index = self.quality_levels.index(assessed_level)

            if target_index > current_index:
                next_level = self.quality_levels[current_index + 1]
                next_requirements = self.quality_requirements[next_level]

                # Method recommendations
                if not features.has_electromagnetic and SurveyMethod.ELECTROMAGNETIC_DETECTION in next_requirements["required_methods"]:
                    recommendations.append("Deploy electromagnetic detection equipment (cable avoidance tools, precision locators)")

                if not features.has_gpr and SurveyMethod.GROUND_PENETRATING_RADAR in next_requirements["required_methods"]:
                    if features.soil_suitability_score < 0.5:
                        recommendations.append("GPR effectiveness limited by soil conditions - consider higher frequency antennas or alternative methods")
                    else:
                        recommendations.append("Deploy ground penetrating radar systems with appropriate antenna frequencies")

                if not features.has_intrusive and next_level == QualityLevel.QL_A:
                    recommendations.append("Implement intrusive investigation methods (trial holes, vacuum excavation)")

                # Accuracy recommendations
                accuracy_threshold = next_requirements["accuracy_threshold"]
                if "horizontal" in accuracy_threshold and features.avg_horizontal_accuracy > accuracy_threshold["horizontal"]:
                    recommendations.append(f"Improve horizontal accuracy to ±{accuracy_threshold['horizontal']:.0f}mm through better equipment calibration and survey methodology")

                if "vertical" in accuracy_threshold and features.avg_vertical_accuracy > accuracy_threshold["vertical"]:
                    recommendations.append(f"Improve vertical accuracy to ±{accuracy_threshold['vertical']:.0f}mm through depth verification methods")

        # Environmental recommendations
        if features.soil_suitability_score < 0.5:
            recommendations.append("Soil conditions limit GPR effectiveness - consider electromagnetic methods as primary detection")

        if features.weather_impact_score < 0.7:
            recommendations.append("Weather conditions impact survey quality - consider rescheduling for optimal conditions")

        if features.site_constraints_score < 0.7:
            recommendations.append("Site constraints limit survey effectiveness - develop mitigation strategies")

        # Coverage recommendations
        if features.survey_coverage < 0.8:
            recommendations.append("Increase survey coverage to ensure comprehensive utility detection")

        # Verification recommendations
        if features.detection_verification_rate < 0.8 and target_level == QualityLevel.QL_A:
            recommendations.append("Increase utility detection verification rate through intrusive investigation")

        return recommendations

    def train_ml_model(self, training_data: List[Tuple[SurveyData, QualityLevel]]) -> None:
        """
        Train machine learning model for quality level prediction.

        Args:
            training_data: List of (survey_data, actual_quality_level) pairs
        """
        self.logger.info(f"Training ML model with {len(training_data)} samples")

        # Extract features and labels
        features_list = []
        labels = []

        for survey_data, quality_level in training_data:
            features = self.extract_features(survey_data)
            features_list.append(features)
            labels.append(quality_level.value)

        # Convert features to numpy array
        feature_matrix = self._features_to_matrix(features_list)

        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Train Random Forest model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        self.ml_model.fit(feature_matrix, encoded_labels)
        self.feature_encoders['quality_level'] = label_encoder

        # Calculate feature importance
        feature_names = self._get_feature_names()
        feature_importance = dict(zip(feature_names, self.ml_model.feature_importances_))

        self.logger.info("ML model training completed")
        self.logger.info(f"Top 5 important features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

    def predict_quality_level_ml(self, survey_data: SurveyData) -> Tuple[QualityLevel, float]:
        """
        Predict quality level using trained machine learning model.

        Args:
            survey_data: Survey data to predict quality level for

        Returns:
            Predicted quality level and confidence score
        """
        if self.ml_model is None:
            raise ValueError("ML model not trained. Call train_ml_model() first.")

        features = self.extract_features(survey_data)
        feature_matrix = self._features_to_matrix([features])

        # Predict
        prediction = self.ml_model.predict(feature_matrix)[0]
        prediction_proba = self.ml_model.predict_proba(feature_matrix)[0]

        # Decode prediction
        predicted_level = QualityLevel(self.feature_encoders['quality_level'].inverse_transform([prediction])[0])
        confidence = np.max(prediction_proba)

        return predicted_level, confidence

    def _features_to_matrix(self, features_list: List[QualityLevelFeatures]) -> np.ndarray:
        """Convert features to matrix for ML processing."""
        matrix = []

        for features in features_list:
            row = [
                float(features.has_electromagnetic),
                float(features.has_gpr),
                float(features.has_intrusive),
                float(features.has_comprehensive_records),
                float(features.has_topographical),
                features.method_execution_quality,
                features.avg_horizontal_accuracy,
                features.avg_vertical_accuracy,
                features.accuracy_confidence,
                features.soil_suitability_score,
                features.weather_impact_score,
                features.site_constraints_score,
                features.detection_count,
                features.detection_confidence,
                features.detection_verification_rate,
                features.deliverable_completeness,
                features.deliverable_quality,
                features.survey_coverage,
                features.method_coverage_adequacy
            ]
            matrix.append(row)

        return np.array(matrix)

    def _get_feature_names(self) -> List[str]:
        """Get feature names for model interpretation."""
        return [
            "has_electromagnetic", "has_gpr", "has_intrusive", "has_comprehensive_records",
            "has_topographical", "method_execution_quality", "avg_horizontal_accuracy",
            "avg_vertical_accuracy", "accuracy_confidence", "soil_suitability_score",
            "weather_impact_score", "site_constraints_score", "detection_count",
            "detection_confidence", "detection_verification_rate", "deliverable_completeness",
            "deliverable_quality", "survey_coverage", "method_coverage_adequacy"
        ]

    def save_model(self, model_path: str) -> None:
        """Save trained model to file."""
        if self.ml_model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.ml_model,
            'encoders': self.feature_encoders,
            'feature_names': self._get_feature_names()
        }

        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(model_path)

        self.ml_model = model_data['model']
        self.feature_encoders = model_data['encoders']

        self.logger.info(f"Model loaded from {model_path}")

    def get_quality_level_probabilities(self, survey_data: SurveyData) -> Dict[QualityLevel, float]:
        """
        Get probability distribution over all quality levels.

        Args:
            survey_data: Survey data to assess

        Returns:
            Dictionary mapping quality levels to probabilities
        """
        if self.ml_model is None:
            raise ValueError("ML model not trained")

        features = self.extract_features(survey_data)
        feature_matrix = self._features_to_matrix([features])

        prediction_proba = self.ml_model.predict_proba(feature_matrix)[0]
        classes = self.feature_encoders['quality_level'].classes_

        probabilities = {}
        for i, class_name in enumerate(classes):
            probabilities[QualityLevel(class_name)] = prediction_proba[i]

        return probabilities