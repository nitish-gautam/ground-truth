"""
Comprehensive ground truth validation tests using University of Twente dataset.

This test suite demonstrates the complete validation framework using real ground truth data
from the University of Twente GPR dataset.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from tests.utils.data_preparation import TwenteDataLoader, create_ground_truth_loader
from tests.validation.accuracy import AccuracyAssessor, DetectedUtility, GroundTruthUtility, create_accuracy_assessor
from tests.validation.pas128 import PAS128ComplianceValidator, QualityLevel, DetectionResult, SurveyDeliverables, create_pas128_validator
from tests.validation.statistical import StatisticalValidator, create_statistical_validator
from tests.validation.environmental import EnvironmentalValidator, EnvironmentalConditions, SurveyResults, create_environmental_validator


@pytest.mark.validation
@pytest.mark.ground_truth
class TestTwenteGroundTruthValidation:
    """Test ground truth validation using Twente dataset."""

    def setup_method(self):
        """Setup test fixtures."""
        # Use test fixture paths
        self.test_data_dir = Path(__file__).parent.parent.parent / "fixtures" / "data"
        self.twente_metadata_path = Path(__file__).parent.parent.parent.parent / "datasets" / "raw" / "twente_gpr" / "Metadata.csv"

        # Initialize components
        self.accuracy_assessor = create_accuracy_assessor()
        self.pas128_validator = create_pas128_validator()
        self.statistical_validator = create_statistical_validator()
        self.environmental_validator = create_environmental_validator()

    @pytest.mark.slow
    def test_load_twente_ground_truth_data(self, twente_metadata_path):
        """Test loading and parsing of Twente ground truth data."""
        if not twente_metadata_path.exists():
            pytest.skip("Twente metadata file not available")

        loader = create_ground_truth_loader(str(twente_metadata_path))
        locations = loader.load_data()

        # Validate data loading
        assert len(locations) > 0, "Should load at least one location"

        # Check data structure
        first_location = next(iter(locations.values()))
        assert first_location.location_id is not None
        assert first_location.environmental_conditions is not None
        assert first_location.survey_metadata is not None

        # Validate statistics
        stats = loader.get_statistics()
        assert stats['total_locations'] > 0
        assert 'weather_conditions' in stats
        assert 'ground_conditions' in stats

    def test_position_accuracy_assessment(self, sample_gpr_detection, sample_ground_truth):
        """Test position accuracy assessment."""
        # Convert sample data to required format
        detections = self._convert_to_detected_utilities(sample_gpr_detection['detected_utilities'])
        ground_truth = self._convert_to_ground_truth_utilities(sample_ground_truth['true_utilities'])

        # Assess position accuracy
        position_accuracy = self.accuracy_assessor.assess_position_accuracy(detections, ground_truth)

        # Validate results
        assert position_accuracy.horizontal_rmse >= 0
        assert position_accuracy.vertical_rmse >= 0
        assert position_accuracy.horizontal_mae >= 0
        assert position_accuracy.vertical_mae >= 0
        assert len(position_accuracy.error_percentiles) > 0

    def test_material_classification_accuracy(self, sample_gpr_detection, sample_ground_truth):
        """Test material classification accuracy assessment."""
        detections = self._convert_to_detected_utilities(sample_gpr_detection['detected_utilities'])
        ground_truth = self._convert_to_ground_truth_utilities(sample_ground_truth['true_utilities'])

        # Assess material classification
        material_accuracy = self.accuracy_assessor.assess_material_classification_accuracy(detections, ground_truth)

        # Validate results
        assert 0 <= material_accuracy.overall_accuracy <= 1
        assert material_accuracy.confusion_matrix is not None
        assert isinstance(material_accuracy.per_class_precision, dict)

    def test_depth_estimation_accuracy(self, sample_gpr_detection, sample_ground_truth):
        """Test depth estimation accuracy assessment."""
        detections = self._convert_to_detected_utilities(sample_gpr_detection['detected_utilities'])
        ground_truth = self._convert_to_ground_truth_utilities(sample_ground_truth['true_utilities'])

        # Assess depth estimation
        depth_accuracy = self.accuracy_assessor.assess_depth_estimation_accuracy(detections, ground_truth)

        # Validate results
        assert depth_accuracy.rmse >= 0
        assert depth_accuracy.mae >= 0
        assert isinstance(depth_accuracy.depth_error_percentiles, dict)

    def test_pas128_compliance_validation(self, sample_gpr_detection, sample_ground_truth):
        """Test PAS 128 compliance validation."""
        # Convert to PAS 128 format
        detection_results = self._convert_to_detection_results(sample_gpr_detection['detected_utilities'])
        ground_truth_utilities = sample_ground_truth['true_utilities']

        # Create survey deliverables
        deliverables = SurveyDeliverables(
            survey_report=True,
            utility_location_plans=True,
            risk_assessment=True,
            detection_survey_results=True
        )

        # Validate compliance for QL-B
        compliance_result = self.pas128_validator.validate_compliance(
            detection_results,
            ground_truth_utilities,
            deliverables,
            QualityLevel.QL_B
        )

        # Validate results
        assert isinstance(compliance_result.compliant, bool)
        assert compliance_result.horizontal_accuracy_mm >= 0
        assert 0 <= compliance_result.compliance_score <= 1
        assert isinstance(compliance_result.method_coverage, dict)

    def test_environmental_factor_validation(self, environmental_test_conditions):
        """Test environmental factor impact validation."""
        # Create mock survey results
        survey_results = []

        for i, condition in enumerate(environmental_test_conditions):
            env_conditions = EnvironmentalConditions(
                weather_condition=condition['weather'],
                ground_condition=condition['ground_condition'],
                ground_permittivity=condition['permittivity'],
                land_cover="Test surface",
                land_use="Test area",
                terrain_levelling="Flat",
                terrain_smoothness="Smooth"
            )

            # Create mock detections and ground truth
            detections = [
                DetectedUtility(
                    x_position=10.0 + i,
                    y_position=20.0 + i,
                    depth=1.5,
                    material="steel",
                    diameter=200,
                    discipline="water",
                    confidence=0.8 + condition.get('expected_performance_bonus', 0.0),
                    detection_id=f"det_{i}"
                )
            ]

            ground_truth = [
                GroundTruthUtility(
                    x_position=10.0 + i,
                    y_position=20.0 + i,
                    depth=1.5,
                    material="steel",
                    diameter=200,
                    discipline="water",
                    utility_id=f"gt_{i}"
                )
            ]

            survey_result = SurveyResults(
                location_id=f"test_location_{i}",
                environmental_conditions=env_conditions,
                detections=detections,
                ground_truth=ground_truth
            )
            survey_results.append(survey_result)

        # Test environmental impact validation
        if len(survey_results) >= 2:
            from tests.validation.environmental import EnvironmentalFactor, PerformanceMetric

            impact_analysis = self.environmental_validator.validate_environmental_impact(
                survey_results,
                EnvironmentalFactor.WEATHER_CONDITION,
                PerformanceMetric.DETECTION_RATE
            )

            # Validate results
            assert impact_analysis.factor == EnvironmentalFactor.WEATHER_CONDITION
            assert impact_analysis.performance_metric == PerformanceMetric.DETECTION_RATE
            assert isinstance(impact_analysis.statistical_significance, dict)
            assert isinstance(impact_analysis.factor_performance, dict)

    def test_statistical_validation_framework(self):
        """Test statistical validation capabilities."""
        # Generate test data
        group1 = np.random.normal(0.8, 0.1, 30)  # High performance group
        group2 = np.random.normal(0.6, 0.1, 30)  # Lower performance group

        # Test statistical comparison
        test_result = self.statistical_validator.compare_groups_statistical_test(
            group1, group2,
            confidence_level=0.95
        )

        # Validate results
        assert test_result.test_name is not None
        assert test_result.p_value >= 0
        assert isinstance(test_result.significant, bool)

        # Test bootstrap confidence intervals
        bootstrap_result = self.statistical_validator.bootstrap_confidence_interval(
            group1, np.mean, n_bootstrap=100
        )

        # Validate results
        assert bootstrap_result.original_statistic > 0
        assert len(bootstrap_result.confidence_interval) == 2
        assert bootstrap_result.confidence_interval[0] < bootstrap_result.confidence_interval[1]

    def test_comprehensive_accuracy_assessment(self, sample_gpr_detection, sample_ground_truth):
        """Test comprehensive accuracy assessment across all metrics."""
        detections = self._convert_to_detected_utilities(sample_gpr_detection['detected_utilities'])
        ground_truth = self._convert_to_ground_truth_utilities(sample_ground_truth['true_utilities'])

        # Perform comprehensive assessment
        results = self.accuracy_assessor.comprehensive_accuracy_assessment(detections, ground_truth)

        # Validate all components are present
        assert 'position_accuracy' in results
        assert 'material_classification' in results
        assert 'depth_estimation' in results
        assert 'discipline_classification' in results
        assert 'detection_performance' in results
        assert 'summary_statistics' in results

        # Validate summary statistics
        summary = results['summary_statistics']
        assert summary['total_detections'] == len(detections)
        assert summary['total_ground_truth'] == len(ground_truth)
        assert 0 <= summary['match_rate'] <= 1

    def _convert_to_detected_utilities(self, detected_utilities: List[Dict]) -> List[DetectedUtility]:
        """Convert sample detection data to DetectedUtility objects."""
        utilities = []
        for i, detection in enumerate(detected_utilities):
            utility = DetectedUtility(
                x_position=detection['x_position'],
                y_position=detection['y_position'],
                depth=detection.get('depth'),
                material=detection.get('material'),
                diameter=detection.get('diameter'),
                discipline=detection.get('discipline'),
                confidence=detection.get('confidence', 0.8),
                detection_id=f"detection_{i}"
            )
            utilities.append(utility)
        return utilities

    def _convert_to_ground_truth_utilities(self, ground_truth_utilities: List[Dict]) -> List[GroundTruthUtility]:
        """Convert sample ground truth data to GroundTruthUtility objects."""
        utilities = []
        for i, gt in enumerate(ground_truth_utilities):
            utility = GroundTruthUtility(
                x_position=gt['x_position'],
                y_position=gt['y_position'],
                depth=gt['depth'],
                material=gt['material'],
                diameter=gt['diameter'],
                discipline=gt['discipline'],
                utility_id=f"ground_truth_{i}"
            )
            utilities.append(utility)
        return utilities

    def _convert_to_detection_results(self, detected_utilities: List[Dict]) -> List[DetectionResult]:
        """Convert sample detection data to DetectionResult objects for PAS 128 validation."""
        from tests.validation.pas128 import DetectionMethod

        results = []
        for detection in detected_utilities:
            result = DetectionResult(
                x_position=detection['x_position'],
                y_position=detection['y_position'],
                depth=detection.get('depth'),
                material=detection.get('material'),
                diameter=detection.get('diameter'),
                discipline=detection.get('discipline'),
                confidence=detection.get('confidence', 0.8),
                detection_method=DetectionMethod.GROUND_PENETRATING_RADAR
            )
            results.append(result)
        return results


@pytest.mark.validation
@pytest.mark.integration
class TestIntegratedValidationWorkflow:
    """Test integrated validation workflow."""

    def setup_method(self):
        """Setup test fixtures."""
        self.accuracy_assessor = create_accuracy_assessor()
        self.pas128_validator = create_pas128_validator()
        self.statistical_validator = create_statistical_validator()

    def test_end_to_end_validation_workflow(self, sample_gpr_detection, sample_ground_truth):
        """Test complete end-to-end validation workflow."""
        # Step 1: Load and prepare data
        detections = self._convert_sample_data(sample_gpr_detection['detected_utilities'])
        ground_truth = self._convert_sample_ground_truth(sample_ground_truth['true_utilities'])

        # Step 2: Accuracy assessment
        accuracy_results = self.accuracy_assessor.comprehensive_accuracy_assessment(detections, ground_truth)

        # Step 3: PAS 128 compliance check
        detection_results = self._to_detection_results(detections)
        deliverables = self._create_sample_deliverables()

        compliance_result = self.pas128_validator.validate_compliance(
            detection_results, sample_ground_truth['true_utilities'], deliverables
        )

        # Step 4: Statistical validation
        # Create multiple test runs for statistical analysis
        performance_scores = [accuracy_results['detection_performance'].f1_score + np.random.normal(0, 0.05) for _ in range(20)]
        baseline_scores = [0.8 + np.random.normal(0, 0.05) for _ in range(20)]

        statistical_result = self.statistical_validator.compare_groups_statistical_test(
            np.array(performance_scores), np.array(baseline_scores)
        )

        # Validate integrated results
        assert accuracy_results is not None
        assert compliance_result is not None
        assert statistical_result is not None

        # Create comprehensive report
        report = {
            'accuracy_assessment': {
                'position_accuracy': accuracy_results['position_accuracy'].horizontal_rmse,
                'detection_performance': accuracy_results['detection_performance'].f1_score,
                'material_accuracy': accuracy_results['material_classification'].overall_accuracy
            },
            'pas128_compliance': {
                'compliant': compliance_result.compliant,
                'quality_level': compliance_result.achieved_quality_level.value if compliance_result.achieved_quality_level else None,
                'compliance_score': compliance_result.compliance_score
            },
            'statistical_validation': {
                'significant_difference': statistical_result.significant,
                'p_value': statistical_result.p_value,
                'effect_size': statistical_result.effect_size
            }
        }

        # Validate report structure
        assert 'accuracy_assessment' in report
        assert 'pas128_compliance' in report
        assert 'statistical_validation' in report

    def _convert_sample_data(self, detected_utilities):
        """Convert sample data for testing."""
        return [
            DetectedUtility(
                x_position=util['x_position'],
                y_position=util['y_position'],
                depth=util.get('depth'),
                material=util.get('material'),
                diameter=util.get('diameter'),
                discipline=util.get('discipline'),
                confidence=util.get('confidence', 0.8),
                detection_id=f"det_{i}"
            )
            for i, util in enumerate(detected_utilities)
        ]

    def _convert_sample_ground_truth(self, ground_truth_utilities):
        """Convert sample ground truth for testing."""
        return [
            GroundTruthUtility(
                x_position=util['x_position'],
                y_position=util['y_position'],
                depth=util['depth'],
                material=util['material'],
                diameter=util['diameter'],
                discipline=util['discipline'],
                utility_id=f"gt_{i}"
            )
            for i, util in enumerate(ground_truth_utilities)
        ]

    def _to_detection_results(self, detections):
        """Convert DetectedUtility to DetectionResult."""
        from tests.validation.pas128 import DetectionMethod

        return [
            DetectionResult(
                x_position=d.x_position,
                y_position=d.y_position,
                depth=d.depth,
                material=d.material,
                diameter=d.diameter,
                discipline=d.discipline,
                confidence=d.confidence,
                detection_method=DetectionMethod.GROUND_PENETRATING_RADAR
            )
            for d in detections
        ]

    def _create_sample_deliverables(self):
        """Create sample survey deliverables."""
        return SurveyDeliverables(
            survey_report=True,
            utility_location_plans=True,
            risk_assessment=True,
            detection_survey_results=True
        )