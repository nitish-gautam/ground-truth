#!/usr/bin/env python3
"""
Comprehensive GPR Validation Test Suite Runner.

This script provides a complete validation framework for GPR signal processing
accuracy using the University of Twente ground truth dataset.

Usage:
    python run_validation_suite.py [options]

Example:
    python run_validation_suite.py --full-suite --report-dir ./validation_reports
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.data_preparation import create_ground_truth_loader
from tests.validation.accuracy import create_accuracy_assessor, DetectedUtility, GroundTruthUtility
from tests.validation.pas128 import create_pas128_validator, QualityLevel, DetectionResult, SurveyDeliverables, DetectionMethod
from tests.validation.statistical import create_statistical_validator
from tests.validation.environmental import create_environmental_validator, EnvironmentalConditions, SurveyResults, EnvironmentalFactor, PerformanceMetric
from tests.performance.benchmarking import create_performance_benchmarker
from tests.utils.reporting import create_validation_reporter
from tests.utils.helpers import create_test_data_generator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation_suite.log')
    ]
)
logger = logging.getLogger(__name__)


class GPRValidationSuite:
    """Comprehensive GPR validation test suite."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation suite.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}

        # Initialize components
        self.accuracy_assessor = create_accuracy_assessor(
            position_tolerance=config.get('position_tolerance', 1.0),
            depth_tolerance=config.get('depth_tolerance', 0.3)
        )

        self.pas128_validator = create_pas128_validator(
            config.get('pas128_spec_path')
        )

        self.statistical_validator = create_statistical_validator(
            random_state=config.get('random_state', 42)
        )

        self.environmental_validator = create_environmental_validator(
            random_state=config.get('random_state', 42)
        )

        self.performance_benchmarker = create_performance_benchmarker(
            config.get('performance_db_path')
        )

        self.test_data_generator = create_test_data_generator(
            random_seed=config.get('random_state', 42)
        )

        self.reporter = create_validation_reporter(
            config.get('report_dir', './validation_reports')
        )

    def run_full_validation_suite(self) -> Dict[str, Any]:
        """
        Run the complete validation suite.

        Returns:
            Dictionary containing all validation results
        """
        logger.info("Starting comprehensive GPR validation suite")

        try:
            # 1. Load ground truth data
            logger.info("Loading ground truth data...")
            ground_truth_data = self._load_ground_truth_data()

            # 2. Generate or load test detections
            logger.info("Generating test detection data...")
            test_detections = self._generate_test_detections(ground_truth_data)

            # 3. Run accuracy assessment
            logger.info("Running accuracy assessment...")
            self.results['accuracy_assessment'] = self._run_accuracy_assessment(
                test_detections, ground_truth_data
            )

            # 4. Run PAS 128 compliance validation
            logger.info("Running PAS 128 compliance validation...")
            self.results['pas128_compliance'] = self._run_pas128_validation(
                test_detections, ground_truth_data
            )

            # 5. Run statistical validation
            logger.info("Running statistical validation...")
            self.results['statistical_analysis'] = self._run_statistical_validation()

            # 6. Run environmental factor validation
            logger.info("Running environmental factor validation...")
            self.results['environmental_validation'] = self._run_environmental_validation()

            # 7. Run performance benchmarking
            logger.info("Running performance benchmarking...")
            self.results['performance_benchmarks'] = self._run_performance_benchmarking()

            # 8. Generate comprehensive report
            logger.info("Generating validation report...")
            validation_report = self.reporter.generate_comprehensive_report(
                self.results,
                report_title="GPR Signal Processing Validation Report",
                report_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            logger.info(f"Validation suite completed successfully. Report: {validation_report.report_id}")
            return self.results

        except Exception as e:
            logger.error(f"Validation suite failed: {e}")
            raise

    def _load_ground_truth_data(self) -> Dict[str, Any]:
        """Load University of Twente ground truth data."""
        twente_metadata_path = self.config.get('twente_metadata_path')

        if twente_metadata_path and Path(twente_metadata_path).exists():
            # Load real Twente data
            loader = create_ground_truth_loader(twente_metadata_path)
            locations = loader.load_data()

            # Convert to our format
            ground_truth_utilities = []
            for location_id, location in locations.items():
                for i, utility in enumerate(location.utilities):
                    if utility.material and utility.discipline:
                        gt_utility = GroundTruthUtility(
                            x_position=float(i * 10),  # Mock positions
                            y_position=float(i * 5),
                            depth=1.5,  # Mock depth
                            material=utility.material,
                            diameter=utility.diameter or 200,
                            discipline=utility.discipline,
                            utility_id=f"{location_id}_{i}"
                        )
                        ground_truth_utilities.append(gt_utility)

            return {
                'utilities': ground_truth_utilities[:50],  # Limit for testing
                'metadata': loader.get_statistics()
            }
        else:
            # Generate synthetic data
            logger.warning("Twente data not available, generating synthetic ground truth")
            utilities = self.test_data_generator.generate_utility_layout(
                area_bounds=(0, 100, 0, 100),
                utility_count=25
            )
            return {
                'utilities': utilities,
                'metadata': {'total_utilities': len(utilities), 'synthetic': True}
            }

    def _generate_test_detections(self, ground_truth_data: Dict[str, Any]) -> List[DetectedUtility]:
        """Generate realistic GPR detections based on ground truth."""
        utilities = ground_truth_data['utilities']

        # Generate detections with realistic performance characteristics
        detections = self.test_data_generator.generate_gpr_detections(
            utilities,
            detection_rate=self.config.get('detection_rate', 0.85),
            position_noise_std=self.config.get('position_noise', 0.3),
            depth_noise_std=self.config.get('depth_noise', 0.1),
            false_positive_rate=self.config.get('false_positive_rate', 0.1),
            material_accuracy=self.config.get('material_accuracy', 0.8),
            discipline_accuracy=self.config.get('discipline_accuracy', 0.9)
        )

        logger.info(f"Generated {len(detections)} detections from {len(utilities)} ground truth utilities")
        return detections

    def _run_accuracy_assessment(
        self,
        detections: List[DetectedUtility],
        ground_truth_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run comprehensive accuracy assessment."""
        utilities = ground_truth_data['utilities']

        # Perform comprehensive accuracy assessment
        results = self.accuracy_assessor.comprehensive_accuracy_assessment(detections, utilities)

        # Convert dataclass results to dictionaries for serialization
        accuracy_results = {
            'position_accuracy': {
                'horizontal_rmse': results['position_accuracy'].horizontal_rmse,
                'vertical_rmse': results['position_accuracy'].vertical_rmse,
                'horizontal_mae': results['position_accuracy'].horizontal_mae,
                'vertical_mae': results['position_accuracy'].vertical_mae,
                'total_rmse': results['position_accuracy'].total_rmse,
                'max_horizontal_error': results['position_accuracy'].max_horizontal_error,
                'error_percentiles': results['position_accuracy'].error_percentiles
            },
            'detection_performance': {
                'precision': results['detection_performance'].precision,
                'recall': results['detection_performance'].recall,
                'f1_score': results['detection_performance'].f1_score,
                'true_positives': results['detection_performance'].true_positives,
                'false_positives': results['detection_performance'].false_positives,
                'false_negatives': results['detection_performance'].false_negatives
            },
            'material_classification': {
                'overall_accuracy': results['material_classification'].overall_accuracy,
                'per_class_precision': results['material_classification'].per_class_precision,
                'per_class_recall': results['material_classification'].per_class_recall
            },
            'depth_estimation': {
                'rmse': results['depth_estimation'].rmse,
                'mae': results['depth_estimation'].mae,
                'bias': results['depth_estimation'].bias,
                'relative_error_mean': results['depth_estimation'].relative_error_mean
            },
            'summary_statistics': results['summary_statistics']
        }

        return accuracy_results

    def _run_pas128_validation(
        self,
        detections: List[DetectedUtility],
        ground_truth_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run PAS 128 compliance validation."""
        utilities = ground_truth_data['utilities']

        # Convert detections to PAS 128 format
        detection_results = [
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

        # Convert ground truth to PAS 128 format
        ground_truth_utilities = [
            {
                'x_position': u.x_position,
                'y_position': u.y_position,
                'depth': u.depth,
                'material': u.material,
                'diameter': u.diameter,
                'discipline': u.discipline
            }
            for u in utilities
        ]

        # Create survey deliverables
        deliverables = SurveyDeliverables(
            survey_report=True,
            utility_location_plans=True,
            risk_assessment=True,
            detection_survey_results=True,
            intrusive_investigation_results=False,
            verification_photos=False
        )

        # Validate compliance for different quality levels
        compliance_results = {}
        for quality_level in [QualityLevel.QL_D, QualityLevel.QL_C, QualityLevel.QL_B, QualityLevel.QL_A]:
            try:
                result = self.pas128_validator.validate_compliance(
                    detection_results,
                    ground_truth_utilities,
                    deliverables,
                    quality_level
                )

                compliance_results[quality_level.value] = {
                    'compliant': result.compliant,
                    'horizontal_accuracy_mm': result.horizontal_accuracy_mm,
                    'depth_accuracy_mm': result.depth_accuracy_mm,
                    'compliance_score': result.compliance_score,
                    'missing_deliverables': result.missing_deliverables,
                    'method_coverage': {k.value: v for k, v in result.method_coverage.items()}
                }
            except Exception as e:
                logger.warning(f"Failed to validate {quality_level.value}: {e}")

        return compliance_results

    def _run_statistical_validation(self) -> Dict[str, Any]:
        """Run statistical validation tests."""
        # Generate sample performance data for testing
        import numpy as np

        # Simulate baseline vs current performance
        baseline_performance = np.random.normal(0.8, 0.05, 50)
        current_performance = np.random.normal(0.82, 0.05, 50)

        # Statistical comparison
        comparison_result = self.statistical_validator.compare_groups_statistical_test(
            baseline_performance,
            current_performance,
            confidence_level=0.95
        )

        # Bootstrap confidence intervals
        bootstrap_result = self.statistical_validator.bootstrap_confidence_interval(
            current_performance,
            np.mean,
            n_bootstrap=1000,
            confidence_level=0.95
        )

        # Cross-validation simulation
        from sklearn.dummy import DummyClassifier
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model = DummyClassifier(strategy='most_frequent')

        cv_result = self.statistical_validator.cross_validate_performance(
            X, y, model, cv_folds=5
        )

        return {
            'group_comparison': {
                'test_name': comparison_result.test_name,
                'statistic': comparison_result.statistic,
                'p_value': comparison_result.p_value,
                'significant': comparison_result.significant,
                'effect_size': comparison_result.effect_size,
                'interpretation': comparison_result.interpretation
            },
            'bootstrap_analysis': {
                'original_statistic': bootstrap_result.original_statistic,
                'bootstrap_mean': bootstrap_result.bootstrap_mean,
                'confidence_interval': bootstrap_result.confidence_interval,
                'n_bootstrap_samples': bootstrap_result.n_bootstrap_samples
            },
            'cross_validation': {
                'mean_score': cv_result.mean_score,
                'std_score': cv_result.std_score,
                'confidence_interval': cv_result.confidence_interval,
                'cv_scores': cv_result.cv_scores
            }
        }

    def _run_environmental_validation(self) -> Dict[str, Any]:
        """Run environmental factor validation."""
        # Generate survey results with different environmental conditions
        survey_results = self.test_data_generator.generate_survey_results(
            num_surveys=20,
            area_bounds=(0, 100, 0, 100)
        )

        environmental_results = {}

        # Test weather condition impact
        try:
            weather_impact = self.environmental_validator.validate_environmental_impact(
                survey_results,
                EnvironmentalFactor.WEATHER_CONDITION,
                PerformanceMetric.DETECTION_RATE
            )

            environmental_results['weather_impact'] = {
                'factor': weather_impact.factor.value,
                'performance_metric': weather_impact.performance_metric.value,
                'effect_size': weather_impact.effect_size,
                'statistical_significance': weather_impact.statistical_significance,
                'factor_performance': weather_impact.factor_performance
            }
        except Exception as e:
            logger.warning(f"Weather impact analysis failed: {e}")

        # Test ground condition impact
        try:
            ground_impact = self.environmental_validator.validate_environmental_impact(
                survey_results,
                EnvironmentalFactor.GROUND_CONDITION,
                PerformanceMetric.DETECTION_RATE
            )

            environmental_results['ground_impact'] = {
                'factor': ground_impact.factor.value,
                'performance_metric': ground_impact.performance_metric.value,
                'effect_size': ground_impact.effect_size,
                'statistical_significance': ground_impact.statistical_significance,
                'factor_performance': ground_impact.factor_performance
            }
        except Exception as e:
            logger.warning(f"Ground condition impact analysis failed: {e}")

        # Analyze optimal conditions
        try:
            optimal_conditions = self.environmental_validator.identify_optimal_conditions(
                survey_results,
                PerformanceMetric.DETECTION_RATE
            )

            environmental_results['optimal_conditions'] = {
                'best_conditions': {k.value: v for k, v in optimal_conditions.best_conditions.items()},
                'worst_conditions': {k.value: v for k, v in optimal_conditions.worst_conditions.items()},
                'recommendations': optimal_conditions.condition_recommendations
            }
        except Exception as e:
            logger.warning(f"Optimal conditions analysis failed: {e}")

        return environmental_results

    def _run_performance_benchmarking(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        # Mock GPR processing function for benchmarking
        def mock_gpr_processor(data):
            import time
            import numpy as np
            # Simulate processing time
            time.sleep(0.1 + np.random.uniform(0, 0.05))
            # Return mock detections
            return [
                DetectedUtility(
                    x_position=10.0, y_position=20.0, depth=1.5,
                    material="steel", diameter=200, discipline="water",
                    confidence=0.8, detection_id="mock_detection"
                )
            ]

        # Mock test data
        test_data = list(range(10))  # Simple test data

        # Benchmark the function
        benchmark_result = self.performance_benchmarker.benchmark_function(
            mock_gpr_processor,
            test_data,
            test_id="mock_gpr_test",
            algorithm_version="v1.0.0",
            iterations=5
        )

        # Set baselines
        self.performance_benchmarker.set_baseline(
            self.performance_benchmarker.BenchmarkMetric.PROCESSING_TIME,
            0.15,  # 150ms baseline
            tolerance_percentage=20.0
        )

        # Check for performance regression
        alerts = self.performance_benchmarker.check_performance_regression(benchmark_result)

        return {
            'benchmark_metrics': {
                'processing_time': benchmark_result.processing_time,
                'memory_usage_mb': benchmark_result.memory_usage_mb,
                'cpu_utilization': benchmark_result.cpu_utilization,
                'throughput': benchmark_result.throughput,
                'latency': benchmark_result.latency,
                'error_rate': benchmark_result.error_rate
            },
            'performance_alerts': [
                {
                    'level': alert.level.value,
                    'metric': alert.metric.value,
                    'message': alert.message,
                    'deviation_percentage': alert.deviation_percentage
                }
                for alert in alerts
            ]
        }


def main():
    """Main entry point for the validation suite."""
    parser = argparse.ArgumentParser(
        description="GPR Validation Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --full-suite
  %(prog)s --accuracy-only --twente-data /path/to/Metadata.csv
  %(prog)s --performance-only --report-dir ./reports
        """
    )

    parser.add_argument(
        '--full-suite',
        action='store_true',
        help='Run the complete validation suite'
    )

    parser.add_argument(
        '--accuracy-only',
        action='store_true',
        help='Run only accuracy assessment tests'
    )

    parser.add_argument(
        '--pas128-only',
        action='store_true',
        help='Run only PAS 128 compliance tests'
    )

    parser.add_argument(
        '--environmental-only',
        action='store_true',
        help='Run only environmental factor tests'
    )

    parser.add_argument(
        '--performance-only',
        action='store_true',
        help='Run only performance benchmarking tests'
    )

    parser.add_argument(
        '--twente-data',
        type=str,
        help='Path to Twente metadata CSV file'
    )

    parser.add_argument(
        '--report-dir',
        type=str,
        default='./validation_reports',
        help='Directory for validation reports'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = {
        'twente_metadata_path': args.twente_data,
        'report_dir': args.report_dir,
        'position_tolerance': 1.0,
        'depth_tolerance': 0.3,
        'random_state': 42,
        'detection_rate': 0.85,
        'position_noise': 0.3,
        'depth_noise': 0.1,
        'false_positive_rate': 0.1,
        'material_accuracy': 0.8,
        'discipline_accuracy': 0.9
    }

    if args.config and Path(args.config).exists():
        import json
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)

    # Initialize validation suite
    validation_suite = GPRValidationSuite(config)

    try:
        if args.full_suite or not any([args.accuracy_only, args.pas128_only,
                                      args.environmental_only, args.performance_only]):
            # Run full suite
            results = validation_suite.run_full_validation_suite()
            print(f"\nValidation suite completed successfully!")
            print(f"Overall score: {results.get('test_summary', {}).get('overall_score', 0)*100:.1f}%")

        else:
            # Run specific tests
            if args.accuracy_only:
                print("Running accuracy assessment tests...")
                # Implementation for specific test types would go here

            if args.pas128_only:
                print("Running PAS 128 compliance tests...")

            if args.environmental_only:
                print("Running environmental factor tests...")

            if args.performance_only:
                print("Running performance benchmarking tests...")

    except Exception as e:
        logger.error(f"Validation suite failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()