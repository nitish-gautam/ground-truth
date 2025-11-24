"""
GPR Testing Framework.

Comprehensive testing and validation framework for Ground Penetrating Radar (GPR)
signal processing systems using the University of Twente ground truth dataset.

This framework provides:

1. Ground Truth Validation - Using real University of Twente GPR dataset
2. Accuracy Assessment - Position, material, depth, and discipline validation
3. PAS 128 Compliance - Professional surveying standard compliance testing
4. Statistical Validation - Comprehensive statistical analysis framework
5. Environmental Factor Validation - Weather, ground, and terrain impact analysis
6. Performance Benchmarking - Real-time monitoring and regression testing
7. Comprehensive Reporting - Interactive reports with visualizations

Usage:
    from tests.validation import create_accuracy_assessor, create_pas128_validator
    from tests.utils.data_preparation import create_ground_truth_loader
    from tests.utils.reporting import create_validation_reporter

    # Load ground truth data
    loader = create_ground_truth_loader("path/to/Metadata.csv")
    locations = loader.load_data()

    # Assess accuracy
    assessor = create_accuracy_assessor()
    results = assessor.comprehensive_accuracy_assessment(detections, ground_truth)

    # Generate report
    reporter = create_validation_reporter("./reports")
    report = reporter.generate_comprehensive_report(results)
"""

__version__ = "1.0.0"
__author__ = "GPR Validation Framework Team"
__email__ = "support@gpr-validation.com"