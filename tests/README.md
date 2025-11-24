# GPR Validation Testing Framework

## Overview

This comprehensive testing framework provides validation capabilities for Ground Penetrating Radar (GPR) signal processing accuracy using real ground truth data from the University of Twente GPR dataset. The framework implements professional surveying standards and regulatory compliance requirements.

## Key Features

### 🎯 **Ground Truth Validation**
- **University of Twente Dataset Integration**: Load and validate against 125 real GPR surveys with confirmed utility locations
- **Comprehensive Data Parser**: Handle multi-line utility data with material, diameter, depth, and discipline information
- **Environmental Context**: Weather conditions, ground types, terrain characteristics, and land use data

### 📏 **Accuracy Assessment Framework**
- **Position Accuracy**: Horizontal and vertical RMSE, MAE, bias calculations with percentile error analysis
- **Material Classification**: Confusion matrix analysis, per-class precision/recall, misclassification tracking
- **Depth Estimation**: RMSE, MAE, bias, relative error statistics with depth-specific analysis
- **Discipline Classification**: Multi-class utility type classification accuracy assessment
- **Detection Performance**: Precision, recall, F1-score, detection rate, false positive analysis

### 🏅 **PAS 128 Compliance Testing**
- **Quality Level Validation**: QL-A (±300mm), QL-B (±500mm), QL-C (±1000mm), QL-D (±2000mm)
- **Method Coverage**: Electromagnetic detection, GPR, intrusive investigation validation
- **Deliverables Compliance**: Survey reports, location plans, risk assessments, verification photos
- **Automated Compliance Scoring**: Overall compliance percentage with detailed breakdown

### 📊 **Statistical Validation Framework**
- **Cross-Validation**: K-fold and stratified cross-validation with confidence intervals
- **Bootstrap Analysis**: Confidence interval estimation for performance metrics
- **Hypothesis Testing**: T-tests, Mann-Whitney U, Kruskal-Wallis, Chi-square tests
- **Effect Size Analysis**: Cohen's d, Glass's delta, Hedge's g, eta-squared calculations
- **Correlation Analysis**: Pearson, Spearman, Kendall correlation with p-value matrices

### 🌍 **Environmental Factor Validation**
- **Weather Impact**: Dry vs Rainy condition performance analysis
- **Ground Condition Effects**: Sandy vs Clayey soil detection accuracy
- **Permittivity Analysis**: Ground relative permittivity vs detection performance
- **Terrain Characteristics**: Levelling, smoothness, cover type impact assessment
- **Interference Factors**: Rubble, tree roots, polluted soil, slag presence effects
- **Optimal Conditions**: ML-based recommendation system for survey planning

### ⚡ **Performance Benchmarking & Monitoring**
- **Real-time Monitoring**: CPU, memory, processing time tracking
- **Baseline Management**: Configurable performance baselines with tolerance levels
- **Regression Testing**: Automated detection of performance degradation
- **A/B Testing**: Statistical comparison of algorithm variants
- **Alert System**: Critical, warning, info level performance alerts
- **Trend Analysis**: Historical performance trend identification

### 📈 **Comprehensive Reporting**
- **Interactive HTML Reports**: Professional validation reports with visualizations
- **Statistical Dashboards**: Plotly-based interactive performance dashboards
- **PDF Summaries**: Executive summary reports for stakeholders
- **JSON Export**: Machine-readable results for programmatic access
- **Visualization Suite**: Matplotlib/Seaborn charts for all validation metrics

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install -r tests/requirements.txt

# Ensure University of Twente data is available
# Place Metadata.csv in datasets/raw/twente_gpr/
```

### Running the Complete Validation Suite

```bash
# Run full validation suite with Twente data
python tests/run_validation_suite.py --full-suite --twente-data datasets/raw/twente_gpr/Metadata.csv

# Generate synthetic data if Twente data unavailable
python tests/run_validation_suite.py --full-suite --report-dir ./validation_reports

# Run specific validation components
python tests/run_validation_suite.py --accuracy-only
python tests/run_validation_suite.py --pas128-only
python tests/run_validation_suite.py --environmental-only
python tests/run_validation_suite.py --performance-only
```

### Using Individual Components

```python
from tests.utils.data_preparation import create_ground_truth_loader
from tests.validation.accuracy import create_accuracy_assessor
from tests.validation.pas128 import create_pas128_validator

# Load ground truth data
loader = create_ground_truth_loader("path/to/Metadata.csv")
locations = loader.load_data()

# Assess detection accuracy
assessor = create_accuracy_assessor()
results = assessor.comprehensive_accuracy_assessment(detections, ground_truth)

# Validate PAS 128 compliance
validator = create_pas128_validator()
compliance = validator.validate_compliance(detections, ground_truth, deliverables)
```

## Framework Architecture

```
tests/
├── conftest.py                    # Global pytest configuration
├── pytest.ini                    # Test execution configuration
├── requirements.txt               # Framework dependencies
├── run_validation_suite.py       # Main test runner script
│
├── validation/                    # Core validation modules
│   ├── ground_truth/             # Twente dataset validation tests
│   ├── accuracy/                 # Position, material, depth accuracy
│   ├── pas128/                   # PAS 128 compliance validation
│   ├── statistical/              # Statistical analysis framework
│   └── environmental/            # Environmental factor validation
│
├── performance/                   # Performance testing
│   ├── benchmarking/             # Performance benchmarking
│   ├── monitoring/               # Real-time monitoring
│   └── regression/               # Regression testing
│
├── utils/                        # Supporting utilities
│   ├── data_preparation/         # Ground truth data loading
│   ├── reporting/                # Comprehensive reporting
│   └── helpers/                  # Test data generation
│
├── unit/                         # Unit tests
├── integration/                  # Integration tests
└── fixtures/                     # Test data fixtures
```

## Key Validation Metrics

### Position Accuracy Metrics
- **Horizontal RMSE**: Root mean square error in horizontal position (mm)
- **Vertical RMSE**: Root mean square error in depth estimation (mm)
- **Mean Absolute Error**: Average absolute position errors
- **Bias Analysis**: Systematic error detection
- **Percentile Analysis**: 50th, 75th, 90th, 95th, 99th percentile errors

### Detection Performance Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Detection Rate**: Percentage of utilities successfully detected
- **False Positive Rate**: Rate of incorrect detections

### PAS 128 Quality Levels
- **QL-A**: ±300mm horizontal, ±300mm depth (requires intrusive verification)
- **QL-B**: ±500mm horizontal (requires detection equipment)
- **QL-C**: ±1000mm horizontal (comprehensive records + reconnaissance)
- **QL-D**: ±2000mm horizontal (desk study only)

### Environmental Impact Factors
- **Weather Conditions**: Dry vs Rainy performance impact
- **Ground Conditions**: Sandy vs Clayey soil effects
- **Ground Permittivity**: Electrical properties impact on signal penetration
- **Terrain Features**: Levelling, smoothness, surface cover effects
- **Interference Sources**: Rubble, vegetation, contamination impact

## Test Execution

### Pytest Commands

```bash
# Run all validation tests
pytest tests/ -v

# Run specific test categories
pytest tests/validation/ -m validation
pytest tests/validation/ -m pas128
pytest tests/validation/ -m environmental
pytest tests/performance/ -m performance

# Run with coverage reporting
pytest tests/ --cov=backend/app --cov-report=html

# Run performance benchmarks
pytest tests/performance/ -m benchmark

# Run slow tests (comprehensive validation)
pytest tests/ -m slow

# Generate HTML test report
pytest tests/ --html=test_report.html --self-contained-html
```

### Configuration Options

```python
# pytest.ini configuration
[tool:pytest]
markers =
    validation: Ground truth validation tests
    pas128: PAS 128 compliance tests
    environmental: Environmental factor tests
    statistical: Statistical analysis tests
    accuracy: Accuracy assessment tests
    performance: Performance benchmarking tests
    slow: Tests that take a long time to run
```

## Ground Truth Data Integration

### University of Twente Dataset

The framework integrates with the comprehensive University of Twente GPR dataset:

- **125 Real GPR Surveys**: Confirmed utility locations with verified characteristics
- **Multi-disciplinary Coverage**: Water, sewer, electricity, telecommunications, gas utilities
- **Material Diversity**: Steel, PVC, asbestos cement, HDPE, concrete, cast iron
- **Environmental Variety**: Different weather, ground, terrain, and land use conditions
- **Survey Context**: Various accuracy requirements and construction scenarios

### Data Loading Example

```python
from tests.utils.data_preparation import create_ground_truth_loader

# Load Twente dataset
loader = create_ground_truth_loader("datasets/raw/twente_gpr/Metadata.csv")
locations = loader.load_data()

# Get specific location
location = loader.get_location("01.1")
print(f"Location has {location.utility_count} utilities")
print(f"Environmental conditions: {location.environmental_conditions.weather}")

# Filter by conditions
dry_sandy_locations = loader.get_locations_by_criteria(
    weather="Dry",
    ground_condition="Sandy"
)

# Export processed data
loader.export_processed_data("processed_ground_truth.json")
```

## Statistical Analysis Capabilities

### Hypothesis Testing

```python
from tests.validation.statistical import create_statistical_validator

validator = create_statistical_validator()

# Compare detection performance between conditions
test_result = validator.compare_groups_statistical_test(
    dry_performance, rainy_performance,
    test_type=StatisticalTest.MANN_WHITNEY,
    confidence_level=0.95
)

print(f"P-value: {test_result.p_value}")
print(f"Significant: {test_result.significant}")
print(f"Effect size: {test_result.effect_size}")
```

### Bootstrap Confidence Intervals

```python
# Calculate confidence intervals for accuracy metrics
bootstrap_result = validator.bootstrap_confidence_interval(
    accuracy_scores,
    statistic_func=np.mean,
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"95% CI: {bootstrap_result.confidence_interval}")
```

### Cross-Validation

```python
# Validate model performance with cross-validation
cv_result = validator.cross_validate_performance(
    X, y, model,
    cv_folds=5,
    scoring_metric='f1',
    stratified=True
)

print(f"CV Score: {cv_result.mean_score:.3f} ± {cv_result.std_score:.3f}")
```

## Environmental Factor Analysis

### Impact Assessment

```python
from tests.validation.environmental import create_environmental_validator, EnvironmentalFactor, PerformanceMetric

validator = create_environmental_validator()

# Analyze weather impact on detection performance
impact_analysis = validator.validate_environmental_impact(
    survey_results,
    EnvironmentalFactor.WEATHER_CONDITION,
    PerformanceMetric.DETECTION_RATE
)

print(f"Effect size: {impact_analysis.effect_size}")
print(f"Statistical significance: {impact_analysis.statistical_significance}")
```

### Optimal Conditions Identification

```python
# Find optimal survey conditions
optimal_conditions = validator.identify_optimal_conditions(
    survey_results,
    PerformanceMetric.DETECTION_RATE
)

print("Best conditions:")
for factor, value in optimal_conditions.best_conditions.items():
    print(f"  {factor.value}: {value}")

print("Recommendations:")
for rec in optimal_conditions.condition_recommendations:
    print(f"  - {rec}")
```

## Performance Monitoring

### Real-time Benchmarking

```python
from tests.performance.benchmarking import create_performance_benchmarker

benchmarker = create_performance_benchmarker()

# Benchmark GPR processing function
metrics = benchmarker.benchmark_function(
    gpr_processing_function,
    test_data,
    ground_truth=ground_truth_data,
    test_id="gpr_v2.0",
    iterations=10
)

print(f"Processing time: {metrics.processing_time:.3f}s")
print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
print(f"Detection accuracy: {metrics.detection_accuracy:.3f}")
```

### Regression Testing

```python
# Compare current vs baseline implementation
regression_result = benchmarker.regression_test(
    "algorithm_update_v2.1",
    current_function,
    baseline_function,
    test_data,
    iterations=5
)

print(f"Regression detected: {regression_result.regression_detected}")
print(f"Performance changes: {regression_result.performance_change}")
```

### A/B Testing

```python
# Compare two algorithm variants
ab_result = benchmarker.ab_test(
    "detection_algorithm_comparison",
    variant_a_function,
    variant_b_function,
    test_data,
    iterations=20
)

print(f"Winner: {ab_result.winner}")
print(f"Statistical significance: {ab_result.statistical_comparison}")
```

## Report Generation

### Comprehensive Validation Reports

```python
from tests.utils.reporting import create_validation_reporter

reporter = create_validation_reporter("./validation_reports")

# Generate comprehensive report
validation_report = reporter.generate_comprehensive_report(
    validation_results,
    report_title="GPR System Validation - Q4 2024",
    report_id="gpr_validation_q4_2024"
)

# Report includes:
# - Interactive HTML dashboard
# - Statistical analysis plots
# - Performance trend charts
# - Environmental impact visualizations
# - PAS 128 compliance status
# - Actionable recommendations
```

## Integration with CI/CD

### Automated Testing Pipeline

```yaml
# .github/workflows/validation.yml
name: GPR Validation Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r tests/requirements.txt

      - name: Run validation suite
        run: |
          python tests/run_validation_suite.py --full-suite

      - name: Upload validation reports
        uses: actions/upload-artifact@v3
        with:
          name: validation-reports
          path: validation_reports/
```

## Best Practices

### 1. **Test Data Management**
- Use real Twente ground truth data when available
- Generate synthetic data with realistic characteristics
- Maintain data versioning and provenance
- Implement data validation and integrity checks

### 2. **Statistical Rigor**
- Use appropriate statistical tests for data types
- Control for multiple comparisons (Bonferroni correction)
- Report effect sizes alongside p-values
- Use bootstrap methods for robust confidence intervals

### 3. **Performance Monitoring**
- Set realistic performance baselines
- Monitor trends over time, not just point measurements
- Use appropriate alert thresholds
- Consider environmental factors in performance assessment

### 4. **Environmental Validation**
- Test across diverse environmental conditions
- Use stratified sampling for environmental factors
- Account for interaction effects between factors
- Validate recommendations with field data

### 5. **Compliance Testing**
- Test against all relevant PAS 128 quality levels
- Maintain audit trails for compliance verification
- Regular updates for evolving standards
- Document deviations and justifications

## Troubleshooting

### Common Issues

1. **Ground Truth Data Loading**
   ```bash
   # Error: Metadata file not found
   # Solution: Verify path and file existence
   ls -la datasets/raw/twente_gpr/Metadata.csv
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Process data in chunks
   loader.load_data(chunk_size=100)
   ```

3. **Statistical Test Failures**
   ```python
   # Ensure sufficient sample sizes
   assert len(group1) >= 30 and len(group2) >= 30
   ```

4. **Performance Monitoring Database Lock**
   ```bash
   # Reset performance database
   rm performance_benchmark.db
   ```

### Configuration Debugging

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Validate configuration
python tests/run_validation_suite.py --config validation_config.json --verbose
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ground-truth

# Install development dependencies
pip install -r tests/requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Pre-commit hooks
pre-commit install
```

### Adding New Validation Components

1. **Create test module** in appropriate validation category
2. **Implement validation logic** with comprehensive error handling
3. **Add statistical analysis** using the statistical validation framework
4. **Update reporting** to include new metrics
5. **Add documentation** and usage examples

### Testing Guidelines

- All validation components must have unit tests
- Integration tests for end-to-end workflows
- Performance tests for computationally intensive operations
- Documentation tests for code examples

## License

This validation framework is part of the Underground Utility Detection Platform project. See the main project LICENSE file for details.

## Contact

For questions, issues, or contributions related to the GPR validation framework, please:

1. Check existing issues in the project repository
2. Create detailed bug reports with reproducible examples
3. Submit feature requests with clear use cases
4. Contribute improvements via pull requests

## Acknowledgments

- **University of Twente**: For providing the comprehensive GPR ground truth dataset
- **PAS 128:2014**: For establishing professional utility detection standards
- **Scientific Community**: For statistical and validation methodologies
- **Open Source Libraries**: scipy, statsmodels, scikit-learn, matplotlib, plotly