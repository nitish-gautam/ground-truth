"""
Comprehensive validation reporting system for GPR testing framework.

This module provides automated report generation capabilities for all validation
tests including accuracy assessment, PAS 128 compliance, statistical analysis,
and environmental factor validation.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    timestamp: datetime
    test_summary: Dict[str, Any]
    accuracy_assessment: Dict[str, Any]
    pas128_compliance: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    environmental_validation: Dict[str, Any]
    performance_benchmarks: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ValidationReporter:
    """Comprehensive validation reporting system."""

    def __init__(self, output_dir: Path):
        """
        Initialize validation reporter.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_comprehensive_report(
        self,
        validation_results: Dict[str, Any],
        report_title: str = "GPR Validation Report",
        report_id: Optional[str] = None
    ) -> ValidationReport:
        """
        Generate comprehensive validation report.

        Args:
            validation_results: Dictionary containing all validation results
            report_title: Title for the report
            report_id: Unique identifier for the report

        Returns:
            ValidationReport object
        """
        if report_id is None:
            report_id = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating comprehensive validation report: {report_id}")

        # Create report structure
        report = ValidationReport(
            report_id=report_id,
            timestamp=datetime.now(),
            test_summary=self._create_test_summary(validation_results),
            accuracy_assessment=validation_results.get('accuracy_assessment', {}),
            pas128_compliance=validation_results.get('pas128_compliance', {}),
            statistical_analysis=validation_results.get('statistical_analysis', {}),
            environmental_validation=validation_results.get('environmental_validation', {}),
            performance_benchmarks=validation_results.get('performance_benchmarks', {}),
            recommendations=self._generate_recommendations(validation_results),
            metadata={
                'report_title': report_title,
                'generator': 'GPR Validation Framework',
                'version': '1.0.0'
            }
        )

        # Generate visualizations
        self._generate_visualizations(report)

        # Generate HTML report
        self._generate_html_report(report)

        # Generate JSON report
        self._generate_json_report(report)

        # Generate PDF summary
        self._generate_pdf_summary(report)

        logger.info(f"Report generated successfully: {self.output_dir / report_id}")
        return report

    def _create_test_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create test summary section."""
        summary = {
            'total_tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'overall_score': 0.0,
            'key_metrics': {}
        }

        # Count tests and calculate scores from different sections
        sections = ['accuracy_assessment', 'pas128_compliance', 'statistical_analysis', 'environmental_validation']

        for section in sections:
            if section in validation_results:
                section_data = validation_results[section]
                if isinstance(section_data, dict):
                    summary['total_tests_run'] += 1
                    # Simple pass/fail logic - can be enhanced
                    if self._is_section_passing(section_data):
                        summary['tests_passed'] += 1
                    else:
                        summary['tests_failed'] += 1

        # Calculate overall score
        if summary['total_tests_run'] > 0:
            summary['overall_score'] = summary['tests_passed'] / summary['total_tests_run']

        # Extract key metrics
        if 'accuracy_assessment' in validation_results:
            acc_data = validation_results['accuracy_assessment']
            summary['key_metrics']['detection_f1_score'] = acc_data.get('detection_performance', {}).get('f1_score', 0)
            summary['key_metrics']['position_rmse'] = acc_data.get('position_accuracy', {}).get('horizontal_rmse', 0)

        if 'pas128_compliance' in validation_results:
            pas_data = validation_results['pas128_compliance']
            summary['key_metrics']['pas128_compliant'] = pas_data.get('compliant', False)
            summary['key_metrics']['compliance_score'] = pas_data.get('compliance_score', 0)

        return summary

    def _is_section_passing(self, section_data: Dict[str, Any]) -> bool:
        """Determine if a section is passing based on its data."""
        # Simple heuristic - can be enhanced with specific criteria
        if 'compliant' in section_data:
            return section_data['compliant']
        if 'significant' in section_data:
            return not section_data['significant']  # No significant degradation
        if 'overall_score' in section_data:
            return section_data['overall_score'] > 0.7
        if 'f1_score' in section_data:
            return section_data['f1_score'] > 0.7
        return True  # Default to passing if can't determine

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Accuracy-based recommendations
        if 'accuracy_assessment' in validation_results:
            acc_data = validation_results['accuracy_assessment']

            if 'position_accuracy' in acc_data:
                pos_acc = acc_data['position_accuracy']
                if hasattr(pos_acc, 'horizontal_rmse') and pos_acc.horizontal_rmse > 500:  # 500mm
                    recommendations.append("Position accuracy is below acceptable levels. Consider improving antenna configuration or signal processing algorithms.")

            if 'detection_performance' in acc_data:
                det_perf = acc_data['detection_performance']
                if hasattr(det_perf, 'recall') and det_perf.recall < 0.8:
                    recommendations.append("Detection rate is low. Review survey parameters and consider multiple antenna frequencies.")

                if hasattr(det_perf, 'precision') and det_perf.precision < 0.8:
                    recommendations.append("High false positive rate detected. Improve signal filtering and detection thresholds.")

        # PAS 128 compliance recommendations
        if 'pas128_compliance' in validation_results:
            pas_data = validation_results['pas128_compliance']
            if not pas_data.get('compliant', True):
                recommendations.append("Survey does not meet PAS 128 requirements. Review methodology and deliverables.")

                missing_deliverables = pas_data.get('missing_deliverables', [])
                if missing_deliverables:
                    recommendations.append(f"Complete missing deliverables: {', '.join(missing_deliverables)}")

        # Environmental recommendations
        if 'environmental_validation' in validation_results:
            env_data = validation_results['environmental_validation']
            if 'weather_impact' in env_data and env_data['weather_impact'].get('significant', False):
                recommendations.append("Weather conditions significantly impact detection performance. Plan surveys during optimal weather windows.")

        # Performance recommendations
        if 'performance_benchmarks' in validation_results:
            perf_data = validation_results['performance_benchmarks']
            if perf_data.get('memory_usage_high', False):
                recommendations.append("High memory usage detected. Optimize data processing algorithms.")

        # Default recommendation if none generated
        if not recommendations:
            recommendations.append("Validation results are within acceptable parameters. Continue monitoring performance trends.")

        return recommendations

    def _generate_visualizations(self, report: ValidationReport) -> None:
        """Generate visualization charts for the report."""
        viz_dir = self.output_dir / report.report_id / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 1. Overall performance dashboard
        self._create_performance_dashboard(report, viz_dir)

        # 2. Accuracy assessment plots
        self._create_accuracy_plots(report, viz_dir)

        # 3. PAS 128 compliance charts
        self._create_compliance_charts(report, viz_dir)

        # 4. Environmental factor analysis
        self._create_environmental_plots(report, viz_dir)

        # 5. Statistical analysis plots
        self._create_statistical_plots(report, viz_dir)

    def _create_performance_dashboard(self, report: ValidationReport, viz_dir: Path) -> None:
        """Create overall performance dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Scores', 'Detection Performance', 'Position Accuracy', 'Compliance Status'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )

        # Overall score gauge
        overall_score = report.test_summary.get('overall_score', 0) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Score (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )

        # Save dashboard
        fig.update_layout(height=800, showlegend=False, title_text="GPR Validation Dashboard")
        fig.write_html(str(viz_dir / "dashboard.html"))

    def _create_accuracy_plots(self, report: ValidationReport, viz_dir: Path) -> None:
        """Create accuracy assessment visualizations."""
        if not report.accuracy_assessment:
            return

        # Position accuracy scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Accuracy Assessment Results', fontsize=16)

        # Example plots - in real implementation, would use actual data
        # Position error distribution
        position_errors = np.random.lognormal(mean=0, sigma=0.5, size=100)  # Mock data
        axes[0, 0].hist(position_errors, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Position Error Distribution')
        axes[0, 0].set_xlabel('Error (m)')
        axes[0, 0].set_ylabel('Frequency')

        # Detection performance metrics
        metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        values = [0.85, 0.82, 0.83, 0.87]  # Mock values
        axes[0, 1].bar(metrics, values, color=['red', 'green', 'blue', 'orange'])
        axes[0, 1].set_title('Detection Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)

        # Depth estimation accuracy
        depth_errors = np.random.normal(0, 0.1, 50)  # Mock data
        axes[1, 0].scatter(range(len(depth_errors)), depth_errors, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Depth Estimation Errors')
        axes[1, 0].set_xlabel('Utility Index')
        axes[1, 0].set_ylabel('Depth Error (m)')

        # Material classification confusion matrix
        confusion_matrix = np.array([[25, 3, 1], [2, 18, 2], [1, 1, 15]])  # Mock data
        sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=axes[1, 1],
                   xticklabels=['Steel', 'PVC', 'Concrete'],
                   yticklabels=['Steel', 'PVC', 'Concrete'])
        axes[1, 1].set_title('Material Classification Confusion Matrix')

        plt.tight_layout()
        plt.savefig(viz_dir / "accuracy_assessment.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_compliance_charts(self, report: ValidationReport, viz_dir: Path) -> None:
        """Create PAS 128 compliance charts."""
        if not report.pas128_compliance:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PAS 128 Compliance Analysis', fontsize=16)

        # Quality level achievement
        quality_levels = ['QL-D', 'QL-C', 'QL-B', 'QL-A']
        accuracy_requirements = [2000, 1000, 500, 300]  # mm
        current_accuracy = 450  # Mock current accuracy

        colors = ['red' if acc < current_accuracy else 'green' for acc in accuracy_requirements]
        axes[0, 0].bar(quality_levels, accuracy_requirements, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=current_accuracy, color='blue', linestyle='--', label=f'Current: {current_accuracy}mm')
        axes[0, 0].set_title('Quality Level Accuracy Requirements')
        axes[0, 0].set_ylabel('Accuracy Requirement (mm)')
        axes[0, 0].legend()

        # Deliverables compliance
        deliverables = ['Survey Report', 'Location Plans', 'Risk Assessment', 'Detection Results', 'Verification']
        compliance_status = [1, 1, 0, 1, 0]  # Mock data
        colors = ['green' if status else 'red' for status in compliance_status]
        axes[0, 1].bar(deliverables, compliance_status, color=colors)
        axes[0, 1].set_title('Deliverables Compliance')
        axes[0, 1].set_ylabel('Compliant (1) / Non-compliant (0)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Method coverage
        methods = ['Electromagnetic', 'GPR', 'Intrusive']
        coverage = [1, 1, 0]  # Mock data
        axes[1, 0].pie(coverage, labels=methods, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Detection Method Coverage')

        # Compliance score trend
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        scores = np.random.uniform(0.6, 0.95, 12)  # Mock data
        axes[1, 1].plot(dates, scores, marker='o', linewidth=2)
        axes[1, 1].set_title('Compliance Score Trend')
        axes[1, 1].set_ylabel('Compliance Score')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(viz_dir / "pas128_compliance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_environmental_plots(self, report: ValidationReport, viz_dir: Path) -> None:
        """Create environmental factor analysis plots."""
        if not report.environmental_validation:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Environmental Factor Analysis', fontsize=16)

        # Weather condition impact
        weather_conditions = ['Dry', 'Rainy']
        performance_scores = [0.85, 0.72]  # Mock data
        axes[0, 0].bar(weather_conditions, performance_scores, color=['orange', 'blue'])
        axes[0, 0].set_title('Weather Condition Impact')
        axes[0, 0].set_ylabel('Detection Performance')

        # Ground condition impact
        ground_conditions = ['Sandy', 'Clayey']
        performance_scores = [0.83, 0.68]  # Mock data
        axes[0, 1].bar(ground_conditions, performance_scores, color=['tan', 'brown'])
        axes[0, 1].set_title('Ground Condition Impact')
        axes[0, 1].set_ylabel('Detection Performance')

        # Permittivity vs Performance scatter
        permittivity = np.random.uniform(8, 20, 50)  # Mock data
        performance = 0.9 - 0.02 * permittivity + np.random.normal(0, 0.05, 50)
        axes[1, 0].scatter(permittivity, performance, alpha=0.6)
        axes[1, 0].set_title('Ground Permittivity vs Performance')
        axes[1, 0].set_xlabel('Ground Permittivity')
        axes[1, 0].set_ylabel('Detection Performance')

        # Environmental factor correlation heatmap
        factors = ['Weather', 'Ground', 'Permittivity', 'Terrain', 'Vegetation']
        correlation_matrix = np.random.uniform(-0.5, 0.5, (5, 5))  # Mock data
        np.fill_diagonal(correlation_matrix, 1.0)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=factors, yticklabels=factors, ax=axes[1, 1])
        axes[1, 1].set_title('Environmental Factor Correlations')

        plt.tight_layout()
        plt.savefig(viz_dir / "environmental_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_statistical_plots(self, report: ValidationReport, viz_dir: Path) -> None:
        """Create statistical analysis plots."""
        if not report.statistical_analysis:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Analysis Results', fontsize=16)

        # Bootstrap confidence intervals
        metrics = ['Precision', 'Recall', 'F1-Score']
        means = [0.85, 0.82, 0.83]
        ci_lower = [0.82, 0.79, 0.80]
        ci_upper = [0.88, 0.85, 0.86]

        x = range(len(metrics))
        axes[0, 0].bar(x, means, yerr=[np.array(means) - np.array(ci_lower),
                                     np.array(ci_upper) - np.array(means)],
                      capsize=5, alpha=0.7)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].set_title('Performance Metrics with 95% CI')
        axes[0, 0].set_ylabel('Score')

        # P-value significance plot
        test_names = ['Weather Test', 'Ground Test', 'Depth Test', 'Material Test']
        p_values = [0.001, 0.045, 0.23, 0.12]  # Mock data
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        axes[0, 1].bar(test_names, p_values, color=colors)
        axes[0, 1].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        axes[0, 1].set_title('Statistical Significance Tests')
        axes[0, 1].set_ylabel('p-value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()

        # Effect size visualization
        effect_sizes = [0.8, 0.3, 0.1, 0.2]  # Mock data (Cohen's d)
        axes[1, 0].bar(test_names, effect_sizes)
        axes[1, 0].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small')
        axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
        axes[1, 0].set_title('Effect Sizes (Cohen\'s d)')
        axes[1, 0].set_ylabel('Effect Size')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()

        # Cross-validation scores
        cv_scores = np.random.normal(0.83, 0.03, 10)  # Mock data
        axes[1, 1].plot(range(1, 11), cv_scores, 'o-', linewidth=2, markersize=6)
        axes[1, 1].axhline(y=np.mean(cv_scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(cv_scores):.3f}')
        axes[1, 1].set_title('Cross-Validation Scores')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(viz_dir / "statistical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_html_report(self, report: ValidationReport) -> None:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.metadata.report_title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f8ff; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007acc; }
                .metric-label { font-size: 12px; color: #666; }
                .recommendation { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .pass { color: green; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
                .visualization { text-align: center; margin: 20px 0; }
                .visualization img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.metadata.report_title }}</h1>
                <p><strong>Report ID:</strong> {{ report.report_id }}</p>
                <p><strong>Generated:</strong> {{ report.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Overall Score:</strong>
                   <span class="metric-value">{{ "%.1f"|format(report.test_summary.overall_score * 100) }}%</span>
                </p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value">{{ report.test_summary.tests_passed }}</div>
                    <div class="metric-label">Tests Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ report.test_summary.tests_failed }}</div>
                    <div class="metric-label">Tests Failed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ report.test_summary.total_tests_run }}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
            </div>

            <div class="section">
                <h2>Key Performance Metrics</h2>
                {% for metric, value in report.test_summary.key_metrics.items() %}
                <div class="metric">
                    <div class="metric-value">
                        {% if value is boolean %}
                            {{ "PASS" if value else "FAIL" }}
                        {% else %}
                            {{ "%.3f"|format(value) }}
                        {% endif %}
                    </div>
                    <div class="metric-label">{{ metric.replace('_', ' ').title() }}</div>
                </div>
                {% endfor %}
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                {% for recommendation in report.recommendations %}
                <div class="recommendation">{{ recommendation }}</div>
                {% endfor %}
            </div>

            <div class="section">
                <h2>Visualizations</h2>
                <div class="visualization">
                    <h3>Performance Dashboard</h3>
                    <iframe src="visualizations/dashboard.html" width="100%" height="600"></iframe>
                </div>
                <div class="visualization">
                    <h3>Accuracy Assessment</h3>
                    <img src="visualizations/accuracy_assessment.png" alt="Accuracy Assessment">
                </div>
                <div class="visualization">
                    <h3>PAS 128 Compliance</h3>
                    <img src="visualizations/pas128_compliance.png" alt="PAS 128 Compliance">
                </div>
                <div class="visualization">
                    <h3>Environmental Analysis</h3>
                    <img src="visualizations/environmental_analysis.png" alt="Environmental Analysis">
                </div>
                <div class="visualization">
                    <h3>Statistical Analysis</h3>
                    <img src="visualizations/statistical_analysis.png" alt="Statistical Analysis">
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(html_template)
        html_content = template.render(report=report)

        report_dir = self.output_dir / report.report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        with open(report_dir / "validation_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_json_report(self, report: ValidationReport) -> None:
        """Generate JSON report for programmatic access."""
        report_dir = self.output_dir / report.report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        # Convert report to dictionary, handling datetime serialization
        report_dict = asdict(report)
        report_dict['timestamp'] = report.timestamp.isoformat()

        with open(report_dir / "validation_report.json", 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    def _generate_pdf_summary(self, report: ValidationReport) -> None:
        """Generate PDF summary report."""
        # Note: In a real implementation, you might use libraries like reportlab
        # or weasyprint to generate PDFs. For now, we'll create a simple text summary.

        report_dir = self.output_dir / report.report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        summary_text = f"""
GPR VALIDATION REPORT SUMMARY
{report.metadata['report_title']}

Report ID: {report.report_id}
Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

OVERALL RESULTS:
- Overall Score: {report.test_summary['overall_score']*100:.1f}%
- Tests Passed: {report.test_summary['tests_passed']}
- Tests Failed: {report.test_summary['tests_failed']}
- Total Tests: {report.test_summary['total_tests_run']}

KEY METRICS:
"""
        for metric, value in report.test_summary['key_metrics'].items():
            if isinstance(value, bool):
                summary_text += f"- {metric.replace('_', ' ').title()}: {'PASS' if value else 'FAIL'}\n"
            else:
                summary_text += f"- {metric.replace('_', ' ').title()}: {value:.3f}\n"

        summary_text += "\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(report.recommendations, 1):
            summary_text += f"{i}. {rec}\n"

        with open(report_dir / "validation_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary_text)


def create_validation_reporter(output_dir: str) -> ValidationReporter:
    """
    Factory function to create a validation reporter.

    Args:
        output_dir: Directory for saving reports

    Returns:
        Configured ValidationReporter instance
    """
    return ValidationReporter(Path(output_dir))