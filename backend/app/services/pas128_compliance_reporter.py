"""
PAS 128 Compliance Scoring and Reporting System

This service provides comprehensive compliance scoring, gap analysis,
and professional reporting for PAS 128 compliance assessments.
"""
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from ..schemas.pas128 import (
    QualityLevel, SurveyMethod, DeliverableType, DetectionMethod,
    SurveyData, ComplianceReport, QualityLevelAssessment,
    EnvironmentalImpactAssessment, ComplianceCheck, UtilityDetection
)

logger = logging.getLogger(__name__)


class ComplianceCategory(Enum):
    """Compliance assessment categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


class RiskLevel(Enum):
    """Risk levels for compliance gaps"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ComplianceMetrics:
    """Comprehensive compliance metrics"""
    overall_score: float
    category: ComplianceCategory
    method_compliance_score: float
    deliverable_compliance_score: float
    accuracy_compliance_score: float
    environmental_suitability_score: float
    quality_level_achievement_score: float
    weighted_score: float
    confidence_level: float


@dataclass
class GapAnalysis:
    """Gap analysis for compliance improvement"""
    gap_id: str
    gap_description: str
    impact_severity: RiskLevel
    affected_areas: List[str]
    current_state: str
    required_state: str
    effort_estimate: str
    cost_estimate: Optional[str]
    timeline_estimate: str
    mitigation_actions: List[str]
    priority_score: float


@dataclass
class RecommendationItem:
    """Individual recommendation item"""
    recommendation_id: str
    category: str
    priority: RiskLevel
    description: str
    rationale: str
    implementation_steps: List[str]
    success_criteria: List[str]
    dependencies: List[str]
    estimated_effort: str
    estimated_cost: Optional[str]
    timeline: str


@dataclass
class ComplianceBenchmark:
    """Compliance benchmarking data"""
    benchmark_type: str
    current_performance: float
    industry_average: float
    best_practice: float
    percentile_ranking: float
    improvement_potential: float


class PAS128ComplianceReporter:
    """
    Comprehensive compliance scoring and reporting system for PAS 128.

    This service provides:
    - Multi-dimensional compliance scoring
    - Professional gap analysis and recommendations
    - Benchmarking against industry standards
    - Risk-based prioritization of improvements
    - Detailed reporting and visualization support
    """

    def __init__(self):
        """Initialize the compliance reporting system."""
        self.logger = logging.getLogger(__name__)

        # Initialize scoring weights
        self.scoring_weights = self._initialize_scoring_weights()

        # Initialize compliance thresholds
        self.compliance_thresholds = self._initialize_compliance_thresholds()

        # Initialize benchmark data
        self.benchmark_data = self._initialize_benchmark_data()

        # Initialize risk assessment criteria
        self.risk_criteria = self._initialize_risk_criteria()

        self.logger.info("PAS 128 Compliance Reporter initialized")

    def _initialize_scoring_weights(self) -> Dict[str, float]:
        """Initialize scoring weights for different compliance aspects."""
        return {
            "method_compliance": 0.25,
            "deliverable_compliance": 0.20,
            "accuracy_compliance": 0.20,
            "environmental_suitability": 0.15,
            "quality_level_achievement": 0.15,
            "documentation_quality": 0.05
        }

    def _initialize_compliance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize compliance thresholds for categorization."""
        return {
            "score_categories": {
                ComplianceCategory.EXCELLENT.value: 0.90,
                ComplianceCategory.GOOD.value: 0.80,
                ComplianceCategory.SATISFACTORY.value: 0.70,
                ComplianceCategory.NEEDS_IMPROVEMENT.value: 0.60,
                ComplianceCategory.POOR.value: 0.0
            },
            "quality_level_thresholds": {
                QualityLevel.QL_A.value: 0.90,
                QualityLevel.QL_B.value: 0.75,
                QualityLevel.QL_C.value: 0.60,
                QualityLevel.QL_D.value: 0.45
            },
            "risk_thresholds": {
                RiskLevel.CRITICAL.value: 0.90,
                RiskLevel.HIGH.value: 0.70,
                RiskLevel.MEDIUM.value: 0.50,
                RiskLevel.LOW.value: 0.30
            }
        }

    def _initialize_benchmark_data(self) -> Dict[str, Dict[str, float]]:
        """Initialize benchmark data for comparison."""
        return {
            "industry_averages": {
                "overall_compliance": 0.75,
                "method_compliance": 0.78,
                "deliverable_compliance": 0.72,
                "accuracy_compliance": 0.70,
                "quality_level_achievement": 0.68
            },
            "best_practices": {
                "overall_compliance": 0.92,
                "method_compliance": 0.95,
                "deliverable_compliance": 0.90,
                "accuracy_compliance": 0.88,
                "quality_level_achievement": 0.85
            },
            "minimum_standards": {
                "overall_compliance": 0.60,
                "method_compliance": 0.65,
                "deliverable_compliance": 0.60,
                "accuracy_compliance": 0.55,
                "quality_level_achievement": 0.50
            }
        }

    def _initialize_risk_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Initialize risk assessment criteria."""
        return {
            "gap_severity_factors": {
                "quality_level_gap": 0.8,
                "method_missing": 0.7,
                "deliverable_missing": 0.6,
                "accuracy_poor": 0.5,
                "environmental_unsuitable": 0.4
            },
            "implementation_factors": {
                "high_cost": 0.8,
                "long_timeline": 0.6,
                "complex_technical": 0.7,
                "regulatory_approval": 0.5,
                "resource_intensive": 0.6
            }
        }

    def calculate_compliance_metrics(self, compliance_report: ComplianceReport) -> ComplianceMetrics:
        """
        Calculate comprehensive compliance metrics from compliance report.

        Args:
            compliance_report: Compliance report to calculate metrics for

        Returns:
            Comprehensive compliance metrics
        """
        self.logger.info(f"Calculating compliance metrics for survey {compliance_report.survey_id}")

        # Extract individual compliance scores
        method_score = self._calculate_method_compliance_score(compliance_report)
        deliverable_score = self._calculate_deliverable_compliance_score(compliance_report)
        accuracy_score = self._calculate_accuracy_compliance_score(compliance_report)
        environmental_score = compliance_report.environmental_impact.overall_environmental_score
        quality_level_score = self._calculate_quality_level_achievement_score(compliance_report)

        # Calculate weighted overall score
        weighted_score = (
            self.scoring_weights["method_compliance"] * method_score +
            self.scoring_weights["deliverable_compliance"] * deliverable_score +
            self.scoring_weights["accuracy_compliance"] * accuracy_score +
            self.scoring_weights["environmental_suitability"] * environmental_score +
            self.scoring_weights["quality_level_achievement"] * quality_level_score
        )

        # Determine compliance category
        category = self._categorize_compliance_score(weighted_score)

        # Calculate confidence level
        confidence_level = compliance_report.quality_level_assessment.confidence

        return ComplianceMetrics(
            overall_score=compliance_report.overall_compliance_score,
            category=category,
            method_compliance_score=method_score,
            deliverable_compliance_score=deliverable_score,
            accuracy_compliance_score=accuracy_score,
            environmental_suitability_score=environmental_score,
            quality_level_achievement_score=quality_level_score,
            weighted_score=weighted_score,
            confidence_level=confidence_level
        )

    def _calculate_method_compliance_score(self, compliance_report: ComplianceReport) -> float:
        """Calculate method compliance score from compliance checks."""
        method_checks = [
            check for check in compliance_report.compliance_checks
            if "method" in check.check_name.lower()
        ]

        if not method_checks:
            return 0.5  # Default if no method checks

        return np.mean([check.score for check in method_checks])

    def _calculate_deliverable_compliance_score(self, compliance_report: ComplianceReport) -> float:
        """Calculate deliverable compliance score."""
        deliverable_checks = [
            check for check in compliance_report.compliance_checks
            if "deliverable" in check.check_name.lower()
        ]

        if not deliverable_checks:
            return 0.5  # Default if no deliverable checks

        return np.mean([check.score for check in deliverable_checks])

    def _calculate_accuracy_compliance_score(self, compliance_report: ComplianceReport) -> float:
        """Calculate accuracy compliance score."""
        accuracy_checks = [
            check for check in compliance_report.compliance_checks
            if "accuracy" in check.check_name.lower()
        ]

        if not accuracy_checks:
            # Fallback to accuracy analysis if available
            accuracy_analysis = compliance_report.accuracy_analysis
            if accuracy_analysis and "total_detections" in accuracy_analysis:
                if accuracy_analysis["total_detections"] > 0:
                    return 0.8  # Assume good accuracy if detections exist
                else:
                    return 0.4  # Poor accuracy if no detections

        return np.mean([check.score for check in accuracy_checks])

    def _calculate_quality_level_achievement_score(self, compliance_report: ComplianceReport) -> float:
        """Calculate quality level achievement score."""
        target_level = compliance_report.target_quality_level
        achieved_level = compliance_report.achieved_quality_level

        # Map quality levels to numerical values
        level_values = {
            QualityLevel.QL_D: 1,
            QualityLevel.QL_C: 2,
            QualityLevel.QL_B: 3,
            QualityLevel.QL_A: 4
        }

        target_value = level_values[target_level]
        achieved_value = level_values[achieved_level]

        # Calculate achievement ratio
        achievement_ratio = achieved_value / target_value
        return min(1.0, achievement_ratio)

    def _categorize_compliance_score(self, score: float) -> ComplianceCategory:
        """Categorize compliance score into descriptive category."""
        thresholds = self.compliance_thresholds["score_categories"]

        if score >= thresholds[ComplianceCategory.EXCELLENT.value]:
            return ComplianceCategory.EXCELLENT
        elif score >= thresholds[ComplianceCategory.GOOD.value]:
            return ComplianceCategory.GOOD
        elif score >= thresholds[ComplianceCategory.SATISFACTORY.value]:
            return ComplianceCategory.SATISFACTORY
        elif score >= thresholds[ComplianceCategory.NEEDS_IMPROVEMENT.value]:
            return ComplianceCategory.NEEDS_IMPROVEMENT
        else:
            return ComplianceCategory.POOR

    def perform_gap_analysis(self, compliance_report: ComplianceReport,
                           metrics: ComplianceMetrics) -> List[GapAnalysis]:
        """
        Perform comprehensive gap analysis for compliance improvement.

        Args:
            compliance_report: Compliance report to analyze
            metrics: Calculated compliance metrics

        Returns:
            List of identified gaps with improvement recommendations
        """
        self.logger.info(f"Performing gap analysis for survey {compliance_report.survey_id}")

        gaps = []
        gap_counter = 1

        # Quality level achievement gap
        if compliance_report.achieved_quality_level != compliance_report.target_quality_level:
            gap = self._analyze_quality_level_gap(compliance_report, gap_counter)
            gaps.append(gap)
            gap_counter += 1

        # Method compliance gaps
        method_gaps = self._analyze_method_gaps(compliance_report, metrics, gap_counter)
        gaps.extend(method_gaps)
        gap_counter += len(method_gaps)

        # Deliverable compliance gaps
        deliverable_gaps = self._analyze_deliverable_gaps(compliance_report, metrics, gap_counter)
        gaps.extend(deliverable_gaps)
        gap_counter += len(deliverable_gaps)

        # Accuracy compliance gaps
        accuracy_gaps = self._analyze_accuracy_gaps(compliance_report, metrics, gap_counter)
        gaps.extend(accuracy_gaps)
        gap_counter += len(accuracy_gaps)

        # Environmental suitability gaps
        environmental_gaps = self._analyze_environmental_gaps(compliance_report, metrics, gap_counter)
        gaps.extend(environmental_gaps)

        # Sort gaps by priority score
        gaps.sort(key=lambda x: x.priority_score, reverse=True)

        return gaps

    def _analyze_quality_level_gap(self, compliance_report: ComplianceReport, gap_id: int) -> GapAnalysis:
        """Analyze quality level achievement gap."""
        target = compliance_report.target_quality_level
        achieved = compliance_report.achieved_quality_level

        # Determine impact severity
        level_diff = abs(
            list(QualityLevel).index(target) - list(QualityLevel).index(achieved)
        )

        if level_diff >= 2:
            severity = RiskLevel.CRITICAL
        elif level_diff == 1:
            severity = RiskLevel.HIGH
        else:
            severity = RiskLevel.MEDIUM

        return GapAnalysis(
            gap_id=f"GAP-{gap_id:03d}",
            gap_description=f"Quality level gap: Target {target.value}, Achieved {achieved.value}",
            impact_severity=severity,
            affected_areas=["Overall Compliance", "Method Execution", "Documentation"],
            current_state=f"Achieved {achieved.value}",
            required_state=f"Target {target.value}",
            effort_estimate="Medium to High",
            cost_estimate="£5,000 - £15,000",
            timeline_estimate="2-4 weeks",
            mitigation_actions=compliance_report.quality_level_assessment.recommendations,
            priority_score=0.9 - (level_diff * 0.1)
        )

    def _analyze_method_gaps(self, compliance_report: ComplianceReport,
                           metrics: ComplianceMetrics, start_id: int) -> List[GapAnalysis]:
        """Analyze method compliance gaps."""
        gaps = []
        gap_id = start_id

        # Find failed method checks
        method_checks = [
            check for check in compliance_report.compliance_checks
            if "method" in check.check_name.lower() and not check.passed
        ]

        for check in method_checks:
            severity = RiskLevel.HIGH if check.score < 0.5 else RiskLevel.MEDIUM

            gap = GapAnalysis(
                gap_id=f"GAP-{gap_id:03d}",
                gap_description=f"Method compliance issue: {check.check_name}",
                impact_severity=severity,
                affected_areas=["Method Execution", "Survey Quality"],
                current_state=f"Score: {check.score:.2f}",
                required_state="Score: ≥0.80",
                effort_estimate="Medium",
                cost_estimate="£2,000 - £8,000",
                timeline_estimate="1-3 weeks",
                mitigation_actions=[
                    f"Address identified gaps: {', '.join(check.gaps[:3])}"
                ],
                priority_score=0.8 - check.score
            )
            gaps.append(gap)
            gap_id += 1

        return gaps

    def _analyze_deliverable_gaps(self, compliance_report: ComplianceReport,
                                metrics: ComplianceMetrics, start_id: int) -> List[GapAnalysis]:
        """Analyze deliverable compliance gaps."""
        gaps = []
        gap_id = start_id

        # Find failed deliverable checks
        deliverable_checks = [
            check for check in compliance_report.compliance_checks
            if "deliverable" in check.check_name.lower() and not check.passed
        ]

        for check in deliverable_checks:
            severity = RiskLevel.MEDIUM if check.score < 0.7 else RiskLevel.LOW

            gap = GapAnalysis(
                gap_id=f"GAP-{gap_id:03d}",
                gap_description=f"Deliverable compliance issue: {check.check_name}",
                impact_severity=severity,
                affected_areas=["Documentation", "Client Deliverables"],
                current_state=f"Score: {check.score:.2f}",
                required_state="Score: ≥0.80",
                effort_estimate="Low to Medium",
                cost_estimate="£1,000 - £5,000",
                timeline_estimate="1-2 weeks",
                mitigation_actions=[
                    f"Improve deliverable quality: {', '.join(check.gaps[:2])}"
                ],
                priority_score=0.6 - (check.score * 0.2)
            )
            gaps.append(gap)
            gap_id += 1

        return gaps

    def _analyze_accuracy_gaps(self, compliance_report: ComplianceReport,
                             metrics: ComplianceMetrics, start_id: int) -> List[GapAnalysis]:
        """Analyze accuracy compliance gaps."""
        gaps = []

        if metrics.accuracy_compliance_score < 0.8:
            gap = GapAnalysis(
                gap_id=f"GAP-{start_id:03d}",
                gap_description="Survey accuracy below requirements",
                impact_severity=RiskLevel.HIGH,
                affected_areas=["Survey Accuracy", "Quality Level Achievement"],
                current_state=f"Accuracy score: {metrics.accuracy_compliance_score:.2f}",
                required_state="Accuracy score: ≥0.80",
                effort_estimate="High",
                cost_estimate="£3,000 - £10,000",
                timeline_estimate="2-4 weeks",
                mitigation_actions=[
                    "Improve survey methodology",
                    "Calibrate equipment",
                    "Increase survey density"
                ],
                priority_score=0.9 - metrics.accuracy_compliance_score
            )
            gaps.append(gap)

        return gaps

    def _analyze_environmental_gaps(self, compliance_report: ComplianceReport,
                                  metrics: ComplianceMetrics, start_id: int) -> List[GapAnalysis]:
        """Analyze environmental suitability gaps."""
        gaps = []

        env_score = metrics.environmental_suitability_score
        if env_score < 0.6:
            severity = RiskLevel.HIGH if env_score < 0.4 else RiskLevel.MEDIUM

            gap = GapAnalysis(
                gap_id=f"GAP-{start_id:03d}",
                gap_description="Environmental conditions limit survey effectiveness",
                impact_severity=severity,
                affected_areas=["Survey Quality", "Method Effectiveness"],
                current_state=f"Environmental score: {env_score:.2f}",
                required_state="Environmental score: ≥0.60",
                effort_estimate="Medium",
                cost_estimate="£1,000 - £5,000",
                timeline_estimate="1-2 weeks",
                mitigation_actions=compliance_report.environmental_impact.recommended_adjustments,
                priority_score=0.7 - env_score
            )
            gaps.append(gap)

        return gaps

    def generate_recommendations(self, compliance_report: ComplianceReport,
                               metrics: ComplianceMetrics,
                               gaps: List[GapAnalysis]) -> List[RecommendationItem]:
        """
        Generate prioritized recommendations for compliance improvement.

        Args:
            compliance_report: Compliance report to base recommendations on
            metrics: Calculated compliance metrics
            gaps: Identified compliance gaps

        Returns:
            List of prioritized recommendations
        """
        self.logger.info(f"Generating recommendations for survey {compliance_report.survey_id}")

        recommendations = []
        rec_counter = 1

        # High-priority recommendations from critical gaps
        critical_gaps = [gap for gap in gaps if gap.impact_severity == RiskLevel.CRITICAL]
        for gap in critical_gaps:
            rec = self._create_recommendation_from_gap(gap, "Critical Issue Resolution", rec_counter)
            recommendations.append(rec)
            rec_counter += 1

        # Method improvement recommendations
        if metrics.method_compliance_score < 0.8:
            method_rec = self._create_method_improvement_recommendation(
                compliance_report, metrics, rec_counter
            )
            recommendations.append(method_rec)
            rec_counter += 1

        # Deliverable improvement recommendations
        if metrics.deliverable_compliance_score < 0.8:
            deliverable_rec = self._create_deliverable_improvement_recommendation(
                compliance_report, metrics, rec_counter
            )
            recommendations.append(deliverable_rec)
            rec_counter += 1

        # Quality level advancement recommendations
        if compliance_report.achieved_quality_level != compliance_report.target_quality_level:
            quality_rec = self._create_quality_level_advancement_recommendation(
                compliance_report, rec_counter
            )
            recommendations.append(quality_rec)
            rec_counter += 1

        # Environmental optimization recommendations
        if metrics.environmental_suitability_score < 0.7:
            env_rec = self._create_environmental_optimization_recommendation(
                compliance_report, metrics, rec_counter
            )
            recommendations.append(env_rec)

        # Sort recommendations by priority
        priority_order = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }
        recommendations.sort(key=lambda x: priority_order[x.priority], reverse=True)

        return recommendations

    def _create_recommendation_from_gap(self, gap: GapAnalysis, category: str, rec_id: int) -> RecommendationItem:
        """Create recommendation item from gap analysis."""
        return RecommendationItem(
            recommendation_id=f"REC-{rec_id:03d}",
            category=category,
            priority=gap.impact_severity,
            description=f"Address {gap.gap_description}",
            rationale=f"This gap affects {', '.join(gap.affected_areas)} and has {gap.impact_severity.value} impact",
            implementation_steps=gap.mitigation_actions[:3],  # Top 3 actions
            success_criteria=[gap.required_state],
            dependencies=[],
            estimated_effort=gap.effort_estimate,
            estimated_cost=gap.cost_estimate,
            timeline=gap.timeline_estimate
        )

    def _create_method_improvement_recommendation(self, compliance_report: ComplianceReport,
                                                metrics: ComplianceMetrics, rec_id: int) -> RecommendationItem:
        """Create method improvement recommendation."""
        return RecommendationItem(
            recommendation_id=f"REC-{rec_id:03d}",
            category="Method Improvement",
            priority=RiskLevel.HIGH,
            description="Improve survey method execution and compliance",
            rationale=f"Method compliance score ({metrics.method_compliance_score:.2f}) below target (0.80)",
            implementation_steps=[
                "Review and update survey methodology procedures",
                "Provide additional operator training",
                "Implement quality control checkpoints",
                "Verify equipment calibration and performance"
            ],
            success_criteria=[
                "Method compliance score ≥ 0.80",
                "All required methods executed to standard",
                "No critical method compliance issues"
            ],
            dependencies=["Equipment availability", "Operator certification"],
            estimated_effort="Medium",
            estimated_cost="£3,000 - £8,000",
            timeline="2-3 weeks"
        )

    def _create_deliverable_improvement_recommendation(self, compliance_report: ComplianceReport,
                                                     metrics: ComplianceMetrics, rec_id: int) -> RecommendationItem:
        """Create deliverable improvement recommendation."""
        return RecommendationItem(
            recommendation_id=f"REC-{rec_id:03d}",
            category="Deliverable Enhancement",
            priority=RiskLevel.MEDIUM,
            description="Enhance deliverable quality and completeness",
            rationale=f"Deliverable compliance score ({metrics.deliverable_compliance_score:.2f}) below target (0.80)",
            implementation_steps=[
                "Review deliverable templates and standards",
                "Implement quality review process",
                "Enhance documentation procedures",
                "Provide training on deliverable requirements"
            ],
            success_criteria=[
                "Deliverable compliance score ≥ 0.80",
                "All required deliverables provided",
                "Deliverables meet format and content standards"
            ],
            dependencies=["Template updates", "Staff training"],
            estimated_effort="Medium",
            estimated_cost="£2,000 - £5,000",
            timeline="1-2 weeks"
        )

    def _create_quality_level_advancement_recommendation(self, compliance_report: ComplianceReport, rec_id: int) -> RecommendationItem:
        """Create quality level advancement recommendation."""
        target = compliance_report.target_quality_level
        achieved = compliance_report.achieved_quality_level

        return RecommendationItem(
            recommendation_id=f"REC-{rec_id:03d}",
            category="Quality Level Advancement",
            priority=RiskLevel.HIGH,
            description=f"Advance from {achieved.value} to {target.value}",
            rationale=f"Current survey achieves {achieved.value} but target is {target.value}",
            implementation_steps=compliance_report.quality_level_assessment.recommendations[:4],
            success_criteria=[f"Achieve {target.value} compliance"],
            dependencies=["Method implementation", "Equipment availability"],
            estimated_effort="High",
            estimated_cost="£5,000 - £15,000",
            timeline="3-6 weeks"
        )

    def _create_environmental_optimization_recommendation(self, compliance_report: ComplianceReport,
                                                        metrics: ComplianceMetrics, rec_id: int) -> RecommendationItem:
        """Create environmental optimization recommendation."""
        return RecommendationItem(
            recommendation_id=f"REC-{rec_id:03d}",
            category="Environmental Optimization",
            priority=RiskLevel.MEDIUM,
            description="Optimize survey methods for environmental conditions",
            rationale=f"Environmental suitability score ({metrics.environmental_suitability_score:.2f}) limits effectiveness",
            implementation_steps=compliance_report.environmental_impact.recommended_adjustments[:3],
            success_criteria=[
                "Environmental suitability score ≥ 0.70",
                "Method effectiveness optimized for conditions"
            ],
            dependencies=["Site access", "Weather conditions"],
            estimated_effort="Low to Medium",
            estimated_cost="£1,000 - £3,000",
            timeline="1-2 weeks"
        )

    def create_benchmark_comparison(self, metrics: ComplianceMetrics) -> List[ComplianceBenchmark]:
        """
        Create benchmark comparison for compliance metrics.

        Args:
            metrics: Compliance metrics to benchmark

        Returns:
            List of benchmark comparisons
        """
        benchmarks = []

        # Overall compliance benchmark
        benchmarks.append(ComplianceBenchmark(
            benchmark_type="Overall Compliance",
            current_performance=metrics.overall_score,
            industry_average=self.benchmark_data["industry_averages"]["overall_compliance"],
            best_practice=self.benchmark_data["best_practices"]["overall_compliance"],
            percentile_ranking=self._calculate_percentile_ranking(
                metrics.overall_score,
                self.benchmark_data["industry_averages"]["overall_compliance"]
            ),
            improvement_potential=self.benchmark_data["best_practices"]["overall_compliance"] - metrics.overall_score
        ))

        # Method compliance benchmark
        benchmarks.append(ComplianceBenchmark(
            benchmark_type="Method Compliance",
            current_performance=metrics.method_compliance_score,
            industry_average=self.benchmark_data["industry_averages"]["method_compliance"],
            best_practice=self.benchmark_data["best_practices"]["method_compliance"],
            percentile_ranking=self._calculate_percentile_ranking(
                metrics.method_compliance_score,
                self.benchmark_data["industry_averages"]["method_compliance"]
            ),
            improvement_potential=self.benchmark_data["best_practices"]["method_compliance"] - metrics.method_compliance_score
        ))

        # Add other benchmarks...
        benchmark_types = [
            ("Deliverable Compliance", metrics.deliverable_compliance_score, "deliverable_compliance"),
            ("Accuracy Compliance", metrics.accuracy_compliance_score, "accuracy_compliance"),
            ("Quality Level Achievement", metrics.quality_level_achievement_score, "quality_level_achievement")
        ]

        for name, score, key in benchmark_types:
            benchmarks.append(ComplianceBenchmark(
                benchmark_type=name,
                current_performance=score,
                industry_average=self.benchmark_data["industry_averages"][key],
                best_practice=self.benchmark_data["best_practices"][key],
                percentile_ranking=self._calculate_percentile_ranking(
                    score, self.benchmark_data["industry_averages"][key]
                ),
                improvement_potential=self.benchmark_data["best_practices"][key] - score
            ))

        return benchmarks

    def _calculate_percentile_ranking(self, score: float, industry_average: float) -> float:
        """Calculate percentile ranking based on score and industry average."""
        # Simplified percentile calculation
        if score >= industry_average * 1.2:
            return 90.0
        elif score >= industry_average * 1.1:
            return 75.0
        elif score >= industry_average:
            return 60.0
        elif score >= industry_average * 0.9:
            return 40.0
        elif score >= industry_average * 0.8:
            return 25.0
        else:
            return 10.0

    def generate_executive_summary(self, compliance_report: ComplianceReport,
                                 metrics: ComplianceMetrics,
                                 gaps: List[GapAnalysis],
                                 recommendations: List[RecommendationItem]) -> Dict[str, Any]:
        """
        Generate executive summary for compliance report.

        Args:
            compliance_report: Complete compliance report
            metrics: Calculated compliance metrics
            gaps: Identified gaps
            recommendations: Generated recommendations

        Returns:
            Executive summary data
        """
        self.logger.info(f"Generating executive summary for survey {compliance_report.survey_id}")

        # Key findings
        key_findings = []
        if metrics.category in [ComplianceCategory.EXCELLENT, ComplianceCategory.GOOD]:
            key_findings.append(f"Survey demonstrates {metrics.category.value} PAS 128 compliance")
        else:
            key_findings.append(f"Survey requires improvement to meet PAS 128 standards")

        if compliance_report.achieved_quality_level != compliance_report.target_quality_level:
            key_findings.append(f"Quality level gap: {compliance_report.achieved_quality_level.value} achieved vs {compliance_report.target_quality_level.value} target")

        # Critical issues
        critical_issues = [gap.gap_description for gap in gaps if gap.impact_severity == RiskLevel.CRITICAL]

        # Priority actions
        high_priority_actions = [
            rec.description for rec in recommendations
            if rec.priority in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ][:3]  # Top 3

        return {
            "survey_id": compliance_report.survey_id,
            "assessment_date": compliance_report.assessment_date.isoformat(),
            "overall_score": metrics.overall_score,
            "compliance_category": metrics.category.value,
            "confidence_level": metrics.confidence_level,
            "target_vs_achieved": {
                "target_quality_level": compliance_report.target_quality_level.value,
                "achieved_quality_level": compliance_report.achieved_quality_level.value,
                "gap_exists": compliance_report.achieved_quality_level != compliance_report.target_quality_level
            },
            "key_findings": key_findings,
            "critical_issues": critical_issues,
            "total_gaps_identified": len(gaps),
            "high_priority_gaps": len([g for g in gaps if g.impact_severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]]),
            "priority_actions": high_priority_actions,
            "estimated_improvement_timeline": self._estimate_improvement_timeline(recommendations),
            "estimated_improvement_cost": self._estimate_improvement_cost(recommendations),
            "next_steps": compliance_report.next_steps[:3]
        }

    def _estimate_improvement_timeline(self, recommendations: List[RecommendationItem]) -> str:
        """Estimate overall improvement timeline from recommendations."""
        high_priority_recs = [
            rec for rec in recommendations
            if rec.priority in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ]

        if len(high_priority_recs) >= 3:
            return "6-12 weeks"
        elif len(high_priority_recs) >= 1:
            return "3-6 weeks"
        else:
            return "1-3 weeks"

    def _estimate_improvement_cost(self, recommendations: List[RecommendationItem]) -> str:
        """Estimate overall improvement cost from recommendations."""
        # Simplified cost estimation
        high_priority_count = len([
            rec for rec in recommendations
            if rec.priority in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ])

        if high_priority_count >= 3:
            return "£10,000 - £25,000"
        elif high_priority_count >= 1:
            return "£5,000 - £15,000"
        else:
            return "£1,000 - £5,000"

    def export_compliance_data(self, compliance_report: ComplianceReport,
                             metrics: ComplianceMetrics,
                             gaps: List[GapAnalysis],
                             recommendations: List[RecommendationItem],
                             output_path: str) -> None:
        """
        Export comprehensive compliance data to file.

        Args:
            compliance_report: Compliance report data
            metrics: Compliance metrics
            gaps: Gap analysis results
            recommendations: Recommendations
            output_path: Output file path
        """
        export_data = {
            "compliance_report": asdict(compliance_report),
            "compliance_metrics": asdict(metrics),
            "gap_analysis": [asdict(gap) for gap in gaps],
            "recommendations": [asdict(rec) for rec in recommendations],
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0"
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Compliance data exported to {output_path}")

    def generate_summary_statistics(self, compliance_reports: List[ComplianceReport]) -> Dict[str, Any]:
        """
        Generate summary statistics across multiple compliance reports.

        Args:
            compliance_reports: List of compliance reports to summarize

        Returns:
            Summary statistics
        """
        if not compliance_reports:
            return {}

        metrics_list = [self.calculate_compliance_metrics(report) for report in compliance_reports]

        return {
            "total_surveys": len(compliance_reports),
            "average_compliance_score": np.mean([m.overall_score for m in metrics_list]),
            "compliance_categories": {
                category.value: len([m for m in metrics_list if m.category == category])
                for category in ComplianceCategory
            },
            "quality_level_distribution": {
                level.value: len([r for r in compliance_reports if r.achieved_quality_level == level])
                for level in QualityLevel
            },
            "average_scores_by_aspect": {
                "method_compliance": np.mean([m.method_compliance_score for m in metrics_list]),
                "deliverable_compliance": np.mean([m.deliverable_compliance_score for m in metrics_list]),
                "accuracy_compliance": np.mean([m.accuracy_compliance_score for m in metrics_list]),
                "environmental_suitability": np.mean([m.environmental_suitability_score for m in metrics_list])
            },
            "trends": {
                "improving_surveys": len([m for m in metrics_list if m.overall_score >= 0.8]),
                "needs_attention": len([m for m in metrics_list if m.overall_score < 0.6])
            }
        }