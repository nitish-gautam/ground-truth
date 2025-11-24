"""
Performance Benchmarking and Monitoring System for GPR Signal Processing.

This module provides comprehensive performance benchmarking capabilities including
automated performance monitoring against baseline, regression testing for model updates,
A/B testing framework for algorithm improvements, and performance degradation detection.
"""

import json
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import queue
import sqlite3
from contextlib import contextmanager
import memory_profiler
import warnings

from ..validation.accuracy.accuracy_assessor import AccuracyAssessor, DetectedUtility, GroundTruthUtility
from ..validation.statistical.statistical_validator import StatisticalValidator


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class BenchmarkMetric(Enum):
    """Available benchmarking metrics."""
    PROCESSING_TIME = "processing_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    DETECTION_ACCURACY = "detection_accuracy"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"


class AlertLevel(Enum):
    """Alert levels for performance monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run."""
    processing_time: float
    memory_usage_mb: float
    cpu_utilization: float
    detection_accuracy: float
    throughput: float  # items per second
    latency: float  # seconds per item
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    test_id: str = ""
    algorithm_version: str = ""
    data_size: int = 0
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkBaseline:
    """Baseline performance metrics for comparison."""
    metric_name: BenchmarkMetric
    baseline_value: float
    tolerance_percentage: float
    created_at: datetime
    description: str = ""
    test_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert notification."""
    level: AlertLevel
    metric: BenchmarkMetric
    current_value: float
    baseline_value: float
    deviation_percentage: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    test_id: str = ""


@dataclass
class RegressionTestResult:
    """Result of regression testing."""
    test_name: str
    baseline_metrics: PerformanceMetrics
    current_metrics: PerformanceMetrics
    performance_change: Dict[BenchmarkMetric, float]  # percentage change
    regression_detected: bool
    alerts: List[PerformanceAlert]
    statistical_significance: Dict[str, Any]


@dataclass
class ABTestResult:
    """Result of A/B testing."""
    test_name: str
    variant_a_metrics: List[PerformanceMetrics]
    variant_b_metrics: List[PerformanceMetrics]
    statistical_comparison: Dict[BenchmarkMetric, Dict[str, Any]]
    winner: str  # "A", "B", or "No significant difference"
    confidence_level: float
    effect_sizes: Dict[BenchmarkMetric, float]


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize performance monitor.

        Args:
            monitoring_interval: Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                memory_usage_mb = memory_info.used / (1024 * 1024)

                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_utilization': cpu_percent,
                    'memory_usage_mb': memory_usage_mb,
                    'memory_percentage': memory_info.percent
                }

                self.metrics_queue.put(metrics)
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def get_recent_metrics(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """Get recent monitoring metrics."""
        metrics = []
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)

        # Drain the queue
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                metric_time = datetime.fromisoformat(metric['timestamp'])
                if metric_time >= cutoff_time:
                    metrics.append(metric)
            except queue.Empty:
                break

        return metrics


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system."""

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the performance benchmarker.

        Args:
            database_path: Path to SQLite database for storing results
        """
        self.database_path = database_path or Path("performance_benchmark.db")
        self.accuracy_assessor = AccuracyAssessor()
        self.statistical_validator = StatisticalValidator()
        self.performance_monitor = PerformanceMonitor()
        self.baselines: Dict[str, BenchmarkBaseline] = {}

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing performance metrics."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    algorithm_version TEXT,
                    timestamp DATETIME,
                    processing_time REAL,
                    memory_usage_mb REAL,
                    cpu_utilization REAL,
                    detection_accuracy REAL,
                    throughput REAL,
                    latency REAL,
                    error_rate REAL,
                    data_size INTEGER,
                    additional_metrics TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    baseline_value REAL,
                    tolerance_percentage REAL,
                    created_at DATETIME,
                    description TEXT,
                    test_conditions TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT,
                    metric TEXT,
                    current_value REAL,
                    baseline_value REAL,
                    deviation_percentage REAL,
                    message TEXT,
                    timestamp DATETIME,
                    test_id TEXT
                )
            """)

            conn.commit()

    @contextmanager
    def _get_db_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def benchmark_function(
        self,
        func: Callable,
        test_data: Any,
        ground_truth: Optional[List[GroundTruthUtility]] = None,
        test_id: str = "",
        algorithm_version: str = "",
        iterations: int = 1
    ) -> PerformanceMetrics:
        """
        Benchmark a function's performance.

        Args:
            func: Function to benchmark
            test_data: Test data to pass to function
            ground_truth: Ground truth for accuracy assessment
            test_id: Identifier for this test
            algorithm_version: Version of the algorithm
            iterations: Number of iterations to run

        Returns:
            PerformanceMetrics with benchmarking results
        """
        logger.info(f"Benchmarking function with {iterations} iterations")

        processing_times = []
        memory_usages = []
        cpu_utilizations = []
        errors = 0

        # Start monitoring
        self.performance_monitor.start_monitoring()

        for i in range(iterations):
            try:
                # Measure processing time
                start_time = time.perf_counter()

                # Monitor memory before execution
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024 * 1024)  # MB

                # Execute function
                result = func(test_data)

                # Measure time
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                processing_times.append(processing_time)

                # Monitor memory after execution
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage = memory_after - memory_before
                memory_usages.append(max(memory_usage, 0))  # Ensure non-negative

                # Get CPU utilization
                cpu_usage = psutil.cpu_percent(interval=0.1)
                cpu_utilizations.append(cpu_usage)

            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
                errors += 1

        # Stop monitoring
        self.performance_monitor.stop_monitoring()

        # Calculate aggregate metrics
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        avg_memory_usage = np.mean(memory_usages) if memory_usages else 0.0
        avg_cpu_utilization = np.mean(cpu_utilizations) if cpu_utilizations else 0.0

        # Calculate accuracy if ground truth provided
        detection_accuracy = 0.0
        if ground_truth and hasattr(result, '__iter__'):
            try:
                # Assume result is list of DetectedUtility objects
                detections = result if isinstance(result, list) else []
                accuracy_results = self.accuracy_assessor.assess_detection_performance(
                    detections, ground_truth
                )
                detection_accuracy = accuracy_results.f1_score
            except Exception as e:
                logger.warning(f"Could not calculate detection accuracy: {e}")

        # Calculate throughput and latency
        data_size = len(test_data) if hasattr(test_data, '__len__') else 1
        throughput = data_size / avg_processing_time if avg_processing_time > 0 else 0.0
        latency = avg_processing_time / data_size if data_size > 0 else avg_processing_time

        # Calculate error rate
        error_rate = errors / iterations if iterations > 0 else 0.0

        metrics = PerformanceMetrics(
            processing_time=avg_processing_time,
            memory_usage_mb=avg_memory_usage,
            cpu_utilization=avg_cpu_utilization,
            detection_accuracy=detection_accuracy,
            throughput=throughput,
            latency=latency,
            error_rate=error_rate,
            test_id=test_id,
            algorithm_version=algorithm_version,
            data_size=data_size
        )

        # Store in database
        self._store_metrics(metrics)

        return metrics

    def set_baseline(
        self,
        metric: BenchmarkMetric,
        baseline_value: float,
        tolerance_percentage: float = 10.0,
        description: str = "",
        test_conditions: Optional[Dict[str, Any]] = None
    ):
        """
        Set baseline for a performance metric.

        Args:
            metric: Benchmark metric
            baseline_value: Baseline value
            tolerance_percentage: Acceptable tolerance percentage
            description: Description of the baseline
            test_conditions: Test conditions when baseline was established
        """
        baseline = BenchmarkBaseline(
            metric_name=metric,
            baseline_value=baseline_value,
            tolerance_percentage=tolerance_percentage,
            created_at=datetime.now(),
            description=description,
            test_conditions=test_conditions or {}
        )

        self.baselines[metric.value] = baseline

        # Store in database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO baselines (metric_name, baseline_value, tolerance_percentage,
                                     created_at, description, test_conditions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric.value, baseline_value, tolerance_percentage,
                baseline.created_at, description, json.dumps(test_conditions or {})
            ))
            conn.commit()

        logger.info(f"Baseline set for {metric.value}: {baseline_value} ±{tolerance_percentage}%")

    def check_performance_regression(
        self,
        current_metrics: PerformanceMetrics,
        confidence_level: float = 0.95
    ) -> List[PerformanceAlert]:
        """
        Check for performance regression against baselines.

        Args:
            current_metrics: Current performance metrics
            confidence_level: Statistical confidence level

        Returns:
            List of performance alerts
        """
        alerts = []

        # Check each metric against its baseline
        metric_values = {
            BenchmarkMetric.PROCESSING_TIME: current_metrics.processing_time,
            BenchmarkMetric.MEMORY_USAGE: current_metrics.memory_usage_mb,
            BenchmarkMetric.CPU_UTILIZATION: current_metrics.cpu_utilization,
            BenchmarkMetric.DETECTION_ACCURACY: current_metrics.detection_accuracy,
            BenchmarkMetric.THROUGHPUT: current_metrics.throughput,
            BenchmarkMetric.LATENCY: current_metrics.latency,
            BenchmarkMetric.ERROR_RATE: current_metrics.error_rate
        }

        for metric, current_value in metric_values.items():
            if metric.value in self.baselines:
                baseline = self.baselines[metric.value]
                deviation = self._calculate_deviation_percentage(
                    current_value, baseline.baseline_value
                )

                # Determine alert level
                alert_level = self._determine_alert_level(
                    metric, deviation, baseline.tolerance_percentage
                )

                if alert_level != AlertLevel.INFO:
                    alert = PerformanceAlert(
                        level=alert_level,
                        metric=metric,
                        current_value=current_value,
                        baseline_value=baseline.baseline_value,
                        deviation_percentage=deviation,
                        message=self._generate_alert_message(
                            metric, current_value, baseline.baseline_value, deviation
                        ),
                        test_id=current_metrics.test_id
                    )
                    alerts.append(alert)

                    # Store alert in database
                    self._store_alert(alert)

        return alerts

    def regression_test(
        self,
        test_name: str,
        current_function: Callable,
        baseline_function: Callable,
        test_data: Any,
        ground_truth: Optional[List[GroundTruthUtility]] = None,
        iterations: int = 5
    ) -> RegressionTestResult:
        """
        Perform regression testing between current and baseline implementations.

        Args:
            test_name: Name of the regression test
            current_function: Current implementation
            baseline_function: Baseline implementation
            test_data: Test data
            ground_truth: Ground truth for accuracy assessment
            iterations: Number of test iterations

        Returns:
            RegressionTestResult with comparison results
        """
        logger.info(f"Running regression test: {test_name}")

        # Benchmark baseline function
        baseline_metrics = self.benchmark_function(
            baseline_function, test_data, ground_truth,
            test_id=f"{test_name}_baseline", iterations=iterations
        )

        # Benchmark current function
        current_metrics = self.benchmark_function(
            current_function, test_data, ground_truth,
            test_id=f"{test_name}_current", iterations=iterations
        )

        # Calculate performance changes
        performance_change = {}
        metrics_to_compare = [
            BenchmarkMetric.PROCESSING_TIME,
            BenchmarkMetric.MEMORY_USAGE,
            BenchmarkMetric.CPU_UTILIZATION,
            BenchmarkMetric.DETECTION_ACCURACY,
            BenchmarkMetric.THROUGHPUT,
            BenchmarkMetric.LATENCY,
            BenchmarkMetric.ERROR_RATE
        ]

        for metric in metrics_to_compare:
            baseline_value = getattr(baseline_metrics, metric.value.replace('_', '_'))
            current_value = getattr(current_metrics, metric.value.replace('_', '_'))
            change = self._calculate_deviation_percentage(current_value, baseline_value)
            performance_change[metric] = change

        # Check for regression
        alerts = self.check_performance_regression(current_metrics)
        regression_detected = any(alert.level == AlertLevel.CRITICAL for alert in alerts)

        # Statistical significance testing
        statistical_significance = self._test_statistical_significance(
            baseline_metrics, current_metrics
        )

        return RegressionTestResult(
            test_name=test_name,
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            performance_change=performance_change,
            regression_detected=regression_detected,
            alerts=alerts,
            statistical_significance=statistical_significance
        )

    def ab_test(
        self,
        test_name: str,
        variant_a_function: Callable,
        variant_b_function: Callable,
        test_data: Any,
        ground_truth: Optional[List[GroundTruthUtility]] = None,
        iterations: int = 10,
        confidence_level: float = 0.95
    ) -> ABTestResult:
        """
        Perform A/B testing between two algorithm variants.

        Args:
            test_name: Name of the A/B test
            variant_a_function: Variant A function
            variant_b_function: Variant B function
            test_data: Test data
            ground_truth: Ground truth for accuracy assessment
            iterations: Number of test iterations per variant
            confidence_level: Statistical confidence level

        Returns:
            ABTestResult with comparison results
        """
        logger.info(f"Running A/B test: {test_name}")

        # Run multiple iterations for each variant
        variant_a_metrics = []
        variant_b_metrics = []

        for i in range(iterations):
            # Test variant A
            metrics_a = self.benchmark_function(
                variant_a_function, test_data, ground_truth,
                test_id=f"{test_name}_a_{i}", iterations=1
            )
            variant_a_metrics.append(metrics_a)

            # Test variant B
            metrics_b = self.benchmark_function(
                variant_b_function, test_data, ground_truth,
                test_id=f"{test_name}_b_{i}", iterations=1
            )
            variant_b_metrics.append(metrics_b)

        # Statistical comparison
        statistical_comparison = self._compare_ab_variants(
            variant_a_metrics, variant_b_metrics, confidence_level
        )

        # Determine winner
        winner = self._determine_ab_winner(statistical_comparison)

        # Calculate effect sizes
        effect_sizes = self._calculate_ab_effect_sizes(
            variant_a_metrics, variant_b_metrics
        )

        return ABTestResult(
            test_name=test_name,
            variant_a_metrics=variant_a_metrics,
            variant_b_metrics=variant_b_metrics,
            statistical_comparison=statistical_comparison,
            winner=winner,
            confidence_level=confidence_level,
            effect_sizes=effect_sizes
        )

    def generate_performance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            start_date: Start date for report period
            end_date: End date for report period

        Returns:
            Dictionary containing performance report
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        with self._get_db_connection() as conn:
            # Get performance metrics
            metrics_df = pd.read_sql_query("""
                SELECT * FROM performance_metrics
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, conn, params=(start_date, end_date))

            # Get alerts
            alerts_df = pd.read_sql_query("""
                SELECT * FROM alerts
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, conn, params=(start_date, end_date))

        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary_statistics': self._calculate_summary_statistics(metrics_df),
            'performance_trends': self._calculate_performance_trends(metrics_df),
            'alert_summary': self._summarize_alerts(alerts_df),
            'baseline_compliance': self._check_baseline_compliance(metrics_df),
            'recommendations': self._generate_performance_recommendations(metrics_df, alerts_df)
        }

        return report

    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    test_id, algorithm_version, timestamp, processing_time,
                    memory_usage_mb, cpu_utilization, detection_accuracy,
                    throughput, latency, error_rate, data_size, additional_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.test_id, metrics.algorithm_version, metrics.timestamp,
                metrics.processing_time, metrics.memory_usage_mb, metrics.cpu_utilization,
                metrics.detection_accuracy, metrics.throughput, metrics.latency,
                metrics.error_rate, metrics.data_size, json.dumps(metrics.additional_metrics)
            ))
            conn.commit()

    def _store_alert(self, alert: PerformanceAlert):
        """Store performance alert in database."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (level, metric, current_value, baseline_value,
                                  deviation_percentage, message, timestamp, test_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.level.value, alert.metric.value, alert.current_value,
                alert.baseline_value, alert.deviation_percentage,
                alert.message, alert.timestamp, alert.test_id
            ))
            conn.commit()

    def _calculate_deviation_percentage(self, current_value: float, baseline_value: float) -> float:
        """Calculate percentage deviation from baseline."""
        if baseline_value == 0:
            return 0.0 if current_value == 0 else 100.0
        return ((current_value - baseline_value) / baseline_value) * 100

    def _determine_alert_level(
        self,
        metric: BenchmarkMetric,
        deviation: float,
        tolerance: float
    ) -> AlertLevel:
        """Determine alert level based on deviation."""
        abs_deviation = abs(deviation)

        # For metrics where lower is better (processing time, memory, latency, error rate)
        lower_is_better = metric in [
            BenchmarkMetric.PROCESSING_TIME,
            BenchmarkMetric.MEMORY_USAGE,
            BenchmarkMetric.LATENCY,
            BenchmarkMetric.ERROR_RATE
        ]

        # For metrics where higher is better (accuracy, throughput)
        higher_is_better = metric in [
            BenchmarkMetric.DETECTION_ACCURACY,
            BenchmarkMetric.THROUGHPUT
        ]

        if abs_deviation <= tolerance:
            return AlertLevel.INFO
        elif abs_deviation <= tolerance * 2:
            # Check if this is a good or bad deviation
            if (lower_is_better and deviation < 0) or (higher_is_better and deviation > 0):
                return AlertLevel.INFO  # Good performance improvement
            else:
                return AlertLevel.WARNING
        else:
            # Check if this is a good or bad deviation
            if (lower_is_better and deviation < 0) or (higher_is_better and deviation > 0):
                return AlertLevel.INFO  # Significant improvement
            else:
                return AlertLevel.CRITICAL

    def _generate_alert_message(
        self,
        metric: BenchmarkMetric,
        current_value: float,
        baseline_value: float,
        deviation: float
    ) -> str:
        """Generate alert message."""
        direction = "increased" if deviation > 0 else "decreased"
        return (f"{metric.value.replace('_', ' ').title()} has {direction} by {abs(deviation):.1f}% "
                f"(current: {current_value:.3f}, baseline: {baseline_value:.3f})")

    def _test_statistical_significance(
        self,
        baseline_metrics: PerformanceMetrics,
        current_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Test statistical significance of performance changes."""
        # This is a simplified version - in practice, you'd need multiple samples
        return {
            'processing_time_change': current_metrics.processing_time - baseline_metrics.processing_time,
            'memory_change': current_metrics.memory_usage_mb - baseline_metrics.memory_usage_mb,
            'accuracy_change': current_metrics.detection_accuracy - baseline_metrics.detection_accuracy,
            'note': 'Full statistical testing requires multiple samples'
        }

    def _compare_ab_variants(
        self,
        variant_a_metrics: List[PerformanceMetrics],
        variant_b_metrics: List[PerformanceMetrics],
        confidence_level: float
    ) -> Dict[BenchmarkMetric, Dict[str, Any]]:
        """Compare A/B test variants statistically."""
        comparison = {}

        metrics_to_compare = [
            BenchmarkMetric.PROCESSING_TIME,
            BenchmarkMetric.MEMORY_USAGE,
            BenchmarkMetric.DETECTION_ACCURACY,
            BenchmarkMetric.THROUGHPUT
        ]

        for metric in metrics_to_compare:
            # Extract values for comparison
            a_values = [getattr(m, metric.value) for m in variant_a_metrics]
            b_values = [getattr(m, metric.value) for m in variant_b_metrics]

            # Perform statistical test
            test_result = self.statistical_validator.compare_groups_statistical_test(
                np.array(a_values), np.array(b_values),
                confidence_level=confidence_level
            )

            comparison[metric] = {
                'variant_a_mean': np.mean(a_values),
                'variant_b_mean': np.mean(b_values),
                'p_value': test_result.p_value,
                'significant': test_result.significant,
                'effect_size': test_result.effect_size
            }

        return comparison

    def _determine_ab_winner(self, statistical_comparison: Dict[BenchmarkMetric, Dict[str, Any]]) -> str:
        """Determine winner of A/B test."""
        significant_improvements = []

        for metric, comparison in statistical_comparison.items():
            if comparison['significant']:
                a_mean = comparison['variant_a_mean']
                b_mean = comparison['variant_b_mean']

                # Determine which variant is better for this metric
                if metric in [BenchmarkMetric.DETECTION_ACCURACY, BenchmarkMetric.THROUGHPUT]:
                    # Higher is better
                    if b_mean > a_mean:
                        significant_improvements.append('B')
                    else:
                        significant_improvements.append('A')
                else:
                    # Lower is better
                    if b_mean < a_mean:
                        significant_improvements.append('B')
                    else:
                        significant_improvements.append('A')

        if not significant_improvements:
            return "No significant difference"

        # Count votes for each variant
        a_votes = significant_improvements.count('A')
        b_votes = significant_improvements.count('B')

        if a_votes > b_votes:
            return "A"
        elif b_votes > a_votes:
            return "B"
        else:
            return "Tie"

    def _calculate_ab_effect_sizes(
        self,
        variant_a_metrics: List[PerformanceMetrics],
        variant_b_metrics: List[PerformanceMetrics]
    ) -> Dict[BenchmarkMetric, float]:
        """Calculate effect sizes for A/B test."""
        effect_sizes = {}

        metrics_to_compare = [
            BenchmarkMetric.PROCESSING_TIME,
            BenchmarkMetric.MEMORY_USAGE,
            BenchmarkMetric.DETECTION_ACCURACY,
            BenchmarkMetric.THROUGHPUT
        ]

        for metric in metrics_to_compare:
            a_values = np.array([getattr(m, metric.value) for m in variant_a_metrics])
            b_values = np.array([getattr(m, metric.value) for m in variant_b_metrics])

            effect_size_analysis = self.statistical_validator.effect_size_analysis(a_values, b_values)
            effect_sizes[metric] = effect_size_analysis['cohens_d']

        return effect_sizes

    def _calculate_summary_statistics(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for performance metrics."""
        if metrics_df.empty:
            return {}

        numeric_columns = ['processing_time', 'memory_usage_mb', 'cpu_utilization',
                          'detection_accuracy', 'throughput', 'latency', 'error_rate']

        summary = {}
        for col in numeric_columns:
            if col in metrics_df.columns:
                summary[col] = {
                    'mean': metrics_df[col].mean(),
                    'std': metrics_df[col].std(),
                    'min': metrics_df[col].min(),
                    'max': metrics_df[col].max(),
                    'median': metrics_df[col].median()
                }

        return summary

    def _calculate_performance_trends(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if metrics_df.empty:
            return {}

        trends = {}
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df = metrics_df.sort_values('timestamp')

        numeric_columns = ['processing_time', 'memory_usage_mb', 'detection_accuracy']

        for col in numeric_columns:
            if col in metrics_df.columns and len(metrics_df[col]) > 1:
                # Calculate linear trend
                x = np.arange(len(metrics_df))
                y = metrics_df[col].values
                slope, intercept = np.polyfit(x, y, 1)

                trends[col] = {
                    'slope': slope,
                    'direction': 'improving' if slope < 0 else 'degrading',
                    'recent_mean': metrics_df[col].tail(5).mean(),
                    'historical_mean': metrics_df[col].head(5).mean()
                }

        return trends

    def _summarize_alerts(self, alerts_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize alerts."""
        if alerts_df.empty:
            return {'total_alerts': 0}

        return {
            'total_alerts': len(alerts_df),
            'critical_alerts': len(alerts_df[alerts_df['level'] == 'critical']),
            'warning_alerts': len(alerts_df[alerts_df['level'] == 'warning']),
            'most_frequent_metric': alerts_df['metric'].mode().iloc[0] if not alerts_df.empty else None
        }

    def _check_baseline_compliance(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Check compliance with established baselines."""
        if metrics_df.empty:
            return {}

        compliance = {}
        for baseline_key, baseline in self.baselines.items():
            metric_column = baseline_key
            if metric_column in metrics_df.columns:
                recent_values = metrics_df[metric_column].tail(10)
                within_tolerance = sum(
                    abs(self._calculate_deviation_percentage(val, baseline.baseline_value))
                    <= baseline.tolerance_percentage
                    for val in recent_values
                )
                compliance[baseline_key] = {
                    'compliance_rate': within_tolerance / len(recent_values),
                    'recent_average': recent_values.mean(),
                    'baseline_value': baseline.baseline_value
                }

        return compliance

    def _generate_performance_recommendations(
        self,
        metrics_df: pd.DataFrame,
        alerts_df: pd.DataFrame
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if metrics_df.empty:
            return recommendations

        # Check for consistent issues
        if not alerts_df.empty:
            frequent_alerts = alerts_df['metric'].value_counts()
            if len(frequent_alerts) > 0:
                most_problematic = frequent_alerts.index[0]
                recommendations.append(
                    f"Focus on optimizing {most_problematic.replace('_', ' ')} - "
                    f"it has generated {frequent_alerts.iloc[0]} alerts recently"
                )

        # Check for performance trends
        recent_accuracy = metrics_df['detection_accuracy'].tail(10).mean()
        historical_accuracy = metrics_df['detection_accuracy'].head(10).mean()

        if recent_accuracy < historical_accuracy * 0.95:
            recommendations.append("Detection accuracy is declining - review recent algorithm changes")

        recent_memory = metrics_df['memory_usage_mb'].tail(10).mean()
        if recent_memory > 1000:  # More than 1GB
            recommendations.append("High memory usage detected - consider memory optimization")

        return recommendations


def create_performance_benchmarker(database_path: Optional[str] = None) -> PerformanceBenchmarker:
    """
    Factory function to create a performance benchmarker.

    Args:
        database_path: Path to SQLite database for storing results

    Returns:
        Configured PerformanceBenchmarker instance
    """
    db_path = Path(database_path) if database_path else None
    return PerformanceBenchmarker(database_path=db_path)