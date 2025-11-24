"""
Statistical Validation Framework for GPR Signal Processing.

This module provides comprehensive statistical analysis capabilities including
cross-validation, bootstrap confidence intervals, hypothesis testing, and
performance metric calculations for GPR detection accuracy validation.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.utils import resample
import logging
from typing import Dict, List, Any, Optional, Tuple, NamedTuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalTest(Enum):
    """Available statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"


class ConfidenceLevel(Enum):
    """Standard confidence levels."""
    CL_90 = 0.90
    CL_95 = 0.95
    CL_99 = 0.99


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    confidence_level: float
    critical_value: Optional[float] = None
    effect_size: Optional[float] = None
    interpretation: str = ""
    significant: bool = False

    def __post_init__(self):
        """Post-initialization processing."""
        alpha = 1 - self.confidence_level
        self.significant = self.p_value < alpha


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""
    original_statistic: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_bootstrap_samples: int


@dataclass
class CrossValidationResult:
    """Result of cross-validation analysis."""
    cv_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    fold_count: int
    scoring_metric: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    sensitivity: float
    auc_score: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None


@dataclass
class RegressionMetrics:
    """Regression analysis metrics."""
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    mae: float
    mape: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None


class StatisticalValidator:
    """Comprehensive statistical validation framework."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the statistical validator.

        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def cross_validate_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        cv_folds: int = 5,
        scoring_metric: str = 'accuracy',
        stratified: bool = True,
        confidence_level: float = 0.95
    ) -> CrossValidationResult:
        """
        Perform k-fold cross-validation with confidence intervals.

        Args:
            X: Feature matrix
            y: Target vector
            model: Machine learning model with fit/predict methods
            cv_folds: Number of cross-validation folds
            scoring_metric: Scoring metric to use
            stratified: Whether to use stratified k-fold
            confidence_level: Confidence level for intervals

        Returns:
            CrossValidationResult with validation metrics
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation with {scoring_metric}")

        # Choose cross-validation strategy
        if stratified and self._is_classification_task(y):
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        cv_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit model and make predictions
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            # Calculate score
            score = self._calculate_score(y_val, predictions, scoring_metric)
            cv_scores.append(score)

        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=cv_folds-1)
        margin_error = t_critical * (std_score / np.sqrt(cv_folds))
        confidence_interval = (mean_score - margin_error, mean_score + margin_error)

        return CrossValidationResult(
            cv_scores=cv_scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=confidence_interval,
            fold_count=cv_folds,
            scoring_metric=scoring_metric
        )

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> BootstrapResult:
        """
        Calculate bootstrap confidence intervals for a statistic.

        Args:
            data: Input data array
            statistic_func: Function to calculate statistic (e.g., np.mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for interval

        Returns:
            BootstrapResult with confidence intervals
        """
        logger.info(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples")

        # Calculate original statistic
        original_statistic = statistic_func(data)

        # Bootstrap sampling
        bootstrap_statistics = []
        for _ in range(n_bootstrap):
            bootstrap_sample = resample(data, random_state=None)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(bootstrap_stat)

        bootstrap_statistics = np.array(bootstrap_statistics)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        confidence_interval = (
            np.percentile(bootstrap_statistics, lower_percentile),
            np.percentile(bootstrap_statistics, upper_percentile)
        )

        return BootstrapResult(
            original_statistic=original_statistic,
            bootstrap_mean=np.mean(bootstrap_statistics),
            bootstrap_std=np.std(bootstrap_statistics),
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            n_bootstrap_samples=n_bootstrap
        )

    def compare_groups_statistical_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        test_type: StatisticalTest = StatisticalTest.T_TEST,
        confidence_level: float = 0.95,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Perform statistical test to compare two groups.

        Args:
            group1: First group data
            group2: Second group data
            test_type: Type of statistical test
            confidence_level: Confidence level for test
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            StatisticalTestResult with test results
        """
        logger.info(f"Performing {test_type.value} test to compare groups")

        if test_type == StatisticalTest.T_TEST:
            result = self._perform_t_test(group1, group2, confidence_level, alternative)
        elif test_type == StatisticalTest.MANN_WHITNEY:
            result = self._perform_mann_whitney_test(group1, group2, confidence_level, alternative)
        elif test_type == StatisticalTest.KOLMOGOROV_SMIRNOV:
            result = self._perform_ks_test(group1, group2, confidence_level, alternative)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        return result

    def compare_multiple_groups(
        self,
        groups: List[np.ndarray],
        group_names: List[str],
        test_type: StatisticalTest = StatisticalTest.ANOVA,
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """
        Perform statistical test to compare multiple groups.

        Args:
            groups: List of group data arrays
            group_names: Names for each group
            test_type: Type of statistical test
            confidence_level: Confidence level for test

        Returns:
            StatisticalTestResult with test results
        """
        logger.info(f"Performing {test_type.value} test to compare {len(groups)} groups")

        if test_type == StatisticalTest.ANOVA:
            result = self._perform_anova_test(groups, confidence_level)
        elif test_type == StatisticalTest.KRUSKAL_WALLIS:
            result = self._perform_kruskal_wallis_test(groups, confidence_level)
        else:
            raise ValueError(f"Unsupported test type for multiple groups: {test_type}")

        return result

    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_labels: Optional[List[str]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC calculation)
            class_labels: Class label names

        Returns:
            PerformanceMetrics with all calculated metrics
        """
        logger.info("Calculating comprehensive performance metrics")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate specificity and sensitivity
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # AUC score (if probabilities provided)
        auc_score = None
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    auc_score = roc_auc_score(y_true, y_proba)
                else:  # Multi-class
                    auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate AUC score: {e}")

        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            sensitivity=sensitivity,
            auc_score=auc_score,
            confusion_matrix=cm,
            classification_report=class_report
        )

    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> RegressionMetrics:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            X: Feature matrix (for adjusted R-squared)

        Returns:
            RegressionMetrics with all calculated metrics
        """
        logger.info("Calculating regression metrics")

        # Calculate residuals
        residuals = y_true - y_pred

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Adjusted R-squared
        adjusted_r_squared = r_squared
        if X is not None and X.shape[0] > X.shape[1]:
            n, p = X.shape
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        # RMSE and MAE
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(residuals / y_true)) * 100 if np.all(y_true != 0) else np.inf

        return RegressionMetrics(
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            rmse=rmse,
            mae=mae,
            mape=mape,
            residuals=residuals,
            fitted_values=y_pred
        )

    def test_environmental_factor_significance(
        self,
        performance_data: pd.DataFrame,
        environmental_factor: str,
        performance_metric: str,
        test_type: StatisticalTest = StatisticalTest.ANOVA
    ) -> StatisticalTestResult:
        """
        Test the statistical significance of environmental factors on performance.

        Args:
            performance_data: DataFrame with performance and environmental data
            environmental_factor: Name of environmental factor column
            performance_metric: Name of performance metric column
            test_type: Statistical test to use

        Returns:
            StatisticalTestResult with significance test results
        """
        logger.info(f"Testing significance of {environmental_factor} on {performance_metric}")

        # Group data by environmental factor
        groups = []
        group_names = []
        for factor_value in performance_data[environmental_factor].unique():
            group_data = performance_data[performance_data[environmental_factor] == factor_value][performance_metric].values
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(str(factor_value))

        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups for comparison, found {len(groups)}")

        if len(groups) == 2:
            return self.compare_groups_statistical_test(groups[0], groups[1], test_type)
        else:
            return self.compare_multiple_groups(groups, group_names, test_type)

    def correlation_analysis(
        self,
        data: pd.DataFrame,
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform correlation analysis on dataset.

        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Tuple of (correlation_matrix, p_values_matrix)
        """
        logger.info(f"Performing {method} correlation analysis")

        # Calculate correlation matrix
        correlation_matrix = data.corr(method=method)

        # Calculate p-values for correlations
        n = len(data)
        p_values = np.zeros((data.shape[1], data.shape[1]))

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i != j:
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(data.iloc[:, i], data.iloc[:, j])
                    else:  # kendall
                        _, p_val = stats.kendalltau(data.iloc[:, i], data.iloc[:, j])
                    p_values[i, j] = p_val

        p_values_df = pd.DataFrame(p_values, index=data.columns, columns=data.columns)

        return correlation_matrix, p_values_df

    def effect_size_analysis(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate effect sizes for group comparisons.

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            Dictionary with various effect size measures
        """
        logger.info("Calculating effect sizes")

        # Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                             (len(group2) - 1) * np.var(group2, ddof=1)) /
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0

        # Glass's delta
        glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1) if np.std(group2, ddof=1) != 0 else 0

        # Hedge's g (bias-corrected Cohen's d)
        n = len(group1) + len(group2)
        hedges_g = cohens_d * (1 - (3 / (4 * n - 9)))

        # Eta-squared (from t-test)
        t_stat, _ = stats.ttest_ind(group1, group2)
        eta_squared = t_stat**2 / (t_stat**2 + len(group1) + len(group2) - 2)

        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'hedges_g': hedges_g,
            'eta_squared': eta_squared
        }

    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Check if the task is classification based on target values."""
        return len(np.unique(y)) < len(y) / 10 or np.issubdtype(y.dtype, np.integer)

    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate score based on specified metric."""
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
        elif metric == 'recall':
            return precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
        elif metric == 'f1':
            return precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
        elif metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unsupported scoring metric: {metric}")

    def _perform_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float,
        alternative: str
    ) -> StatisticalTestResult:
        """Perform independent samples t-test."""
        statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)

        # Calculate effect size (Cohen's d)
        effect_size = self.effect_size_analysis(group1, group2)['cohens_d']

        # Critical value
        alpha = 1 - confidence_level
        df = len(group1) + len(group2) - 2
        critical_value = stats.t.ppf(1 - alpha/2, df) if alternative == 'two-sided' else stats.t.ppf(1 - alpha, df)

        return StatisticalTestResult(
            test_name="Independent Samples T-Test",
            statistic=statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            critical_value=critical_value,
            effect_size=effect_size,
            interpretation=self._interpret_t_test_result(statistic, p_value, effect_size, alpha)
        )

    def _perform_mann_whitney_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float,
        alternative: str
    ) -> StatisticalTestResult:
        """Perform Mann-Whitney U test."""
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

        return StatisticalTestResult(
            test_name="Mann-Whitney U Test",
            statistic=statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            interpretation=self._interpret_nonparametric_result(p_value, 1 - confidence_level)
        )

    def _perform_ks_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float,
        alternative: str
    ) -> StatisticalTestResult:
        """Perform Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(group1, group2, alternative=alternative)

        return StatisticalTestResult(
            test_name="Kolmogorov-Smirnov Test",
            statistic=statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            interpretation=self._interpret_ks_result(statistic, p_value, 1 - confidence_level)
        )

    def _perform_anova_test(
        self,
        groups: List[np.ndarray],
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform one-way ANOVA."""
        statistic, p_value = stats.f_oneway(*groups)

        # Calculate eta-squared (effect size)
        total_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(group) * (np.mean(group) - total_mean)**2 for group in groups)
        ss_total = sum(np.sum((group - total_mean)**2) for group in groups)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0

        return StatisticalTestResult(
            test_name="One-Way ANOVA",
            statistic=statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            effect_size=eta_squared,
            interpretation=self._interpret_anova_result(statistic, p_value, eta_squared, 1 - confidence_level)
        )

    def _perform_kruskal_wallis_test(
        self,
        groups: List[np.ndarray],
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Kruskal-Wallis test."""
        statistic, p_value = stats.kruskal(*groups)

        return StatisticalTestResult(
            test_name="Kruskal-Wallis Test",
            statistic=statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            interpretation=self._interpret_nonparametric_result(p_value, 1 - confidence_level)
        )

    def _interpret_t_test_result(
        self,
        statistic: float,
        p_value: float,
        effect_size: float,
        alpha: float
    ) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < alpha else "not significant"
        effect_interpretation = self._interpret_effect_size(effect_size)
        return f"The difference between groups is {significance} (p={p_value:.4f}). {effect_interpretation}"

    def _interpret_nonparametric_result(self, p_value: float, alpha: float) -> str:
        """Interpret non-parametric test results."""
        significance = "significant" if p_value < alpha else "not significant"
        return f"The difference between groups is {significance} (p={p_value:.4f})."

    def _interpret_ks_result(self, statistic: float, p_value: float, alpha: float) -> str:
        """Interpret Kolmogorov-Smirnov test results."""
        significance = "significant" if p_value < alpha else "not significant"
        return f"The distributions are {significance}ly different (KS statistic={statistic:.4f}, p={p_value:.4f})."

    def _interpret_anova_result(
        self,
        statistic: float,
        p_value: float,
        eta_squared: float,
        alpha: float
    ) -> str:
        """Interpret ANOVA results."""
        significance = "significant" if p_value < alpha else "not significant"
        effect_interpretation = self._interpret_eta_squared(eta_squared)
        return f"The group differences are {significance} (F={statistic:.4f}, p={p_value:.4f}). {effect_interpretation}"

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "The effect size is negligible."
        elif abs_effect < 0.5:
            return "The effect size is small."
        elif abs_effect < 0.8:
            return "The effect size is medium."
        else:
            return "The effect size is large."

    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "The effect size is negligible."
        elif eta_squared < 0.06:
            return "The effect size is small."
        elif eta_squared < 0.14:
            return "The effect size is medium."
        else:
            return "The effect size is large."


def create_statistical_validator(random_state: int = 42) -> StatisticalValidator:
    """
    Factory function to create a statistical validator.

    Args:
        random_state: Random state for reproducible results

    Returns:
        Configured StatisticalValidator instance
    """
    return StatisticalValidator(random_state=random_state)