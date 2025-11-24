"""
Environmental Factor Validation Framework for GPR Detection Performance.

This module provides comprehensive validation of environmental factor impacts on
GPR detection accuracy, including weather conditions, ground conditions, terrain
characteristics, and other environmental variables that affect detection performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import warnings

from ..statistical.statistical_validator import StatisticalValidator, StatisticalTest
from ..accuracy.accuracy_assessor import AccuracyAssessor, DetectedUtility, GroundTruthUtility


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class EnvironmentalFactor(Enum):
    """Environmental factors that can affect GPR performance."""
    WEATHER_CONDITION = "weather_condition"
    GROUND_CONDITION = "ground_condition"
    GROUND_PERMITTIVITY = "ground_permittivity"
    LAND_COVER = "land_cover"
    LAND_USE = "land_use"
    TERRAIN_LEVELLING = "terrain_levelling"
    TERRAIN_SMOOTHNESS = "terrain_smoothness"
    RUBBLE_PRESENCE = "rubble_presence"
    TREE_ROOTS_PRESENCE = "tree_roots_presence"
    POLLUTED_SOIL_PRESENCE = "polluted_soil_presence"
    BLAST_FURNACE_SLAG_PRESENCE = "blast_furnace_slag_presence"


class PerformanceMetric(Enum):
    """Performance metrics for environmental analysis."""
    DETECTION_RATE = "detection_rate"
    POSITION_ACCURACY = "position_accuracy"
    DEPTH_ACCURACY = "depth_accuracy"
    MATERIAL_ACCURACY = "material_accuracy"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    OVERALL_F1_SCORE = "overall_f1_score"


@dataclass
class EnvironmentalConditions:
    """Environmental conditions for a survey location."""
    weather_condition: str
    ground_condition: str
    ground_permittivity: float
    land_cover: str
    land_use: str
    terrain_levelling: str
    terrain_smoothness: str
    rubble_presence: bool = False
    tree_roots_presence: bool = False
    polluted_soil_presence: bool = False
    blast_furnace_slag_presence: bool = False


@dataclass
class SurveyResults:
    """Survey results for a specific location."""
    location_id: str
    environmental_conditions: EnvironmentalConditions
    detections: List[DetectedUtility]
    ground_truth: List[GroundTruthUtility]
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnvironmentalImpactAnalysis:
    """Results of environmental factor impact analysis."""
    factor: EnvironmentalFactor
    performance_metric: PerformanceMetric
    statistical_significance: Dict[str, Any]
    effect_size: float
    factor_performance: Dict[str, Dict[str, float]]  # factor_value -> metrics
    correlation_coefficient: Optional[float] = None
    feature_importance: Optional[float] = None


@dataclass
class FactorCorrelationAnalysis:
    """Correlation analysis between environmental factors and performance."""
    correlation_matrix: pd.DataFrame
    p_value_matrix: pd.DataFrame
    significant_correlations: List[Tuple[str, str, float, float]]  # (factor1, factor2, correlation, p_value)


@dataclass
class OptimalConditionsAnalysis:
    """Analysis of optimal environmental conditions for GPR performance."""
    best_conditions: Dict[EnvironmentalFactor, Any]
    worst_conditions: Dict[EnvironmentalFactor, Any]
    performance_ranking: List[Tuple[str, float]]  # (condition_combination, performance_score)
    condition_recommendations: List[str]


class EnvironmentalValidator:
    """Validator for environmental factor impacts on GPR detection performance."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the environmental validator.

        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.statistical_validator = StatisticalValidator(random_state=random_state)
        self.accuracy_assessor = AccuracyAssessor()

    def validate_environmental_impact(
        self,
        survey_results: List[SurveyResults],
        target_factor: EnvironmentalFactor,
        target_metric: PerformanceMetric,
        confidence_level: float = 0.95
    ) -> EnvironmentalImpactAnalysis:
        """
        Validate the impact of an environmental factor on detection performance.

        Args:
            survey_results: List of survey results with environmental conditions
            target_factor: Environmental factor to analyze
            target_metric: Performance metric to evaluate
            confidence_level: Statistical confidence level

        Returns:
            EnvironmentalImpactAnalysis with impact assessment
        """
        logger.info(f"Validating impact of {target_factor.value} on {target_metric.value}")

        # Calculate performance metrics for each survey
        performance_data = self._calculate_performance_metrics(survey_results)

        # Group data by environmental factor
        factor_groups = self._group_by_environmental_factor(performance_data, target_factor)

        if len(factor_groups) < 2:
            raise ValueError(f"Need at least 2 groups for factor {target_factor.value}, found {len(factor_groups)}")

        # Perform statistical significance testing
        statistical_result = self._test_factor_significance(
            factor_groups, target_metric, confidence_level
        )

        # Calculate effect size
        effect_size = self._calculate_environmental_effect_size(factor_groups, target_metric)

        # Calculate performance by factor value
        factor_performance = self._calculate_factor_performance(factor_groups, target_metric)

        # Calculate correlation if factor is numeric
        correlation_coefficient = self._calculate_factor_correlation(
            performance_data, target_factor, target_metric
        )

        # Calculate feature importance using random forest
        feature_importance = self._calculate_feature_importance(
            performance_data, target_factor, target_metric
        )

        return EnvironmentalImpactAnalysis(
            factor=target_factor,
            performance_metric=target_metric,
            statistical_significance=statistical_result,
            effect_size=effect_size,
            factor_performance=factor_performance,
            correlation_coefficient=correlation_coefficient,
            feature_importance=feature_importance
        )

    def analyze_all_environmental_factors(
        self,
        survey_results: List[SurveyResults],
        target_metric: PerformanceMetric = PerformanceMetric.DETECTION_RATE,
        confidence_level: float = 0.95
    ) -> Dict[EnvironmentalFactor, EnvironmentalImpactAnalysis]:
        """
        Analyze the impact of all environmental factors on a performance metric.

        Args:
            survey_results: List of survey results
            target_metric: Performance metric to evaluate
            confidence_level: Statistical confidence level

        Returns:
            Dictionary mapping factors to their impact analyses
        """
        logger.info(f"Analyzing all environmental factors impact on {target_metric.value}")

        results = {}

        for factor in EnvironmentalFactor:
            try:
                analysis = self.validate_environmental_impact(
                    survey_results, factor, target_metric, confidence_level
                )
                results[factor] = analysis
            except Exception as e:
                logger.warning(f"Failed to analyze factor {factor.value}: {e}")

        return results

    def correlation_analysis_all_factors(
        self,
        survey_results: List[SurveyResults]
    ) -> FactorCorrelationAnalysis:
        """
        Perform correlation analysis between all environmental factors and performance metrics.

        Args:
            survey_results: List of survey results

        Returns:
            FactorCorrelationAnalysis with correlation results
        """
        logger.info("Performing correlation analysis for all environmental factors")

        # Prepare data for correlation analysis
        correlation_data = self._prepare_correlation_data(survey_results)

        # Calculate correlation matrix
        correlation_matrix, p_value_matrix = self.statistical_validator.correlation_analysis(
            correlation_data, method='spearman'
        )

        # Find significant correlations
        significant_correlations = []
        for i, factor1 in enumerate(correlation_matrix.index):
            for j, factor2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    correlation = correlation_matrix.iloc[i, j]
                    p_value = p_value_matrix.iloc[i, j]
                    if p_value < 0.05 and abs(correlation) > 0.3:  # Significant and meaningful
                        significant_correlations.append((factor1, factor2, correlation, p_value))

        return FactorCorrelationAnalysis(
            correlation_matrix=correlation_matrix,
            p_value_matrix=p_value_matrix,
            significant_correlations=significant_correlations
        )

    def identify_optimal_conditions(
        self,
        survey_results: List[SurveyResults],
        target_metric: PerformanceMetric = PerformanceMetric.DETECTION_RATE
    ) -> OptimalConditionsAnalysis:
        """
        Identify optimal environmental conditions for GPR performance.

        Args:
            survey_results: List of survey results
            target_metric: Performance metric to optimize

        Returns:
            OptimalConditionsAnalysis with recommendations
        """
        logger.info(f"Identifying optimal conditions for {target_metric.value}")

        # Calculate performance metrics
        performance_data = self._calculate_performance_metrics(survey_results)

        # Find best and worst performing conditions
        best_conditions, worst_conditions = self._find_extreme_conditions(
            performance_data, target_metric
        )

        # Rank condition combinations
        performance_ranking = self._rank_condition_combinations(
            performance_data, target_metric
        )

        # Generate recommendations
        recommendations = self._generate_condition_recommendations(
            best_conditions, worst_conditions, performance_ranking
        )

        return OptimalConditionsAnalysis(
            best_conditions=best_conditions,
            worst_conditions=worst_conditions,
            performance_ranking=performance_ranking,
            condition_recommendations=recommendations
        )

    def environmental_factor_interaction_analysis(
        self,
        survey_results: List[SurveyResults],
        factors: List[EnvironmentalFactor],
        target_metric: PerformanceMetric
    ) -> Dict[str, Any]:
        """
        Analyze interactions between multiple environmental factors.

        Args:
            survey_results: List of survey results
            factors: List of environmental factors to analyze
            target_metric: Performance metric to evaluate

        Returns:
            Dictionary with interaction analysis results
        """
        logger.info(f"Analyzing interactions between {len(factors)} environmental factors")

        # Prepare data
        performance_data = self._calculate_performance_metrics(survey_results)

        # Create interaction terms
        interaction_data = self._create_interaction_terms(performance_data, factors)

        # Fit random forest model to capture interactions
        rf_model = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, max_depth=10
        )

        # Prepare features and target
        feature_columns = [f.value for f in factors] + [
            f"{f1.value}_x_{f2.value}" for i, f1 in enumerate(factors)
            for f2 in factors[i+1:]
        ]

        X = interaction_data[feature_columns]
        y = interaction_data[target_metric.value]

        # Encode categorical variables
        X_encoded = self._encode_categorical_features(X)

        # Fit model
        rf_model.fit(X_encoded, y)

        # Get feature importances
        feature_importances = dict(zip(feature_columns, rf_model.feature_importances_))

        # Identify most important interactions
        interaction_importances = {
            k: v for k, v in feature_importances.items() if '_x_' in k
        }

        return {
            'main_effects': {k: v for k, v in feature_importances.items() if '_x_' not in k},
            'interaction_effects': interaction_importances,
            'model_score': rf_model.score(X_encoded, y),
            'most_important_interaction': max(interaction_importances.items(), key=lambda x: x[1]) if interaction_importances else None
        }

    def _calculate_performance_metrics(self, survey_results: List[SurveyResults]) -> pd.DataFrame:
        """Calculate performance metrics for all survey results."""
        data = []

        for survey in survey_results:
            # Calculate accuracy metrics
            comprehensive_results = self.accuracy_assessor.comprehensive_accuracy_assessment(
                survey.detections, survey.ground_truth
            )

            # Extract environmental conditions
            env_conditions = survey.environmental_conditions

            # Compile data row
            row = {
                'location_id': survey.location_id,
                'weather_condition': env_conditions.weather_condition,
                'ground_condition': env_conditions.ground_condition,
                'ground_permittivity': env_conditions.ground_permittivity,
                'land_cover': env_conditions.land_cover,
                'land_use': env_conditions.land_use,
                'terrain_levelling': env_conditions.terrain_levelling,
                'terrain_smoothness': env_conditions.terrain_smoothness,
                'rubble_presence': env_conditions.rubble_presence,
                'tree_roots_presence': env_conditions.tree_roots_presence,
                'polluted_soil_presence': env_conditions.polluted_soil_presence,
                'blast_furnace_slag_presence': env_conditions.blast_furnace_slag_presence,
                'detection_rate': comprehensive_results['detection_performance'].recall,
                'position_accuracy': comprehensive_results['position_accuracy'].horizontal_rmse,
                'depth_accuracy': comprehensive_results['depth_estimation'].rmse,
                'material_accuracy': comprehensive_results['material_classification'].overall_accuracy,
                'false_positive_rate': comprehensive_results['detection_performance'].false_positive_rate,
                'overall_f1_score': comprehensive_results['detection_performance'].f1_score
            }

            data.append(row)

        return pd.DataFrame(data)

    def _group_by_environmental_factor(
        self,
        performance_data: pd.DataFrame,
        factor: EnvironmentalFactor
    ) -> Dict[str, pd.DataFrame]:
        """Group performance data by environmental factor values."""
        factor_column = factor.value
        if factor_column not in performance_data.columns:
            raise ValueError(f"Factor {factor_column} not found in performance data")

        groups = {}
        for factor_value in performance_data[factor_column].unique():
            if pd.notna(factor_value):  # Skip NaN values
                group_data = performance_data[performance_data[factor_column] == factor_value]
                groups[str(factor_value)] = group_data

        return groups

    def _test_factor_significance(
        self,
        factor_groups: Dict[str, pd.DataFrame],
        target_metric: PerformanceMetric,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Test statistical significance of environmental factor impact."""
        metric_column = target_metric.value

        # Extract performance values for each group
        group_values = []
        group_names = []

        for group_name, group_data in factor_groups.items():
            if metric_column in group_data.columns:
                values = group_data[metric_column].dropna().values
                if len(values) > 0:
                    group_values.append(values)
                    group_names.append(group_name)

        if len(group_values) < 2:
            return {'error': 'Insufficient groups for statistical testing'}

        # Choose appropriate test
        if len(group_values) == 2:
            # Two-group comparison
            test_result = self.statistical_validator.compare_groups_statistical_test(
                group_values[0], group_values[1],
                test_type=StatisticalTest.MANN_WHITNEY,
                confidence_level=confidence_level
            )
        else:
            # Multi-group comparison
            test_result = self.statistical_validator.compare_multiple_groups(
                group_values, group_names,
                test_type=StatisticalTest.KRUSKAL_WALLIS,
                confidence_level=confidence_level
            )

        return {
            'test_name': test_result.test_name,
            'statistic': test_result.statistic,
            'p_value': test_result.p_value,
            'significant': test_result.significant,
            'effect_size': test_result.effect_size,
            'interpretation': test_result.interpretation
        }

    def _calculate_environmental_effect_size(
        self,
        factor_groups: Dict[str, pd.DataFrame],
        target_metric: PerformanceMetric
    ) -> float:
        """Calculate effect size for environmental factor impact."""
        metric_column = target_metric.value

        # Calculate overall variance
        all_values = []
        for group_data in factor_groups.values():
            if metric_column in group_data.columns:
                values = group_data[metric_column].dropna().values
                all_values.extend(values)

        if len(all_values) < 2:
            return 0.0

        total_variance = np.var(all_values, ddof=1)

        # Calculate between-group variance
        group_means = []
        group_sizes = []

        for group_data in factor_groups.values():
            if metric_column in group_data.columns:
                values = group_data[metric_column].dropna().values
                if len(values) > 0:
                    group_means.append(np.mean(values))
                    group_sizes.append(len(values))

        if len(group_means) < 2:
            return 0.0

        overall_mean = np.average(group_means, weights=group_sizes)
        between_group_variance = np.average(
            [(mean - overall_mean)**2 for mean in group_means],
            weights=group_sizes
        )

        # Calculate eta-squared (effect size)
        eta_squared = between_group_variance / total_variance if total_variance > 0 else 0.0
        return eta_squared

    def _calculate_factor_performance(
        self,
        factor_groups: Dict[str, pd.DataFrame],
        target_metric: PerformanceMetric
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance statistics for each factor value."""
        metric_column = target_metric.value
        factor_performance = {}

        for factor_value, group_data in factor_groups.items():
            if metric_column in group_data.columns:
                values = group_data[metric_column].dropna().values
                if len(values) > 0:
                    factor_performance[factor_value] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

        return factor_performance

    def _calculate_factor_correlation(
        self,
        performance_data: pd.DataFrame,
        factor: EnvironmentalFactor,
        target_metric: PerformanceMetric
    ) -> Optional[float]:
        """Calculate correlation between numeric factor and performance metric."""
        factor_column = factor.value
        metric_column = target_metric.value

        if factor_column not in performance_data.columns or metric_column not in performance_data.columns:
            return None

        # Only calculate correlation for numeric factors
        if factor == EnvironmentalFactor.GROUND_PERMITTIVITY:
            factor_values = performance_data[factor_column].dropna()
            metric_values = performance_data[metric_column].dropna()

            if len(factor_values) > 2 and len(metric_values) > 2:
                correlation, p_value = stats.spearmanr(factor_values, metric_values)
                return correlation if p_value < 0.05 else None

        return None

    def _calculate_feature_importance(
        self,
        performance_data: pd.DataFrame,
        target_factor: EnvironmentalFactor,
        target_metric: PerformanceMetric
    ) -> Optional[float]:
        """Calculate feature importance using random forest."""
        try:
            # Prepare features (all environmental factors)
            feature_columns = [f.value for f in EnvironmentalFactor]
            target_column = target_metric.value

            # Filter available columns
            available_features = [col for col in feature_columns if col in performance_data.columns]

            if len(available_features) < 2 or target_column not in performance_data.columns:
                return None

            X = performance_data[available_features].copy()
            y = performance_data[target_column].dropna()

            # Encode categorical variables
            X_encoded = self._encode_categorical_features(X)

            # Fit random forest
            rf = RandomForestRegressor(n_estimators=50, random_state=self.random_state, max_depth=5)
            rf.fit(X_encoded, y)

            # Get importance for target factor
            target_index = available_features.index(target_factor.value)
            return rf.feature_importances_[target_index]

        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
            return None

    def _prepare_correlation_data(self, survey_results: List[SurveyResults]) -> pd.DataFrame:
        """Prepare data for correlation analysis."""
        performance_data = self._calculate_performance_metrics(survey_results)

        # Select numeric columns for correlation
        numeric_columns = []
        for col in performance_data.columns:
            if performance_data[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)

        return performance_data[numeric_columns]

    def _find_extreme_conditions(
        self,
        performance_data: pd.DataFrame,
        target_metric: PerformanceMetric
    ) -> Tuple[Dict[EnvironmentalFactor, Any], Dict[EnvironmentalFactor, Any]]:
        """Find best and worst performing environmental conditions."""
        metric_column = target_metric.value

        # Find best and worst performing locations
        best_location = performance_data.loc[performance_data[metric_column].idxmax()]
        worst_location = performance_data.loc[performance_data[metric_column].idxmin()]

        best_conditions = {}
        worst_conditions = {}

        for factor in EnvironmentalFactor:
            factor_column = factor.value
            if factor_column in performance_data.columns:
                best_conditions[factor] = best_location[factor_column]
                worst_conditions[factor] = worst_location[factor_column]

        return best_conditions, worst_conditions

    def _rank_condition_combinations(
        self,
        performance_data: pd.DataFrame,
        target_metric: PerformanceMetric
    ) -> List[Tuple[str, float]]:
        """Rank different condition combinations by performance."""
        metric_column = target_metric.value

        # Create condition combination strings
        condition_cols = [f.value for f in EnvironmentalFactor if f.value in performance_data.columns]
        performance_data['condition_combination'] = performance_data[condition_cols].apply(
            lambda row: '_'.join([f"{col}:{row[col]}" for col in condition_cols]), axis=1
        )

        # Group by condition combination and calculate mean performance
        combination_performance = performance_data.groupby('condition_combination')[metric_column].mean()

        # Sort by performance
        ranking = [(combo, perf) for combo, perf in combination_performance.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking[:10]  # Return top 10

    def _generate_condition_recommendations(
        self,
        best_conditions: Dict[EnvironmentalFactor, Any],
        worst_conditions: Dict[EnvironmentalFactor, Any],
        performance_ranking: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate recommendations for optimal environmental conditions."""
        recommendations = []

        # Analyze best vs worst conditions
        for factor, best_value in best_conditions.items():
            worst_value = worst_conditions[factor]
            if best_value != worst_value:
                recommendations.append(
                    f"Prefer {factor.value.replace('_', ' ')}: {best_value} over {worst_value}"
                )

        # Add general recommendations based on top performers
        if performance_ranking:
            best_combo = performance_ranking[0][0]
            recommendations.append(f"Best overall combination: {best_combo}")

        return recommendations

    def _create_interaction_terms(
        self,
        performance_data: pd.DataFrame,
        factors: List[EnvironmentalFactor]
    ) -> pd.DataFrame:
        """Create interaction terms between environmental factors."""
        interaction_data = performance_data.copy()

        # Create pairwise interactions
        for i, factor1 in enumerate(factors):
            for factor2 in factors[i+1:]:
                col1 = factor1.value
                col2 = factor2.value

                if col1 in interaction_data.columns and col2 in interaction_data.columns:
                    # For numeric columns, multiply values
                    if (interaction_data[col1].dtype in ['int64', 'float64'] and
                        interaction_data[col2].dtype in ['int64', 'float64']):
                        interaction_data[f"{col1}_x_{col2}"] = interaction_data[col1] * interaction_data[col2]
                    else:
                        # For categorical, create combined categories
                        interaction_data[f"{col1}_x_{col2}"] = (
                            interaction_data[col1].astype(str) + "_x_" + interaction_data[col2].astype(str)
                        )

        return interaction_data

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for machine learning models."""
        X_encoded = X.copy()

        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype == 'bool':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        return X_encoded


def create_environmental_validator(random_state: int = 42) -> EnvironmentalValidator:
    """
    Factory function to create an environmental validator.

    Args:
        random_state: Random state for reproducible results

    Returns:
        Configured EnvironmentalValidator instance
    """
    return EnvironmentalValidator(random_state=random_state)