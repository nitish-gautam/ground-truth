"""
Environmental correlation analysis service
=========================================

Comprehensive analysis of environmental factors and their impact on GPR
detection accuracy, signal quality, and utility visibility.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRSurvey, GPRScan, GPRProcessingResult
from ..models.environmental import EnvironmentalData, WeatherCondition, GroundCondition
from ..models.validation import ValidationResult, AccuracyMetrics


class EnvironmentalCorrelationAnalyzer(LoggerMixin):
    """Service for analyzing environmental correlations with GPR performance."""

    def __init__(self):
        super().__init__()
        self.correlation_methods = {
            "pearson": pearsonr,
            "spearman": spearmanr,
        }

    async def perform_comprehensive_analysis(
        self,
        db: AsyncSession,
        analysis_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive environmental correlation analysis."""
        self.log_operation_start("comprehensive_environmental_analysis")

        try:
            # Load environmental and performance data
            env_data = await self._load_environmental_data(db)
            performance_data = await self._load_performance_data(db)

            if not env_data or not performance_data:
                raise ValueError("Insufficient data for analysis")

            # Merge datasets
            merged_data = await self._merge_datasets(env_data, performance_data)

            # Perform various correlation analyses
            results = {
                "weather_impact_analysis": await self._analyze_weather_impact(merged_data),
                "ground_condition_analysis": await self._analyze_ground_conditions(merged_data),
                "utility_density_analysis": await self._analyze_utility_density(merged_data),
                "soil_composition_analysis": await self._analyze_soil_composition(merged_data),
                "terrain_characteristics_analysis": await self._analyze_terrain_characteristics(merged_data),
                "seasonal_variations": await self._analyze_seasonal_variations(merged_data),
                "multivariate_analysis": await self._perform_multivariate_analysis(merged_data),
                "predictive_model": await self._build_predictive_model(merged_data)
            }

            # Generate summary statistics
            results["summary"] = await self._generate_analysis_summary(results)

            # Store results in database
            await self._store_analysis_results(db, results)

            self.log_operation_complete("comprehensive_environmental_analysis", 0)

            return results

        except Exception as e:
            self.log_operation_error("comprehensive_environmental_analysis", e)
            raise

    async def _load_environmental_data(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Load environmental data from database."""
        try:
            # Load environmental data with related survey information
            result = await db.execute(
                select(EnvironmentalData, GPRSurvey)
                .join(GPRSurvey, EnvironmentalData.survey_id == GPRSurvey.id)
                .order_by(EnvironmentalData.created_at)
            )

            env_records = []
            for env_data, survey in result.all():
                record = {
                    "survey_id": str(survey.id),
                    "location_id": survey.location_id,
                    "land_use": env_data.land_use,
                    "land_cover": env_data.land_cover,
                    "ground_condition": env_data.ground_condition,
                    "ground_permittivity": env_data.ground_relative_permittivity,
                    "groundwater_level": env_data.relative_groundwater_level,
                    "terrain_levelling": env_data.terrain_levelling,
                    "terrain_smoothness": env_data.terrain_smoothness,
                    "weather_condition": env_data.weather_condition,
                    "amount_of_utilities": env_data.amount_of_utilities,
                    "utility_crossing": env_data.utility_crossing,
                    "utility_path_linear": env_data.utility_path_linear,
                    "rubble_presence": env_data.rubble_presence,
                    "tree_roots_presence": env_data.tree_roots_presence,
                    "polluted_soil_presence": env_data.polluted_soil_presence,
                    "blast_furnace_slag_presence": env_data.blast_furnace_slag_presence
                }
                env_records.append(record)

            return env_records

        except Exception as e:
            self.log_operation_error("load_environmental_data", e)
            raise

    async def _load_performance_data(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Load GPR performance data from database."""
        try:
            # Load processing results with scan and survey information
            result = await db.execute(
                select(GPRProcessingResult, GPRScan, GPRSurvey)
                .join(GPRScan, GPRProcessingResult.scan_id == GPRScan.id)
                .join(GPRSurvey, GPRScan.survey_id == GPRSurvey.id)
                .where(GPRProcessingResult.status == "completed")
                .order_by(GPRProcessingResult.processing_timestamp)
            )

            performance_records = []
            for processing_result, scan, survey in result.all():
                record = {
                    "survey_id": str(survey.id),
                    "scan_id": str(scan.id),
                    "location_id": survey.location_id,
                    "overall_confidence": processing_result.overall_confidence,
                    "detection_count": processing_result.detection_count,
                    "signal_quality_score": scan.signal_quality_score,
                    "noise_level": scan.noise_level,
                    "data_completeness": scan.data_completeness,
                    "processing_algorithm": processing_result.processing_algorithm,
                    "environmental_impact_score": processing_result.environmental_impact_score,
                    "weather_correlation": processing_result.weather_correlation,
                    "ground_condition_impact": processing_result.ground_condition_impact
                }
                performance_records.append(record)

            return performance_records

        except Exception as e:
            self.log_operation_error("load_performance_data", e)
            raise

    async def _merge_datasets(
        self,
        env_data: List[Dict[str, Any]],
        performance_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Merge environmental and performance datasets."""
        try:
            # Convert to DataFrames
            env_df = pd.DataFrame(env_data)
            perf_df = pd.DataFrame(performance_data)

            # Merge on survey_id
            merged_df = pd.merge(env_df, perf_df, on="survey_id", how="inner")

            # Clean and prepare data
            merged_df = await self._clean_merged_data(merged_df)

            return merged_df

        except Exception as e:
            self.log_operation_error("merge_datasets", e)
            raise

    async def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare merged dataset for analysis."""
        try:
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna('unknown')

            # Convert boolean columns
            boolean_columns = [
                'utility_crossing', 'utility_path_linear', 'rubble_presence',
                'tree_roots_presence', 'polluted_soil_presence', 'blast_furnace_slag_presence'
            ]
            for col in boolean_columns:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            # Create derived features
            df['ground_complexity_score'] = (
                df.get('rubble_presence', 0) +
                df.get('tree_roots_presence', 0) +
                df.get('polluted_soil_presence', 0) +
                df.get('blast_furnace_slag_presence', 0)
            )

            # Normalize utility density
            if 'amount_of_utilities' in df.columns:
                df['utility_density_normalized'] = df['amount_of_utilities'] / df['amount_of_utilities'].max()

            return df

        except Exception as e:
            self.log_operation_error("clean_merged_data", e)
            raise

    async def _analyze_weather_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of weather conditions on GPR performance."""
        try:
            weather_analysis = {}

            if 'weather_condition' in df.columns and 'overall_confidence' in df.columns:
                # Group by weather condition
                weather_groups = df.groupby('weather_condition')['overall_confidence']

                weather_stats = {}
                for weather, group in weather_groups:
                    weather_stats[weather] = {
                        "mean_confidence": float(group.mean()),
                        "std_confidence": float(group.std()),
                        "count": len(group),
                        "median_confidence": float(group.median())
                    }

                weather_analysis["confidence_by_weather"] = weather_stats

                # Statistical significance test
                weather_values = [group.values for _, group in weather_groups]
                if len(weather_values) > 1:
                    from scipy.stats import f_oneway
                    f_stat, p_value = f_oneway(*weather_values)
                    weather_analysis["anova_test"] = {
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }

            # Correlation with signal quality
            if 'signal_quality_score' in df.columns:
                weather_encoded = LabelEncoder().fit_transform(df['weather_condition'].fillna('unknown'))
                correlation, p_value = pearsonr(weather_encoded, df['signal_quality_score'].fillna(0))

                weather_analysis["signal_quality_correlation"] = {
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }

            self.log_environmental_correlation("weather_impact", len(weather_analysis))

            return weather_analysis

        except Exception as e:
            self.log_operation_error("analyze_weather_impact", e)
            return {"error": str(e)}

    async def _analyze_ground_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of ground conditions on GPR performance."""
        try:
            ground_analysis = {}

            # Ground condition vs. detection performance
            if 'ground_condition' in df.columns and 'detection_count' in df.columns:
                ground_groups = df.groupby('ground_condition')['detection_count']

                ground_stats = {}
                for condition, group in ground_groups:
                    ground_stats[condition] = {
                        "mean_detections": float(group.mean()),
                        "std_detections": float(group.std()),
                        "count": len(group),
                        "success_rate": float((group > 0).mean())
                    }

                ground_analysis["detections_by_ground_condition"] = ground_stats

            # Ground permittivity correlation
            if 'ground_permittivity' in df.columns and 'overall_confidence' in df.columns:
                valid_data = df.dropna(subset=['ground_permittivity', 'overall_confidence'])
                if len(valid_data) > 5:
                    correlation, p_value = pearsonr(
                        valid_data['ground_permittivity'],
                        valid_data['overall_confidence']
                    )

                    ground_analysis["permittivity_correlation"] = {
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }

            # Groundwater level impact
            if 'groundwater_level' in df.columns:
                groundwater_groups = df.groupby('groundwater_level')['signal_quality_score']

                groundwater_stats = {}
                for level, group in groundwater_groups:
                    if len(group) > 0:
                        groundwater_stats[level] = {
                            "mean_signal_quality": float(group.mean()),
                            "count": len(group)
                        }

                ground_analysis["groundwater_impact"] = groundwater_stats

            self.log_environmental_correlation("ground_conditions", len(ground_analysis))

            return ground_analysis

        except Exception as e:
            self.log_operation_error("analyze_ground_conditions", e)
            return {"error": str(e)}

    async def _analyze_utility_density(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationship between utility density and detection accuracy."""
        try:
            utility_analysis = {}

            if 'amount_of_utilities' in df.columns and 'overall_confidence' in df.columns:
                valid_data = df.dropna(subset=['amount_of_utilities', 'overall_confidence'])

                if len(valid_data) > 5:
                    # Correlation analysis
                    correlation, p_value = pearsonr(
                        valid_data['amount_of_utilities'],
                        valid_data['overall_confidence']
                    )

                    utility_analysis["density_confidence_correlation"] = {
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }

                    # Binned analysis
                    utility_bins = pd.cut(valid_data['amount_of_utilities'], bins=5, labels=False)
                    binned_stats = valid_data.groupby(utility_bins)['overall_confidence'].agg([
                        'mean', 'std', 'count'
                    ])

                    utility_analysis["binned_analysis"] = binned_stats.to_dict('index')

            # Utility crossing impact
            if 'utility_crossing' in df.columns:
                crossing_impact = df.groupby('utility_crossing')['detection_count'].agg([
                    'mean', 'std', 'count'
                ])

                utility_analysis["crossing_impact"] = crossing_impact.to_dict('index')

            # Path linearity impact
            if 'utility_path_linear' in df.columns:
                linearity_impact = df.groupby('utility_path_linear')['overall_confidence'].agg([
                    'mean', 'std', 'count'
                ])

                utility_analysis["linearity_impact"] = linearity_impact.to_dict('index')

            self.log_environmental_correlation("utility_density", len(utility_analysis))

            return utility_analysis

        except Exception as e:
            self.log_operation_error("analyze_utility_density", e)
            return {"error": str(e)}

    async def _analyze_soil_composition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze soil composition effects on GPR performance."""
        try:
            soil_analysis = {}

            # Ground complexity score impact
            if 'ground_complexity_score' in df.columns:
                complexity_groups = df.groupby('ground_complexity_score')['signal_quality_score']

                complexity_stats = {}
                for score, group in complexity_groups:
                    complexity_stats[str(score)] = {
                        "mean_signal_quality": float(group.mean()),
                        "count": len(group),
                        "std": float(group.std())
                    }

                soil_analysis["complexity_impact"] = complexity_stats

            # Individual contamination factors
            contamination_factors = [
                'rubble_presence', 'tree_roots_presence',
                'polluted_soil_presence', 'blast_furnace_slag_presence'
            ]

            factor_impacts = {}
            for factor in contamination_factors:
                if factor in df.columns and 'overall_confidence' in df.columns:
                    factor_groups = df.groupby(factor)['overall_confidence']
                    factor_impacts[factor] = factor_groups.agg(['mean', 'count']).to_dict()

            soil_analysis["contamination_impacts"] = factor_impacts

            self.log_environmental_correlation("soil_composition", len(soil_analysis))

            return soil_analysis

        except Exception as e:
            self.log_operation_error("analyze_soil_composition", e)
            return {"error": str(e)}

    async def _analyze_terrain_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze terrain characteristics impact on GPR performance."""
        try:
            terrain_analysis = {}

            # Terrain levelling impact
            if 'terrain_levelling' in df.columns:
                levelling_groups = df.groupby('terrain_levelling')['signal_quality_score']
                terrain_analysis["levelling_impact"] = levelling_groups.agg([
                    'mean', 'std', 'count'
                ]).to_dict('index')

            # Terrain smoothness impact
            if 'terrain_smoothness' in df.columns:
                smoothness_groups = df.groupby('terrain_smoothness')['overall_confidence']
                terrain_analysis["smoothness_impact"] = smoothness_groups.agg([
                    'mean', 'std', 'count'
                ]).to_dict('index')

            # Combined terrain quality score
            if all(col in df.columns for col in ['terrain_levelling', 'terrain_smoothness']):
                # Create combined terrain quality
                terrain_quality_map = {
                    ('Flat', 'Smooth'): 5,
                    ('Flat', 'Rough'): 4,
                    ('Uneven', 'Smooth'): 3,
                    ('Uneven', 'Rough'): 2,
                    ('Steep', 'Smooth'): 2,
                    ('Steep', 'Rough'): 1
                }

                df['terrain_quality'] = df.apply(
                    lambda row: terrain_quality_map.get(
                        (row['terrain_levelling'], row['terrain_smoothness']), 3
                    ), axis=1
                )

                # Correlation with performance
                if 'overall_confidence' in df.columns:
                    correlation, p_value = pearsonr(df['terrain_quality'], df['overall_confidence'])
                    terrain_analysis["quality_correlation"] = {
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }

            self.log_environmental_correlation("terrain_characteristics", len(terrain_analysis))

            return terrain_analysis

        except Exception as e:
            self.log_operation_error("analyze_terrain_characteristics", e)
            return {"error": str(e)}

    async def _analyze_seasonal_variations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal variations in GPR performance (placeholder)."""
        # This would require timestamp data to analyze seasonal patterns
        return {"status": "seasonal_analysis_requires_temporal_data"}

    async def _perform_multivariate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform multivariate analysis to identify combined factor effects."""
        try:
            multivariate_results = {}

            # Prepare features for analysis
            feature_columns = [
                'ground_permittivity', 'amount_of_utilities', 'ground_complexity_score',
                'utility_crossing', 'utility_path_linear'
            ]

            available_features = [col for col in feature_columns if col in df.columns]

            if len(available_features) >= 3 and 'overall_confidence' in df.columns:
                # Prepare data
                feature_data = df[available_features].fillna(0)
                target_data = df['overall_confidence'].fillna(0)

                # Correlation matrix
                correlation_matrix = feature_data.corrwith(target_data)
                multivariate_results["feature_correlations"] = correlation_matrix.to_dict()

                # Principal Component Analysis
                if len(feature_data) > 10:
                    from sklearn.decomposition import PCA

                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_data)

                    pca = PCA(n_components=min(3, len(available_features)))
                    pca_components = pca.fit_transform(scaled_features)

                    multivariate_results["pca_analysis"] = {
                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                        "components": pca.components_.tolist(),
                        "feature_names": available_features
                    }

            return multivariate_results

        except Exception as e:
            self.log_operation_error("perform_multivariate_analysis", e)
            return {"error": str(e)}

    async def _build_predictive_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build predictive model for GPR performance based on environmental factors."""
        try:
            model_results = {}

            # Prepare features and target
            feature_columns = [
                'ground_permittivity', 'amount_of_utilities', 'ground_complexity_score',
                'utility_crossing', 'utility_path_linear'
            ]

            available_features = [col for col in feature_columns if col in df.columns]

            if len(available_features) >= 3 and 'overall_confidence' in df.columns:
                X = df[available_features].fillna(0)
                y = df['overall_confidence'].fillna(0)

                if len(X) > 20:  # Minimum samples for meaningful model
                    # Random Forest model
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

                    # Cross-validation
                    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

                    # Fit model for feature importance
                    rf_model.fit(X, y)

                    model_results["predictive_model"] = {
                        "model_type": "RandomForestRegressor",
                        "cv_scores": cv_scores.tolist(),
                        "mean_cv_score": float(cv_scores.mean()),
                        "std_cv_score": float(cv_scores.std()),
                        "feature_importance": dict(zip(available_features, rf_model.feature_importances_)),
                        "feature_names": available_features
                    }

            return model_results

        except Exception as e:
            self.log_operation_error("build_predictive_model", e)
            return {"error": str(e)}

    async def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all analyses."""
        summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "key_findings": [],
            "recommendations": [],
            "data_quality_notes": []
        }

        # Extract key findings from each analysis
        if "weather_impact_analysis" in results:
            weather_results = results["weather_impact_analysis"]
            if "anova_test" in weather_results and weather_results["anova_test"].get("significant"):
                summary["key_findings"].append("Weather conditions have statistically significant impact on GPR confidence")

        if "ground_condition_analysis" in results:
            ground_results = results["ground_condition_analysis"]
            if "permittivity_correlation" in ground_results:
                corr = ground_results["permittivity_correlation"].get("correlation", 0)
                if abs(corr) > 0.3:
                    summary["key_findings"].append(f"Ground permittivity shows moderate correlation ({corr:.2f}) with detection confidence")

        # Generate recommendations
        summary["recommendations"].extend([
            "Consider weather conditions when planning GPR surveys",
            "Account for ground permittivity in detection confidence calculations",
            "Higher utility density areas may require specialized processing techniques",
            "Terrain quality should be factored into survey planning"
        ])

        return summary

    async def _store_analysis_results(self, db: AsyncSession, results: Dict[str, Any]) -> None:
        """Store analysis results in database (placeholder)."""
        # This would store results in a dedicated analysis results table
        pass