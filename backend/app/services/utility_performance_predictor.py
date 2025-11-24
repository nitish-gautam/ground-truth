"""
Utility Detection Performance Prediction System
==============================================

Advanced predictive modeling system for GPR utility detection performance
using comprehensive environmental, material, and survey context features
from the University of Twente dataset.

Builds multiple prediction models for:
- Detection accuracy prediction
- Signal quality forecasting
- Utility visibility assessment
- Survey success probability
- Optimal survey condition recommendations
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import (
    cross_val_score, train_test_split, GridSearchCV,
    RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, explained_variance_score
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    SelectFromModel, RFE
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRSurvey, GPRScan, GPRProcessingResult
from ..models.environmental import EnvironmentalData
from ..models.utilities import Utility, UtilityDetection
from ..models.ml_analytics import FeatureVector, MLModel, ModelPerformance
from .twente_feature_extractor import TwenteFeatureExtractor
from .material_classification_system import MaterialClassificationSystem

warnings.filterwarnings('ignore', category=FutureWarning)


class UtilityPerformancePredictor(LoggerMixin):
    """Advanced utility detection performance prediction system."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = TwenteFeatureExtractor()
        self.material_classifier = MaterialClassificationSystem()

        # Prediction targets
        self.prediction_targets = {
            'detection_accuracy': 'regression',
            'signal_quality': 'regression',
            'utility_visibility': 'classification',
            'survey_success': 'classification',
            'detection_difficulty': 'regression'
        }

        # Model configurations
        self.regression_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(random_state=42, max_iter=500)
        }

        self.classification_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svc': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(random_state=42, max_iter=500)
        }

        # Scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }

        # Trained models storage
        self.trained_models = {}
        self.feature_selectors = {}
        self.performance_metrics = {}

    async def build_comprehensive_prediction_system(
        self,
        metadata_df: pd.DataFrame,
        performance_data: Optional[pd.DataFrame] = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build comprehensive utility detection performance prediction system."""
        self.log_operation_start("comprehensive_prediction_system")

        if config is None:
            config = self._get_default_config()

        try:
            # Extract comprehensive features
            feature_extraction_results = await self.feature_extractor.extract_comprehensive_features(
                metadata_df, performance_data
            )

            # Create synthetic performance targets (since we don't have real performance data)
            synthetic_targets = await self._create_synthetic_performance_targets(
                metadata_df, feature_extraction_results, config
            )

            # Prepare training datasets
            training_datasets = await self._prepare_training_datasets(
                feature_extraction_results, synthetic_targets, config
            )

            # Build prediction models
            prediction_models = await self._build_prediction_models(
                training_datasets, config
            )

            # Model evaluation and validation
            model_evaluation = await self._evaluate_prediction_models(
                prediction_models, training_datasets, config
            )

            # Feature importance analysis
            feature_importance = await self._analyze_feature_importance(
                prediction_models, training_datasets, config
            )

            # Performance factor analysis
            performance_factors = await self._analyze_performance_factors(
                training_datasets, prediction_models, config
            )

            # Prediction insights and recommendations
            prediction_insights = await self._generate_prediction_insights(
                prediction_models, feature_importance, performance_factors
            )

            # Scenario analysis
            scenario_analysis = await self._perform_scenario_analysis(
                prediction_models, training_datasets, config
            )

            # Optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                prediction_models, feature_importance, scenario_analysis
            )

            results = {
                "feature_extraction": feature_extraction_results,
                "synthetic_targets": synthetic_targets,
                "training_datasets": training_datasets,
                "prediction_models": prediction_models,
                "model_evaluation": model_evaluation,
                "feature_importance": feature_importance,
                "performance_factors": performance_factors,
                "prediction_insights": prediction_insights,
                "scenario_analysis": scenario_analysis,
                "optimization_recommendations": optimization_recommendations,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "config": config,
                    "model_count": len(prediction_models),
                    "feature_count": len(feature_extraction_results.get('environmental_features', {}))
                }
            }

            self.log_operation_complete("comprehensive_prediction_system", len(results))
            return results

        except Exception as e:
            self.log_operation_error("comprehensive_prediction_system", e)
            raise

    async def _create_synthetic_performance_targets(
        self,
        metadata_df: pd.DataFrame,
        features: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create synthetic performance targets based on environmental and material factors."""
        try:
            synthetic_targets = {}

            # Create detection accuracy target
            detection_accuracy = []
            for idx, row in metadata_df.iterrows():
                base_accuracy = 0.7  # Base accuracy

                # Weather impact
                if row.get('Weather condition') == 'Dry':
                    base_accuracy += 0.15
                elif row.get('Weather condition') == 'Rainy':
                    base_accuracy -= 0.2

                # Ground condition impact
                if row.get('Ground condition') == 'Sandy':
                    base_accuracy += 0.1
                elif row.get('Ground condition') == 'Clayey':
                    base_accuracy -= 0.15

                # Ground permittivity impact
                permittivity = row.get('Ground relative permittivity', 10)
                if permittivity < 8:
                    base_accuracy += 0.1
                elif permittivity > 15:
                    base_accuracy -= 0.1

                # Utility density impact
                utility_count = row.get('Amount of utilities', 5)
                if utility_count > 15:
                    base_accuracy -= 0.2
                elif utility_count < 3:
                    base_accuracy += 0.1

                # Utility crossing penalty
                if row.get('Utility crossing') is True:
                    base_accuracy -= 0.15

                # Environmental contamination impact
                contamination_penalty = 0
                if row.get('Rubble presence') is True:
                    contamination_penalty += 0.05
                if row.get('Tree roots presence') is True:
                    contamination_penalty += 0.05
                if row.get('Polluted soil presence') is True:
                    contamination_penalty += 0.1

                base_accuracy -= contamination_penalty

                # Add some noise
                noise = np.random.normal(0, 0.05)
                final_accuracy = np.clip(base_accuracy + noise, 0.1, 0.95)

                detection_accuracy.append(final_accuracy)

            synthetic_targets['detection_accuracy'] = detection_accuracy

            # Create signal quality target
            signal_quality = []
            for idx, row in metadata_df.iterrows():
                base_quality = 0.75  # Base signal quality

                # Weather impact (stronger for signal quality)
                if row.get('Weather condition') == 'Dry':
                    base_quality += 0.2
                elif row.get('Weather condition') == 'Rainy':
                    base_quality -= 0.3

                # Ground permittivity impact (major factor for signal quality)
                permittivity = row.get('Ground relative permittivity', 10)
                if permittivity < 8:
                    base_quality += 0.15
                elif permittivity > 15:
                    base_quality -= 0.25

                # Terrain impact
                if row.get('Terrain levelling') == 'Flat' and row.get('Terrain smoothness') == 'Smooth':
                    base_quality += 0.1
                elif row.get('Terrain levelling') == 'Steep' or row.get('Terrain smoothness') == 'Rough':
                    base_quality -= 0.15

                # Add noise
                noise = np.random.normal(0, 0.08)
                final_quality = np.clip(base_quality + noise, 0.1, 0.95)

                signal_quality.append(final_quality)

            synthetic_targets['signal_quality'] = signal_quality

            # Create utility visibility classification target
            utility_visibility = []
            for accuracy in detection_accuracy:
                if accuracy > 0.8:
                    visibility = 'high'
                elif accuracy > 0.6:
                    visibility = 'medium'
                else:
                    visibility = 'low'
                utility_visibility.append(visibility)

            synthetic_targets['utility_visibility'] = utility_visibility

            # Create survey success classification target
            survey_success = []
            for accuracy, quality in zip(detection_accuracy, signal_quality):
                combined_score = (accuracy + quality) / 2
                success = 'success' if combined_score > 0.65 else 'failure'
                survey_success.append(success)

            synthetic_targets['survey_success'] = survey_success

            # Create detection difficulty target
            detection_difficulty = []
            for idx, row in metadata_df.iterrows():
                difficulty_score = 0

                # Environmental factors
                if row.get('Weather condition') == 'Rainy':
                    difficulty_score += 2
                if row.get('Ground condition') == 'Clayey':
                    difficulty_score += 2

                # Utility complexity
                utility_count = row.get('Amount of utilities', 5)
                if utility_count > 15:
                    difficulty_score += 3
                elif utility_count > 10:
                    difficulty_score += 2

                if row.get('Utility crossing') is True:
                    difficulty_score += 2

                # Contamination
                contamination_factors = [
                    'Rubble presence', 'Tree roots presence',
                    'Polluted soil presence', 'Blast-furnace slag presence'
                ]
                for factor in contamination_factors:
                    if row.get(factor) is True:
                        difficulty_score += 1

                # Normalize to 0-1 scale
                normalized_difficulty = min(difficulty_score / 10, 1.0)
                detection_difficulty.append(normalized_difficulty)

            synthetic_targets['detection_difficulty'] = detection_difficulty

            # Add metadata
            synthetic_targets['metadata'] = {
                'creation_method': 'rule_based_synthesis',
                'factors_considered': [
                    'weather_condition', 'ground_condition', 'ground_permittivity',
                    'utility_density', 'utility_crossing', 'environmental_contamination',
                    'terrain_characteristics'
                ],
                'target_count': len(synthetic_targets),
                'sample_count': len(metadata_df)
            }

            return synthetic_targets

        except Exception as e:
            self.log_operation_error("create_synthetic_performance_targets", e)
            raise

    async def _prepare_training_datasets(
        self,
        features: Dict[str, Any],
        targets: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare comprehensive training datasets for prediction models."""
        try:
            training_datasets = {}

            # Combine all feature sets into a comprehensive feature matrix
            feature_df = await self.feature_extractor._combine_feature_sets(
                features, pd.DataFrame()  # Empty df since features are already extracted
            )

            # Select numeric features for modeling
            numeric_features = feature_df.select_dtypes(include=[np.number])

            # Remove columns with too many missing values
            missing_threshold = config.get('missing_threshold', 0.5)
            valid_features = numeric_features.loc[:, numeric_features.isnull().mean() < missing_threshold]

            # Fill remaining missing values
            feature_matrix = valid_features.fillna(valid_features.median())

            feature_names = list(feature_matrix.columns)

            # Prepare datasets for each prediction target
            for target_name, target_values in targets.items():
                if target_name == 'metadata':
                    continue

                target_type = self.prediction_targets.get(target_name, 'regression')

                # Align target values with feature matrix
                if len(target_values) != len(feature_matrix):
                    min_length = min(len(target_values), len(feature_matrix))
                    target_values = target_values[:min_length]
                    current_feature_matrix = feature_matrix.iloc[:min_length]
                else:
                    current_feature_matrix = feature_matrix

                # Prepare target variable
                if target_type == 'classification':
                    # Encode categorical targets
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(target_values)
                    classes = label_encoder.classes_
                else:
                    y = np.array(target_values)
                    classes = None

                X = current_feature_matrix.values

                # Feature scaling
                scaler_type = config.get('scaler_type', 'standard')
                scaler = self.scalers[scaler_type]
                X_scaled = scaler.fit_transform(X)

                # Feature selection
                if config.get('feature_selection', True):
                    n_features = min(config.get('max_features', 20), X.shape[1])

                    if target_type == 'regression':
                        selector = SelectKBest(score_func=f_regression, k=n_features)
                    else:
                        selector = SelectKBest(score_func=chi2, k=n_features)

                    X_selected = selector.fit_transform(X_scaled, y)
                    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
                else:
                    X_selected = X_scaled
                    selected_features = feature_names
                    selector = None

                # Train-test split
                test_size = config.get('test_size', 0.2)
                stratify = y if target_type == 'classification' else None

                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=test_size, random_state=42, stratify=stratify
                )

                dataset = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'X_full': X_selected,
                    'y_full': y,
                    'feature_names': selected_features,
                    'scaler': scaler,
                    'feature_selector': selector,
                    'target_type': target_type,
                    'classes': classes.tolist() if classes is not None else None,
                    'original_feature_count': len(feature_names),
                    'selected_feature_count': len(selected_features)
                }

                training_datasets[target_name] = dataset

            return training_datasets

        except Exception as e:
            self.log_operation_error("prepare_training_datasets", e)
            raise

    async def _build_prediction_models(
        self,
        training_datasets: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build prediction models for all targets."""
        try:
            prediction_models = {}

            for target_name, dataset in training_datasets.items():
                target_type = dataset['target_type']
                X_train = dataset['X_train']
                y_train = dataset['y_train']

                target_models = {}

                # Select appropriate models based on target type
                if target_type == 'regression':
                    models_to_train = self.regression_models
                else:
                    models_to_train = self.classification_models

                # Train each model
                for model_name, model in models_to_train.items():
                    try:
                        # Clone the model to avoid interference
                        from sklearn.base import clone
                        model_instance = clone(model)

                        # Train the model
                        model_instance.fit(X_train, y_train)

                        # Store trained model
                        target_models[model_name] = {
                            'model': model_instance,
                            'model_type': target_type,
                            'feature_count': X_train.shape[1],
                            'training_samples': X_train.shape[0]
                        }

                        # Store in class for later use
                        model_key = f"{target_name}_{model_name}"
                        self.trained_models[model_key] = model_instance

                    except Exception as model_error:
                        target_models[model_name] = {'error': str(model_error)}
                        self.log_model_training_error(f"{target_name}_{model_name}", model_error)

                prediction_models[target_name] = target_models

            return prediction_models

        except Exception as e:
            self.log_operation_error("build_prediction_models", e)
            raise

    async def _evaluate_prediction_models(
        self,
        prediction_models: Dict[str, Any],
        training_datasets: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate prediction model performance."""
        try:
            model_evaluation = {}

            for target_name, target_models in prediction_models.items():
                dataset = training_datasets[target_name]
                target_type = dataset['target_type']
                X_test = dataset['X_test']
                y_test = dataset['y_test']
                X_full = dataset['X_full']
                y_full = dataset['y_full']

                target_evaluation = {}

                for model_name, model_info in target_models.items():
                    if 'error' in model_info:
                        target_evaluation[model_name] = model_info
                        continue

                    model = model_info['model']

                    try:
                        # Make predictions
                        y_pred = model.predict(X_test)

                        # Calculate metrics based on target type
                        if target_type == 'regression':
                            metrics = {
                                'mse': float(mean_squared_error(y_test, y_pred)),
                                'mae': float(mean_absolute_error(y_test, y_pred)),
                                'r2': float(r2_score(y_test, y_pred)),
                                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                                'explained_variance': float(explained_variance_score(y_test, y_pred))
                            }
                        else:
                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_test, y_pred, average='weighted'
                            )

                            metrics = {
                                'accuracy': float(accuracy),
                                'precision': float(precision),
                                'recall': float(recall),
                                'f1_score': float(f1)
                            }

                            # Add ROC AUC if binary classification
                            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                                y_proba = model.predict_proba(X_test)[:, 1]
                                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))

                        # Cross-validation scores
                        cv_folds = config.get('cv_folds', 5)
                        scoring = 'r2' if target_type == 'regression' else 'accuracy'

                        cv_scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring=scoring)
                        metrics['cv_scores'] = cv_scores.tolist()
                        metrics['cv_mean'] = float(cv_scores.mean())
                        metrics['cv_std'] = float(cv_scores.std())

                        # Feature importance (if available)
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = dict(zip(
                                dataset['feature_names'],
                                model.feature_importances_
                            ))
                            metrics['feature_importance'] = feature_importance

                        target_evaluation[model_name] = {
                            'metrics': metrics,
                            'predictions_sample': y_pred[:10].tolist(),
                            'ground_truth_sample': y_test[:10].tolist()
                        }

                    except Exception as eval_error:
                        target_evaluation[model_name] = {'evaluation_error': str(eval_error)}

                model_evaluation[target_name] = target_evaluation

            return model_evaluation

        except Exception as e:
            self.log_operation_error("evaluate_prediction_models", e)
            raise

    async def _analyze_feature_importance(
        self,
        prediction_models: Dict[str, Any],
        training_datasets: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze feature importance across all models."""
        try:
            feature_importance_analysis = {}

            # Aggregate feature importance across all models and targets
            all_feature_importances = {}

            for target_name, target_models in prediction_models.items():
                target_importance = {}

                for model_name, model_info in target_models.items():
                    if 'error' in model_info or 'model' not in model_info:
                        continue

                    model = model_info['model']

                    if hasattr(model, 'feature_importances_'):
                        dataset = training_datasets[target_name]
                        feature_names = dataset['feature_names']
                        importances = model.feature_importances_

                        feature_importance = dict(zip(feature_names, importances))
                        target_importance[model_name] = feature_importance

                        # Aggregate for overall analysis
                        for feature, importance in feature_importance.items():
                            if feature not in all_feature_importances:
                                all_feature_importances[feature] = []
                            all_feature_importances[feature].append(importance)

                feature_importance_analysis[target_name] = target_importance

            # Calculate overall feature rankings
            overall_rankings = {}
            for feature, importances in all_feature_importances.items():
                overall_rankings[feature] = {
                    'mean_importance': float(np.mean(importances)),
                    'std_importance': float(np.std(importances)),
                    'max_importance': float(np.max(importances)),
                    'model_count': len(importances)
                }

            # Sort by mean importance
            sorted_features = sorted(
                overall_rankings.items(),
                key=lambda x: x[1]['mean_importance'],
                reverse=True
            )

            feature_importance_analysis['overall_rankings'] = dict(sorted_features)
            feature_importance_analysis['top_features'] = dict(sorted_features[:10])

            return feature_importance_analysis

        except Exception as e:
            self.log_operation_error("analyze_feature_importance", e)
            return {"error": str(e)}

    async def _analyze_performance_factors(
        self,
        training_datasets: Dict[str, Any],
        prediction_models: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze factors affecting prediction performance."""
        try:
            performance_factors = {}

            # Model performance comparison
            model_performance_comparison = {}
            for target_name in training_datasets.keys():
                target_comparison = {}

                # Compare models for this target
                for model_name in self.regression_models.keys():
                    if model_name in prediction_models.get(target_name, {}):
                        model_info = prediction_models[target_name][model_name]
                        if 'metrics' in model_info:
                            target_comparison[model_name] = model_info['metrics']

                model_performance_comparison[target_name] = target_comparison

            performance_factors['model_performance_comparison'] = model_performance_comparison

            # Feature count vs performance analysis
            feature_performance_analysis = {}
            for target_name, dataset in training_datasets.items():
                feature_count = dataset['selected_feature_count']
                original_count = dataset['original_feature_count']

                feature_performance_analysis[target_name] = {
                    'original_feature_count': original_count,
                    'selected_feature_count': feature_count,
                    'feature_reduction_ratio': feature_count / original_count if original_count > 0 else 0
                }

            performance_factors['feature_performance_analysis'] = feature_performance_analysis

            # Data quality impact analysis
            data_quality_analysis = {}
            for target_name, dataset in training_datasets.items():
                training_size = dataset['X_train'].shape[0]
                test_size = dataset['X_test'].shape[0]

                data_quality_analysis[target_name] = {
                    'training_samples': training_size,
                    'test_samples': test_size,
                    'total_samples': training_size + test_size,
                    'train_test_ratio': training_size / (training_size + test_size)
                }

            performance_factors['data_quality_analysis'] = data_quality_analysis

            return performance_factors

        except Exception as e:
            self.log_operation_error("analyze_performance_factors", e)
            return {"error": str(e)}

    async def _generate_prediction_insights(
        self,
        prediction_models: Dict[str, Any],
        feature_importance: Dict[str, Any],
        performance_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights from prediction models."""
        try:
            insights = {}

            # Best performing models for each target
            best_models = {}
            for target_name, models in prediction_models.items():
                best_score = -float('inf')
                best_model_name = None

                for model_name, model_info in models.items():
                    if 'metrics' in model_info:
                        # Use appropriate metric for comparison
                        score = model_info['metrics'].get('r2', model_info['metrics'].get('accuracy', 0))
                        if score > best_score:
                            best_score = score
                            best_model_name = model_name

                if best_model_name:
                    best_models[target_name] = {
                        'model': best_model_name,
                        'score': best_score
                    }

            insights['best_models'] = best_models

            # Key prediction factors
            top_features = feature_importance.get('top_features', {})
            insights['key_prediction_factors'] = list(top_features.keys())[:5]

            # Prediction reliability assessment
            reliability_assessment = {}
            for target_name in prediction_models.keys():
                model_scores = []
                for model_name, model_info in prediction_models[target_name].items():
                    if 'metrics' in model_info:
                        cv_mean = model_info['metrics'].get('cv_mean', 0)
                        model_scores.append(cv_mean)

                if model_scores:
                    reliability_assessment[target_name] = {
                        'mean_cv_score': float(np.mean(model_scores)),
                        'std_cv_score': float(np.std(model_scores)),
                        'model_agreement': float(np.std(model_scores) < 0.1)  # Low std indicates agreement
                    }

            insights['reliability_assessment'] = reliability_assessment

            return insights

        except Exception as e:
            self.log_operation_error("generate_prediction_insights", e)
            return {"error": str(e)}

    async def _perform_scenario_analysis(
        self,
        prediction_models: Dict[str, Any],
        training_datasets: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform scenario analysis for different survey conditions."""
        try:
            scenario_analysis = {}

            # Define test scenarios
            scenarios = {
                'optimal_conditions': {
                    'weather_dry': 1.0,
                    'ground_sandy': 1.0,
                    'low_permittivity': 1.0,
                    'low_utility_density': 1.0,
                    'no_crossing': 1.0,
                    'no_contamination': 1.0
                },
                'challenging_conditions': {
                    'weather_rainy': 1.0,
                    'ground_clayey': 1.0,
                    'high_permittivity': 1.0,
                    'high_utility_density': 1.0,
                    'utility_crossing': 1.0,
                    'contamination_present': 1.0
                },
                'moderate_conditions': {
                    'weather_mixed': 0.5,
                    'ground_mixed': 0.5,
                    'medium_permittivity': 0.5,
                    'medium_utility_density': 0.5,
                    'some_crossing': 0.5,
                    'some_contamination': 0.5
                }
            }

            # For each scenario, create synthetic feature vectors and predict
            for scenario_name, scenario_features in scenarios.items():
                scenario_predictions = {}

                for target_name, target_models in prediction_models.items():
                    if not target_models:
                        continue

                    # Use the best model for this target
                    best_model_name = None
                    best_score = -float('inf')

                    for model_name, model_info in target_models.items():
                        if 'metrics' in model_info:
                            score = model_info['metrics'].get('cv_mean', 0)
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name

                    if best_model_name and 'model' in target_models[best_model_name]:
                        model = target_models[best_model_name]['model']
                        dataset = training_datasets[target_name]

                        # Create synthetic feature vector for scenario
                        feature_vector = self._create_scenario_feature_vector(
                            scenario_features, dataset['feature_names']
                        )

                        try:
                            prediction = model.predict([feature_vector])[0]
                            scenario_predictions[target_name] = {
                                'prediction': float(prediction),
                                'model_used': best_model_name
                            }
                        except Exception as pred_error:
                            scenario_predictions[target_name] = {'error': str(pred_error)}

                scenario_analysis[scenario_name] = scenario_predictions

            return scenario_analysis

        except Exception as e:
            self.log_operation_error("perform_scenario_analysis", e)
            return {"error": str(e)}

    def _create_scenario_feature_vector(
        self,
        scenario_features: Dict[str, float],
        feature_names: List[str]
    ) -> np.ndarray:
        """Create a feature vector for a given scenario."""
        # This is a simplified implementation
        # In practice, this would map scenario conditions to actual feature values
        feature_vector = np.zeros(len(feature_names))

        # Simple mapping of scenario features to model features
        for i, feature_name in enumerate(feature_names):
            # Default value
            feature_vector[i] = 0.5

            # Apply scenario-specific modifications
            if 'weather' in feature_name.lower():
                if 'weather_dry' in scenario_features:
                    feature_vector[i] = scenario_features['weather_dry']
                elif 'weather_rainy' in scenario_features:
                    feature_vector[i] = 1.0 - scenario_features['weather_rainy']

            elif 'ground' in feature_name.lower() and 'condition' in feature_name.lower():
                if 'ground_sandy' in scenario_features:
                    feature_vector[i] = scenario_features['ground_sandy']
                elif 'ground_clayey' in scenario_features:
                    feature_vector[i] = 1.0 - scenario_features['ground_clayey']

            elif 'utility' in feature_name.lower() and 'density' in feature_name.lower():
                if 'low_utility_density' in scenario_features:
                    feature_vector[i] = 1.0 - scenario_features['low_utility_density']
                elif 'high_utility_density' in scenario_features:
                    feature_vector[i] = scenario_features['high_utility_density']

        return feature_vector

    async def _generate_optimization_recommendations(
        self,
        prediction_models: Dict[str, Any],
        feature_importance: Dict[str, Any],
        scenario_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization recommendations for improving detection performance."""
        try:
            recommendations = {}

            # Environmental condition recommendations
            environmental_recommendations = []
            if 'optimal_conditions' in scenario_analysis:
                optimal_predictions = scenario_analysis['optimal_conditions']
                challenging_predictions = scenario_analysis['challenging_conditions']

                for target in optimal_predictions:
                    optimal_score = optimal_predictions[target].get('prediction', 0)
                    challenging_score = challenging_predictions.get(target, {}).get('prediction', 0)

                    if optimal_score > challenging_score:
                        improvement = optimal_score - challenging_score
                        environmental_recommendations.append({
                            'target': target,
                            'improvement_potential': float(improvement),
                            'recommendation': f"Optimize conditions for {target} - potential improvement: {improvement:.2f}"
                        })

            recommendations['environmental_optimization'] = environmental_recommendations

            # Feature-based recommendations
            top_features = feature_importance.get('top_features', {})
            feature_recommendations = []

            for feature, importance_info in list(top_features.items())[:5]:
                feature_recommendations.append({
                    'feature': feature,
                    'importance': importance_info.get('mean_importance', 0),
                    'recommendation': f"Focus on optimizing {feature} - high impact on prediction accuracy"
                })

            recommendations['feature_optimization'] = feature_recommendations

            # Survey planning recommendations
            survey_recommendations = [
                {
                    'category': 'Weather Planning',
                    'recommendation': 'Schedule surveys during dry conditions when possible',
                    'impact': 'High'
                },
                {
                    'category': 'Site Preparation',
                    'recommendation': 'Remove surface debris and vegetation where feasible',
                    'impact': 'Medium'
                },
                {
                    'category': 'Equipment Configuration',
                    'recommendation': 'Adjust GPR frequency based on utility density and material types',
                    'impact': 'High'
                },
                {
                    'category': 'Data Processing',
                    'recommendation': 'Apply environmental-specific processing algorithms',
                    'impact': 'Medium'
                }
            ]

            recommendations['survey_planning'] = survey_recommendations

            return recommendations

        except Exception as e:
            self.log_operation_error("generate_optimization_recommendations", e)
            return {"error": str(e)}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for prediction system."""
        return {
            'test_size': 0.2,
            'cv_folds': 5,
            'random_state': 42,
            'scaler_type': 'standard',
            'feature_selection': True,
            'max_features': 20,
            'missing_threshold': 0.5,
            'grid_search': False,
            'cross_validation': True
        }

    # Logging methods
    def log_model_training_error(self, model_name: str, error: Exception):
        """Log model training errors."""
        self.logger.error(f"Model training failed",
                         extra={
                             "model_name": model_name,
                             "error": str(error),
                             "operation": "model_training_error"
                         })