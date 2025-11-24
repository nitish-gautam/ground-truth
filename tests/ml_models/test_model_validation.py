#!/usr/bin/env python3
"""
ML Model Validation and Accuracy Testing Suite

This module provides comprehensive validation for machine learning models including:
1. Material Classification Models
2. GPR Signal Processing Models
3. Environmental Correlation Models
4. Utility Detection Models
5. Model Performance Metrics
6. Cross-validation and Testing
7. Model Drift Detection
8. Accuracy Benchmarking

Test components:
- Model loading and initialization validation
- Prediction accuracy testing
- Performance metric calculation
- Cross-validation framework
- Model robustness testing
- Feature importance validation
- Model interpretability testing
- Regression and classification metrics
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, MagicMock
import tempfile
import logging
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MLModelValidator:
    """Comprehensive ML model validation framework."""

    def __init__(self, model_path: Optional[Path] = None, model_type: str = 'classification'):
        """
        Initialize ML model validator.

        Args:
            model_path: Path to trained model file
            model_type: Type of model ('classification' or 'regression')
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model = None
        self.validation_results = {}

    def load_model(self) -> Dict[str, Any]:
        """Load and validate model file."""
        result = {
            'model_loaded': False,
            'model_type': self.model_type,
            'model_info': {},
            'loading_errors': []
        }

        try:
            if self.model_path and self.model_path.exists():
                # Try to load model
                self.model = joblib.load(self.model_path)
                result['model_loaded'] = True

                # Extract model information
                result['model_info'] = {
                    'model_class': type(self.model).__name__,
                    'has_predict_method': hasattr(self.model, 'predict'),
                    'has_predict_proba_method': hasattr(self.model, 'predict_proba'),
                    'has_feature_importances': hasattr(self.model, 'feature_importances_'),
                    'model_file_size_mb': self.model_path.stat().st_size / (1024 * 1024)
                }

            elif self.model_path:
                result['loading_errors'].append(f"Model file not found: {self.model_path}")
            else:
                # Create mock model for testing
                if self.model_type == 'classification':
                    self.model = DummyClassifier(strategy='most_frequent')
                else:
                    self.model = DummyRegressor(strategy='mean')
                result['model_loaded'] = True
                result['model_info']['model_class'] = 'Mock' + type(self.model).__name__

        except Exception as e:
            result['loading_errors'].append(f"Failed to load model: {str(e)}")
            logger.error(f"Model loading error: {e}")

        return result

    def validate_model_predictions(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Validate model predictions and calculate metrics."""
        result = {
            'prediction_valid': False,
            'metrics': {},
            'prediction_errors': [],
            'sample_predictions': {}
        }

        try:
            if self.model is None:
                result['prediction_errors'].append("Model not loaded")
                return result

            # Fit model if it's a dummy model
            if isinstance(self.model, (DummyClassifier, DummyRegressor)):
                self.model.fit(X_test, y_test)

            # Make predictions
            start_time = time.time()
            predictions = self.model.predict(X_test)
            prediction_time = (time.time() - start_time) * 1000

            result['sample_predictions'] = {
                'prediction_time_ms': prediction_time,
                'predictions_shape': predictions.shape,
                'first_5_predictions': predictions[:5].tolist() if len(predictions) > 0 else [],
                'prediction_distribution': self._get_prediction_distribution(predictions)
            }

            # Calculate metrics based on model type
            if self.model_type == 'classification':
                result['metrics'] = self._calculate_classification_metrics(y_test, predictions)
            else:
                result['metrics'] = self._calculate_regression_metrics(y_test, predictions)

            # Test prediction probabilities if available
            if hasattr(self.model, 'predict_proba') and self.model_type == 'classification':
                try:
                    probabilities = self.model.predict_proba(X_test)
                    result['sample_predictions']['probabilities_available'] = True
                    result['sample_predictions']['probability_shape'] = probabilities.shape
                    result['sample_predictions']['first_5_probabilities'] = probabilities[:5].tolist()
                except Exception as e:
                    result['prediction_errors'].append(f"Probability prediction failed: {str(e)}")

            result['prediction_valid'] = len(result['prediction_errors']) == 0

        except Exception as e:
            result['prediction_errors'].append(f"Prediction validation failed: {str(e)}")
            logger.error(f"Prediction validation error: {e}")

        return result

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {}

        try:
            # Basic metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

            # Precision, recall, F1-score
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics['precision_weighted'] = float(precision)
            metrics['recall_weighted'] = float(recall)
            metrics['f1_score_weighted'] = float(f1)

            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = \
                precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            metrics['per_class_metrics'] = {}

            for i, label in enumerate(unique_labels):
                if i < len(precision_per_class):
                    metrics['per_class_metrics'][str(label)] = {
                        'precision': float(precision_per_class[i]),
                        'recall': float(recall_per_class[i]),
                        'f1_score': float(f1_per_class[i]),
                        'support': int(support_per_class[i])
                    }

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            # Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            metrics['true_class_distribution'] = dict(zip(unique.astype(str), counts.astype(int)))

            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            metrics['predicted_class_distribution'] = dict(zip(unique_pred.astype(str), counts_pred.astype(int)))

        except Exception as e:
            metrics['calculation_error'] = str(e)

        return metrics

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        metrics = {}

        try:
            # Basic metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2_score'] = float(r2_score(y_true, y_pred))

            # Additional metrics
            residuals = y_true - y_pred
            metrics['mean_residual'] = float(np.mean(residuals))
            metrics['std_residual'] = float(np.std(residuals))
            metrics['max_absolute_error'] = float(np.max(np.abs(residuals)))

            # Percentage errors
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['mape'] = float(mape)

            # Distribution statistics
            metrics['true_value_stats'] = {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true))
            }

            metrics['predicted_value_stats'] = {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            }

        except Exception as e:
            metrics['calculation_error'] = str(e)

        return metrics

    def _get_prediction_distribution(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Get distribution of predictions."""
        distribution = {}

        try:
            if self.model_type == 'classification':
                unique, counts = np.unique(predictions, return_counts=True)
                distribution['class_counts'] = dict(zip(unique.astype(str), counts.astype(int)))
            else:
                distribution['statistics'] = {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions)),
                    'percentiles': {
                        '25': float(np.percentile(predictions, 25)),
                        '50': float(np.percentile(predictions, 50)),
                        '75': float(np.percentile(predictions, 75))
                    }
                }

        except Exception as e:
            distribution['error'] = str(e)

        return distribution

    def validate_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Validate model using cross-validation."""
        result = {
            'cv_valid': False,
            'cv_scores': [],
            'cv_statistics': {},
            'cv_errors': []
        }

        try:
            if self.model is None:
                result['cv_errors'].append("Model not loaded")
                return result

            # Prepare cross-validation
            if self.model_type == 'classification':
                cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                cv_strategy = cv_folds
                scoring = 'neg_mean_squared_error'

            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=cv_strategy, scoring=scoring)

            if self.model_type == 'regression':
                cv_scores = -cv_scores  # Convert negative MSE to positive

            result['cv_scores'] = cv_scores.tolist()
            result['cv_statistics'] = {
                'mean_score': float(np.mean(cv_scores)),
                'std_score': float(np.std(cv_scores)),
                'min_score': float(np.min(cv_scores)),
                'max_score': float(np.max(cv_scores)),
                'scoring_metric': scoring
            }

            result['cv_valid'] = True

        except Exception as e:
            result['cv_errors'].append(f"Cross-validation failed: {str(e)}")
            logger.error(f"Cross-validation error: {e}")

        return result

    def validate_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate feature importance analysis."""
        result = {
            'feature_importance_available': False,
            'feature_importances': {},
            'top_features': [],
            'feature_errors': []
        }

        try:
            if self.model is None:
                result['feature_errors'].append("Model not loaded")
                return result

            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                result['feature_importance_available'] = True

                # Create feature importance dictionary
                if feature_names:
                    result['feature_importances'] = dict(zip(feature_names, importances.tolist()))
                else:
                    result['feature_importances'] = {f'feature_{i}': imp for i, imp in enumerate(importances)}

                # Get top features
                if feature_names:
                    feature_importance_pairs = list(zip(feature_names, importances))
                else:
                    feature_importance_pairs = [(f'feature_{i}', imp) for i, imp in enumerate(importances)]

                # Sort by importance and get top 10
                sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
                result['top_features'] = [
                    {'feature': name, 'importance': float(importance)}
                    for name, importance in sorted_features[:10]
                ]

            elif hasattr(self.model, 'coef_'):
                # For linear models, use coefficients as feature importance
                coef = self.model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)  # Average across classes for multi-class
                else:
                    coef = np.abs(coef)

                result['feature_importance_available'] = True

                if feature_names:
                    result['feature_importances'] = dict(zip(feature_names, coef.tolist()))
                else:
                    result['feature_importances'] = {f'feature_{i}': imp for i, imp in enumerate(coef)}

                # Get top features
                if feature_names:
                    feature_importance_pairs = list(zip(feature_names, coef))
                else:
                    feature_importance_pairs = [(f'feature_{i}', imp) for i, imp in enumerate(coef)]

                sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
                result['top_features'] = [
                    {'feature': name, 'importance': float(importance)}
                    for name, importance in sorted_features[:10]
                ]

            else:
                result['feature_errors'].append("Model does not support feature importance")

        except Exception as e:
            result['feature_errors'].append(f"Feature importance validation failed: {str(e)}")

        return result

    def validate_model_robustness(self, X_test: np.ndarray, y_test: np.ndarray,
                                noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, Any]:
        """Validate model robustness to input noise."""
        result = {
            'robustness_tests': {},
            'baseline_performance': {},
            'robustness_scores': {},
            'robustness_errors': []
        }

        try:
            if self.model is None:
                result['robustness_errors'].append("Model not loaded")
                return result

            # Get baseline performance
            baseline_predictions = self.model.predict(X_test)
            if self.model_type == 'classification':
                baseline_score = accuracy_score(y_test, baseline_predictions)
                result['baseline_performance']['accuracy'] = float(baseline_score)
            else:
                baseline_score = r2_score(y_test, baseline_predictions)
                result['baseline_performance']['r2_score'] = float(baseline_score)

            # Test robustness with different noise levels
            for noise_level in noise_levels:
                noise_result = {}

                try:
                    # Add noise to input features
                    noise = np.random.normal(0, noise_level * np.std(X_test, axis=0), X_test.shape)
                    X_noisy = X_test + noise

                    # Make predictions on noisy data
                    noisy_predictions = self.model.predict(X_noisy)

                    if self.model_type == 'classification':
                        noisy_score = accuracy_score(y_test, noisy_predictions)
                        performance_drop = baseline_score - noisy_score
                        noise_result = {
                            'accuracy': float(noisy_score),
                            'performance_drop': float(performance_drop),
                            'relative_performance': float(noisy_score / baseline_score) if baseline_score > 0 else 0
                        }
                    else:
                        noisy_score = r2_score(y_test, noisy_predictions)
                        performance_drop = baseline_score - noisy_score
                        noise_result = {
                            'r2_score': float(noisy_score),
                            'performance_drop': float(performance_drop),
                            'relative_performance': float(noisy_score / baseline_score) if baseline_score > 0 else 0
                        }

                    result['robustness_tests'][f'noise_{noise_level}'] = noise_result

                except Exception as e:
                    result['robustness_errors'].append(f"Noise level {noise_level} test failed: {str(e)}")

            # Calculate overall robustness score
            if result['robustness_tests']:
                relative_performances = [
                    test['relative_performance'] for test in result['robustness_tests'].values()
                ]
                result['robustness_scores']['mean_relative_performance'] = float(np.mean(relative_performances))
                result['robustness_scores']['min_relative_performance'] = float(np.min(relative_performances))

        except Exception as e:
            result['robustness_errors'].append(f"Robustness validation failed: {str(e)}")

        return result

    def benchmark_prediction_performance(self, X_test: np.ndarray, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model prediction performance."""
        result = {
            'performance_benchmarks': {},
            'benchmark_errors': []
        }

        try:
            if self.model is None:
                result['benchmark_errors'].append("Model not loaded")
                return result

            prediction_times = []
            memory_usage = []

            for i in range(iterations):
                start_time = time.time()

                # Make prediction
                predictions = self.model.predict(X_test)

                prediction_time = (time.time() - start_time) * 1000  # milliseconds
                prediction_times.append(prediction_time)

            result['performance_benchmarks'] = {
                'mean_prediction_time_ms': float(np.mean(prediction_times)),
                'std_prediction_time_ms': float(np.std(prediction_times)),
                'min_prediction_time_ms': float(np.min(prediction_times)),
                'max_prediction_time_ms': float(np.max(prediction_times)),
                'throughput_predictions_per_second': float(len(X_test) / (np.mean(prediction_times) / 1000)),
                'iterations_tested': iterations,
                'samples_per_iteration': len(X_test)
            }

        except Exception as e:
            result['benchmark_errors'].append(f"Performance benchmarking failed: {str(e)}")

        return result


class MaterialClassificationValidator(MLModelValidator):
    """Specialized validator for material classification models."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize material classification validator."""
        super().__init__(model_path, 'classification')
        self.material_classes = [
            'steel', 'polyVinylChloride', 'concrete', 'cast_iron', 'polyethylene'
        ]

    def validate_material_prediction_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Validate material-specific prediction accuracy."""
        result = super().validate_model_predictions(X_test, y_test)

        # Add material-specific analysis
        if result['prediction_valid'] and 'metrics' in result:
            result['material_analysis'] = self._analyze_material_performance(y_test, self.model.predict(X_test))

        return result

    def _analyze_material_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze performance by material type."""
        analysis = {}

        try:
            # Convert labels to material names if they are numeric
            material_mapping = {i: material for i, material in enumerate(self.material_classes)}

            for material_id, material_name in material_mapping.items():
                material_mask = y_true == material_id
                if np.any(material_mask):
                    material_accuracy = accuracy_score(
                        y_true[material_mask], y_pred[material_mask]
                    )
                    analysis[material_name] = {
                        'accuracy': float(material_accuracy),
                        'sample_count': int(np.sum(material_mask)),
                        'correctly_classified': int(np.sum(y_pred[material_mask] == material_id))
                    }

        except Exception as e:
            analysis['analysis_error'] = str(e)

        return analysis


class GPRSignalValidator(MLModelValidator):
    """Specialized validator for GPR signal processing models."""

    def __init__(self, model_path: Optional[Path] = None, model_type: str = 'regression'):
        """Initialize GPR signal validator."""
        super().__init__(model_path, model_type)

    def validate_depth_estimation_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Validate depth estimation accuracy for GPR signals."""
        result = super().validate_model_predictions(X_test, y_test)

        if result['prediction_valid'] and self.model_type == 'regression':
            # Add depth-specific analysis
            predictions = self.model.predict(X_test)
            result['depth_analysis'] = self._analyze_depth_performance(y_test, predictions)

        return result

    def _analyze_depth_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze depth estimation performance."""
        analysis = {}

        try:
            # Depth range analysis
            depth_ranges = [(0, 1), (1, 2), (2, 3), (3, float('inf'))]
            range_names = ['shallow', 'medium', 'deep', 'very_deep']

            for (min_depth, max_depth), range_name in zip(depth_ranges, range_names):
                if max_depth == float('inf'):
                    mask = y_true >= min_depth
                else:
                    mask = (y_true >= min_depth) & (y_true < max_depth)

                if np.any(mask):
                    range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                    range_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

                    analysis[f'{range_name}_depths'] = {
                        'mae': float(range_mae),
                        'rmse': float(range_rmse),
                        'sample_count': int(np.sum(mask)),
                        'depth_range': f'{min_depth}-{max_depth}m'
                    }

        except Exception as e:
            analysis['analysis_error'] = str(e)

        return analysis


class TestMLModelValidation:
    """Test suite for ML model validation."""

    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)  # 3 classes

        return X, y

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 2 + 1  # Depth values

        return X, y

    @pytest.fixture
    def trained_classification_model(self, sample_classification_data):
        """Create a trained classification model for testing."""
        X, y = sample_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def trained_regression_model(self, sample_regression_data):
        """Create a trained regression model for testing."""
        X, y = sample_regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_model_loading_validation(self, trained_classification_model, tmp_path):
        """Test model loading validation."""
        # Save model to temporary file
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(trained_classification_model, model_path)

        # Test loading
        validator = MLModelValidator(model_path, 'classification')
        result = validator.load_model()

        assert result['model_loaded'] is True
        assert result['model_info']['model_class'] == 'RandomForestClassifier'
        assert result['model_info']['has_predict_method'] is True

    def test_classification_model_validation(self, sample_classification_data):
        """Test classification model validation."""
        X, y = sample_classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        # Create and train model
        validator = MLModelValidator(model_type='classification')
        validator.model = RandomForestClassifier(n_estimators=10, random_state=42)
        validator.model.fit(X_train, y_train)

        # Validate predictions
        result = validator.validate_model_predictions(X_test, y_test)

        assert result['prediction_valid'] is True
        assert 'accuracy' in result['metrics']
        assert 'precision_weighted' in result['metrics']
        assert 'confusion_matrix' in result['metrics']

    def test_regression_model_validation(self, sample_regression_data):
        """Test regression model validation."""
        X, y = sample_regression_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        # Create and train model
        validator = MLModelValidator(model_type='regression')
        validator.model = RandomForestRegressor(n_estimators=10, random_state=42)
        validator.model.fit(X_train, y_train)

        # Validate predictions
        result = validator.validate_model_predictions(X_test, y_test)

        assert result['prediction_valid'] is True
        assert 'mse' in result['metrics']
        assert 'rmse' in result['metrics']
        assert 'r2_score' in result['metrics']

    def test_cross_validation(self, sample_classification_data):
        """Test cross-validation functionality."""
        X, y = sample_classification_data

        validator = MLModelValidator(model_type='classification')
        validator.model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = validator.validate_cross_validation(X, y, cv_folds=3)

        assert result['cv_valid'] is True
        assert len(result['cv_scores']) == 3
        assert 'mean_score' in result['cv_statistics']

    def test_feature_importance_validation(self, sample_classification_data):
        """Test feature importance validation."""
        X, y = sample_classification_data

        validator = MLModelValidator(model_type='classification')
        validator.model = RandomForestClassifier(n_estimators=10, random_state=42)
        validator.model.fit(X, y)

        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        result = validator.validate_feature_importance(feature_names)

        assert result['feature_importance_available'] is True
        assert len(result['feature_importances']) == len(feature_names)
        assert len(result['top_features']) <= 10

    def test_model_robustness_validation(self, sample_classification_data):
        """Test model robustness validation."""
        X, y = sample_classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        validator = MLModelValidator(model_type='classification')
        validator.model = RandomForestClassifier(n_estimators=10, random_state=42)
        validator.model.fit(X_train, y_train)

        result = validator.validate_model_robustness(X_test, y_test, noise_levels=[0.01, 0.1])

        assert 'baseline_performance' in result
        assert 'robustness_tests' in result
        assert len(result['robustness_tests']) == 2

    def test_performance_benchmarking(self, sample_classification_data):
        """Test performance benchmarking."""
        X, y = sample_classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        validator = MLModelValidator(model_type='classification')
        validator.model = RandomForestClassifier(n_estimators=10, random_state=42)
        validator.model.fit(X_train, y_train)

        result = validator.benchmark_prediction_performance(X_test, iterations=5)

        assert 'performance_benchmarks' in result
        assert 'mean_prediction_time_ms' in result['performance_benchmarks']
        assert 'throughput_predictions_per_second' in result['performance_benchmarks']

    def test_material_classification_validator(self, sample_classification_data):
        """Test material classification validator."""
        X, y = sample_classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        validator = MaterialClassificationValidator()
        validator.model = RandomForestClassifier(n_estimators=10, random_state=42)
        validator.model.fit(X_train, y_train)

        result = validator.validate_material_prediction_accuracy(X_test, y_test)

        assert result['prediction_valid'] is True
        assert 'material_analysis' in result

    def test_gpr_signal_validator(self, sample_regression_data):
        """Test GPR signal validator."""
        X, y = sample_regression_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        validator = GPRSignalValidator(model_type='regression')
        validator.model = RandomForestRegressor(n_estimators=10, random_state=42)
        validator.model.fit(X_train, y_train)

        result = validator.validate_depth_estimation_accuracy(X_test, y_test)

        assert result['prediction_valid'] is True
        assert 'depth_analysis' in result

    def test_invalid_model_handling(self):
        """Test handling of invalid models."""
        validator = MLModelValidator(Path("nonexistent_model.pkl"), 'classification')
        result = validator.load_model()

        assert result['model_loaded'] is False
        assert len(result['loading_errors']) > 0

    def test_model_without_feature_importance(self, sample_classification_data):
        """Test handling of models without feature importance."""
        X, y = sample_classification_data

        validator = MLModelValidator(model_type='classification')
        validator.model = LogisticRegression(random_state=42)
        validator.model.fit(X, y)

        result = validator.validate_feature_importance()

        # Logistic regression has coef_ instead of feature_importances_
        assert result['feature_importance_available'] is True


@pytest.mark.integration
class TestMLModelIntegration:
    """Integration tests for ML model validation."""

    def test_end_to_end_model_validation(self, sample_classification_data, tmp_path):
        """Test end-to-end model validation workflow."""
        X, y = sample_classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        # Train and save model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        model_path = tmp_path / "integration_test_model.pkl"
        joblib.dump(model, model_path)

        # Validate model
        validator = MLModelValidator(model_path, 'classification')

        # Run all validation steps
        load_result = validator.load_model()
        prediction_result = validator.validate_model_predictions(X_test, y_test)
        cv_result = validator.validate_cross_validation(X, y)
        importance_result = validator.validate_feature_importance()
        robustness_result = validator.validate_model_robustness(X_test, y_test)
        performance_result = validator.benchmark_prediction_performance(X_test)

        # Assert all steps completed successfully
        assert load_result['model_loaded'] is True
        assert prediction_result['prediction_valid'] is True
        assert cv_result['cv_valid'] is True
        assert importance_result['feature_importance_available'] is True
        assert 'robustness_tests' in robustness_result
        assert 'performance_benchmarks' in performance_result


def create_ml_model_validation_suite(model_path: Optional[Path] = None,
                                   model_type: str = 'classification') -> MLModelValidator:
    """Factory function to create ML model validation suite."""
    return MLModelValidator(model_path, model_type)


def create_material_classification_suite(model_path: Optional[Path] = None) -> MaterialClassificationValidator:
    """Factory function to create material classification validation suite."""
    return MaterialClassificationValidator(model_path)


def create_gpr_signal_validation_suite(model_path: Optional[Path] = None,
                                     model_type: str = 'regression') -> GPRSignalValidator:
    """Factory function to create GPR signal validation suite."""
    return GPRSignalValidator(model_path, model_type)


if __name__ == '__main__':
    # Run ML model validation as standalone script
    import argparse

    parser = argparse.ArgumentParser(description='ML Model Validation Suite')
    parser.add_argument('--model-path', type=str, help='Path to trained model file')
    parser.add_argument('--model-type', type=str, choices=['classification', 'regression'],
                       default='classification', help='Type of model')
    parser.add_argument('--test-data', type=str, help='Path to test data CSV')
    parser.add_argument('--full-validation', action='store_true', help='Run full validation suite')

    args = parser.parse_args()

    # Create validator
    model_path = Path(args.model_path) if args.model_path else None
    validator = create_ml_model_validation_suite(model_path, args.model_type)

    if args.full_validation:
        print("Running comprehensive ML model validation...")

        load_result = validator.load_model()
        print(f"Model loading: {'PASS' if load_result['model_loaded'] else 'FAIL'}")

        if args.test_data and load_result['model_loaded']:
            # Load test data and run validation
            test_df = pd.read_csv(args.test_data)
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values

            prediction_result = validator.validate_model_predictions(X_test, y_test)
            print(f"Prediction validation: {'PASS' if prediction_result['prediction_valid'] else 'FAIL'}")

            cv_result = validator.validate_cross_validation(X_test, y_test)
            print(f"Cross-validation: {'PASS' if cv_result['cv_valid'] else 'FAIL'}")

            importance_result = validator.validate_feature_importance()
            print(f"Feature importance: {'AVAILABLE' if importance_result['feature_importance_available'] else 'NOT AVAILABLE'}")

    else:
        load_result = validator.load_model()
        print(f"Model validation result: {json.dumps(load_result, indent=2)}")