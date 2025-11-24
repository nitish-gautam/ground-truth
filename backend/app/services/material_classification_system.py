"""
Material Classification System for GPR Utility Detection
=======================================================

Advanced material classification system specifically designed for the University
of Twente dataset, handling 10+ different utility material types found in real
GPR surveys. Implements predictive models for material detection, property
analysis, and performance correlation.

Real materials in dataset:
- Steel
- polyVinylChloride (PVC)
- asbestosCement
- highDensityPolyEthylene (HDPE)
- polyEthylene
- paperInsulatedLeadCovered
- Concrete
- Clay
- Cast Iron
- Copper
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    cross_val_score, train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import joblib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRSurvey, GPRScan, GPRProcessingResult
from ..models.environmental import EnvironmentalData
from ..models.utilities import Utility, UtilityDetection
from ..models.ml_analytics import FeatureVector, MLModel, ModelPerformance


class MaterialClassificationSystem(LoggerMixin):
    """Advanced material classification system for GPR utility detection."""

    def __init__(self):
        super().__init__()

        # Real material types from Twente dataset
        self.material_types = {
            'steel': {
                'conductivity': 'high',
                'magnetic': True,
                'corrosion_risk': 'high',
                'typical_diameters': [100, 150, 200, 300, 500],
                'detection_difficulty': 'easy',
                'age_category': 'older'
            },
            'polyVinylChloride': {
                'conductivity': 'none',
                'magnetic': False,
                'corrosion_risk': 'none',
                'typical_diameters': [110, 125, 160, 200],
                'detection_difficulty': 'difficult',
                'age_category': 'modern'
            },
            'asbestosCement': {
                'conductivity': 'none',
                'magnetic': False,
                'corrosion_risk': 'low',
                'typical_diameters': [100, 150, 200, 300],
                'detection_difficulty': 'moderate',
                'age_category': 'older'
            },
            'highDensityPolyEthylene': {
                'conductivity': 'none',
                'magnetic': False,
                'corrosion_risk': 'none',
                'typical_diameters': [40, 63, 90, 110],
                'detection_difficulty': 'difficult',
                'age_category': 'modern'
            },
            'polyEthylene': {
                'conductivity': 'none',
                'magnetic': False,
                'corrosion_risk': 'none',
                'typical_diameters': [32, 50, 75, 100],
                'detection_difficulty': 'very_difficult',
                'age_category': 'modern'
            },
            'paperInsulatedLeadCovered': {
                'conductivity': 'high',
                'magnetic': False,
                'corrosion_risk': 'moderate',
                'typical_diameters': [75, 100, 150],
                'detection_difficulty': 'easy',
                'age_category': 'legacy'
            },
            'concrete': {
                'conductivity': 'low',
                'magnetic': False,
                'corrosion_risk': 'low',
                'typical_diameters': [300, 400, 500, 600, 800, 1000],
                'detection_difficulty': 'moderate',
                'age_category': 'older'
            },
            'clay': {
                'conductivity': 'none',
                'magnetic': False,
                'corrosion_risk': 'none',
                'typical_diameters': [100, 150, 200, 300],
                'detection_difficulty': 'very_difficult',
                'age_category': 'legacy'
            },
            'cast_iron': {
                'conductivity': 'high',
                'magnetic': True,
                'corrosion_risk': 'high',
                'typical_diameters': [100, 150, 200, 300, 400],
                'detection_difficulty': 'easy',
                'age_category': 'older'
            },
            'copper': {
                'conductivity': 'very_high',
                'magnetic': False,
                'corrosion_risk': 'low',
                'typical_diameters': [15, 22, 28, 35, 50],
                'detection_difficulty': 'very_easy',
                'age_category': 'modern'
            }
        }

        # Detection difficulty scores
        self.difficulty_scores = {
            'very_easy': 1.0,
            'easy': 0.8,
            'moderate': 0.6,
            'difficult': 0.4,
            'very_difficult': 0.2
        }

        # Initialize encoders and models
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.classification_models = {}
        self.feature_selectors = {}

    async def build_comprehensive_material_classification(
        self,
        metadata_df: pd.DataFrame,
        performance_data: Optional[pd.DataFrame] = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build comprehensive material classification system."""
        self.log_operation_start("comprehensive_material_classification")

        if config is None:
            config = self._get_default_config()

        try:
            # Parse and prepare material data
            material_data = await self._parse_material_data(metadata_df)

            # Extract material features
            material_features = await self._extract_material_features(material_data, config)

            # Build classification models
            classification_models = await self._build_classification_models(
                material_features, config
            )

            # Material property analysis
            property_analysis = await self._analyze_material_properties(
                material_data, material_features, config
            )

            # Detection performance correlation
            detection_correlation = await self._analyze_detection_performance_correlation(
                material_data, performance_data, config
            )

            # Environmental interaction analysis
            environmental_interaction = await self._analyze_material_environment_interaction(
                material_data, metadata_df, config
            )

            # Material clustering analysis
            clustering_analysis = await self._perform_material_clustering(
                material_features, config
            )

            # Predictive insights
            predictive_insights = await self._generate_predictive_insights(
                classification_models, property_analysis, detection_correlation
            )

            # Practical recommendations
            recommendations = await self._generate_material_recommendations(
                property_analysis, detection_correlation, environmental_interaction
            )

            results = {
                "material_data": material_data,
                "material_features": material_features,
                "classification_models": classification_models,
                "property_analysis": property_analysis,
                "detection_correlation": detection_correlation,
                "environmental_interaction": environmental_interaction,
                "clustering_analysis": clustering_analysis,
                "predictive_insights": predictive_insights,
                "recommendations": recommendations,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "config": config,
                    "material_types_analyzed": len(self.material_types)
                }
            }

            self.log_operation_complete("comprehensive_material_classification", len(results))
            return results

        except Exception as e:
            self.log_operation_error("comprehensive_material_classification", e)
            raise

    async def _parse_material_data(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Parse material data from Twente metadata format."""
        try:
            material_records = []

            for idx, row in metadata_df.iterrows():
                location_id = row.get('LocationID', f'location_{idx}')

                # Parse material lists
                materials_list = row.get('utility_materials_list', [])
                diameters_list = row.get('utility_diameters_list', [])
                disciplines_list = row.get('utility_disciplines_list', [])

                if isinstance(materials_list, list) and materials_list:
                    for i, material in enumerate(materials_list):
                        if material and material.strip():
                            # Clean material name
                            cleaned_material = material.strip().lower()

                            # Map to standard names
                            standard_material = self._map_to_standard_material(cleaned_material)

                            # Get corresponding diameter and discipline
                            diameter = None
                            if isinstance(diameters_list, list) and i < len(diameters_list):
                                try:
                                    diameter = float(diameters_list[i]) if diameters_list[i] else None
                                except (ValueError, TypeError):
                                    diameter = None

                            discipline = None
                            if isinstance(disciplines_list, list) and i < len(disciplines_list):
                                discipline = disciplines_list[i] if disciplines_list[i] else None

                            material_record = {
                                'location_id': location_id,
                                'material_raw': material,
                                'material_standard': standard_material,
                                'diameter': diameter,
                                'discipline': discipline,
                                'material_index': i,
                                'ground_condition': row.get('Ground condition'),
                                'ground_permittivity': row.get('Ground relative permittivity'),
                                'weather_condition': row.get('Weather condition'),
                                'land_use': row.get('Land use'),
                                'utility_count': row.get('Amount of utilities'),
                                'crossing': row.get('Utility crossing'),
                                'linear_path': row.get('Utility path linear')
                            }
                            material_records.append(material_record)

            material_df = pd.DataFrame(material_records)

            # Add material properties
            material_df = await self._add_material_properties(material_df)

            # Statistical summary
            summary_stats = await self._generate_material_summary_stats(material_df)

            return {
                'material_dataframe': material_df,
                'summary_statistics': summary_stats,
                'total_materials': len(material_df),
                'unique_materials': material_df['material_standard'].nunique() if not material_df.empty else 0,
                'unique_locations': material_df['location_id'].nunique() if not material_df.empty else 0
            }

        except Exception as e:
            self.log_operation_error("parse_material_data", e)
            raise

    def _map_to_standard_material(self, raw_material: str) -> str:
        """Map raw material names to standard classifications."""
        raw_material = raw_material.lower().strip()

        # Direct matches
        if raw_material in self.material_types:
            return raw_material

        # Pattern matching for variations
        material_patterns = {
            'steel': ['steel', 'staal'],
            'polyVinylChloride': ['pvc', 'polyvinylchloride', 'poly vinyl chloride'],
            'asbestosCement': ['asbestos', 'cement', 'asbestoscement'],
            'highDensityPolyEthylene': ['hdpe', 'highdensitypolyethylene', 'high density polyethylene'],
            'polyEthylene': ['pe', 'polyethylene', 'poly ethylene'],
            'paperInsulatedLeadCovered': ['pilc', 'lead', 'paper', 'paperinsulated'],
            'concrete': ['concrete', 'beton'],
            'clay': ['clay', 'klei', 'ceramic'],
            'cast_iron': ['cast iron', 'iron', 'ijzer', 'castiron'],
            'copper': ['copper', 'koper', 'cu']
        }

        for standard_name, patterns in material_patterns.items():
            for pattern in patterns:
                if pattern in raw_material:
                    return standard_name

        return 'unknown'

    async def _add_material_properties(self, material_df: pd.DataFrame) -> pd.DataFrame:
        """Add material properties to the dataframe."""
        try:
            if material_df.empty:
                return material_df

            # Add properties from material_types dictionary
            for prop in ['conductivity', 'magnetic', 'corrosion_risk', 'detection_difficulty', 'age_category']:
                material_df[prop] = material_df['material_standard'].map(
                    lambda x: self.material_types.get(x, {}).get(prop, 'unknown')
                )

            # Add detection difficulty scores
            material_df['detection_difficulty_score'] = material_df['detection_difficulty'].map(
                lambda x: self.difficulty_scores.get(x, 0.5)
            )

            # Add conductivity scores
            conductivity_scores = {
                'very_high': 1.0, 'high': 0.8, 'moderate': 0.6,
                'low': 0.4, 'none': 0.0, 'unknown': 0.3
            }
            material_df['conductivity_score'] = material_df['conductivity'].map(
                lambda x: conductivity_scores.get(x, 0.3)
            )

            # Add magnetic properties (binary)
            material_df['is_magnetic'] = material_df['magnetic'].astype(int)

            # Add age category scores
            age_scores = {'legacy': 1, 'older': 2, 'modern': 3, 'unknown': 0}
            material_df['age_score'] = material_df['age_category'].map(
                lambda x: age_scores.get(x, 0)
            )

            # Add diameter categories
            material_df['diameter_category'] = material_df['diameter'].apply(self._categorize_diameter)

            # Add material-specific features
            material_df = await self._add_material_specific_features(material_df)

            return material_df

        except Exception as e:
            self.log_operation_error("add_material_properties", e)
            return material_df

    def _categorize_diameter(self, diameter: Optional[float]) -> str:
        """Categorize utility diameter."""
        if diameter is None or pd.isna(diameter):
            return 'unknown'

        if diameter <= 50:
            return 'small'
        elif diameter <= 150:
            return 'medium'
        elif diameter <= 300:
            return 'large'
        else:
            return 'very_large'

    async def _add_material_specific_features(self, material_df: pd.DataFrame) -> pd.DataFrame:
        """Add material-specific engineered features."""
        try:
            # Corrosion risk in different ground conditions
            corrosion_ground_risk = []
            for _, row in material_df.iterrows():
                base_risk = row.get('corrosion_risk', 'unknown')
                ground_condition = row.get('ground_condition', 'unknown')

                risk_score = 0
                if base_risk == 'high':
                    risk_score = 3
                elif base_risk == 'moderate':
                    risk_score = 2
                elif base_risk == 'low':
                    risk_score = 1

                # Increase risk in clayey conditions
                if ground_condition == 'Clayey':
                    risk_score += 1

                corrosion_ground_risk.append(risk_score)

            material_df['corrosion_ground_risk_score'] = corrosion_ground_risk

            # Detection suitability score
            detection_suitability = []
            for _, row in material_df.iterrows():
                score = row.get('detection_difficulty_score', 0.5)

                # Adjust for environmental conditions
                if row.get('weather_condition') == 'Rainy':
                    score *= 0.8  # Reduce in rainy conditions

                if row.get('ground_permittivity', 0) > 15:
                    score *= 0.9  # Reduce for high permittivity

                detection_suitability.append(score)

            material_df['detection_suitability_score'] = detection_suitability

            # Material compatibility with GPR frequency
            # Higher frequency better for smaller diameters, lower frequency for larger
            frequency_compatibility = []
            for _, row in material_df.iterrows():
                diameter = row.get('diameter', 100)
                material = row.get('material_standard', 'unknown')

                # Base compatibility
                if material in ['steel', 'copper', 'cast_iron']:
                    base_compat = 0.9  # Metal pipes are easier to detect
                elif material in ['polyVinylChloride', 'polyEthylene']:
                    base_compat = 0.4  # Plastic pipes are harder
                else:
                    base_compat = 0.6  # Moderate for others

                # Adjust for diameter
                if diameter and diameter > 200:
                    base_compat += 0.1  # Larger pipes easier to detect
                elif diameter and diameter < 50:
                    base_compat -= 0.1  # Smaller pipes harder

                frequency_compatibility.append(min(1.0, max(0.0, base_compat)))

            material_df['gpr_frequency_compatibility'] = frequency_compatibility

            return material_df

        except Exception as e:
            self.log_operation_error("add_material_specific_features", e)
            return material_df

    async def _extract_material_features(
        self,
        material_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive features for material classification."""
        try:
            material_df = material_data.get('material_dataframe', pd.DataFrame())

            if material_df.empty:
                return {"error": "No material data available for feature extraction"}

            features = {}

            # Basic material distribution features
            features['material_distribution'] = material_df['material_standard'].value_counts().to_dict()
            features['diameter_distribution'] = material_df['diameter_category'].value_counts().to_dict()
            features['conductivity_distribution'] = material_df['conductivity'].value_counts().to_dict()

            # Statistical features
            numeric_columns = material_df.select_dtypes(include=[np.number]).columns
            statistical_features = {}
            for col in numeric_columns:
                if material_df[col].notna().sum() > 0:
                    statistical_features[col] = {
                        'mean': float(material_df[col].mean()),
                        'std': float(material_df[col].std()),
                        'min': float(material_df[col].min()),
                        'max': float(material_df[col].max()),
                        'median': float(material_df[col].median())
                    }

            features['statistical_features'] = statistical_features

            # Material property correlation features
            property_correlations = await self._analyze_material_property_correlations(material_df)
            features['property_correlations'] = property_correlations

            # Environmental context features
            environmental_features = await self._extract_environmental_context_features(material_df)
            features['environmental_context'] = environmental_features

            # Detection complexity features
            complexity_features = await self._extract_detection_complexity_features(material_df)
            features['detection_complexity'] = complexity_features

            # Material clustering features
            if len(material_df) > 10:
                clustering_features = await self._extract_material_clustering_features(material_df)
                features['clustering_features'] = clustering_features

            return features

        except Exception as e:
            self.log_operation_error("extract_material_features", e)
            return {"error": str(e)}

    async def _build_classification_models(
        self,
        material_features: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build material classification models."""
        try:
            classification_results = {}

            # Check if we have the material dataframe
            if 'material_dataframe' not in material_features:
                return {"error": "Material dataframe not found in features"}

            # For this implementation, we'll create synthetic training data
            # In a real scenario, this would use actual GPR signal data
            training_data = await self._create_training_dataset(material_features, config)

            if training_data['X'].shape[0] < 10:
                return {"error": "Insufficient data for model training"}

            X = training_data['X']
            y = training_data['y']
            feature_names = training_data['feature_names']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=config.get('random_state', 42),
                stratify=y if len(np.unique(y)) > 1 else None
            )

            # Build multiple classification models
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=config.get('random_state', 42)
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    random_state=config.get('random_state', 42)
                ),
                'logistic_regression': LogisticRegression(
                    random_state=config.get('random_state', 42),
                    max_iter=1000
                )
            }

            # Train and evaluate models
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)

                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

                    model_results = {
                        'model_type': model_name,
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'cv_scores': cv_scores.tolist(),
                        'cv_mean': float(cv_scores.mean()),
                        'cv_std': float(cv_scores.std()),
                        'feature_names': feature_names
                    }

                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_names, model.feature_importances_))
                        model_results['feature_importance'] = feature_importance

                    # Classification report
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    model_results['classification_report'] = class_report

                    classification_results[model_name] = model_results

                    # Store model for later use
                    self.classification_models[model_name] = model

                except Exception as model_error:
                    classification_results[model_name] = {'error': str(model_error)}

            return classification_results

        except Exception as e:
            self.log_operation_error("build_classification_models", e)
            return {"error": str(e)}

    async def _create_training_dataset(
        self,
        material_features: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create training dataset for material classification."""
        # This is a simplified implementation
        # In practice, this would use actual GPR signal features

        material_df = material_features.get('material_dataframe', pd.DataFrame())

        if material_df.empty:
            return {"X": np.array([]), "y": np.array([]), "feature_names": []}

        # Select numeric features
        feature_columns = [
            'detection_difficulty_score', 'conductivity_score', 'is_magnetic',
            'age_score', 'detection_suitability_score', 'gpr_frequency_compatibility'
        ]

        available_features = [col for col in feature_columns if col in material_df.columns]

        if not available_features:
            return {"X": np.array([]), "y": np.array([]), "feature_names": []}

        X = material_df[available_features].fillna(0).values
        y = material_df['material_standard'].values

        # Filter out unknown materials
        mask = y != 'unknown'
        X = X[mask]
        y = y[mask]

        # Encode labels
        if len(y) > 0:
            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = np.array([])

        return {
            "X": X,
            "y": y_encoded,
            "feature_names": available_features,
            "label_encoder": self.label_encoder
        }

    async def _analyze_material_properties(
        self,
        material_data: Dict[str, Any],
        material_features: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze material properties and their relationships."""
        try:
            material_df = material_data.get('material_dataframe', pd.DataFrame())

            if material_df.empty:
                return {"error": "No material data available"}

            property_analysis = {}

            # Conductivity analysis
            conductivity_analysis = material_df.groupby('conductivity').agg({
                'detection_difficulty_score': ['mean', 'std', 'count'],
                'diameter': ['mean', 'std']
            }).to_dict()
            property_analysis['conductivity_analysis'] = conductivity_analysis

            # Magnetic property analysis
            if 'is_magnetic' in material_df.columns:
                magnetic_analysis = material_df.groupby('is_magnetic').agg({
                    'detection_difficulty_score': ['mean', 'std', 'count'],
                    'conductivity_score': ['mean', 'std']
                }).to_dict()
                property_analysis['magnetic_analysis'] = magnetic_analysis

            # Age category analysis
            age_analysis = material_df.groupby('age_category').agg({
                'detection_difficulty_score': ['mean', 'std', 'count'],
                'diameter': ['mean', 'std']
            }).to_dict()
            property_analysis['age_analysis'] = age_analysis

            # Diameter impact analysis
            if 'diameter' in material_df.columns:
                diameter_bins = pd.cut(material_df['diameter'].dropna(), bins=5, labels=False)
                diameter_analysis = material_df.groupby(diameter_bins).agg({
                    'detection_difficulty_score': ['mean', 'std', 'count']
                }).to_dict()
                property_analysis['diameter_analysis'] = diameter_analysis

            # Material-specific detection scores
            material_detection_scores = material_df.groupby('material_standard').agg({
                'detection_difficulty_score': ['mean', 'std', 'count'],
                'detection_suitability_score': ['mean', 'std'],
                'gpr_frequency_compatibility': ['mean', 'std']
            }).to_dict()
            property_analysis['material_detection_scores'] = material_detection_scores

            # Correlation analysis
            numeric_columns = material_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                correlation_matrix = material_df[numeric_columns].corr()
                property_analysis['correlation_matrix'] = correlation_matrix.to_dict()

            return property_analysis

        except Exception as e:
            self.log_operation_error("analyze_material_properties", e)
            return {"error": str(e)}

    async def _analyze_detection_performance_correlation(
        self,
        material_data: Dict[str, Any],
        performance_data: Optional[pd.DataFrame],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between material properties and detection performance."""
        try:
            correlation_analysis = {}

            material_df = material_data.get('material_dataframe', pd.DataFrame())

            if material_df.empty:
                return {"error": "No material data available"}

            # Internal correlations within material data
            internal_correlations = {}

            # Conductivity vs detection difficulty
            if 'conductivity_score' in material_df.columns and 'detection_difficulty_score' in material_df.columns:
                corr, p_value = stats.pearsonr(
                    material_df['conductivity_score'].fillna(0),
                    material_df['detection_difficulty_score'].fillna(0)
                )
                internal_correlations['conductivity_vs_detection_difficulty'] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }

            # Diameter vs detection scores
            if 'diameter' in material_df.columns:
                valid_diameter_mask = material_df['diameter'].notna()
                if valid_diameter_mask.sum() > 5:
                    corr, p_value = stats.pearsonr(
                        material_df.loc[valid_diameter_mask, 'diameter'],
                        material_df.loc[valid_diameter_mask, 'detection_difficulty_score']
                    )
                    internal_correlations['diameter_vs_detection_difficulty'] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

            correlation_analysis['internal_correlations'] = internal_correlations

            # External performance correlation (if performance data available)
            if performance_data is not None:
                # Placeholder for actual performance correlation
                correlation_analysis['external_performance'] = {
                    "status": "performance_data_available_but_correlation_not_implemented"
                }
            else:
                correlation_analysis['external_performance'] = {
                    "status": "no_performance_data_available"
                }

            # Material-specific performance insights
            performance_insights = await self._generate_material_performance_insights(material_df)
            correlation_analysis['performance_insights'] = performance_insights

            return correlation_analysis

        except Exception as e:
            self.log_operation_error("analyze_detection_performance_correlation", e)
            return {"error": str(e)}

    async def _generate_material_performance_insights(self, material_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights about material detection performance."""
        insights = {}

        # Best performing materials (easiest to detect)
        if 'detection_difficulty_score' in material_df.columns:
            best_materials = material_df.groupby('material_standard')['detection_difficulty_score'].mean().sort_values(ascending=False)
            insights['easiest_to_detect'] = best_materials.head(5).to_dict()
            insights['hardest_to_detect'] = best_materials.tail(5).to_dict()

        # Material recommendations by condition
        if 'ground_condition' in material_df.columns:
            condition_recommendations = {}
            for condition in material_df['ground_condition'].unique():
                if pd.notna(condition):
                    condition_materials = material_df[material_df['ground_condition'] == condition]
                    if len(condition_materials) > 0:
                        avg_scores = condition_materials.groupby('material_standard')['detection_suitability_score'].mean()
                        condition_recommendations[condition] = avg_scores.sort_values(ascending=False).head(3).to_dict()

            insights['condition_specific_recommendations'] = condition_recommendations

        return insights

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for material classification."""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'feature_selection_k': 10,
            'grid_search': False,
            'model_types': ['random_forest', 'gradient_boosting', 'logistic_regression']
        }

    async def _generate_material_summary_stats(self, material_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for material data."""
        if material_df.empty:
            return {"error": "No material data available"}

        summary = {
            'total_materials': len(material_df),
            'unique_materials': material_df['material_standard'].nunique(),
            'unique_locations': material_df['location_id'].nunique(),
            'materials_with_diameter': material_df['diameter'].notna().sum(),
            'materials_with_discipline': material_df['discipline'].notna().sum()
        }

        # Material type distribution
        material_dist = material_df['material_standard'].value_counts()
        summary['material_distribution'] = material_dist.to_dict()

        # Diameter statistics
        if material_df['diameter'].notna().sum() > 0:
            diameter_stats = material_df['diameter'].describe()
            summary['diameter_statistics'] = diameter_stats.to_dict()

        return summary

    # Placeholder methods for additional analyses
    async def _analyze_material_property_correlations(self, material_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between material properties."""
        return {"status": "material_property_correlations_placeholder"}

    async def _extract_environmental_context_features(self, material_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features related to environmental context."""
        return {"status": "environmental_context_features_placeholder"}

    async def _extract_detection_complexity_features(self, material_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features related to detection complexity."""
        return {"status": "detection_complexity_features_placeholder"}

    async def _extract_material_clustering_features(self, material_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for material clustering analysis."""
        return {"status": "material_clustering_features_placeholder"}

    async def _analyze_material_environment_interaction(self, material_data: Dict[str, Any], metadata_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interaction between materials and environmental conditions."""
        return {"status": "material_environment_interaction_placeholder"}

    async def _perform_material_clustering(self, material_features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis on materials."""
        return {"status": "material_clustering_placeholder"}

    async def _generate_predictive_insights(self, classification_models: Dict[str, Any], property_analysis: Dict[str, Any], detection_correlation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictive insights from models and analyses."""
        return {"status": "predictive_insights_placeholder"}

    async def _generate_material_recommendations(self, property_analysis: Dict[str, Any], detection_correlation: Dict[str, Any], environmental_interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate practical recommendations for material detection."""
        return {"status": "material_recommendations_placeholder"}