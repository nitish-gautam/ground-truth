"""
Twente Dataset Advanced Feature Extraction Service
=================================================

Comprehensive feature extraction and environmental correlation analysis
specifically designed for the University of Twente GPR utility detection dataset.

Processes all 25+ metadata fields including environmental conditions, utility
characteristics, material properties, and survey contexts to build predictive
models for GPR detection performance.
"""

import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRSurvey, GPRScan, GPRProcessingResult
from ..models.environmental import EnvironmentalData
from ..models.utilities import UtilityRecord
from ..models.ml_analytics import FeatureVector, MLModel, ModelPerformance


class TwenteFeatureExtractor(LoggerMixin):
    """Advanced feature extraction service for Twente GPR dataset."""

    def __init__(self):
        super().__init__()
        self.material_types = {
            'steel': 1, 'polyVinylChloride': 2, 'asbestosCement': 3,
            'highDensityPolyEthylene': 4, 'polyEthylene': 5,
            'paperInsulatedLeadCovered': 6, 'concrete': 7, 'clay': 8,
            'cast_iron': 9, 'copper': 10, 'unknown': 0
        }

        self.utility_disciplines = {
            'electricity': 1, 'water': 2, 'sewer': 3,
            'telecommunications': 4, 'oilGasChemicals': 5, 'unknown': 0
        }

        self.land_use_types = {
            'Residential': 1, 'Commercial': 2, 'Public institutions and service': 3,
            'Industrial': 4, 'Agricultural': 5, 'Recreational': 6, 'unknown': 0
        }

        # Initialize encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()

    async def extract_comprehensive_features(
        self,
        metadata_df: pd.DataFrame,
        performance_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Extract comprehensive features from Twente metadata."""
        self.log_operation_start("twente_feature_extraction")

        try:
            # Parse and clean the metadata
            cleaned_df = await self._parse_twente_metadata(metadata_df)

            # Extract feature categories
            features = {
                "environmental_features": await self._extract_environmental_features(cleaned_df),
                "utility_features": await self._extract_utility_features(cleaned_df),
                "material_features": await self._extract_material_features(cleaned_df),
                "survey_context_features": await self._extract_survey_context_features(cleaned_df),
                "spatial_features": await self._extract_spatial_features(cleaned_df),
                "complexity_features": await self._extract_complexity_features(cleaned_df),
                "derived_features": await self._create_derived_features(cleaned_df)
            }

            # Combine all features
            combined_features = await self._combine_feature_sets(features, cleaned_df)

            # Perform correlation analysis if performance data available
            if performance_data is not None:
                correlation_analysis = await self._perform_correlation_analysis(
                    combined_features, performance_data
                )
                features["correlation_analysis"] = correlation_analysis

            # Statistical analysis
            features["statistical_summary"] = await self._generate_statistical_summary(combined_features)

            # Feature importance analysis
            features["feature_importance"] = await self._analyze_feature_importance(combined_features)

            self.log_feature_extraction_complete("twente_comprehensive", len(combined_features))
            return features

        except Exception as e:
            self.log_operation_error("twente_feature_extraction", e)
            raise

    async def _parse_twente_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and clean the Twente metadata CSV structure."""
        try:
            # Handle the multi-line utility information format
            cleaned_rows = []

            for idx, row in df.iterrows():
                if pd.isna(row.iloc[0]) or str(row.iloc[0]).strip() == '':
                    continue

                # Parse utility disciplines (multi-line format)
                utility_disciplines = []
                utility_materials = []
                utility_diameters = []

                # Extract utility discipline
                discipline_text = str(row.get('Utility discipline', ''))
                if discipline_text and discipline_text != 'nan':
                    disciplines = [d.strip() for d in discipline_text.split('\n') if d.strip()]
                    utility_disciplines.extend(disciplines)

                # Extract utility materials
                material_text = str(row.get('Utility material', ''))
                if material_text and material_text != 'nan':
                    materials = [m.strip() for m in material_text.split('\n') if m.strip()]
                    utility_materials.extend(materials)

                # Extract utility diameters
                diameter_text = str(row.get('Utility diameter', ''))
                if diameter_text and diameter_text != 'nan':
                    diameters = [d.strip() for d in diameter_text.split('\n') if d.strip()]
                    utility_diameters.extend(diameters)

                # Create cleaned row
                cleaned_row = row.copy()
                cleaned_row['utility_disciplines_list'] = utility_disciplines
                cleaned_row['utility_materials_list'] = utility_materials
                cleaned_row['utility_diameters_list'] = utility_diameters

                cleaned_rows.append(cleaned_row)

            cleaned_df = pd.DataFrame(cleaned_rows)

            # Clean column names (remove extra spaces, special characters)
            cleaned_df.columns = [col.strip() for col in cleaned_df.columns]

            # Convert numeric columns
            numeric_columns = ['Ground relative permittivity', 'Amount of utilities']
            for col in numeric_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

            # Convert boolean columns
            boolean_columns = [
                'Exact location accuracy required', 'Rubble presence',
                'Tree roots presence', 'Polluted soil presence',
                'Blast-furnace slag presence', 'Utility crossing', 'Utility path linear'
            ]
            for col in boolean_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].map({'Yes': True, 'No': False})

            self.log_data_processing("parse_twente_metadata", len(cleaned_df))
            return cleaned_df

        except Exception as e:
            self.log_operation_error("parse_twente_metadata", e)
            raise

    async def _extract_environmental_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract environmental condition features."""
        try:
            env_features = {}

            # Ground condition features
            if 'Ground condition' in df.columns:
                ground_conditions = df['Ground condition'].value_counts()
                env_features['ground_condition_distribution'] = ground_conditions.to_dict()

                # Encode ground conditions
                if 'Ground condition' not in self.label_encoders:
                    self.label_encoders['Ground condition'] = LabelEncoder()
                    self.label_encoders['Ground condition'].fit(df['Ground condition'].fillna('unknown'))

                env_features['ground_condition_encoded'] = self.label_encoders['Ground condition'].transform(
                    df['Ground condition'].fillna('unknown')
                ).tolist()

            # Ground permittivity analysis
            if 'Ground relative permittivity' in df.columns:
                permittivity_data = df['Ground relative permittivity'].dropna()
                if len(permittivity_data) > 0:
                    env_features['permittivity_stats'] = {
                        'mean': float(permittivity_data.mean()),
                        'std': float(permittivity_data.std()),
                        'min': float(permittivity_data.min()),
                        'max': float(permittivity_data.max()),
                        'median': float(permittivity_data.median()),
                        'range': float(permittivity_data.max() - permittivity_data.min())
                    }

                    # Categorize permittivity
                    env_features['permittivity_categories'] = self._categorize_permittivity(permittivity_data).tolist()

            # Weather condition analysis
            if 'Weather condition' in df.columns:
                weather_dist = df['Weather condition'].value_counts()
                env_features['weather_distribution'] = weather_dist.to_dict()

                # Weather impact score (dry = 1.0, rainy = 0.6)
                weather_impact_map = {'Dry': 1.0, 'Rainy': 0.6, 'Cloudy': 0.8}
                env_features['weather_impact_scores'] = df['Weather condition'].map(weather_impact_map).fillna(0.8).tolist()

            # Terrain characteristics
            terrain_features = self._extract_terrain_features(df)
            env_features.update(terrain_features)

            # Subsurface contamination features
            contamination_features = self._extract_contamination_features(df)
            env_features.update(contamination_features)

            return env_features

        except Exception as e:
            self.log_operation_error("extract_environmental_features", e)
            return {"error": str(e)}

    async def _extract_utility_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract utility-specific features from the parsed data."""
        try:
            utility_features = {}

            # Utility density analysis
            if 'Amount of utilities' in df.columns:
                utility_counts = df['Amount of utilities'].dropna()
                if len(utility_counts) > 0:
                    utility_features['utility_density_stats'] = {
                        'mean': float(utility_counts.mean()),
                        'std': float(utility_counts.std()),
                        'min': int(utility_counts.min()),
                        'max': int(utility_counts.max()),
                        'median': float(utility_counts.median())
                    }

                    # Utility density categories
                    utility_features['utility_density_categories'] = self._categorize_utility_density(utility_counts).tolist()

            # Utility discipline analysis
            all_disciplines = []
            for disciplines_list in df['utility_disciplines_list']:
                if isinstance(disciplines_list, list):
                    all_disciplines.extend([d for d in disciplines_list if d and d.strip()])

            if all_disciplines:
                discipline_counts = pd.Series(all_disciplines).value_counts()
                utility_features['discipline_distribution'] = discipline_counts.to_dict()
                utility_features['unique_disciplines'] = list(discipline_counts.index)

                # Create discipline diversity score for each location
                discipline_diversity = []
                for disciplines_list in df['utility_disciplines_list']:
                    if isinstance(disciplines_list, list) and disciplines_list:
                        unique_disciplines = len(set(d for d in disciplines_list if d and d.strip()))
                        discipline_diversity.append(unique_disciplines)
                    else:
                        discipline_diversity.append(0)

                utility_features['discipline_diversity_scores'] = discipline_diversity

            # Utility crossing and path analysis
            if 'Utility crossing' in df.columns:
                crossing_rate = df['Utility crossing'].mean() if df['Utility crossing'].notna().any() else 0
                utility_features['utility_crossing_rate'] = float(crossing_rate)

            if 'Utility path linear' in df.columns:
                linearity_rate = df['Utility path linear'].mean() if df['Utility path linear'].notna().any() else 0
                utility_features['utility_linearity_rate'] = float(linearity_rate)

            # Utility complexity score
            complexity_scores = self._calculate_utility_complexity_scores(df)
            utility_features['utility_complexity_scores'] = complexity_scores

            return utility_features

        except Exception as e:
            self.log_operation_error("extract_utility_features", e)
            return {"error": str(e)}

    async def _extract_material_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract material classification and property features."""
        try:
            material_features = {}

            # Analyze material distribution
            all_materials = []
            for materials_list in df['utility_materials_list']:
                if isinstance(materials_list, list):
                    all_materials.extend([m for m in materials_list if m and m.strip()])

            if all_materials:
                material_counts = pd.Series(all_materials).value_counts()
                material_features['material_distribution'] = material_counts.to_dict()
                material_features['unique_materials'] = list(material_counts.index)

                # Material property analysis
                material_properties = self._analyze_material_properties(all_materials)
                material_features.update(material_properties)

                # Material diversity score for each location
                material_diversity = []
                for materials_list in df['utility_materials_list']:
                    if isinstance(materials_list, list) and materials_list:
                        unique_materials = len(set(m for m in materials_list if m and m.strip()))
                        material_diversity.append(unique_materials)
                    else:
                        material_diversity.append(0)

                material_features['material_diversity_scores'] = material_diversity

            # Diameter analysis
            all_diameters = []
            for diameters_list in df['utility_diameters_list']:
                if isinstance(diameters_list, list):
                    for d in diameters_list:
                        if d and d.strip():
                            try:
                                diameter_val = float(d.strip())
                                all_diameters.append(diameter_val)
                            except ValueError:
                                continue

            if all_diameters:
                diameter_array = np.array(all_diameters)
                material_features['diameter_stats'] = {
                    'mean': float(diameter_array.mean()),
                    'std': float(diameter_array.std()),
                    'min': float(diameter_array.min()),
                    'max': float(diameter_array.max()),
                    'median': float(np.median(diameter_array))
                }

                # Diameter categories
                material_features['diameter_categories'] = self._categorize_diameters(diameter_array)

                # Average diameter per location
                avg_diameters = []
                for diameters_list in df['utility_diameters_list']:
                    if isinstance(diameters_list, list) and diameters_list:
                        valid_diameters = []
                        for d in diameters_list:
                            if d and d.strip():
                                try:
                                    valid_diameters.append(float(d.strip()))
                                except ValueError:
                                    continue
                        avg_diameters.append(np.mean(valid_diameters) if valid_diameters else 0)
                    else:
                        avg_diameters.append(0)

                material_features['average_diameters'] = avg_diameters

            return material_features

        except Exception as e:
            self.log_operation_error("extract_material_features", e)
            return {"error": str(e)}

    async def _extract_survey_context_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract survey context and objective features."""
        try:
            context_features = {}

            # Survey objective analysis
            if 'Utility surveying objective' in df.columns:
                objectives = df['Utility surveying objective'].value_counts()
                context_features['survey_objective_distribution'] = objectives.to_dict()

                # Encode objectives
                if 'Utility surveying objective' not in self.label_encoders:
                    self.label_encoders['Utility surveying objective'] = LabelEncoder()
                    self.label_encoders['Utility surveying objective'].fit(
                        df['Utility surveying objective'].fillna('unknown')
                    )

                context_features['survey_objective_encoded'] = self.label_encoders['Utility surveying objective'].transform(
                    df['Utility surveying objective'].fillna('unknown')
                ).tolist()

            # Construction context
            if 'Construction workers' in df.columns:
                construction_types = df['Construction workers'].value_counts()
                context_features['construction_context_distribution'] = construction_types.to_dict()

            # Accuracy requirements
            if 'Exact location accuracy required' in df.columns:
                accuracy_required_rate = df['Exact location accuracy required'].mean()
                context_features['accuracy_requirement_rate'] = float(accuracy_required_rate) if not pd.isna(accuracy_required_rate) else 0

            # Land use context
            if 'Land use' in df.columns:
                land_use_dist = df['Land use'].value_counts()
                context_features['land_use_distribution'] = land_use_dist.to_dict()

                # Encode land use
                if 'Land use' not in self.label_encoders:
                    self.label_encoders['Land use'] = LabelEncoder()
                    self.label_encoders['Land use'].fit(df['Land use'].fillna('unknown'))

                context_features['land_use_encoded'] = self.label_encoders['Land use'].transform(
                    df['Land use'].fillna('unknown')
                ).tolist()

            # Complementary works impact
            if 'Complementary works' in df.columns:
                has_complementary_work = df['Complementary works'].notna() & (df['Complementary works'] != 'None')
                context_features['complementary_work_rate'] = float(has_complementary_work.mean())

            return context_features

        except Exception as e:
            self.log_operation_error("extract_survey_context_features", e)
            return {"error": str(e)}

    async def _extract_spatial_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract spatial and location-based features."""
        try:
            spatial_features = {}

            # Location ID patterns
            if 'LocationID' in df.columns:
                location_ids = df['LocationID'].dropna()

                # Extract location patterns
                location_patterns = {}
                for loc_id in location_ids:
                    # Extract numeric parts and patterns
                    pattern = re.sub(r'\d+', 'N', str(loc_id))
                    location_patterns[pattern] = location_patterns.get(pattern, 0) + 1

                spatial_features['location_patterns'] = location_patterns

                # Location sequence analysis
                sequences = []
                for loc_id in location_ids:
                    # Extract sequence number (e.g., '01.1' -> 1, '01.2' -> 2)
                    match = re.search(r'(\d+)\.(\d+)', str(loc_id))
                    if match:
                        sequences.append(int(match.group(2)))

                if sequences:
                    spatial_features['location_sequence_stats'] = {
                        'min': min(sequences),
                        'max': max(sequences),
                        'unique_count': len(set(sequences))
                    }

            # Land cover spatial features
            if 'Land cover' in df.columns:
                land_cover_dist = df['Land cover'].value_counts()
                spatial_features['land_cover_distribution'] = land_cover_dist.to_dict()

                # Surface permeability score
                permeability_map = {
                    'Grass': 1.0, 'Dirt': 0.9, 'Gravel': 0.7,
                    'Brick road concrete': 0.3, 'Concrete': 0.1, 'Asphalt': 0.1
                }
                spatial_features['surface_permeability_scores'] = df['Land cover'].map(permeability_map).fillna(0.5).tolist()

            # Land type features
            if 'Land type' in df.columns:
                land_type_dist = df['Land type'].value_counts()
                spatial_features['land_type_distribution'] = land_type_dist.to_dict()

            return spatial_features

        except Exception as e:
            self.log_operation_error("extract_spatial_features", e)
            return {"error": str(e)}

    async def _extract_complexity_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract complexity and difficulty assessment features."""
        try:
            complexity_features = {}

            # Environmental complexity score
            env_complexity_scores = []
            for idx, row in df.iterrows():
                score = 0

                # Weather impact
                if row.get('Weather condition') == 'Rainy':
                    score += 2
                elif row.get('Weather condition') == 'Cloudy':
                    score += 1

                # Ground condition complexity
                if row.get('Ground condition') == 'Clayey':
                    score += 2
                elif row.get('Ground condition') == 'Mixed':
                    score += 1

                # Contamination factors
                contamination_factors = [
                    'Rubble presence', 'Tree roots presence',
                    'Polluted soil presence', 'Blast-furnace slag presence'
                ]
                for factor in contamination_factors:
                    if row.get(factor) is True:
                        score += 1

                # Terrain difficulty
                if row.get('Terrain levelling') == 'Steep':
                    score += 2
                elif row.get('Terrain levelling') == 'Uneven':
                    score += 1

                if row.get('Terrain smoothness') == 'Rough':
                    score += 2

                env_complexity_scores.append(score)

            complexity_features['environmental_complexity_scores'] = env_complexity_scores

            # Utility detection difficulty score
            detection_difficulty_scores = []
            for idx, row in df.iterrows():
                score = 0

                # Utility density impact
                utility_count = row.get('Amount of utilities', 0)
                if utility_count > 15:
                    score += 3
                elif utility_count > 10:
                    score += 2
                elif utility_count > 5:
                    score += 1

                # Utility crossing complexity
                if row.get('Utility crossing') is True:
                    score += 2

                # Path non-linearity
                if row.get('Utility path linear') is False:
                    score += 2

                # Material diversity impact
                materials_list = row.get('utility_materials_list', [])
                if isinstance(materials_list, list) and len(materials_list) > 0:
                    unique_materials = len(set(m for m in materials_list if m and m.strip()))
                    if unique_materials > 5:
                        score += 2
                    elif unique_materials > 3:
                        score += 1

                detection_difficulty_scores.append(score)

            complexity_features['detection_difficulty_scores'] = detection_difficulty_scores

            # Overall survey complexity
            overall_complexity = []
            for env_score, det_score in zip(env_complexity_scores, detection_difficulty_scores):
                overall_complexity.append(env_score + det_score)

            complexity_features['overall_complexity_scores'] = overall_complexity

            # Complexity distribution analysis
            if overall_complexity:
                complexity_array = np.array(overall_complexity)
                complexity_features['complexity_stats'] = {
                    'mean': float(complexity_array.mean()),
                    'std': float(complexity_array.std()),
                    'min': int(complexity_array.min()),
                    'max': int(complexity_array.max()),
                    'quartiles': [float(q) for q in np.percentile(complexity_array, [25, 50, 75])]
                }

            return complexity_features

        except Exception as e:
            self.log_operation_error("extract_complexity_features", e)
            return {"error": str(e)}

    async def _create_derived_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create derived features from combinations of existing features."""
        try:
            derived_features = {}

            # Utility density per area type
            if 'Amount of utilities' in df.columns and 'Land use' in df.columns:
                density_by_land_use = df.groupby('Land use')['Amount of utilities'].agg(['mean', 'std']).to_dict()
                derived_features['utility_density_by_land_use'] = density_by_land_use

            # Material-ground condition interaction
            if 'Ground condition' in df.columns:
                ground_material_interaction = []
                for idx, row in df.iterrows():
                    ground_cond = row.get('Ground condition', 'unknown')
                    materials_list = row.get('utility_materials_list', [])

                    # Check for problematic material-ground combinations
                    score = 0
                    if isinstance(materials_list, list):
                        for material in materials_list:
                            if material and ground_cond == 'Clayey' and 'steel' in material.lower():
                                score += 2  # Steel in clay = higher corrosion risk
                            elif material and ground_cond == 'Sandy' and 'polyVinylChloride' in material.lower():
                                score += 1  # PVC in sand = good combination

                    ground_material_interaction.append(score)

                derived_features['ground_material_interaction_scores'] = ground_material_interaction

            # Weather-ground permittivity interaction
            if 'Weather condition' in df.columns and 'Ground relative permittivity' in df.columns:
                weather_permittivity_scores = []
                for idx, row in df.iterrows():
                    weather = row.get('Weather condition', 'unknown')
                    permittivity = row.get('Ground relative permittivity', 0)

                    # Calculate combined impact score
                    if weather == 'Rainy' and permittivity > 15:
                        score = 0.3  # High moisture + high permittivity = poor conditions
                    elif weather == 'Dry' and permittivity < 10:
                        score = 1.0  # Good conditions
                    else:
                        score = 0.7  # Moderate conditions

                    weather_permittivity_scores.append(score)

                derived_features['weather_permittivity_scores'] = weather_permittivity_scores

            # Survey efficiency indicator
            survey_efficiency = []
            for idx, row in df.iterrows():
                score = 1.0

                # Reduce efficiency for complex conditions
                if row.get('Exact location accuracy required') is True:
                    score *= 0.8

                if row.get('Utility crossing') is True:
                    score *= 0.7

                if row.get('Weather condition') == 'Rainy':
                    score *= 0.6

                survey_efficiency.append(score)

            derived_features['survey_efficiency_scores'] = survey_efficiency

            return derived_features

        except Exception as e:
            self.log_operation_error("create_derived_features", e)
            return {"error": str(e)}

    def _categorize_permittivity(self, permittivity_data: pd.Series) -> pd.Series:
        """Categorize ground permittivity into discrete categories."""
        return pd.cut(
            permittivity_data,
            bins=[0, 8, 12, 16, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )

    def _categorize_utility_density(self, utility_counts: pd.Series) -> pd.Series:
        """Categorize utility density into discrete levels."""
        return pd.cut(
            utility_counts,
            bins=[0, 3, 8, 15, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )

    def _categorize_diameters(self, diameters: np.ndarray) -> Dict[str, int]:
        """Categorize utility diameters into size categories."""
        categories = {
            'Small (≤50mm)': np.sum(diameters <= 50),
            'Medium (51-150mm)': np.sum((diameters > 50) & (diameters <= 150)),
            'Large (151-500mm)': np.sum((diameters > 150) & (diameters <= 500)),
            'Very Large (>500mm)': np.sum(diameters > 500)
        }
        return categories

    def _extract_terrain_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract terrain-related features."""
        terrain_features = {}

        if 'Terrain levelling' in df.columns:
            levelling_dist = df['Terrain levelling'].value_counts()
            terrain_features['terrain_levelling_distribution'] = levelling_dist.to_dict()

            # Terrain difficulty score
            difficulty_map = {'Flat': 1, 'Uneven': 2, 'Steep': 3}
            terrain_features['terrain_difficulty_scores'] = df['Terrain levelling'].map(difficulty_map).fillna(2).tolist()

        if 'Terrain smoothness' in df.columns:
            smoothness_dist = df['Terrain smoothness'].value_counts()
            terrain_features['terrain_smoothness_distribution'] = smoothness_dist.to_dict()

            # Smoothness impact score
            smoothness_map = {'Smooth': 1.0, 'Rough': 0.6}
            terrain_features['terrain_smoothness_scores'] = df['Terrain smoothness'].map(smoothness_map).fillna(0.8).tolist()

        return terrain_features

    def _extract_contamination_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract subsurface contamination features."""
        contamination_features = {}

        contamination_factors = [
            'Rubble presence', 'Tree roots presence',
            'Polluted soil presence', 'Blast-furnace slag presence'
        ]

        for factor in contamination_factors:
            if factor in df.columns:
                presence_rate = df[factor].mean() if df[factor].notna().any() else 0
                contamination_features[f'{factor.lower().replace(" ", "_")}_rate'] = float(presence_rate)

        # Total contamination score
        contamination_scores = []
        for idx, row in df.iterrows():
            score = sum(1 for factor in contamination_factors if row.get(factor) is True)
            contamination_scores.append(score)

        contamination_features['total_contamination_scores'] = contamination_scores

        return contamination_features

    def _analyze_material_properties(self, materials: List[str]) -> Dict[str, Any]:
        """Analyze material properties and characteristics."""
        material_properties = {}

        # Conductivity categories
        conductive_materials = ['steel', 'paperInsulatedLeadCovered', 'copper']
        non_conductive_materials = ['polyVinylChloride', 'polyEthylene', 'highDensityPolyEthylene']

        conductive_count = sum(1 for m in materials if any(cond in m for cond in conductive_materials))
        non_conductive_count = sum(1 for m in materials if any(non_cond in m for non_cond in non_conductive_materials))

        material_properties['conductive_material_ratio'] = conductive_count / len(materials) if materials else 0
        material_properties['non_conductive_material_ratio'] = non_conductive_count / len(materials) if materials else 0

        # Material age/durability categories
        older_materials = ['asbestosCement', 'paperInsulatedLeadCovered']
        modern_materials = ['highDensityPolyEthylene', 'polyEthylene']

        older_count = sum(1 for m in materials if any(old in m for old in older_materials))
        modern_count = sum(1 for m in materials if any(mod in m for mod in modern_materials))

        material_properties['older_material_ratio'] = older_count / len(materials) if materials else 0
        material_properties['modern_material_ratio'] = modern_count / len(materials) if materials else 0

        return material_properties

    def _calculate_utility_complexity_scores(self, df: pd.DataFrame) -> List[float]:
        """Calculate utility configuration complexity scores."""
        complexity_scores = []

        for idx, row in df.iterrows():
            score = 0.0

            # Base utility count impact
            utility_count = row.get('Amount of utilities', 0)
            score += min(utility_count * 0.1, 2.0)  # Cap at 2.0

            # Crossing penalty
            if row.get('Utility crossing') is True:
                score += 1.0

            # Non-linear path penalty
            if row.get('Utility path linear') is False:
                score += 0.5

            # Material diversity impact
            materials_list = row.get('utility_materials_list', [])
            if isinstance(materials_list, list) and materials_list:
                unique_materials = len(set(m for m in materials_list if m and m.strip()))
                score += unique_materials * 0.1

            # Discipline diversity impact
            disciplines_list = row.get('utility_disciplines_list', [])
            if isinstance(disciplines_list, list) and disciplines_list:
                unique_disciplines = len(set(d for d in disciplines_list if d and d.strip()))
                score += unique_disciplines * 0.15

            complexity_scores.append(score)

        return complexity_scores

    async def _combine_feature_sets(self, features: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature sets into a comprehensive feature matrix."""
        try:
            # Start with the original dataframe structure
            combined_df = df.copy()

            # Add environmental features
            env_features = features.get('environmental_features', {})
            for key, values in env_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'env_{key}'] = values

            # Add utility features
            utility_features = features.get('utility_features', {})
            for key, values in utility_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'utility_{key}'] = values

            # Add material features
            material_features = features.get('material_features', {})
            for key, values in material_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'material_{key}'] = values

            # Add survey context features
            context_features = features.get('survey_context_features', {})
            for key, values in context_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'context_{key}'] = values

            # Add spatial features
            spatial_features = features.get('spatial_features', {})
            for key, values in spatial_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'spatial_{key}'] = values

            # Add complexity features
            complexity_features = features.get('complexity_features', {})
            for key, values in complexity_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'complexity_{key}'] = values

            # Add derived features
            derived_features = features.get('derived_features', {})
            for key, values in derived_features.items():
                if isinstance(values, list) and len(values) == len(df):
                    combined_df[f'derived_{key}'] = values

            return combined_df

        except Exception as e:
            self.log_operation_error("combine_feature_sets", e)
            return df

    async def _perform_correlation_analysis(
        self,
        feature_df: pd.DataFrame,
        performance_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis between features and performance."""
        try:
            correlation_results = {}

            # Merge feature and performance data
            merged_df = pd.merge(feature_df, performance_df, on='LocationID', how='inner')

            if len(merged_df) < 5:
                return {"error": "Insufficient data for correlation analysis"}

            # Select numeric feature columns
            feature_columns = [col for col in merged_df.columns if col.startswith(('env_', 'utility_', 'material_', 'complexity_', 'derived_'))]
            numeric_features = merged_df[feature_columns].select_dtypes(include=[np.number])

            # Performance metrics to correlate against
            performance_metrics = ['detection_accuracy', 'confidence_score', 'signal_quality']
            available_metrics = [metric for metric in performance_metrics if metric in merged_df.columns]

            for metric in available_metrics:
                if metric in merged_df.columns:
                    metric_correlations = {}

                    for feature in numeric_features.columns:
                        if merged_df[feature].notna().sum() > 3:  # Need at least 4 valid values
                            try:
                                correlation, p_value = pearsonr(
                                    merged_df[feature].fillna(0),
                                    merged_df[metric].fillna(0)
                                )
                                metric_correlations[feature] = {
                                    'correlation': float(correlation),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05
                                }
                            except Exception:
                                continue

                    correlation_results[f'{metric}_correlations'] = metric_correlations

            # Feature importance using Random Forest
            if available_metrics and len(numeric_features.columns) > 0:
                for metric in available_metrics:
                    if metric in merged_df.columns:
                        X = numeric_features.fillna(0)
                        y = merged_df[metric].fillna(0)

                        if len(X) > 10:  # Minimum samples for RF
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X, y)

                            feature_importance = dict(zip(X.columns, rf.feature_importances_))
                            correlation_results[f'{metric}_feature_importance'] = feature_importance

            return correlation_results

        except Exception as e:
            self.log_operation_error("perform_correlation_analysis", e)
            return {"error": str(e)}

    async def _generate_statistical_summary(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary of features."""
        try:
            summary = {}

            # Basic statistics for numeric columns
            numeric_columns = feature_df.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if feature_df[col].notna().sum() > 0:
                    summary[col] = {
                        'count': int(feature_df[col].notna().sum()),
                        'mean': float(feature_df[col].mean()),
                        'std': float(feature_df[col].std()),
                        'min': float(feature_df[col].min()),
                        'max': float(feature_df[col].max()),
                        'median': float(feature_df[col].median())
                    }

            # Categorical statistics
            categorical_columns = feature_df.select_dtypes(include=['object', 'category']).columns

            for col in categorical_columns:
                if feature_df[col].notna().sum() > 0:
                    value_counts = feature_df[col].value_counts()
                    summary[f'{col}_distribution'] = value_counts.to_dict()

            # Data quality metrics
            summary['data_quality'] = {
                'total_samples': len(feature_df),
                'missing_data_percentage': float((feature_df.isnull().sum().sum() / (len(feature_df) * len(feature_df.columns))) * 100),
                'complete_cases': int(feature_df.dropna().shape[0])
            }

            return summary

        except Exception as e:
            self.log_operation_error("generate_statistical_summary", e)
            return {"error": str(e)}

    async def _analyze_feature_importance(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance using various methods."""
        try:
            importance_analysis = {}

            # Select numeric features
            numeric_features = feature_df.select_dtypes(include=[np.number])

            if len(numeric_features.columns) > 3:
                # Variance analysis
                feature_variances = numeric_features.var().sort_values(ascending=False)
                importance_analysis['variance_ranking'] = feature_variances.to_dict()

                # Correlation with target-like features
                if 'complexity_overall_complexity_scores' in numeric_features.columns:
                    target_correlations = numeric_features.corrwith(
                        numeric_features['complexity_overall_complexity_scores']
                    ).abs().sort_values(ascending=False)
                    importance_analysis['complexity_correlations'] = target_correlations.to_dict()

                # Principal Component Analysis
                if len(numeric_features) > 10:
                    try:
                        scaler = StandardScaler()
                        scaled_features = scaler.fit_transform(numeric_features.fillna(0))

                        pca = PCA(n_components=min(5, len(numeric_features.columns)))
                        pca.fit(scaled_features)

                        # Feature loadings for first few components
                        feature_loadings = {}
                        for i, component in enumerate(pca.components_[:3]):  # First 3 components
                            loadings = dict(zip(numeric_features.columns, component))
                            feature_loadings[f'PC{i+1}'] = {
                                k: float(v) for k, v in sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                            }

                        importance_analysis['pca_feature_loadings'] = feature_loadings
                        importance_analysis['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()

                    except Exception as pca_error:
                        importance_analysis['pca_error'] = str(pca_error)

            return importance_analysis

        except Exception as e:
            self.log_operation_error("analyze_feature_importance", e)
            return {"error": str(e)}

    # Logging methods
    def log_feature_extraction_complete(self, extraction_type: str, feature_count: int):
        """Log completion of feature extraction."""
        self.logger.info(f"Feature extraction completed",
                        extra={
                            "extraction_type": extraction_type,
                            "feature_count": feature_count,
                            "operation": "feature_extraction_complete"
                        })

    def log_environmental_correlation(self, analysis_type: str, correlation_count: int):
        """Log environmental correlation analysis."""
        self.logger.info(f"Environmental correlation analysis completed",
                        extra={
                            "analysis_type": analysis_type,
                            "correlation_count": correlation_count,
                            "operation": "environmental_correlation"
                        })

    def log_data_processing(self, process_type: str, record_count: int):
        """Log data processing operations."""
        self.logger.info(f"Data processing completed",
                        extra={
                            "process_type": process_type,
                            "record_count": record_count,
                            "operation": "data_processing"
                        })