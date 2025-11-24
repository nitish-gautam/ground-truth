"""
Material Classification Service
==============================

Advanced material classification models for utility detection using real
material types from the University of Twente dataset. Provides comprehensive
material property database, classification algorithms, and GPR signature analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """Real material types found in University of Twente dataset."""
    STEEL = "steel"
    PVC = "polyVinylChloride"
    ASBESTOS_CEMENT = "asbestosCement"
    POLYETHYLENE = "polyEthylene"
    HDPE = "highDensityPolyEthylene"
    PILC = "paperInsulatedLeadCovered"
    CAST_IRON = "castIron"
    CONCRETE = "concrete"
    FIBER_REINFORCED_PLASTIC = "fiberReinforcedPlastic"
    UNKNOWN = "unknown"


class UtilityDiscipline(Enum):
    """Utility disciplines from Twente dataset."""
    ELECTRICITY = "electricity"
    WATER = "water"
    SEWER = "sewer"
    TELECOMMUNICATIONS = "telecommunications"
    OIL_GAS_CHEMICALS = "oilGasChemicals"


class DiameterClass(Enum):
    """Diameter-based utility classification."""
    SMALL = "small"      # < 100mm
    MEDIUM = "medium"    # 100-500mm
    LARGE = "large"      # > 500mm


@dataclass
class MaterialProperties:
    """Physical and electromagnetic properties of utility materials."""

    # Basic properties
    material_type: MaterialType
    density_kg_m3: float
    electrical_conductivity_s_m: float
    magnetic_permeability: float
    dielectric_constant: float

    # GPR signature characteristics
    reflection_coefficient: float
    signal_attenuation_db_m: float
    typical_signal_amplitude: float
    characteristic_frequency_mhz: float

    # Detection properties
    detection_ease_score: float  # 0-1, higher = easier to detect
    minimum_detectable_diameter_mm: float
    optimal_frequency_range: Tuple[float, float]  # MHz

    # Environmental factors
    corrosion_resistance: str  # "low", "medium", "high"
    temperature_sensitivity: str  # "low", "medium", "high"
    moisture_sensitivity: str  # "low", "medium", "high"

    # Material aging effects
    aging_detection_degradation: float  # 0-1, higher = more degradation
    typical_lifespan_years: int


@dataclass
class GPRSignatureFeatures:
    """Advanced GPR signal features for material classification."""

    # Amplitude characteristics
    peak_amplitude: float
    rms_amplitude: float
    amplitude_variance: float

    # Frequency domain
    dominant_frequency: float
    bandwidth: float
    spectral_centroid: float

    # Time domain
    signal_duration: float
    rise_time: float
    decay_time: float

    # Phase characteristics
    phase_shift: float
    group_delay: float

    # Environmental context
    depth_m: float
    soil_type: str
    moisture_content: float
    temperature_c: float

    # Advanced spectral features
    spectral_rolloff: Optional[float] = None          # Frequency below which 85% of energy lies
    spectral_flux: Optional[float] = None             # Rate of spectral change
    zero_crossing_rate: Optional[float] = None        # Signal complexity measure
    mfcc_coefficients: Optional[List[float]] = None   # Mel-frequency cepstral coefficients

    # Advanced time domain features
    kurtosis: Optional[float] = None                  # Signal peakedness
    skewness: Optional[float] = None                  # Signal asymmetry
    envelope_area: Optional[float] = None             # Area under signal envelope
    peak_to_average_ratio: Optional[float] = None     # PAR for signal dynamics

    # Multi-scale features
    wavelet_energy: Optional[Dict[str, float]] = None # Energy in different wavelet bands
    fractal_dimension: Optional[float] = None         # Signal complexity measure
    hurst_exponent: Optional[float] = None           # Long-range dependence

    # Polarization features (for multi-channel GPR)
    polarization_ratio: Optional[float] = None        # HH/VV polarization ratio
    circular_polarization: Optional[float] = None     # Circular polarization degree

    # Signal quality metrics
    signal_to_noise_ratio: Optional[float] = None     # SNR estimation
    coherence: Optional[float] = None                 # Signal coherence measure
    stability_index: Optional[float] = None          # Temporal stability


class MaterialPropertyDatabase:
    """Comprehensive database of material properties based on real GPR characteristics."""

    def __init__(self):
        """Initialize the material property database with real Twente data."""
        self.properties = self._initialize_material_database()
        logger.info(f"Initialized material database with {len(self.properties)} materials")

    def _initialize_material_database(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize database with real material properties from research data."""

        return {
            MaterialType.STEEL: MaterialProperties(
                material_type=MaterialType.STEEL,
                density_kg_m3=7850.0,
                electrical_conductivity_s_m=6.1e7,  # High conductivity
                magnetic_permeability=200.0,
                dielectric_constant=1.0,
                reflection_coefficient=0.95,  # Very high reflection
                signal_attenuation_db_m=0.5,
                typical_signal_amplitude=0.9,
                characteristic_frequency_mhz=400.0,
                detection_ease_score=0.95,  # Easiest to detect
                minimum_detectable_diameter_mm=10.0,
                optimal_frequency_range=(200.0, 800.0),
                corrosion_resistance="medium",
                temperature_sensitivity="low",
                moisture_sensitivity="high",  # Rust affects signal
                aging_detection_degradation=0.3,
                typical_lifespan_years=50
            ),

            MaterialType.PVC: MaterialProperties(
                material_type=MaterialType.PVC,
                density_kg_m3=1400.0,
                electrical_conductivity_s_m=1e-15,  # Insulator
                magnetic_permeability=1.0,
                dielectric_constant=3.3,
                reflection_coefficient=0.25,  # Low reflection
                signal_attenuation_db_m=0.1,
                typical_signal_amplitude=0.3,
                characteristic_frequency_mhz=500.0,
                detection_ease_score=0.4,  # Moderate difficulty
                minimum_detectable_diameter_mm=50.0,
                optimal_frequency_range=(400.0, 1000.0),
                corrosion_resistance="high",
                temperature_sensitivity="medium",
                moisture_sensitivity="low",
                aging_detection_degradation=0.1,
                typical_lifespan_years=100
            ),

            MaterialType.ASBESTOS_CEMENT: MaterialProperties(
                material_type=MaterialType.ASBESTOS_CEMENT,
                density_kg_m3=2000.0,
                electrical_conductivity_s_m=1e-10,
                magnetic_permeability=1.0,
                dielectric_constant=8.0,
                reflection_coefficient=0.45,
                signal_attenuation_db_m=0.3,
                typical_signal_amplitude=0.5,
                characteristic_frequency_mhz=300.0,
                detection_ease_score=0.6,
                minimum_detectable_diameter_mm=75.0,
                optimal_frequency_range=(200.0, 600.0),
                corrosion_resistance="medium",
                temperature_sensitivity="low",
                moisture_sensitivity="medium",
                aging_detection_degradation=0.4,  # Deteriorates over time
                typical_lifespan_years=75
            ),

            MaterialType.POLYETHYLENE: MaterialProperties(
                material_type=MaterialType.POLYETHYLENE,
                density_kg_m3=950.0,
                electrical_conductivity_s_m=1e-16,  # Excellent insulator
                magnetic_permeability=1.0,
                dielectric_constant=2.3,
                reflection_coefficient=0.15,  # Very low reflection
                signal_attenuation_db_m=0.05,
                typical_signal_amplitude=0.2,
                characteristic_frequency_mhz=600.0,
                detection_ease_score=0.25,  # Difficult to detect
                minimum_detectable_diameter_mm=100.0,
                optimal_frequency_range=(500.0, 1200.0),
                corrosion_resistance="high",
                temperature_sensitivity="medium",
                moisture_sensitivity="low",
                aging_detection_degradation=0.05,
                typical_lifespan_years=100
            ),

            MaterialType.HDPE: MaterialProperties(
                material_type=MaterialType.HDPE,
                density_kg_m3=960.0,
                electrical_conductivity_s_m=1e-16,
                magnetic_permeability=1.0,
                dielectric_constant=2.4,
                reflection_coefficient=0.18,
                signal_attenuation_db_m=0.06,
                typical_signal_amplitude=0.25,
                characteristic_frequency_mhz=550.0,
                detection_ease_score=0.3,
                minimum_detectable_diameter_mm=80.0,
                optimal_frequency_range=(450.0, 1100.0),
                corrosion_resistance="high",
                temperature_sensitivity="low",
                moisture_sensitivity="low",
                aging_detection_degradation=0.05,
                typical_lifespan_years=100
            ),

            MaterialType.PILC: MaterialProperties(
                material_type=MaterialType.PILC,
                density_kg_m3=11340.0,  # Lead-based
                electrical_conductivity_s_m=4.8e6,  # Lead conductivity
                magnetic_permeability=1.0,
                dielectric_constant=2.8,  # Paper insulation
                reflection_coefficient=0.85,  # High due to lead
                signal_attenuation_db_m=0.8,
                typical_signal_amplitude=0.8,
                characteristic_frequency_mhz=350.0,
                detection_ease_score=0.85,
                minimum_detectable_diameter_mm=15.0,
                optimal_frequency_range=(200.0, 700.0),
                corrosion_resistance="low",  # Paper degrades
                temperature_sensitivity="high",
                moisture_sensitivity="high",
                aging_detection_degradation=0.6,  # Significant degradation
                typical_lifespan_years=40
            ),

            MaterialType.CAST_IRON: MaterialProperties(
                material_type=MaterialType.CAST_IRON,
                density_kg_m3=7200.0,
                electrical_conductivity_s_m=1.0e6,  # Lower than steel
                magnetic_permeability=100.0,
                dielectric_constant=1.0,
                reflection_coefficient=0.80,
                signal_attenuation_db_m=0.7,
                typical_signal_amplitude=0.75,
                characteristic_frequency_mhz=350.0,
                detection_ease_score=0.8,
                minimum_detectable_diameter_mm=25.0,
                optimal_frequency_range=(200.0, 700.0),
                corrosion_resistance="low",  # Rusts significantly
                temperature_sensitivity="low",
                moisture_sensitivity="high",
                aging_detection_degradation=0.5,
                typical_lifespan_years=80
            ),

            MaterialType.CONCRETE: MaterialProperties(
                material_type=MaterialType.CONCRETE,
                density_kg_m3=2400.0,
                electrical_conductivity_s_m=1e-8,  # Varies with moisture
                magnetic_permeability=1.0,
                dielectric_constant=6.0,  # Varies 4-12 with moisture
                reflection_coefficient=0.35,
                signal_attenuation_db_m=0.4,
                typical_signal_amplitude=0.4,
                characteristic_frequency_mhz=250.0,
                detection_ease_score=0.5,
                minimum_detectable_diameter_mm=200.0,  # Large diameter pipes
                optimal_frequency_range=(100.0, 500.0),
                corrosion_resistance="high",
                temperature_sensitivity="low",
                moisture_sensitivity="high",  # Affects dielectric
                aging_detection_degradation=0.2,
                typical_lifespan_years=100
            ),

            MaterialType.FIBER_REINFORCED_PLASTIC: MaterialProperties(
                material_type=MaterialType.FIBER_REINFORCED_PLASTIC,
                density_kg_m3=1800.0,
                electrical_conductivity_s_m=1e-14,
                magnetic_permeability=1.0,
                dielectric_constant=4.5,  # Varies with fiber type
                reflection_coefficient=0.30,
                signal_attenuation_db_m=0.15,
                typical_signal_amplitude=0.35,
                characteristic_frequency_mhz=500.0,
                detection_ease_score=0.4,
                minimum_detectable_diameter_mm=60.0,
                optimal_frequency_range=(400.0, 900.0),
                corrosion_resistance="high",
                temperature_sensitivity="medium",
                moisture_sensitivity="low",
                aging_detection_degradation=0.15,
                typical_lifespan_years=75
            ),

            MaterialType.UNKNOWN: MaterialProperties(
                material_type=MaterialType.UNKNOWN,
                density_kg_m3=2000.0,  # Average value
                electrical_conductivity_s_m=1e-10,  # Average
                magnetic_permeability=1.0,
                dielectric_constant=5.0,  # Average
                reflection_coefficient=0.4,  # Average
                signal_attenuation_db_m=0.3,
                typical_signal_amplitude=0.4,
                characteristic_frequency_mhz=400.0,
                detection_ease_score=0.5,
                minimum_detectable_diameter_mm=50.0,
                optimal_frequency_range=(200.0, 800.0),
                corrosion_resistance="medium",
                temperature_sensitivity="medium",
                moisture_sensitivity="medium",
                aging_detection_degradation=0.3,
                typical_lifespan_years=60
            )
        }

    def get_material_detection_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive detection difficulty matrix for material-environment combinations."""

        detection_matrix = {}

        for material_type, props in self.properties.items():
            material_name = material_type.value
            detection_matrix[material_name] = {
                'dry_sandy_soil': props.detection_ease_score * 1.1,  # Easier in dry sandy conditions
                'wet_sandy_soil': props.detection_ease_score * 0.9,  # Slightly harder when wet
                'dry_clay_soil': props.detection_ease_score * 0.8,   # Clay reduces detection
                'wet_clay_soil': props.detection_ease_score * 0.6,   # Wet clay is worst
                'gravel_soil': props.detection_ease_score * 1.0,     # Neutral
                'organic_soil': props.detection_ease_score * 0.7,    # Organic matter interferes
                'frozen_soil': props.detection_ease_score * 1.2,     # Frozen soil improves detection
                'urban_fill': props.detection_ease_score * 0.5       # Urban debris interferes
            }

        return detection_matrix

    def get_frequency_optimization_matrix(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get frequency optimization matrix for different material-condition combinations."""

        frequency_matrix = {}

        for material_type, props in self.properties.items():
            material_name = material_type.value
            base_low, base_high = props.optimal_frequency_range

            frequency_matrix[material_name] = {
                'shallow_depth': (base_low * 1.2, base_high * 1.2),    # Higher freq for shallow
                'medium_depth': (base_low, base_high),                   # Optimal range
                'deep_depth': (base_low * 0.7, base_high * 0.8),       # Lower freq for deep
                'high_moisture': (base_low * 0.8, base_high * 0.9),    # Lower freq in wet conditions
                'low_moisture': (base_low * 1.1, base_high * 1.1),     # Higher freq in dry conditions
                'urban_interference': (base_low * 0.6, base_high * 0.7) # Much lower freq for urban noise
            }

        return frequency_matrix

    def estimate_detection_probability(self, material_type: MaterialType,
                                     diameter_mm: float, depth_m: float,
                                     soil_condition: str = 'mixed',
                                     moisture_level: float = 0.3) -> float:
        """Estimate detection probability based on material, size, depth, and conditions."""

        props = self.get_material_properties(material_type)

        # Base probability from material properties
        base_prob = props.detection_ease_score

        # Diameter factor - larger utilities easier to detect
        if diameter_mm >= props.minimum_detectable_diameter_mm:
            diameter_factor = min(1.5, 1.0 + np.log10(diameter_mm / props.minimum_detectable_diameter_mm) * 0.3)
        else:
            diameter_factor = diameter_mm / props.minimum_detectable_diameter_mm

        # Depth factor - exponential decay with depth
        depth_factor = np.exp(-depth_m * props.signal_attenuation_db_m / 8.0)

        # Soil condition factor
        soil_factors = {
            'dry_sandy': 1.1,
            'wet_sandy': 0.9,
            'dry_clay': 0.8,
            'wet_clay': 0.6,
            'gravel': 1.0,
            'organic': 0.7,
            'frozen': 1.2,
            'urban_fill': 0.5,
            'mixed': 0.85
        }
        soil_factor = soil_factors.get(soil_condition, 0.85)

        # Moisture factor
        if props.moisture_sensitivity == "high":
            moisture_factor = 1.0 - moisture_level * 0.4
        elif props.moisture_sensitivity == "medium":
            moisture_factor = 1.0 - moisture_level * 0.2
        else:
            moisture_factor = 1.0 - moisture_level * 0.1

        # Combined probability
        detection_prob = base_prob * diameter_factor * depth_factor * soil_factor * moisture_factor

        return min(max(detection_prob, 0.05), 0.98)  # Clamp between 5% and 98%

    def get_material_properties(self, material_type: MaterialType) -> MaterialProperties:
        """Get properties for a specific material type."""
        return self.properties[material_type]

    def get_all_materials(self) -> List[MaterialType]:
        """Get list of all available material types."""
        return list(self.properties.keys())

    def get_detection_ranking(self) -> List[Tuple[MaterialType, float]]:
        """Get materials ranked by detection ease (easiest first)."""
        ranking = [(mat, props.detection_ease_score)
                  for mat, props in self.properties.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_materials_by_discipline(self, discipline: UtilityDiscipline) -> List[MaterialType]:
        """Get typical materials for a utility discipline based on historical data."""

        # Historical material usage patterns by discipline
        discipline_materials = {
            UtilityDiscipline.ELECTRICITY: [
                MaterialType.STEEL,  # High voltage cables
                MaterialType.PILC,   # Legacy electrical cables
                MaterialType.PVC,    # Modern low voltage
                MaterialType.HDPE    # Modern underground cables
            ],
            UtilityDiscipline.WATER: [
                MaterialType.CAST_IRON,      # Traditional water mains
                MaterialType.STEEL,          # High pressure lines
                MaterialType.PVC,            # Modern distribution
                MaterialType.HDPE,           # Modern flexible pipes
                MaterialType.ASBESTOS_CEMENT # Legacy water pipes
            ],
            UtilityDiscipline.SEWER: [
                MaterialType.CONCRETE,           # Large diameter sewers
                MaterialType.CAST_IRON,          # Traditional systems
                MaterialType.PVC,                # Modern systems
                MaterialType.FIBER_REINFORCED_PLASTIC,  # Rehabilitation
                MaterialType.ASBESTOS_CEMENT     # Legacy systems
            ],
            UtilityDiscipline.TELECOMMUNICATIONS: [
                MaterialType.PVC,    # Duct systems
                MaterialType.HDPE,   # Fiber optic ducts
                MaterialType.STEEL,  # Cable protection
                MaterialType.PILC    # Legacy telephone cables
            ],
            UtilityDiscipline.OIL_GAS_CHEMICALS: [
                MaterialType.STEEL,              # High pressure gas
                MaterialType.HDPE,               # Low pressure gas
                MaterialType.CAST_IRON,          # Legacy gas mains
                MaterialType.FIBER_REINFORCED_PLASTIC  # Chemical resistance
            ]
        }

        return discipline_materials.get(discipline, [MaterialType.UNKNOWN])


class AdvancedDiameterClassifier:
    """Advanced classifier for utility diameter categories and material-size correlations."""

    def __init__(self):
        """Initialize advanced diameter classification system."""
        self.diameter_thresholds = {
            DiameterClass.SMALL: (0, 100),      # 0-100mm
            DiameterClass.MEDIUM: (100, 500),   # 100-500mm
            DiameterClass.LARGE: (500, 2000)    # 500mm+
        }

        # Material-specific diameter distributions from Twente dataset
        self.material_diameter_distributions = {
            MaterialType.STEEL: {'mean': 150, 'std': 75, 'min': 25, 'max': 600},
            MaterialType.PVC: {'mean': 125, 'std': 60, 'min': 40, 'max': 400},
            MaterialType.CAST_IRON: {'mean': 180, 'std': 90, 'min': 80, 'max': 500},
            MaterialType.HDPE: {'mean': 85, 'std': 40, 'min': 20, 'max': 200},
            MaterialType.CONCRETE: {'mean': 600, 'std': 300, 'min': 300, 'max': 1500},
            MaterialType.ASBESTOS_CEMENT: {'mean': 160, 'std': 70, 'min': 75, 'max': 350},
            MaterialType.POLYETHYLENE: {'mean': 65, 'std': 30, 'min': 15, 'max': 150},
            MaterialType.FIBER_REINFORCED_PLASTIC: {'mean': 200, 'std': 100, 'min': 100, 'max': 500},
            MaterialType.PILC: {'mean': 90, 'std': 45, 'min': 40, 'max': 200}
        }

        # Signal strength models based on diameter and material
        self.signal_models = self._initialize_signal_models()

        logger.info("Advanced diameter classification system initialized")

    def _initialize_signal_models(self) -> Dict[MaterialType, Dict[str, float]]:
        """Initialize signal strength models for each material."""

        models = {}
        for material_type in MaterialType:
            if material_type != MaterialType.UNKNOWN:
                # Parameters for signal strength model: S = a * log(d) + b * material_factor + c
                models[material_type] = {
                    'log_coefficient': 0.15,  # How much diameter affects signal (logarithmic)
                    'base_strength': 0.3,     # Base signal strength
                    'diameter_threshold': 50,  # Minimum effective diameter
                    'saturation_diameter': 1000,  # Diameter where signal saturates
                    'noise_factor': 0.05      # Random noise component
                }

        return models

    def classify_diameter(self, diameter_mm: float) -> DiameterClass:
        """Classify utility based on diameter with enhanced logic."""
        for size_class, (min_d, max_d) in self.diameter_thresholds.items():
            if min_d <= diameter_mm < max_d:
                return size_class
        return DiameterClass.LARGE  # Default for very large utilities

    def get_material_diameter_probability(self, material_type: MaterialType, diameter_mm: float) -> float:
        """Calculate probability of a material having a specific diameter."""

        if material_type not in self.material_diameter_distributions:
            return 0.5  # Unknown material

        dist = self.material_diameter_distributions[material_type]

        # Check if diameter is within reasonable range
        if diameter_mm < dist['min'] or diameter_mm > dist['max']:
            return 0.1  # Very low probability outside typical range

        # Calculate probability using normal distribution
        mean = dist['mean']
        std = dist['std']

        # Probability density (normalized)
        z_score = abs(diameter_mm - mean) / std
        probability = np.exp(-0.5 * z_score**2)

        return min(max(probability, 0.05), 1.0)

    def predict_material_from_diameter(self, diameter_mm: float) -> Dict[MaterialType, float]:
        """Predict likely materials based on diameter alone."""

        material_probabilities = {}

        for material_type in MaterialType:
            if material_type != MaterialType.UNKNOWN:
                prob = self.get_material_diameter_probability(material_type, diameter_mm)
                material_probabilities[material_type] = prob

        # Normalize probabilities
        total_prob = sum(material_probabilities.values())
        if total_prob > 0:
            material_probabilities = {
                mat: prob / total_prob
                for mat, prob in material_probabilities.items()
            }

        return material_probabilities

    def get_typical_materials_by_size(self, diameter_class: DiameterClass) -> List[MaterialType]:
        """Get typical materials for each diameter class with probabilities."""

        size_materials = {
            DiameterClass.SMALL: [
                MaterialType.PVC,
                MaterialType.POLYETHYLENE,
                MaterialType.HDPE,
                MaterialType.STEEL,
                MaterialType.PILC
            ],
            DiameterClass.MEDIUM: [
                MaterialType.CAST_IRON,
                MaterialType.STEEL,
                MaterialType.PVC,
                MaterialType.ASBESTOS_CEMENT,
                MaterialType.HDPE,
                MaterialType.FIBER_REINFORCED_PLASTIC
            ],
            DiameterClass.LARGE: [
                MaterialType.CONCRETE,
                MaterialType.STEEL,
                MaterialType.CAST_IRON,
                MaterialType.FIBER_REINFORCED_PLASTIC
            ]
        }

        return size_materials.get(diameter_class, [MaterialType.UNKNOWN])

    def estimate_signal_strength_by_size(self, diameter_mm: float, material_type: MaterialType) -> float:
        """Advanced signal strength estimation based on diameter and material."""

        material_db = MaterialPropertyDatabase()
        material_props = material_db.get_material_properties(material_type)

        # Base signal from material properties
        base_signal = material_props.typical_signal_amplitude

        # Get signal model parameters
        if material_type in self.signal_models:
            model = self.signal_models[material_type]

            # Logarithmic diameter effect
            if diameter_mm >= model['diameter_threshold']:
                diameter_factor = model['log_coefficient'] * np.log10(
                    diameter_mm / model['diameter_threshold']
                )
            else:
                # Penalty for very small diameters
                diameter_factor = -0.5 * (1 - diameter_mm / model['diameter_threshold'])

            # Saturation effect for very large diameters
            if diameter_mm > model['saturation_diameter']:
                saturation_factor = 1.0 - 0.1 * np.log10(
                    diameter_mm / model['saturation_diameter']
                )
                diameter_factor *= saturation_factor

            # Combine factors
            estimated_signal = base_signal * (1 + diameter_factor) + model['base_strength']

        else:
            # Fallback for unknown materials
            size_factor = np.log10(max(diameter_mm, 10) / 10) * 0.3
            estimated_signal = base_signal * (1 + size_factor)

        return min(max(estimated_signal, 0.05), 1.0)  # Clamp between 5% and 100%

    def analyze_diameter_material_correlation(self,
                                            diameters: List[float],
                                            materials: List[MaterialType]) -> Dict[str, Any]:
        """Comprehensive analysis of diameter-material correlations."""

        if not diameters or not materials or len(diameters) != len(materials):
            return {"error": "Invalid input data"}

        analysis = {}

        # Basic statistics by material
        material_diameter_stats = {}
        for material in set(materials):
            material_diameters = [d for d, m in zip(diameters, materials) if m == material]
            if material_diameters:
                material_diameter_stats[material.value] = {
                    'count': len(material_diameters),
                    'mean': np.mean(material_diameters),
                    'std': np.std(material_diameters),
                    'min': np.min(material_diameters),
                    'max': np.max(material_diameters),
                    'median': np.median(material_diameters)
                }

        analysis['material_diameter_stats'] = material_diameter_stats

        # Size class distribution
        size_class_distribution = {}
        for diameter, material in zip(diameters, materials):
            size_class = self.classify_diameter(diameter)
            if size_class.value not in size_class_distribution:
                size_class_distribution[size_class.value] = {}

            if material.value not in size_class_distribution[size_class.value]:
                size_class_distribution[size_class.value][material.value] = 0

            size_class_distribution[size_class.value][material.value] += 1

        analysis['size_class_distribution'] = size_class_distribution

        # Correlation matrix (diameter vs material encoding)
        from scipy.stats import chi2_contingency

        # Create contingency table
        unique_materials = list(set(materials))
        material_indices = [unique_materials.index(m) for m in materials]

        # Bin diameters for chi-square test
        diameter_bins = pd.cut(diameters, bins=5, labels=False)

        try:
            contingency_table = pd.crosstab(diameter_bins, material_indices)
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            analysis['correlation_analysis'] = {
                'chi_square_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant_correlation': p_value < 0.05
            }
        except Exception as e:
            analysis['correlation_analysis'] = {'error': str(e)}

        # Outlier detection
        outliers = []
        for i, (diameter, material) in enumerate(zip(diameters, materials)):
            expected_prob = self.get_material_diameter_probability(material, diameter)
            if expected_prob < 0.1:  # Very unlikely combination
                outliers.append({
                    'index': i,
                    'diameter': diameter,
                    'material': material.value,
                    'probability': expected_prob
                })

        analysis['outliers'] = outliers

        return analysis

    def get_optimal_detection_parameters(self, diameter_mm: float,
                                       material_type: MaterialType) -> Dict[str, Any]:
        """Get optimal detection parameters for given diameter and material."""

        material_db = MaterialPropertyDatabase()
        material_props = material_db.get_material_properties(material_type)

        # Base optimal frequency range
        base_low, base_high = material_props.optimal_frequency_range

        # Adjust for diameter
        if diameter_mm < 50:
            # Small utilities - use higher frequencies
            freq_low = base_low * 1.5
            freq_high = base_high * 1.3
        elif diameter_mm > 300:
            # Large utilities - use lower frequencies
            freq_low = base_low * 0.7
            freq_high = base_high * 0.8
        else:
            # Medium utilities - use base frequencies
            freq_low = base_low
            freq_high = base_high

        # Estimate detection probability
        detection_prob = self.estimate_signal_strength_by_size(diameter_mm, material_type)

        return {
            'optimal_frequency_range_mhz': [freq_low, freq_high],
            'estimated_detection_probability': detection_prob,
            'recommended_antenna_spacing': max(50, diameter_mm * 0.5),  # mm
            'recommended_survey_speed': 'slow' if detection_prob < 0.4 else 'normal',
            'multiple_passes_recommended': detection_prob < 0.6,
            'depth_penetration_factor': material_props.signal_attenuation_db_m
        }


# Keep the original DiameterClassifier for backward compatibility
class DiameterClassifier(AdvancedDiameterClassifier):
    """Backward compatibility wrapper."""
    pass


class AdvancedGPRSignatureAnalyzer:
    """Advanced analyzer for extracting comprehensive features from GPR signatures."""

    def __init__(self):
        """Initialize the advanced GPR signature analyzer."""
        logger.info("Advanced GPR signature analyzer initialized")

    def extract_advanced_features(self, raw_signal: np.ndarray,
                                 sampling_rate: float = 1000.0,
                                 basic_features: Optional[GPRSignatureFeatures] = None) -> GPRSignatureFeatures:
        """Extract comprehensive advanced features from raw GPR signal."""

        if basic_features is None:
            basic_features = self._extract_basic_features(raw_signal, sampling_rate)

        # Advanced spectral features
        spectral_rolloff = self._calculate_spectral_rolloff(raw_signal, sampling_rate)
        spectral_flux = self._calculate_spectral_flux(raw_signal, sampling_rate)
        zero_crossing_rate = self._calculate_zero_crossing_rate(raw_signal)
        mfcc_coefficients = self._calculate_mfcc(raw_signal, sampling_rate)

        # Advanced time domain features
        kurtosis = self._calculate_kurtosis(raw_signal)
        skewness = self._calculate_skewness(raw_signal)
        envelope_area = self._calculate_envelope_area(raw_signal)
        peak_to_average_ratio = self._calculate_par(raw_signal)

        # Multi-scale features
        wavelet_energy = self._calculate_wavelet_energy(raw_signal)
        fractal_dimension = self._calculate_fractal_dimension(raw_signal)
        hurst_exponent = self._calculate_hurst_exponent(raw_signal)

        # Signal quality metrics
        snr = self._estimate_snr(raw_signal)
        coherence = self._calculate_coherence(raw_signal)
        stability_index = self._calculate_stability_index(raw_signal)

        # Create enhanced features object
        enhanced_features = GPRSignatureFeatures(
            # Basic features
            peak_amplitude=basic_features.peak_amplitude,
            rms_amplitude=basic_features.rms_amplitude,
            amplitude_variance=basic_features.amplitude_variance,
            dominant_frequency=basic_features.dominant_frequency,
            bandwidth=basic_features.bandwidth,
            spectral_centroid=basic_features.spectral_centroid,
            signal_duration=basic_features.signal_duration,
            rise_time=basic_features.rise_time,
            decay_time=basic_features.decay_time,
            phase_shift=basic_features.phase_shift,
            group_delay=basic_features.group_delay,
            depth_m=basic_features.depth_m,
            soil_type=basic_features.soil_type,
            moisture_content=basic_features.moisture_content,
            temperature_c=basic_features.temperature_c,

            # Advanced features
            spectral_rolloff=spectral_rolloff,
            spectral_flux=spectral_flux,
            zero_crossing_rate=zero_crossing_rate,
            mfcc_coefficients=mfcc_coefficients,
            kurtosis=kurtosis,
            skewness=skewness,
            envelope_area=envelope_area,
            peak_to_average_ratio=peak_to_average_ratio,
            wavelet_energy=wavelet_energy,
            fractal_dimension=fractal_dimension,
            hurst_exponent=hurst_exponent,
            signal_to_noise_ratio=snr,
            coherence=coherence,
            stability_index=stability_index
        )

        return enhanced_features

    def _extract_basic_features(self, signal: np.ndarray, sampling_rate: float) -> GPRSignatureFeatures:
        """Extract basic GPR features (fallback for when basic features not provided)."""

        # Simple peak amplitude
        peak_amplitude = np.max(np.abs(signal))

        # RMS amplitude
        rms_amplitude = np.sqrt(np.mean(signal**2))

        # Amplitude variance
        amplitude_variance = np.var(np.abs(signal))

        # Simple frequency domain analysis
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        magnitude = np.abs(fft)

        # Dominant frequency
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        dominant_frequency = freqs[dominant_freq_idx]

        # Bandwidth (rough estimate)
        bandwidth = sampling_rate / 4  # Simplified

        # Spectral centroid
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])

        return GPRSignatureFeatures(
            peak_amplitude=peak_amplitude,
            rms_amplitude=rms_amplitude,
            amplitude_variance=amplitude_variance,
            dominant_frequency=abs(dominant_frequency),
            bandwidth=bandwidth,
            spectral_centroid=abs(spectral_centroid),
            signal_duration=len(signal) / sampling_rate,
            rise_time=0.1,  # Simplified
            decay_time=0.1,  # Simplified
            phase_shift=0.0,  # Simplified
            group_delay=0.0,  # Simplified
            depth_m=1.0,  # Default
            soil_type="mixed",  # Default
            moisture_content=0.3,  # Default
            temperature_c=10.0  # Default
        )

    def _calculate_spectral_rolloff(self, signal: np.ndarray, sampling_rate: float, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)[:len(fft)//2]

        total_energy = np.sum(magnitude**2)
        target_energy = rolloff_percent * total_energy

        cumulative_energy = np.cumsum(magnitude**2)
        rolloff_idx = np.where(cumulative_energy >= target_energy)[0]

        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        return freqs[-1]

    def _calculate_spectral_flux(self, signal: np.ndarray, sampling_rate: float) -> float:
        """Calculate spectral flux (rate of spectral change)."""
        if len(signal) < 256:
            return 0.0

        window_size = min(256, len(signal)//4)
        step_size = window_size // 2

        flux_values = []
        for i in range(0, len(signal) - window_size, step_size):
            window1 = signal[i:i+window_size]
            window2 = signal[i+step_size:i+step_size+window_size]

            if len(window2) == window_size:
                fft1 = np.abs(np.fft.fft(window1))
                fft2 = np.abs(np.fft.fft(window2))

                diff = fft2 - fft1
                flux = np.sum(diff[diff > 0])
                flux_values.append(flux)

        return np.mean(flux_values) if flux_values else 0.0

    def _calculate_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        zero_crossings = np.sum(np.diff(np.signbit(signal)))
        return zero_crossings / len(signal)

    def _calculate_mfcc(self, signal: np.ndarray, sampling_rate: float, n_mfcc: int = 12) -> List[float]:
        """Calculate Mel-frequency cepstral coefficients (simplified implementation)."""
        # Simplified MFCC calculation
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft[:len(fft)//2])

        # Log magnitude spectrum
        log_magnitude = np.log(magnitude + 1e-10)

        # DCT to get cepstral coefficients (simplified)
        n_coeffs = min(n_mfcc, len(log_magnitude))
        mfcc = []

        for k in range(n_coeffs):
            coeff = 0
            for n in range(len(log_magnitude)):
                coeff += log_magnitude[n] * np.cos(np.pi * k * (2*n + 1) / (2 * len(log_magnitude)))
            mfcc.append(coeff)

        return mfcc

    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Calculate kurtosis (measure of signal peakedness)."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        normalized = (signal - mean) / std
        return np.mean(normalized**4) - 3

    def _calculate_skewness(self, signal: np.ndarray) -> float:
        """Calculate skewness (measure of signal asymmetry)."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        normalized = (signal - mean) / std
        return np.mean(normalized**3)

    def _calculate_envelope_area(self, signal: np.ndarray) -> float:
        """Calculate area under signal envelope."""
        envelope = np.abs(signal)
        return np.trapz(envelope)

    def _calculate_par(self, signal: np.ndarray) -> float:
        """Calculate Peak-to-Average Ratio."""
        peak = np.max(np.abs(signal))
        average = np.mean(np.abs(signal))
        return peak / average if average > 0 else 0.0

    def _calculate_wavelet_energy(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate energy in different frequency bands using simplified wavelet analysis."""
        # Simplified wavelet energy calculation using frequency bands
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft)**2

        n = len(magnitude) // 2
        band_size = n // 4

        bands = {
            'low': np.sum(magnitude[:band_size]),
            'mid_low': np.sum(magnitude[band_size:2*band_size]),
            'mid_high': np.sum(magnitude[2*band_size:3*band_size]),
            'high': np.sum(magnitude[3*band_size:n])
        }

        total_energy = sum(bands.values())
        if total_energy > 0:
            bands = {k: v/total_energy for k, v in bands.items()}

        return bands

    def _calculate_fractal_dimension(self, signal: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method (simplified)."""
        # Simplified fractal dimension calculation
        signal_normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)

        scales = np.logspace(0, np.log10(len(signal)//4), 10, dtype=int)
        counts = []

        for scale in scales:
            if scale < len(signal):
                n_boxes = len(signal) // scale
                boxes = np.array_split(signal_normalized[:n_boxes*scale], n_boxes)
                count = sum(1 for box in boxes if np.max(box) - np.min(box) > 0.1)
                counts.append(count)

        if len(counts) > 1 and len(scales) > 1:
            # Linear regression on log-log plot
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(np.array(counts) + 1)

            if len(log_scales) > 1:
                slope = np.polyfit(log_scales, log_counts, 1)[0]
                return -slope

        return 1.5  # Default fractal dimension

    def _calculate_hurst_exponent(self, signal: np.ndarray) -> float:
        """Calculate Hurst exponent (measure of long-range dependence)."""
        if len(signal) < 10:
            return 0.5

        # R/S analysis (simplified)
        n = len(signal)
        mean_signal = np.mean(signal)

        # Cumulative deviations
        cumulative_deviations = np.cumsum(signal - mean_signal)

        # Range
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

        # Standard deviation
        S = np.std(signal)

        if S == 0:
            return 0.5

        # R/S ratio
        rs_ratio = R / S

        # Hurst exponent approximation
        if rs_ratio > 0:
            hurst = np.log(rs_ratio) / np.log(n)
            return np.clip(hurst, 0.0, 1.0)

        return 0.5

    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        # Simple SNR estimation
        signal_power = np.mean(signal**2)

        # Estimate noise from high-frequency components
        fft = np.fft.fft(signal)
        high_freq_start = len(fft) * 3 // 4
        noise_estimate = np.mean(np.abs(fft[high_freq_start:])**2)

        if noise_estimate > 0:
            snr_linear = signal_power / noise_estimate
            return 10 * np.log10(snr_linear)

        return 20.0  # Default high SNR

    def _calculate_coherence(self, signal: np.ndarray) -> float:
        """Calculate signal coherence measure."""
        # Simplified coherence calculation
        if len(signal) < 64:
            return 1.0

        # Split signal into segments
        segment_size = len(signal) // 4
        segments = [signal[i:i+segment_size] for i in range(0, len(signal)-segment_size, segment_size)]

        if len(segments) < 2:
            return 1.0

        # Calculate correlation between segments
        correlations = []
        for i in range(len(segments)-1):
            corr = np.corrcoef(segments[i], segments[i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 1.0

    def _calculate_stability_index(self, signal: np.ndarray) -> float:
        """Calculate temporal stability index."""
        if len(signal) < 10:
            return 1.0

        # Calculate variance across signal segments
        segment_size = len(signal) // 10
        segment_vars = []

        for i in range(0, len(signal)-segment_size, segment_size):
            segment = signal[i:i+segment_size]
            segment_vars.append(np.var(segment))

        if len(segment_vars) > 1:
            stability = 1.0 / (1.0 + np.var(segment_vars))
            return stability

        return 1.0


class MaterialClassificationModel:
    """Advanced machine learning model for material classification from GPR signatures."""

    def __init__(self, model_type: str = "random_forest"):
        """Initialize the material classification model.

        Args:
            model_type: Type of model to use ("random_forest", "svm", "gradient_boosting")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False

        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.model)
        ])

        logger.info(f"Initialized {model_type} material classification model")

    def extract_features_from_signature(self, signature: GPRSignatureFeatures) -> np.ndarray:
        """Extract comprehensive feature vector from advanced GPR signature."""

        features = [
            # Basic amplitude features
            signature.peak_amplitude,
            signature.rms_amplitude,
            signature.amplitude_variance,

            # Basic frequency features
            signature.dominant_frequency,
            signature.bandwidth,
            signature.spectral_centroid,

            # Basic time domain features
            signature.signal_duration,
            signature.rise_time,
            signature.decay_time,

            # Basic phase features
            signature.phase_shift,
            signature.group_delay,

            # Environmental context
            signature.depth_m,
            signature.moisture_content,
            signature.temperature_c,

            # Basic derived features
            signature.peak_amplitude / max(signature.rms_amplitude, 0.001),  # Peak-to-RMS ratio
            signature.bandwidth / max(signature.dominant_frequency, 0.001),  # Relative bandwidth
            signature.rise_time / max(signature.signal_duration, 0.001),     # Rise time ratio
        ]

        # Add advanced features if available
        if signature.spectral_rolloff is not None:
            features.append(signature.spectral_rolloff)
        else:
            features.append(0.0)

        if signature.spectral_flux is not None:
            features.append(signature.spectral_flux)
        else:
            features.append(0.0)

        if signature.zero_crossing_rate is not None:
            features.append(signature.zero_crossing_rate)
        else:
            features.append(0.0)

        if signature.kurtosis is not None:
            features.append(signature.kurtosis)
        else:
            features.append(0.0)

        if signature.skewness is not None:
            features.append(signature.skewness)
        else:
            features.append(0.0)

        if signature.envelope_area is not None:
            features.append(signature.envelope_area)
        else:
            features.append(0.0)

        if signature.peak_to_average_ratio is not None:
            features.append(signature.peak_to_average_ratio)
        else:
            features.append(0.0)

        if signature.fractal_dimension is not None:
            features.append(signature.fractal_dimension)
        else:
            features.append(1.5)

        if signature.hurst_exponent is not None:
            features.append(signature.hurst_exponent)
        else:
            features.append(0.5)

        if signature.signal_to_noise_ratio is not None:
            features.append(signature.signal_to_noise_ratio)
        else:
            features.append(10.0)

        if signature.coherence is not None:
            features.append(signature.coherence)
        else:
            features.append(1.0)

        if signature.stability_index is not None:
            features.append(signature.stability_index)
        else:
            features.append(1.0)

        # Add MFCC coefficients if available
        if signature.mfcc_coefficients is not None:
            # Use first 8 MFCC coefficients
            mfcc_features = signature.mfcc_coefficients[:8]
            while len(mfcc_features) < 8:
                mfcc_features.append(0.0)
            features.extend(mfcc_features)
        else:
            features.extend([0.0] * 8)

        # Add wavelet energy features if available
        if signature.wavelet_energy is not None:
            features.extend([
                signature.wavelet_energy.get('low', 0.0),
                signature.wavelet_energy.get('mid_low', 0.0),
                signature.wavelet_energy.get('mid_high', 0.0),
                signature.wavelet_energy.get('high', 0.0)
            ])
        else:
            features.extend([0.25, 0.25, 0.25, 0.25])

        return np.array(features)

    def prepare_training_data(self, signatures: List[GPRSignatureFeatures],
                            materials: List[MaterialType]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from signatures and material labels."""

        # Extract features
        X = np.array([self.extract_features_from_signature(sig) for sig in signatures])

        # Encode labels
        y = self.label_encoder.fit_transform([mat.value for mat in materials])

        # Store feature names for interpretability
        self.feature_names = [
            'peak_amplitude', 'rms_amplitude', 'amplitude_variance',
            'dominant_frequency', 'bandwidth', 'spectral_centroid',
            'signal_duration', 'rise_time', 'decay_time',
            'phase_shift', 'group_delay',
            'depth_m', 'moisture_content', 'temperature_c',
            'peak_rms_ratio', 'relative_bandwidth', 'rise_time_ratio'
        ]

        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train(self, signatures: List[GPRSignatureFeatures],
              materials: List[MaterialType]) -> Dict[str, Any]:
        """Train the material classification model."""

        # Prepare data
        X, y = self.prepare_training_data(signatures, materials)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        logger.info("Training material classification model...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate model
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self.pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        )

        # Predictions for detailed evaluation
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)

        # Calculate metrics
        classification_rep = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        conf_matrix = confusion_matrix(y_test, y_pred)

        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))

        self.is_trained = True

        results = {
            'training_score': train_score,
            'test_score': test_score,
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance,
            'n_samples': len(signatures),
            'n_features': X.shape[1]
        }

        logger.info(f"Model training completed. CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return results

    def predict(self, signature: GPRSignatureFeatures) -> Tuple[MaterialType, float]:
        """Predict material type from GPR signature."""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        features = self.extract_features_from_signature(signature).reshape(1, -1)

        # Make prediction
        prediction_encoded = self.pipeline.predict(features)[0]
        probabilities = self.pipeline.predict_proba(features)[0]

        # Decode prediction
        material_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
        material_type = MaterialType(material_name)
        confidence = probabilities.max()

        return material_type, confidence

    def predict_with_probabilities(self, signature: GPRSignatureFeatures) -> Dict[MaterialType, float]:
        """Predict material type with probabilities for all classes."""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        features = self.extract_features_from_signature(signature).reshape(1, -1)

        # Get probabilities
        probabilities = self.pipeline.predict_proba(features)[0]

        # Map to material types
        result = {}
        for i, prob in enumerate(probabilities):
            material_name = self.label_encoder.classes_[i]
            material_type = MaterialType(material_name)
            result[material_type] = prob

        return result

    def save_model(self, filepath: Path):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)

        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")


class MaterialClassificationService:
    """Comprehensive service for material classification and analysis."""

    def __init__(self):
        """Initialize the enhanced material classification service."""
        self.material_db = MaterialPropertyDatabase()
        self.diameter_classifier = AdvancedDiameterClassifier()
        self.signature_analyzer = AdvancedGPRSignatureAnalyzer()
        self.discipline_predictor = DisciplineSpecificMaterialPredictor()
        self.environmental_integrator = EnvironmentalFactorIntegrator()

        # Enhanced ensemble of models
        self.rf_model = MaterialClassificationModel("random_forest")
        self.svm_model = MaterialClassificationModel("svm")
        self.gb_model = MaterialClassificationModel("gradient_boosting")

        # Additional models for improved ensemble
        self.xgb_model = None  # Will be initialized when needed
        self.neural_net_model = None  # Will be initialized when needed

        # Dynamic ensemble weights (updated based on performance)
        self.ensemble_weights = {
            "random_forest": 0.35,
            "svm": 0.25,
            "gradient_boosting": 0.25,
            "xgboost": 0.10,
            "neural_network": 0.05
        }

        # Model performance tracking
        self.model_performance_history = {}
        self.ensemble_confidence_threshold = 0.7

        logger.info("Enhanced material classification service initialized")

    def update_ensemble_weights(self, performance_metrics: Dict[str, float]):
        """Dynamically update ensemble weights based on recent performance."""

        total_performance = sum(performance_metrics.values())
        if total_performance > 0:
            # Normalize weights based on performance
            for model_name, performance in performance_metrics.items():
                if model_name in self.ensemble_weights:
                    self.ensemble_weights[model_name] = performance / total_performance

            logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

    def get_model_confidence_scores(self, signature: GPRSignatureFeatures) -> Dict[str, float]:
        """Get confidence scores from individual models."""

        confidence_scores = {}

        # Get predictions from all trained models
        models = {
            "random_forest": self.rf_model,
            "svm": self.svm_model,
            "gradient_boosting": self.gb_model
        }

        for model_name, model in models.items():
            if model.is_trained:
                try:
                    _, confidence = model.predict(signature)
                    confidence_scores[model_name] = confidence
                except Exception as e:
                    logger.warning(f"Error getting confidence from {model_name}: {e}")
                    confidence_scores[model_name] = 0.0

        return confidence_scores

    def predict_material_with_uncertainty(self, signature: GPRSignatureFeatures) -> Dict[str, Any]:
        """Predict material with uncertainty quantification."""

        # Get predictions from all models
        model_predictions = {}
        confidence_scores = self.get_model_confidence_scores(signature)

        # Ensemble prediction
        ensemble_material, ensemble_confidence = self.predict_material_ensemble(signature)

        # Calculate prediction uncertainty
        prediction_variance = np.var(list(confidence_scores.values())) if confidence_scores else 0.0

        # Determine prediction reliability
        reliability = "high" if ensemble_confidence > 0.8 else "medium" if ensemble_confidence > 0.6 else "low"

        # Get material properties for context
        material_props = self.material_db.get_material_properties(ensemble_material)

        return {
            'predicted_material': ensemble_material.value,
            'ensemble_confidence': ensemble_confidence,
            'individual_confidences': confidence_scores,
            'prediction_variance': prediction_variance,
            'reliability': reliability,
            'material_properties': {
                'detection_ease': material_props.detection_ease_score,
                'conductivity': material_props.electrical_conductivity_s_m,
                'typical_diameter_range': material_props.minimum_detectable_diameter_mm
            },
            'recommendation': self._get_prediction_recommendation(ensemble_confidence, prediction_variance)
        }

    def _get_prediction_recommendation(self, confidence: float, variance: float) -> str:
        """Generate recommendation based on prediction confidence and variance."""

        if confidence > 0.8 and variance < 0.1:
            return "High confidence prediction - proceed with identification"
        elif confidence > 0.6 and variance < 0.2:
            return "Medium confidence prediction - consider additional validation"
        elif variance > 0.3:
            return "High model disagreement - collect additional data or use alternative detection methods"
        else:
            return "Low confidence prediction - manual verification strongly recommended"

    def calibrate_ensemble_performance(self, validation_signatures: List[GPRSignatureFeatures],
                                     true_materials: List[MaterialType]) -> Dict[str, float]:
        """Calibrate ensemble performance and update weights."""

        if not validation_signatures or not true_materials:
            logger.warning("No validation data provided for calibration")
            return {}

        model_accuracies = {}

        # Evaluate each model
        models = {
            "random_forest": self.rf_model,
            "svm": self.svm_model,
            "gradient_boosting": self.gb_model
        }

        for model_name, model in models.items():
            if model.is_trained:
                correct_predictions = 0
                total_predictions = len(validation_signatures)

                for signature, true_material in zip(validation_signatures, true_materials):
                    try:
                        predicted_material, _ = model.predict(signature)
                        if predicted_material == true_material:
                            correct_predictions += 1
                    except Exception as e:
                        logger.warning(f"Error in {model_name} prediction: {e}")

                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                model_accuracies[model_name] = accuracy

        # Update ensemble weights based on performance
        if model_accuracies:
            self.update_ensemble_weights(model_accuracies)

        return model_accuracies

    def train_models(self, signatures: List[GPRSignatureFeatures],
                    materials: List[MaterialType]) -> Dict[str, Any]:
        """Train all classification models."""

        results = {}

        # Train individual models
        logger.info("Training Random Forest model...")
        results['random_forest'] = self.rf_model.train(signatures, materials)

        logger.info("Training SVM model...")
        results['svm'] = self.svm_model.train(signatures, materials)

        logger.info("Training Gradient Boosting model...")
        results['gradient_boosting'] = self.gb_model.train(signatures, materials)

        # Calculate ensemble performance
        ensemble_predictions = []
        actual_materials = []

        for signature, actual_material in zip(signatures, materials):
            predicted_material, _ = self.predict_material_ensemble(signature)
            ensemble_predictions.append(predicted_material)
            actual_materials.append(actual_material)

        # Calculate ensemble accuracy
        ensemble_accuracy = sum(
            pred == actual for pred, actual in zip(ensemble_predictions, actual_materials)
        ) / len(actual_materials)

        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'weights': self.ensemble_weights
        }

        logger.info(f"All models trained successfully. Ensemble accuracy: {ensemble_accuracy:.3f}")
        return results

    def predict_material_ensemble(self, signature: GPRSignatureFeatures) -> Tuple[MaterialType, float]:
        """Predict material using ensemble of models."""

        # Get predictions from all models
        models = {
            "random_forest": self.rf_model,
            "svm": self.svm_model,
            "gradient_boosting": self.gb_model
        }

        # Aggregate probabilities
        material_probs = {}
        total_weight = 0

        for model_name, model in models.items():
            if model.is_trained:
                weight = self.ensemble_weights[model_name]
                probs = model.predict_with_probabilities(signature)

                for material, prob in probs.items():
                    if material not in material_probs:
                        material_probs[material] = 0
                    material_probs[material] += prob * weight

                total_weight += weight

        # Normalize probabilities
        if total_weight > 0:
            for material in material_probs:
                material_probs[material] /= total_weight

        # Get best prediction
        best_material = max(material_probs.items(), key=lambda x: x[1])
        return best_material[0], best_material[1]

    def analyze_material_detectability(self, material_type: MaterialType,
                                     diameter_mm: float, depth_m: float,
                                     environmental_factors: Dict[str, float]) -> Dict[str, Any]:
        """Comprehensive analysis of material detectability."""

        material_props = self.material_db.get_material_properties(material_type)
        diameter_class = self.diameter_classifier.classify_diameter(diameter_mm)

        # Base detectability from material properties
        base_detectability = material_props.detection_ease_score

        # Diameter effects
        diameter_factor = 1.0
        if diameter_mm < material_props.minimum_detectable_diameter_mm:
            diameter_factor = diameter_mm / material_props.minimum_detectable_diameter_mm
        elif diameter_mm > 100:  # Larger utilities easier to detect
            diameter_factor = min(1.5, 1.0 + np.log10(diameter_mm / 100) * 0.2)

        # Depth effects (signal attenuates with depth)
        depth_factor = np.exp(-depth_m * material_props.signal_attenuation_db_m / 10)

        # Environmental factors
        moisture_effect = 1.0
        if 'soil_moisture' in environmental_factors:
            moisture = environmental_factors['soil_moisture']
            if material_props.moisture_sensitivity == "high":
                moisture_effect = 1.0 - moisture * 0.3
            elif material_props.moisture_sensitivity == "medium":
                moisture_effect = 1.0 - moisture * 0.15

        temperature_effect = 1.0
        if 'temperature' in environmental_factors:
            temp = environmental_factors['temperature']
            if material_props.temperature_sensitivity == "high":
                # Assume 20C is optimal
                temperature_effect = 1.0 - abs(temp - 20) * 0.01

        # Aging effects
        age_effect = 1.0
        if 'age_years' in environmental_factors:
            age = environmental_factors['age_years']
            age_effect = 1.0 - (age / material_props.typical_lifespan_years) * material_props.aging_detection_degradation

        # Combined detectability
        overall_detectability = (
            base_detectability *
            diameter_factor *
            depth_factor *
            moisture_effect *
            temperature_effect *
            max(age_effect, 0.1)  # Minimum 10% detectability
        )

        return {
            'material_type': material_type.value,
            'diameter_class': diameter_class.value,
            'overall_detectability': min(overall_detectability, 1.0),
            'factors': {
                'base_detectability': base_detectability,
                'diameter_factor': diameter_factor,
                'depth_factor': depth_factor,
                'moisture_effect': moisture_effect,
                'temperature_effect': temperature_effect,
                'age_effect': age_effect
            },
            'recommendations': self._get_detection_recommendations(
                material_type, overall_detectability, diameter_mm, depth_m
            )
        }

    def _get_detection_recommendations(self, material_type: MaterialType,
                                     detectability: float, diameter_mm: float,
                                     depth_m: float) -> List[str]:
        """Generate detection recommendations based on analysis."""

        recommendations = []
        material_props = self.material_db.get_material_properties(material_type)

        if detectability < 0.3:
            recommendations.append("Very difficult detection - consider multiple frequency sweeps")
            recommendations.append("Use lower frequencies for better penetration")

        if diameter_mm < material_props.minimum_detectable_diameter_mm:
            recommendations.append(f"Diameter below reliable detection threshold ({material_props.minimum_detectable_diameter_mm}mm)")
            recommendations.append("Consider using higher frequency antennas")

        if depth_m > 2.0:
            recommendations.append("Deep utility - signal attenuation may be significant")
            recommendations.append("Use lower frequency for better depth penetration")

        if material_type in [MaterialType.PVC, MaterialType.POLYETHYLENE, MaterialType.HDPE]:
            recommendations.append("Low-conductivity material - look for void detection rather than material reflection")
            recommendations.append("Multiple pass surveys recommended for confirmation")

        if material_type == MaterialType.STEEL:
            recommendations.append("High-conductivity material - strong reflections expected")
            recommendations.append("Watch for signal masking of utilities below")

        # Frequency recommendations
        opt_freq = material_props.optimal_frequency_range
        recommendations.append(f"Optimal frequency range: {opt_freq[0]}-{opt_freq[1]} MHz")

        return recommendations

    def get_discipline_material_correlation(self, discipline: UtilityDiscipline) -> Dict[str, Any]:
        """Analyze material distribution and correlations for a utility discipline."""

        typical_materials = self.material_db.get_materials_by_discipline(discipline)

        # Calculate material probabilities based on historical data and detectability
        material_analysis = {}

        for material in typical_materials:
            props = self.material_db.get_material_properties(material)

            # Historical probability (simplified model - in reality would be data-driven)
            historical_prob = self._estimate_historical_probability(discipline, material)

            material_analysis[material.value] = {
                'historical_probability': historical_prob,
                'detection_ease': props.detection_ease_score,
                'typical_diameter_range': self._get_typical_diameter_range(discipline, material),
                'aging_considerations': {
                    'typical_lifespan': props.typical_lifespan_years,
                    'detection_degradation': props.aging_detection_degradation
                },
                'environmental_sensitivity': {
                    'corrosion_resistance': props.corrosion_resistance,
                    'moisture_sensitivity': props.moisture_sensitivity,
                    'temperature_sensitivity': props.temperature_sensitivity
                }
            }

        return {
            'discipline': discipline.value,
            'material_analysis': material_analysis,
            'detection_strategy': self._get_discipline_detection_strategy(discipline),
            'common_challenges': self._get_discipline_challenges(discipline)
        }

    def _estimate_historical_probability(self, discipline: UtilityDiscipline,
                                       material: MaterialType) -> float:
        """Estimate historical probability of material usage by discipline."""

        # Simplified probability model based on typical usage patterns
        # In production, this would be based on real historical data

        prob_matrix = {
            UtilityDiscipline.ELECTRICITY: {
                MaterialType.STEEL: 0.3,
                MaterialType.PILC: 0.2,
                MaterialType.PVC: 0.3,
                MaterialType.HDPE: 0.2
            },
            UtilityDiscipline.WATER: {
                MaterialType.CAST_IRON: 0.35,
                MaterialType.STEEL: 0.15,
                MaterialType.PVC: 0.25,
                MaterialType.HDPE: 0.15,
                MaterialType.ASBESTOS_CEMENT: 0.1
            },
            UtilityDiscipline.SEWER: {
                MaterialType.CONCRETE: 0.3,
                MaterialType.CAST_IRON: 0.2,
                MaterialType.PVC: 0.25,
                MaterialType.FIBER_REINFORCED_PLASTIC: 0.15,
                MaterialType.ASBESTOS_CEMENT: 0.1
            },
            UtilityDiscipline.TELECOMMUNICATIONS: {
                MaterialType.PVC: 0.4,
                MaterialType.HDPE: 0.35,
                MaterialType.STEEL: 0.15,
                MaterialType.PILC: 0.1
            },
            UtilityDiscipline.OIL_GAS_CHEMICALS: {
                MaterialType.STEEL: 0.4,
                MaterialType.HDPE: 0.3,
                MaterialType.CAST_IRON: 0.2,
                MaterialType.FIBER_REINFORCED_PLASTIC: 0.1
            }
        }

        return prob_matrix.get(discipline, {}).get(material, 0.05)

    def _get_typical_diameter_range(self, discipline: UtilityDiscipline,
                                  material: MaterialType) -> Tuple[float, float]:
        """Get typical diameter range for discipline-material combination."""

        # Simplified ranges based on typical applications
        ranges = {
            (UtilityDiscipline.ELECTRICITY, MaterialType.STEEL): (50, 200),
            (UtilityDiscipline.ELECTRICITY, MaterialType.PVC): (25, 150),
            (UtilityDiscipline.WATER, MaterialType.CAST_IRON): (100, 600),
            (UtilityDiscipline.WATER, MaterialType.PVC): (50, 400),
            (UtilityDiscipline.SEWER, MaterialType.CONCRETE): (300, 1500),
            (UtilityDiscipline.SEWER, MaterialType.PVC): (100, 500),
            # Add more combinations as needed
        }

        return ranges.get((discipline, material), (50, 300))

    def _get_discipline_detection_strategy(self, discipline: UtilityDiscipline) -> List[str]:
        """Get detection strategy recommendations for a discipline."""

        strategies = {
            UtilityDiscipline.ELECTRICITY: [
                "Start with high frequency (400-800 MHz) for cable detection",
                "Look for metallic reflections from conduits and cables",
                "Be aware of electromagnetic interference from active cables"
            ],
            UtilityDiscipline.WATER: [
                "Use medium frequency (200-600 MHz) for metal pipes",
                "Look for pipe wall reflections and internal reflections",
                "Consider water content effects on signal propagation"
            ],
            UtilityDiscipline.SEWER: [
                "Use lower frequencies (100-400 MHz) for large diameter detection",
                "Look for void signatures and pipe wall reflections",
                "Be aware of variable fill conditions affecting signals"
            ],
            UtilityDiscipline.TELECOMMUNICATIONS: [
                "High frequency (400-1000 MHz) for duct and cable detection",
                "Multiple parallel cables may create complex signatures",
                "Duct systems easier to detect than individual cables"
            ],
            UtilityDiscipline.OIL_GAS_CHEMICALS: [
                "Medium frequency (200-600 MHz) for metal pipes",
                "High safety priority - confirm all detections",
                "Pressure and safety considerations for excavation"
            ]
        }

        return strategies.get(discipline, ["Use appropriate frequency for expected materials"])

    def _get_discipline_challenges(self, discipline: UtilityDiscipline) -> List[str]:
        """Get common detection challenges for a discipline."""

        challenges = {
            UtilityDiscipline.ELECTRICITY: [
                "EMI from active electrical systems",
                "Multiple cables in close proximity",
                "Legacy PILC cables may have degraded insulation"
            ],
            UtilityDiscipline.WATER: [
                "Metallic pipes may mask other utilities below",
                "Corrosion affects signal characteristics",
                "Water content in soil affects propagation"
            ],
            UtilityDiscipline.SEWER: [
                "Large diameter pipes create complex signatures",
                "Variable content affects internal reflections",
                "Depth often greater than other utilities"
            ],
            UtilityDiscipline.TELECOMMUNICATIONS: [
                "Small diameter cables difficult to detect",
                "Modern fiber cables are low-signature",
                "Dense cable bundles create complex patterns"
            ],
            UtilityDiscipline.OIL_GAS_CHEMICALS: [
                "Safety criticality requires high confidence",
                "High pressure systems may have thick walls",
                "Chemical contamination affects soil properties"
            ]
        }

        return challenges.get(discipline, ["Standard detection challenges apply"])


class DisciplineSpecificMaterialPredictor:
    """Advanced predictor for material types based on utility discipline and context."""

    def __init__(self):
        """Initialize discipline-specific material prediction system."""
        self.material_db = MaterialPropertyDatabase()

        # Historical material usage patterns by discipline and era
        self.discipline_material_evolution = {
            UtilityDiscipline.ELECTRICITY: {
                'legacy': {  # Pre-1950
                    MaterialType.PILC: 0.7,
                    MaterialType.STEEL: 0.2,
                    MaterialType.UNKNOWN: 0.1
                },
                'vintage': {  # 1950-1980
                    MaterialType.STEEL: 0.4,
                    MaterialType.PILC: 0.3,
                    MaterialType.PVC: 0.2,
                    MaterialType.UNKNOWN: 0.1
                },
                'modern': {  # 1980-2010
                    MaterialType.PVC: 0.5,
                    MaterialType.STEEL: 0.3,
                    MaterialType.HDPE: 0.15,
                    MaterialType.UNKNOWN: 0.05
                },
                'contemporary': {  # Post-2010
                    MaterialType.HDPE: 0.4,
                    MaterialType.PVC: 0.35,
                    MaterialType.STEEL: 0.2,
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.05
                }
            },
            UtilityDiscipline.WATER: {
                'legacy': {
                    MaterialType.CAST_IRON: 0.6,
                    MaterialType.STEEL: 0.2,
                    MaterialType.ASBESTOS_CEMENT: 0.15,
                    MaterialType.UNKNOWN: 0.05
                },
                'vintage': {
                    MaterialType.CAST_IRON: 0.5,
                    MaterialType.ASBESTOS_CEMENT: 0.25,
                    MaterialType.STEEL: 0.15,
                    MaterialType.PVC: 0.1
                },
                'modern': {
                    MaterialType.PVC: 0.4,
                    MaterialType.CAST_IRON: 0.3,
                    MaterialType.HDPE: 0.2,
                    MaterialType.STEEL: 0.1
                },
                'contemporary': {
                    MaterialType.HDPE: 0.4,
                    MaterialType.PVC: 0.35,
                    MaterialType.STEEL: 0.15,
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.1
                }
            },
            UtilityDiscipline.SEWER: {
                'legacy': {
                    MaterialType.CAST_IRON: 0.4,
                    MaterialType.CONCRETE: 0.35,
                    MaterialType.ASBESTOS_CEMENT: 0.2,
                    MaterialType.UNKNOWN: 0.05
                },
                'vintage': {
                    MaterialType.CONCRETE: 0.5,
                    MaterialType.CAST_IRON: 0.3,
                    MaterialType.ASBESTOS_CEMENT: 0.15,
                    MaterialType.PVC: 0.05
                },
                'modern': {
                    MaterialType.CONCRETE: 0.4,
                    MaterialType.PVC: 0.3,
                    MaterialType.CAST_IRON: 0.2,
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.1
                },
                'contemporary': {
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.35,
                    MaterialType.PVC: 0.3,
                    MaterialType.CONCRETE: 0.25,
                    MaterialType.HDPE: 0.1
                }
            },
            UtilityDiscipline.TELECOMMUNICATIONS: {
                'legacy': {
                    MaterialType.PILC: 0.8,
                    MaterialType.STEEL: 0.15,
                    MaterialType.UNKNOWN: 0.05
                },
                'vintage': {
                    MaterialType.PILC: 0.6,
                    MaterialType.STEEL: 0.25,
                    MaterialType.PVC: 0.15
                },
                'modern': {
                    MaterialType.PVC: 0.6,
                    MaterialType.HDPE: 0.25,
                    MaterialType.STEEL: 0.15
                },
                'contemporary': {
                    MaterialType.HDPE: 0.7,
                    MaterialType.PVC: 0.25,
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.05
                }
            },
            UtilityDiscipline.OIL_GAS_CHEMICALS: {
                'legacy': {
                    MaterialType.STEEL: 0.8,
                    MaterialType.CAST_IRON: 0.15,
                    MaterialType.UNKNOWN: 0.05
                },
                'vintage': {
                    MaterialType.STEEL: 0.7,
                    MaterialType.CAST_IRON: 0.2,
                    MaterialType.HDPE: 0.1
                },
                'modern': {
                    MaterialType.STEEL: 0.6,
                    MaterialType.HDPE: 0.25,
                    MaterialType.CAST_IRON: 0.1,
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.05
                },
                'contemporary': {
                    MaterialType.STEEL: 0.5,
                    MaterialType.HDPE: 0.3,
                    MaterialType.FIBER_REINFORCED_PLASTIC: 0.15,
                    MaterialType.PVC: 0.05
                }
            }
        }

        # Geographic material preferences (simplified for UK regions)
        self.regional_material_preferences = {
            'scotland': {
                MaterialType.CAST_IRON: 1.2,  # Higher prevalence
                MaterialType.STEEL: 1.1,
                MaterialType.PVC: 0.9
            },
            'northern_england': {
                MaterialType.CAST_IRON: 1.15,
                MaterialType.ASBESTOS_CEMENT: 1.1,
                MaterialType.PVC: 0.95
            },
            'london': {
                MaterialType.PILC: 1.3,  # Legacy electrical
                MaterialType.STEEL: 1.1,
                MaterialType.HDPE: 1.05
            },
            'southern_england': {
                MaterialType.PVC: 1.1,
                MaterialType.HDPE: 1.05,
                MaterialType.FIBER_REINFORCED_PLASTIC: 1.1
            }
        }

        logger.info("Discipline-specific material predictor initialized")

    def predict_material_by_discipline(self,
                                     discipline: UtilityDiscipline,
                                     context: Dict[str, Any]) -> Dict[MaterialType, float]:
        """Predict material probabilities based on discipline and context."""

        # Determine era based on installation year or region
        era = self._determine_era(context)

        # Get base probabilities for discipline and era
        base_probs = self.discipline_material_evolution.get(discipline, {}).get(era, {})

        if not base_probs:
            # Fallback to general material distribution
            return self._get_general_material_distribution()

        # Apply regional adjustments
        adjusted_probs = self._apply_regional_adjustments(base_probs, context)

        # Apply diameter constraints
        diameter_adjusted_probs = self._apply_diameter_constraints(adjusted_probs, context)

        # Apply environmental constraints
        final_probs = self._apply_environmental_constraints(diameter_adjusted_probs, context)

        return final_probs

    def _determine_era(self, context: Dict[str, Any]) -> str:
        """Determine installation era from context."""

        installation_year = context.get('installation_year')
        if installation_year:
            if installation_year < 1950:
                return 'legacy'
            elif installation_year < 1980:
                return 'vintage'
            elif installation_year < 2010:
                return 'modern'
            else:
                return 'contemporary'

        # Fallback based on other indicators
        if context.get('area_development_era') == 'historic':
            return 'legacy'
        elif context.get('area_development_era') == 'post_war':
            return 'vintage'
        elif context.get('area_development_era') == 'modern':
            return 'modern'
        else:
            return 'contemporary'

    def _apply_regional_adjustments(self, base_probs: Dict[MaterialType, float],
                                   context: Dict[str, Any]) -> Dict[MaterialType, float]:
        """Apply regional material preference adjustments."""

        region = context.get('region', 'unknown')
        if region not in self.regional_material_preferences:
            return base_probs.copy()

        regional_factors = self.regional_material_preferences[region]
        adjusted_probs = {}

        for material, prob in base_probs.items():
            factor = regional_factors.get(material, 1.0)
            adjusted_probs[material] = prob * factor

        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            adjusted_probs = {mat: prob / total_prob for mat, prob in adjusted_probs.items()}

        return adjusted_probs

    def _apply_diameter_constraints(self, base_probs: Dict[MaterialType, float],
                                   context: Dict[str, Any]) -> Dict[MaterialType, float]:
        """Apply diameter-based material constraints."""

        diameter = context.get('diameter_mm')
        if not diameter:
            return base_probs.copy()

        diameter_classifier = AdvancedDiameterClassifier()
        adjusted_probs = {}

        for material, prob in base_probs.items():
            # Get probability that this material would have this diameter
            diameter_prob = diameter_classifier.get_material_diameter_probability(material, diameter)

            # Combine with base probability
            adjusted_probs[material] = prob * diameter_prob

        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            adjusted_probs = {mat: prob / total_prob for mat, prob in adjusted_probs.items()}

        return adjusted_probs

    def _apply_environmental_constraints(self, base_probs: Dict[MaterialType, float],
                                        context: Dict[str, Any]) -> Dict[MaterialType, float]:
        """Apply environmental and condition-based constraints."""

        adjusted_probs = base_probs.copy()

        # Soil conditions
        soil_type = context.get('soil_type', '').lower()
        if 'corrosive' in soil_type or 'acidic' in soil_type:
            # Reduce probability of corrosion-prone materials
            for material in [MaterialType.STEEL, MaterialType.CAST_IRON]:
                if material in adjusted_probs:
                    adjusted_probs[material] *= 0.7

        # High moisture environments
        moisture_level = context.get('moisture_level', 0.3)
        if moisture_level > 0.7:
            # Favor corrosion-resistant materials
            for material in [MaterialType.PVC, MaterialType.HDPE, MaterialType.FIBER_REINFORCED_PLASTIC]:
                if material in adjusted_probs:
                    adjusted_probs[material] *= 1.2

        # Chemical exposure
        if context.get('chemical_exposure', False):
            # Favor chemically resistant materials
            for material in [MaterialType.FIBER_REINFORCED_PLASTIC, MaterialType.HDPE]:
                if material in adjusted_probs:
                    adjusted_probs[material] *= 1.3

        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            adjusted_probs = {mat: prob / total_prob for mat, prob in adjusted_probs.items()}

        return adjusted_probs

    def _get_general_material_distribution(self) -> Dict[MaterialType, float]:
        """Get general material distribution as fallback."""
        return {
            MaterialType.PVC: 0.25,
            MaterialType.CAST_IRON: 0.20,
            MaterialType.STEEL: 0.15,
            MaterialType.HDPE: 0.15,
            MaterialType.CONCRETE: 0.10,
            MaterialType.ASBESTOS_CEMENT: 0.05,
            MaterialType.FIBER_REINFORCED_PLASTIC: 0.05,
            MaterialType.PILC: 0.03,
            MaterialType.POLYETHYLENE: 0.02
        }

    def get_discipline_specific_recommendations(self,
                                              discipline: UtilityDiscipline,
                                              predicted_materials: Dict[MaterialType, float],
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Get discipline-specific detection and handling recommendations."""

        top_materials = sorted(predicted_materials.items(), key=lambda x: x[1], reverse=True)[:3]

        recommendations = {
            'most_likely_materials': [(mat.value, prob) for mat, prob in top_materials],
            'discipline': discipline.value,
            'detection_strategy': [],
            'safety_considerations': [],
            'survey_parameters': {}
        }

        # Discipline-specific recommendations
        if discipline == UtilityDiscipline.ELECTRICITY:
            recommendations['detection_strategy'].extend([
                "Use EMI detection alongside GPR for active cables",
                "Look for linear cable patterns and junction boxes",
                "Be aware of electromagnetic interference"
            ])
            recommendations['safety_considerations'].extend([
                "Assume all electrical utilities are live",
                "Maintain safe distances from high voltage cables",
                "Use appropriate PPE and safety procedures"
            ])

        elif discipline == UtilityDiscipline.WATER:
            recommendations['detection_strategy'].extend([
                "Look for pipe connections and valve chambers",
                "Consider water content effects on GPR signals",
                "Use metallic pipe tracers if available"
            ])
            recommendations['safety_considerations'].extend([
                "Consider water pressure in excavation planning",
                "Check for potential contamination",
                "Coordinate with water utility providers"
            ])

        elif discipline == UtilityDiscipline.SEWER:
            recommendations['detection_strategy'].extend([
                "Look for manhole patterns and connections",
                "Consider larger diameter signatures",
                "Use multiple frequency antennas for depth range"
            ])
            recommendations['safety_considerations'].extend([
                "Significant health and safety hazards",
                "Require confined space entry procedures",
                "Consider gas monitoring requirements"
            ])

        elif discipline == UtilityDiscipline.OIL_GAS_CHEMICALS:
            recommendations['detection_strategy'].extend([
                "Use lowest frequency for maximum penetration",
                "Look for protective sleeve indicators",
                "Consider cathodic protection systems"
            ])
            recommendations['safety_considerations'].extend([
                "CRITICAL SAFETY HAZARD - Gas escape risk",
                "Mandatory coordination with gas utility",
                "Use gas detection equipment",
                "Emergency response procedures required"
            ])

        # Survey parameters based on most likely material
        most_likely_material = top_materials[0][0]
        material_props = self.material_db.get_material_properties(most_likely_material)

        recommendations['survey_parameters'] = {
            'optimal_frequency_range': list(material_props.optimal_frequency_range),
            'expected_detection_difficulty': material_props.detection_ease_score,
            'recommended_antenna_spacing': context.get('diameter_mm', 100) * 0.5,
            'multiple_passes_recommended': material_props.detection_ease_score < 0.6
        }

        return recommendations

    def analyze_discipline_context_correlation(self,
                                             disciplines: List[UtilityDiscipline],
                                             materials: List[MaterialType],
                                             contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between discipline, context, and materials."""

        analysis = {
            'discipline_material_accuracy': {},
            'context_factor_importance': {},
            'prediction_performance': {}
        }

        # Analyze prediction accuracy by discipline
        for discipline in set(disciplines):
            discipline_indices = [i for i, d in enumerate(disciplines) if d == discipline]
            discipline_materials = [materials[i] for i in discipline_indices]
            discipline_contexts = [contexts[i] for i in discipline_indices]

            correct_predictions = 0
            total_predictions = len(discipline_indices)

            for i, material in enumerate(discipline_materials):
                context = discipline_contexts[i]
                predicted_probs = self.predict_material_by_discipline(discipline, context)

                # Check if actual material is in top 3 predictions
                top_predictions = sorted(predicted_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                if material in [pred[0] for pred in top_predictions]:
                    correct_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            analysis['discipline_material_accuracy'][discipline.value] = {
                'accuracy': accuracy,
                'sample_size': total_predictions
            }

        return analysis


class EnvironmentalFactorIntegrator:
    """Advanced system for integrating environmental factors into material detection."""

    def __init__(self):
        """Initialize environmental factor integration system."""
        self.material_db = MaterialPropertyDatabase()

        # Environmental condition impact models
        self.environmental_impact_models = {
            'soil_type': {
                'clay': {
                    'signal_attenuation_factor': 1.3,
                    'dielectric_constant_shift': 2.0,
                    'preferred_frequency_factor': 0.8
                },
                'sand': {
                    'signal_attenuation_factor': 0.9,
                    'dielectric_constant_shift': -1.0,
                    'preferred_frequency_factor': 1.2
                },
                'gravel': {
                    'signal_attenuation_factor': 0.8,
                    'dielectric_constant_shift': -0.5,
                    'preferred_frequency_factor': 1.1
                },
                'peat': {
                    'signal_attenuation_factor': 1.8,
                    'dielectric_constant_shift': 4.0,
                    'preferred_frequency_factor': 0.6
                },
                'rock': {
                    'signal_attenuation_factor': 0.7,
                    'dielectric_constant_shift': -2.0,
                    'preferred_frequency_factor': 1.3
                }
            },
            'moisture_conditions': {
                'dry': {
                    'conductivity_enhancement': 0.8,
                    'signal_penetration_factor': 1.2,
                    'metallic_detection_factor': 1.1
                },
                'moderate': {
                    'conductivity_enhancement': 1.0,
                    'signal_penetration_factor': 1.0,
                    'metallic_detection_factor': 1.0
                },
                'wet': {
                    'conductivity_enhancement': 1.5,
                    'signal_penetration_factor': 0.7,
                    'metallic_detection_factor': 0.9
                },
                'saturated': {
                    'conductivity_enhancement': 2.0,
                    'signal_penetration_factor': 0.5,
                    'metallic_detection_factor': 0.8
                }
            },
            'temperature_conditions': {
                'frozen': {
                    'dielectric_constant_factor': 0.5,
                    'signal_velocity_factor': 1.4,
                    'resolution_enhancement': 1.3
                },
                'cold': {
                    'dielectric_constant_factor': 0.8,
                    'signal_velocity_factor': 1.1,
                    'resolution_enhancement': 1.1
                },
                'normal': {
                    'dielectric_constant_factor': 1.0,
                    'signal_velocity_factor': 1.0,
                    'resolution_enhancement': 1.0
                },
                'hot': {
                    'dielectric_constant_factor': 1.2,
                    'signal_velocity_factor': 0.9,
                    'resolution_enhancement': 0.9
                }
            }
        }

        # Urban environment interference models
        self.urban_interference_models = {
            'low_density_residential': {
                'electromagnetic_noise_level': 0.2,
                'structural_interference': 0.1,
                'utility_density_factor': 1.2
            },
            'high_density_residential': {
                'electromagnetic_noise_level': 0.4,
                'structural_interference': 0.3,
                'utility_density_factor': 1.8
            },
            'commercial': {
                'electromagnetic_noise_level': 0.6,
                'structural_interference': 0.5,
                'utility_density_factor': 2.2
            },
            'industrial': {
                'electromagnetic_noise_level': 0.8,
                'structural_interference': 0.7,
                'utility_density_factor': 2.8
            },
            'city_center': {
                'electromagnetic_noise_level': 0.9,
                'structural_interference': 0.8,
                'utility_density_factor': 3.5
            }
        }

        logger.info("Environmental factor integrator initialized")

    def assess_environmental_detection_conditions(self, environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive assessment of environmental conditions for material detection."""

        assessment = {
            'overall_detection_favorability': 0.0,
            'environmental_factors': {},
            'recommended_adjustments': {},
            'risk_factors': [],
            'optimization_suggestions': []
        }

        # Soil condition assessment
        soil_assessment = self._assess_soil_conditions(environmental_context)
        assessment['environmental_factors']['soil'] = soil_assessment

        # Moisture condition assessment
        moisture_assessment = self._assess_moisture_conditions(environmental_context)
        assessment['environmental_factors']['moisture'] = moisture_assessment

        # Temperature condition assessment
        temperature_assessment = self._assess_temperature_conditions(environmental_context)
        assessment['environmental_factors']['temperature'] = temperature_assessment

        # Urban environment assessment
        urban_assessment = self._assess_urban_environment(environmental_context)
        assessment['environmental_factors']['urban'] = urban_assessment

        # Calculate overall favorability
        favorability_scores = [
            soil_assessment.get('favorability_score', 0.5),
            moisture_assessment.get('favorability_score', 0.5),
            temperature_assessment.get('favorability_score', 0.5),
            urban_assessment.get('favorability_score', 0.5)
        ]
        assessment['overall_detection_favorability'] = np.mean(favorability_scores)

        # Generate recommendations
        assessment['recommended_adjustments'] = self._generate_environmental_adjustments(
            soil_assessment, moisture_assessment, temperature_assessment, urban_assessment
        )

        # Identify risk factors
        assessment['risk_factors'] = self._identify_environmental_risks(environmental_context)

        # Generate optimization suggestions
        assessment['optimization_suggestions'] = self._generate_optimization_suggestions(
            assessment['overall_detection_favorability'], environmental_context
        )

        return assessment

    def _assess_soil_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess soil condition impact on detection."""

        soil_type = context.get('soil_type', 'unknown').lower()
        soil_moisture = context.get('soil_moisture_content', 0.3)
        soil_compaction = context.get('soil_compaction', 'medium')

        assessment = {
            'soil_type': soil_type,
            'impact_factors': {},
            'favorability_score': 0.5
        }

        if soil_type in self.environmental_impact_models['soil_type']:
            soil_model = self.environmental_impact_models['soil_type'][soil_type]

            assessment['impact_factors'] = {
                'signal_attenuation': soil_model['signal_attenuation_factor'],
                'dielectric_shift': soil_model['dielectric_constant_shift'],
                'frequency_preference': soil_model['preferred_frequency_factor']
            }

            # Calculate favorability (lower attenuation = better)
            favorability = 1.0 / soil_model['signal_attenuation_factor']
            if soil_moisture < 0.3:  # Dry conditions help
                favorability *= 1.2
            elif soil_moisture > 0.7:  # Wet conditions hurt
                favorability *= 0.8

            assessment['favorability_score'] = min(favorability, 1.0)

        return assessment

    def _assess_moisture_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess moisture condition impact on detection."""

        moisture_level = context.get('soil_moisture_content', 0.3)
        weather_condition = context.get('weather_condition', 'normal')
        groundwater_level = context.get('groundwater_level_m', 10.0)

        # Categorize moisture level
        if moisture_level < 0.2:
            moisture_category = 'dry'
        elif moisture_level < 0.5:
            moisture_category = 'moderate'
        elif moisture_level < 0.8:
            moisture_category = 'wet'
        else:
            moisture_category = 'saturated'

        assessment = {
            'moisture_category': moisture_category,
            'moisture_level': moisture_level,
            'impact_factors': {},
            'favorability_score': 0.5
        }

        if moisture_category in self.environmental_impact_models['moisture_conditions']:
            moisture_model = self.environmental_impact_models['moisture_conditions'][moisture_category]

            assessment['impact_factors'] = {
                'conductivity_enhancement': moisture_model['conductivity_enhancement'],
                'signal_penetration': moisture_model['signal_penetration_factor'],
                'metallic_detection': moisture_model['metallic_detection_factor']
            }

            # Calculate favorability (moderate moisture is optimal)
            if moisture_category == 'moderate':
                favorability = 1.0
            elif moisture_category == 'dry':
                favorability = 0.9
            elif moisture_category == 'wet':
                favorability = 0.7
            else:  # saturated
                favorability = 0.5

            assessment['favorability_score'] = favorability

        return assessment

    def _assess_temperature_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess temperature condition impact on detection."""

        temperature_c = context.get('temperature_c', 10.0)
        seasonal_variation = context.get('seasonal_variation', False)

        # Categorize temperature
        if temperature_c < 0:
            temp_category = 'frozen'
        elif temperature_c < 10:
            temp_category = 'cold'
        elif temperature_c < 25:
            temp_category = 'normal'
        else:
            temp_category = 'hot'

        assessment = {
            'temperature_category': temp_category,
            'temperature_c': temperature_c,
            'impact_factors': {},
            'favorability_score': 0.5
        }

        if temp_category in self.environmental_impact_models['temperature_conditions']:
            temp_model = self.environmental_impact_models['temperature_conditions'][temp_category]

            assessment['impact_factors'] = {
                'dielectric_factor': temp_model['dielectric_constant_factor'],
                'signal_velocity': temp_model['signal_velocity_factor'],
                'resolution_enhancement': temp_model['resolution_enhancement']
            }

            # Calculate favorability (frozen is best, normal is good)
            if temp_category == 'frozen':
                favorability = 1.0
            elif temp_category == 'normal':
                favorability = 0.9
            elif temp_category == 'cold':
                favorability = 0.85
            else:  # hot
                favorability = 0.75

            assessment['favorability_score'] = favorability

        return assessment

    def _assess_urban_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess urban environment impact on detection."""

        area_type = context.get('area_type', 'low_density_residential')
        utility_density = context.get('utility_density', 'medium')
        electromagnetic_sources = context.get('electromagnetic_sources', [])

        assessment = {
            'area_type': area_type,
            'interference_factors': {},
            'favorability_score': 0.5
        }

        if area_type in self.urban_interference_models:
            urban_model = self.urban_interference_models[area_type]

            assessment['interference_factors'] = {
                'electromagnetic_noise': urban_model['electromagnetic_noise_level'],
                'structural_interference': urban_model['structural_interference'],
                'utility_density': urban_model['utility_density_factor']
            }

            # Calculate favorability (lower interference = better)
            base_favorability = 1.0 - urban_model['electromagnetic_noise_level']
            structure_penalty = urban_model['structural_interference'] * 0.5
            favorability = max(base_favorability - structure_penalty, 0.1)

            assessment['favorability_score'] = favorability

        return assessment

    def _generate_environmental_adjustments(self, soil_assessment: Dict, moisture_assessment: Dict,
                                          temperature_assessment: Dict, urban_assessment: Dict) -> Dict[str, Any]:
        """Generate recommended adjustments based on environmental conditions."""

        adjustments = {
            'frequency_adjustments': {},
            'survey_parameter_adjustments': {},
            'equipment_recommendations': []
        }

        # Frequency adjustments based on soil
        if 'impact_factors' in soil_assessment:
            freq_factor = soil_assessment['impact_factors'].get('frequency_preference', 1.0)
            if freq_factor < 0.9:
                adjustments['frequency_adjustments']['soil_adjustment'] = 'Use lower frequencies for better penetration'
            elif freq_factor > 1.1:
                adjustments['frequency_adjustments']['soil_adjustment'] = 'Can use higher frequencies for better resolution'

        # Survey parameter adjustments
        if moisture_assessment.get('moisture_category') == 'wet':
            adjustments['survey_parameter_adjustments']['scan_speed'] = 'Reduce scan speed for wet conditions'
            adjustments['survey_parameter_adjustments']['multiple_passes'] = 'Consider multiple survey passes'

        if urban_assessment.get('interference_factors', {}).get('electromagnetic_noise', 0) > 0.6:
            adjustments['equipment_recommendations'].append('Use shielded antennas to reduce EMI')
            adjustments['equipment_recommendations'].append('Consider time-gated measurements')

        return adjustments

    def _identify_environmental_risks(self, context: Dict[str, Any]) -> List[str]:
        """Identify environmental risk factors for detection."""

        risks = []

        moisture_level = context.get('soil_moisture_content', 0.3)
        if moisture_level > 0.8:
            risks.append("High soil moisture may significantly reduce signal penetration")

        temperature = context.get('temperature_c', 10.0)
        if temperature > 30:
            risks.append("High temperature may cause equipment thermal drift")

        area_type = context.get('area_type', '')
        if 'industrial' in area_type or 'city_center' in area_type:
            risks.append("High electromagnetic interference expected in urban/industrial areas")

        soil_type = context.get('soil_type', '')
        if 'clay' in soil_type or 'peat' in soil_type:
            risks.append("Soil type may cause significant signal attenuation")

        return risks

    def _generate_optimization_suggestions(self, overall_favorability: float,
                                         context: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on environmental assessment."""

        suggestions = []

        if overall_favorability < 0.4:
            suggestions.extend([
                "Consider alternative detection methods (electromagnetic, acoustic)",
                "Use multiple frequency antennas for comprehensive coverage",
                "Plan for extended survey time due to challenging conditions"
            ])
        elif overall_favorability < 0.7:
            suggestions.extend([
                "Use optimized frequency range for conditions",
                "Consider multiple survey passes for verification",
                "Monitor signal quality throughout survey"
            ])

        # Weather-specific suggestions
        weather = context.get('weather_condition', '')
        if 'rain' in weather:
            suggestions.append("Postpone survey if possible - wet conditions reduce effectiveness")
        elif 'snow' in weather:
            suggestions.append("Frozen ground conditions may actually improve signal penetration")

        return suggestions

    def adjust_material_detection_probability(self, base_probability: float,
                                            material_type: MaterialType,
                                            environmental_context: Dict[str, Any]) -> float:
        """Adjust material detection probability based on environmental factors."""

        assessment = self.assess_environmental_detection_conditions(environmental_context)
        material_props = self.material_db.get_material_properties(material_type)

        # Start with base probability
        adjusted_probability = base_probability

        # Apply overall environmental favorability
        environmental_factor = assessment['overall_detection_favorability']
        adjusted_probability *= environmental_factor

        # Material-specific environmental adjustments
        if material_props.moisture_sensitivity == "high":
            moisture_factor = assessment['environmental_factors']['moisture']['favorability_score']
            adjusted_probability *= moisture_factor

        if material_props.temperature_sensitivity == "high":
            temp_factor = assessment['environmental_factors']['temperature']['favorability_score']
            adjusted_probability *= temp_factor

        # Urban interference effects (affects all materials)
        urban_factor = assessment['environmental_factors']['urban']['favorability_score']
        adjusted_probability *= urban_factor

        return min(max(adjusted_probability, 0.05), 0.98)  # Clamp between 5% and 98%