"""
Test data generation utilities for GPR validation testing.

This module provides utilities to generate synthetic test data for various
validation scenarios when real data is not available.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
from datetime import datetime, timedelta

from tests.validation.accuracy import DetectedUtility, GroundTruthUtility
from tests.validation.environmental import EnvironmentalConditions, SurveyResults


class TestDataGenerator:
    """Generator for synthetic test data."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize test data generator.

        Args:
            random_seed: Random seed for reproducible data generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def generate_utility_layout(
        self,
        area_bounds: Tuple[float, float, float, float],  # (min_x, max_x, min_y, max_y)
        utility_count: int = 10,
        depth_range: Tuple[float, float] = (0.5, 3.0),
        materials: Optional[List[str]] = None,
        disciplines: Optional[List[str]] = None
    ) -> List[GroundTruthUtility]:
        """
        Generate a realistic utility layout.

        Args:
            area_bounds: Survey area boundaries (min_x, max_x, min_y, max_y)
            utility_count: Number of utilities to generate
            depth_range: Depth range in meters
            materials: List of possible materials
            disciplines: List of possible disciplines

        Returns:
            List of GroundTruthUtility objects
        """
        if materials is None:
            materials = ["steel", "polyVinylChloride", "asbestosCement", "highDensityPolyEthylene", "concrete"]

        if disciplines is None:
            disciplines = ["water", "sewer", "electricity", "telecommunications", "oilGasChemicals"]

        min_x, max_x, min_y, max_y = area_bounds
        utilities = []

        for i in range(utility_count):
            # Generate position with some clustering tendency
            if i > 0 and random.random() < 0.3:  # 30% chance of clustering
                # Place near existing utility
                existing = random.choice(utilities)
                x_pos = existing.x_position + np.random.normal(0, 2.0)
                y_pos = existing.y_position + np.random.normal(0, 2.0)
                # Clamp to bounds
                x_pos = max(min_x, min(max_x, x_pos))
                y_pos = max(min_y, min(max_y, y_pos))
            else:
                # Random position
                x_pos = np.random.uniform(min_x, max_x)
                y_pos = np.random.uniform(min_y, max_y)

            # Generate depth (shallower utilities are more common)
            depth = np.random.exponential(scale=1.0) + depth_range[0]
            depth = min(depth, depth_range[1])

            # Select material and discipline
            material = random.choice(materials)
            discipline = random.choice(disciplines)

            # Generate diameter based on discipline
            diameter = self._generate_realistic_diameter(discipline)

            utility = GroundTruthUtility(
                x_position=x_pos,
                y_position=y_pos,
                depth=depth,
                material=material,
                diameter=diameter,
                discipline=discipline,
                utility_id=f"utility_{i:03d}"
            )
            utilities.append(utility)

        return utilities

    def generate_gpr_detections(
        self,
        ground_truth: List[GroundTruthUtility],
        detection_rate: float = 0.85,
        position_noise_std: float = 0.3,
        depth_noise_std: float = 0.1,
        false_positive_rate: float = 0.1,
        material_accuracy: float = 0.8,
        discipline_accuracy: float = 0.9
    ) -> List[DetectedUtility]:
        """
        Generate realistic GPR detections based on ground truth.

        Args:
            ground_truth: List of ground truth utilities
            detection_rate: Probability of detecting each utility
            position_noise_std: Standard deviation of position noise (meters)
            depth_noise_std: Standard deviation of depth noise (meters)
            false_positive_rate: Rate of false positive detections
            material_accuracy: Accuracy of material classification
            discipline_accuracy: Accuracy of discipline classification

        Returns:
            List of DetectedUtility objects
        """
        detections = []
        detection_id = 0

        # Generate true positive detections
        for gt_utility in ground_truth:
            if np.random.random() < detection_rate:
                # Add position noise
                x_detected = gt_utility.x_position + np.random.normal(0, position_noise_std)
                y_detected = gt_utility.y_position + np.random.normal(0, position_noise_std)

                # Add depth noise
                depth_detected = gt_utility.depth + np.random.normal(0, depth_noise_std)
                depth_detected = max(0.1, depth_detected)  # Ensure positive depth

                # Material classification (with some errors)
                if np.random.random() < material_accuracy:
                    material_detected = gt_utility.material
                else:
                    # Random misclassification
                    materials = ["steel", "polyVinylChloride", "asbestosCement", "highDensityPolyEthylene"]
                    material_detected = random.choice([m for m in materials if m != gt_utility.material])

                # Discipline classification (with some errors)
                if np.random.random() < discipline_accuracy:
                    discipline_detected = gt_utility.discipline
                else:
                    # Random misclassification
                    disciplines = ["water", "sewer", "electricity", "telecommunications"]
                    discipline_detected = random.choice([d for d in disciplines if d != gt_utility.discipline])

                # Generate confidence score
                confidence = self._generate_confidence_score(gt_utility, position_noise_std, depth_noise_std)

                detection = DetectedUtility(
                    x_position=x_detected,
                    y_position=y_detected,
                    depth=depth_detected,
                    material=material_detected,
                    diameter=gt_utility.diameter,
                    discipline=discipline_detected,
                    confidence=confidence,
                    detection_id=f"detection_{detection_id:03d}"
                )
                detections.append(detection)
                detection_id += 1

        # Generate false positive detections
        num_false_positives = int(len(detections) * false_positive_rate)
        area_bounds = self._calculate_area_bounds(ground_truth)

        for i in range(num_false_positives):
            # Random position in survey area
            x_fp = np.random.uniform(area_bounds[0], area_bounds[1])
            y_fp = np.random.uniform(area_bounds[2], area_bounds[3])
            depth_fp = np.random.uniform(0.5, 2.5)

            # Random properties
            materials = ["steel", "polyVinylChloride", "unknown"]
            disciplines = ["unknown", "electricity", "telecommunications"]

            detection = DetectedUtility(
                x_position=x_fp,
                y_position=y_fp,
                depth=depth_fp,
                material=random.choice(materials),
                diameter=random.choice([100, 125, 200]),
                discipline=random.choice(disciplines),
                confidence=np.random.uniform(0.3, 0.7),  # Lower confidence for false positives
                detection_id=f"detection_{detection_id:03d}"
            )
            detections.append(detection)
            detection_id += 1

        return detections

    def generate_environmental_conditions(
        self,
        scenario: str = "mixed"
    ) -> EnvironmentalConditions:
        """
        Generate environmental conditions for testing.

        Args:
            scenario: Type of scenario ("dry_sandy", "wet_clayey", "mixed")

        Returns:
            EnvironmentalConditions object
        """
        if scenario == "dry_sandy":
            return EnvironmentalConditions(
                weather_condition="Dry",
                ground_condition="Sandy",
                ground_permittivity=np.random.uniform(8.0, 12.0),
                land_cover=random.choice(["Grass / vegetation", "Concrete surfacing"]),
                land_use="Residential - quiet suburbia",
                terrain_levelling="Flat",
                terrain_smoothness="Smooth",
                rubble_presence=False,
                tree_roots_presence=random.choice([True, False]),
                polluted_soil_presence=False,
                blast_furnace_slag_presence=False
            )
        elif scenario == "wet_clayey":
            return EnvironmentalConditions(
                weather_condition="Rainy",
                ground_condition="Clayey",
                ground_permittivity=np.random.uniform(15.0, 20.0),
                land_cover=random.choice(["Brick road concrete", "Asphalt"]),
                land_use="Commercial and industrial",
                terrain_levelling=random.choice(["Flat", "Steep"]),
                terrain_smoothness=random.choice(["Smooth", "Rough"]),
                rubble_presence=random.choice([True, False]),
                tree_roots_presence=False,
                polluted_soil_presence=random.choice([True, False]),
                blast_furnace_slag_presence=random.choice([True, False])
            )
        else:  # mixed
            return EnvironmentalConditions(
                weather_condition=random.choice(["Dry", "Rainy"]),
                ground_condition=random.choice(["Sandy", "Clayey"]),
                ground_permittivity=np.random.uniform(8.0, 20.0),
                land_cover=random.choice([
                    "Grass / vegetation", "Concrete surfacing", "Brick road concrete", "Asphalt"
                ]),
                land_use=random.choice([
                    "Residential - quiet suburbia",
                    "Commercial and industrial",
                    "High density residental - land-high rise, town and city centre with high population"
                ]),
                terrain_levelling=random.choice(["Flat", "Steep"]),
                terrain_smoothness=random.choice(["Smooth", "Rough"]),
                rubble_presence=random.choice([True, False]),
                tree_roots_presence=random.choice([True, False]),
                polluted_soil_presence=random.choice([True, False]),
                blast_furnace_slag_presence=random.choice([True, False])
            )

    def generate_survey_results(
        self,
        num_surveys: int = 20,
        area_bounds: Tuple[float, float, float, float] = (0, 100, 0, 100),
        utilities_per_survey_range: Tuple[int, int] = (5, 15)
    ) -> List[SurveyResults]:
        """
        Generate multiple survey results for environmental testing.

        Args:
            num_surveys: Number of surveys to generate
            area_bounds: Survey area boundaries
            utilities_per_survey_range: Range of utilities per survey

        Returns:
            List of SurveyResults objects
        """
        survey_results = []

        for i in range(num_surveys):
            # Generate number of utilities for this survey
            utility_count = np.random.randint(utilities_per_survey_range[0], utilities_per_survey_range[1] + 1)

            # Generate ground truth
            ground_truth = self.generate_utility_layout(area_bounds, utility_count)

            # Generate environmental conditions
            env_conditions = self.generate_environmental_conditions()

            # Adjust detection performance based on environmental conditions
            detection_rate = self._calculate_environmental_detection_rate(env_conditions)
            position_noise = self._calculate_environmental_position_noise(env_conditions)

            # Generate detections
            detections = self.generate_gpr_detections(
                ground_truth,
                detection_rate=detection_rate,
                position_noise_std=position_noise
            )

            survey_result = SurveyResults(
                location_id=f"survey_{i:03d}",
                environmental_conditions=env_conditions,
                detections=detections,
                ground_truth=ground_truth
            )
            survey_results.append(survey_result)

        return survey_results

    def generate_performance_time_series(
        self,
        num_points: int = 100,
        base_performance: float = 0.8,
        trend_slope: float = 0.0,
        noise_std: float = 0.05,
        seasonal_amplitude: float = 0.1,
        anomaly_probability: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate time series performance data for monitoring tests.

        Args:
            num_points: Number of data points
            base_performance: Base performance level
            trend_slope: Linear trend (positive = improving)
            noise_std: Standard deviation of noise
            seasonal_amplitude: Amplitude of seasonal variation
            anomaly_probability: Probability of anomalies

        Returns:
            DataFrame with timestamp and performance columns
        """
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=num_points),
            end=datetime.now(),
            periods=num_points
        )

        performance_values = []

        for i, timestamp in enumerate(timestamps):
            # Base trend
            trend_value = base_performance + (trend_slope * i / num_points)

            # Seasonal component (weekly cycle)
            seasonal_value = seasonal_amplitude * np.sin(2 * np.pi * i / 7)

            # Random noise
            noise_value = np.random.normal(0, noise_std)

            # Anomalies
            if np.random.random() < anomaly_probability:
                anomaly_value = np.random.normal(0, noise_std * 5)  # Larger anomaly
            else:
                anomaly_value = 0

            performance = trend_value + seasonal_value + noise_value + anomaly_value
            performance = max(0, min(1, performance))  # Clamp to [0, 1]

            performance_values.append(performance)

        return pd.DataFrame({
            'timestamp': timestamps,
            'performance': performance_values,
            'processing_time': np.random.lognormal(mean=0, sigma=0.3, size=num_points),
            'memory_usage': np.random.lognormal(mean=6, sigma=0.2, size=num_points),  # ~400MB average
            'cpu_utilization': np.random.beta(2, 5, size=num_points) * 100  # Skewed toward lower usage
        })

    def _generate_realistic_diameter(self, discipline: str) -> float:
        """Generate realistic diameter based on utility discipline."""
        diameter_ranges = {
            "water": [100, 125, 150, 200, 250, 300, 400],
            "sewer": [200, 300, 400, 500, 600, 800, 1000],
            "electricity": [16, 25, 40, 50, 75, 100],
            "telecommunications": [16, 20, 25, 32, 40, 50],
            "oilGasChemicals": [100, 150, 200, 300, 400, 500]
        }

        if discipline in diameter_ranges:
            return random.choice(diameter_ranges[discipline])
        else:
            return random.choice([100, 125, 150, 200])

    def _generate_confidence_score(
        self,
        gt_utility: GroundTruthUtility,
        position_noise: float,
        depth_noise: float
    ) -> float:
        """Generate realistic confidence score based on detection quality."""
        # Base confidence
        base_confidence = 0.8

        # Depth factor (deeper utilities are harder to detect)
        depth_factor = max(0.2, 1.0 - (gt_utility.depth - 0.5) * 0.2)

        # Material factor (some materials are easier to detect)
        material_factors = {
            "steel": 1.0,
            "concrete": 0.9,
            "polyVinylChloride": 0.7,
            "asbestosCement": 0.8,
            "highDensityPolyEthylene": 0.6
        }
        material_factor = material_factors.get(gt_utility.material, 0.7)

        # Noise factor (more noise = lower confidence)
        noise_factor = max(0.3, 1.0 - (position_noise + depth_noise))

        # Combine factors
        confidence = base_confidence * depth_factor * material_factor * noise_factor

        # Add some randomness
        confidence += np.random.normal(0, 0.05)

        return max(0.1, min(0.99, confidence))

    def _calculate_area_bounds(self, utilities: List[GroundTruthUtility]) -> Tuple[float, float, float, float]:
        """Calculate bounding box of utilities."""
        if not utilities:
            return (0, 100, 0, 100)

        x_positions = [u.x_position for u in utilities]
        y_positions = [u.y_position for u in utilities]

        return (
            min(x_positions) - 10,
            max(x_positions) + 10,
            min(y_positions) - 10,
            max(y_positions) + 10
        )

    def _calculate_environmental_detection_rate(self, env_conditions: EnvironmentalConditions) -> float:
        """Calculate detection rate based on environmental conditions."""
        base_rate = 0.85

        # Weather impact
        if env_conditions.weather_condition == "Rainy":
            base_rate *= 0.9

        # Ground condition impact
        if env_conditions.ground_condition == "Clayey":
            base_rate *= 0.8

        # Permittivity impact (higher permittivity = more attenuation)
        if env_conditions.ground_permittivity > 15:
            base_rate *= 0.85

        # Terrain impact
        if env_conditions.terrain_smoothness == "Rough":
            base_rate *= 0.95

        # Interference factors
        if env_conditions.rubble_presence:
            base_rate *= 0.9
        if env_conditions.tree_roots_presence:
            base_rate *= 0.95

        return max(0.3, min(0.99, base_rate))

    def _calculate_environmental_position_noise(self, env_conditions: EnvironmentalConditions) -> float:
        """Calculate position noise based on environmental conditions."""
        base_noise = 0.3

        # Weather impact
        if env_conditions.weather_condition == "Rainy":
            base_noise *= 1.2

        # Ground condition impact
        if env_conditions.ground_condition == "Clayey":
            base_noise *= 1.3

        # Terrain impact
        if env_conditions.terrain_smoothness == "Rough":
            base_noise *= 1.1

        return max(0.1, min(1.0, base_noise))


def create_test_data_generator(random_seed: int = 42) -> TestDataGenerator:
    """
    Factory function to create a test data generator.

    Args:
        random_seed: Random seed for reproducible data generation

    Returns:
        Configured TestDataGenerator instance
    """
    return TestDataGenerator(random_seed=random_seed)