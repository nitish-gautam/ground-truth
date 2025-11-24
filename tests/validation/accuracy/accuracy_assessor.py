"""
Accuracy Assessment Framework for GPR Detection Validation.

This module provides comprehensive accuracy assessment capabilities for evaluating
GPR detection performance against ground truth data, including position accuracy,
material classification accuracy, depth estimation accuracy, and utility discipline
classification validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support
)
from scipy.spatial.distance import cdist
from collections import defaultdict


logger = logging.getLogger(__name__)


class AccuracyMetric(Enum):
    """Available accuracy metrics."""
    HORIZONTAL_RMSE = "horizontal_rmse"
    VERTICAL_RMSE = "vertical_rmse"
    TOTAL_POSITION_RMSE = "total_position_rmse"
    HORIZONTAL_MAE = "horizontal_mae"
    VERTICAL_MAE = "vertical_mae"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    DETECTION_RATE = "detection_rate"
    FALSE_POSITIVE_RATE = "false_positive_rate"


@dataclass
class PositionAccuracy:
    """Position accuracy metrics."""
    horizontal_rmse: float
    vertical_rmse: float
    horizontal_mae: float
    vertical_mae: float
    horizontal_std: float
    vertical_std: float
    horizontal_bias: float
    vertical_bias: float
    total_rmse: float
    max_horizontal_error: float
    max_vertical_error: float
    error_percentiles: Dict[int, Tuple[float, float]]  # percentile -> (horizontal, vertical)


@dataclass
class MaterialClassificationAccuracy:
    """Material classification accuracy metrics."""
    overall_accuracy: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    support_per_class: Dict[str, int]
    misclassification_matrix: Dict[Tuple[str, str], int]


@dataclass
class DepthEstimationAccuracy:
    """Depth estimation accuracy metrics."""
    rmse: float
    mae: float
    bias: float
    std_error: float
    relative_error_mean: float
    relative_error_std: float
    depth_error_percentiles: Dict[int, float]
    max_error: float
    min_error: float


@dataclass
class DisciplineClassificationAccuracy:
    """Utility discipline classification accuracy metrics."""
    overall_accuracy: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    per_discipline_precision: Dict[str, float]
    per_discipline_recall: Dict[str, float]
    per_discipline_f1: Dict[str, float]
    support_per_discipline: Dict[str, int]


@dataclass
class DetectionPerformance:
    """Overall detection performance metrics."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    detection_rate: float
    false_positive_rate: float
    utility_count_accuracy: float


@dataclass
class DetectedUtility:
    """Detected utility information."""
    x_position: float
    y_position: float
    depth: Optional[float]
    material: Optional[str]
    diameter: Optional[float]
    discipline: Optional[str]
    confidence: float
    detection_id: str


@dataclass
class GroundTruthUtility:
    """Ground truth utility information."""
    x_position: float
    y_position: float
    depth: float
    material: str
    diameter: float
    discipline: str
    utility_id: str


@dataclass
class UtilityMatch:
    """Matched detected utility to ground truth."""
    detected: DetectedUtility
    ground_truth: GroundTruthUtility
    position_error: float
    depth_error: Optional[float]
    material_match: bool
    discipline_match: bool


class AccuracyAssessor:
    """Comprehensive accuracy assessment for GPR detection validation."""

    def __init__(self, position_tolerance: float = 1.0, depth_tolerance: float = 0.3):
        """
        Initialize the accuracy assessor.

        Args:
            position_tolerance: Maximum distance (meters) for considering a detection as a match
            depth_tolerance: Maximum depth difference (meters) for depth accuracy assessment
        """
        self.position_tolerance = position_tolerance
        self.depth_tolerance = depth_tolerance

    def assess_position_accuracy(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> PositionAccuracy:
        """
        Assess position accuracy of detections against ground truth.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            PositionAccuracy metrics
        """
        logger.info("Assessing position accuracy")

        matches = self._match_utilities(detections, ground_truth)

        if not matches:
            logger.warning("No matches found for position accuracy assessment")
            return self._empty_position_accuracy()

        # Calculate horizontal and vertical errors
        horizontal_errors = []
        vertical_errors = []

        for match in matches:
            # Horizontal error (Euclidean distance in XY plane)
            dx = match.detected.x_position - match.ground_truth.x_position
            dy = match.detected.y_position - match.ground_truth.y_position
            horizontal_error = math.sqrt(dx*dx + dy*dy)
            horizontal_errors.append(horizontal_error)

            # Vertical error (depth difference)
            if match.detected.depth is not None:
                vertical_error = abs(match.detected.depth - match.ground_truth.depth)
                vertical_errors.append(vertical_error)

        # Calculate statistics
        horizontal_errors = np.array(horizontal_errors)
        vertical_errors = np.array(vertical_errors) if vertical_errors else np.array([])

        # Horizontal metrics
        horizontal_rmse = np.sqrt(np.mean(horizontal_errors**2))
        horizontal_mae = np.mean(horizontal_errors)
        horizontal_std = np.std(horizontal_errors)
        horizontal_bias = np.mean(horizontal_errors)  # For symmetric errors, bias ≈ MAE

        # Vertical metrics
        vertical_rmse = np.sqrt(np.mean(vertical_errors**2)) if len(vertical_errors) > 0 else 0.0
        vertical_mae = np.mean(vertical_errors) if len(vertical_errors) > 0 else 0.0
        vertical_std = np.std(vertical_errors) if len(vertical_errors) > 0 else 0.0
        vertical_bias = np.mean(vertical_errors) if len(vertical_errors) > 0 else 0.0

        # Total position RMSE (3D if depth available)
        if len(vertical_errors) > 0:
            total_rmse = np.sqrt(np.mean(horizontal_errors**2 + vertical_errors**2))
        else:
            total_rmse = horizontal_rmse

        # Maximum errors
        max_horizontal_error = np.max(horizontal_errors)
        max_vertical_error = np.max(vertical_errors) if len(vertical_errors) > 0 else 0.0

        # Error percentiles
        percentiles = [50, 75, 90, 95, 99]
        error_percentiles = {}
        for p in percentiles:
            h_percentile = np.percentile(horizontal_errors, p)
            v_percentile = np.percentile(vertical_errors, p) if len(vertical_errors) > 0 else 0.0
            error_percentiles[p] = (h_percentile, v_percentile)

        return PositionAccuracy(
            horizontal_rmse=horizontal_rmse,
            vertical_rmse=vertical_rmse,
            horizontal_mae=horizontal_mae,
            vertical_mae=vertical_mae,
            horizontal_std=horizontal_std,
            vertical_std=vertical_std,
            horizontal_bias=horizontal_bias,
            vertical_bias=vertical_bias,
            total_rmse=total_rmse,
            max_horizontal_error=max_horizontal_error,
            max_vertical_error=max_vertical_error,
            error_percentiles=error_percentiles
        )

    def assess_material_classification_accuracy(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> MaterialClassificationAccuracy:
        """
        Assess material classification accuracy.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            MaterialClassificationAccuracy metrics
        """
        logger.info("Assessing material classification accuracy")

        matches = self._match_utilities(detections, ground_truth)

        if not matches:
            logger.warning("No matches found for material classification assessment")
            return self._empty_material_accuracy()

        # Extract material classifications
        predicted_materials = []
        true_materials = []

        for match in matches:
            if match.detected.material and match.ground_truth.material:
                predicted_materials.append(match.detected.material)
                true_materials.append(match.ground_truth.material)

        if not predicted_materials:
            logger.warning("No material information available for classification assessment")
            return self._empty_material_accuracy()

        # Calculate metrics
        overall_accuracy = accuracy_score(true_materials, predicted_materials)

        # Get unique materials
        all_materials = sorted(list(set(true_materials + predicted_materials)))

        # Confusion matrix
        cm = confusion_matrix(true_materials, predicted_materials, labels=all_materials)

        # Classification report
        class_report = classification_report(
            true_materials, predicted_materials,
            labels=all_materials, output_dict=True, zero_division=0
        )

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_materials, predicted_materials,
            labels=all_materials, zero_division=0
        )

        per_class_precision = {material: precision[i] for i, material in enumerate(all_materials)}
        per_class_recall = {material: recall[i] for i, material in enumerate(all_materials)}
        per_class_f1 = {material: f1[i] for i, material in enumerate(all_materials)}
        support_per_class = {material: support[i] for i, material in enumerate(all_materials)}

        # Misclassification matrix
        misclassification_matrix = {}
        for i, true_mat in enumerate(all_materials):
            for j, pred_mat in enumerate(all_materials):
                if i != j:  # Only misclassifications
                    misclassification_matrix[(true_mat, pred_mat)] = cm[i, j]

        return MaterialClassificationAccuracy(
            overall_accuracy=overall_accuracy,
            confusion_matrix=cm,
            classification_report=class_report,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            support_per_class=support_per_class,
            misclassification_matrix=misclassification_matrix
        )

    def assess_depth_estimation_accuracy(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> DepthEstimationAccuracy:
        """
        Assess depth estimation accuracy.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            DepthEstimationAccuracy metrics
        """
        logger.info("Assessing depth estimation accuracy")

        matches = self._match_utilities(detections, ground_truth)

        depth_errors = []
        relative_errors = []

        for match in matches:
            if match.detected.depth is not None:
                absolute_error = abs(match.detected.depth - match.ground_truth.depth)
                depth_errors.append(absolute_error)

                # Relative error (percentage)
                if match.ground_truth.depth > 0:
                    relative_error = absolute_error / match.ground_truth.depth
                    relative_errors.append(relative_error)

        if not depth_errors:
            logger.warning("No depth information available for accuracy assessment")
            return self._empty_depth_accuracy()

        depth_errors = np.array(depth_errors)
        relative_errors = np.array(relative_errors)

        # Calculate metrics
        rmse = np.sqrt(np.mean(depth_errors**2))
        mae = np.mean(depth_errors)
        bias = np.mean([match.detected.depth - match.ground_truth.depth
                      for match in matches if match.detected.depth is not None])
        std_error = np.std(depth_errors)

        relative_error_mean = np.mean(relative_errors) if len(relative_errors) > 0 else 0.0
        relative_error_std = np.std(relative_errors) if len(relative_errors) > 0 else 0.0

        # Error percentiles
        percentiles = [50, 75, 90, 95, 99]
        depth_error_percentiles = {p: np.percentile(depth_errors, p) for p in percentiles}

        max_error = np.max(depth_errors)
        min_error = np.min(depth_errors)

        return DepthEstimationAccuracy(
            rmse=rmse,
            mae=mae,
            bias=bias,
            std_error=std_error,
            relative_error_mean=relative_error_mean,
            relative_error_std=relative_error_std,
            depth_error_percentiles=depth_error_percentiles,
            max_error=max_error,
            min_error=min_error
        )

    def assess_discipline_classification_accuracy(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> DisciplineClassificationAccuracy:
        """
        Assess utility discipline classification accuracy.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            DisciplineClassificationAccuracy metrics
        """
        logger.info("Assessing discipline classification accuracy")

        matches = self._match_utilities(detections, ground_truth)

        if not matches:
            logger.warning("No matches found for discipline classification assessment")
            return self._empty_discipline_accuracy()

        # Extract discipline classifications
        predicted_disciplines = []
        true_disciplines = []

        for match in matches:
            if match.detected.discipline and match.ground_truth.discipline:
                predicted_disciplines.append(match.detected.discipline)
                true_disciplines.append(match.ground_truth.discipline)

        if not predicted_disciplines:
            logger.warning("No discipline information available for classification assessment")
            return self._empty_discipline_accuracy()

        # Calculate metrics
        overall_accuracy = accuracy_score(true_disciplines, predicted_disciplines)

        # Get unique disciplines
        all_disciplines = sorted(list(set(true_disciplines + predicted_disciplines)))

        # Confusion matrix
        cm = confusion_matrix(true_disciplines, predicted_disciplines, labels=all_disciplines)

        # Classification report
        class_report = classification_report(
            true_disciplines, predicted_disciplines,
            labels=all_disciplines, output_dict=True, zero_division=0
        )

        # Per-discipline metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_disciplines, predicted_disciplines,
            labels=all_disciplines, zero_division=0
        )

        per_discipline_precision = {discipline: precision[i] for i, discipline in enumerate(all_disciplines)}
        per_discipline_recall = {discipline: recall[i] for i, discipline in enumerate(all_disciplines)}
        per_discipline_f1 = {discipline: f1[i] for i, discipline in enumerate(all_disciplines)}
        support_per_discipline = {discipline: support[i] for i, discipline in enumerate(all_disciplines)}

        return DisciplineClassificationAccuracy(
            overall_accuracy=overall_accuracy,
            confusion_matrix=cm,
            classification_report=class_report,
            per_discipline_precision=per_discipline_precision,
            per_discipline_recall=per_discipline_recall,
            per_discipline_f1=per_discipline_f1,
            support_per_discipline=support_per_discipline
        )

    def assess_detection_performance(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> DetectionPerformance:
        """
        Assess overall detection performance.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            DetectionPerformance metrics
        """
        logger.info("Assessing detection performance")

        matches = self._match_utilities(detections, ground_truth)

        # Count true positives (matched detections)
        true_positives = len(matches)

        # Count false positives (unmatched detections)
        matched_detection_ids = {match.detected.detection_id for match in matches}
        false_positives = len([d for d in detections if d.detection_id not in matched_detection_ids])

        # Count false negatives (unmatched ground truth)
        matched_gt_ids = {match.ground_truth.utility_id for match in matches}
        false_negatives = len([gt for gt in ground_truth if gt.utility_id not in matched_gt_ids])

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        detection_rate = recall  # Same as recall
        false_positive_rate = false_positives / len(detections) if len(detections) > 0 else 0.0

        # Utility count accuracy
        predicted_count = len(detections)
        true_count = len(ground_truth)
        utility_count_accuracy = 1.0 - abs(predicted_count - true_count) / max(true_count, 1)

        return DetectionPerformance(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            utility_count_accuracy=utility_count_accuracy
        )

    def comprehensive_accuracy_assessment(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive accuracy assessment across all metrics.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            Dictionary containing all accuracy assessment results
        """
        logger.info("Performing comprehensive accuracy assessment")

        results = {
            'position_accuracy': self.assess_position_accuracy(detections, ground_truth),
            'material_classification': self.assess_material_classification_accuracy(detections, ground_truth),
            'depth_estimation': self.assess_depth_estimation_accuracy(detections, ground_truth),
            'discipline_classification': self.assess_discipline_classification_accuracy(detections, ground_truth),
            'detection_performance': self.assess_detection_performance(detections, ground_truth),
            'summary_statistics': self._calculate_summary_statistics(detections, ground_truth)
        }

        return results

    def _match_utilities(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> List[UtilityMatch]:
        """
        Match detected utilities to ground truth utilities based on position.

        Args:
            detections: List of detected utilities
            ground_truth: List of ground truth utilities

        Returns:
            List of utility matches
        """
        if not detections or not ground_truth:
            return []

        matches = []
        used_ground_truth = set()

        # Create position arrays for distance calculation
        detection_positions = np.array([[d.x_position, d.y_position] for d in detections])
        gt_positions = np.array([[gt.x_position, gt.y_position] for gt in ground_truth])

        # Calculate distance matrix
        distances = cdist(detection_positions, gt_positions, metric='euclidean')

        # Find best matches
        for i, detection in enumerate(detections):
            min_distance = float('inf')
            best_match_idx = -1

            for j, gt_utility in enumerate(ground_truth):
                if j in used_ground_truth:
                    continue

                distance = distances[i, j]
                if distance < min_distance and distance <= self.position_tolerance:
                    min_distance = distance
                    best_match_idx = j

            if best_match_idx != -1:
                used_ground_truth.add(best_match_idx)
                gt_utility = ground_truth[best_match_idx]

                # Calculate errors
                depth_error = None
                if detection.depth is not None:
                    depth_error = abs(detection.depth - gt_utility.depth)

                material_match = (detection.material == gt_utility.material) if detection.material else False
                discipline_match = (detection.discipline == gt_utility.discipline) if detection.discipline else False

                match = UtilityMatch(
                    detected=detection,
                    ground_truth=gt_utility,
                    position_error=min_distance,
                    depth_error=depth_error,
                    material_match=material_match,
                    discipline_match=discipline_match
                )
                matches.append(match)

        return matches

    def _calculate_summary_statistics(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the assessment."""
        matches = self._match_utilities(detections, ground_truth)

        return {
            'total_detections': len(detections),
            'total_ground_truth': len(ground_truth),
            'total_matches': len(matches),
            'match_rate': len(matches) / len(ground_truth) if ground_truth else 0.0,
            'average_confidence': np.mean([d.confidence for d in detections]) if detections else 0.0,
            'detections_with_depth': sum(1 for d in detections if d.depth is not None),
            'detections_with_material': sum(1 for d in detections if d.material is not None),
            'detections_with_discipline': sum(1 for d in detections if d.discipline is not None)
        }

    def _empty_position_accuracy(self) -> PositionAccuracy:
        """Return empty position accuracy metrics."""
        return PositionAccuracy(
            horizontal_rmse=0.0, vertical_rmse=0.0, horizontal_mae=0.0, vertical_mae=0.0,
            horizontal_std=0.0, vertical_std=0.0, horizontal_bias=0.0, vertical_bias=0.0,
            total_rmse=0.0, max_horizontal_error=0.0, max_vertical_error=0.0,
            error_percentiles={}
        )

    def _empty_material_accuracy(self) -> MaterialClassificationAccuracy:
        """Return empty material classification accuracy metrics."""
        return MaterialClassificationAccuracy(
            overall_accuracy=0.0, confusion_matrix=np.array([]), classification_report={},
            per_class_precision={}, per_class_recall={}, per_class_f1={},
            support_per_class={}, misclassification_matrix={}
        )

    def _empty_depth_accuracy(self) -> DepthEstimationAccuracy:
        """Return empty depth estimation accuracy metrics."""
        return DepthEstimationAccuracy(
            rmse=0.0, mae=0.0, bias=0.0, std_error=0.0,
            relative_error_mean=0.0, relative_error_std=0.0,
            depth_error_percentiles={}, max_error=0.0, min_error=0.0
        )

    def _empty_discipline_accuracy(self) -> DisciplineClassificationAccuracy:
        """Return empty discipline classification accuracy metrics."""
        return DisciplineClassificationAccuracy(
            overall_accuracy=0.0, confusion_matrix=np.array([]), classification_report={},
            per_discipline_precision={}, per_discipline_recall={}, per_discipline_f1={},
            support_per_discipline={}
        )


def create_accuracy_assessor(
    position_tolerance: float = 1.0,
    depth_tolerance: float = 0.3
) -> AccuracyAssessor:
    """
    Factory function to create an accuracy assessor.

    Args:
        position_tolerance: Maximum distance (meters) for considering a detection as a match
        depth_tolerance: Maximum depth difference (meters) for depth accuracy assessment

    Returns:
        Configured AccuracyAssessor instance
    """
    return AccuracyAssessor(position_tolerance=position_tolerance, depth_tolerance=depth_tolerance)