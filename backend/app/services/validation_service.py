"""
Ground truth validation and accuracy assessment service
======================================================

Comprehensive service for validating GPR detections against ground truth data
and calculating accuracy metrics according to PAS 128 standards.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
from scipy.spatial.distance import cdist
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRScan, GPRProcessingResult
from ..models.validation import GroundTruthData, ValidationResult, AccuracyMetrics
from ..models.environmental import EnvironmentalData


class ValidationService(LoggerMixin):
    """Service for ground truth validation and accuracy assessment."""

    def __init__(self):
        super().__init__()
        self.pas128_thresholds = {
            "QL-A": {"horizontal": 0.15, "vertical": 0.05},  # meters
            "QL-B": {"horizontal": 0.5, "vertical": 0.15},
            "QL-C": {"horizontal": 1.0, "vertical": 0.5},
            "QL-D": {"horizontal": 2.0, "vertical": 1.0}
        }

    async def validate_scan_results(
        self,
        db: AsyncSession,
        scan_id: str,
        validation_config: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate detection results for a single scan against ground truth."""
        self.log_operation_start("validate_scan_results", scan_id=scan_id)

        try:
            # Load scan and processing results
            scan = await self._load_scan(db, scan_id)
            if not scan:
                raise ValueError(f"Scan not found: {scan_id}")

            processing_results = await self._load_processing_results(db, scan_id)
            ground_truth_data = await self._load_ground_truth(db, scan.survey_id)

            if not ground_truth_data:
                self.logger.warning(f"No ground truth data found for scan {scan_id}")
                return []

            # Extract detections from processing results
            detections = await self._extract_detections(processing_results)

            # Perform validation
            validation_results = []
            for ground_truth in ground_truth_data:
                validation_result = await self._validate_single_utility(
                    db, scan_id, ground_truth, detections, validation_config
                )
                validation_results.append(validation_result)

            # Check for false positives (detections without corresponding ground truth)
            false_positives = await self._identify_false_positives(
                detections, ground_truth_data, validation_config
            )

            for fp_detection in false_positives:
                fp_result = ValidationResult(
                    ground_truth_id=None,  # No corresponding ground truth
                    scan_id=scan_id,
                    validation_method="automated_comparison",
                    validation_timestamp=datetime.now(),
                    detection_result="false_positive",
                    predicted_x=fp_detection.get("x"),
                    predicted_y=fp_detection.get("y"),
                    predicted_depth=fp_detection.get("depth"),
                    detection_confidence=fp_detection.get("confidence", 0.0),
                    validation_notes="No corresponding ground truth utility found"
                )
                validation_results.append(fp_result)

            # Store validation results
            for result in validation_results:
                db.add(result)

            await db.commit()

            self.log_accuracy_assessment("scan_validation", len(validation_results), scan_id=scan_id)

            return validation_results

        except Exception as e:
            await db.rollback()
            self.log_operation_error("validate_scan_results", e, scan_id=scan_id)
            raise

    async def _validate_single_utility(
        self,
        db: AsyncSession,
        scan_id: str,
        ground_truth: GroundTruthData,
        detections: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate detection against a single ground truth utility."""
        try:
            # Find closest detection to ground truth location
            closest_detection, distance = await self._find_closest_detection(
                ground_truth, detections
            )

            # Determine detection result based on distance thresholds
            max_distance = config.get("max_detection_distance", 2.0)  # meters
            detection_result = "false_negative"  # Default

            if closest_detection and distance <= max_distance:
                detection_result = "true_positive"

                # Calculate position and depth errors
                position_error = distance
                depth_error = abs(
                    closest_detection.get("depth", 0) - (ground_truth.utility_depth or 0)
                )

                # Check material and diameter predictions
                material_match = (
                    closest_detection.get("material") == ground_truth.utility_material
                    if closest_detection.get("material") and ground_truth.utility_material
                    else None
                )

                diameter_error_percent = None
                if (closest_detection.get("diameter") and ground_truth.utility_diameter):
                    predicted_diameter = closest_detection["diameter"]
                    actual_diameter = ground_truth.utility_diameter
                    diameter_error_percent = abs(predicted_diameter - actual_diameter) / actual_diameter * 100

                # Determine PAS 128 quality level
                pas128_quality = await self._determine_pas128_quality(
                    position_error, depth_error
                )

                validation_result = ValidationResult(
                    ground_truth_id=ground_truth.id,
                    scan_id=scan_id,
                    validation_method="automated_comparison",
                    validation_timestamp=datetime.now(),
                    detection_result=detection_result,
                    detection_confidence=closest_detection.get("confidence"),
                    predicted_x=closest_detection.get("x"),
                    predicted_y=closest_detection.get("y"),
                    predicted_depth=closest_detection.get("depth"),
                    position_error_m=position_error,
                    depth_error_m=depth_error,
                    predicted_material=closest_detection.get("material"),
                    predicted_diameter=closest_detection.get("diameter"),
                    material_match=material_match,
                    diameter_error_percent=diameter_error_percent,
                    pas128_quality_level=pas128_quality,
                    signal_strength=closest_detection.get("signal_strength"),
                    signal_clarity=closest_detection.get("signal_clarity")
                )

            else:
                # False negative - utility not detected
                validation_result = ValidationResult(
                    ground_truth_id=ground_truth.id,
                    scan_id=scan_id,
                    validation_method="automated_comparison",
                    validation_timestamp=datetime.now(),
                    detection_result="false_negative",
                    validation_notes=f"No detection within {max_distance}m of ground truth location"
                )

            return validation_result

        except Exception as e:
            self.log_operation_error("validate_single_utility", e)
            raise

    async def _find_closest_detection(
        self,
        ground_truth: GroundTruthData,
        detections: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Find the closest detection to a ground truth utility."""
        if not detections:
            return None, float('inf')

        # Ground truth position
        gt_x = ground_truth.start_x or 0
        gt_y = ground_truth.start_y or 0

        closest_detection = None
        min_distance = float('inf')

        for detection in detections:
            det_x = detection.get("x", 0)
            det_y = detection.get("y", 0)

            distance = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)

            if distance < min_distance:
                min_distance = distance
                closest_detection = detection

        return closest_detection, min_distance

    async def _determine_pas128_quality(
        self,
        position_error: float,
        depth_error: float
    ) -> Optional[str]:
        """Determine PAS 128 quality level based on position and depth errors."""
        for quality_level, thresholds in self.pas128_thresholds.items():
            if (position_error <= thresholds["horizontal"] and
                depth_error <= thresholds["vertical"]):
                return quality_level

        return None  # Doesn't meet any quality level

    async def calculate_accuracy_metrics(
        self,
        db: AsyncSession,
        survey_id: Optional[str] = None,
        timeframe_days: Optional[int] = None
    ) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics."""
        self.log_operation_start("calculate_accuracy_metrics", survey_id=survey_id)

        try:
            # Build query for validation results
            query = select(ValidationResult)

            if survey_id:
                query = query.join(GroundTruthData).where(
                    GroundTruthData.survey_id == survey_id
                )

            if timeframe_days:
                cutoff_date = datetime.now() - timedelta(days=timeframe_days)
                query = query.where(ValidationResult.validation_timestamp >= cutoff_date)

            result = await db.execute(query)
            validation_results = result.scalars().all()

            if not validation_results:
                raise ValueError("No validation results found for specified criteria")

            # Count detection results
            detection_counts = {}
            for result in validation_results:
                detection_counts[result.detection_result] = detection_counts.get(
                    result.detection_result, 0
                ) + 1

            true_positives = detection_counts.get("true_positive", 0)
            false_positives = detection_counts.get("false_positive", 0)
            false_negatives = detection_counts.get("false_negative", 0)
            true_negatives = detection_counts.get("true_negative", 0)

            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / len(validation_results)
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

            # Position and depth accuracy
            position_errors = [r.position_error_m for r in validation_results
                             if r.position_error_m is not None and r.detection_result == "true_positive"]
            depth_errors = [r.depth_error_m for r in validation_results
                          if r.depth_error_m is not None and r.detection_result == "true_positive"]

            # PAS 128 compliance
            pas128_counts = {}
            for result in validation_results:
                if result.pas128_quality_level:
                    pas128_counts[result.pas128_quality_level] = pas128_counts.get(
                        result.pas128_quality_level, 0
                    ) + 1

            total_with_quality = sum(pas128_counts.values())

            # Create accuracy metrics record
            accuracy_metrics = AccuracyMetrics(
                context_type="survey" if survey_id else "global",
                context_id=survey_id or "all",
                context_description=f"Accuracy metrics for {'survey ' + survey_id if survey_id else 'all data'}",
                total_ground_truth_utilities=true_positives + false_negatives,
                total_detections=true_positives + false_positives,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                true_negatives=true_negatives,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                specificity=specificity,
                mean_position_error_m=np.mean(position_errors) if position_errors else None,
                median_position_error_m=np.median(position_errors) if position_errors else None,
                position_error_std_m=np.std(position_errors) if position_errors else None,
                position_accuracy_within_1m=(np.array(position_errors) <= 1.0).mean() if position_errors else None,
                position_accuracy_within_2m=(np.array(position_errors) <= 2.0).mean() if position_errors else None,
                mean_depth_error_m=np.mean(depth_errors) if depth_errors else None,
                median_depth_error_m=np.median(depth_errors) if depth_errors else None,
                depth_error_std_m=np.std(depth_errors) if depth_errors else None,
                depth_accuracy_within_10cm=(np.array(depth_errors) <= 0.1).mean() if depth_errors else None,
                depth_accuracy_within_25cm=(np.array(depth_errors) <= 0.25).mean() if depth_errors else None,
                ql_a_compliance=pas128_counts.get("QL-A", 0) / total_with_quality if total_with_quality > 0 else None,
                ql_b_compliance=pas128_counts.get("QL-B", 0) / total_with_quality if total_with_quality > 0 else None,
                ql_c_compliance=pas128_counts.get("QL-C", 0) / total_with_quality if total_with_quality > 0 else None,
                ql_d_compliance=pas128_counts.get("QL-D", 0) / total_with_quality if total_with_quality > 0 else None,
                sample_size=len(validation_results),
                calculation_method="automated_validation",
                calculation_timestamp=datetime.now()
            )

            db.add(accuracy_metrics)
            await db.commit()

            self.log_accuracy_assessment("accuracy_metrics", accuracy, sample_size=len(validation_results))

            return accuracy_metrics

        except Exception as e:
            await db.rollback()
            self.log_operation_error("calculate_accuracy_metrics", e)
            raise

    async def _load_scan(self, db: AsyncSession, scan_id: str) -> Optional:
        """Load scan from database."""
        result = await db.execute(select(GPRScan).where(GPRScan.id == scan_id))
        return result.scalar_one_or_none()

    async def _load_processing_results(self, db: AsyncSession, scan_id: str) -> List:
        """Load processing results for a scan."""
        result = await db.execute(
            select(GPRProcessingResult).where(GPRProcessingResult.scan_id == scan_id)
        )
        return result.scalars().all()

    async def _load_ground_truth(self, db: AsyncSession, survey_id: str) -> List[GroundTruthData]:
        """Load ground truth data for a survey."""
        result = await db.execute(
            select(GroundTruthData).where(GroundTruthData.survey_id == survey_id)
        )
        return result.scalars().all()

    async def _extract_detections(self, processing_results: List) -> List[Dict[str, Any]]:
        """Extract detection data from processing results."""
        detections = []

        for result in processing_results:
            if result.utility_predictions:
                for prediction in result.utility_predictions:
                    detection = {
                        "x": prediction.get("x", 0),
                        "y": prediction.get("y", 0),
                        "depth": prediction.get("depth", 0),
                        "confidence": prediction.get("confidence", 0),
                        "material": prediction.get("material"),
                        "diameter": prediction.get("diameter"),
                        "signal_strength": prediction.get("signal_strength"),
                        "signal_clarity": prediction.get("signal_clarity")
                    }
                    detections.append(detection)

        return detections

    async def _identify_false_positives(
        self,
        detections: List[Dict[str, Any]],
        ground_truth_data: List[GroundTruthData],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify detections that don't correspond to any ground truth utility."""
        false_positives = []
        max_distance = config.get("max_detection_distance", 2.0)

        for detection in detections:
            det_x = detection.get("x", 0)
            det_y = detection.get("y", 0)

            # Check if detection is close to any ground truth utility
            is_near_ground_truth = False

            for gt in ground_truth_data:
                gt_x = gt.start_x or 0
                gt_y = gt.start_y or 0
                distance = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)

                if distance <= max_distance:
                    is_near_ground_truth = True
                    break

            if not is_near_ground_truth:
                false_positives.append(detection)

        return false_positives