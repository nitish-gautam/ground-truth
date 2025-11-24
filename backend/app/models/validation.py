"""
Validation and ground truth models
=================================

Database models for ground truth data, validation results, and accuracy
assessment based on known utility locations from Twente dataset.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, JSON, ForeignKey,
    Index, CheckConstraint, Integer, Text
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class GroundTruthData(BaseModel):
    """Ground truth utility locations and characteristics."""

    __tablename__ = "ground_truth_data"

    # Survey relationship
    survey_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_surveys.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Utility identification
    utility_id: Mapped[str] = mapped_column(String(100), nullable=False)
    utility_name: Mapped[Optional[str]] = mapped_column(String(255))

    # Utility characteristics from Twente dataset
    utility_discipline: Mapped[str] = mapped_column(String(100), nullable=False)
    utility_material: Mapped[Optional[str]] = mapped_column(String(100))
    utility_diameter: Mapped[Optional[float]] = mapped_column(Float)  # in meters
    utility_depth: Mapped[Optional[float]] = mapped_column(Float)     # in meters

    # Spatial information
    start_x: Mapped[Optional[float]] = mapped_column(Float)
    start_y: Mapped[Optional[float]] = mapped_column(Float)
    end_x: Mapped[Optional[float]] = mapped_column(Float)
    end_y: Mapped[Optional[float]] = mapped_column(Float)
    centerline_coordinates: Mapped[Optional[List[List[float]]]] = mapped_column(JSON)

    # Geometric properties
    length_meters: Mapped[Optional[float]] = mapped_column(Float)
    orientation_degrees: Mapped[Optional[float]] = mapped_column(Float)
    is_linear: Mapped[Optional[bool]] = mapped_column(Boolean)
    has_bends: Mapped[Optional[bool]] = mapped_column(Boolean)
    bend_locations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    # Utility status and condition
    utility_status: Mapped[Optional[str]] = mapped_column(String(50))  # active, abandoned, out_of_service
    installation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_maintenance_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    condition_assessment: Mapped[Optional[str]] = mapped_column(String(100))

    # Additional information from survey
    additional_utility_information: Mapped[Optional[str]] = mapped_column(Text)
    survey_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Data quality and confidence
    location_accuracy_m: Mapped[Optional[float]] = mapped_column(Float)
    depth_accuracy_m: Mapped[Optional[float]] = mapped_column(Float)
    confidence_level: Mapped[Optional[str]] = mapped_column(String(20))
    verification_method: Mapped[Optional[str]] = mapped_column(String(100))
    verification_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Source information
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    source_quality: Mapped[Optional[str]] = mapped_column(String(50))
    record_keeper: Mapped[Optional[str]] = mapped_column(String(100))

    # Relationships
    survey: Mapped["GPRSurvey"] = relationship("GPRSurvey")
    validation_results: Mapped[List["ValidationResult"]] = relationship(
        "ValidationResult",
        back_populates="ground_truth",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "utility_discipline IN ('electricity', 'water', 'sewer', 'telecommunications', "
            "'gas', 'oilGasChemicals', 'heating', 'other', 'unknown')",
            name="check_utility_discipline"
        ),
        CheckConstraint(
            "utility_diameter > 0 OR utility_diameter IS NULL",
            name="check_utility_diameter"
        ),
        CheckConstraint(
            "utility_depth >= 0 OR utility_depth IS NULL",
            name="check_utility_depth"
        ),
        CheckConstraint(
            "length_meters > 0 OR length_meters IS NULL",
            name="check_length_positive"
        ),
        CheckConstraint(
            "orientation_degrees >= 0 AND orientation_degrees < 360 OR orientation_degrees IS NULL",
            name="check_orientation_range"
        ),
        CheckConstraint(
            "location_accuracy_m >= 0 OR location_accuracy_m IS NULL",
            name="check_location_accuracy"
        ),
        CheckConstraint(
            "depth_accuracy_m >= 0 OR depth_accuracy_m IS NULL",
            name="check_depth_accuracy"
        ),
        CheckConstraint(
            "confidence_level IN ('very_low', 'low', 'medium', 'high', 'very_high') OR confidence_level IS NULL",
            name="check_confidence_level"
        ),
        Index("idx_ground_truth_survey", "survey_id"),
        Index("idx_ground_truth_utility", "utility_id"),
        Index("idx_ground_truth_discipline", "utility_discipline"),
        Index("idx_ground_truth_material", "utility_material"),
        Index("idx_ground_truth_position", "start_x", "start_y"),
    )


class ValidationResult(BaseModel):
    """Results from comparing GPR detections with ground truth."""

    __tablename__ = "validation_results"

    # Ground truth relationship
    ground_truth_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("ground_truth_data.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # GPR scan relationship
    scan_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Processing result relationship (optional)
    processing_result_id: Mapped[Optional[UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_processing_results.id", ondelete="SET NULL"),
        index=True
    )

    # Validation metadata
    validation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    validation_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    validator_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Detection results
    detection_result: Mapped[str] = mapped_column(String(20), nullable=False)  # true_positive, false_positive, etc.
    detection_confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Position accuracy
    predicted_x: Mapped[Optional[float]] = mapped_column(Float)
    predicted_y: Mapped[Optional[float]] = mapped_column(Float)
    predicted_depth: Mapped[Optional[float]] = mapped_column(Float)
    position_error_m: Mapped[Optional[float]] = mapped_column(Float)
    depth_error_m: Mapped[Optional[float]] = mapped_column(Float)

    # Attribute accuracy
    predicted_material: Mapped[Optional[str]] = mapped_column(String(100))
    predicted_diameter: Mapped[Optional[float]] = mapped_column(Float)
    material_match: Mapped[Optional[bool]] = mapped_column(Boolean)
    diameter_error_percent: Mapped[Optional[float]] = mapped_column(Float)

    # Signal characteristics
    signal_strength: Mapped[Optional[float]] = mapped_column(Float)
    signal_clarity: Mapped[Optional[float]] = mapped_column(Float)
    background_noise_level: Mapped[Optional[float]] = mapped_column(Float)

    # Environmental impact on detection
    environmental_factor_impact: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    weather_impact_score: Mapped[Optional[float]] = mapped_column(Float)
    ground_condition_impact_score: Mapped[Optional[float]] = mapped_column(Float)

    # Quality metrics
    detection_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    pas128_quality_level: Mapped[Optional[str]] = mapped_column(String(10))

    # Additional validation data
    validation_notes: Mapped[Optional[str]] = mapped_column(Text)
    manual_review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    review_status: Mapped[Optional[str]] = mapped_column(String(50))
    reviewer_comments: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    ground_truth: Mapped["GroundTruthData"] = relationship(
        "GroundTruthData",
        back_populates="validation_results"
    )
    scan: Mapped["GPRScan"] = relationship("GPRScan")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "detection_result IN ('true_positive', 'false_positive', 'false_negative', 'true_negative')",
            name="check_detection_result"
        ),
        CheckConstraint(
            "detection_confidence >= 0 AND detection_confidence <= 1 OR detection_confidence IS NULL",
            name="check_detection_confidence"
        ),
        CheckConstraint(
            "position_error_m >= 0 OR position_error_m IS NULL",
            name="check_position_error"
        ),
        CheckConstraint(
            "depth_error_m >= 0 OR depth_error_m IS NULL",
            name="check_depth_error"
        ),
        CheckConstraint(
            "predicted_diameter > 0 OR predicted_diameter IS NULL",
            name="check_predicted_diameter"
        ),
        CheckConstraint(
            "detection_quality_score >= 0 AND detection_quality_score <= 1 OR detection_quality_score IS NULL",
            name="check_quality_score"
        ),
        CheckConstraint(
            "pas128_quality_level IN ('QL-A', 'QL-B', 'QL-C', 'QL-D') OR pas128_quality_level IS NULL",
            name="check_pas128_quality"
        ),
        Index("idx_validation_ground_truth", "ground_truth_id"),
        Index("idx_validation_scan", "scan_id"),
        Index("idx_validation_result", "detection_result"),
        Index("idx_validation_timestamp", "validation_timestamp"),
    )


class AccuracyMetrics(BaseModel):
    """Aggregated accuracy metrics for surveys, algorithms, or conditions."""

    __tablename__ = "accuracy_metrics"

    # Context identification
    context_type: Mapped[str] = mapped_column(String(50), nullable=False)  # survey, algorithm, condition, etc.
    context_id: Mapped[str] = mapped_column(String(255), nullable=False)
    context_description: Mapped[Optional[str]] = mapped_column(String(500))

    # Time period
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Basic detection metrics
    total_ground_truth_utilities: Mapped[int] = mapped_column(Integer, nullable=False)
    total_detections: Mapped[int] = mapped_column(Integer, nullable=False)
    true_positives: Mapped[int] = mapped_column(Integer, nullable=False)
    false_positives: Mapped[int] = mapped_column(Integer, nullable=False)
    false_negatives: Mapped[int] = mapped_column(Integer, nullable=False)
    true_negatives: Mapped[int] = mapped_column(Integer, nullable=False)

    # Derived accuracy metrics
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    specificity: Mapped[float] = mapped_column(Float, nullable=False)

    # Position accuracy metrics
    mean_position_error_m: Mapped[Optional[float]] = mapped_column(Float)
    median_position_error_m: Mapped[Optional[float]] = mapped_column(Float)
    position_error_std_m: Mapped[Optional[float]] = mapped_column(Float)
    position_accuracy_within_1m: Mapped[Optional[float]] = mapped_column(Float)
    position_accuracy_within_2m: Mapped[Optional[float]] = mapped_column(Float)

    # Depth accuracy metrics
    mean_depth_error_m: Mapped[Optional[float]] = mapped_column(Float)
    median_depth_error_m: Mapped[Optional[float]] = mapped_column(Float)
    depth_error_std_m: Mapped[Optional[float]] = mapped_column(Float)
    depth_accuracy_within_10cm: Mapped[Optional[float]] = mapped_column(Float)
    depth_accuracy_within_25cm: Mapped[Optional[float]] = mapped_column(Float)

    # Material classification accuracy
    material_classification_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    diameter_estimation_accuracy: Mapped[Optional[float]] = mapped_column(Float)

    # PAS 128 compliance metrics
    ql_a_compliance: Mapped[Optional[float]] = mapped_column(Float)
    ql_b_compliance: Mapped[Optional[float]] = mapped_column(Float)
    ql_c_compliance: Mapped[Optional[float]] = mapped_column(Float)
    ql_d_compliance: Mapped[Optional[float]] = mapped_column(Float)

    # Environmental correlation
    weather_impact_correlation: Mapped[Optional[float]] = mapped_column(Float)
    ground_condition_correlation: Mapped[Optional[float]] = mapped_column(Float)
    utility_density_correlation: Mapped[Optional[float]] = mapped_column(Float)

    # Statistical confidence
    confidence_interval_95: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    statistical_significance: Mapped[Optional[float]] = mapped_column(Float)

    # Calculation metadata
    calculation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    calculation_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    calculation_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Additional metrics
    additional_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Constraints
    __table_args__ = (
        CheckConstraint("total_ground_truth_utilities >= 0", name="check_total_ground_truth"),
        CheckConstraint("total_detections >= 0", name="check_total_detections"),
        CheckConstraint("true_positives >= 0", name="check_true_positives"),
        CheckConstraint("false_positives >= 0", name="check_false_positives"),
        CheckConstraint("false_negatives >= 0", name="check_false_negatives"),
        CheckConstraint("true_negatives >= 0", name="check_true_negatives"),
        CheckConstraint("precision >= 0 AND precision <= 1", name="check_precision"),
        CheckConstraint("recall >= 0 AND recall <= 1", name="check_recall"),
        CheckConstraint("f1_score >= 0 AND f1_score <= 1", name="check_f1_score"),
        CheckConstraint("accuracy >= 0 AND accuracy <= 1", name="check_accuracy"),
        CheckConstraint("sample_size > 0", name="check_sample_size"),
        Index("idx_accuracy_context", "context_type", "context_id"),
        Index("idx_accuracy_timestamp", "calculation_timestamp"),
        Index("idx_accuracy_period", "start_date", "end_date"),
    )