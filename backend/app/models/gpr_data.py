"""
GPR data models
==============

Database models for GPR surveys, scans, signal data, and processing results.
Handles the core GPR data from Twente dataset and processing pipeline.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, CheckConstraint, LargeBinary
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class GPRSurvey(BaseModel):
    """GPR survey session containing multiple scans."""

    __tablename__ = "gpr_surveys"

    # Basic survey information
    survey_name: Mapped[str] = mapped_column(String(255), nullable=False)
    location_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    survey_objective: Mapped[Optional[str]] = mapped_column(String(500))
    survey_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Survey configuration
    equipment_model: Mapped[Optional[str]] = mapped_column(String(100))
    antenna_frequency: Mapped[Optional[float]] = mapped_column(Float)
    sampling_frequency: Mapped[Optional[float]] = mapped_column(Float)
    time_window: Mapped[Optional[float]] = mapped_column(Float)

    # Survey status and quality
    status: Mapped[str] = mapped_column(
        String(50),
        default="planned",
        server_default="planned"
    )
    quality_level: Mapped[Optional[str]] = mapped_column(String(10))  # PAS 128 QL-A to QL-D
    completion_percentage: Mapped[float] = mapped_column(Float, default=0.0)

    # Spatial information
    survey_area_description: Mapped[Optional[str]] = mapped_column(String(500))
    coordinate_system: Mapped[Optional[str]] = mapped_column(String(100))

    # Relationships
    scans: Mapped[List["GPRScan"]] = relationship(
        "GPRScan",
        back_populates="survey",
        cascade="all, delete-orphan"
    )

    environmental_data: Mapped[List["EnvironmentalData"]] = relationship(
        "EnvironmentalData",
        back_populates="survey",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('planned', 'active', 'completed', 'validated', 'archived')",
            name="check_survey_status"
        ),
        CheckConstraint(
            "quality_level IN ('QL-A', 'QL-B', 'QL-C', 'QL-D') OR quality_level IS NULL",
            name="check_quality_level"
        ),
        CheckConstraint(
            "completion_percentage >= 0 AND completion_percentage <= 100",
            name="check_completion_percentage"
        ),
        Index("idx_gpr_surveys_location_date", "location_id", "survey_date"),
        Index("idx_gpr_surveys_status", "status"),
    )


class GPRScan(BaseModel):
    """Individual GPR scan within a survey."""

    __tablename__ = "gpr_scans"

    # Survey relationship
    survey_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_surveys.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Scan identification
    scan_number: Mapped[int] = mapped_column(Integer, nullable=False)
    scan_name: Mapped[Optional[str]] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)

    # Scan parameters
    start_position: Mapped[Optional[float]] = mapped_column(Float)  # meters
    end_position: Mapped[Optional[float]] = mapped_column(Float)    # meters
    scan_length: Mapped[Optional[float]] = mapped_column(Float)     # meters
    trace_count: Mapped[Optional[int]] = mapped_column(Integer)
    samples_per_trace: Mapped[Optional[int]] = mapped_column(Integer)

    # Data format information
    data_format: Mapped[Optional[str]] = mapped_column(String(50))  # DT1, DZT, etc.
    header_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Processing status
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_status: Mapped[Optional[str]] = mapped_column(String(100))
    processing_errors: Mapped[Optional[str]] = mapped_column(Text)

    # Quality metrics
    signal_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    noise_level: Mapped[Optional[float]] = mapped_column(Float)
    data_completeness: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    survey: Mapped["GPRSurvey"] = relationship("GPRSurvey", back_populates="scans")
    signal_data: Mapped[List["GPRSignalData"]] = relationship(
        "GPRSignalData",
        back_populates="scan",
        cascade="all, delete-orphan"
    )
    processing_results: Mapped[List["GPRProcessingResult"]] = relationship(
        "GPRProcessingResult",
        back_populates="scan",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("scan_number > 0", name="check_scan_number_positive"),
        CheckConstraint("trace_count > 0 OR trace_count IS NULL", name="check_trace_count"),
        CheckConstraint("samples_per_trace > 0 OR samples_per_trace IS NULL", name="check_samples_per_trace"),
        CheckConstraint(
            "signal_quality_score >= 0 AND signal_quality_score <= 1 OR signal_quality_score IS NULL",
            name="check_signal_quality_score"
        ),
        Index("idx_gpr_scans_survey_scan", "survey_id", "scan_number"),
        Index("idx_gpr_scans_processing", "is_processed", "processing_status"),
    )


class GPRSignalData(BaseModel):
    """Raw and processed GPR signal data."""

    __tablename__ = "gpr_signal_data"

    # Scan relationship
    scan_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Data identification
    trace_number: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)  # raw, filtered, processed

    # Signal data (stored as binary for efficiency)
    signal_data: Mapped[Optional[bytes]] = mapped_column(LargeBinary)

    # Signal metadata
    time_zero: Mapped[Optional[float]] = mapped_column(Float)
    depth_calibration: Mapped[Optional[float]] = mapped_column(Float)
    amplitude_scaling: Mapped[Optional[float]] = mapped_column(Float)

    # Processing parameters
    filter_applied: Mapped[Optional[str]] = mapped_column(String(200))
    processing_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Quality indicators
    snr_db: Mapped[Optional[float]] = mapped_column(Float)  # Signal-to-noise ratio
    rms_amplitude: Mapped[Optional[float]] = mapped_column(Float)
    peak_amplitude: Mapped[Optional[float]] = mapped_column(Float)

    # Spatial information
    position_x: Mapped[Optional[float]] = mapped_column(Float)
    position_y: Mapped[Optional[float]] = mapped_column(Float)
    gps_latitude: Mapped[Optional[float]] = mapped_column(Float)
    gps_longitude: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    scan: Mapped["GPRScan"] = relationship("GPRScan", back_populates="signal_data")

    # Constraints
    __table_args__ = (
        CheckConstraint("trace_number > 0", name="check_trace_number_positive"),
        CheckConstraint(
            "data_type IN ('raw', 'filtered', 'processed', 'analyzed')",
            name="check_data_type"
        ),
        Index("idx_gpr_signal_data_scan_trace", "scan_id", "trace_number"),
        Index("idx_gpr_signal_data_type", "data_type"),
        Index("idx_gpr_signal_data_position", "position_x", "position_y"),
    )


class GPRProcessingResult(BaseModel):
    """Results from GPR signal processing and analysis."""

    __tablename__ = "gpr_processing_results"

    # Scan relationship
    scan_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Processing information
    processing_algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    processing_version: Mapped[Optional[str]] = mapped_column(String(50))
    processing_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    processing_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    # Processing parameters
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Results
    detected_features: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    hyperbola_detections: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    utility_predictions: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    # Confidence and quality metrics
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float)
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    false_positive_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Feature extraction results
    feature_vector: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    statistical_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    frequency_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)

    # Environmental correlation
    environmental_impact_score: Mapped[Optional[float]] = mapped_column(Float)
    weather_correlation: Mapped[Optional[float]] = mapped_column(Float)
    ground_condition_impact: Mapped[Optional[float]] = mapped_column(Float)

    # Processing status
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    scan: Mapped["GPRScan"] = relationship("GPRScan", back_populates="processing_results")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="check_processing_status"
        ),
        CheckConstraint(
            "overall_confidence >= 0 AND overall_confidence <= 1 OR overall_confidence IS NULL",
            name="check_overall_confidence"
        ),
        CheckConstraint("detection_count >= 0", name="check_detection_count"),
        Index("idx_gpr_processing_scan_algorithm", "scan_id", "processing_algorithm"),
        Index("idx_gpr_processing_timestamp", "processing_timestamp"),
        Index("idx_gpr_processing_status", "status"),
    )