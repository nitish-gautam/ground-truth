"""
HS2 Progress Verification Models
=================================

SQLAlchemy models for progress tracking, point cloud comparison, and schedule management.

Tables:
- HS2ProgressSnapshot: Timeline tracking of progress measurements
- HS2PointCloudComparison: BIM vs reality capture comparison results
- HS2ScheduleMilestone: Milestone tracking with dependency management
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class HS2ProgressSnapshot(Base):
    """
    Progress snapshot for timeline tracking.

    Stores periodic measurements of physical, cost, and schedule progress
    for each asset. Used for progress trending, earned value management,
    and anomaly detection.
    """

    __tablename__ = "hs2_progress_snapshots"

    # Foreign Keys
    asset_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("hs2_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Snapshot metadata
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )

    # Progress Metrics (0-100%)
    physical_progress_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        CheckConstraint("physical_progress_pct >= 0 AND physical_progress_pct <= 100"),
        nullable=True
    )
    cost_progress_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        CheckConstraint("cost_progress_pct >= 0 AND cost_progress_pct <= 100"),
        nullable=True
    )
    schedule_progress_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        CheckConstraint("schedule_progress_pct >= 0 AND schedule_progress_pct <= 100"),
        nullable=True
    )

    # Earned Value Management Metrics
    planned_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)  # PV
    earned_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)  # EV
    actual_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)  # AC
    cost_variance: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)  # CV = EV - AC
    schedule_variance: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)  # SV = EV - PV
    cost_performance_index: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 3), nullable=True)  # CPI = EV / AC
    schedule_performance_index: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 3), nullable=True)  # SPI = EV / PV

    # Point Cloud Data
    point_cloud_file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deviation_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
        comment="Average deviation in mm"
    )

    # Anomalies and Issues
    anomalies: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"type": "cost_overrun", "severity": "high", "description": "..."}]'
    )

    # Metadata
    data_source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'manual'"),
        comment="Source: manual, point_cloud, survey, bim_comparison"
    )
    confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
        server_default="100.00"
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    asset = relationship("HS2Asset", back_populates="progress_snapshots")

    # Indexes
    __table_args__ = (
        Index("idx_progress_asset_date", "asset_id", "snapshot_date"),
        Index("idx_progress_snapshot_date", "snapshot_date"),
        Index("idx_progress_anomalies", "anomalies", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return (
            f"<HS2ProgressSnapshot(asset_id={self.asset_id}, "
            f"snapshot_date={self.snapshot_date}, physical={self.physical_progress_pct}%)>"
        )


class HS2PointCloudComparison(Base):
    """
    Point cloud comparison results.

    Stores results from comparing BIM models against reality capture data
    (LiDAR scans, drone imagery). Used for automated progress verification.
    """

    __tablename__ = "hs2_point_cloud_comparisons"

    # Foreign Keys
    asset_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("hs2_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # File References (MinIO paths)
    baseline_file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="BIM model or baseline scan file path"
    )
    current_file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Current site scan (LAS/LAZ) file path"
    )
    comparison_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )

    # Volume Analysis
    volume_difference_m3: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 3), nullable=True)
    volume_planned_m3: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 3), nullable=True)
    volume_actual_m3: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 3), nullable=True)

    # Surface Deviation Analysis (in mm)
    surface_deviation_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3), nullable=True)
    surface_deviation_max: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3), nullable=True)
    surface_deviation_min: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3), nullable=True)
    surface_deviation_std: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3), nullable=True)

    # Completion Metrics
    completion_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        CheckConstraint("completion_percentage >= 0 AND completion_percentage <= 100"),
        nullable=True,
        index=True
    )
    points_within_tolerance_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Percentage of points within tolerance threshold"
    )
    tolerance_threshold_mm: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        server_default="50.00",
        comment="Tolerance threshold in mm (e.g., ±50mm)"
    )

    # Visualization Data
    heatmap_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment='{"points": [...], "colors": [...], "bounds": {...}}'
    )
    hotspots: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"location": [x,y,z], "deviation": 150.5, "severity": "high"}]'
    )

    # Processing Metadata
    processing_time_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    point_count_baseline: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    point_count_current: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    algorithm_version: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'v1.0'")
    )

    # Quality Metrics
    confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
        server_default="95.00"
    )
    quality_flags: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"flag": "incomplete_scan", "description": "..."}]'
    )

    # Metadata
    processed_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Service or user that ran comparison"
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    asset = relationship("HS2Asset", back_populates="point_cloud_comparisons")

    # Indexes
    __table_args__ = (
        Index("idx_pointcloud_asset_date", "asset_id", "comparison_date"),
        Index("idx_pointcloud_comparison_date", "comparison_date"),
        Index("idx_pointcloud_completion", "completion_percentage"),
    )

    def __repr__(self) -> str:
        return (
            f"<HS2PointCloudComparison(asset_id={self.asset_id}, "
            f"date={self.comparison_date}, completion={self.completion_percentage}%)>"
        )


class HS2ScheduleMilestone(Base):
    """
    Schedule milestone tracking.

    Manages project milestones with dependency tracking, critical path analysis,
    and schedule variance analysis. Used for Gantt charts and delay prediction.
    """

    __tablename__ = "hs2_schedule_milestones"

    # Foreign Keys
    asset_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("hs2_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Milestone Details
    milestone_name: Mapped[str] = mapped_column(String(255), nullable=False)
    milestone_code: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="e.g., VIA-001-FOUND"
    )
    milestone_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="foundation, structure, completion, inspection"
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Schedule Dates
    planned_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    baseline_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    forecast_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    early_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    late_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    early_finish: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    late_finish: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Duration
    planned_duration_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    actual_duration_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'not_started'"),
        index=True,
        comment="not_started, in_progress, completed, delayed, cancelled"
    )
    completion_percentage: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        CheckConstraint("completion_percentage >= 0 AND completion_percentage <= 100"),
        nullable=False,
        server_default="0.00"
    )

    # Dependencies
    predecessors: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"milestone_id": "uuid", "type": "finish_to_start", "lag_days": 0}]'
    )
    successors: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"milestone_id": "uuid", "type": "finish_to_start"}]'
    )

    # Critical Path Analysis
    is_critical_path: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
        index=True
    )
    float_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
        comment="Total float (slack)"
    )

    # Variance Analysis
    schedule_variance_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Actual vs planned (negative = delay)"
    )
    variance_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Risk and Issues
    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'low'"),
        comment="low, medium, high, critical"
    )
    risk_factors: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"factor": "weather", "impact": "2 days delay"}]'
    )

    # Resource Allocation
    assigned_contractor: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    estimated_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    actual_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)

    # Relationships
    asset = relationship("HS2Asset", back_populates="schedule_milestones")

    # Indexes
    __table_args__ = (
        Index("idx_milestone_asset", "asset_id"),
        Index("idx_milestone_dates", "planned_date", "actual_date"),
        Index("idx_milestone_status", "status"),
        Index("idx_milestone_critical_path", "is_critical_path", postgresql_where="is_critical_path = true"),
        Index("idx_milestone_type", "milestone_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<HS2ScheduleMilestone(asset_id={self.asset_id}, "
            f"name={self.milestone_name}, status={self.status})>"
        )
