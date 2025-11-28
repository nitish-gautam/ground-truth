"""
HS2 Progress Verification Pydantic Schemas
==========================================

Request and response models for Progress Verification, Point Cloud Comparison,
and Schedule Milestone endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict


# ==================== Progress Snapshot Schemas ====================

class ProgressSnapshotBase(BaseModel):
    """Base progress snapshot model."""
    asset_id: UUID = Field(..., description="Asset UUID")
    snapshot_date: datetime = Field(..., description="Snapshot timestamp")

    # Progress percentages
    physical_progress_pct: Optional[Decimal] = Field(None, ge=0, le=100, description="Physical progress %")
    cost_progress_pct: Optional[Decimal] = Field(None, ge=0, le=100, description="Cost progress %")
    schedule_progress_pct: Optional[Decimal] = Field(None, ge=0, le=100, description="Schedule progress %")

    # Earned Value Management
    planned_value: Optional[Decimal] = Field(None, description="Planned Value (PV)")
    earned_value: Optional[Decimal] = Field(None, description="Earned Value (EV)")
    actual_cost: Optional[Decimal] = Field(None, description="Actual Cost (AC)")

    # Point cloud data
    point_cloud_file_path: Optional[str] = Field(None, description="Path to point cloud file")
    deviation_score: Optional[Decimal] = Field(None, description="Average deviation in mm")

    # Anomalies
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")

    # Metadata
    data_source: str = Field(default="manual", description="Data source")
    confidence_score: Decimal = Field(default=Decimal("100.00"), ge=0, le=100, description="Confidence score")
    notes: Optional[str] = Field(None, description="Additional notes")


class ProgressSnapshotCreate(ProgressSnapshotBase):
    """Progress snapshot creation model."""
    pass


class ProgressSnapshotUpdate(BaseModel):
    """Progress snapshot update model - all fields optional."""
    physical_progress_pct: Optional[Decimal] = Field(None, ge=0, le=100)
    cost_progress_pct: Optional[Decimal] = Field(None, ge=0, le=100)
    schedule_progress_pct: Optional[Decimal] = Field(None, ge=0, le=100)
    planned_value: Optional[Decimal] = None
    earned_value: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    point_cloud_file_path: Optional[str] = None
    deviation_score: Optional[Decimal] = None
    anomalies: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None


class ProgressSnapshotResponse(ProgressSnapshotBase):
    """Progress snapshot response model with full details."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    cost_variance: Optional[Decimal] = Field(None, description="Cost Variance (CV = EV - AC)")
    schedule_variance: Optional[Decimal] = Field(None, description="Schedule Variance (SV = EV - PV)")
    cost_performance_index: Optional[Decimal] = Field(None, description="CPI = EV / AC")
    schedule_performance_index: Optional[Decimal] = Field(None, description="SPI = EV / PV")
    created_at: datetime
    updated_at: datetime


class ProgressTimelineResponse(BaseModel):
    """Progress timeline with multiple snapshots."""
    asset_id: UUID
    snapshots: List[ProgressSnapshotResponse]
    metrics: Dict[str, Any] = Field(
        ...,
        description="Aggregated metrics",
        examples=[{
            "avg_physical_progress": 75.5,
            "avg_cost_performance_index": 1.05,
            "trend": "improving"
        }]
    )


# ==================== Point Cloud Comparison Schemas ====================

class PointCloudComparisonBase(BaseModel):
    """Base point cloud comparison model."""
    asset_id: UUID = Field(..., description="Asset UUID")
    baseline_file_path: str = Field(..., description="BIM model or baseline scan path")
    current_file_path: str = Field(..., description="Current site scan (LAS/LAZ) path")
    comparison_date: datetime = Field(..., description="Comparison timestamp")

    # Volume analysis
    volume_difference_m3: Optional[Decimal] = Field(None, description="Volume difference in m³")
    volume_planned_m3: Optional[Decimal] = Field(None, description="Planned volume in m³")
    volume_actual_m3: Optional[Decimal] = Field(None, description="Actual volume in m³")

    # Surface deviation (mm)
    surface_deviation_avg: Optional[Decimal] = Field(None, description="Average deviation in mm")
    surface_deviation_max: Optional[Decimal] = Field(None, description="Max deviation in mm")
    surface_deviation_min: Optional[Decimal] = Field(None, description="Min deviation in mm")
    surface_deviation_std: Optional[Decimal] = Field(None, description="Std deviation in mm")

    # Completion metrics
    completion_percentage: Optional[Decimal] = Field(None, ge=0, le=100, description="Completion %")
    points_within_tolerance_pct: Optional[Decimal] = Field(None, ge=0, le=100, description="% within tolerance")
    tolerance_threshold_mm: Decimal = Field(default=Decimal("50.00"), description="Tolerance threshold in mm")

    # Visualization
    heatmap_data: Optional[Dict[str, Any]] = Field(None, description="Heatmap visualization data")
    hotspots: List[Dict[str, Any]] = Field(default_factory=list, description="High-deviation hotspots")

    # Processing metadata
    processing_time_seconds: Optional[int] = Field(None, description="Processing time")
    point_count_baseline: Optional[int] = Field(None, description="Baseline point count")
    point_count_current: Optional[int] = Field(None, description="Current point count")

    # Quality
    confidence_score: Decimal = Field(default=Decimal("95.00"), ge=0, le=100, description="Confidence score")
    quality_flags: List[Dict[str, Any]] = Field(default_factory=list, description="Quality flags")

    notes: Optional[str] = Field(None, description="Additional notes")


class PointCloudComparisonCreate(PointCloudComparisonBase):
    """Point cloud comparison creation model."""
    pass


class PointCloudComparisonResponse(PointCloudComparisonBase):
    """Point cloud comparison response model."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    algorithm_version: str = Field(..., description="Algorithm version")
    processed_by: Optional[str] = Field(None, description="Processing service/user")
    created_at: datetime
    updated_at: datetime


class PointCloudComparisonListResponse(BaseModel):
    """List of point cloud comparisons for an asset."""
    asset_id: UUID
    comparisons: List[PointCloudComparisonResponse]
    total_count: int


# ==================== Schedule Milestone Schemas ====================

class ScheduleMilestoneBase(BaseModel):
    """Base schedule milestone model."""
    asset_id: UUID = Field(..., description="Asset UUID")
    milestone_name: str = Field(..., description="Milestone name", max_length=255)
    milestone_code: Optional[str] = Field(None, description="Milestone code", max_length=50)
    milestone_type: Optional[str] = Field(None, description="Type: foundation, structure, completion")
    description: Optional[str] = Field(None, description="Milestone description")

    # Schedule dates
    planned_date: Optional[datetime] = Field(None, description="Original planned date")
    baseline_date: Optional[datetime] = Field(None, description="Baseline date after changes")
    forecast_date: Optional[datetime] = Field(None, description="ML-predicted completion date")
    actual_date: Optional[datetime] = Field(None, description="Actual completion date")

    # Duration
    planned_duration_days: Optional[int] = Field(None, ge=0, description="Planned duration in days")
    actual_duration_days: Optional[int] = Field(None, ge=0, description="Actual duration in days")

    # Status
    status: str = Field(default="not_started", description="Status")
    completion_percentage: Decimal = Field(default=Decimal("0.00"), ge=0, le=100, description="Completion %")

    # Dependencies
    predecessors: List[Dict[str, Any]] = Field(default_factory=list, description="Predecessor milestones")
    successors: List[Dict[str, Any]] = Field(default_factory=list, description="Successor milestones")

    # Critical path
    is_critical_path: bool = Field(default=False, description="On critical path?")
    float_days: int = Field(default=0, description="Total float (slack) in days")

    # Variance
    schedule_variance_days: Optional[int] = Field(None, description="Schedule variance (negative = delay)")
    variance_reason: Optional[str] = Field(None, description="Reason for variance")

    # Risk
    risk_level: str = Field(default="low", description="Risk level: low, medium, high, critical")
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Risk factors")

    # Resources
    assigned_contractor: Optional[str] = Field(None, max_length=255, description="Assigned contractor")
    estimated_cost: Optional[Decimal] = Field(None, description="Estimated cost")
    actual_cost: Optional[Decimal] = Field(None, description="Actual cost")


class ScheduleMilestoneCreate(ScheduleMilestoneBase):
    """Schedule milestone creation model."""
    pass


class ScheduleMilestoneUpdate(BaseModel):
    """Schedule milestone update model - all fields optional."""
    milestone_name: Optional[str] = Field(None, max_length=255)
    milestone_code: Optional[str] = Field(None, max_length=50)
    milestone_type: Optional[str] = None
    description: Optional[str] = None
    planned_date: Optional[datetime] = None
    baseline_date: Optional[datetime] = None
    forecast_date: Optional[datetime] = None
    actual_date: Optional[datetime] = None
    planned_duration_days: Optional[int] = Field(None, ge=0)
    actual_duration_days: Optional[int] = Field(None, ge=0)
    status: Optional[str] = None
    completion_percentage: Optional[Decimal] = Field(None, ge=0, le=100)
    predecessors: Optional[List[Dict[str, Any]]] = None
    successors: Optional[List[Dict[str, Any]]] = None
    is_critical_path: Optional[bool] = None
    float_days: Optional[int] = None
    schedule_variance_days: Optional[int] = None
    variance_reason: Optional[str] = None
    risk_level: Optional[str] = None
    risk_factors: Optional[List[Dict[str, Any]]] = None
    assigned_contractor: Optional[str] = Field(None, max_length=255)
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None


class ScheduleMilestoneResponse(ScheduleMilestoneBase):
    """Schedule milestone response model."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    early_start: Optional[datetime] = Field(None, description="Earliest start date")
    late_start: Optional[datetime] = Field(None, description="Latest start date")
    early_finish: Optional[datetime] = Field(None, description="Earliest finish date")
    late_finish: Optional[datetime] = Field(None, description="Latest finish date")
    created_at: datetime
    updated_at: datetime


class GanttChartResponse(BaseModel):
    """Gantt chart data for an asset."""
    asset_id: UUID
    milestones: List[ScheduleMilestoneResponse]
    critical_path: List[UUID] = Field(..., description="List of milestone IDs on critical path")
    project_start: Optional[datetime]
    project_end: Optional[datetime]
    total_duration_days: Optional[int]
    metrics: Dict[str, Any] = Field(
        ...,
        description="Schedule metrics",
        examples=[{
            "on_time_milestones": 15,
            "delayed_milestones": 3,
            "avg_delay_days": 5
        }]
    )


# ==================== Earned Value Management Schemas ====================

class EarnedValueAnalysis(BaseModel):
    """Earned Value Management analysis for an asset."""
    asset_id: UUID
    analysis_date: datetime

    # EVM Metrics
    budget_at_completion: Decimal = Field(..., description="BAC - Total budget")
    planned_value: Decimal = Field(..., description="PV - Budgeted cost of work scheduled")
    earned_value: Decimal = Field(..., description="EV - Budgeted cost of work performed")
    actual_cost: Decimal = Field(..., description="AC - Actual cost of work performed")

    # Variances
    cost_variance: Decimal = Field(..., description="CV = EV - AC")
    schedule_variance: Decimal = Field(..., description="SV = EV - PV")

    # Performance Indices
    cost_performance_index: Decimal = Field(..., description="CPI = EV / AC")
    schedule_performance_index: Decimal = Field(..., description="SPI = EV / PV")

    # Forecasts
    estimate_at_completion: Decimal = Field(..., description="EAC = BAC / CPI")
    estimate_to_complete: Decimal = Field(..., description="ETC = EAC - AC")
    variance_at_completion: Decimal = Field(..., description="VAC = BAC - EAC")

    # Completion forecast
    percent_complete: Decimal = Field(..., ge=0, le=100, description="% Complete")
    forecast_completion_date: Optional[datetime] = Field(None, description="Forecasted completion")

    # Status
    performance_status: str = Field(
        ...,
        description="Overall performance status",
        examples=["On Track", "Over Budget", "Behind Schedule", "Critical"]
    )


# ==================== Cost-Progress Alignment Schemas ====================

class CostProgressAlignment(BaseModel):
    """Cost-progress alignment analysis."""
    asset_id: UUID
    analysis_date: datetime

    # Progress metrics
    physical_progress_pct: Decimal = Field(..., ge=0, le=100, description="Physical progress %")
    cost_progress_pct: Decimal = Field(..., ge=0, le=100, description="Cost progress %")
    schedule_progress_pct: Decimal = Field(..., ge=0, le=100, description="Schedule progress %")

    # Financial metrics
    budget_allocated: Decimal
    budget_spent: Decimal
    budget_remaining: Decimal
    burn_rate: Decimal = Field(..., description="Average daily spend")

    # Alignment score
    alignment_score: Decimal = Field(..., ge=0, le=100, description="Alignment score (100 = perfect)")
    alignment_status: str = Field(..., examples=["Aligned", "Overspending", "Underspending", "Critical"])

    # Anomalies
    anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected misalignments",
        examples=[[
            {"type": "cost_overrun", "severity": "high", "description": "40% over budget with only 30% progress"},
            {"type": "underspending", "severity": "medium", "description": "20% spent but 50% progress claimed"}
        ]]
    )


# ==================== Progress Summary Schemas ====================

class ProgressVerificationSummary(BaseModel):
    """Comprehensive progress verification summary for an asset."""
    asset_id: UUID
    asset_name: str
    asset_type: str

    # Latest snapshot
    latest_snapshot: Optional[ProgressSnapshotResponse] = None

    # Latest comparison
    latest_comparison: Optional[PointCloudComparisonResponse] = None

    # Schedule summary
    total_milestones: int
    completed_milestones: int
    in_progress_milestones: int
    delayed_milestones: int

    # Earned Value
    earned_value_analysis: Optional[EarnedValueAnalysis] = None

    # Cost-Progress
    cost_progress_alignment: Optional[CostProgressAlignment] = None

    # Overall status
    overall_status: str = Field(..., examples=["On Track", "At Risk", "Critical"])
    confidence_score: Decimal = Field(..., ge=0, le=100, description="Overall confidence")
    last_updated: datetime


# ==================== Alert Schemas ====================

class ProgressAlert(BaseModel):
    """Progress verification alert."""
    alert_id: UUID
    asset_id: UUID
    alert_type: str = Field(..., examples=["cost_overrun", "schedule_delay", "quality_issue"])
    severity: str = Field(..., examples=["low", "medium", "high", "critical"])
    title: str = Field(..., max_length=255)
    description: str
    detected_at: datetime
    source: str = Field(..., examples=["point_cloud_comparison", "earned_value", "milestone_tracking"])
    data: Dict[str, Any] = Field(..., description="Alert-specific data")
    status: str = Field(default="active", examples=["active", "acknowledged", "resolved"])
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class ProgressAlertListResponse(BaseModel):
    """List of progress alerts."""
    alerts: List[ProgressAlert]
    total_count: int
    active_count: int
    critical_count: int
