"""
HS2 Pydantic Schemas
===================

Request and response models for HS2 Assurance Intelligence Demonstrator endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict


# ==================== Asset Schemas ====================

class AssetBase(BaseModel):
    """Base asset model."""
    asset_id: str = Field(..., description="Unique asset identifier", examples=["BR001-N1-V01"])
    asset_name: str = Field(..., description="Asset name", examples=["London-Birmingham Viaduct Section 1"])
    asset_type: str = Field(..., description="Type of asset", examples=["Viaduct", "Bridge", "Tunnel"])
    route_section: str = Field(..., description="HS2 route section", examples=["Phase 1", "Phase 2a", "Phase 2b"])
    contractor: str = Field(..., description="Contractor responsible", examples=["Balfour Beatty", "Mace", "EKFB JV"])
    design_status: Optional[str] = Field(None, description="Design approval status", examples=["Approved", "Pending", "In Progress"])
    construction_status: Optional[str] = Field(None, description="Construction status", examples=["Complete", "In Progress", "Not Started"])
    planned_completion_date: Optional[datetime] = Field(None, description="Planned completion date")


class AssetCreate(AssetBase):
    """Asset creation model."""
    pass


class AssetUpdate(BaseModel):
    """Asset update model - all fields optional."""
    asset_name: Optional[str] = None
    asset_type: Optional[str] = None
    route_section: Optional[str] = None
    contractor: Optional[str] = None
    design_status: Optional[str] = None
    construction_status: Optional[str] = None
    planned_completion_date: Optional[datetime] = None
    readiness_status: Optional[str] = None


class AssetReadinessSummary(BaseModel):
    """Asset readiness summary with evaluation results."""
    deliverables_submitted: int = Field(..., description="Number of deliverables submitted")
    deliverables_required: int = Field(..., description="Total deliverables required")
    deliverables_completion_pct: float = Field(..., description="Deliverables completion percentage")
    
    certificates_issued: int = Field(..., description="Number of certificates issued")
    certificates_required: int = Field(..., description="Total certificates required")
    certificates_completion_pct: float = Field(..., description="Certificates completion percentage")
    
    cost_variance_pct: float = Field(..., description="Cost variance percentage")
    schedule_variance_days: int = Field(..., description="Schedule variance in days")
    
    critical_risks: int = Field(..., description="Number of critical risks")
    major_risks: int = Field(..., description="Number of major risks")
    minor_risks: int = Field(..., description="Number of minor risks")
    
    overall_readiness: str = Field(..., description="Overall readiness status", examples=["Ready", "Not Ready", "At Risk"])
    taem_score: float = Field(..., description="TAEM evaluation score (0-100)")
    last_evaluation: Optional[datetime] = Field(None, description="Last evaluation timestamp")


class AssetResponse(AssetBase):
    """Asset response model with full details."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    readiness_status: str = Field(..., description="Current readiness status", examples=["Ready", "Not Ready", "At Risk"])
    taem_evaluation_score: Optional[float] = Field(None, description="Latest TAEM score")
    created_at: datetime
    updated_at: datetime


class AssetDetailResponse(AssetResponse):
    """Detailed asset response with readiness summary."""
    readiness_summary: Optional[AssetReadinessSummary] = None


# ==================== Deliverable Schemas ====================

class DeliverableBase(BaseModel):
    """Base deliverable model."""
    deliverable_type: str = Field(..., description="Type of deliverable", examples=["Design Report", "As-Built Drawing", "Safety Certificate"])
    deliverable_name: str = Field(..., description="Deliverable name")
    required_by_date: Optional[datetime] = Field(None, description="Required submission date")
    responsible_party: Optional[str] = Field(None, description="Party responsible for submission")


class DeliverableCreate(DeliverableBase):
    """Deliverable creation model."""
    asset_id: UUID = Field(..., description="Associated asset ID")


class DeliverableUpdate(BaseModel):
    """Deliverable update model."""
    deliverable_type: Optional[str] = None
    deliverable_name: Optional[str] = None
    status: Optional[str] = None
    submission_date: Optional[datetime] = None
    approval_status: Optional[str] = None
    required_by_date: Optional[datetime] = None
    responsible_party: Optional[str] = None
    notes: Optional[str] = None


class DeliverableResponse(DeliverableBase):
    """Deliverable response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    asset_id: UUID
    status: str = Field(..., description="Submission status", examples=["Submitted", "Pending", "Approved", "Rejected"])
    submission_date: Optional[datetime] = Field(None, description="Actual submission date")
    approval_status: Optional[str] = Field(None, description="Approval status")
    days_overdue: Optional[int] = Field(None, description="Days overdue (negative if ahead)")
    created_at: datetime
    updated_at: datetime


# ==================== Cost Tracking Schemas ====================

class CostResponse(BaseModel):
    """Cost tracking response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    asset_id: UUID
    budget_amount: Decimal = Field(..., description="Budgeted amount")
    actual_amount: Decimal = Field(..., description="Actual spent amount")
    forecast_amount: Decimal = Field(..., description="Forecasted final amount")
    variance_amount: Decimal = Field(..., description="Variance amount (actual - budget)")
    variance_pct: float = Field(..., description="Variance percentage")
    cost_category: Optional[str] = Field(None, description="Cost category", examples=["Labour", "Materials", "Equipment"])
    reporting_period: Optional[str] = Field(None, description="Reporting period")
    created_at: datetime
    updated_at: datetime


# ==================== Certificate Schemas ====================

class CertificateResponse(BaseModel):
    """Certificate response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    asset_id: UUID
    certificate_type: str = Field(..., description="Type of certificate", examples=["Safety", "Quality", "Environmental"])
    certificate_name: str = Field(..., description="Certificate name")
    issuing_authority: Optional[str] = Field(None, description="Authority that issued certificate")
    issue_date: Optional[datetime] = Field(None, description="Date issued")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    status: str = Field(..., description="Certificate status", examples=["Valid", "Expired", "Pending"])
    days_until_expiry: Optional[int] = Field(None, description="Days until expiry (negative if expired)")
    created_at: datetime
    updated_at: datetime


# ==================== TAEM Rule Schemas ====================

class TAEMRuleBase(BaseModel):
    """Base TAEM rule model."""
    rule_code: str = Field(..., description="Rule identifier code", examples=["TAEM-001", "TAEM-002"])
    rule_name: str = Field(..., description="Rule name", examples=["Deliverables Completeness Check"])
    rule_description: str = Field(..., description="Detailed rule description")
    rule_category: str = Field(..., description="Rule category", examples=["Deliverables", "Costs", "Certificates", "Schedule"])
    severity: str = Field(..., description="Rule severity", examples=["Critical", "Major", "Minor"])
    weight: float = Field(..., description="Rule weight in overall score (0-1)")


class TAEMRuleResponse(TAEMRuleBase):
    """TAEM rule response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_active: bool = Field(..., description="Whether rule is active")
    threshold_value: Optional[float] = Field(None, description="Threshold value for rule evaluation")
    created_at: datetime
    updated_at: datetime


class TAEMRuleUpdate(BaseModel):
    """TAEM rule update model for tinkerability."""
    rule_name: Optional[str] = None
    rule_description: Optional[str] = None
    severity: Optional[str] = None
    weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_active: Optional[bool] = None
    threshold_value: Optional[float] = None


# ==================== Rule Evaluation Schemas ====================

class RuleEvaluationResult(BaseModel):
    """Individual rule evaluation result."""
    rule_code: str = Field(..., description="Rule code")
    rule_name: str = Field(..., description="Rule name")
    status: str = Field(..., description="Evaluation status", examples=["Pass", "Fail", "Warning"])
    score: float = Field(..., description="Rule score (0-100)")
    weight: float = Field(..., description="Rule weight")
    weighted_score: float = Field(..., description="Weighted contribution to total score")
    message: str = Field(..., description="Evaluation message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional evaluation details")


class AssetEvaluationResponse(BaseModel):
    """Asset TAEM evaluation response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    asset_id: UUID
    evaluation_date: datetime = Field(..., description="Date of evaluation")
    overall_score: float = Field(..., description="Overall TAEM score (0-100)")
    readiness_status: str = Field(..., description="Determined readiness status", examples=["Ready", "Not Ready", "At Risk"])
    rules_evaluated: int = Field(..., description="Number of rules evaluated")
    rules_passed: int = Field(..., description="Number of rules passed")
    rules_failed: int = Field(..., description="Number of rules failed")
    rule_results: List[RuleEvaluationResult] = Field(default_factory=list, description="Individual rule results")
    created_at: datetime


class AssetEvaluationRequest(BaseModel):
    """Request to trigger asset evaluation."""
    force_refresh: bool = Field(default=False, description="Force re-evaluation even if recent evaluation exists")


# ==================== Dashboard Schemas ====================

class AssetTypeBreakdown(BaseModel):
    """Breakdown by asset type."""
    asset_type: str = Field(..., description="Asset type")
    total: int = Field(..., description="Total assets")
    ready: int = Field(..., description="Ready assets")
    not_ready: int = Field(..., description="Not ready assets")
    at_risk: int = Field(..., description="At risk assets")
    ready_pct: float = Field(..., description="Percentage ready")


class ContractorBreakdown(BaseModel):
    """Breakdown by contractor."""
    contractor: str = Field(..., description="Contractor name")
    total: int = Field(..., description="Total assets")
    ready: int = Field(..., description="Ready assets")
    not_ready: int = Field(..., description="Not ready assets")
    at_risk: int = Field(..., description="At risk assets")
    ready_pct: float = Field(..., description="Percentage ready")
    avg_taem_score: float = Field(..., description="Average TAEM score")


class RouteBreakdown(BaseModel):
    """Breakdown by route section."""
    route_section: str = Field(..., description="Route section")
    total: int = Field(..., description="Total assets")
    ready: int = Field(..., description="Ready assets")
    not_ready: int = Field(..., description="Not ready assets")
    at_risk: int = Field(..., description="At risk assets")
    ready_pct: float = Field(..., description="Percentage ready")


class DashboardSummary(BaseModel):
    """Overall dashboard summary."""
    total_assets: int = Field(..., description="Total number of assets")
    ready: int = Field(..., description="Ready assets")
    not_ready: int = Field(..., description="Not ready assets")
    at_risk: int = Field(..., description="At risk assets")
    ready_pct: float = Field(..., description="Percentage ready")
    not_ready_pct: float = Field(..., description="Percentage not ready")
    at_risk_pct: float = Field(..., description="Percentage at risk")
    avg_taem_score: float = Field(..., description="Average TAEM score across all assets")
    last_updated: datetime = Field(..., description="Last dashboard update")
    
    by_asset_type: List[AssetTypeBreakdown] = Field(default_factory=list, description="Breakdown by asset type")
    by_contractor: List[ContractorBreakdown] = Field(default_factory=list, description="Breakdown by contractor")
    by_route: List[RouteBreakdown] = Field(default_factory=list, description="Breakdown by route section")


# ==================== Pagination Schemas ====================

class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    total: int = Field(..., description="Total number of items")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Number of items per page")
    items: List[Any] = Field(..., description="List of items")


class AssetPaginatedResponse(BaseModel):
    """Paginated asset response."""
    total: int = Field(..., description="Total number of assets")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Number of items per page")
    items: List[AssetResponse] = Field(..., description="List of assets")


class DeliverablePaginatedResponse(BaseModel):
    """Paginated deliverable response."""
    total: int = Field(..., description="Total number of deliverables")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Number of items per page")
    items: List[DeliverableResponse] = Field(..., description="List of deliverables")


class EvaluationPaginatedResponse(BaseModel):
    """Paginated evaluation response."""
    total: int = Field(..., description="Total number of evaluations")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Number of items per page")
    items: List[AssetEvaluationResponse] = Field(..., description="List of evaluations")
