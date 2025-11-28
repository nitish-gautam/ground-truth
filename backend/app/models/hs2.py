"""
HS2 Assurance Intelligence Database Models
==========================================

SQLAlchemy models for HS2 infrastructure assets, deliverables, costs,
certificates, TAEM rules, and evaluation results.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Boolean, Text,
    ForeignKey, Numeric, JSON, Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from app.models.base import Base


# ==================== HS2 Asset Model ====================

class HS2Asset(Base):
    """
    HS2 Infrastructure Asset Model
    
    Represents physical infrastructure assets (viaducts, bridges, tunnels, OLE masts)
    with readiness tracking and TAEM evaluation scores.
    """
    __tablename__ = "hs2_assets"
    
    # Asset identification
    asset_id = Column(String(50), unique=True, nullable=False, index=True, 
                     comment="Unique asset identifier (e.g., VA-001, BR-001)")
    asset_name = Column(String(255), nullable=False, 
                       comment="Human-readable asset name")
    asset_type = Column(String(50), nullable=False, index=True,
                       comment="Type: Viaduct, Bridge, Tunnel, OLE Mast")
    
    # Asset location and assignment
    route_section = Column(String(100), nullable=False, index=True,
                          comment="HS2 route section (e.g., London-Euston)")
    contractor = Column(String(100), nullable=False, index=True,
                       comment="Responsible contractor")
    location_text = Column(Text, nullable=True,
                          comment="Human-readable location description")
    
    # Status tracking
    design_status = Column(String(50), nullable=True,
                          comment="Design approval status")
    construction_status = Column(String(50), nullable=True,
                               comment="Construction status")
    readiness_status = Column(String(50), nullable=False, default="Not Ready", index=True,
                             comment="Overall readiness: Ready, Not Ready, At Risk")
    
    # Dates
    planned_completion_date = Column(DateTime, nullable=True,
                                    comment="Target completion date")
    
    # TAEM evaluation
    taem_evaluation_score = Column(Float, nullable=True,
                                   comment="Latest TAEM score (0-100)")
    last_evaluation_date = Column(DateTime, nullable=True,
                                 comment="Timestamp of last TAEM evaluation")
    
    # Metadata (flexible JSON for asset-specific attributes)
    asset_metadata = Column(JSON, nullable=True, default=dict,
                     comment="Asset-specific metadata (height, span, etc.)")
    
    # Relationships
    deliverables = relationship("HS2Deliverable", back_populates="asset", cascade="all, delete-orphan")
    costs = relationship("HS2Cost", back_populates="asset", cascade="all, delete-orphan")
    certificates = relationship("HS2Certificate", back_populates="asset", cascade="all, delete-orphan")
    evaluations = relationship("HS2Evaluation", back_populates="asset", cascade="all, delete-orphan")

    # Progress Verification Relationships
    progress_snapshots = relationship("HS2ProgressSnapshot", back_populates="asset", cascade="all, delete-orphan")
    point_cloud_comparisons = relationship("HS2PointCloudComparison", back_populates="asset", cascade="all, delete-orphan")
    schedule_milestones = relationship("HS2ScheduleMilestone", back_populates="asset", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_hs2_assets_type_status", "asset_type", "readiness_status"),
        Index("idx_hs2_assets_contractor_status", "contractor", "readiness_status"),
        CheckConstraint("readiness_status IN ('Ready', 'Not Ready', 'At Risk')", name="check_readiness_status"),
    )
    
    def __repr__(self):
        return f"<HS2Asset {self.asset_id}: {self.asset_name} ({self.readiness_status})>"


# ==================== Deliverable Model ====================

class HS2Deliverable(Base):
    """
    HS2 Deliverable Tracking Model
    
    Tracks submission and approval status of critical project deliverables
    (design certificates, assurance sign-offs, test reports, etc.).
    """
    __tablename__ = "hs2_deliverables"
    
    # Foreign keys
    asset_id = Column(PGUUID(as_uuid=True), ForeignKey("hs2_assets.id", ondelete="CASCADE"), 
                     nullable=False, index=True)
    
    # Deliverable identification
    deliverable_id = Column(String(50), unique=True, nullable=False, index=True,
                           comment="Unique deliverable ID (e.g., DEL-VA-001-01)")
    deliverable_name = Column(String(255), nullable=False,
                             comment="Deliverable name")
    deliverable_type = Column(String(100), nullable=False, index=True,
                             comment="Type: Design Certificate, Test Report, etc.")
    
    # Status tracking
    status = Column(String(50), nullable=False, default="Not Started", index=True,
                   comment="Status: Submitted, Pending, Overdue, Not Started")
    approval_status = Column(String(50), nullable=True,
                            comment="Approval status: Approved, Rejected, Under Review")
    
    # Dates
    due_date = Column(DateTime, nullable=True,
                     comment="Required submission date")
    submission_date = Column(DateTime, nullable=True,
                           comment="Actual submission date")
    approval_date = Column(DateTime, nullable=True,
                         comment="Date of approval/rejection")
    
    # Responsibility and tracking
    responsible_party = Column(String(100), nullable=True,
                              comment="Party responsible for delivery")
    document_reference = Column(String(100), nullable=True,
                               comment="Document management system reference")
    days_overdue = Column(Integer, nullable=True,
                         comment="Calculated days overdue (negative if ahead)")
    
    # Priority and notes
    priority = Column(String(20), nullable=False, default="Minor",
                     comment="Priority: Critical, Major, Minor")
    notes = Column(Text, nullable=True,
                  comment="Additional notes or comments")
    
    # Relationship
    asset = relationship("HS2Asset", back_populates="deliverables")
    
    # Indexes
    __table_args__ = (
        Index("idx_hs2_deliverables_asset_status", "asset_id", "status"),
        Index("idx_hs2_deliverables_type_priority", "deliverable_type", "priority"),
        CheckConstraint("status IN ('Submitted', 'Pending', 'Overdue', 'Not Started', 'Approved')", 
                       name="check_deliverable_status"),
        CheckConstraint("priority IN ('Critical', 'Major', 'Minor')", name="check_priority"),
    )
    
    def __repr__(self):
        return f"<HS2Deliverable {self.deliverable_id}: {self.deliverable_name} ({self.status})>"


# ==================== Cost Tracking Model ====================

class HS2Cost(Base):
    """
    HS2 Cost Tracking Model
    
    Tracks budget, actual spend, and cost variance for assets.
    """
    __tablename__ = "hs2_costs"
    
    # Foreign keys
    asset_id = Column(PGUUID(as_uuid=True), ForeignKey("hs2_assets.id", ondelete="CASCADE"), 
                     nullable=False, index=True)
    
    # Cost identification
    cost_line_id = Column(String(50), unique=True, nullable=False, index=True,
                         comment="Unique cost line ID (e.g., COST-VA-001)")
    
    # Financial data (using Numeric for precise decimal handling)
    budget_amount = Column(Numeric(15, 2), nullable=False,
                          comment="Budgeted amount (£)")
    actual_amount = Column(Numeric(15, 2), nullable=False, default=0,
                         comment="Actual spent amount (£)")
    forecast_amount = Column(Numeric(15, 2), nullable=True,
                           comment="Forecasted final amount (£)")
    
    # Variance calculations
    variance_amount = Column(Numeric(15, 2), nullable=True,
                            comment="Variance amount (actual - budget)")
    variance_pct = Column(Float, nullable=True,
                         comment="Variance percentage")
    
    # Categorization
    cost_category = Column(String(50), nullable=True, index=True,
                          comment="Category: Labour, Materials, Equipment, etc.")
    reporting_period = Column(String(50), nullable=True,
                             comment="Reporting period (e.g., 2024-Q1)")
    
    # Status
    status = Column(String(50), nullable=False, default="On Budget",
                   comment="Status: On Budget, Over Budget, Under Budget")
    
    # Notes
    notes = Column(Text, nullable=True,
                  comment="Cost variance explanation or notes")
    
    # Relationship
    asset = relationship("HS2Asset", back_populates="costs")
    
    # Indexes
    __table_args__ = (
        Index("idx_hs2_costs_asset_status", "asset_id", "status"),
        CheckConstraint("status IN ('On Budget', 'Over Budget', 'Under Budget', 'At Risk')", 
                       name="check_cost_status"),
    )
    
    def __repr__(self):
        return f"<HS2Cost {self.cost_line_id}: Budget £{self.budget_amount}, Actual £{self.actual_amount}>"


# ==================== Certificate Model ====================

class HS2Certificate(Base):
    """
    HS2 Certificate Model
    
    Tracks certificates, qualifications, and approvals with expiry dates.
    """
    __tablename__ = "hs2_certificates"
    
    # Foreign keys
    asset_id = Column(PGUUID(as_uuid=True), ForeignKey("hs2_assets.id", ondelete="CASCADE"), 
                     nullable=False, index=True)
    
    # Certificate identification
    certificate_id = Column(String(50), unique=True, nullable=False, index=True,
                           comment="Unique certificate ID (e.g., CERT-VA-001-01)")
    certificate_name = Column(String(255), nullable=False,
                             comment="Certificate name")
    certificate_type = Column(String(100), nullable=False, index=True,
                             comment="Type: Design, Welding, Concrete, NDT, QA")
    
    # Issuing information
    issuing_authority = Column(String(150), nullable=True,
                              comment="Authority that issued the certificate")
    issue_date = Column(DateTime, nullable=True,
                       comment="Date certificate was issued")
    expiry_date = Column(DateTime, nullable=True,
                        comment="Certificate expiry date")
    
    # Status
    status = Column(String(50), nullable=False, default="Valid", index=True,
                   comment="Status: Valid, Expired, Expiring Soon, Pending")
    days_until_expiry = Column(Integer, nullable=True,
                              comment="Days until expiry (negative if expired)")
    
    # Document reference
    document_reference = Column(String(100), nullable=True,
                               comment="Document management system reference")
    
    # Notes
    notes = Column(Text, nullable=True,
                  comment="Additional notes")
    
    # Relationship
    asset = relationship("HS2Asset", back_populates="certificates")
    
    # Indexes
    __table_args__ = (
        Index("idx_hs2_certificates_asset_status", "asset_id", "status"),
        Index("idx_hs2_certificates_expiry", "expiry_date"),
        CheckConstraint("status IN ('Valid', 'Expired', 'Expiring Soon', 'Pending')", 
                       name="check_certificate_status"),
    )
    
    def __repr__(self):
        return f"<HS2Certificate {self.certificate_id}: {self.certificate_name} ({self.status})>"


# ==================== TAEM Rule Model ====================

class HS2Rule(Base):
    """
    TAEM Rule Model
    
    Defines evaluation rules for the Technical Assurance and Evaluation Model.
    Supports tinkerability - rules can be enabled/disabled and weights adjusted.
    """
    __tablename__ = "hs2_taem_rules"
    
    # Rule identification
    rule_code = Column(String(50), unique=True, nullable=False, index=True,
                      comment="Unique rule code (e.g., TAEM-001)")
    rule_name = Column(String(255), nullable=False,
                      comment="Human-readable rule name")
    rule_description = Column(Text, nullable=False,
                            comment="Detailed rule description")
    
    # Rule categorization
    rule_category = Column(String(50), nullable=False, index=True,
                          comment="Category: Deliverables, Costs, Certificates, Schedule")
    severity = Column(String(20), nullable=False,
                     comment="Severity: Critical, Major, Minor")
    
    # Rule evaluation parameters
    weight = Column(Float, nullable=False, default=1.0,
                   comment="Rule weight in overall score (0-1)")
    threshold_value = Column(Float, nullable=True,
                            comment="Threshold value for pass/fail")
    
    # Rule implementation (JSON for flexible rule logic)
    rule_logic = Column(JSON, nullable=True, default=dict,
                       comment="Rule evaluation logic configuration")
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True,
                      comment="Whether rule is active in evaluations")
    
    # Indexes
    __table_args__ = (
        Index("idx_hs2_rules_category_active", "rule_category", "is_active"),
        CheckConstraint("severity IN ('Critical', 'Major', 'Minor')", name="check_severity"),
        CheckConstraint("weight >= 0 AND weight <= 1", name="check_weight_range"),
    )
    
    def __repr__(self):
        return f"<HS2Rule {self.rule_code}: {self.rule_name} (Weight: {self.weight})>"


# ==================== Evaluation Results Model ====================

class HS2Evaluation(Base):
    """
    HS2 Evaluation Results Model
    
    Stores historical TAEM evaluation results for audit trail and trend analysis.
    """
    __tablename__ = "hs2_evaluations"
    
    # Foreign keys
    asset_id = Column(PGUUID(as_uuid=True), ForeignKey("hs2_assets.id", ondelete="CASCADE"), 
                     nullable=False, index=True)
    
    # Evaluation metadata
    evaluation_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True,
                           comment="Timestamp of evaluation")
    
    # Overall results
    overall_score = Column(Float, nullable=False,
                          comment="Overall TAEM score (0-100)")
    readiness_status = Column(String(50), nullable=False,
                             comment="Determined readiness: Ready, Not Ready, At Risk")
    
    # Rule statistics
    rules_evaluated = Column(Integer, nullable=False, default=0,
                           comment="Total number of rules evaluated")
    rules_passed = Column(Integer, nullable=False, default=0,
                        comment="Number of rules passed")
    rules_failed = Column(Integer, nullable=False, default=0,
                        comment="Number of rules failed")
    
    # Detailed results (JSON array of rule evaluation results)
    rule_results = Column(JSON, nullable=True, default=list,
                         comment="Detailed results for each rule")
    
    # Evaluation context
    evaluation_trigger = Column(String(50), nullable=True,
                               comment="What triggered evaluation: Manual, Scheduled, Data Update")
    evaluator = Column(String(100), nullable=True,
                      comment="User or system that triggered evaluation")
    
    # Relationship
    asset = relationship("HS2Asset", back_populates="evaluations")
    
    # Indexes
    __table_args__ = (
        Index("idx_hs2_evaluations_asset_date", "asset_id", "evaluation_date"),
        Index("idx_hs2_evaluations_status", "readiness_status"),
        CheckConstraint("overall_score >= 0 AND overall_score <= 100", name="check_score_range"),
        CheckConstraint("readiness_status IN ('Ready', 'Not Ready', 'At Risk')", 
                       name="check_eval_readiness_status"),
    )
    
    def __repr__(self):
        return f"<HS2Evaluation {self.id}: Asset {self.asset_id} - Score {self.overall_score} ({self.readiness_status})>"
