"""
Utility infrastructure models
============================

Database models for utility records, materials, disciplines, and
infrastructure characteristics based on Twente dataset structure.
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


class UtilityDiscipline(BaseModel):
    """Utility discipline/service type classification."""

    __tablename__ = "utility_disciplines"

    # Discipline identification
    discipline_code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    discipline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    discipline_category: Mapped[str] = mapped_column(String(100), nullable=False)

    # Description and standards
    description: Mapped[Optional[str]] = mapped_column(Text)
    industry_standards: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    regulatory_body: Mapped[Optional[str]] = mapped_column(String(255))

    # GPR detection characteristics
    typical_depth_range_min_m: Mapped[Optional[float]] = mapped_column(Float)
    typical_depth_range_max_m: Mapped[Optional[float]] = mapped_column(Float)
    typical_diameter_range_min_m: Mapped[Optional[float]] = mapped_column(Float)
    typical_diameter_range_max_m: Mapped[Optional[float]] = mapped_column(Float)

    # Detection difficulty factors
    detection_difficulty_score: Mapped[Optional[float]] = mapped_column(Float)
    common_materials: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    signal_characteristics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Safety and regulatory
    safety_criticality: Mapped[Optional[str]] = mapped_column(String(50))
    strike_risk_level: Mapped[Optional[str]] = mapped_column(String(50))
    emergency_contact_required: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    utility_records: Mapped[List["UtilityRecord"]] = relationship(
        "UtilityRecord",
        back_populates="discipline",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "discipline_category IN ('utilities', 'telecommunications', 'energy', 'water', 'waste', 'transportation')",
            name="check_discipline_category"
        ),
        CheckConstraint(
            "typical_depth_range_min_m >= 0 OR typical_depth_range_min_m IS NULL",
            name="check_depth_min"
        ),
        CheckConstraint(
            "typical_depth_range_max_m >= typical_depth_range_min_m OR typical_depth_range_max_m IS NULL",
            name="check_depth_max"
        ),
        CheckConstraint(
            "detection_difficulty_score >= 0 AND detection_difficulty_score <= 1 OR detection_difficulty_score IS NULL",
            name="check_difficulty_score"
        ),
        CheckConstraint(
            "safety_criticality IN ('low', 'medium', 'high', 'critical') OR safety_criticality IS NULL",
            name="check_safety_criticality"
        ),
        Index("idx_utility_disciplines_code", "discipline_code"),
        Index("idx_utility_disciplines_category", "discipline_category"),
    )


class UtilityMaterial(BaseModel):
    """Utility material types and their GPR signature characteristics."""

    __tablename__ = "utility_materials"

    # Material identification
    material_code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    material_name: Mapped[str] = mapped_column(String(255), nullable=False)
    material_category: Mapped[str] = mapped_column(String(100), nullable=False)

    # Physical properties
    density_kg_m3: Mapped[Optional[float]] = mapped_column(Float)
    electrical_conductivity: Mapped[Optional[float]] = mapped_column(Float)
    magnetic_permeability: Mapped[Optional[float]] = mapped_column(Float)
    dielectric_constant: Mapped[Optional[float]] = mapped_column(Float)

    # GPR signature characteristics
    reflection_coefficient: Mapped[Optional[float]] = mapped_column(Float)
    signal_attenuation_db_m: Mapped[Optional[float]] = mapped_column(Float)
    typical_signal_amplitude: Mapped[Optional[float]] = mapped_column(Float)
    characteristic_frequency_mhz: Mapped[Optional[float]] = mapped_column(Float)

    # Detection properties
    detection_ease_score: Mapped[Optional[float]] = mapped_column(Float)
    minimum_detectable_diameter_m: Mapped[Optional[float]] = mapped_column(Float)
    optimal_frequency_range: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))

    # Environmental sensitivity
    corrosion_resistance: Mapped[Optional[str]] = mapped_column(String(50))
    temperature_sensitivity: Mapped[Optional[str]] = mapped_column(String(50))
    moisture_sensitivity: Mapped[Optional[str]] = mapped_column(String(50))

    # Installation and lifecycle
    typical_lifespan_years: Mapped[Optional[int]] = mapped_column(Integer)
    installation_method: Mapped[Optional[str]] = mapped_column(String(100))
    maintenance_frequency_years: Mapped[Optional[float]] = mapped_column(Float)

    # Material properties affecting GPR
    gpr_properties: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    calibration_factors: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)

    # Relationships
    utility_records: Mapped[List["UtilityRecord"]] = relationship(
        "UtilityRecord",
        back_populates="material",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "material_category IN ('metal', 'plastic', 'concrete', 'ceramic', 'composite', 'fiber')",
            name="check_material_category"
        ),
        CheckConstraint(
            "density_kg_m3 > 0 OR density_kg_m3 IS NULL",
            name="check_density_positive"
        ),
        CheckConstraint(
            "electrical_conductivity >= 0 OR electrical_conductivity IS NULL",
            name="check_conductivity"
        ),
        CheckConstraint(
            "dielectric_constant >= 1 OR dielectric_constant IS NULL",
            name="check_dielectric"
        ),
        CheckConstraint(
            "detection_ease_score >= 0 AND detection_ease_score <= 1 OR detection_ease_score IS NULL",
            name="check_detection_ease"
        ),
        CheckConstraint(
            "minimum_detectable_diameter_m > 0 OR minimum_detectable_diameter_m IS NULL",
            name="check_min_diameter"
        ),
        Index("idx_utility_materials_code", "material_code"),
        Index("idx_utility_materials_category", "material_category"),
    )


class UtilityRecord(BaseModel):
    """Individual utility records with comprehensive characteristics."""

    __tablename__ = "utility_records"

    # Basic identification
    utility_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    utility_name: Mapped[Optional[str]] = mapped_column(String(255))
    external_id: Mapped[Optional[str]] = mapped_column(String(255))  # ID from external system

    # Relationships to lookup tables
    discipline_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("utility_disciplines.id"),
        nullable=False,
        index=True
    )
    material_id: Mapped[Optional[UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("utility_materials.id"),
        index=True
    )

    # Physical characteristics
    diameter_m: Mapped[Optional[float]] = mapped_column(Float)
    wall_thickness_m: Mapped[Optional[float]] = mapped_column(Float)
    length_m: Mapped[Optional[float]] = mapped_column(Float)
    nominal_depth_m: Mapped[Optional[float]] = mapped_column(Float)
    actual_depth_m: Mapped[Optional[float]] = mapped_column(Float)

    # Geometric properties
    start_coordinates: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))  # [x, y, z]
    end_coordinates: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))    # [x, y, z]
    centerline_path: Mapped[Optional[List[List[float]]]] = mapped_column(JSON)     # [[x,y,z], ...]
    is_linear: Mapped[bool] = mapped_column(Boolean, default=True)
    orientation_degrees: Mapped[Optional[float]] = mapped_column(Float)

    # Installation information
    installation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    installation_method: Mapped[Optional[str]] = mapped_column(String(100))
    installer_organization: Mapped[Optional[str]] = mapped_column(String(255))
    installation_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Operational status
    operational_status: Mapped[str] = mapped_column(String(50), default="active")
    service_pressure_bar: Mapped[Optional[float]] = mapped_column(Float)
    flow_capacity: Mapped[Optional[float]] = mapped_column(Float)
    operating_temperature_c: Mapped[Optional[float]] = mapped_column(Float)

    # Condition and maintenance
    condition_rating: Mapped[Optional[str]] = mapped_column(String(50))
    last_inspection_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_maintenance_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    maintenance_history: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    # Ownership and responsibility
    owner_organization: Mapped[Optional[str]] = mapped_column(String(255))
    operator_organization: Mapped[Optional[str]] = mapped_column(String(255))
    contact_information: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON)

    # Risk and safety
    safety_class: Mapped[Optional[str]] = mapped_column(String(50))
    strike_risk_level: Mapped[Optional[str]] = mapped_column(String(50))
    emergency_response_required: Mapped[bool] = mapped_column(Boolean, default=False)
    hazardous_contents: Mapped[bool] = mapped_column(Boolean, default=False)

    # Data quality and source
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    data_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    location_accuracy_m: Mapped[Optional[float]] = mapped_column(Float)
    attribute_completeness: Mapped[Optional[float]] = mapped_column(Float)
    last_verified_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Survey context
    survey_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    detection_difficulty_factors: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))

    # Additional metadata
    additional_attributes: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    regulatory_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    discipline: Mapped["UtilityDiscipline"] = relationship(
        "UtilityDiscipline",
        back_populates="utility_records"
    )
    material: Mapped[Optional["UtilityMaterial"]] = relationship(
        "UtilityMaterial",
        back_populates="utility_records"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("diameter_m > 0 OR diameter_m IS NULL", name="check_diameter_positive"),
        CheckConstraint("wall_thickness_m >= 0 OR wall_thickness_m IS NULL", name="check_wall_thickness"),
        CheckConstraint("length_m > 0 OR length_m IS NULL", name="check_length_positive"),
        CheckConstraint("nominal_depth_m >= 0 OR nominal_depth_m IS NULL", name="check_nominal_depth"),
        CheckConstraint("actual_depth_m >= 0 OR actual_depth_m IS NULL", name="check_actual_depth"),
        CheckConstraint(
            "orientation_degrees >= 0 AND orientation_degrees < 360 OR orientation_degrees IS NULL",
            name="check_orientation_range"
        ),
        CheckConstraint(
            "operational_status IN ('active', 'inactive', 'abandoned', 'decommissioned', 'unknown')",
            name="check_operational_status"
        ),
        CheckConstraint(
            "condition_rating IN ('excellent', 'good', 'fair', 'poor', 'critical', 'unknown') OR condition_rating IS NULL",
            name="check_condition_rating"
        ),
        CheckConstraint(
            "safety_class IN ('low', 'medium', 'high', 'critical') OR safety_class IS NULL",
            name="check_safety_class"
        ),
        CheckConstraint(
            "data_quality_score >= 0 AND data_quality_score <= 1 OR data_quality_score IS NULL",
            name="check_data_quality_score"
        ),
        CheckConstraint(
            "location_accuracy_m >= 0 OR location_accuracy_m IS NULL",
            name="check_location_accuracy"
        ),
        CheckConstraint(
            "attribute_completeness >= 0 AND attribute_completeness <= 1 OR attribute_completeness IS NULL",
            name="check_attribute_completeness"
        ),
        Index("idx_utility_records_utility_id", "utility_id"),
        Index("idx_utility_records_discipline", "discipline_id"),
        Index("idx_utility_records_material", "material_id"),
        Index("idx_utility_records_status", "operational_status"),
        Index("idx_utility_records_position", "start_coordinates"),
        Index("idx_utility_records_depth", "actual_depth_m"),
    )