"""
Environmental data models
========================

Database models for environmental factors that affect GPR signal quality
and utility detection accuracy, based on Twente dataset metadata.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, JSON, ForeignKey,
    Index, CheckConstraint, Integer
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class EnvironmentalData(BaseModel):
    """Environmental conditions during GPR survey."""

    __tablename__ = "environmental_data"

    # Survey relationship
    survey_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_surveys.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Location and land use characteristics
    land_use: Mapped[Optional[str]] = mapped_column(String(200))
    land_cover: Mapped[Optional[str]] = mapped_column(String(200))
    land_type: Mapped[Optional[str]] = mapped_column(String(100))

    # Ground conditions (critical for GPR signal propagation)
    ground_condition: Mapped[Optional[str]] = mapped_column(String(100))
    ground_relative_permittivity: Mapped[Optional[float]] = mapped_column(Float)
    relative_groundwater_level: Mapped[Optional[str]] = mapped_column(String(100))

    # Terrain characteristics
    terrain_levelling: Mapped[Optional[str]] = mapped_column(String(100))
    terrain_smoothness: Mapped[Optional[str]] = mapped_column(String(100))

    # Subsurface conditions
    rubble_presence: Mapped[Optional[bool]] = mapped_column(Boolean)
    tree_roots_presence: Mapped[Optional[bool]] = mapped_column(Boolean)
    polluted_soil_presence: Mapped[Optional[bool]] = mapped_column(Boolean)
    blast_furnace_slag_presence: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Weather conditions
    weather_condition: Mapped[Optional[str]] = mapped_column(String(100))
    temperature_celsius: Mapped[Optional[float]] = mapped_column(Float)
    humidity_percentage: Mapped[Optional[float]] = mapped_column(Float)
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float)

    # Utility density and complexity
    amount_of_utilities: Mapped[Optional[int]] = mapped_column(Integer)
    utility_crossing: Mapped[Optional[bool]] = mapped_column(Boolean)
    utility_path_linear: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Construction context
    construction_workers: Mapped[Optional[str]] = mapped_column(String(200))
    complementary_works: Mapped[Optional[str]] = mapped_column(String(200))
    exact_location_accuracy_required: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Additional metadata
    measurement_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    measurement_notes: Mapped[Optional[str]] = mapped_column(String(1000))
    data_source: Mapped[Optional[str]] = mapped_column(String(100))

    # Calculated impact scores
    signal_propagation_score: Mapped[Optional[float]] = mapped_column(Float)
    detection_difficulty_score: Mapped[Optional[float]] = mapped_column(Float)
    environmental_complexity: Mapped[Optional[float]] = mapped_column(Float)

    # Additional parameters as JSON
    additional_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Relationships
    survey: Mapped["GPRSurvey"] = relationship("GPRSurvey", back_populates="environmental_data")
    weather_conditions: Mapped[list["WeatherCondition"]] = relationship(
        "WeatherCondition",
        back_populates="environmental_data",
        cascade="all, delete-orphan"
    )
    ground_conditions: Mapped[list["GroundCondition"]] = relationship(
        "GroundCondition",
        back_populates="environmental_data",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "ground_relative_permittivity > 0 OR ground_relative_permittivity IS NULL",
            name="check_permittivity_positive"
        ),
        CheckConstraint(
            "temperature_celsius >= -50 AND temperature_celsius <= 60 OR temperature_celsius IS NULL",
            name="check_temperature_range"
        ),
        CheckConstraint(
            "humidity_percentage >= 0 AND humidity_percentage <= 100 OR humidity_percentage IS NULL",
            name="check_humidity_range"
        ),
        CheckConstraint(
            "precipitation_mm >= 0 OR precipitation_mm IS NULL",
            name="check_precipitation_positive"
        ),
        CheckConstraint(
            "amount_of_utilities >= 0 OR amount_of_utilities IS NULL",
            name="check_utility_count_positive"
        ),
        CheckConstraint(
            "signal_propagation_score >= 0 AND signal_propagation_score <= 1 OR signal_propagation_score IS NULL",
            name="check_propagation_score"
        ),
        CheckConstraint(
            "detection_difficulty_score >= 0 AND detection_difficulty_score <= 1 OR detection_difficulty_score IS NULL",
            name="check_difficulty_score"
        ),
        Index("idx_environmental_survey", "survey_id"),
        Index("idx_environmental_ground_condition", "ground_condition"),
        Index("idx_environmental_weather", "weather_condition"),
        Index("idx_environmental_land_use", "land_use"),
    )


class WeatherCondition(BaseModel):
    """Detailed weather condition measurements."""

    __tablename__ = "weather_conditions"

    # Environmental data relationship
    environmental_data_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("environmental_data.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Timestamp for this measurement
    measurement_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Weather parameters
    condition_description: Mapped[str] = mapped_column(String(100), nullable=False)
    temperature_celsius: Mapped[Optional[float]] = mapped_column(Float)
    humidity_percentage: Mapped[Optional[float]] = mapped_column(Float)
    pressure_hpa: Mapped[Optional[float]] = mapped_column(Float)
    wind_speed_kmh: Mapped[Optional[float]] = mapped_column(Float)
    wind_direction_degrees: Mapped[Optional[float]] = mapped_column(Float)
    precipitation_rate_mmh: Mapped[Optional[float]] = mapped_column(Float)
    visibility_km: Mapped[Optional[float]] = mapped_column(Float)

    # Weather impact on GPR
    expected_signal_attenuation: Mapped[Optional[float]] = mapped_column(Float)
    moisture_impact_score: Mapped[Optional[float]] = mapped_column(Float)

    # Data source
    weather_station_id: Mapped[Optional[str]] = mapped_column(String(100))
    measurement_source: Mapped[Optional[str]] = mapped_column(String(100))

    # Relationships
    environmental_data: Mapped["EnvironmentalData"] = relationship(
        "EnvironmentalData",
        back_populates="weather_conditions"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "temperature_celsius >= -50 AND temperature_celsius <= 60 OR temperature_celsius IS NULL",
            name="check_weather_temperature"
        ),
        CheckConstraint(
            "humidity_percentage >= 0 AND humidity_percentage <= 100 OR humidity_percentage IS NULL",
            name="check_weather_humidity"
        ),
        CheckConstraint(
            "wind_direction_degrees >= 0 AND wind_direction_degrees < 360 OR wind_direction_degrees IS NULL",
            name="check_wind_direction"
        ),
        Index("idx_weather_environmental_time", "environmental_data_id", "measurement_time"),
        Index("idx_weather_condition", "condition_description"),
    )


class GroundCondition(BaseModel):
    """Detailed ground condition analysis."""

    __tablename__ = "ground_conditions"

    # Environmental data relationship
    environmental_data_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("environmental_data.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Ground composition
    soil_type: Mapped[str] = mapped_column(String(100), nullable=False)
    clay_percentage: Mapped[Optional[float]] = mapped_column(Float)
    sand_percentage: Mapped[Optional[float]] = mapped_column(Float)
    silt_percentage: Mapped[Optional[float]] = mapped_column(Float)
    organic_content_percentage: Mapped[Optional[float]] = mapped_column(Float)

    # Physical properties
    density_kg_m3: Mapped[Optional[float]] = mapped_column(Float)
    porosity_percentage: Mapped[Optional[float]] = mapped_column(Float)
    moisture_content_percentage: Mapped[Optional[float]] = mapped_column(Float)
    compaction_level: Mapped[Optional[str]] = mapped_column(String(50))

    # Electrical properties (critical for GPR)
    electrical_conductivity: Mapped[Optional[float]] = mapped_column(Float)
    dielectric_constant: Mapped[Optional[float]] = mapped_column(Float)
    magnetic_permeability: Mapped[Optional[float]] = mapped_column(Float)

    # Contamination and obstacles
    contamination_level: Mapped[Optional[str]] = mapped_column(String(50))
    metal_content_ppm: Mapped[Optional[float]] = mapped_column(Float)
    salt_content_ppm: Mapped[Optional[float]] = mapped_column(Float)

    # GPR signal impact
    signal_penetration_depth_m: Mapped[Optional[float]] = mapped_column(Float)
    signal_attenuation_db_m: Mapped[Optional[float]] = mapped_column(Float)
    reflection_coefficient: Mapped[Optional[float]] = mapped_column(Float)

    # Measurement details
    measurement_depth_m: Mapped[Optional[float]] = mapped_column(Float)
    measurement_method: Mapped[Optional[str]] = mapped_column(String(100))
    lab_analysis_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Additional properties
    additional_properties: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Relationships
    environmental_data: Mapped["EnvironmentalData"] = relationship(
        "EnvironmentalData",
        back_populates="ground_conditions"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "clay_percentage >= 0 AND clay_percentage <= 100 OR clay_percentage IS NULL",
            name="check_clay_percentage"
        ),
        CheckConstraint(
            "sand_percentage >= 0 AND sand_percentage <= 100 OR sand_percentage IS NULL",
            name="check_sand_percentage"
        ),
        CheckConstraint(
            "silt_percentage >= 0 AND silt_percentage <= 100 OR silt_percentage IS NULL",
            name="check_silt_percentage"
        ),
        CheckConstraint(
            "organic_content_percentage >= 0 AND organic_content_percentage <= 100 OR organic_content_percentage IS NULL",
            name="check_organic_content"
        ),
        CheckConstraint(
            "porosity_percentage >= 0 AND porosity_percentage <= 100 OR porosity_percentage IS NULL",
            name="check_porosity"
        ),
        CheckConstraint(
            "moisture_content_percentage >= 0 OR moisture_content_percentage IS NULL",
            name="check_moisture_content"
        ),
        CheckConstraint(
            "electrical_conductivity >= 0 OR electrical_conductivity IS NULL",
            name="check_conductivity"
        ),
        CheckConstraint(
            "dielectric_constant >= 1 OR dielectric_constant IS NULL",
            name="check_dielectric_constant"
        ),
        Index("idx_ground_environmental", "environmental_data_id"),
        Index("idx_ground_soil_type", "soil_type"),
        Index("idx_ground_contamination", "contamination_level"),
    )