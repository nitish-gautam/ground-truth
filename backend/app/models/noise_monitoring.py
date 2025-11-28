"""
Database models for noise monitoring data
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class NoiseMonitoringMeasurement(Base):
    """Individual noise measurement record"""
    __tablename__ = "noise_monitoring_measurements"

    id = Column(Integer, primary_key=True, index=True)

    # Temporal data
    timestamp = Column(DateTime, nullable=False, index=True)
    month = Column(String(50), nullable=False, index=True)  # e.g., "December_2024"

    # Location data
    location_id = Column(String(50), nullable=False, index=True)  # e.g., "TOS-N1"
    area = Column(String(50), nullable=False, index=True)  # North, Central, South
    council = Column(String(100), nullable=False, index=True)  # e.g., "Birmingham"

    # Measurement data
    period_hours = Column(Float, nullable=False)  # Duration in hours
    avg_noise_db = Column(Float, nullable=False)  # LpAeq,T - Average noise level
    max_noise_db = Column(Float, nullable=False)  # LpAF,Max - Maximum noise level
    background_noise_db = Column(Float, nullable=False)  # LpA90,T - Background noise

    # Compliance
    is_violation = Column(Integer, default=0, index=True)  # 1 if above limit (75 dB)

    # Metadata
    source_file = Column(String(255), nullable=True)  # Original Excel file
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_month_council', 'month', 'council'),
        Index('idx_month_area', 'month', 'area'),
        Index('idx_timestamp_location', 'timestamp', 'location_id'),
    )

class NoiseMonitoringLocation(Base):
    """Unique monitoring locations"""
    __tablename__ = "noise_monitoring_locations"

    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(String(50), unique=True, nullable=False, index=True)
    area = Column(String(50), nullable=False)
    council = Column(String(100), nullable=False)

    # GPS coordinates (if available in metadata)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
