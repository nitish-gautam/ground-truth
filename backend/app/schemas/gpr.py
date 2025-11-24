"""
GPR data Pydantic schemas
========================

Request and response models for GPR data endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class GPRSurveyBase(BaseModel):
    """Base GPR survey model."""
    survey_name: str = Field(..., description="Survey name")
    location_id: str = Field(..., description="Location identifier")
    survey_objective: Optional[str] = Field(None, description="Survey objective")
    survey_date: Optional[datetime] = Field(None, description="Survey date")
    equipment_model: Optional[str] = Field(None, description="GPR equipment model")
    antenna_frequency: Optional[float] = Field(None, description="Antenna frequency in MHz")
    sampling_frequency: Optional[float] = Field(None, description="Sampling frequency")
    time_window: Optional[float] = Field(None, description="Time window in ns")
    quality_level: Optional[str] = Field(None, description="PAS 128 quality level")


class GPRSurveyCreate(GPRSurveyBase):
    """GPR survey creation model."""
    pass


class GPRSurveyResponse(GPRSurveyBase):
    """GPR survey response model."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    completion_percentage: float
    created_at: datetime
    updated_at: datetime


class GPRScanBase(BaseModel):
    """Base GPR scan model."""
    scan_number: int = Field(..., description="Scan number")
    scan_name: Optional[str] = Field(None, description="Scan name")
    file_path: str = Field(..., description="File path")
    start_position: Optional[float] = Field(None, description="Start position in meters")
    end_position: Optional[float] = Field(None, description="End position in meters")
    scan_length: Optional[float] = Field(None, description="Scan length in meters")


class GPRScanResponse(GPRScanBase):
    """GPR scan response model."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    survey_id: UUID
    file_size_bytes: Optional[int]
    trace_count: Optional[int]
    samples_per_trace: Optional[int]
    data_format: Optional[str]
    is_processed: bool
    processing_status: Optional[str]
    signal_quality_score: Optional[float]
    created_at: datetime


class GPRScanStatistics(BaseModel):
    """GPR scan statistics."""
    model_config = ConfigDict(from_attributes=True)

    total_surveys: int = Field(..., description="Total number of surveys")
    total_scans: int = Field(..., description="Total number of scans")
    total_signal_records: int = Field(..., description="Total signal data records")
    processing_status_distribution: Dict[str, int] = Field(..., description="Processing status distribution")
    last_updated: datetime = Field(..., description="Last update timestamp")