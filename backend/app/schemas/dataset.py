"""
Dataset-related Pydantic schemas
===============================

Request and response models for dataset management endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class DatasetType(str, Enum):
    """Dataset type enumeration."""
    GPR_SCANS = "gpr_scans"
    IMAGES = "images"
    METADATA = "metadata"
    GROUND_TRUTH = "ground_truth"


class DatasetStatus(str, Enum):
    """Dataset processing status."""
    AVAILABLE = "available"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    NOT_FOUND = "not_found"


class DatasetInfo(BaseModel):
    """Information about a dataset."""
    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., description="Dataset name")
    type: DatasetType = Field(..., description="Type of dataset")
    description: str = Field(..., description="Dataset description")
    path: str = Field(..., description="File system path to dataset")
    file_count: int = Field(..., ge=0, description="Number of files in dataset")
    total_size_mb: float = Field(..., ge=0, description="Total size in megabytes")
    has_metadata: bool = Field(..., description="Whether dataset has metadata file")
    has_ground_truth: bool = Field(..., description="Whether dataset has ground truth data")
    status: DatasetStatus = Field(..., description="Current processing status")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class TwenteDatasetStatus(BaseModel):
    """Status of Twente GPR dataset processing."""
    model_config = ConfigDict(from_attributes=True)

    total_zip_files: int = Field(..., ge=0, description="Total number of ZIP files")
    processed_zip_files: int = Field(..., ge=0, description="Number of processed ZIP files")
    total_scans: int = Field(..., ge=0, description="Total number of GPR scans")
    processed_scans: int = Field(..., ge=0, description="Number of processed scans")
    total_processed: int = Field(..., ge=0, description="Total records processed")

    metadata_loaded: bool = Field(..., description="Whether metadata CSV is loaded")
    metadata_records: int = Field(..., ge=0, description="Number of metadata records")

    processing_status: DatasetStatus = Field(..., description="Overall processing status")
    error_count: int = Field(..., ge=0, description="Number of processing errors")
    last_processed_at: Optional[datetime] = Field(None, description="Last processing timestamp")

    progress_percentage: float = Field(..., ge=0, le=100, description="Processing progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class MojahidDatasetStatus(BaseModel):
    """Status of Mojahid images dataset processing."""
    model_config = ConfigDict(from_attributes=True)

    categories: List[str] = Field(..., description="Available image categories")
    category_counts: Dict[str, int] = Field(..., description="Image count per category")
    total_images: int = Field(..., ge=0, description="Total number of images")
    processed_images: int = Field(..., ge=0, description="Number of processed images")

    processing_status: DatasetStatus = Field(..., description="Overall processing status")
    error_count: int = Field(..., ge=0, description="Number of processing errors")
    last_processed_at: Optional[datetime] = Field(None, description="Last processing timestamp")

    progress_percentage: float = Field(..., ge=0, le=100, description="Processing progress percentage")
    augmented_data_included: bool = Field(..., description="Whether augmented data is included")


class FileProcessingStatus(BaseModel):
    """Status of individual file processing."""
    model_config = ConfigDict(from_attributes=True)

    file_path: str = Field(..., description="Path to the file")
    file_name: str = Field(..., description="Name of the file")
    file_size_mb: float = Field(..., ge=0, description="File size in megabytes")

    status: DatasetStatus = Field(..., description="Processing status")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Processing duration")

    records_extracted: int = Field(..., ge=0, description="Number of records extracted")
    features_extracted: int = Field(..., ge=0, description="Number of features extracted")

    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    processing_notes: Optional[str] = Field(None, description="Additional processing notes")


class BatchProcessingRequest(BaseModel):
    """Request for batch file processing."""
    model_config = ConfigDict(from_attributes=True)

    file_paths: List[str] = Field(..., min_items=1, description="List of file paths to process")
    batch_size: int = Field(default=10, ge=1, le=100, description="Number of files to process in each batch")
    processing_options: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration options")
    priority: int = Field(default=5, ge=1, le=10, description="Processing priority (1=highest, 10=lowest)")

    # Processing options
    extract_features: bool = Field(default=True, description="Whether to extract features from data")
    create_ground_truth: bool = Field(default=True, description="Whether to create ground truth records")
    perform_validation: bool = Field(default=False, description="Whether to perform validation")

    # Notification settings
    notify_on_completion: bool = Field(default=False, description="Send notification when complete")
    notification_email: Optional[str] = Field(None, description="Email for completion notification")


class DatasetLoadRequest(BaseModel):
    """Request for loading a specific dataset."""
    model_config = ConfigDict(from_attributes=True)

    dataset_name: str = Field(..., description="Name of the dataset to load")
    force_reload: bool = Field(default=False, description="Force reload even if already loaded")
    batch_size: int = Field(default=10, ge=1, le=50, description="Batch size for processing")

    # Filtering options
    file_pattern: Optional[str] = Field(None, description="File pattern to match (regex)")
    max_files: Optional[int] = Field(None, ge=1, description="Maximum number of files to process")
    skip_existing: bool = Field(default=True, description="Skip files that are already processed")

    # Processing options
    processing_options: Dict[str, Any] = Field(default_factory=dict, description="Additional processing options")


class DatasetStatistics(BaseModel):
    """Statistical summary of a dataset."""
    model_config = ConfigDict(from_attributes=True)

    dataset_name: str = Field(..., description="Name of the dataset")
    total_files: int = Field(..., ge=0, description="Total number of files")
    total_size_gb: float = Field(..., ge=0, description="Total size in gigabytes")

    # Data distribution
    file_type_distribution: Dict[str, int] = Field(..., description="Count of files by type")
    size_distribution: Dict[str, int] = Field(..., description="File size distribution")

    # Quality metrics
    data_quality_score: float = Field(..., ge=0, le=1, description="Overall data quality score")
    completeness_percentage: float = Field(..., ge=0, le=100, description="Data completeness percentage")

    # Processing metrics
    processing_time_hours: float = Field(..., ge=0, description="Total processing time in hours")
    success_rate: float = Field(..., ge=0, le=1, description="Processing success rate")

    # Temporal information
    date_range_start: Optional[datetime] = Field(None, description="Earliest data timestamp")
    date_range_end: Optional[datetime] = Field(None, description="Latest data timestamp")
    last_updated: datetime = Field(..., description="Last statistics update")


class DatasetValidationResult(BaseModel):
    """Results of dataset validation."""
    model_config = ConfigDict(from_attributes=True)

    dataset_name: str = Field(..., description="Name of the validated dataset")
    validation_timestamp: datetime = Field(..., description="When validation was performed")

    # Validation status
    is_valid: bool = Field(..., description="Overall validation result")
    validation_score: float = Field(..., ge=0, le=1, description="Validation score (0-1)")

    # Validation checks
    file_integrity_passed: bool = Field(..., description="File integrity check result")
    schema_validation_passed: bool = Field(..., description="Schema validation result")
    data_consistency_passed: bool = Field(..., description="Data consistency check result")

    # Issues found
    missing_files: List[str] = Field(default_factory=list, description="List of missing expected files")
    corrupted_files: List[str] = Field(default_factory=list, description="List of corrupted files")
    schema_violations: List[Dict[str, str]] = Field(default_factory=list, description="Schema validation errors")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for fixing issues")
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues requiring immediate attention")

    # Statistics
    total_files_checked: int = Field(..., ge=0, description="Total number of files checked")
    files_with_issues: int = Field(..., ge=0, description="Number of files with issues")
    validation_duration_seconds: float = Field(..., ge=0, description="Time taken for validation")


class UploadResponse(BaseModel):
    """Response for file upload operations."""
    model_config = ConfigDict(from_attributes=True)

    message: str = Field(..., description="Response message")
    filename: str = Field(..., description="Name of uploaded file")
    file_size_mb: float = Field(..., ge=0, description="File size in megabytes")
    upload_id: str = Field(..., description="Unique identifier for this upload")

    processing_started: bool = Field(..., description="Whether processing has started")
    estimated_processing_time_minutes: Optional[float] = Field(None, description="Estimated processing time")

    status_check_url: str = Field(..., description="URL to check processing status")
    uploaded_at: datetime = Field(..., description="Upload timestamp")