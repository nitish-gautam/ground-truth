"""
Machine learning and analytics models
====================================

Database models for ML model management, feature vectors, performance metrics,
and training sessions for GPR-based utility detection.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, JSON, ForeignKey,
    Index, CheckConstraint, Integer, Text, LargeBinary
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class MLModel(BaseModel):
    """Machine learning model registry and metadata."""

    __tablename__ = "ml_models"

    # Model identification
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    model_architecture: Mapped[Optional[str]] = mapped_column(String(100))

    # Model purpose and domain
    purpose: Mapped[str] = mapped_column(String(200), nullable=False)
    target_task: Mapped[str] = mapped_column(String(100), nullable=False)  # detection, classification, regression
    domain_specific: Mapped[Optional[str]] = mapped_column(String(100))    # utility_type, material_type, etc.

    # Development stage
    development_stage: Mapped[str] = mapped_column(String(50), nullable=False, default="development")
    is_production_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    deployment_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Model parameters and configuration
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    preprocessing_pipeline: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    feature_engineering: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Input/output specifications
    input_features: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    input_shape: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    output_classes: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    output_shape: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))

    # Training information
    training_dataset_size: Mapped[Optional[int]] = mapped_column(Integer)
    validation_dataset_size: Mapped[Optional[int]] = mapped_column(Integer)
    test_dataset_size: Mapped[Optional[int]] = mapped_column(Integer)
    training_duration_hours: Mapped[Optional[float]] = mapped_column(Float)

    # Model storage
    model_file_path: Mapped[Optional[str]] = mapped_column(String(500))
    model_serialized: Mapped[Optional[bytes]] = mapped_column(LargeBinary)  # For small models
    model_checksum: Mapped[Optional[str]] = mapped_column(String(100))
    model_size_mb: Mapped[Optional[float]] = mapped_column(Float)

    # Framework and dependencies
    framework: Mapped[Optional[str]] = mapped_column(String(100))
    framework_version: Mapped[Optional[str]] = mapped_column(String(50))
    python_version: Mapped[Optional[str]] = mapped_column(String(20))
    dependencies: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON)

    # Documentation and metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    training_notes: Mapped[Optional[str]] = mapped_column(Text)
    known_limitations: Mapped[Optional[str]] = mapped_column(Text)
    performance_characteristics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Author and ownership
    created_by_user: Mapped[Optional[str]] = mapped_column(String(255))
    research_team: Mapped[Optional[str]] = mapped_column(String(255))
    contact_email: Mapped[Optional[str]] = mapped_column(String(255))

    # Relationships
    training_sessions: Mapped[List["TrainingSession"]] = relationship(
        "TrainingSession",
        back_populates="model",
        cascade="all, delete-orphan"
    )
    performance_metrics: Mapped[List["ModelPerformance"]] = relationship(
        "ModelPerformance",
        back_populates="model",
        cascade="all, delete-orphan"
    )
    feature_vectors: Mapped[List["FeatureVector"]] = relationship(
        "FeatureVector",
        back_populates="model",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "model_type IN ('neural_network', 'random_forest', 'svm', 'xgboost', 'cnn', 'rnn', 'lstm', 'transformer')",
            name="check_model_type"
        ),
        CheckConstraint(
            "target_task IN ('detection', 'classification', 'regression', 'segmentation', 'clustering')",
            name="check_target_task"
        ),
        CheckConstraint(
            "development_stage IN ('development', 'training', 'validation', 'testing', 'production', 'deprecated')",
            name="check_development_stage"
        ),
        CheckConstraint(
            "training_dataset_size > 0 OR training_dataset_size IS NULL",
            name="check_training_size"
        ),
        CheckConstraint(
            "training_duration_hours >= 0 OR training_duration_hours IS NULL",
            name="check_training_duration"
        ),
        Index("idx_ml_models_name_version", "model_name", "model_version"),
        Index("idx_ml_models_type", "model_type"),
        Index("idx_ml_models_stage", "development_stage"),
        Index("idx_ml_models_production", "is_production_ready"),
    )


class TrainingSession(BaseModel):
    """Individual model training session records."""

    __tablename__ = "training_sessions"

    # Model relationship
    model_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Training session identification
    session_name: Mapped[Optional[str]] = mapped_column(String(255))
    session_purpose: Mapped[str] = mapped_column(String(200), nullable=False)

    # Training configuration
    training_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    data_configuration: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Dataset information
    training_data_sources: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    test_samples: Mapped[Optional[int]] = mapped_column(Integer)

    # Training timeline
    training_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    total_training_time_hours: Mapped[Optional[float]] = mapped_column(Float)

    # Training progress
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs_planned: Mapped[Optional[int]] = mapped_column(Integer)
    current_status: Mapped[str] = mapped_column(String(50), default="running")

    # Performance during training
    best_validation_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    best_validation_loss: Mapped[Optional[float]] = mapped_column(Float)
    final_training_loss: Mapped[Optional[float]] = mapped_column(Float)
    convergence_epoch: Mapped[Optional[int]] = mapped_column(Integer)

    # Resource utilization
    gpu_hours_used: Mapped[Optional[float]] = mapped_column(Float)
    cpu_hours_used: Mapped[Optional[float]] = mapped_column(Float)
    memory_peak_gb: Mapped[Optional[float]] = mapped_column(Float)
    computational_cost: Mapped[Optional[float]] = mapped_column(Float)

    # Training logs and artifacts
    training_logs: Mapped[Optional[str]] = mapped_column(Text)
    training_artifacts: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    checkpoint_paths: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))

    # Early stopping and optimization
    early_stopping_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    early_stopping_epoch: Mapped[Optional[int]] = mapped_column(Integer)
    learning_rate_schedule: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Training environment
    environment_info: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON)
    random_seed: Mapped[Optional[int]] = mapped_column(Integer)
    reproducibility_hash: Mapped[Optional[str]] = mapped_column(String(100))

    # Results and outcomes
    training_success: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    training_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    model: Mapped["MLModel"] = relationship("MLModel", back_populates="training_sessions")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "current_status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="check_training_status"
        ),
        CheckConstraint("training_samples > 0", name="check_training_samples"),
        CheckConstraint("validation_samples > 0", name="check_validation_samples"),
        CheckConstraint("epochs_completed >= 0", name="check_epochs_completed"),
        CheckConstraint(
            "best_validation_accuracy >= 0 AND best_validation_accuracy <= 1 OR best_validation_accuracy IS NULL",
            name="check_validation_accuracy"
        ),
        Index("idx_training_sessions_model", "model_id"),
        Index("idx_training_sessions_status", "current_status"),
        Index("idx_training_sessions_started", "training_started_at"),
    )


class FeatureVector(BaseModel):
    """Feature vectors extracted from GPR data for ML training."""

    __tablename__ = "feature_vectors"

    # Model relationship (optional - features can exist without a specific model)
    model_id: Mapped[Optional[UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="SET NULL"),
        index=True
    )

    # Source data relationship
    scan_id: Mapped[Optional[UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("gpr_scans.id", ondelete="CASCADE"),
        index=True
    )

    # Feature identification
    feature_set_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(50), nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(100), nullable=False)

    # Feature data
    feature_vector: Mapped[List[float]] = mapped_column(ARRAY(Float), nullable=False)
    feature_names: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    feature_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)

    # Feature categories
    statistical_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    frequency_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    spatial_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    environmental_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)

    # Target labels (for supervised learning)
    target_label: Mapped[Optional[str]] = mapped_column(String(100))
    target_value: Mapped[Optional[float]] = mapped_column(Float)
    target_confidence: Mapped[Optional[float]] = mapped_column(Float)
    ground_truth_available: Mapped[bool] = mapped_column(Boolean, default=False)

    # Feature quality and preprocessing
    feature_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    preprocessing_applied: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    normalization_method: Mapped[Optional[str]] = mapped_column(String(100))
    feature_scaling: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)

    # Data split assignment
    data_split: Mapped[Optional[str]] = mapped_column(String(20))  # train, validation, test
    cross_validation_fold: Mapped[Optional[int]] = mapped_column(Integer)

    # Extraction metadata
    extraction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    extraction_duration_ms: Mapped[Optional[float]] = mapped_column(Float)
    extraction_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    model: Mapped[Optional["MLModel"]] = relationship("MLModel", back_populates="feature_vectors")
    scan: Mapped[Optional["GPRScan"]] = relationship("GPRScan")

    # Constraints
    __table_args__ = (
        CheckConstraint("feature_dimensions > 0", name="check_feature_dimensions"),
        CheckConstraint(
            "target_confidence >= 0 AND target_confidence <= 1 OR target_confidence IS NULL",
            name="check_target_confidence"
        ),
        CheckConstraint(
            "feature_quality_score >= 0 AND feature_quality_score <= 1 OR feature_quality_score IS NULL",
            name="check_feature_quality"
        ),
        CheckConstraint(
            "data_split IN ('train', 'validation', 'test') OR data_split IS NULL",
            name="check_data_split"
        ),
        CheckConstraint("usage_count >= 0", name="check_usage_count"),
        Index("idx_feature_vectors_model", "model_id"),
        Index("idx_feature_vectors_scan", "scan_id"),
        Index("idx_feature_vectors_set", "feature_set_name", "feature_version"),
        Index("idx_feature_vectors_split", "data_split"),
        Index("idx_feature_vectors_extraction", "extraction_timestamp"),
    )


class ModelPerformance(BaseModel):
    """Model performance metrics and evaluation results."""

    __tablename__ = "model_performance"

    # Model relationship
    model_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Evaluation context
    evaluation_dataset: Mapped[str] = mapped_column(String(255), nullable=False)
    evaluation_type: Mapped[str] = mapped_column(String(100), nullable=False)  # validation, test, production
    evaluation_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Dataset characteristics
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    positive_samples: Mapped[Optional[int]] = mapped_column(Integer)
    negative_samples: Mapped[Optional[int]] = mapped_column(Integer)
    class_distribution: Mapped[Optional[Dict[str, int]]] = mapped_column(JSON)

    # Classification metrics
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    precision: Mapped[Optional[float]] = mapped_column(Float)
    recall: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    auc_roc: Mapped[Optional[float]] = mapped_column(Float)
    auc_pr: Mapped[Optional[float]] = mapped_column(Float)

    # Regression metrics
    mae: Mapped[Optional[float]] = mapped_column(Float)        # Mean Absolute Error
    mse: Mapped[Optional[float]] = mapped_column(Float)        # Mean Squared Error
    rmse: Mapped[Optional[float]] = mapped_column(Float)       # Root Mean Squared Error
    r2_score: Mapped[Optional[float]] = mapped_column(Float)   # R-squared

    # Confusion matrix and detailed results
    confusion_matrix: Mapped[Optional[List[List[int]]]] = mapped_column(JSON)
    classification_report: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    per_class_metrics: Mapped[Optional[Dict[str, Dict[str, float]]]] = mapped_column(JSON)

    # GPR-specific metrics
    utility_detection_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    material_classification_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    depth_estimation_mae: Mapped[Optional[float]] = mapped_column(Float)
    position_estimation_mae: Mapped[Optional[float]] = mapped_column(Float)

    # Environmental performance
    performance_by_weather: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    performance_by_ground_condition: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    performance_by_utility_type: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)

    # Inference performance
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    throughput_samples_per_second: Mapped[Optional[float]] = mapped_column(Float)
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float)

    # Statistical confidence
    confidence_interval_95: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    statistical_significance: Mapped[Optional[float]] = mapped_column(Float)
    cross_validation_scores: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    cross_validation_std: Mapped[Optional[float]] = mapped_column(Float)

    # Evaluation configuration
    evaluation_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    threshold_optimization: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    evaluation_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Performance trends
    performance_trend: Mapped[Optional[str]] = mapped_column(String(20))  # improving, stable, degrading
    compared_to_baseline: Mapped[Optional[float]] = mapped_column(Float)
    compared_to_previous: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    model: Mapped["MLModel"] = relationship("MLModel", back_populates="performance_metrics")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "evaluation_type IN ('validation', 'test', 'production', 'cross_validation')",
            name="check_evaluation_type"
        ),
        CheckConstraint("dataset_size > 0", name="check_dataset_size"),
        CheckConstraint(
            "accuracy >= 0 AND accuracy <= 1 OR accuracy IS NULL",
            name="check_accuracy_range"
        ),
        CheckConstraint(
            "precision >= 0 AND precision <= 1 OR precision IS NULL",
            name="check_precision_range"
        ),
        CheckConstraint(
            "recall >= 0 AND recall <= 1 OR recall IS NULL",
            name="check_recall_range"
        ),
        CheckConstraint(
            "f1_score >= 0 AND f1_score <= 1 OR f1_score IS NULL",
            name="check_f1_range"
        ),
        CheckConstraint(
            "auc_roc >= 0 AND auc_roc <= 1 OR auc_roc IS NULL",
            name="check_auc_roc_range"
        ),
        CheckConstraint(
            "inference_time_ms >= 0 OR inference_time_ms IS NULL",
            name="check_inference_time"
        ),
        CheckConstraint(
            "performance_trend IN ('improving', 'stable', 'degrading') OR performance_trend IS NULL",
            name="check_performance_trend"
        ),
        Index("idx_model_performance_model", "model_id"),
        Index("idx_model_performance_type", "evaluation_type"),
        Index("idx_model_performance_timestamp", "evaluation_timestamp"),
        Index("idx_model_performance_accuracy", "accuracy"),
    )