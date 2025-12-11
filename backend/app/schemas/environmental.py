"""
Environmental Analysis Schemas
==============================

Pydantic schemas for environmental correlation analysis API endpoints.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class AnalysisConfig(BaseModel):
    """Configuration parameters for environmental analysis."""

    correlation_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum correlation coefficient to consider significant"
    )
    significance_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="P-value threshold for statistical significance"
    )
    pca_components: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Number of principal components to extract"
    )
    factor_count: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of factors for factor analysis"
    )
    n_clusters: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Number of clusters for K-means clustering"
    )
    dbscan_eps: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="DBSCAN epsilon parameter for density clustering"
    )
    dbscan_min_samples: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Minimum samples per DBSCAN cluster"
    )
    cross_validation_folds: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of folds for cross-validation"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducible results"
    )
    feature_selection_k: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Number of top features to select"
    )
    multiple_comparison_method: str = Field(
        default="bonferroni",
        description="Method for multiple comparison correction"
    )
    enable_advanced_clustering: bool = Field(
        default=True,
        description="Enable additional clustering algorithms"
    )
    enable_material_analysis: bool = Field(
        default=True,
        description="Enable detailed material property analysis"
    )
    enable_performance_prediction: bool = Field(
        default=True,
        description="Enable predictive modeling"
    )

    @field_validator('multiple_comparison_method')
    @classmethod
    def validate_comparison_method(cls, v):
        allowed_methods = ['bonferroni', 'fdr', 'benjamini_hochberg']
        if v not in allowed_methods:
            raise ValueError(f"multiple_comparison_method must be one of {allowed_methods}")
        return v


class EnvironmentalAnalysisRequest(BaseModel):
    """Request schema for comprehensive environmental analysis."""

    metadata_csv_path: Optional[str] = Field(
        None,
        description="Path to Twente metadata CSV file"
    )
    metadata_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Metadata as list of dictionaries (alternative to CSV path)"
    )
    performance_csv_path: Optional[str] = Field(
        None,
        description="Optional path to performance metrics CSV file"
    )
    performance_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Performance data as list of dictionaries"
    )
    analysis_config: Optional[AnalysisConfig] = Field(
        None,
        description="Configuration parameters for analysis"
    )
    include_detailed_results: bool = Field(
        default=True,
        description="Include detailed analysis results in response"
    )
    generate_report: bool = Field(
        default=True,
        description="Generate comprehensive analysis report"
    )

    @field_validator('metadata_csv_path', 'performance_csv_path')
    @classmethod
    def validate_csv_path(cls, v):
        if v is not None and not v.endswith('.csv'):
            raise ValueError("CSV file path must end with .csv")
        return v

    class Config:
        schema_extra = {
            "example": {
                "metadata_csv_path": "/path/to/twente_metadata.csv",
                "performance_csv_path": "/path/to/performance_metrics.csv",
                "analysis_config": {
                    "correlation_threshold": 0.4,
                    "significance_level": 0.05,
                    "n_clusters": 6,
                    "enable_material_analysis": True
                },
                "include_detailed_results": True,
                "generate_report": True
            }
        }


class DataSummary(BaseModel):
    """Summary of analyzed dataset."""

    total_samples: int = Field(description="Total number of survey samples")
    total_features: int = Field(description="Total number of features/columns")
    metadata_file: Optional[str] = Field(None, description="Name of metadata file")
    performance_file: Optional[str] = Field(None, description="Name of performance file")
    analysis_timestamp: Optional[str] = Field(None, description="Timestamp of analysis")
    missing_data_percentage: Optional[float] = Field(None, description="Percentage of missing data")
    completeness_percentage: Optional[float] = Field(None, description="Percentage of complete records")


class CorrelationAnalysisResult(BaseModel):
    """Results from correlation analysis."""

    pearson_correlation: Optional[Dict[str, Any]] = Field(None, description="Pearson correlation results")
    spearman_correlation: Optional[Dict[str, Any]] = Field(None, description="Spearman correlation results")
    environmental_factor_correlations: Optional[Dict[str, Any]] = Field(None, description="Environmental factor correlations")
    environmental_interactions: Optional[Dict[str, Any]] = Field(None, description="Environmental interactions")
    permittivity_correlations: Optional[Dict[str, Any]] = Field(None, description="Permittivity correlation analysis")


class MaterialClassificationResult(BaseModel):
    """Results from material classification analysis."""

    material_property_analysis: Optional[Dict[str, Any]] = Field(None, description="Material property analysis")
    detectability_model: Optional[Dict[str, Any]] = Field(None, description="Material detectability model")
    age_classification: Optional[Dict[str, Any]] = Field(None, description="Material age classification")
    conductivity_groups: Optional[Dict[str, Any]] = Field(None, description="Material conductivity grouping")
    material_environment_interactions: Optional[Dict[str, Any]] = Field(None, description="Material-environment interactions")
    corrosion_risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Corrosion risk assessment")


class PerformancePredictionResult(BaseModel):
    """Results from performance prediction models."""

    environmental_impact_predictor: Optional[Dict[str, Any]] = Field(None, description="Environmental impact prediction model")
    detection_difficulty_predictor: Optional[Dict[str, Any]] = Field(None, description="Detection difficulty prediction model")
    signal_quality_predictor: Optional[Dict[str, Any]] = Field(None, description="Signal quality prediction model")
    utility_material_predictor: Optional[Dict[str, Any]] = Field(None, description="Utility material prediction model")
    optimal_conditions_predictor: Optional[Dict[str, Any]] = Field(None, description="Optimal conditions prediction model")
    actual_performance_predictor: Optional[Dict[str, Any]] = Field(None, description="Actual performance prediction model")


class SignificanceTestingResult(BaseModel):
    """Results from statistical significance testing."""

    weather_condition_significance: Optional[Dict[str, Any]] = Field(None, description="Weather condition significance tests")
    ground_condition_significance: Optional[Dict[str, Any]] = Field(None, description="Ground condition significance tests")
    permittivity_significance: Optional[Dict[str, Any]] = Field(None, description="Permittivity significance tests")
    utility_configuration_significance: Optional[Dict[str, Any]] = Field(None, description="Utility configuration significance tests")
    material_type_significance: Optional[Dict[str, Any]] = Field(None, description="Material type significance tests")
    contamination_significance: Optional[Dict[str, Any]] = Field(None, description="Contamination factor significance tests")
    land_use_significance: Optional[Dict[str, Any]] = Field(None, description="Land use significance tests")
    multiple_comparison_corrections: Optional[Dict[str, Any]] = Field(None, description="Multiple comparison corrections")


class FeatureImportanceResult(BaseModel):
    """Results from feature importance analysis."""

    variance_ranking: Optional[Dict[str, float]] = Field(None, description="Variance-based feature ranking")
    random_forest_importance: Optional[Dict[str, float]] = Field(None, description="Random Forest feature importance")
    mutual_information_importance: Optional[Dict[str, float]] = Field(None, description="Mutual information importance")
    pca_feature_loadings: Optional[Dict[str, Any]] = Field(None, description="PCA feature loadings")


class ActionableInsights(BaseModel):
    """Actionable insights and recommendations."""

    environmental_optimization: List[str] = Field(description="Environmental optimization recommendations")
    material_optimization: List[str] = Field(description="Material-specific optimization recommendations")
    performance_insights: List[str] = Field(description="Performance optimization insights")
    priority_actions: List[str] = Field(description="Priority action items")


class ComprehensiveReport(BaseModel):
    """Comprehensive analysis report."""

    executive_summary: str = Field(description="Executive summary of analysis")
    key_findings: List[str] = Field(description="Key findings from analysis")
    analysis_scope: str = Field(description="Scope of analysis performed")
    statistical_confidence: str = Field(description="Statistical confidence assessment")
    recommendations: str = Field(description="Overall recommendations")


class EnvironmentalAnalysisResponse(BaseModel):
    """Response schema for comprehensive environmental analysis."""

    success: bool = Field(description="Whether analysis completed successfully")
    message: str = Field(description="Response message")
    data_summary: DataSummary = Field(description="Summary of analyzed data")

    # Main analysis results
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Complete analysis results")

    # Structured result components
    correlation_analysis: Optional[CorrelationAnalysisResult] = Field(None, description="Correlation analysis results")
    material_classification: Optional[MaterialClassificationResult] = Field(None, description="Material classification results")
    performance_prediction: Optional[PerformancePredictionResult] = Field(None, description="Performance prediction results")
    significance_testing: Optional[SignificanceTestingResult] = Field(None, description="Statistical significance testing results")
    feature_importance: Optional[FeatureImportanceResult] = Field(None, description="Feature importance analysis results")
    actionable_insights: Optional[ActionableInsights] = Field(None, description="Actionable insights and recommendations")
    comprehensive_report: Optional[ComprehensiveReport] = Field(None, description="Comprehensive analysis report")

    # Analysis metadata
    analysis_timestamp: Optional[datetime] = Field(None, description="Timestamp of analysis completion")
    analysis_duration_seconds: Optional[float] = Field(None, description="Duration of analysis in seconds")
    analysis_version: Optional[str] = Field(None, description="Version of analysis algorithm used")

    # Error information (if any)
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details if analysis failed")
    warnings: Optional[List[str]] = Field(None, description="Analysis warnings")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Comprehensive environmental analysis completed successfully",
                "data_summary": {
                    "total_samples": 150,
                    "total_features": 25,
                    "metadata_file": "twente_metadata.csv",
                    "analysis_timestamp": "2024-09-21T10:30:00Z",
                    "completeness_percentage": 95.5
                },
                "actionable_insights": {
                    "environmental_optimization": [
                        "Schedule surveys during dry weather conditions for optimal signal quality",
                        "Prefer sandy ground locations when possible for better signal penetration"
                    ],
                    "material_optimization": [
                        "High material detectability: Standard GPR protocols should be effective"
                    ],
                    "performance_insights": [
                        "Weather conditions have the strongest impact on GPR performance",
                        "Ground permittivity correlation with detection success is significant"
                    ],
                    "priority_actions": [
                        "Implement weather-based survey scheduling protocols",
                        "Develop ground condition assessment procedures"
                    ]
                },
                "analysis_timestamp": "2024-09-21T10:30:15Z",
                "analysis_duration_seconds": 45.2,
                "analysis_version": "1.0.0"
            }
        }


class MaterialProperty(BaseModel):
    """Material property definition for reference."""

    material_name: str = Field(description="Name of the material")
    conductivity: str = Field(description="Electrical conductivity level")
    permittivity: str = Field(description="Relative permittivity level")
    detectability: str = Field(description="GPR detectability rating")
    corrosion_risk: str = Field(description="Corrosion risk level")
    age_category: str = Field(description="Age category (legacy, traditional, modern)")


class EnvironmentalCoefficient(BaseModel):
    """Environmental impact coefficient definition."""

    factor_name: str = Field(description="Name of environmental factor")
    factor_value: str = Field(description="Specific value of the factor")
    impact_coefficient: float = Field(description="Impact coefficient on GPR performance")
    impact_description: str = Field(description="Description of the impact")


class AnalysisCapabilities(BaseModel):
    """Analysis capabilities and supported features."""

    environmental_factors_analyzed: List[str] = Field(description="List of environmental factors analyzed")
    analysis_methods: List[str] = Field(description="Statistical and ML methods used")
    prediction_models: List[str] = Field(description="Available prediction models")
    material_analysis: List[str] = Field(description="Material analysis capabilities")
    output_features: List[str] = Field(description="Analysis output features")
    supported_datasets: List[str] = Field(description="Supported dataset types")