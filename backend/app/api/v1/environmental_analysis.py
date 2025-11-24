"""
Environmental Analysis API Endpoints
====================================

API endpoints for comprehensive environmental correlation analysis
of GPR utility detection data using the Twente dataset.
"""

from typing import Dict, Any, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.logging_config import LoggerMixin
from ...services.comprehensive_environmental_analyzer import ComprehensiveEnvironmentalAnalyzer
from ...schemas.environmental import (
    EnvironmentalAnalysisRequest,
    EnvironmentalAnalysisResponse,
    AnalysisConfig
)

router = APIRouter(prefix="/environmental-analysis", tags=["Environmental Analysis"])


class EnvironmentalAnalysisAPI(LoggerMixin):
    """API handler for environmental analysis operations."""

    def __init__(self):
        super().__init__()
        self.analyzer = ComprehensiveEnvironmentalAnalyzer()


analysis_api = EnvironmentalAnalysisAPI()


@router.post("/comprehensive-analysis", response_model=EnvironmentalAnalysisResponse)
async def perform_comprehensive_environmental_analysis(
    request: EnvironmentalAnalysisRequest,
    db: AsyncSession = Depends(get_db)
) -> EnvironmentalAnalysisResponse:
    """
    Perform comprehensive environmental correlation analysis on Twente GPR dataset.

    This endpoint provides:
    - Environmental factor correlation analysis
    - Material classification system
    - Utility detection performance prediction
    - Statistical significance testing
    - Feature importance ranking
    - Actionable insights and recommendations

    Args:
        request: Analysis request with configuration parameters
        db: Database session

    Returns:
        Comprehensive analysis results with insights and recommendations
    """
    try:
        analysis_api.log_operation_start("comprehensive_environmental_analysis_api")

        # Validate request parameters
        if not request.metadata_csv_path and not request.metadata_data:
            raise HTTPException(
                status_code=400,
                detail="Either metadata_csv_path or metadata_data must be provided"
            )

        # Load metadata
        if request.metadata_csv_path:
            try:
                metadata_df = pd.read_csv(request.metadata_csv_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading metadata CSV: {str(e)}"
                )
        else:
            metadata_df = pd.DataFrame(request.metadata_data)

        # Load performance data if provided
        performance_df = None
        if request.performance_csv_path:
            try:
                performance_df = pd.read_csv(request.performance_csv_path)
            except Exception as e:
                analysis_api.logger.warning(f"Could not load performance data: {str(e)}")
        elif request.performance_data:
            performance_df = pd.DataFrame(request.performance_data)

        # Prepare analysis configuration
        analysis_config = request.analysis_config.dict() if request.analysis_config else None

        # Perform comprehensive analysis
        analysis_results = await analysis_api.analyzer.perform_comprehensive_analysis(
            metadata_df=metadata_df,
            performance_data=performance_df,
            analysis_config=analysis_config
        )

        # Prepare response
        response = EnvironmentalAnalysisResponse(
            success=True,
            message="Comprehensive environmental analysis completed successfully",
            analysis_results=analysis_results,
            data_summary={
                "total_samples": len(metadata_df),
                "total_features": len(metadata_df.columns),
                "analysis_timestamp": analysis_results.get("analysis_metadata", {}).get("timestamp")
            }
        )

        analysis_api.log_operation_complete("comprehensive_environmental_analysis_api", len(analysis_results))
        return response

    except HTTPException:
        raise
    except Exception as e:
        analysis_api.log_operation_error("comprehensive_environmental_analysis_api", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during analysis: {str(e)}"
        )


@router.post("/upload-analysis")
async def upload_and_analyze_dataset(
    metadata_file: UploadFile = File(..., description="Twente metadata CSV file"),
    performance_file: Optional[UploadFile] = File(None, description="Optional performance data CSV"),
    config: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> EnvironmentalAnalysisResponse:
    """
    Upload Twente dataset files and perform comprehensive environmental analysis.

    Args:
        metadata_file: CSV file containing Twente metadata with 25+ fields
        performance_file: Optional CSV file with performance metrics
        config: Optional JSON string with analysis configuration
        db: Database session

    Returns:
        Comprehensive analysis results
    """
    try:
        analysis_api.log_operation_start("upload_and_analyze_dataset")

        # Validate file types
        if not metadata_file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Metadata file must be a CSV file"
            )

        if performance_file and not performance_file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Performance file must be a CSV file"
            )

        # Read metadata file
        try:
            metadata_content = await metadata_file.read()
            metadata_df = pd.read_csv(pd.io.common.StringIO(metadata_content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading metadata file: {str(e)}"
            )

        # Read performance file if provided
        performance_df = None
        if performance_file:
            try:
                performance_content = await performance_file.read()
                performance_df = pd.read_csv(pd.io.common.StringIO(performance_content.decode('utf-8')))
            except Exception as e:
                analysis_api.logger.warning(f"Could not read performance file: {str(e)}")

        # Parse configuration if provided
        analysis_config = None
        if config:
            try:
                import json
                analysis_config = json.loads(config)
            except Exception as e:
                analysis_api.logger.warning(f"Could not parse config: {str(e)}")

        # Perform analysis
        analysis_results = await analysis_api.analyzer.perform_comprehensive_analysis(
            metadata_df=metadata_df,
            performance_data=performance_df,
            analysis_config=analysis_config
        )

        # Create response
        response = EnvironmentalAnalysisResponse(
            success=True,
            message=f"Analysis completed for {metadata_file.filename}",
            analysis_results=analysis_results,
            data_summary={
                "metadata_file": metadata_file.filename,
                "performance_file": performance_file.filename if performance_file else None,
                "total_samples": len(metadata_df),
                "total_features": len(metadata_df.columns),
                "analysis_timestamp": analysis_results.get("analysis_metadata", {}).get("timestamp")
            }
        )

        analysis_api.log_operation_complete("upload_and_analyze_dataset", len(analysis_results))
        return response

    except HTTPException:
        raise
    except Exception as e:
        analysis_api.log_operation_error("upload_and_analyze_dataset", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during file upload and analysis: {str(e)}"
        )


@router.get("/analysis-capabilities")
async def get_analysis_capabilities() -> Dict[str, Any]:
    """
    Get comprehensive list of environmental analysis capabilities.

    Returns:
        Dictionary describing all available analysis features
    """
    return {
        "environmental_factors_analyzed": [
            "Weather conditions (Dry, Rainy, Cloudy)",
            "Ground conditions (Sandy, Clayey, Mixed)",
            "Ground relative permittivity (8.16-19.46 range)",
            "Terrain characteristics (levelling, smoothness)",
            "Land use and land cover types",
            "Contamination factors (rubble, tree roots, pollution, slag)",
            "Utility density and configuration",
            "Material types (10+ different materials)",
            "Utility disciplines (electricity, water, sewer, telecom, gas)",
            "Survey context and objectives"
        ],
        "analysis_methods": [
            "Pearson and Spearman correlation analysis",
            "Principal Component Analysis (PCA)",
            "Factor Analysis",
            "K-Means and DBSCAN clustering",
            "Random Forest feature importance",
            "Statistical significance testing (ANOVA, Chi-square)",
            "Multiple comparison corrections (Bonferroni, FDR)",
            "Environmental impact quantification"
        ],
        "prediction_models": [
            "Environmental impact prediction",
            "Detection difficulty prediction",
            "Signal quality prediction",
            "Material classification prediction",
            "Optimal survey conditions prediction"
        ],
        "material_analysis": [
            "Material property analysis (conductivity, detectability, age)",
            "Material-environment interaction analysis",
            "Corrosion risk assessment",
            "Age-based material classification",
            "Conductivity-based grouping"
        ],
        "output_features": [
            "Statistical correlation matrices",
            "Feature importance rankings",
            "Environmental clustering results",
            "Predictive model performance metrics",
            "Actionable insights and recommendations",
            "Statistical significance reports",
            "Impact quantification scores"
        ],
        "supported_datasets": [
            "University of Twente GPR Metadata (25+ fields)",
            "Optional performance metrics data",
            "Custom environmental datasets with similar structure"
        ]
    }


@router.get("/sample-config")
async def get_sample_analysis_config() -> Dict[str, Any]:
    """
    Get sample analysis configuration for comprehensive environmental analysis.

    Returns:
        Sample configuration dictionary with explanations
    """
    return {
        "sample_config": {
            "correlation_threshold": 0.4,
            "significance_level": 0.05,
            "pca_components": 10,
            "factor_count": 5,
            "n_clusters": 6,
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 5,
            "cross_validation_folds": 5,
            "random_state": 42,
            "feature_selection_k": 15,
            "multiple_comparison_method": "bonferroni",
            "enable_advanced_clustering": True,
            "enable_material_analysis": True,
            "enable_performance_prediction": True
        },
        "parameter_explanations": {
            "correlation_threshold": "Minimum correlation coefficient to consider significant (0.0-1.0)",
            "significance_level": "P-value threshold for statistical significance (typically 0.05)",
            "pca_components": "Number of principal components to extract",
            "factor_count": "Number of factors for factor analysis",
            "n_clusters": "Number of clusters for K-means clustering",
            "dbscan_eps": "DBSCAN epsilon parameter for density clustering",
            "dbscan_min_samples": "Minimum samples per DBSCAN cluster",
            "cross_validation_folds": "Number of folds for cross-validation",
            "random_state": "Random seed for reproducible results",
            "feature_selection_k": "Number of top features to select",
            "multiple_comparison_method": "Method for multiple comparison correction",
            "enable_advanced_clustering": "Enable additional clustering algorithms",
            "enable_material_analysis": "Enable detailed material property analysis",
            "enable_performance_prediction": "Enable predictive modeling"
        }
    }


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for environmental analysis service."""
    return {
        "status": "healthy",
        "service": "Environmental Analysis API",
        "version": "1.0.0",
        "analyzer": "ComprehensiveEnvironmentalAnalyzer"
    }