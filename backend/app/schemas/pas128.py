"""
PAS 128:2022 Compliance Checking Schemas

This module defines Pydantic schemas for PAS 128 compliance checking,
quality level determination, and compliance reporting.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class QualityLevel(str, Enum):
    """PAS 128 Quality Levels"""
    QL_D = "QL-D"
    QL_C = "QL-C"
    QL_B = "QL-B"
    QL_A = "QL-A"


class DetectionMethod(str, Enum):
    """Detection methods available for utility surveys"""
    ELECTROMAGNETIC = "electromagnetic"
    GROUND_PENETRATING_RADAR = "ground_penetrating_radar"
    RADIO_DETECTION = "radio_detection"
    VISUAL_INSPECTION = "visual_inspection"
    TOPOGRAPHICAL_SURVEY = "topographical_survey"


class SurveyMethod(str, Enum):
    """Survey methods for different quality levels"""
    RECORDS_SEARCH = "records_search"
    SITE_RECONNAISSANCE = "site_reconnaissance"
    COMPREHENSIVE_RECORDS = "comprehensive_records"
    TOPOGRAPHICAL_SURVEY = "topographical_survey"
    ELECTROMAGNETIC_DETECTION = "electromagnetic_detection"
    GROUND_PENETRATING_RADAR = "ground_penetrating_radar"
    TRIAL_HOLES = "trial_holes"
    VACUUM_EXCAVATION = "vacuum_excavation"
    HAND_DIGGING = "hand_digging"


class DeliverableType(str, Enum):
    """Required deliverables for each quality level"""
    SURVEY_REPORT = "survey_report"
    UTILITY_LOCATION_PLANS = "utility_location_plans"
    RISK_ASSESSMENT = "risk_assessment"
    DETECTION_SURVEY_RESULTS = "detection_survey_results"
    INTRUSIVE_INVESTIGATION_RESULTS = "intrusive_investigation_results"
    VERIFICATION_PHOTOS = "verification_photos"


class EnvironmentalCondition(BaseModel):
    """Environmental conditions affecting survey methods"""
    soil_type: str = Field(..., description="Primary soil type (clay, sand, gravel, etc.)")
    moisture_content: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture percentage")
    ground_conditions: List[str] = Field(default_factory=list, description="Ground condition factors")
    weather_conditions: Optional[str] = Field(None, description="Weather during survey")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    site_constraints: List[str] = Field(default_factory=list, description="Site-specific constraints")


class AccuracyMeasurement(BaseModel):
    """Accuracy measurement for utility detection"""
    horizontal_accuracy: float = Field(..., description="Horizontal accuracy in millimeters")
    vertical_accuracy: Optional[float] = Field(None, description="Vertical accuracy in millimeters")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    measurement_method: str = Field(..., description="Method used for accuracy measurement")


class DetectionEquipment(BaseModel):
    """Detection equipment specifications"""
    equipment_type: DetectionMethod
    model: str = Field(..., description="Equipment model/brand")
    frequency_range: Optional[str] = Field(None, description="Operating frequency range")
    calibration_date: Optional[datetime] = Field(None, description="Last calibration date")
    operator_certification: Optional[str] = Field(None, description="Operator certification level")


class MethodExecution(BaseModel):
    """Details of executed survey method"""
    method: SurveyMethod
    execution_date: datetime
    equipment_used: List[DetectionEquipment] = Field(default_factory=list)
    coverage_area: Optional[float] = Field(None, description="Coverage area in square meters")
    execution_quality: float = Field(..., ge=0, le=1, description="Execution quality score (0-1)")
    limitations_encountered: List[str] = Field(default_factory=list)
    results_summary: Optional[str] = Field(None, description="Summary of method results")


class DeliverableItem(BaseModel):
    """Individual deliverable item"""
    deliverable_type: DeliverableType
    provided: bool = Field(..., description="Whether deliverable was provided")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality score if provided")
    completeness_score: Optional[float] = Field(None, ge=0, le=1, description="Completeness score")
    file_path: Optional[str] = Field(None, description="Path to deliverable file")
    notes: Optional[str] = Field(None, description="Additional notes on deliverable")


class UtilityDetection(BaseModel):
    """Individual utility detection result"""
    utility_id: str = Field(..., description="Unique identifier for detected utility")
    utility_type: str = Field(..., description="Type of utility (gas, water, electric, etc.)")
    detection_method: DetectionMethod
    location: Dict[str, float] = Field(..., description="X, Y coordinates")
    depth: Optional[float] = Field(None, description="Depth in meters")
    accuracy: AccuracyMeasurement
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence (0-1)")
    verified: bool = Field(default=False, description="Whether detection was verified")
    verification_method: Optional[str] = Field(None, description="Method used for verification")


class SurveyData(BaseModel):
    """Core survey data for compliance checking"""
    survey_id: str = Field(..., description="Unique survey identifier")
    survey_date: datetime
    site_location: Dict[str, Any] = Field(..., description="Site location details")
    environmental_conditions: EnvironmentalCondition
    methods_executed: List[MethodExecution]
    deliverables: List[DeliverableItem]
    utility_detections: List[UtilityDetection] = Field(default_factory=list)
    target_quality_level: QualityLevel = Field(..., description="Target quality level for survey")
    survey_extent: Dict[str, Any] = Field(..., description="Survey boundary and extent information")


class QualityLevelRequirements(BaseModel):
    """Requirements for a specific quality level"""
    quality_level: QualityLevel
    required_methods: List[SurveyMethod]
    required_deliverables: List[DeliverableType]
    accuracy_requirements: Dict[str, float]  # horizontal, vertical accuracy in mm
    minimum_coverage: Optional[float] = Field(None, description="Minimum coverage percentage")


class ComplianceCheck(BaseModel):
    """Individual compliance check result"""
    check_name: str = Field(..., description="Name of compliance check")
    passed: bool = Field(..., description="Whether check passed")
    score: float = Field(..., ge=0, le=1, description="Compliance score (0-1)")
    details: str = Field(..., description="Detailed explanation of check result")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements checked")
    gaps: List[str] = Field(default_factory=list, description="Identified compliance gaps")


class QualityLevelAssessment(BaseModel):
    """Assessment of achievable quality level"""
    assessed_quality_level: QualityLevel
    confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    methods_compliance: Dict[SurveyMethod, bool] = Field(default_factory=dict)
    deliverables_compliance: Dict[DeliverableType, bool] = Field(default_factory=dict)
    accuracy_compliance: Dict[str, bool] = Field(default_factory=dict)
    limiting_factors: List[str] = Field(default_factory=list, description="Factors preventing higher quality level")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")


class EnvironmentalImpactAssessment(BaseModel):
    """Assessment of environmental factors on survey methods"""
    soil_impact_on_gpr: float = Field(..., ge=0, le=1, description="Impact of soil conditions on GPR effectiveness")
    weather_impact: float = Field(..., ge=0, le=1, description="Impact of weather on survey quality")
    site_constraints_impact: float = Field(..., ge=0, le=1, description="Impact of site constraints")
    overall_environmental_score: float = Field(..., ge=0, le=1, description="Overall environmental suitability")
    method_effectiveness: Dict[DetectionMethod, float] = Field(default_factory=dict)
    recommended_adjustments: List[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    """Comprehensive PAS 128 compliance report"""
    survey_id: str
    assessment_date: datetime = Field(default_factory=datetime.now)
    target_quality_level: QualityLevel
    achieved_quality_level: QualityLevel
    overall_compliance_score: float = Field(..., ge=0, le=1, description="Overall compliance score")

    # Detailed assessments
    quality_level_assessment: QualityLevelAssessment
    environmental_impact: EnvironmentalImpactAssessment
    compliance_checks: List[ComplianceCheck]

    # Method and deliverable analysis
    methods_analysis: Dict[SurveyMethod, Dict[str, Any]] = Field(default_factory=dict)
    deliverables_analysis: Dict[DeliverableType, Dict[str, Any]] = Field(default_factory=dict)
    accuracy_analysis: Dict[str, Any] = Field(default_factory=dict)

    # Recommendations and gaps
    critical_gaps: List[str] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

    # Compliance summary
    compliance_summary: Dict[str, Any] = Field(default_factory=dict)


class ComplianceRequest(BaseModel):
    """Request for PAS 128 compliance checking"""
    survey_data: SurveyData
    assessment_options: Dict[str, Any] = Field(default_factory=dict)
    include_recommendations: bool = Field(default=True)
    include_environmental_analysis: bool = Field(default=True)
    detailed_analysis: bool = Field(default=False)


class QualityLevelDeterminationRequest(BaseModel):
    """Request for automated quality level determination"""
    survey_data: SurveyData
    consider_environmental_factors: bool = Field(default=True)
    conservative_assessment: bool = Field(default=True, description="Use conservative quality level determination")


class BatchComplianceRequest(BaseModel):
    """Request for batch compliance checking of multiple surveys"""
    surveys: List[SurveyData]
    assessment_options: Dict[str, Any] = Field(default_factory=dict)
    generate_summary_report: bool = Field(default=True)


class BatchComplianceResponse(BaseModel):
    """Response for batch compliance checking"""
    total_surveys: int
    processed_surveys: int
    compliance_reports: List[ComplianceReport]
    summary_statistics: Dict[str, Any] = Field(default_factory=dict)
    batch_recommendations: List[str] = Field(default_factory=list)


# Validation schemas for API responses
class ComplianceCheckResponse(BaseModel):
    """Response for compliance check API endpoint"""
    success: bool
    compliance_report: Optional[ComplianceReport] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class QualityLevelResponse(BaseModel):
    """Response for quality level determination API endpoint"""
    success: bool
    quality_level_assessment: Optional[QualityLevelAssessment] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None