-- ================================================================
-- COMPREHENSIVE GROUND TRUTH VALIDATION SCHEMA
-- Underground Utility Detection Platform - Accuracy Assessment System
-- ================================================================

-- Validation method registry
CREATE TABLE validation_methods (
    id SERIAL PRIMARY KEY,
    method_code VARCHAR(20) UNIQUE NOT NULL,
    method_name VARCHAR(100) NOT NULL,
    description TEXT,
    accuracy_range_mm NUMRANGE, -- Expected accuracy range in millimeters
    cost_factor DECIMAL(5,2), -- Relative cost factor (1.0 = baseline)
    time_factor DECIMAL(5,2), -- Relative time factor (1.0 = baseline)
    invasiveness_level VARCHAR(20) CHECK (invasiveness_level IN ('non_invasive', 'minimally_invasive', 'invasive', 'highly_invasive')),
    pas128_recognized BOOLEAN DEFAULT TRUE
);

-- Insert validation methods
INSERT INTO validation_methods (method_code, method_name, description, accuracy_range_mm, cost_factor, time_factor, invasiveness_level, pas128_recognized) VALUES
('TRIAL_TRENCH', 'Trial Trench', 'Excavated trench for direct utility exposure', '[10,50)', 3.5, 2.8, 'highly_invasive', true),
('VACUUM_EXC', 'Vacuum Excavation', 'Pneumatic vacuum excavation around utilities', '[15,75)', 2.8, 2.2, 'invasive', true),
('HAND_DIG', 'Hand Digging', 'Manual excavation with hand tools', '[20,100)', 1.5, 3.5, 'invasive', true),
('POTHOLING', 'Potholing', 'Small exploratory excavations', '[25,100)', 2.0, 1.8, 'invasive', true),
('RECORDS_CHK', 'Records Check', 'Verification against utility records', '[500,2000)', 0.3, 0.5, 'non_invasive', true),
('EM_SURVEY', 'Electromagnetic Survey', 'Cable locator verification', '[100,500)', 0.8, 0.7, 'non_invasive', true),
('CCTV_INSP', 'CCTV Inspection', 'Camera inspection of accessible utilities', '[50,200)', 1.2, 1.0, 'minimally_invasive', true),
('CORE_DRILL', 'Core Drilling', 'Small diameter core samples', '[30,150)', 1.8, 1.5, 'invasive', true),
('PROBE_ROD', 'Probe Rods', 'Physical probing with metal rods', '[100,300)', 0.6, 0.8, 'minimally_invasive', false),
('UTILITY_MAP', 'Utility Mapping', 'Existing utility mapping verification', '[300,1500)', 0.2, 0.3, 'non_invasive', false);

-- ================================================================
-- ENHANCED GROUND TRUTH VALIDATION CAMPAIGNS
-- ================================================================

CREATE TABLE validation_campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Campaign identification
    campaign_name VARCHAR(255) NOT NULL,
    campaign_reference VARCHAR(100) UNIQUE,

    -- Campaign scope and objectives
    primary_objective TEXT NOT NULL,
    secondary_objectives TEXT[],
    target_accuracy_mm INTEGER,
    minimum_confidence_level DECIMAL(3,2) DEFAULT 0.95,

    -- Planning and execution
    planned_start_date DATE NOT NULL,
    planned_end_date DATE,
    actual_start_date DATE,
    actual_end_date DATE,

    -- Responsible parties
    campaign_manager VARCHAR(255),
    responsible_engineer VARCHAR(255),
    validation_team TEXT[], -- Array of team member names
    external_contractors TEXT[],

    -- Standards and protocols
    validation_standard VARCHAR(100) DEFAULT 'PAS128:2022',
    validation_protocol_document TEXT,
    quality_assurance_plan TEXT,

    -- Budget and resources
    allocated_budget DECIMAL(12,2),
    actual_cost DECIMAL(12,2),
    resource_constraints TEXT[],

    -- Results summary (automatically updated)
    total_planned_validations INTEGER DEFAULT 0,
    total_completed_validations INTEGER DEFAULT 0,
    total_correct_detections INTEGER DEFAULT 0,
    total_false_positives INTEGER DEFAULT 0,
    total_false_negatives INTEGER DEFAULT 0,
    total_missed_utilities INTEGER DEFAULT 0,

    -- Performance metrics (calculated)
    overall_accuracy_percentage DECIMAL(5,2),
    precision_percentage DECIMAL(5,2),
    recall_percentage DECIMAL(5,2),
    f1_score DECIMAL(5,4),

    -- Campaign status
    status VARCHAR(50) DEFAULT 'planned' CHECK (status IN ('planned', 'active', 'completed', 'cancelled', 'on_hold')),
    completion_notes TEXT,
    lessons_learned TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- DETAILED GROUND TRUTH VALIDATION RECORDS
-- ================================================================

CREATE TABLE ground_truth_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES validation_campaigns(id) ON DELETE CASCADE,
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    detected_utility_id UUID REFERENCES detected_utilities(id), -- NULL if validating missed detection

    -- Validation identification
    validation_reference VARCHAR(100),
    validation_sequence_number INTEGER,

    -- Validation location (precise coordinates)
    validation_point GEOMETRY(POINT, 4326) NOT NULL,
    osgb36_coordinates GEOMETRY(POINT, 27700),
    elevation_m DECIMAL(8,3),
    location_accuracy_estimate_mm INTEGER,

    -- Validation method details
    validation_method_id INTEGER NOT NULL REFERENCES validation_methods(id),
    secondary_validation_methods INTEGER[], -- Additional verification methods
    validation_depth_m DECIMAL(6,3),
    excavation_dimensions_m VARCHAR(100), -- "2.0x1.0x1.5" (LxWxD)

    -- Timing
    validation_date DATE NOT NULL,
    validation_start_time TIME,
    validation_end_time TIME,
    duration_hours DECIMAL(5,2),

    -- Personnel
    validator_name VARCHAR(255) NOT NULL,
    witness_names TEXT[],
    supervisor_name VARCHAR(255),

    -- == GROUND TRUTH FINDINGS ==
    utility_present BOOLEAN NOT NULL,

    -- If utility present
    actual_utility_type VARCHAR(100), -- water, sewer, gas, electric, telecom, etc.
    actual_utility_subtype VARCHAR(100), -- main, service, distribution
    actual_material VARCHAR(100),
    actual_diameter_mm INTEGER,
    actual_depth_m DECIMAL(6,3),
    actual_orientation_degrees DECIMAL(5,2), -- 0-360 degrees from north
    actual_condition VARCHAR(50), -- good, fair, poor, damaged

    -- Multi-utility scenarios
    additional_utilities_found INTEGER DEFAULT 0,
    bundle_configuration BOOLEAN DEFAULT FALSE,
    utility_spacing_mm INTEGER, -- For bundled utilities

    -- Physical measurements
    depth_measurement_points JSONB, -- Array of depth measurements
    depth_measurement_accuracy_mm INTEGER,
    diameter_measurement_method VARCHAR(100),
    material_identification_method VARCHAR(100),

    -- == ACCURACY ASSESSMENT ==
    -- Horizontal position accuracy
    horizontal_position_error_mm INTEGER,
    horizontal_position_error_direction_degrees DECIMAL(5,2),

    -- Depth accuracy
    depth_error_mm INTEGER, -- Positive = deeper than predicted, negative = shallower
    depth_error_percentage DECIMAL(5,2),

    -- Detection quality assessment
    detection_result VARCHAR(50) CHECK (detection_result IN ('true_positive', 'false_positive', 'false_negative', 'true_negative')),
    detection_confidence_assessment VARCHAR(50), -- over_confident, under_confident, appropriate

    -- Material and size accuracy
    material_identification_correct BOOLEAN,
    diameter_error_mm INTEGER,
    diameter_error_percentage DECIMAL(5,2),

    -- == VALIDATION CONFIDENCE AND QUALITY ==
    validation_confidence VARCHAR(20) CHECK (validation_confidence IN ('very_low', 'low', 'medium', 'high', 'very_high')),
    validation_quality_score DECIMAL(3,2) CHECK (validation_quality_score >= 0 AND validation_quality_score <= 1),
    measurement_uncertainty_mm INTEGER,

    -- Factors affecting validation quality
    access_difficulties TEXT[],
    weather_conditions VARCHAR(100),
    ground_conditions_encountered VARCHAR(100),
    interference_factors TEXT[],

    -- == DOCUMENTATION ==
    photo_references TEXT[], -- File paths to photos
    video_references TEXT[], -- File paths to videos
    measurement_sketches TEXT[], -- File paths to sketches/drawings
    field_notes TEXT,
    survey_drawings TEXT[], -- As-built drawings

    -- == SAFETY AND COMPLIANCE ==
    safety_incidents TEXT[],
    environmental_precautions TEXT[],
    permits_required TEXT[],
    utility_company_notifications BOOLEAN DEFAULT FALSE,

    -- == ADDITIONAL FINDINGS ==
    unexpected_discoveries TEXT[],
    utility_condition_issues TEXT[],
    recommended_follow_up_actions TEXT[],

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- ACCURACY ASSESSMENT AGGREGATION
-- ================================================================

CREATE TABLE accuracy_assessment_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES validation_campaigns(id) ON DELETE CASCADE,

    -- Assessment period
    assessment_date DATE DEFAULT CURRENT_DATE,
    data_cutoff_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Sample size
    total_validations INTEGER NOT NULL,
    valid_detections INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    true_negatives INTEGER,

    -- Overall accuracy metrics
    overall_accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    specificity DECIMAL(5,4),

    -- Spatial accuracy metrics
    mean_horizontal_error_mm DECIMAL(7,2),
    std_horizontal_error_mm DECIMAL(7,2),
    median_horizontal_error_mm DECIMAL(7,2),
    horizontal_error_95_percentile_mm DECIMAL(7,2),

    mean_depth_error_mm DECIMAL(7,2),
    std_depth_error_mm DECIMAL(7,2),
    median_depth_error_mm DECIMAL(7,2),
    depth_error_95_percentile_mm DECIMAL(7,2),

    -- Depth-based performance
    depth_accuracy_by_range JSONB, -- Performance by depth ranges
    shallow_utilities_accuracy DECIMAL(5,4), -- 0-1m
    medium_utilities_accuracy DECIMAL(5,4), -- 1-2m
    deep_utilities_accuracy DECIMAL(5,4), -- >2m

    -- Utility type performance
    utility_type_accuracy JSONB, -- Performance by utility type

    -- Environmental factor correlations
    ground_condition_performance JSONB,
    weather_impact_analysis JSONB,

    -- PAS 128 compliance assessment
    pas128_ql_a_compliance BOOLEAN DEFAULT FALSE, -- ±40mm
    pas128_ql_b_compliance BOOLEAN DEFAULT FALSE, -- ±200mm
    pas128_ql_c_compliance BOOLEAN DEFAULT FALSE, -- ±500mm
    pas128_ql_d_compliance BOOLEAN DEFAULT FALSE, -- ±1000mm
    recommended_quality_level VARCHAR(10),

    -- Statistical confidence
    confidence_interval_95_lower DECIMAL(5,4),
    confidence_interval_95_upper DECIMAL(5,4),
    sample_size_adequate BOOLEAN,
    recommended_additional_samples INTEGER,

    -- Assessment metadata
    assessment_method VARCHAR(100),
    statistical_tests_applied TEXT[],
    assessment_notes TEXT,
    assessed_by VARCHAR(255),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- VALIDATION METHOD EFFECTIVENESS TRACKING
-- ================================================================

CREATE TABLE validation_method_effectiveness (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_method_id INTEGER NOT NULL REFERENCES validation_methods(id),

    -- Analysis period
    analysis_start_date DATE NOT NULL,
    analysis_end_date DATE NOT NULL,
    total_validations_analyzed INTEGER,

    -- Effectiveness metrics
    average_validation_time_hours DECIMAL(5,2),
    average_cost_per_validation DECIMAL(10,2),
    success_rate DECIMAL(5,4), -- Rate of successful validations

    -- Accuracy metrics by this method
    mean_measurement_error_mm DECIMAL(7,2),
    measurement_precision_mm DECIMAL(7,2),
    measurement_bias_mm DECIMAL(7,2),

    -- Conditions analysis
    optimal_ground_conditions TEXT[],
    challenging_ground_conditions TEXT[],
    weather_limitations TEXT[],
    depth_limitations_m DECIMAL(6,3),

    -- Recommendations
    recommended_use_cases TEXT[],
    not_recommended_conditions TEXT[],
    alternative_methods_suggested INTEGER[],

    -- Cost-benefit analysis
    cost_effectiveness_score DECIMAL(5,4),
    time_effectiveness_score DECIMAL(5,4),
    overall_effectiveness_score DECIMAL(5,4),

    -- Analysis metadata
    analysis_method VARCHAR(100),
    analyst_name VARCHAR(255),
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- UTILITY-SPECIFIC VALIDATION PERFORMANCE
-- ================================================================

CREATE TABLE utility_type_validation_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    utility_discipline VARCHAR(50) NOT NULL,
    utility_material VARCHAR(100),
    diameter_range_mm NUMRANGE,

    -- Performance metrics
    detection_rate DECIMAL(5,4),
    false_positive_rate DECIMAL(5,4),
    position_accuracy_mm DECIMAL(7,2),
    depth_accuracy_mm DECIMAL(7,2),

    -- Material identification accuracy
    material_identification_rate DECIMAL(5,4),
    diameter_estimation_accuracy DECIMAL(5,4),

    -- Depth-dependent performance
    shallow_detection_rate DECIMAL(5,4), -- 0-1m
    medium_detection_rate DECIMAL(5,4), -- 1-2m
    deep_detection_rate DECIMAL(5,4), -- >2m

    -- Environmental factor impacts
    dry_ground_performance DECIMAL(5,4),
    wet_ground_performance DECIMAL(5,4),
    contaminated_ground_performance DECIMAL(5,4),

    -- Sample statistics
    total_samples INTEGER,
    validation_campaigns_included TEXT[],
    analysis_date_range TSRANGE,

    -- Statistical confidence
    confidence_level DECIMAL(3,2) DEFAULT 0.95,
    margin_of_error DECIMAL(5,4),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- ================================================================

-- Validation campaigns indexes
CREATE INDEX idx_validation_campaigns_project ON validation_campaigns(project_id);
CREATE INDEX idx_validation_campaigns_dates ON validation_campaigns(actual_start_date, actual_end_date);
CREATE INDEX idx_validation_campaigns_status ON validation_campaigns(status);

-- Ground truth validations indexes
CREATE INDEX idx_ground_truth_campaign ON ground_truth_validations(campaign_id);
CREATE INDEX idx_ground_truth_survey ON ground_truth_validations(survey_id);
CREATE INDEX idx_ground_truth_detected_utility ON ground_truth_validations(detected_utility_id);
CREATE INDEX idx_ground_truth_method ON ground_truth_validations(validation_method_id);
CREATE INDEX idx_ground_truth_date ON ground_truth_validations(validation_date);
CREATE INDEX idx_ground_truth_result ON ground_truth_validations(detection_result);
CREATE INDEX idx_ground_truth_utility_type ON ground_truth_validations(actual_utility_type);

-- Spatial indexes
CREATE INDEX idx_ground_truth_location ON ground_truth_validations USING GIST(validation_point);
CREATE INDEX idx_ground_truth_osgb36 ON ground_truth_validations USING GIST(osgb36_coordinates);

-- Accuracy assessment indexes
CREATE INDEX idx_accuracy_assessment_campaign ON accuracy_assessment_summary(campaign_id);
CREATE INDEX idx_accuracy_assessment_date ON accuracy_assessment_summary(assessment_date);

-- Performance tracking indexes
CREATE INDEX idx_method_effectiveness_method ON validation_method_effectiveness(validation_method_id);
CREATE INDEX idx_utility_performance_type ON utility_type_validation_performance(utility_discipline);
CREATE INDEX idx_utility_performance_material ON utility_type_validation_performance(utility_material);

-- ================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE ANALYSIS
-- ================================================================

-- Validation campaign performance summary
CREATE MATERIALIZED VIEW validation_performance_summary AS
SELECT
    vc.id as campaign_id,
    vc.campaign_name,
    vc.validation_standard,
    COUNT(gtv.id) as total_validations,
    COUNT(CASE WHEN gtv.detection_result = 'true_positive' THEN 1 END) as true_positives,
    COUNT(CASE WHEN gtv.detection_result = 'false_positive' THEN 1 END) as false_positives,
    COUNT(CASE WHEN gtv.detection_result = 'false_negative' THEN 1 END) as false_negatives,
    ROUND(AVG(gtv.horizontal_position_error_mm), 2) as avg_horizontal_error_mm,
    ROUND(AVG(gtv.depth_error_mm), 2) as avg_depth_error_mm,
    ROUND(STDDEV(gtv.horizontal_position_error_mm), 2) as std_horizontal_error_mm,
    ROUND(STDDEV(gtv.depth_error_mm), 2) as std_depth_error_mm,
    COUNT(CASE WHEN gtv.validation_confidence IN ('high', 'very_high') THEN 1 END) as high_confidence_validations
FROM validation_campaigns vc
LEFT JOIN ground_truth_validations gtv ON vc.id = gtv.campaign_id
GROUP BY vc.id, vc.campaign_name, vc.validation_standard;

-- PAS 128 compliance assessment view
CREATE MATERIALIZED VIEW pas128_compliance_assessment AS
SELECT
    vc.id as campaign_id,
    vc.campaign_name,
    COUNT(gtv.id) as total_validations,
    -- QL-A: ±40mm horizontal, ±75mm depth
    COUNT(CASE WHEN ABS(gtv.horizontal_position_error_mm) <= 40 AND ABS(gtv.depth_error_mm) <= 75 THEN 1 END) as ql_a_compliant,
    -- QL-B: ±200mm horizontal, ±300mm depth
    COUNT(CASE WHEN ABS(gtv.horizontal_position_error_mm) <= 200 AND ABS(gtv.depth_error_mm) <= 300 THEN 1 END) as ql_b_compliant,
    -- QL-C: ±500mm horizontal, ±500mm depth
    COUNT(CASE WHEN ABS(gtv.horizontal_position_error_mm) <= 500 AND ABS(gtv.depth_error_mm) <= 500 THEN 1 END) as ql_c_compliant,
    -- QL-D: ±1000mm horizontal, ±1000mm depth
    COUNT(CASE WHEN ABS(gtv.horizontal_position_error_mm) <= 1000 AND ABS(gtv.depth_error_mm) <= 1000 THEN 1 END) as ql_d_compliant,
    ROUND(
        CAST(COUNT(CASE WHEN ABS(gtv.horizontal_position_error_mm) <= 40 AND ABS(gtv.depth_error_mm) <= 75 THEN 1 END) AS DECIMAL) /
        NULLIF(COUNT(gtv.id), 0) * 100, 2
    ) as ql_a_percentage,
    ROUND(
        CAST(COUNT(CASE WHEN ABS(gtv.horizontal_position_error_mm) <= 200 AND ABS(gtv.depth_error_mm) <= 300 THEN 1 END) AS DECIMAL) /
        NULLIF(COUNT(gtv.id), 0) * 100, 2
    ) as ql_b_percentage
FROM validation_campaigns vc
LEFT JOIN ground_truth_validations gtv ON vc.id = gtv.campaign_id
WHERE gtv.utility_present = true
GROUP BY vc.id, vc.campaign_name;

-- ================================================================
-- UTILITY FUNCTIONS FOR ACCURACY ASSESSMENT
-- ================================================================

-- Function to calculate detection performance metrics
CREATE OR REPLACE FUNCTION calculate_detection_metrics(
    campaign_uuid UUID
) RETURNS TABLE (
    accuracy DECIMAL,
    precision_val DECIMAL,
    recall_val DECIMAL,
    f1_score DECIMAL,
    specificity DECIMAL
) AS $$
DECLARE
    tp INTEGER; -- True positives
    fp INTEGER; -- False positives
    fn INTEGER; -- False negatives
    tn INTEGER; -- True negatives
BEGIN
    SELECT
        COUNT(CASE WHEN detection_result = 'true_positive' THEN 1 END),
        COUNT(CASE WHEN detection_result = 'false_positive' THEN 1 END),
        COUNT(CASE WHEN detection_result = 'false_negative' THEN 1 END),
        COUNT(CASE WHEN detection_result = 'true_negative' THEN 1 END)
    INTO tp, fp, fn, tn
    FROM ground_truth_validations
    WHERE campaign_id = campaign_uuid;

    RETURN QUERY SELECT
        ROUND(CAST(tp + tn AS DECIMAL) / NULLIF(tp + fp + fn + tn, 0), 4) as accuracy,
        ROUND(CAST(tp AS DECIMAL) / NULLIF(tp + fp, 0), 4) as precision_val,
        ROUND(CAST(tp AS DECIMAL) / NULLIF(tp + fn, 0), 4) as recall_val,
        ROUND(2.0 * CAST(tp AS DECIMAL) / NULLIF(2 * tp + fp + fn, 0), 4) as f1_score,
        ROUND(CAST(tn AS DECIMAL) / NULLIF(tn + fp, 0), 4) as specificity;
END;
$$ LANGUAGE plpgsql;

-- Function to update campaign summary statistics
CREATE OR REPLACE FUNCTION update_campaign_statistics(campaign_uuid UUID)
RETURNS void AS $$
DECLARE
    metrics RECORD;
BEGIN
    SELECT * INTO metrics FROM calculate_detection_metrics(campaign_uuid);

    UPDATE validation_campaigns SET
        total_completed_validations = (
            SELECT COUNT(*) FROM ground_truth_validations WHERE campaign_id = campaign_uuid
        ),
        total_correct_detections = (
            SELECT COUNT(*) FROM ground_truth_validations
            WHERE campaign_id = campaign_uuid AND detection_result = 'true_positive'
        ),
        total_false_positives = (
            SELECT COUNT(*) FROM ground_truth_validations
            WHERE campaign_id = campaign_uuid AND detection_result = 'false_positive'
        ),
        total_false_negatives = (
            SELECT COUNT(*) FROM ground_truth_validations
            WHERE campaign_id = campaign_uuid AND detection_result = 'false_negative'
        ),
        overall_accuracy_percentage = metrics.accuracy * 100,
        precision_percentage = metrics.precision_val * 100,
        recall_percentage = metrics.recall_val * 100,
        f1_score = metrics.f1_score,
        updated_at = NOW()
    WHERE id = campaign_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to refresh validation analysis views
CREATE OR REPLACE FUNCTION refresh_validation_analysis_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW validation_performance_summary;
    REFRESH MATERIALIZED VIEW pas128_compliance_assessment;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ================================================================

-- Trigger to update campaign statistics when validation records change
CREATE OR REPLACE FUNCTION trigger_update_campaign_statistics()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        PERFORM update_campaign_statistics(NEW.campaign_id);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM update_campaign_statistics(OLD.campaign_id);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_campaign_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON ground_truth_validations
    FOR EACH ROW EXECUTE FUNCTION trigger_update_campaign_statistics();

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE validation_campaigns IS 'Comprehensive validation campaign management with planning, execution, and results tracking';
COMMENT ON TABLE ground_truth_validations IS 'Detailed ground truth validation records with precise accuracy assessment';
COMMENT ON TABLE accuracy_assessment_summary IS 'Aggregated accuracy metrics and statistical analysis for validation campaigns';
COMMENT ON TABLE validation_method_effectiveness IS 'Comparative analysis of different validation methods effectiveness';
COMMENT ON TABLE utility_type_validation_performance IS 'Performance metrics broken down by utility type and characteristics';

COMMENT ON MATERIALIZED VIEW validation_performance_summary IS 'Summary view of validation campaign performance metrics';
COMMENT ON MATERIALIZED VIEW pas128_compliance_assessment IS 'PAS 128 standard compliance assessment for quality level determination';