-- ================================================================
-- Underground Utility Detection Platform - Phase 1A Database Schema
-- GPR Infrastructure and Feature Analysis Database Design
-- ================================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ================================================================
-- CORE PROJECT MANAGEMENT TABLES
-- ================================================================

-- Projects table for organizing surveys and datasets
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    client_organization VARCHAR(255),
    project_manager VARCHAR(255),
    start_date DATE,
    end_date DATE,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived', 'cancelled')),
    compliance_standards TEXT[], -- ['PAS128:2022', 'CDM2015']
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sites/Locations within projects
CREATE TABLE survey_sites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    site_name VARCHAR(255) NOT NULL,
    address TEXT,
    location GEOMETRY(POINT, 4326), -- WGS84 coordinates
    osgb36_coordinates GEOMETRY(POINT, 27700), -- UK National Grid
    site_boundary GEOMETRY(POLYGON, 4326),
    land_use VARCHAR(100), -- From Twente metadata categories
    land_type VARCHAR(100), -- Sidewalk, Street, Greenery, etc.
    ground_condition VARCHAR(100), -- Sandy, Clayey, etc.
    ground_relative_permittivity DECIMAL(5,2),
    relative_groundwater_level VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- GPR SURVEY DATA TABLES - TWENTE DATASET INTEGRATION
-- ================================================================

-- Main GPR surveys table with comprehensive metadata from Twente dataset
CREATE TABLE gpr_surveys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    site_id UUID NOT NULL REFERENCES survey_sites(id) ON DELETE CASCADE,
    location_id VARCHAR(50) NOT NULL, -- From Twente: 01.1, 01.2, etc.

    -- Survey objectives and context (25+ metadata fields from Twente)
    utility_surveying_objective VARCHAR(100) NOT NULL,
    construction_workers VARCHAR(100),
    exact_location_accuracy_required BOOLEAN DEFAULT FALSE,
    complementary_works TEXT[],

    -- Environmental conditions
    land_cover VARCHAR(100), -- Brick road concrete, Grass/vegetation, etc.
    terrain_levelling VARCHAR(50), -- Flat, Steep
    terrain_smoothness VARCHAR(50), -- Smooth, Rough
    weather_condition VARCHAR(50), -- Dry, Rainy

    -- Ground contamination indicators
    rubble_presence BOOLEAN DEFAULT FALSE,
    tree_roots_presence BOOLEAN DEFAULT FALSE,
    polluted_soil_presence BOOLEAN DEFAULT FALSE,
    blast_furnace_slag_presence BOOLEAN DEFAULT FALSE,

    -- Utility infrastructure characteristics
    amount_of_utilities INTEGER DEFAULT 0,
    utility_crossing BOOLEAN DEFAULT FALSE,
    utility_path_linear BOOLEAN DEFAULT TRUE,

    -- Survey metadata
    survey_date DATE NOT NULL,
    surveyor_name VARCHAR(255),
    equipment_model VARCHAR(255) DEFAULT 'GPR 500MHz',
    antenna_frequency INTEGER DEFAULT 500, -- MHz
    trace_spacing DECIMAL(4,2) DEFAULT 0.02, -- meters
    samples_per_trace INTEGER DEFAULT 512,
    time_range_ns INTEGER DEFAULT 50, -- nanoseconds

    -- Quality and validation
    quality_level VARCHAR(10), -- PAS128 QL-A to QL-D
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    validation_method VARCHAR(100), -- trial_trench, excavation, etc.

    -- File references
    radargram_files TEXT[], -- SEG-Y file paths
    survey_map_path TEXT,
    ground_truth_map_path TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- GPR Signal Processing Results
CREATE TABLE gpr_signal_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    trace_number INTEGER NOT NULL,
    sample_number INTEGER NOT NULL,

    -- Raw signal characteristics
    amplitude DECIMAL(10,6) NOT NULL,
    time_ns DECIMAL(8,3) NOT NULL, -- nanoseconds
    depth_estimate_m DECIMAL(6,3), -- meters, calculated from velocity model

    -- Processed signal features
    frequency_dominant_mhz DECIMAL(6,2),
    signal_strength_db DECIMAL(6,2),
    noise_level_db DECIMAL(6,2),
    snr_ratio DECIMAL(6,2), -- Signal-to-noise ratio

    -- Processing flags
    time_zero_corrected BOOLEAN DEFAULT FALSE,
    background_removed BOOLEAN DEFAULT FALSE,
    gain_corrected BOOLEAN DEFAULT FALSE,
    filtered BOOLEAN DEFAULT FALSE,

    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(survey_id, trace_number, sample_number)
);

-- Index for efficient signal data queries
CREATE INDEX idx_gpr_signal_survey_trace ON gpr_signal_data(survey_id, trace_number);
CREATE INDEX idx_gpr_signal_depth ON gpr_signal_data(depth_estimate_m) WHERE depth_estimate_m IS NOT NULL;

-- ================================================================
-- UTILITY DETECTION AND CLASSIFICATION TABLES
-- ================================================================

-- Detected utilities from GPR analysis
CREATE TABLE detected_utilities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Spatial information
    detection_line VARCHAR(50), -- Survey line identifier
    position_along_line_m DECIMAL(8,3) NOT NULL,
    depth_m DECIMAL(6,3) NOT NULL,
    coordinates GEOMETRY(POINT, 4326),

    -- Utility characteristics from Twente dataset
    utility_discipline VARCHAR(50) NOT NULL, -- electricity, water, sewer, telecommunications, oilGasChemicals
    utility_material VARCHAR(100), -- From Twente: steel, polyVinylChloride, asbestosCement, etc.
    utility_diameter_mm INTEGER,
    additional_info TEXT, -- bundle, cover, manhole, etc.

    -- Detection confidence and validation
    detection_method VARCHAR(100), -- hyperbola_detection, amplitude_analysis, etc.
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    false_positive_probability DECIMAL(3,2),

    -- Ground truth validation (when available)
    ground_truth_confirmed BOOLEAN,
    ground_truth_depth_m DECIMAL(6,3),
    ground_truth_material VARCHAR(100),
    ground_truth_diameter_mm INTEGER,
    validation_method VARCHAR(100), -- trial_trench, excavation, records_check

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Utility crossing analysis
CREATE TABLE utility_crossings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    utility1_id UUID NOT NULL REFERENCES detected_utilities(id),
    utility2_id UUID NOT NULL REFERENCES detected_utilities(id),

    crossing_type VARCHAR(50), -- perpendicular, parallel, bundle
    separation_distance_m DECIMAL(6,3),
    depth_difference_m DECIMAL(6,3),
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- MOJAHID IMAGE DATASET INTEGRATION
-- ================================================================

-- GPR Image Classification Data
CREATE TABLE gpr_image_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Image metadata
    image_filename VARCHAR(255) NOT NULL,
    image_path TEXT NOT NULL,
    image_category VARCHAR(50) NOT NULL, -- cavities, utilities, intact, augmented_cavities, etc.
    image_type VARCHAR(20) DEFAULT 'original', -- original, augmented

    -- Image characteristics
    width_pixels INTEGER,
    height_pixels INTEGER,
    bit_depth INTEGER DEFAULT 8,
    color_space VARCHAR(20) DEFAULT 'grayscale',

    -- ML Classification results
    predicted_class VARCHAR(50),
    prediction_confidence DECIMAL(3,2),
    feature_vector vector(512), -- For similarity searches

    -- Ground truth labels
    ground_truth_class VARCHAR(50),
    manually_verified BOOLEAN DEFAULT FALSE,
    verification_date TIMESTAMP WITH TIME ZONE,
    verified_by VARCHAR(255),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Image feature annotations for object detection
CREATE TABLE image_annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES gpr_image_data(id) ON DELETE CASCADE,

    -- Bounding box coordinates (normalized 0-1)
    bbox_x_min DECIMAL(6,4) CHECK (bbox_x_min >= 0 AND bbox_x_min <= 1),
    bbox_y_min DECIMAL(6,4) CHECK (bbox_y_min >= 0 AND bbox_y_min <= 1),
    bbox_x_max DECIMAL(6,4) CHECK (bbox_x_max >= 0 AND bbox_x_max <= 1),
    bbox_y_max DECIMAL(6,4) CHECK (bbox_y_max >= 0 AND bbox_y_max <= 1),

    -- Annotation details
    annotation_class VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2),
    annotator VARCHAR(255),
    annotation_method VARCHAR(50), -- manual, automated, semi_automated

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- FEATURE EXTRACTION AND ANALYSIS TABLES
-- ================================================================

-- Extracted features from GPR signals
CREATE TABLE signal_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Hyperbola detection features
    hyperbola_apex_depth_m DECIMAL(6,3),
    hyperbola_width_m DECIMAL(6,3),
    hyperbola_curvature DECIMAL(8,4),
    hyperbola_symmetry_score DECIMAL(3,2),

    -- Amplitude-based features
    peak_amplitude DECIMAL(10,6),
    average_amplitude DECIMAL(10,6),
    amplitude_variance DECIMAL(10,6),
    amplitude_skewness DECIMAL(8,4),
    amplitude_kurtosis DECIMAL(8,4),

    -- Frequency domain features
    dominant_frequency_mhz DECIMAL(6,2),
    frequency_bandwidth_mhz DECIMAL(6,2),
    spectral_centroid DECIMAL(8,4),
    spectral_rolloff DECIMAL(8,4),
    spectral_flatness DECIMAL(6,4),

    -- Time-domain statistical features
    zero_crossing_rate DECIMAL(6,4),
    energy DECIMAL(12,6),
    rms_value DECIMAL(10,6),
    crest_factor DECIMAL(6,3),

    -- Texture and pattern features
    local_binary_pattern vector(256),
    gradient_magnitude_histogram vector(64),

    extraction_method VARCHAR(100),
    extraction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Environmental correlation analysis
CREATE TABLE environmental_correlations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Environmental factor correlations
    moisture_impact_factor DECIMAL(6,4),
    soil_type_impact_factor DECIMAL(6,4),
    weather_impact_factor DECIMAL(6,4),
    surface_material_impact_factor DECIMAL(6,4),

    -- Detection performance metrics
    detection_accuracy DECIMAL(5,4),
    false_positive_rate DECIMAL(5,4),
    false_negative_rate DECIMAL(5,4),
    depth_estimation_error_m DECIMAL(6,3),

    -- Analysis metadata
    correlation_method VARCHAR(100),
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confidence_interval DECIMAL(5,4)
);

-- ================================================================
-- ML MODEL PERFORMANCE TRACKING
-- ================================================================

-- ML Models registry
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL, -- classification, detection, depth_estimation, etc.
    model_version VARCHAR(50) NOT NULL,

    -- Model architecture details
    architecture VARCHAR(100), -- ResNet50, YOLOv8, RandomForest, etc.
    framework VARCHAR(50), -- tensorflow, pytorch, sklearn
    input_shape TEXT,
    output_shape TEXT,
    parameters_count BIGINT,

    -- Training details
    training_dataset_size INTEGER,
    validation_dataset_size INTEGER,
    test_dataset_size INTEGER,
    training_duration_hours DECIMAL(8,2),

    -- Model files
    model_file_path TEXT,
    weights_file_path TEXT,
    config_file_path TEXT,

    -- Status and metadata
    status VARCHAR(50) DEFAULT 'training' CHECK (status IN ('training', 'trained', 'deployed', 'deprecated')),
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_at TIMESTAMP WITH TIME ZONE
);

-- Model performance metrics tracking
CREATE TABLE model_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,

    -- Evaluation dataset
    evaluation_dataset VARCHAR(255),
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sample_count INTEGER,

    -- Classification metrics
    accuracy DECIMAL(6,4),
    precision_macro DECIMAL(6,4),
    recall_macro DECIMAL(6,4),
    f1_score_macro DECIMAL(6,4),
    auc_roc DECIMAL(6,4),

    -- Detection metrics (for object detection models)
    map_50 DECIMAL(6,4), -- mAP at IoU=0.5
    map_75 DECIMAL(6,4), -- mAP at IoU=0.75
    map_50_95 DECIMAL(6,4), -- mAP averaged over IoU thresholds

    -- Regression metrics (for depth estimation)
    mse DECIMAL(8,6),
    rmse DECIMAL(8,6),
    mae DECIMAL(8,6),
    r_squared DECIMAL(6,4),

    -- Per-class performance (JSON format)
    class_specific_metrics JSONB,

    -- Confusion matrix (for classification)
    confusion_matrix JSONB,

    -- Additional metrics
    inference_time_ms DECIMAL(8,3),
    memory_usage_mb DECIMAL(8,2),

    notes TEXT
);

-- Model prediction tracking for continuous monitoring
CREATE TABLE model_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    survey_id UUID REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    image_id UUID REFERENCES gpr_image_data(id) ON DELETE CASCADE,

    -- Prediction details
    prediction_class VARCHAR(100),
    confidence_score DECIMAL(5,4),
    prediction_probabilities JSONB, -- All class probabilities

    -- For object detection
    bounding_boxes JSONB,

    -- For utility depth estimation
    predicted_depth_m DECIMAL(6,3),
    depth_confidence DECIMAL(5,4),

    -- Ground truth comparison (when available)
    ground_truth_class VARCHAR(100),
    ground_truth_depth_m DECIMAL(6,3),
    prediction_correct BOOLEAN,
    absolute_error DECIMAL(6,3),

    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT check_reference CHECK (
        (survey_id IS NOT NULL AND image_id IS NULL) OR
        (survey_id IS NULL AND image_id IS NOT NULL)
    )
);

-- ================================================================
-- GROUND TRUTH VALIDATION TABLES
-- ================================================================

-- Ground truth validation campaigns
CREATE TABLE validation_campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    campaign_name VARCHAR(255) NOT NULL,
    validation_method VARCHAR(100) NOT NULL, -- trial_trench, excavation, records_verification

    start_date DATE NOT NULL,
    end_date DATE,
    responsible_engineer VARCHAR(255),
    validation_standard VARCHAR(100), -- PAS128:2022, etc.

    -- Results summary
    total_validations INTEGER DEFAULT 0,
    correct_detections INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual validation records
CREATE TABLE ground_truth_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES validation_campaigns(id) ON DELETE CASCADE,
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    detected_utility_id UUID REFERENCES detected_utilities(id),

    -- Validation location
    validation_point GEOMETRY(POINT, 4326),
    depth_validated_m DECIMAL(6,3),

    -- Ground truth findings
    utility_present BOOLEAN NOT NULL,
    actual_utility_type VARCHAR(100),
    actual_material VARCHAR(100),
    actual_diameter_mm INTEGER,
    actual_depth_m DECIMAL(6,3),

    -- Validation details
    validation_date DATE NOT NULL,
    validation_method VARCHAR(100),
    validator_name VARCHAR(255),
    excavation_depth_m DECIMAL(6,3),

    -- Quality assessment
    validation_confidence VARCHAR(20) CHECK (validation_confidence IN ('low', 'medium', 'high')),
    notes TEXT,
    photo_references TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- COMPLIANCE AND REPORTING TABLES
-- ================================================================

-- PAS 128 compliance tracking
CREATE TABLE pas128_compliance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Quality level assignments
    quality_level_assigned VARCHAR(10) NOT NULL CHECK (quality_level_assigned IN ('QL-A', 'QL-B', 'QL-C', 'QL-D')),
    quality_criteria_met JSONB, -- Detailed criteria checklist

    -- Survey completeness
    survey_extent_documented BOOLEAN DEFAULT FALSE,
    coordinate_system_recorded BOOLEAN DEFAULT FALSE,
    limitations_documented BOOLEAN DEFAULT FALSE,
    confidence_scores_assigned BOOLEAN DEFAULT FALSE,

    -- Compliance validation
    compliant BOOLEAN DEFAULT FALSE,
    compliance_issues TEXT[],
    compliance_check_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checked_by VARCHAR(255),

    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- ================================================================

-- Survey data indexes
CREATE INDEX idx_gpr_surveys_project ON gpr_surveys(project_id);
CREATE INDEX idx_gpr_surveys_site ON gpr_surveys(site_id);
CREATE INDEX idx_gpr_surveys_date ON gpr_surveys(survey_date);
CREATE INDEX idx_gpr_surveys_location ON gpr_surveys(location_id);

-- Utility detection indexes
CREATE INDEX idx_detected_utilities_survey ON detected_utilities(survey_id);
CREATE INDEX idx_detected_utilities_type ON detected_utilities(utility_discipline);
CREATE INDEX idx_detected_utilities_depth ON detected_utilities(depth_m);
CREATE INDEX idx_detected_utilities_confidence ON detected_utilities(confidence_score);

-- Spatial indexes
CREATE INDEX idx_survey_sites_location ON survey_sites USING GIST(location);
CREATE INDEX idx_detected_utilities_coordinates ON detected_utilities USING GIST(coordinates);

-- Image data indexes
CREATE INDEX idx_gpr_image_category ON gpr_image_data(image_category);
CREATE INDEX idx_gpr_image_survey ON gpr_image_data(survey_id);
CREATE INDEX idx_gpr_image_vector ON gpr_image_data USING ivfflat(feature_vector vector_cosine_ops);

-- ML performance indexes
CREATE INDEX idx_model_performance_model ON model_performance_metrics(model_id);
CREATE INDEX idx_model_performance_date ON model_performance_metrics(evaluation_date);
CREATE INDEX idx_model_predictions_model ON model_predictions(model_id);
CREATE INDEX idx_model_predictions_timestamp ON model_predictions(prediction_timestamp);

-- ================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ================================================================

-- Survey summary statistics
CREATE MATERIALIZED VIEW survey_statistics AS
SELECT
    p.id as project_id,
    p.name as project_name,
    COUNT(DISTINCT s.id) as total_surveys,
    COUNT(DISTINCT du.id) as total_utilities_detected,
    AVG(s.confidence_score) as avg_confidence_score,
    COUNT(DISTINCT CASE WHEN gtv.utility_present THEN gtv.id END) as validated_detections,
    COUNT(DISTINCT ss.id) as total_sites
FROM projects p
LEFT JOIN gpr_surveys s ON p.id = s.project_id
LEFT JOIN detected_utilities du ON s.id = du.survey_id
LEFT JOIN ground_truth_validations gtv ON du.id = gtv.detected_utility_id
LEFT JOIN survey_sites ss ON p.id = ss.project_id
GROUP BY p.id, p.name;

-- Create refresh function for materialized views
CREATE OR REPLACE FUNCTION refresh_survey_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW survey_statistics;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- AUDIT TRIGGERS
-- ================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to relevant tables
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gpr_surveys_updated_at BEFORE UPDATE ON gpr_surveys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE projects IS 'Main project organization table for managing utility detection surveys';
COMMENT ON TABLE gpr_surveys IS 'GPR survey data with comprehensive metadata from Twente dataset (25+ fields)';
COMMENT ON TABLE gpr_signal_data IS 'Raw and processed GPR signal characteristics for feature analysis';
COMMENT ON TABLE detected_utilities IS 'Utility detection results with ground truth validation';
COMMENT ON TABLE gpr_image_data IS 'Integration with Mojahid image dataset for ML training';
COMMENT ON TABLE ml_models IS 'ML model registry for tracking performance across different approaches';
COMMENT ON TABLE model_performance_metrics IS 'Comprehensive ML model performance tracking';
COMMENT ON TABLE ground_truth_validations IS 'Ground truth validation results for accuracy assessment';
COMMENT ON TABLE pas128_compliance IS 'PAS 128:2022 compliance tracking and reporting';

-- ================================================================
-- INITIAL DATA SETUP
-- ================================================================

-- Insert sample project for Twente dataset
INSERT INTO projects (name, description, compliance_standards) VALUES
('Twente University GPR Research Dataset', 'Ground-truthed GPR dataset from 13 construction sites in Netherlands', ARRAY['PAS128:2022']);

-- Create views for dataset integration
CREATE VIEW twente_dataset_view AS
SELECT
    s.*,
    COUNT(du.id) as detected_utilities_count,
    AVG(du.confidence_score) as avg_detection_confidence
FROM gpr_surveys s
LEFT JOIN detected_utilities du ON s.id = du.survey_id
WHERE s.location_id LIKE '%.%' -- Twente format: 01.1, 01.2, etc.
GROUP BY s.id;

CREATE VIEW mojahid_dataset_view AS
SELECT
    img.*,
    ann.annotation_class,
    COUNT(ann.id) as annotation_count
FROM gpr_image_data img
LEFT JOIN image_annotations ann ON img.id = ann.image_id
WHERE img.image_category IN ('cavities', 'utilities', 'intact', 'augmented_cavities', 'augmented_utilities', 'augmented_intact')
GROUP BY img.id, ann.annotation_class;