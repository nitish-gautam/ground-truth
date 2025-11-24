-- ================================================================
-- PAS 128:2022 COMPLIANCE INTEGRATION SCHEMA
-- Underground Utility Detection Platform - Quality Level Determination
-- ================================================================

-- PAS 128 Quality Levels definition
CREATE TABLE pas128_quality_levels (
    id SERIAL PRIMARY KEY,
    quality_level_code VARCHAR(10) NOT NULL UNIQUE, -- QL-A, QL-B, QL-C, QL-D
    quality_level_name VARCHAR(100) NOT NULL,

    -- Accuracy requirements
    horizontal_accuracy_mm INTEGER NOT NULL,
    vertical_accuracy_mm INTEGER NOT NULL,
    depth_accuracy_percentage DECIMAL(5,2),

    -- Method requirements
    required_detection_methods TEXT[] NOT NULL,
    minimum_verification_percentage DECIMAL(5,2) DEFAULT 0.0,

    -- Documentation requirements
    survey_method_documented BOOLEAN DEFAULT TRUE,
    limitations_documented BOOLEAN DEFAULT TRUE,
    coordinate_system_specified BOOLEAN DEFAULT TRUE,
    confidence_levels_assigned BOOLEAN DEFAULT TRUE,

    -- Quality assurance requirements
    quality_control_checks TEXT[],
    verification_methods TEXT[],

    description TEXT,
    typical_applications TEXT[]
);

-- Insert PAS 128:2022 quality levels
INSERT INTO pas128_quality_levels (
    quality_level_code, quality_level_name,
    horizontal_accuracy_mm, vertical_accuracy_mm, depth_accuracy_percentage,
    required_detection_methods, minimum_verification_percentage,
    quality_control_checks, typical_applications
) VALUES
(
    'QL-A', 'Quality Level A - Excavation Required',
    40, 75, 10.0,
    ARRAY['trial_excavation', 'vacuum_excavation', 'hand_excavation'],
    100.0,
    ARRAY['direct_visual_verification', 'measurement_verification', 'photographic_evidence'],
    ARRAY['sensitive_excavation', 'precision_works', 'critical_infrastructure']
),
(
    'QL-B', 'Quality Level B - Survey Grade',
    200, 300, 15.0,
    ARRAY['gpr_survey', 'electromagnetic_detection', 'utility_records'],
    20.0,
    ARRAY['cross_verification', 'multiple_methods', 'spot_verification'],
    ARRAY['design_applications', 'planning_permission', 'construction_planning']
),
(
    'QL-C', 'Quality Level C - General Information',
    500, 500, 25.0,
    ARRAY['desktop_study', 'site_reconnaissance', 'utility_records'],
    10.0,
    ARRAY['records_verification', 'visual_inspection', 'basic_survey'],
    ARRAY['feasibility_studies', 'route_planning', 'preliminary_design']
),
(
    'QL-D', 'Quality Level D - Schematic',
    1000, 1000, 50.0,
    ARRAY['desktop_study', 'existing_records'],
    0.0,
    ARRAY['records_compilation', 'data_consistency_check'],
    ARRAY['initial_assessment', 'route_selection', 'concept_design']
);

-- Detection methods registry aligned with PAS 128
CREATE TABLE pas128_detection_methods (
    id SERIAL PRIMARY KEY,
    method_code VARCHAR(50) NOT NULL UNIQUE,
    method_name VARCHAR(200) NOT NULL,
    method_category VARCHAR(100) NOT NULL, -- electromagnetic, gpr, excavation, etc.

    -- PAS 128 compliance
    pas128_recognized BOOLEAN DEFAULT TRUE,
    applicable_quality_levels VARCHAR(10)[],

    -- Method characteristics
    typical_accuracy_horizontal_mm INTEGER,
    typical_accuracy_vertical_mm INTEGER,
    detection_depth_range_m NUMRANGE,
    suitable_utility_types TEXT[],

    -- Environmental limitations
    ground_condition_limitations TEXT[],
    weather_limitations TEXT[],
    surface_limitations TEXT[],

    -- Method requirements
    operator_certification_required BOOLEAN DEFAULT FALSE,
    equipment_calibration_required BOOLEAN DEFAULT TRUE,
    minimum_training_hours INTEGER DEFAULT 0,

    -- Documentation requirements
    method_statement_required BOOLEAN DEFAULT TRUE,
    calibration_records_required BOOLEAN DEFAULT TRUE,
    operating_procedures TEXT[],

    method_description TEXT,
    reference_standards TEXT[]
);

-- Insert PAS 128 recognized detection methods
INSERT INTO pas128_detection_methods (
    method_code, method_name, method_category,
    applicable_quality_levels, typical_accuracy_horizontal_mm, typical_accuracy_vertical_mm,
    detection_depth_range_m, suitable_utility_types, operator_certification_required
) VALUES
-- Excavation methods (QL-A)
('TRIAL_EXC', 'Trial Excavation', 'excavation', ARRAY['QL-A'], 10, 25, '[0.1,3.0)',
 ARRAY['all_utility_types'], true),
('VAC_EXC', 'Vacuum Excavation', 'excavation', ARRAY['QL-A', 'QL-B'], 15, 50, '[0.1,2.5)',
 ARRAY['all_utility_types'], true),
('HAND_EXC', 'Hand Excavation', 'excavation', ARRAY['QL-A', 'QL-B'], 25, 75, '[0.1,2.0)',
 ARRAY['shallow_utilities'], false),

-- GPR methods (QL-B, QL-C)
('GPR_500', 'Ground Penetrating Radar 500MHz', 'gpr', ARRAY['QL-B', 'QL-C'], 100, 200, '[0.1,3.0)',
 ARRAY['non_metallic', 'metallic', 'voids'], true),
('GPR_250', 'Ground Penetrating Radar 250MHz', 'gpr', ARRAY['QL-B', 'QL-C'], 150, 300, '[0.5,5.0)',
 ARRAY['deep_utilities', 'large_diameter'], true),

-- Electromagnetic methods (QL-B, QL-C)
('EM_CABLE', 'Cable Detection (Electromagnetic)', 'electromagnetic', ARRAY['QL-B', 'QL-C'], 150, 250, '[0.1,3.0)',
 ARRAY['metallic_utilities', 'cables'], true),
('EM_PIPE', 'Pipe Detection (Electromagnetic)', 'electromagnetic', ARRAY['QL-B', 'QL-C'], 200, 400, '[0.1,4.0)',
 ARRAY['metallic_pipes'], true),

-- Records and desktop methods (QL-C, QL-D)
('UTIL_REC', 'Utility Records Search', 'desktop', ARRAY['QL-C', 'QL-D'], 500, 1000, '[0,10)',
 ARRAY['all_utility_types'], false),
('SITE_REC', 'Site Reconnaissance', 'visual', ARRAY['QL-C', 'QL-D'], 1000, 2000, '[0,5)',
 ARRAY['surface_features'], false);

-- ================================================================
-- ENHANCED PAS 128 COMPLIANCE TRACKING
-- ================================================================

CREATE TABLE pas128_compliance_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Assessment identification
    assessment_reference VARCHAR(100) UNIQUE,
    assessment_date DATE DEFAULT CURRENT_DATE,
    assessor_name VARCHAR(255) NOT NULL,
    assessor_certification VARCHAR(255),

    -- Target quality level
    target_quality_level_id INTEGER NOT NULL REFERENCES pas128_quality_levels(id),
    achieved_quality_level_id INTEGER REFERENCES pas128_quality_levels(id),

    -- Method compliance
    detection_methods_used INTEGER[] NOT NULL, -- References to pas128_detection_methods
    primary_method_id INTEGER NOT NULL REFERENCES pas128_detection_methods(id),
    verification_methods_used INTEGER[],

    -- Accuracy assessment
    horizontal_accuracy_achieved_mm INTEGER,
    vertical_accuracy_achieved_mm INTEGER,
    depth_accuracy_achieved_percentage DECIMAL(5,2),

    -- Statistical validation
    sample_size_total INTEGER,
    sample_size_verified INTEGER,
    verification_percentage DECIMAL(5,2),
    accuracy_confidence_level DECIMAL(5,2) DEFAULT 95.0,

    -- Detailed compliance checks
    survey_method_documented BOOLEAN DEFAULT FALSE,
    coordinate_system_specified BOOLEAN DEFAULT FALSE,
    limitations_documented BOOLEAN DEFAULT FALSE,
    confidence_levels_assigned BOOLEAN DEFAULT FALSE,
    equipment_calibrated BOOLEAN DEFAULT FALSE,
    operator_certified BOOLEAN DEFAULT FALSE,

    -- Quality control results
    qc_checks_performed TEXT[],
    qc_results JSONB, -- Detailed QC check results
    non_conformances TEXT[],
    corrective_actions TEXT[],

    -- Verification results
    verification_sample_locations GEOMETRY(MULTIPOINT, 4326),
    verification_methods_results JSONB,
    verification_accuracy_statistics JSONB,

    -- Environmental conditions compliance
    ground_conditions_suitable BOOLEAN DEFAULT TRUE,
    weather_conditions_suitable BOOLEAN DEFAULT TRUE,
    access_conditions_adequate BOOLEAN DEFAULT TRUE,

    -- Documentation compliance
    method_statements_provided TEXT[],
    calibration_certificates TEXT[],
    survey_reports TEXT[],
    as_built_drawings TEXT[],
    photographic_evidence TEXT[],

    -- Overall compliance determination
    compliant BOOLEAN DEFAULT FALSE,
    compliance_score DECIMAL(5,2), -- 0-100 scale
    compliance_grade VARCHAR(20), -- Excellent, Good, Adequate, Poor, Non-compliant

    -- Issues and recommendations
    compliance_issues TEXT[],
    recommendations TEXT[],
    follow_up_actions_required TEXT[],

    -- Review and approval
    reviewed_by VARCHAR(255),
    review_date DATE,
    approved_by VARCHAR(255),
    approval_date DATE,
    approval_status VARCHAR(50) DEFAULT 'pending' CHECK (approval_status IN ('pending', 'approved', 'conditional', 'rejected')),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- QUALITY LEVEL DETERMINATION LOGIC
-- ================================================================

CREATE TABLE quality_level_determination_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Rule identification
    rule_name VARCHAR(255) NOT NULL,
    rule_description TEXT,
    rule_category VARCHAR(100), -- accuracy_based, method_based, application_based

    -- Conditions
    application_context VARCHAR(100), -- construction, design, feasibility, etc.
    required_accuracy_horizontal_mm INTEGER,
    required_accuracy_vertical_mm INTEGER,
    required_depth_accuracy_percentage DECIMAL(5,2),

    -- Project risk factors
    project_risk_level VARCHAR(50), -- low, medium, high, critical
    consequences_of_error VARCHAR(100), -- minor, moderate, major, catastrophic
    utility_density VARCHAR(50), -- low, medium, high, very_high
    utility_criticality VARCHAR(50), -- standard, important, critical

    -- Environmental factors
    suitable_ground_conditions TEXT[],
    suitable_weather_conditions TEXT[],
    depth_range_m NUMRANGE,

    -- Recommended quality level
    recommended_quality_level_id INTEGER NOT NULL REFERENCES pas128_quality_levels(id),
    alternative_quality_levels INTEGER[], -- Alternative acceptable levels

    -- Supporting rationale
    technical_justification TEXT,
    cost_benefit_analysis TEXT,
    risk_assessment TEXT,

    -- Rule status
    rule_status VARCHAR(50) DEFAULT 'active' CHECK (rule_status IN ('active', 'inactive', 'under_review')),
    effective_date DATE DEFAULT CURRENT_DATE,
    review_date DATE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert standard determination rules
INSERT INTO quality_level_determination_rules (
    rule_name, rule_description, rule_category,
    required_accuracy_horizontal_mm, required_accuracy_vertical_mm,
    project_risk_level, consequences_of_error,
    recommended_quality_level_id, technical_justification
) VALUES
(
    'Critical Infrastructure QL-A',
    'High-risk excavation near critical infrastructure requiring maximum accuracy',
    'risk_based',
    40, 75, 'critical', 'catastrophic',
    1, -- QL-A
    'Direct excavation verification required due to potential for service disruption and safety risks'
),
(
    'Standard Construction QL-B',
    'Standard construction projects requiring survey-grade accuracy',
    'application_based',
    200, 300, 'medium', 'moderate',
    2, -- QL-B
    'Balanced approach providing adequate accuracy for most construction applications'
),
(
    'Feasibility Study QL-C',
    'Preliminary studies and route planning applications',
    'application_based',
    500, 500, 'low', 'minor',
    3, -- QL-C
    'Sufficient accuracy for planning and preliminary design phases'
),
(
    'Desktop Study QL-D',
    'Initial assessment and concept development',
    'application_based',
    1000, 1000, 'low', 'minor',
    4, -- QL-D
    'Records-based information adequate for initial planning'
);

-- ================================================================
-- AUTOMATED QUALITY LEVEL DETERMINATION
-- ================================================================

-- Function to determine appropriate quality level
CREATE OR REPLACE FUNCTION determine_quality_level(
    survey_uuid UUID,
    application_context VARCHAR DEFAULT 'construction',
    project_risk VARCHAR DEFAULT 'medium',
    required_horizontal_accuracy_mm INTEGER DEFAULT NULL,
    required_vertical_accuracy_mm INTEGER DEFAULT NULL
) RETURNS TABLE (
    recommended_quality_level VARCHAR,
    quality_level_id INTEGER,
    confidence_score DECIMAL,
    justification TEXT,
    alternative_levels VARCHAR[]
) AS $$
DECLARE
    survey_record RECORD;
    environmental_record RECORD;
    rule_record RECORD;
    best_match_score DECIMAL := 0;
    best_rule_id UUID;
BEGIN
    -- Get survey and environmental data
    SELECT s.*, em.amount_of_utilities, em.utility_crossing, em.ground_condition_id
    INTO survey_record
    FROM gpr_surveys s
    LEFT JOIN environmental_metadata em ON s.id = em.survey_id
    WHERE s.id = survey_uuid;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Survey not found: %', survey_uuid;
    END IF;

    -- Find best matching rule
    FOR rule_record IN
        SELECT qr.*, ql.quality_level_code, ql.quality_level_name
        FROM quality_level_determination_rules qr
        JOIN pas128_quality_levels ql ON qr.recommended_quality_level_id = ql.id
        WHERE qr.rule_status = 'active'
        ORDER BY qr.required_accuracy_horizontal_mm ASC
    LOOP
        DECLARE
            match_score DECIMAL := 0;
        BEGIN
            -- Score based on accuracy requirements
            IF required_horizontal_accuracy_mm IS NOT NULL THEN
                IF rule_record.required_accuracy_horizontal_mm <= required_horizontal_accuracy_mm THEN
                    match_score := match_score + 30;
                END IF;
            END IF;

            -- Score based on application context
            IF rule_record.application_context = application_context THEN
                match_score := match_score + 25;
            END IF;

            -- Score based on project risk
            IF rule_record.project_risk_level = project_risk THEN
                match_score := match_score + 20;
            END IF;

            -- Score based on utility density
            IF survey_record.amount_of_utilities IS NOT NULL THEN
                CASE
                    WHEN survey_record.amount_of_utilities >= 5 AND rule_record.utility_density = 'very_high' THEN
                        match_score := match_score + 15;
                    WHEN survey_record.amount_of_utilities >= 3 AND rule_record.utility_density = 'high' THEN
                        match_score := match_score + 15;
                    WHEN survey_record.amount_of_utilities >= 1 AND rule_record.utility_density IN ('medium', 'high') THEN
                        match_score := match_score + 10;
                    WHEN survey_record.amount_of_utilities = 0 AND rule_record.utility_density = 'low' THEN
                        match_score := match_score + 15;
                END CASE;
            END IF;

            -- Update best match
            IF match_score > best_match_score THEN
                best_match_score := match_score;
                best_rule_id := rule_record.id;
            END IF;
        END;
    END LOOP;

    -- Return results for best matching rule
    RETURN QUERY
    SELECT
        ql.quality_level_code,
        ql.id,
        ROUND(best_match_score / 90.0, 2) as confidence_score, -- Normalized to 0-1
        qr.technical_justification,
        ARRAY(
            SELECT ql_alt.quality_level_code
            FROM pas128_quality_levels ql_alt
            WHERE ql_alt.id = ANY(qr.alternative_quality_levels)
        )
    FROM quality_level_determination_rules qr
    JOIN pas128_quality_levels ql ON qr.recommended_quality_level_id = ql.id
    WHERE qr.id = best_rule_id;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- COMPLIANCE VALIDATION FUNCTIONS
-- ================================================================

-- Function to validate survey compliance with quality level
CREATE OR REPLACE FUNCTION validate_pas128_compliance(
    survey_uuid UUID,
    target_quality_level VARCHAR
) RETURNS TABLE (
    is_compliant BOOLEAN,
    compliance_score DECIMAL,
    horizontal_accuracy_met BOOLEAN,
    vertical_accuracy_met BOOLEAN,
    method_requirements_met BOOLEAN,
    documentation_complete BOOLEAN,
    issues TEXT[],
    recommendations TEXT[]
) AS $$
DECLARE
    survey_record RECORD;
    quality_level_record RECORD;
    validation_stats RECORD;
    compliance_issues TEXT[] := ARRAY[]::TEXT[];
    compliance_recommendations TEXT[] := ARRAY[]::TEXT[];
    total_score DECIMAL := 0;
    max_score DECIMAL := 100;
    h_accuracy_ok BOOLEAN := FALSE;
    v_accuracy_ok BOOLEAN := FALSE;
    methods_ok BOOLEAN := FALSE;
    docs_ok BOOLEAN := FALSE;
BEGIN
    -- Get quality level requirements
    SELECT * INTO quality_level_record
    FROM pas128_quality_levels
    WHERE quality_level_code = target_quality_level;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Quality level not found: %', target_quality_level;
    END IF;

    -- Get survey data
    SELECT s.* INTO survey_record
    FROM gpr_surveys s
    WHERE s.id = survey_uuid;

    -- Get validation statistics
    SELECT
        AVG(ABS(horizontal_position_error_mm)) as avg_horizontal_error,
        AVG(ABS(depth_error_mm)) as avg_vertical_error,
        COUNT(*) as total_validations,
        COUNT(CASE WHEN utility_present THEN 1 END) as confirmed_utilities
    INTO validation_stats
    FROM ground_truth_validations
    WHERE survey_id = survey_uuid;

    -- Check horizontal accuracy
    IF validation_stats.avg_horizontal_error <= quality_level_record.horizontal_accuracy_mm THEN
        h_accuracy_ok := TRUE;
        total_score := total_score + 30;
    ELSE
        compliance_issues := compliance_issues ||
            format('Horizontal accuracy requirement not met: %s mm achieved vs %s mm required',
                   validation_stats.avg_horizontal_error, quality_level_record.horizontal_accuracy_mm);
        compliance_recommendations := compliance_recommendations ||
            'Consider using more precise detection methods or additional verification';
    END IF;

    -- Check vertical accuracy
    IF validation_stats.avg_vertical_error <= quality_level_record.vertical_accuracy_mm THEN
        v_accuracy_ok := TRUE;
        total_score := total_score + 30;
    ELSE
        compliance_issues := compliance_issues ||
            format('Vertical accuracy requirement not met: %s mm achieved vs %s mm required',
                   validation_stats.avg_vertical_error, quality_level_record.vertical_accuracy_mm);
        compliance_recommendations := compliance_recommendations ||
            'Improve depth estimation methods or increase verification sampling';
    END IF;

    -- Check method requirements (simplified check)
    IF survey_record.equipment_model IS NOT NULL AND survey_record.surveyor_name IS NOT NULL THEN
        methods_ok := TRUE;
        total_score := total_score + 20;
    ELSE
        compliance_issues := compliance_issues || 'Method documentation incomplete';
        compliance_recommendations := compliance_recommendations ||
            'Ensure all detection methods and equipment are properly documented';
    END IF;

    -- Check documentation requirements
    IF survey_record.survey_map_path IS NOT NULL THEN
        docs_ok := TRUE;
        total_score := total_score + 20;
    ELSE
        compliance_issues := compliance_issues || 'Required documentation missing';
        compliance_recommendations := compliance_recommendations ||
            'Complete all required documentation and drawings';
    END IF;

    RETURN QUERY SELECT
        (total_score >= 80), -- 80% threshold for compliance
        ROUND(total_score, 2),
        h_accuracy_ok,
        v_accuracy_ok,
        methods_ok,
        docs_ok,
        compliance_issues,
        compliance_recommendations;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- PERFORMANCE INDEXES
-- ================================================================

-- PAS 128 specific indexes
CREATE INDEX idx_pas128_assessments_survey ON pas128_compliance_assessments(survey_id);
CREATE INDEX idx_pas128_assessments_target_ql ON pas128_compliance_assessments(target_quality_level_id);
CREATE INDEX idx_pas128_assessments_achieved_ql ON pas128_compliance_assessments(achieved_quality_level_id);
CREATE INDEX idx_pas128_assessments_compliant ON pas128_compliance_assessments(compliant);
CREATE INDEX idx_pas128_assessments_date ON pas128_compliance_assessments(assessment_date);

-- Detection methods indexes
CREATE INDEX idx_detection_methods_category ON pas128_detection_methods(method_category);
CREATE INDEX idx_detection_methods_quality_levels ON pas128_detection_methods USING GIN(applicable_quality_levels);

-- Quality determination rules indexes
CREATE INDEX idx_ql_rules_accuracy ON quality_level_determination_rules(required_accuracy_horizontal_mm, required_accuracy_vertical_mm);
CREATE INDEX idx_ql_rules_context ON quality_level_determination_rules(application_context, project_risk_level);
CREATE INDEX idx_ql_rules_recommended ON quality_level_determination_rules(recommended_quality_level_id);

-- ================================================================
-- MATERIALIZED VIEW FOR COMPLIANCE REPORTING
-- ================================================================

CREATE MATERIALIZED VIEW pas128_compliance_summary AS
SELECT
    p.name as project_name,
    ql.quality_level_code,
    ql.quality_level_name,
    COUNT(pca.id) as total_assessments,
    COUNT(CASE WHEN pca.compliant = true THEN 1 END) as compliant_assessments,
    ROUND(AVG(pca.compliance_score), 2) as average_compliance_score,
    ROUND(AVG(pca.horizontal_accuracy_achieved_mm), 2) as avg_horizontal_accuracy,
    ROUND(AVG(pca.vertical_accuracy_achieved_mm), 2) as avg_vertical_accuracy,
    COUNT(CASE WHEN pca.approval_status = 'approved' THEN 1 END) as approved_assessments
FROM projects p
JOIN gpr_surveys s ON p.id = s.project_id
JOIN pas128_compliance_assessments pca ON s.id = pca.survey_id
JOIN pas128_quality_levels ql ON pca.target_quality_level_id = ql.id
GROUP BY p.name, ql.quality_level_code, ql.quality_level_name
ORDER BY p.name, ql.quality_level_code;

-- Function to refresh PAS 128 views
CREATE OR REPLACE FUNCTION refresh_pas128_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW pas128_compliance_summary;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE pas128_quality_levels IS 'PAS 128:2022 quality level definitions with accuracy and method requirements';
COMMENT ON TABLE pas128_detection_methods IS 'PAS 128 recognized detection methods with compliance specifications';
COMMENT ON TABLE pas128_compliance_assessments IS 'Comprehensive PAS 128 compliance assessments and validation';
COMMENT ON TABLE quality_level_determination_rules IS 'Business rules for automated quality level determination';

COMMENT ON FUNCTION determine_quality_level IS 'Automated quality level determination based on project requirements and risk factors';
COMMENT ON FUNCTION validate_pas128_compliance IS 'Comprehensive PAS 128 compliance validation with detailed feedback';

COMMENT ON MATERIALIZED VIEW pas128_compliance_summary IS 'Summary of PAS 128 compliance performance across projects and quality levels';