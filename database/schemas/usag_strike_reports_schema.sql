-- ================================================================
-- USAG STRIKE REPORTS SCHEMA
-- Underground Utility Detection Platform - Historical Incident Data Analysis
-- ================================================================

-- Strike incident categories and classification
CREATE TABLE strike_incident_categories (
    id SERIAL PRIMARY KEY,
    category_code VARCHAR(20) UNIQUE NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    severity_level VARCHAR(20) CHECK (severity_level IN ('minor', 'moderate', 'major', 'critical', 'catastrophic')),
    typical_causes TEXT[],
    prevention_measures TEXT[]
);

-- Insert strike incident categories
INSERT INTO strike_incident_categories (category_code, category_name, description, severity_level, typical_causes) VALUES
('ELEC_STRIKE', 'Electrical Utility Strike', 'Damage to electrical cables or equipment', 'critical',
 ARRAY['inadequate_location', 'excavation_error', 'equipment_failure', 'procedural_violation']),
('GAS_STRIKE', 'Gas Pipeline Strike', 'Damage to gas distribution or transmission lines', 'catastrophic',
 ARRAY['inadequate_location', 'excavation_error', 'equipment_failure', 'emergency_response']),
('WATER_STRIKE', 'Water Main Strike', 'Damage to water distribution infrastructure', 'moderate',
 ARRAY['inadequate_location', 'excavation_error', 'aging_infrastructure', 'corrosion']),
('SEWER_STRIKE', 'Sewer Line Strike', 'Damage to wastewater collection systems', 'moderate',
 ARRAY['inadequate_location', 'excavation_error', 'blockage_clearance', 'maintenance_work']),
('TELECOM_STRIKE', 'Telecommunications Strike', 'Damage to fiber optic or copper communication lines', 'major',
 ARRAY['inadequate_location', 'excavation_error', 'network_congestion', 'upgrade_work']),
('MULTI_UTIL', 'Multiple Utility Strike', 'Damage affecting multiple utility types simultaneously', 'critical',
 ARRAY['complex_utility_crossing', 'inadequate_survey', 'bundled_utilities', 'cascade_failure']);

-- Utility types involved in strikes
CREATE TABLE utility_types (
    id SERIAL PRIMARY KEY,
    utility_code VARCHAR(20) UNIQUE NOT NULL,
    utility_name VARCHAR(100) NOT NULL,
    utility_category VARCHAR(50), -- power, communications, water, gas, etc.
    criticality_level VARCHAR(20) CHECK (criticality_level IN ('low', 'medium', 'high', 'critical')),
    typical_materials TEXT[],
    typical_depths_m NUMRANGE,
    regulatory_requirements TEXT[]
);

-- Insert utility types
INSERT INTO utility_types (utility_code, utility_name, utility_category, criticality_level, typical_materials, typical_depths_m) VALUES
('ELEC_LV', 'Low Voltage Electricity', 'power', 'high', ARRAY['copper', 'aluminum', 'XLPE_cable'], '[0.45,1.2)'),
('ELEC_HV', 'High Voltage Electricity', 'power', 'critical', ARRAY['copper', 'aluminum', 'oil_filled'], '[0.9,2.5)'),
('GAS_LP', 'Low Pressure Gas', 'gas', 'critical', ARRAY['polyethylene', 'steel', 'cast_iron'], '[0.6,1.8)'),
('GAS_MP', 'Medium Pressure Gas', 'gas', 'critical', ARRAY['steel', 'polyethylene'], '[0.9,2.4)'),
('GAS_HP', 'High Pressure Gas', 'gas', 'critical', ARRAY['steel', 'welded_steel'], '[1.2,4.0)'),
('WATER_MAIN', 'Water Distribution Main', 'water', 'medium', ARRAY['ductile_iron', 'PVC', 'concrete'], '[0.9,3.0)'),
('WATER_SERV', 'Water Service Connection', 'water', 'low', ARRAY['copper', 'plastic', 'lead'], '[0.6,1.5)'),
('SEWER_MAIN', 'Sewer Main', 'wastewater', 'medium', ARRAY['vitrified_clay', 'concrete', 'PVC'], '[1.2,4.0)'),
('SEWER_SERV', 'Sewer Service', 'wastewater', 'low', ARRAY['PVC', 'clay', 'cast_iron'], '[0.9,2.4)'),
('FIBER_OPT', 'Fiber Optic Cable', 'communications', 'high', ARRAY['fiber_optic', 'armored_cable'], '[0.6,1.8)'),
('TELECOM_CU', 'Copper Telecommunications', 'communications', 'medium', ARRAY['copper', 'lead_sheath'], '[0.6,1.5)'),
('CABLE_TV', 'Cable Television', 'communications', 'low', ARRAY['coaxial', 'fiber_optic'], '[0.45,1.2)');

-- ================================================================
-- USAG STRIKE INCIDENT REPORTS
-- ================================================================

CREATE TABLE usag_strike_incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Incident identification
    incident_reference VARCHAR(100) UNIQUE NOT NULL,
    usag_report_number VARCHAR(50),
    external_reference_numbers TEXT[], -- Other agency reference numbers

    -- Incident classification
    incident_category_id INTEGER NOT NULL REFERENCES strike_incident_categories(id),
    utility_type_id INTEGER NOT NULL REFERENCES utility_types(id),
    additional_utilities_affected INTEGER[] REFERENCES utility_types(id), -- For multi-utility incidents

    -- Temporal information
    incident_date DATE NOT NULL,
    incident_time TIME,
    report_date DATE NOT NULL,
    investigation_completion_date DATE,

    -- Location information
    incident_location GEOMETRY(POINT, 4326) NOT NULL,
    osgb36_coordinates GEOMETRY(POINT, 27700),
    location_description TEXT NOT NULL,
    address TEXT,
    postcode VARCHAR(10),
    local_authority VARCHAR(100),
    parliamentary_constituency VARCHAR(100),

    -- Site characteristics
    site_type VARCHAR(100), -- residential, commercial, industrial, highway, etc.
    land_use_context VARCHAR(100),
    excavation_type VARCHAR(100), -- planned_excavation, emergency_repair, maintenance, etc.
    excavation_method VARCHAR(100), -- hand_digging, mechanical_excavator, drilling, etc.

    -- Incident details
    incident_description TEXT NOT NULL,
    immediate_cause VARCHAR(200),
    root_cause_analysis TEXT,
    contributing_factors TEXT[],

    -- Pre-incident planning and surveys
    utility_survey_conducted BOOLEAN DEFAULT FALSE,
    survey_method VARCHAR(100), -- CAT_scanner, GPR, records_search, etc.
    survey_quality_level VARCHAR(10), -- PAS128 QL-A, QL-B, etc.
    survey_date DATE,
    survey_contractor VARCHAR(255),
    utility_records_consulted BOOLEAN DEFAULT FALSE,
    permit_to_dig_obtained BOOLEAN DEFAULT FALSE,

    -- Damage assessment
    utility_damaged BOOLEAN DEFAULT TRUE,
    damage_extent VARCHAR(100), -- minor, moderate, severe, complete_severance
    damage_description TEXT,
    repair_required BOOLEAN DEFAULT TRUE,
    replacement_required BOOLEAN DEFAULT FALSE,

    -- Service disruption
    service_interruption BOOLEAN DEFAULT FALSE,
    customers_affected INTEGER DEFAULT 0,
    duration_of_outage_hours DECIMAL(6,2),
    priority_customers_affected BOOLEAN DEFAULT FALSE, -- hospitals, schools, etc.

    -- Safety and emergency response
    safety_incident BOOLEAN DEFAULT FALSE,
    injuries_reported INTEGER DEFAULT 0,
    fatalities_reported INTEGER DEFAULT 0,
    emergency_services_called BOOLEAN DEFAULT FALSE,
    evacuation_required BOOLEAN DEFAULT FALSE,
    area_evacuated_radius_m INTEGER,

    -- Environmental impact
    environmental_damage BOOLEAN DEFAULT FALSE,
    environmental_description TEXT,
    contamination_risk VARCHAR(50), -- none, low, medium, high
    environmental_cleanup_required BOOLEAN DEFAULT FALSE,

    -- Regulatory and legal
    hse_notifiable BOOLEAN DEFAULT FALSE,
    hse_notification_number VARCHAR(50),
    police_involvement BOOLEAN DEFAULT FALSE,
    enforcement_action BOOLEAN DEFAULT FALSE,
    legal_proceedings BOOLEAN DEFAULT FALSE,

    -- Financial impact
    estimated_repair_cost DECIMAL(12,2),
    estimated_disruption_cost DECIMAL(12,2),
    insurance_claim_value DECIMAL(12,2),
    compensation_paid DECIMAL(12,2),

    -- Responsible parties
    excavation_contractor VARCHAR(255),
    client_organization VARCHAR(255),
    utility_owner VARCHAR(255),
    project_manager VARCHAR(255),

    -- Investigation findings
    investigation_status VARCHAR(50) DEFAULT 'pending' CHECK (investigation_status IN ('pending', 'ongoing', 'completed', 'closed')),
    fault_determination VARCHAR(100),
    liability_assignment VARCHAR(100),
    lessons_learned TEXT,

    -- Preventive measures
    recommended_actions TEXT[],
    industry_guidance_updates TEXT[],
    training_requirements TEXT[],
    procedural_changes TEXT[],

    -- Follow-up and monitoring
    follow_up_required BOOLEAN DEFAULT FALSE,
    follow_up_actions TEXT[],
    follow_up_completion_date DATE,
    similar_incidents_flagged BOOLEAN DEFAULT FALSE,

    -- Data sources and quality
    data_source VARCHAR(100), -- usag_report, company_report, investigation, etc.
    data_quality_score DECIMAL(3,2) CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    data_completeness_score DECIMAL(3,2) CHECK (data_completeness_score >= 0 AND data_completeness_score <= 1),
    verification_status VARCHAR(50) DEFAULT 'unverified',

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- STRIKE INCIDENT ANALYSIS AND PATTERNS
-- ================================================================

CREATE TABLE strike_pattern_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Analysis identification
    analysis_name VARCHAR(255) NOT NULL,
    analysis_period_start DATE NOT NULL,
    analysis_period_end DATE NOT NULL,
    analysis_description TEXT,

    -- Spatial analysis
    analysis_region GEOMETRY(POLYGON, 4326),
    hotspot_locations GEOMETRY(MULTIPOINT, 4326),
    spatial_clustering_detected BOOLEAN DEFAULT FALSE,
    cluster_analysis_results JSONB,

    -- Temporal analysis
    seasonal_patterns JSONB, -- Monthly/quarterly trends
    day_of_week_patterns JSONB,
    time_of_day_patterns JSONB,
    trend_analysis JSONB, -- Increasing/decreasing trends

    -- Incident characteristics analysis
    most_common_utilities TEXT[],
    most_common_causes TEXT[],
    severity_distribution JSONB,
    cost_impact_analysis JSONB,

    -- Correlation analysis
    utility_type_correlations JSONB,
    site_type_correlations JSONB,
    excavation_method_correlations JSONB,
    weather_correlations JSONB,

    -- Risk factors identification
    high_risk_locations GEOMETRY(MULTIPOINT, 4326),
    high_risk_activities TEXT[],
    high_risk_contractors TEXT[],
    risk_scoring_model JSONB,

    -- Prevention effectiveness
    survey_effectiveness_analysis JSONB,
    training_impact_analysis JSONB,
    regulatory_impact_analysis JSONB,

    -- Predictions and forecasting
    predicted_hotspots GEOMETRY(MULTIPOINT, 4326),
    risk_forecast_6months JSONB,
    prevention_recommendations TEXT[],

    -- Analysis metadata
    analysis_method VARCHAR(100),
    statistical_methods_used TEXT[],
    sample_size INTEGER,
    confidence_level DECIMAL(3,2) DEFAULT 0.95,
    analyst_name VARCHAR(255),
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- STRIKE PREVENTION MEASURES TRACKING
-- ================================================================

CREATE TABLE strike_prevention_measures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Measure identification
    measure_name VARCHAR(255) NOT NULL,
    measure_category VARCHAR(100), -- technology, training, regulation, procedure
    measure_description TEXT,

    -- Implementation details
    implementation_date DATE,
    implementation_scope VARCHAR(100), -- national, regional, company, project
    target_audience TEXT[], -- contractors, engineers, operators, etc.

    -- Effectiveness tracking
    baseline_incident_rate DECIMAL(8,4), -- Incidents per unit (time/projects/etc.)
    post_implementation_rate DECIMAL(8,4),
    effectiveness_percentage DECIMAL(5,2),
    statistical_significance DECIMAL(6,4),

    -- Cost-benefit analysis
    implementation_cost DECIMAL(12,2),
    annual_operating_cost DECIMAL(12,2),
    estimated_savings_annual DECIMAL(12,2),
    payback_period_months INTEGER,

    -- Adoption and compliance
    adoption_rate DECIMAL(5,2), -- Percentage of target audience adopting
    compliance_rate DECIMAL(5,2), -- Percentage of correct implementation
    barriers_to_adoption TEXT[],
    success_factors TEXT[],

    -- Performance monitoring
    key_performance_indicators JSONB,
    monitoring_frequency VARCHAR(50),
    last_review_date DATE,
    next_review_date DATE,

    -- Measure status
    measure_status VARCHAR(50) DEFAULT 'active' CHECK (measure_status IN ('proposed', 'pilot', 'active', 'superseded', 'discontinued')),
    discontinuation_reason TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- PERFORMANCE INDEXES
-- ================================================================

-- Strike incidents indexes
CREATE INDEX idx_strike_incidents_date ON usag_strike_incidents(incident_date);
CREATE INDEX idx_strike_incidents_category ON usag_strike_incidents(incident_category_id);
CREATE INDEX idx_strike_incidents_utility_type ON usag_strike_incidents(utility_type_id);
CREATE INDEX idx_strike_incidents_contractor ON usag_strike_incidents(excavation_contractor);
CREATE INDEX idx_strike_incidents_cost ON usag_strike_incidents(estimated_repair_cost);
CREATE INDEX idx_strike_incidents_customers ON usag_strike_incidents(customers_affected);
CREATE INDEX idx_strike_incidents_survey ON usag_strike_incidents(utility_survey_conducted, survey_quality_level);

-- Spatial indexes
CREATE INDEX idx_strike_incidents_location ON usag_strike_incidents USING GIST(incident_location);
CREATE INDEX idx_strike_incidents_osgb36 ON usag_strike_incidents USING GIST(osgb36_coordinates);
CREATE INDEX idx_pattern_analysis_region ON strike_pattern_analysis USING GIST(analysis_region);
CREATE INDEX idx_pattern_analysis_hotspots ON strike_pattern_analysis USING GIST(hotspot_locations);

-- Analysis indexes
CREATE INDEX idx_pattern_analysis_period ON strike_pattern_analysis(analysis_period_start, analysis_period_end);
CREATE INDEX idx_prevention_measures_category ON strike_prevention_measures(measure_category);
CREATE INDEX idx_prevention_measures_status ON strike_prevention_measures(measure_status);

-- ================================================================
-- MATERIALIZED VIEWS FOR INCIDENT ANALYSIS
-- ================================================================

-- Annual strike statistics
CREATE MATERIALIZED VIEW annual_strike_statistics AS
SELECT
    EXTRACT(YEAR FROM incident_date) as incident_year,
    COUNT(*) as total_incidents,
    COUNT(CASE WHEN safety_incident = true THEN 1 END) as safety_incidents,
    COUNT(CASE WHEN service_interruption = true THEN 1 END) as service_disruptions,
    SUM(customers_affected) as total_customers_affected,
    SUM(estimated_repair_cost) as total_repair_costs,
    AVG(duration_of_outage_hours) as avg_outage_duration,
    COUNT(CASE WHEN utility_survey_conducted = true THEN 1 END) as surveys_conducted,
    ROUND(
        CAST(COUNT(CASE WHEN utility_survey_conducted = true THEN 1 END) AS DECIMAL) /
        NULLIF(COUNT(*), 0) * 100, 2
    ) as survey_percentage
FROM usag_strike_incidents
GROUP BY EXTRACT(YEAR FROM incident_date)
ORDER BY incident_year;

-- Utility type strike analysis
CREATE MATERIALIZED VIEW utility_type_strike_analysis AS
SELECT
    ut.utility_name,
    ut.utility_category,
    ut.criticality_level,
    COUNT(usi.id) as total_strikes,
    ROUND(AVG(usi.estimated_repair_cost), 2) as avg_repair_cost,
    SUM(usi.customers_affected) as total_customers_affected,
    COUNT(CASE WHEN usi.safety_incident = true THEN 1 END) as safety_incidents,
    COUNT(CASE WHEN usi.utility_survey_conducted = true THEN 1 END) as surveyed_incidents,
    ROUND(
        CAST(COUNT(CASE WHEN usi.utility_survey_conducted = true THEN 1 END) AS DECIMAL) /
        NULLIF(COUNT(usi.id), 0) * 100, 2
    ) as survey_rate_percentage
FROM utility_types ut
LEFT JOIN usag_strike_incidents usi ON ut.id = usi.utility_type_id
GROUP BY ut.utility_name, ut.utility_category, ut.criticality_level
ORDER BY total_strikes DESC;

-- Contractor performance analysis
CREATE MATERIALIZED VIEW contractor_performance_analysis AS
SELECT
    excavation_contractor,
    COUNT(*) as total_incidents,
    COUNT(CASE WHEN utility_survey_conducted = true THEN 1 END) as incidents_with_survey,
    ROUND(
        CAST(COUNT(CASE WHEN utility_survey_conducted = true THEN 1 END) AS DECIMAL) /
        NULLIF(COUNT(*), 0) * 100, 2
    ) as survey_compliance_percentage,
    COUNT(CASE WHEN safety_incident = true THEN 1 END) as safety_incidents,
    SUM(estimated_repair_cost) as total_costs,
    AVG(estimated_repair_cost) as avg_cost_per_incident,
    COUNT(CASE WHEN survey_quality_level IN ('QL-A', 'QL-B') THEN 1 END) as high_quality_surveys
FROM usag_strike_incidents
WHERE excavation_contractor IS NOT NULL
GROUP BY excavation_contractor
HAVING COUNT(*) >= 3 -- Only contractors with 3+ incidents
ORDER BY total_incidents DESC;

-- ================================================================
-- UTILITY FUNCTIONS FOR STRIKE ANALYSIS
-- ================================================================

-- Function to calculate strike risk score for a location
CREATE OR REPLACE FUNCTION calculate_strike_risk_score(
    location_point GEOMETRY,
    radius_meters INTEGER DEFAULT 1000,
    analysis_period_years INTEGER DEFAULT 5
) RETURNS DECIMAL AS $$
DECLARE
    risk_score DECIMAL := 0;
    recent_incidents INTEGER;
    high_severity_incidents INTEGER;
    avg_cost DECIMAL;
    utility_variety INTEGER;
BEGIN
    -- Count recent incidents within radius
    SELECT COUNT(*)
    INTO recent_incidents
    FROM usag_strike_incidents
    WHERE ST_DWithin(
        ST_Transform(incident_location, 3857),
        ST_Transform(location_point, 3857),
        radius_meters
    )
    AND incident_date >= CURRENT_DATE - (analysis_period_years * INTERVAL '1 year');

    -- Count high severity incidents
    SELECT COUNT(*)
    INTO high_severity_incidents
    FROM usag_strike_incidents usi
    JOIN strike_incident_categories sic ON usi.incident_category_id = sic.id
    WHERE ST_DWithin(
        ST_Transform(usi.incident_location, 3857),
        ST_Transform(location_point, 3857),
        radius_meters
    )
    AND usi.incident_date >= CURRENT_DATE - (analysis_period_years * INTERVAL '1 year')
    AND sic.severity_level IN ('critical', 'catastrophic');

    -- Get average cost of incidents
    SELECT COALESCE(AVG(estimated_repair_cost), 0)
    INTO avg_cost
    FROM usag_strike_incidents
    WHERE ST_DWithin(
        ST_Transform(incident_location, 3857),
        ST_Transform(location_point, 3857),
        radius_meters
    )
    AND incident_date >= CURRENT_DATE - (analysis_period_years * INTERVAL '1 year');

    -- Count utility type variety
    SELECT COUNT(DISTINCT utility_type_id)
    INTO utility_variety
    FROM usag_strike_incidents
    WHERE ST_DWithin(
        ST_Transform(incident_location, 3857),
        ST_Transform(location_point, 3857),
        radius_meters
    )
    AND incident_date >= CURRENT_DATE - (analysis_period_years * INTERVAL '1 year');

    -- Calculate composite risk score (0-100 scale)
    risk_score :=
        (recent_incidents * 10) +
        (high_severity_incidents * 20) +
        (LEAST(avg_cost / 10000, 25)) +
        (utility_variety * 5);

    RETURN LEAST(risk_score, 100.0);
END;
$$ LANGUAGE plpgsql;

-- Function to refresh strike analysis views
CREATE OR REPLACE FUNCTION refresh_strike_analysis_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW annual_strike_statistics;
    REFRESH MATERIALIZED VIEW utility_type_strike_analysis;
    REFRESH MATERIALIZED VIEW contractor_performance_analysis;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- TRIGGERS FOR DATA QUALITY
-- ================================================================

-- Function to validate incident data quality
CREATE OR REPLACE FUNCTION validate_incident_data_quality()
RETURNS TRIGGER AS $$
BEGIN
    -- Calculate data quality score based on completeness
    NEW.data_quality_score :=
        CASE WHEN NEW.incident_description IS NOT NULL THEN 0.2 ELSE 0 END +
        CASE WHEN NEW.location_description IS NOT NULL THEN 0.2 ELSE 0 END +
        CASE WHEN NEW.immediate_cause IS NOT NULL THEN 0.2 ELSE 0 END +
        CASE WHEN NEW.excavation_contractor IS NOT NULL THEN 0.2 ELSE 0 END +
        CASE WHEN NEW.estimated_repair_cost IS NOT NULL THEN 0.2 ELSE 0 END;

    -- Calculate data completeness score
    NEW.data_completeness_score :=
        (CASE WHEN NEW.incident_date IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN NEW.incident_location IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN NEW.incident_description IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN NEW.utility_type_id IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN NEW.incident_category_id IS NOT NULL THEN 1 ELSE 0 END) / 5.0;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_incident_quality_trigger
    BEFORE INSERT OR UPDATE ON usag_strike_incidents
    FOR EACH ROW EXECUTE FUNCTION validate_incident_data_quality();

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE usag_strike_incidents IS 'Comprehensive USAG strike incident reports with detailed analysis fields';
COMMENT ON TABLE strike_incident_categories IS 'Classification system for different types of utility strike incidents';
COMMENT ON TABLE utility_types IS 'Standardized utility type definitions with characteristics and risk levels';
COMMENT ON TABLE strike_pattern_analysis IS 'Spatial and temporal pattern analysis of strike incidents';
COMMENT ON TABLE strike_prevention_measures IS 'Tracking of prevention measures and their effectiveness';

COMMENT ON FUNCTION calculate_strike_risk_score IS 'Calculates composite risk score for utility strikes at a given location';

COMMENT ON MATERIALIZED VIEW annual_strike_statistics IS 'Annual summary statistics of utility strike incidents';
COMMENT ON MATERIALIZED VIEW utility_type_strike_analysis IS 'Strike analysis broken down by utility type and characteristics';
COMMENT ON MATERIALIZED VIEW contractor_performance_analysis IS 'Performance analysis of excavation contractors regarding strike prevention';