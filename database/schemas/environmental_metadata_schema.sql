-- ================================================================
-- COMPREHENSIVE ENVIRONMENTAL METADATA SCHEMA
-- Underground Utility Detection Platform - Twente Dataset Integration
-- All 25+ Environmental and Survey Metadata Fields
-- ================================================================

-- Environmental conditions lookup tables for normalization
CREATE TABLE ground_conditions (
    id SERIAL PRIMARY KEY,
    condition_code VARCHAR(20) UNIQUE NOT NULL,
    condition_name VARCHAR(100) NOT NULL,
    description TEXT,
    typical_permittivity_range NUMRANGE,
    penetration_characteristics TEXT
);

CREATE TABLE weather_conditions (
    id SERIAL PRIMARY KEY,
    condition_code VARCHAR(20) UNIQUE NOT NULL,
    condition_name VARCHAR(100) NOT NULL,
    impact_on_gpr TEXT,
    recommended_settings JSONB
);

CREATE TABLE land_use_types (
    id SERIAL PRIMARY KEY,
    land_use_code VARCHAR(20) UNIQUE NOT NULL,
    land_use_name VARCHAR(100) NOT NULL,
    typical_utilities TEXT[],
    access_constraints TEXT
);

CREATE TABLE surface_materials (
    id SERIAL PRIMARY KEY,
    material_code VARCHAR(20) UNIQUE NOT NULL,
    material_name VARCHAR(100) NOT NULL,
    dielectric_properties JSONB,
    signal_attenuation_factor DECIMAL(6,4)
);

-- ================================================================
-- ENHANCED ENVIRONMENTAL METADATA TABLE
-- All Twente Dataset Fields Properly Structured
-- ================================================================

CREATE TABLE environmental_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- == LOCATION AND SITE CONTEXT (Fields 1-4) ==
    location_id VARCHAR(50) NOT NULL, -- 01.1, 01.2, etc. (Twente format)
    site_classification VARCHAR(100), -- Construction site type
    surveying_objective VARCHAR(200) NOT NULL, -- Primary survey objective
    secondary_objectives TEXT[], -- Additional survey goals

    -- == CONSTRUCTION AND WORK CONTEXT (Fields 5-7) ==
    construction_workers_present VARCHAR(50), -- "Yes", "No", "Unknown"
    construction_phase VARCHAR(100), -- Planning, Active, Completed
    complementary_works TEXT[], -- Array of concurrent work activities
    exact_location_accuracy_required BOOLEAN DEFAULT FALSE,

    -- == ENVIRONMENTAL CONDITIONS (Fields 8-12) ==
    ground_condition_id INTEGER REFERENCES ground_conditions(id),
    ground_condition_details TEXT, -- Additional ground description
    ground_relative_permittivity DECIMAL(6,3) CHECK (ground_relative_permittivity > 0),
    permittivity_measurement_method VARCHAR(100),
    permittivity_confidence VARCHAR(20) CHECK (permittivity_confidence IN ('low', 'medium', 'high')),

    -- == GROUNDWATER CONDITIONS (Fields 13-15) ==
    relative_groundwater_level VARCHAR(100), -- "High", "Medium", "Low", "Dry"
    groundwater_depth_estimate_m DECIMAL(6,3),
    groundwater_seasonal_variation TEXT,
    drainage_conditions VARCHAR(100),

    -- == SURFACE AND TERRAIN (Fields 16-20) ==
    land_cover_id INTEGER REFERENCES surface_materials(id),
    land_cover_details TEXT,
    terrain_levelling VARCHAR(50) CHECK (terrain_levelling IN ('Flat', 'Gently Sloping', 'Steep', 'Irregular')),
    terrain_smoothness VARCHAR(50) CHECK (terrain_smoothness IN ('Smooth', 'Slightly Rough', 'Rough', 'Very Rough')),
    surface_accessibility VARCHAR(100), -- Access constraints for survey equipment

    -- == WEATHER CONDITIONS (Fields 21-23) ==
    weather_condition_id INTEGER REFERENCES weather_conditions(id),
    temperature_celsius DECIMAL(5,2),
    humidity_percentage DECIMAL(5,2),
    precipitation_24h_mm DECIMAL(6,2) DEFAULT 0.0,
    wind_speed_ms DECIMAL(5,2),
    atmospheric_pressure_hpa DECIMAL(7,2),

    -- == SOIL CONTAMINATION AND DISTURBANCES (Fields 24-28) ==
    rubble_presence BOOLEAN DEFAULT FALSE,
    rubble_type VARCHAR(100), -- Type of rubble/debris
    rubble_density VARCHAR(50), -- Sparse, Moderate, Dense

    tree_roots_presence BOOLEAN DEFAULT FALSE,
    vegetation_density VARCHAR(50), -- None, Light, Moderate, Dense
    root_depth_estimate_m DECIMAL(5,2),

    polluted_soil_presence BOOLEAN DEFAULT FALSE,
    pollution_type VARCHAR(100), -- Chemical, petroleum, etc.
    pollution_severity VARCHAR(50),

    blast_furnace_slag_presence BOOLEAN DEFAULT FALSE,
    slag_characteristics TEXT,

    other_ground_disturbances TEXT[], -- Additional disturbances

    -- == UTILITY INFRASTRUCTURE CONTEXT (Fields 29-35) ==
    amount_of_utilities INTEGER DEFAULT 0 CHECK (amount_of_utilities >= 0),
    utility_density VARCHAR(50), -- Low, Medium, High, Very High

    utility_crossing BOOLEAN DEFAULT FALSE,
    crossing_types TEXT[], -- Types of crossings present
    crossing_complexity VARCHAR(50),

    utility_path_linear BOOLEAN DEFAULT TRUE,
    path_complexity VARCHAR(50), -- Simple, Moderate, Complex

    known_utility_records_available BOOLEAN DEFAULT FALSE,
    utility_records_quality VARCHAR(50), -- Poor, Fair, Good, Excellent
    records_last_updated DATE,

    -- == LAND USE AND ZONING (Additional Context) ==
    land_use_id INTEGER REFERENCES land_use_types(id),
    zoning_classification VARCHAR(100),
    development_age_years INTEGER,
    infrastructure_modernization_level VARCHAR(50),

    -- == SURVEY CONDITIONS AND CONSTRAINTS ==
    survey_time_constraints TEXT, -- Time limitations for survey
    access_permissions_required BOOLEAN DEFAULT FALSE,
    safety_considerations TEXT[],
    equipment_limitations TEXT[],

    -- == GEOLOGICAL CONTEXT ==
    geological_formation VARCHAR(100),
    soil_layer_structure JSONB, -- Layered soil description
    bedrock_depth_estimate_m DECIMAL(6,3),
    geological_complexity VARCHAR(50),

    -- == ELECTROMAGNETIC ENVIRONMENT ==
    electromagnetic_interference_sources TEXT[],
    nearby_metallic_structures TEXT[],
    power_lines_proximity_m DECIMAL(6,2),
    radio_transmitters_nearby BOOLEAN DEFAULT FALSE,

    -- == METADATA AND QUALITY ==
    metadata_completeness_score DECIMAL(3,2) CHECK (metadata_completeness_score >= 0 AND metadata_completeness_score <= 1),
    data_collection_method VARCHAR(100), -- Site visit, records, estimation
    data_reliability_score DECIMAL(3,2) CHECK (data_reliability_score >= 0 AND data_reliability_score <= 1),
    verification_status VARCHAR(50) DEFAULT 'unverified',
    collected_by VARCHAR(255),
    collection_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- ENVIRONMENTAL IMPACT ANALYSIS TABLE
-- ================================================================

CREATE TABLE environmental_impact_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    environmental_metadata_id UUID NOT NULL REFERENCES environmental_metadata(id) ON DELETE CASCADE,
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Signal propagation impacts
    estimated_signal_velocity_m_ns DECIMAL(6,4),
    velocity_uncertainty_m_ns DECIMAL(6,4),
    attenuation_coefficient_db_m DECIMAL(8,4),

    -- Detection performance impacts
    expected_penetration_depth_m DECIMAL(6,3),
    resolution_degradation_factor DECIMAL(5,4),
    noise_level_increase_db DECIMAL(6,2),

    -- Environmental factor correlations
    moisture_correlation_coefficient DECIMAL(6,4),
    temperature_correlation_coefficient DECIMAL(6,4),
    soil_type_correlation_coefficient DECIMAL(6,4),
    contamination_impact_factor DECIMAL(6,4),

    -- Calibration requirements
    calibration_adjustments_required BOOLEAN DEFAULT FALSE,
    recommended_antenna_frequency_mhz INTEGER,
    recommended_trace_spacing_m DECIMAL(5,3),
    recommended_gain_settings JSONB,

    -- Prediction confidence
    prediction_confidence_score DECIMAL(3,2),
    uncertainty_sources TEXT[],

    -- Analysis metadata
    analysis_algorithm VARCHAR(100),
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analyst_name VARCHAR(255),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- TEMPORAL ENVIRONMENTAL TRACKING
-- ================================================================

CREATE TABLE environmental_monitoring (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID NOT NULL REFERENCES survey_sites(id) ON DELETE CASCADE,

    -- Monitoring period
    monitoring_date DATE NOT NULL,
    monitoring_time TIME,

    -- Real-time environmental conditions
    air_temperature_celsius DECIMAL(5,2),
    ground_temperature_celsius DECIMAL(5,2),
    relative_humidity_percentage DECIMAL(5,2),
    soil_moisture_percentage DECIMAL(5,2),

    -- Weather conditions
    weather_condition VARCHAR(100),
    precipitation_current_mm_h DECIMAL(6,3),
    wind_speed_current_ms DECIMAL(5,2),

    -- Ground conditions
    ground_state VARCHAR(50), -- Frozen, Thawed, Saturated, Dry
    surface_water_presence BOOLEAN DEFAULT FALSE,

    -- Electromagnetic environment
    em_noise_level_db DECIMAL(6,2),
    radio_frequency_interference BOOLEAN DEFAULT FALSE,

    -- Equipment performance correlation
    signal_quality_impact VARCHAR(50),
    calibration_drift_detected BOOLEAN DEFAULT FALSE,

    -- Monitoring equipment
    monitoring_equipment VARCHAR(255),
    measurement_accuracy VARCHAR(100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- LOOKUP TABLE DATA POPULATION
-- ================================================================

-- Insert ground conditions
INSERT INTO ground_conditions (condition_code, condition_name, description, typical_permittivity_range, penetration_characteristics) VALUES
('SAND', 'Sandy', 'Predominantly sandy soil composition', '[3,6)', 'Good penetration, low attenuation'),
('CLAY', 'Clayey', 'Clay-rich soil with high moisture retention', '[8,25)', 'Moderate penetration, higher attenuation'),
('SILT', 'Silty', 'Fine-grained silty soil', '[5,15)', 'Moderate penetration, variable attenuation'),
('GRAVEL', 'Gravelly', 'Coarse gravel and stone mixture', '[4,8)', 'Variable penetration, scattering effects'),
('ORGANIC', 'Organic', 'Organic-rich soil with vegetation', '[6,20)', 'Poor penetration, high attenuation'),
('MIXED', 'Mixed', 'Heterogeneous soil composition', '[5,20)', 'Highly variable characteristics'),
('ROCK', 'Rocky', 'Bedrock or rocky substrate', '[8,12)', 'Limited penetration, strong reflections'),
('FILL', 'Engineered Fill', 'Artificial fill material', '[4,15)', 'Variable, depends on composition');

-- Insert weather conditions
INSERT INTO weather_conditions (condition_code, condition_name, impact_on_gpr, recommended_settings) VALUES
('DRY', 'Dry', 'Optimal conditions for GPR surveying', '{"gain": "standard", "frequency": "500MHz"}'),
('HUMID', 'Humid', 'Slight increase in signal attenuation', '{"gain": "increased", "frequency": "500MHz"}'),
('RAINY', 'Rainy', 'Significant impact on surface coupling', '{"gain": "high", "frequency": "400MHz", "note": "Consider postponing"}'),
('SNOW', 'Snowy', 'Surface coupling issues, cold affects electronics', '{"gain": "high", "frequency": "400MHz", "antenna_protection": "required"}'),
('STORM', 'Stormy', 'Unsuitable for surveying', '{"recommendation": "postpone_survey"}'),
('FOG', 'Foggy', 'Minimal direct impact on GPR', '{"gain": "standard", "frequency": "500MHz"}');

-- Insert land use types
INSERT INTO land_use_types (land_use_code, land_use_name, typical_utilities, access_constraints) VALUES
('RESID', 'Residential', ARRAY['water', 'sewer', 'gas', 'electricity', 'telecommunications'], 'Property access permissions required'),
('COMM', 'Commercial', ARRAY['water', 'sewer', 'electricity', 'telecommunications', 'HVAC'], 'Business hours restrictions'),
('INDUST', 'Industrial', ARRAY['water', 'sewer', 'electricity', 'gas', 'steam', 'chemicals'], 'Safety clearances required'),
('ROAD', 'Roadway', ARRAY['water', 'sewer', 'electricity', 'telecommunications', 'traffic signals'], 'Traffic management needed'),
('PARK', 'Parkland', ARRAY['water', 'electricity', 'irrigation'], 'Minimal access restrictions'),
('AGRIC', 'Agricultural', ARRAY['water', 'irrigation', 'drainage'], 'Seasonal access limitations'),
('CONSTR', 'Construction Site', ARRAY['temporary utilities', 'power', 'water'], 'Active construction hazards');

-- Insert surface materials
INSERT INTO surface_materials (material_code, material_name, dielectric_properties, signal_attenuation_factor) VALUES
('ASPH', 'Asphalt', '{"permittivity": 6.5, "conductivity": 0.001}', 0.15),
('CONC', 'Concrete', '{"permittivity": 8.5, "conductivity": 0.01}', 0.25),
('BRICK', 'Brick Road', '{"permittivity": 7.2, "conductivity": 0.005}', 0.20),
('GRAVEL', 'Gravel', '{"permittivity": 5.8, "conductivity": 0.0001}', 0.10),
('GRASS', 'Grass/Vegetation', '{"permittivity": 12.0, "conductivity": 0.01, "moisture_dependent": true}', 0.35),
('SOIL', 'Bare Soil', '{"permittivity": 9.0, "conductivity": 0.005, "moisture_dependent": true}', 0.30),
('PAVE', 'Paver Stones', '{"permittivity": 7.0, "conductivity": 0.002}', 0.18),
('DIRT', 'Dirt Road', '{"permittivity": 8.0, "conductivity": 0.008, "moisture_dependent": true}', 0.28);

-- ================================================================
-- PERFORMANCE INDEXES
-- ================================================================

-- Environmental metadata indexes
CREATE INDEX idx_environmental_metadata_survey ON environmental_metadata(survey_id);
CREATE INDEX idx_environmental_metadata_location ON environmental_metadata(location_id);
CREATE INDEX idx_environmental_ground_condition ON environmental_metadata(ground_condition_id);
CREATE INDEX idx_environmental_weather ON environmental_metadata(weather_condition_id);
CREATE INDEX idx_environmental_land_use ON environmental_metadata(land_use_id);
CREATE INDEX idx_environmental_permittivity ON environmental_metadata(ground_relative_permittivity);
CREATE INDEX idx_environmental_utilities_count ON environmental_metadata(amount_of_utilities);

-- Environmental impact analysis indexes
CREATE INDEX idx_impact_analysis_env_metadata ON environmental_impact_analysis(environmental_metadata_id);
CREATE INDEX idx_impact_analysis_survey ON environmental_impact_analysis(survey_id);
CREATE INDEX idx_impact_velocity ON environmental_impact_analysis(estimated_signal_velocity_m_ns);
CREATE INDEX idx_impact_penetration ON environmental_impact_analysis(expected_penetration_depth_m);

-- Environmental monitoring indexes
CREATE INDEX idx_monitoring_site ON environmental_monitoring(site_id);
CREATE INDEX idx_monitoring_date ON environmental_monitoring(monitoring_date);
CREATE INDEX idx_monitoring_conditions ON environmental_monitoring(air_temperature_celsius, relative_humidity_percentage);

-- Lookup table indexes
CREATE INDEX idx_ground_conditions_permittivity ON ground_conditions USING GIST(typical_permittivity_range);
CREATE INDEX idx_surface_materials_attenuation ON surface_materials(signal_attenuation_factor);

-- ================================================================
-- MATERIALIZED VIEWS FOR ENVIRONMENTAL ANALYSIS
-- ================================================================

-- Environmental conditions summary
CREATE MATERIALIZED VIEW environmental_conditions_summary AS
SELECT
    em.survey_id,
    em.location_id,
    gc.condition_name as ground_condition,
    em.ground_relative_permittivity,
    wc.condition_name as weather_condition,
    sm.material_name as surface_material,
    lu.land_use_name as land_use,
    em.amount_of_utilities,
    em.utility_crossing,
    em.rubble_presence,
    em.tree_roots_presence,
    em.polluted_soil_presence,
    em.blast_furnace_slag_presence,
    eia.estimated_signal_velocity_m_ns,
    eia.expected_penetration_depth_m,
    eia.prediction_confidence_score
FROM environmental_metadata em
LEFT JOIN ground_conditions gc ON em.ground_condition_id = gc.id
LEFT JOIN weather_conditions wc ON em.weather_condition_id = wc.id
LEFT JOIN surface_materials sm ON em.land_cover_id = sm.id
LEFT JOIN land_use_types lu ON em.land_use_id = lu.id
LEFT JOIN environmental_impact_analysis eia ON em.id = eia.environmental_metadata_id;

-- Environmental factor correlations
CREATE MATERIALIZED VIEW environmental_correlations_summary AS
SELECT
    gc.condition_name as ground_condition,
    AVG(em.ground_relative_permittivity) as avg_permittivity,
    COUNT(DISTINCT em.survey_id) as survey_count,
    AVG(eia.estimated_signal_velocity_m_ns) as avg_velocity,
    AVG(eia.expected_penetration_depth_m) as avg_penetration,
    AVG(eia.prediction_confidence_score) as avg_confidence,
    STDDEV(eia.estimated_signal_velocity_m_ns) as velocity_std_dev
FROM environmental_metadata em
JOIN ground_conditions gc ON em.ground_condition_id = gc.id
LEFT JOIN environmental_impact_analysis eia ON em.id = eia.environmental_metadata_id
GROUP BY gc.condition_name
HAVING COUNT(DISTINCT em.survey_id) >= 3;

-- ================================================================
-- UTILITY FUNCTIONS
-- ================================================================

-- Function to calculate environmental impact score
CREATE OR REPLACE FUNCTION calculate_environmental_impact_score(
    survey_uuid UUID
) RETURNS DECIMAL AS $$
DECLARE
    impact_score DECIMAL := 1.0;
    env_record RECORD;
BEGIN
    SELECT em.* INTO env_record
    FROM environmental_metadata em
    WHERE em.survey_id = survey_uuid;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    -- Adjust score based on contamination factors
    IF env_record.rubble_presence THEN
        impact_score := impact_score * 0.85;
    END IF;

    IF env_record.tree_roots_presence THEN
        impact_score := impact_score * 0.90;
    END IF;

    IF env_record.polluted_soil_presence THEN
        impact_score := impact_score * 0.75;
    END IF;

    IF env_record.blast_furnace_slag_presence THEN
        impact_score := impact_score * 0.70;
    END IF;

    -- Adjust for utility complexity
    IF env_record.utility_crossing THEN
        impact_score := impact_score * 0.80;
    END IF;

    RETURN ROUND(impact_score, 4);
END;
$$ LANGUAGE plpgsql;

-- Function to refresh environmental analysis views
CREATE OR REPLACE FUNCTION refresh_environmental_analysis_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW environmental_conditions_summary;
    REFRESH MATERIALIZED VIEW environmental_correlations_summary;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE environmental_metadata IS 'Comprehensive environmental metadata with all 25+ Twente dataset fields for correlation analysis';
COMMENT ON TABLE environmental_impact_analysis IS 'Analysis of environmental factors impact on GPR signal characteristics and detection performance';
COMMENT ON TABLE environmental_monitoring IS 'Temporal tracking of environmental conditions for long-term analysis';
COMMENT ON TABLE ground_conditions IS 'Standardized ground condition types with dielectric properties';
COMMENT ON TABLE weather_conditions IS 'Weather condition impacts and recommended GPR settings';
COMMENT ON TABLE land_use_types IS 'Land use classifications with typical utility expectations';
COMMENT ON TABLE surface_materials IS 'Surface material properties affecting GPR signal propagation';