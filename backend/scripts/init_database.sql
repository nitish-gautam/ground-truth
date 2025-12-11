-- Database Initialization Script
-- ================================
-- Creates tables for LiDAR, Hyperspectral, and BIM data models
-- Requires: PostgreSQL 12+ with PostGIS extension

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS lidar_elevation_profiles CASCADE;
DROP TABLE IF EXISTS lidar_point_cloud_coverage CASCADE;
DROP TABLE IF EXISTS lidar_dtm_tiles CASCADE;
DROP TABLE IF EXISTS concrete_strength_calibrations CASCADE;
DROP TABLE IF EXISTS hyperspectral_analyses CASCADE;
DROP TABLE IF EXISTS hyperspectral_material_samples CASCADE;
DROP TABLE IF EXISTS bim_elements CASCADE;
DROP TABLE IF EXISTS architectural_scans CASCADE;
DROP TABLE IF EXISTS bim_test_models CASCADE;

-- ============================================================================
-- LIDAR TABLES
-- ============================================================================

-- LiDAR DTM Tiles
CREATE TABLE lidar_dtm_tiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tile_name VARCHAR(50) UNIQUE NOT NULL,
    grid_reference VARCHAR(10),
    file_path TEXT,
    file_size_mb FLOAT,
    bounds GEOMETRY(POLYGON, 27700),  -- British National Grid
    resolution_meters FLOAT DEFAULT 1.0,
    min_elevation FLOAT,
    max_elevation FLOAT,
    mean_elevation FLOAT,
    std_elevation FLOAT,
    capture_year INTEGER,
    capture_date DATE,
    source VARCHAR(100),
    dataset_name VARCHAR(100),
    is_accessible BOOLEAN DEFAULT TRUE,
    is_processed BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_lidar_dtm_bounds ON lidar_dtm_tiles USING GIST(bounds);
CREATE INDEX idx_lidar_dtm_tile_name ON lidar_dtm_tiles(tile_name);
CREATE INDEX idx_lidar_dtm_year ON lidar_dtm_tiles(capture_year);

-- LiDAR Point Cloud Coverage (Historical)
CREATE TABLE lidar_point_cloud_coverage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    year INTEGER NOT NULL,
    tile_reference VARCHAR(10),
    tile_name VARCHAR(50),
    coverage_area GEOMETRY(POLYGON, 27700),
    data_available BOOLEAN DEFAULT TRUE,
    data_quality VARCHAR(20),
    point_density FLOAT,
    has_rgb BOOLEAN DEFAULT FALSE,
    has_intensity BOOLEAN DEFAULT FALSE,
    has_classification BOOLEAN DEFAULT FALSE,
    provider VARCHAR(100),
    capture_date_start DATE,
    capture_date_end DATE,
    source_format VARCHAR(20),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_lidar_coverage_area ON lidar_point_cloud_coverage USING GIST(coverage_area);
CREATE INDEX idx_lidar_coverage_year ON lidar_point_cloud_coverage(year);

-- LiDAR Elevation Profiles
CREATE TABLE lidar_elevation_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_name VARCHAR(200) NOT NULL,
    description TEXT,
    line_geometry GEOMETRY(LINESTRING, 27700) NOT NULL,
    num_samples INTEGER NOT NULL,
    profile_length_m FLOAT NOT NULL,
    min_elevation FLOAT,
    max_elevation FLOAT,
    elevation_gain FLOAT,
    elevation_loss FLOAT,
    elevation_data JSONB NOT NULL,
    purpose VARCHAR(50),
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_profile_line ON lidar_elevation_profiles USING GIST(line_geometry);

-- ============================================================================
-- HYPERSPECTRAL TABLES
-- ============================================================================

-- Hyperspectral Material Samples (Training Data)
CREATE TABLE hyperspectral_material_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id VARCHAR(50) UNIQUE NOT NULL,
    sample_name VARCHAR(200),
    material_type VARCHAR(50) NOT NULL,  -- concrete, asphalt, steel, etc.
    material_subtype VARCHAR(100),
    surface_condition VARCHAR(50),
    surface_age VARCHAR(50),
    moisture_level VARCHAR(50),
    image_path TEXT,
    image_format VARCHAR(20),
    resolution VARCHAR(50),
    spectral_signature JSONB,
    num_bands INTEGER DEFAULT 204,
    wavelength_range_nm VARCHAR(50) DEFAULT '400-1000',
    is_specim_compatible BOOLEAN DEFAULT TRUE,
    spectral_resolution_nm FLOAT,
    source VARCHAR(100),
    dataset_name VARCHAR(100) DEFAULT 'UMKC',
    quality_label VARCHAR(20) DEFAULT 'training',  -- training, validation, test
    is_augmented BOOLEAN DEFAULT FALSE,
    parent_sample_id UUID REFERENCES hyperspectral_material_samples(id),
    augmentation_method VARCHAR(50),
    ground_truth_strength_mpa FLOAT,
    ground_truth_moisture_pct FLOAT,
    ground_truth_defects JSONB,
    capture_date DATE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hyper_sample_material ON hyperspectral_material_samples(material_type);
CREATE INDEX idx_hyper_sample_quality ON hyperspectral_material_samples(quality_label);
CREATE INDEX idx_hyper_sample_augmented ON hyperspectral_material_samples(is_augmented);

-- Hyperspectral Analyses (Production Results)
CREATE TABLE hyperspectral_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_name VARCHAR(200),
    description TEXT,
    image_path TEXT,
    image_url TEXT,
    image_metadata JSONB,
    predicted_material VARCHAR(50),
    confidence_score FLOAT,
    material_probabilities JSONB,
    predicted_strength_mpa FLOAT,
    strength_confidence FLOAT,
    strength_range_min FLOAT,
    strength_range_max FLOAT,
    model_r_squared FLOAT,
    model_mae FLOAT,
    curing_quality_score FLOAT,
    moisture_content_pct FLOAT,
    defects_detected JSONB,
    defect_count INTEGER DEFAULT 0,
    defect_locations JSONB,
    defect_severity VARCHAR(20),
    spectral_signature JSONB,
    key_wavelengths JSONB,
    processing_time_ms INTEGER,
    calibration_id UUID REFERENCES concrete_strength_calibrations(id),
    project_id VARCHAR(100),
    location_info JSONB,
    analyzed_by VARCHAR(100),
    analyzed_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hyper_analysis_material ON hyperspectral_analyses(predicted_material);
CREATE INDEX idx_hyper_analysis_date ON hyperspectral_analyses(analyzed_at);
CREATE INDEX idx_hyper_analysis_project ON hyperspectral_analyses(project_id);

-- Concrete Strength Calibrations
CREATE TABLE concrete_strength_calibrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    calibration_name VARCHAR(200) NOT NULL,
    description TEXT,
    test_location VARCHAR(20) DEFAULT 'lab',  -- lab, field
    num_samples INTEGER NOT NULL,
    strength_range_min FLOAT,
    strength_range_max FLOAT,
    key_wavelengths FLOAT[],
    spectral_coefficients JSONB,
    model_type VARCHAR(50) DEFAULT 'regression',
    model_parameters JSONB,
    r_squared FLOAT NOT NULL,
    mae FLOAT NOT NULL,
    rmse FLOAT,
    training_data_ids UUID[],
    is_validated BOOLEAN DEFAULT FALSE,
    validation_date DATE,
    validated_by VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_calibration_location ON concrete_strength_calibrations(test_location);
CREATE INDEX idx_calibration_validated ON concrete_strength_calibrations(is_validated);
CREATE INDEX idx_calibration_r_squared ON concrete_strength_calibrations(r_squared DESC);

-- ============================================================================
-- BIM/IFC TABLES
-- ============================================================================

-- BIM Test Models
CREATE TABLE bim_test_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(200) NOT NULL,
    description TEXT,
    file_path TEXT,
    file_size_kb FLOAT,
    ifc_version VARCHAR(20) NOT NULL,  -- 2x3, 4.0.2.1, 4.3.2.0
    schema_name VARCHAR(50),
    num_elements INTEGER,
    element_counts JSONB,
    -- IFC 4.3 Infrastructure Features (CRITICAL for HS2)
    has_alignment BOOLEAN DEFAULT FALSE,
    has_bridge BOOLEAN DEFAULT FALSE,
    has_tunnel BOOLEAN DEFAULT FALSE,
    has_earthworks BOOLEAN DEFAULT FALSE,
    project_name VARCHAR(200),
    project_description TEXT,
    purpose VARCHAR(50),  -- test, training, production
    complexity_level VARCHAR(20),  -- simple, moderate, complex
    import_date TIMESTAMP,
    is_validated BOOLEAN DEFAULT FALSE,
    validation_errors TEXT[],
    validation_warnings TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_bim_ifc_version ON bim_test_models(ifc_version);
CREATE INDEX idx_bim_alignment ON bim_test_models(has_alignment);
CREATE INDEX idx_bim_infrastructure ON bim_test_models(has_bridge, has_tunnel, has_earthworks);

-- Architectural Scans
CREATE TABLE architectural_scans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_name VARCHAR(200) NOT NULL,
    description TEXT,
    building_type VARCHAR(100),
    building_name VARCHAR(200),
    location TEXT,
    complexity_level VARCHAR(20),  -- simple, moderate, complex
    num_floors INTEGER,
    total_area_m2 FLOAT,
    capture_method VARCHAR(50),  -- laser_scan, photogrammetry, structured_light
    point_cloud_path TEXT,
    point_count BIGINT,
    mesh_path TEXT,
    has_textures BOOLEAN DEFAULT FALSE,
    capture_date DATE,
    scan_provider VARCHAR(100),
    scan_equipment VARCHAR(100),
    coordinate_system VARCHAR(50),
    bounding_box JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_arch_scan_building_type ON architectural_scans(building_type);
CREATE INDEX idx_arch_scan_complexity ON architectural_scans(complexity_level);

-- BIM Elements (Extracted from IFC)
CREATE TABLE bim_elements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bim_model_id UUID REFERENCES bim_test_models(id) ON DELETE CASCADE,
    global_id VARCHAR(50) NOT NULL,
    ifc_type VARCHAR(50) NOT NULL,
    name VARCHAR(200),
    description TEXT,
    tag VARCHAR(100),
    element_properties JSONB,
    quantities JSONB,
    material_info JSONB,
    geometry_wkt TEXT,
    bounding_box JSONB,
    storey VARCHAR(100),
    space VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_bim_elem_model ON bim_elements(bim_model_id);
CREATE INDEX idx_bim_elem_type ON bim_elements(ifc_type);
CREATE INDEX idx_bim_elem_global_id ON bim_elements(global_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER update_lidar_dtm_updated_at
    BEFORE UPDATE ON lidar_dtm_tiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hyper_sample_updated_at
    BEFORE UPDATE ON hyperspectral_material_samples
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_calibration_updated_at
    BEFORE UPDATE ON concrete_strength_calibrations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bim_model_updated_at
    BEFORE UPDATE ON bim_test_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Accessible DTM Tiles Summary
CREATE VIEW v_accessible_dtm_tiles AS
SELECT
    tile_name,
    grid_reference,
    resolution_meters,
    min_elevation,
    max_elevation,
    capture_year,
    source,
    is_accessible
FROM lidar_dtm_tiles
WHERE is_accessible = TRUE
ORDER BY tile_name;

-- View: Hyperspectral Training Data Summary
CREATE VIEW v_hyperspectral_training_summary AS
SELECT
    material_type,
    COUNT(*) as sample_count,
    SUM(CASE WHEN is_augmented THEN 1 ELSE 0 END) as augmented_count,
    SUM(CASE WHEN quality_label = 'training' THEN 1 ELSE 0 END) as training_count,
    SUM(CASE WHEN quality_label = 'validation' THEN 1 ELSE 0 END) as validation_count,
    SUM(CASE WHEN quality_label = 'test' THEN 1 ELSE 0 END) as test_count,
    AVG(ground_truth_strength_mpa) as avg_strength_mpa,
    STDDEV(ground_truth_strength_mpa) as std_strength_mpa
FROM hyperspectral_material_samples
GROUP BY material_type
ORDER BY sample_count DESC;

-- View: BIM Infrastructure Models (IFC 4.3 Features)
CREATE VIEW v_bim_infrastructure_models AS
SELECT
    model_name,
    ifc_version,
    has_alignment,
    has_bridge,
    has_tunnel,
    has_earthworks,
    num_elements,
    purpose,
    import_date
FROM bim_test_models
WHERE has_alignment OR has_bridge OR has_tunnel OR has_earthworks
ORDER BY import_date DESC;

-- ============================================================================
-- GRANTS (Optional - adjust based on your user setup)
-- ============================================================================

-- Grant permissions to application user
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gpr_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gpr_app_user;

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Verify PostGIS installation
SELECT PostGIS_Full_Version();

-- Count tables created
SELECT COUNT(*) as tables_created
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_type = 'BASE TABLE'
AND table_name IN (
    'lidar_dtm_tiles', 'lidar_point_cloud_coverage', 'lidar_elevation_profiles',
    'hyperspectral_material_samples', 'hyperspectral_analyses', 'concrete_strength_calibrations',
    'bim_test_models', 'architectural_scans', 'bim_elements'
);

COMMENT ON TABLE lidar_dtm_tiles IS 'LiDAR Digital Terrain Model tiles - 1m resolution elevation data';
COMMENT ON TABLE hyperspectral_material_samples IS 'Training samples for Specim IQ hyperspectral analysis';
COMMENT ON TABLE hyperspectral_analyses IS 'Production hyperspectral analysis results - concrete quality assessment';
COMMENT ON TABLE bim_test_models IS 'BIM/IFC models with IFC 4.3 infrastructure support for HS2';
