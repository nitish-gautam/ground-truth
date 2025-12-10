-- ============================================================================
-- HS2 AUTOMATED PROGRESS ASSURANCE - DATABASE SCHEMA
-- ============================================================================
-- Patent-Pending: Multi-Spectral Data Fusion for Non-Destructive Quality Assurance
--
-- This schema supports:
-- 1. Hyperspectral imaging for material quality verification
-- 2. LiDAR-based progress monitoring
-- 3. BIM-to-reality comparison and deviation analysis
-- 4. Automated progress reporting
--
-- Version: 1.0
-- Created: 2025-01-09
-- ============================================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "postgis_raster";  -- For raster hyperspectral data

-- ============================================================================
-- SECTION 1: HYPERSPECTRAL IMAGING
-- ============================================================================

-- Hyperspectral scans metadata
CREATE TABLE IF NOT EXISTS hyperspectral_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    site_location VARCHAR(255) NOT NULL,
    scan_date TIMESTAMPTZ NOT NULL,

    -- Camera specifications
    camera_model VARCHAR(100),  -- e.g., 'Specim IQ', 'Corning microHSI', 'Headwall Photonics'
    camera_serial_number VARCHAR(100),
    wavelength_range VARCHAR(50) NOT NULL,  -- e.g., '400-2500nm', '400-1000nm'
    wavelength_min_nm INTEGER,  -- e.g., 400
    wavelength_max_nm INTEGER,  -- e.g., 2500
    spectral_resolution FLOAT,  -- nm per band (e.g., 10nm)
    band_count INTEGER NOT NULL,  -- e.g., 204 bands
    spatial_resolution FLOAT,  -- meters per pixel (e.g., 0.01 = 1cm)
    swath_width_m FLOAT,  -- Width of scan area in meters
    scan_speed_ms FLOAT,  -- Scan speed in meters per second

    -- Environmental conditions (critical for calibration)
    solar_angle FLOAT,  -- Degrees
    solar_azimuth FLOAT,  -- Degrees
    atmospheric_conditions JSONB,  -- {temperature, humidity, visibility, pressure}
    weather VARCHAR(50),  -- 'Clear', 'Cloudy', 'Overcast', 'Rain'
    surface_temperature_c FLOAT,
    ambient_temperature_c FLOAT,
    relative_humidity FLOAT,  -- Percentage

    -- File storage
    raw_file_path TEXT NOT NULL,  -- S3: hyperspectral-data/raw/YYYY/MM/DD/scan_id.hdr
    processed_file_path TEXT,  -- S3: hyperspectral-data/processed/scan_id_corrected.tif
    calibration_file_path TEXT,  -- S3: hyperspectral-data/calibration/scan_id_cal.json
    thumbnail_path TEXT,  -- S3: hyperspectral-data/thumbnails/scan_id.jpg (RGB preview)
    file_size_bytes BIGINT,
    format VARCHAR(20) DEFAULT 'ENVI',  -- 'ENVI', 'HDF5', 'GeoTIFF'
    data_type VARCHAR(20),  -- 'BSQ' (Band Sequential), 'BIL' (Band Interleaved by Line), 'BIP' (Band Interleaved by Pixel)

    -- Geospatial (WGS84 - EPSG:4326)
    location GEOGRAPHY(POINT, 4326),  -- Center point of scan
    coverage_area GEOGRAPHY(POLYGON, 4326),  -- Footprint of scan
    elevation_m FLOAT,
    coordinate_system VARCHAR(100) DEFAULT 'EPSG:4326',

    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
    processing_started_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ,
    processing_duration_seconds INTEGER,
    error_message TEXT,

    -- Quality metrics
    image_quality_score FLOAT,  -- 0-100, calculated from SNR, spatial coverage
    signal_to_noise_ratio FLOAT,  -- SNR across all bands
    spatial_coverage_percent FLOAT,  -- % of intended area captured

    -- Metadata
    operator_name VARCHAR(255),
    notes TEXT,
    tags TEXT[],  -- Array of tags: ['concrete', 'phase1', 'building_a']

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Material quality assessments from hyperspectral analysis
CREATE TABLE IF NOT EXISTS material_quality_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES hyperspectral_scans(id) ON DELETE CASCADE,

    -- Material identification
    material_type VARCHAR(100) NOT NULL,  -- 'concrete', 'steel', 'asphalt', 'soil', 'wood'
    material_subtype VARCHAR(100),  -- 'C40 concrete', 'Grade 500 steel', 'Asphalt Mix Type 1'
    material_age_days INTEGER,  -- Days since placement/installation
    specification_reference VARCHAR(255),  -- Reference to project spec (e.g., 'BS 8500-1:2015')

    -- Spatial location
    location_in_scan GEOGRAPHY(POINT, 4326),  -- GPS coordinates
    pixel_coordinates JSONB,  -- {x: 1024, y: 768, width: 50, height: 50}
    region_area_m2 FLOAT,
    depth_from_surface_mm FLOAT,  -- For sub-surface analysis

    -- Quality metrics (concrete-specific)
    predicted_strength_mpa FLOAT,  -- Compressive strength prediction
    strength_confidence FLOAT CHECK (strength_confidence >= 0 AND strength_confidence <= 100),  -- 0-100%
    specification_strength_mpa FLOAT,  -- Required strength from specs
    meets_specification BOOLEAN,
    strength_margin_mpa FLOAT,  -- Actual - Required (positive = over-spec, negative = under-spec)

    -- Quality metrics (general)
    moisture_content_percent FLOAT,  -- % moisture by weight
    density_kg_m3 FLOAT,
    surface_temperature_c FLOAT,
    chloride_content_percent FLOAT,  -- For corrosion risk

    -- Defects detected
    defects_detected JSONB,  -- Array of {type, severity, location_px, confidence, area_m2, depth_mm}
    /*
    Example defects_detected:
    [
        {
            "type": "void",
            "severity": "critical",
            "location_px": {"x": 512, "y": 384},
            "confidence": 0.92,
            "area_m2": 0.045,
            "depth_mm": 150
        },
        {
            "type": "crack",
            "severity": "moderate",
            "location_px": {"x": 600, "y": 400},
            "confidence": 0.88,
            "width_mm": 2.5,
            "length_mm": 350
        }
    ]
    */
    defect_count INTEGER DEFAULT 0,
    critical_defects INTEGER DEFAULT 0,
    major_defects INTEGER DEFAULT 0,
    minor_defects INTEGER DEFAULT 0,

    -- Spectral evidence (material fingerprint)
    spectral_signature JSONB,  -- Key wavelengths and reflectance values
    /*
    Example spectral_signature:
    {
        "wavelengths_nm": [400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        "reflectance": [0.15, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32, 0.34, 0.36],
        "absorption_features": [
            {"wavelength": 1450, "type": "water", "depth": 0.15},
            {"wavelength": 1950, "type": "water", "depth": 0.20}
        ]
    }
    */
    spectral_match_score FLOAT CHECK (spectral_match_score >= 0 AND spectral_match_score <= 100),  -- Similarity to reference library (0-100)
    reference_material_id UUID,  -- FK to spectral_library

    -- Overall quality score
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 100),  -- 0-100 composite score
    quality_grade VARCHAR(10),  -- 'A' (Excellent), 'B' (Good), 'C' (Acceptable), 'D' (Poor), 'F' (Fail)
    pass_fail_status VARCHAR(20),  -- 'PASS', 'FAIL', 'CONDITIONAL'

    -- Model metadata
    model_name VARCHAR(100),  -- 'concrete_strength_cnn_v2.1'
    model_version VARCHAR(50),
    model_confidence FLOAT CHECK (model_confidence >= 0 AND model_confidence <= 100),
    inference_date TIMESTAMPTZ DEFAULT NOW(),

    -- Actions required
    requires_verification BOOLEAN DEFAULT FALSE,
    requires_remediation BOOLEAN DEFAULT FALSE,
    recommended_action TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spectral library (reference materials database)
CREATE TABLE IF NOT EXISTS spectral_library (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Material identity
    material_name VARCHAR(255) NOT NULL,  -- 'C40 Concrete, 28-day cure, Type I Portland Cement'
    material_category VARCHAR(100) NOT NULL,  -- 'Concrete', 'Steel', 'Asphalt', 'Soil', 'Wood'
    material_grade VARCHAR(100),  -- 'C40', 'Grade 500', etc.
    manufacturer VARCHAR(255),
    batch_number VARCHAR(100),
    material_properties JSONB,  -- Known physical properties (strength, density, porosity, etc.)
    /*
    Example material_properties:
    {
        "compressive_strength_mpa": 42.5,
        "density_kg_m3": 2400,
        "water_cement_ratio": 0.45,
        "cement_type": "Type I Portland",
        "aggregate_type": "Limestone",
        "curing_method": "Moist curing, 28 days"
    }
    */

    -- Spectral signature
    wavelengths FLOAT[] NOT NULL,  -- Array of wavelengths (nm): [400, 410, 420, ..., 2500]
    reflectance_values FLOAT[] NOT NULL,  -- Corresponding reflectance (0-1): [0.15, 0.16, ...]
    spectral_curve JSONB,  -- Full spectral curve data with metadata
    spectral_resolution FLOAT,  -- nm per band
    measurement_geometry VARCHAR(100),  -- 'Nadir', '45°/0°', 'Hemispherical'

    -- Acquisition conditions
    acquisition_date DATE,
    lab_conditions JSONB,  -- {lighting, angle, atmospheric, temperature}
    calibration_method VARCHAR(100),  -- 'Spectralon White Reference', 'Dark Current Subtraction'
    instrument_model VARCHAR(100),

    -- Validation data (ground truth from destructive testing)
    lab_test_results JSONB,  -- Destructive test results for validation
    /*
    Example lab_test_results:
    {
        "compressive_strength_mpa": 42.5,
        "test_standard": "BS EN 12390-3:2019",
        "test_date": "2025-01-09",
        "lab_name": "Acme Testing Laboratory",
        "certificate_number": "CERT-2025-001234"
    }
    */
    sample_source VARCHAR(255),  -- 'Site A, Building 3, Column C12'
    validation_confidence FLOAT CHECK (validation_confidence >= 0 AND validation_confidence <= 100),

    -- Usage metadata
    is_validated BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,  -- Number of times referenced in assessments
    last_used_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hyperspectral-LiDAR fusion (combined analysis)
CREATE TABLE IF NOT EXISTS hyperspectral_lidar_fusion (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hyperspectral_scan_id UUID REFERENCES hyperspectral_scans(id) ON DELETE CASCADE,
    lidar_scan_id UUID,  -- Will reference progress_lidar_scans(id) once created

    -- Alignment data
    transformation_matrix JSONB,  -- 4x4 transformation for co-registration
    /*
    Example transformation_matrix:
    {
        "matrix": [
            [0.9998, -0.0175, 0.0087, 100.5],
            [0.0175, 0.9998, -0.0052, 200.3],
            [-0.0087, 0.0052, 0.9999, 50.1],
            [0, 0, 0, 1]
        ],
        "scale": 1.0,
        "rotation_deg": {"x": 0.5, "y": -1.0, "z": 0.3},
        "translation_m": {"x": 100.5, "y": 200.3, "z": 50.1}
    }
    */
    alignment_error_m FLOAT,  -- RMS alignment error in meters
    alignment_method VARCHAR(100),  -- 'ICP', 'Feature-based', 'Manual', 'GPS-assisted'
    alignment_iterations INTEGER,

    -- Fused data products
    fused_point_cloud_path TEXT,  -- S3: 3D points with spectral attributes (XYZ + RGB + Spectral bands)
    material_mapped_mesh_path TEXT,  -- S3: 3D mesh with material quality overlay
    quality_heatmap_path TEXT,  -- S3: 2D/3D heatmap visualization

    -- Quality assessment
    alignment_quality VARCHAR(50),  -- 'Excellent', 'Good', 'Fair', 'Poor'
    coverage_percentage FLOAT CHECK (coverage_percentage >= 0 AND coverage_percentage <= 100),  -- % of LiDAR points with hyperspectral data
    overlap_area_m2 FLOAT,

    -- Processing metadata
    fusion_method VARCHAR(100),  -- 'Nearest Neighbor', 'Kriging Interpolation', 'Inverse Distance Weighted'
    processing_duration_seconds INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SECTION 2: LIDAR PROGRESS MONITORING
-- ============================================================================

-- LiDAR scans for progress monitoring
CREATE TABLE IF NOT EXISTS progress_lidar_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    site_location VARCHAR(255) NOT NULL,
    scan_date TIMESTAMPTZ NOT NULL,

    -- Scanner specifications
    scanner_model VARCHAR(100) NOT NULL,  -- 'Leica RTC360', 'Faro Focus S350', 'Trimble TX8'
    scanner_serial_number VARCHAR(100),
    point_count BIGINT,  -- Number of 3D points (can be billions)
    point_density FLOAT,  -- Points per m²
    scan_quality VARCHAR(50),  -- 'High', 'Medium', 'Low'
    scan_duration_minutes INTEGER,
    scan_resolution_mm FLOAT,  -- Point spacing at 10m distance

    -- File storage
    raw_file_path TEXT NOT NULL,  -- S3: lidar-scans/raw/YYYY/MM/DD/ (LAZ/LAS/E57)
    processed_file_path TEXT,  -- S3: lidar-scans/processed/ (cleaned, aligned)
    potree_octree_path TEXT,  -- S3: lidar-scans/potree/ (web visualization format)
    thumbnail_path TEXT,  -- S3: lidar-scans/thumbnails/ (preview image)
    file_size_bytes BIGINT,
    file_format VARCHAR(20),  -- 'LAZ', 'LAS', 'E57', 'PTS'
    las_version VARCHAR(10),  -- '1.2', '1.4'
    point_data_format INTEGER,  -- LAS format code (0-10)

    -- Geospatial
    location GEOGRAPHY(POINT, 4326),  -- Scanner location
    coverage_area GEOGRAPHY(POLYGON, 4326),  -- Scan footprint
    coordinate_system VARCHAR(100) DEFAULT 'EPSG:27700',  -- British National Grid
    bounds_min_x FLOAT,
    bounds_min_y FLOAT,
    bounds_min_z FLOAT,
    bounds_max_x FLOAT,
    bounds_max_y FLOAT,
    bounds_max_z FLOAT,

    -- Classification statistics (LAS classification codes)
    ground_points BIGINT,  -- Class 2
    vegetation_points BIGINT,  -- Class 3-5
    building_points BIGINT,  -- Class 6
    unclassified_points BIGINT,  -- Class 1

    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_started_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ,
    processing_duration_seconds INTEGER,
    error_message TEXT,

    -- Metadata
    operator_name VARCHAR(255),
    weather_conditions VARCHAR(100),
    notes TEXT,
    tags TEXT[],

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SECTION 3: BIM INTEGRATION
-- ============================================================================

-- BIM model uploads
CREATE TABLE IF NOT EXISTS bim_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    model_description TEXT,
    upload_date TIMESTAMPTZ DEFAULT NOW(),

    -- File metadata
    file_path TEXT NOT NULL,  -- S3: bim-models/ifc/
    file_format VARCHAR(20) NOT NULL,  -- 'IFC', 'Revit', 'glTF', 'OBJ'
    file_size_bytes BIGINT,

    -- IFC-specific metadata
    ifc_schema VARCHAR(50),  -- 'IFC4', 'IFC2x3', 'IFC4.3'
    ifc_file_type VARCHAR(50),  -- 'IFC', 'IFCXML', 'IFCZIP'
    element_count INTEGER,
    discipline VARCHAR(100),  -- 'Architectural', 'Structural', 'MEP', 'Civil'
    lod_level VARCHAR(10),  -- 'LOD 100', 'LOD 200', 'LOD 300', 'LOD 400', 'LOD 500'

    -- Spatial bounds (project coordinate system)
    bounds_min_x FLOAT,
    bounds_min_y FLOAT,
    bounds_min_z FLOAT,
    bounds_max_x FLOAT,
    bounds_max_y FLOAT,
    bounds_max_z FLOAT,
    coordinate_system VARCHAR(100),

    -- Model metadata
    software_used VARCHAR(100),  -- 'Revit 2024', 'ArchiCAD 27'
    author VARCHAR(255),
    organization VARCHAR(255),
    project_phase VARCHAR(100),  -- 'Design', 'Construction', 'As-Built'

    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    processed_at TIMESTAMPTZ,
    validation_status VARCHAR(50),  -- 'Valid', 'Invalid', 'Warnings'
    validation_errors JSONB,

    -- Metadata
    is_baseline BOOLEAN DEFAULT FALSE,  -- Is this the baseline/reference model?
    is_active BOOLEAN DEFAULT TRUE,
    tags TEXT[],

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SECTION 4: BIM-LIDAR ALIGNMENT AND DEVIATION ANALYSIS
-- ============================================================================

-- BIM-LiDAR alignment results
CREATE TABLE IF NOT EXISTS bim_lidar_alignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bim_model_id UUID REFERENCES bim_models(id) ON DELETE CASCADE,
    lidar_scan_id UUID REFERENCES progress_lidar_scans(id) ON DELETE CASCADE,

    -- Alignment transformation (4x4 homogeneous transformation matrix)
    transformation_matrix JSONB NOT NULL,  -- Full 4x4 matrix
    rotation JSONB,  -- Quaternion or Euler angles
    translation JSONB,  -- X, Y, Z offset in meters
    scale FLOAT DEFAULT 1.0,

    -- Alignment quality metrics
    alignment_method VARCHAR(100) NOT NULL,  -- 'ICP', 'Feature-based', 'Manual', 'GPS-assisted'
    alignment_error_m FLOAT,  -- RMS error in meters
    max_error_m FLOAT,
    iterations_required INTEGER,
    convergence_achieved BOOLEAN,
    convergence_threshold_m FLOAT,

    -- Point correspondences
    corresponding_points INTEGER,  -- Number of point pairs used
    inlier_percentage FLOAT,  -- % of points within tolerance
    outlier_points INTEGER,

    -- Alignment metadata
    aligned_by VARCHAR(255),  -- User or 'System'
    alignment_confidence FLOAT CHECK (alignment_confidence >= 0 AND alignment_confidence <= 100),  -- 0-100%
    alignment_duration_seconds INTEGER,
    manual_adjustments_applied BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Deviation analysis results (element-level)
CREATE TABLE IF NOT EXISTS progress_deviation_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alignment_id UUID REFERENCES bim_lidar_alignments(id) ON DELETE CASCADE,

    -- BIM element identification
    bim_element_id VARCHAR(255) NOT NULL,  -- IFC element GUID
    element_type VARCHAR(100),  -- 'IfcWall', 'IfcBeam', 'IfcColumn', 'IfcSlab'
    element_name VARCHAR(255),
    element_description TEXT,
    storey VARCHAR(100),  -- Building storey/level

    -- Deviation metrics (statistical analysis)
    mean_deviation_mm FLOAT,  -- Average deviation
    max_deviation_mm FLOAT,  -- Maximum deviation
    min_deviation_mm FLOAT,  -- Minimum deviation
    std_deviation_mm FLOAT,  -- Standard deviation
    rms_deviation_mm FLOAT,  -- Root Mean Square deviation

    -- Volume analysis
    designed_volume_m3 FLOAT,
    as_built_volume_m3 FLOAT,
    volume_difference_m3 FLOAT,
    volume_variance_percent FLOAT,

    -- Area analysis
    designed_area_m2 FLOAT,
    as_built_area_m2 FLOAT,
    area_difference_m2 FLOAT,

    -- Severity classification
    severity VARCHAR(50) NOT NULL,  -- 'None', 'Minor', 'Moderate', 'Major', 'Critical'
    within_tolerance BOOLEAN,
    tolerance_threshold_mm FLOAT DEFAULT 10.0,  -- Project-specific tolerance
    requires_action BOOLEAN DEFAULT FALSE,
    action_priority VARCHAR(20),  -- 'Low', 'Medium', 'High', 'Urgent'

    -- Spatial location
    location GEOGRAPHY(POINT, 4326),  -- Center point of element
    deviation_geometry GEOGRAPHY(POLYGON, 4326),  -- Polygon showing deviation area

    -- Color coding for visualization
    color_code VARCHAR(7),  -- Hex color: '#00FF00' (green), '#FFFF00' (yellow), '#FF0000' (red)
    visualization_mesh_path TEXT,  -- S3: Path to colored mesh for this element

    -- Analysis metadata
    analysis_method VARCHAR(100),  -- 'Voxel Comparison', 'Surface Distance', 'Point-to-Mesh'
    point_count_used INTEGER,  -- Number of LiDAR points used in analysis
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 100),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SECTION 5: PROGRESS TRACKING AND REPORTING
-- ============================================================================

-- Progress snapshots (time-series tracking)
CREATE TABLE IF NOT EXISTS progress_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    snapshot_date TIMESTAMPTZ NOT NULL,
    snapshot_name VARCHAR(255),  -- 'Week 23', 'Milestone 5', 'Monthly - January 2025'

    -- Data sources
    lidar_scan_id UUID REFERENCES progress_lidar_scans(id),
    bim_model_id UUID REFERENCES bim_models(id),
    hyperspectral_scan_id UUID REFERENCES hyperspectral_scans(id),

    -- Progress metrics (volume-based)
    percent_complete FLOAT CHECK (percent_complete >= 0 AND percent_complete <= 100),  -- 0-100%
    completed_volume_m3 FLOAT,
    planned_volume_m3 FLOAT,
    variance_volume_m3 FLOAT,
    variance_percent FLOAT,

    -- Progress metrics (count-based)
    total_elements INTEGER,
    completed_elements INTEGER,
    in_progress_elements INTEGER,
    not_started_elements INTEGER,

    -- Schedule metrics
    planned_completion_date DATE,
    predicted_completion_date DATE,
    schedule_variance_days INTEGER,  -- Negative = ahead, Positive = behind
    percent_time_elapsed FLOAT,  -- % of project duration elapsed

    -- Quality metrics (from hyperspectral analysis)
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 100),  -- 0-100 aggregate score
    defects_detected INTEGER DEFAULT 0,
    critical_issues INTEGER DEFAULT 0,
    major_issues INTEGER DEFAULT 0,
    minor_issues INTEGER DEFAULT 0,

    -- Deviation metrics (from BIM-LiDAR comparison)
    elements_within_tolerance INTEGER,
    elements_minor_deviation INTEGER,
    elements_major_deviation INTEGER,
    average_deviation_mm FLOAT,

    -- Cost metrics (if available)
    planned_cost DECIMAL(15, 2),
    actual_cost DECIMAL(15, 2),
    cost_variance DECIMAL(15, 2),

    -- Weather data (for correlation analysis)
    weather_summary JSONB,  -- {avg_temp, total_rain_mm, wind_speed, etc}

    -- Metadata
    created_by VARCHAR(255),
    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Automated progress assurance reports
CREATE TABLE IF NOT EXISTS progress_assurance_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    snapshot_id UUID REFERENCES progress_snapshots(id) ON DELETE CASCADE,
    report_date TIMESTAMPTZ DEFAULT NOW(),

    -- Report metadata
    report_type VARCHAR(100) NOT NULL,  -- 'Weekly', 'Monthly', 'Milestone', 'On-Demand', 'Executive'
    report_title VARCHAR(255) NOT NULL,
    report_subtitle VARCHAR(255),
    generated_by VARCHAR(100) DEFAULT 'System',  -- 'System' or user name
    generation_duration_seconds INTEGER,

    -- Report content (text summaries)
    executive_summary TEXT,
    progress_analysis TEXT,
    material_quality_summary TEXT,
    deviation_summary TEXT,
    risk_assessment TEXT,
    recommendations TEXT,
    next_steps TEXT,

    -- Structured report data
    key_metrics JSONB,  -- {percent_complete, schedule_variance, quality_score, etc}
    progress_charts JSONB,  -- Chart data/config for visualization
    deviation_statistics JSONB,
    quality_statistics JSONB,
    material_test_results JSONB,

    -- Visualizations (S3 paths)
    color_coded_3d_model_path TEXT,  -- S3: 3D model with color-coded deviations
    material_quality_heatmap_path TEXT,  -- S3: Heatmap showing material quality
    progress_timeline_chart_path TEXT,  -- S3: Timeline/Gantt chart
    deviation_charts_path TEXT[],  -- S3: Array of chart image paths

    -- File outputs
    pdf_report_path TEXT,  -- S3: reports/progress/PROJECT_ID/YYYY/MM/report_id.pdf
    excel_data_path TEXT,  -- S3: reports/progress/PROJECT_ID/YYYY/MM/report_id.xlsx
    html_report_path TEXT,  -- S3: Interactive HTML report

    -- Distribution
    recipients TEXT[],  -- Array of email addresses
    sent_at TIMESTAMPTZ,
    delivery_status VARCHAR(50),  -- 'Draft', 'Sent', 'Failed'

    -- Approval workflow
    requires_approval BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(255),
    approved_at TIMESTAMPTZ,
    approval_status VARCHAR(50),  -- 'Pending', 'Approved', 'Rejected'

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Hyperspectral indexes
CREATE INDEX idx_hyperspectral_scans_project ON hyperspectral_scans(project_id);
CREATE INDEX idx_hyperspectral_scans_date ON hyperspectral_scans(scan_date DESC);
CREATE INDEX idx_hyperspectral_scans_status ON hyperspectral_scans(processing_status);
CREATE INDEX idx_hyperspectral_scans_location ON hyperspectral_scans USING GIST(location);
CREATE INDEX idx_hyperspectral_scans_coverage ON hyperspectral_scans USING GIST(coverage_area);

CREATE INDEX idx_material_quality_scan ON material_quality_assessments(scan_id);
CREATE INDEX idx_material_quality_type ON material_quality_assessments(material_type);
CREATE INDEX idx_material_quality_score ON material_quality_assessments(quality_score DESC);
CREATE INDEX idx_material_quality_grade ON material_quality_assessments(quality_grade);
CREATE INDEX idx_material_quality_defects ON material_quality_assessments(defect_count DESC);
CREATE INDEX idx_material_quality_location ON material_quality_assessments USING GIST(location_in_scan);

CREATE INDEX idx_spectral_library_category ON spectral_library(material_category);
CREATE INDEX idx_spectral_library_active ON spectral_library(is_active) WHERE is_active = TRUE;

CREATE INDEX idx_hyper_lidar_fusion_hyper ON hyperspectral_lidar_fusion(hyperspectral_scan_id);
CREATE INDEX idx_hyper_lidar_fusion_lidar ON hyperspectral_lidar_fusion(lidar_scan_id);

-- LiDAR indexes
CREATE INDEX idx_progress_lidar_project ON progress_lidar_scans(project_id);
CREATE INDEX idx_progress_lidar_date ON progress_lidar_scans(scan_date DESC);
CREATE INDEX idx_progress_lidar_status ON progress_lidar_scans(processing_status);
CREATE INDEX idx_progress_lidar_location ON progress_lidar_scans USING GIST(location);
CREATE INDEX idx_progress_lidar_coverage ON progress_lidar_scans USING GIST(coverage_area);

-- BIM indexes
CREATE INDEX idx_bim_models_project ON bim_models(project_id);
CREATE INDEX idx_bim_models_active ON bim_models(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_bim_models_baseline ON bim_models(is_baseline) WHERE is_baseline = TRUE;

-- Alignment and deviation indexes
CREATE INDEX idx_bim_lidar_align_bim ON bim_lidar_alignments(bim_model_id);
CREATE INDEX idx_bim_lidar_align_lidar ON bim_lidar_alignments(lidar_scan_id);

CREATE INDEX idx_deviation_alignment ON progress_deviation_analysis(alignment_id);
CREATE INDEX idx_deviation_severity ON progress_deviation_analysis(severity);
CREATE INDEX idx_deviation_tolerance ON progress_deviation_analysis(within_tolerance);
CREATE INDEX idx_deviation_location ON progress_deviation_analysis USING GIST(location);
CREATE INDEX idx_deviation_element ON progress_deviation_analysis(bim_element_id);

-- Progress tracking indexes
CREATE INDEX idx_snapshots_project ON progress_snapshots(project_id);
CREATE INDEX idx_snapshots_date ON progress_snapshots(snapshot_date DESC);
CREATE INDEX idx_snapshots_lidar ON progress_snapshots(lidar_scan_id);
CREATE INDEX idx_snapshots_bim ON progress_snapshots(bim_model_id);
CREATE INDEX idx_snapshots_hyper ON progress_snapshots(hyperspectral_scan_id);

CREATE INDEX idx_reports_project ON progress_assurance_reports(project_id);
CREATE INDEX idx_reports_snapshot ON progress_assurance_reports(snapshot_id);
CREATE INDEX idx_reports_date ON progress_assurance_reports(report_date DESC);
CREATE INDEX idx_reports_type ON progress_assurance_reports(report_type);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER update_hyperspectral_scans_updated_at
    BEFORE UPDATE ON hyperspectral_scans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spectral_library_updated_at
    BEFORE UPDATE ON spectral_library
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_progress_lidar_scans_updated_at
    BEFORE UPDATE ON progress_lidar_scans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bim_models_updated_at
    BEFORE UPDATE ON bim_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE hyperspectral_scans IS 'Hyperspectral imaging scans (Specim IQ: 204 spectral bands, 400-1000nm) for material quality verification';
COMMENT ON TABLE material_quality_assessments IS 'AI-powered material quality predictions from hyperspectral analysis (concrete strength, defects, etc)';
COMMENT ON TABLE spectral_library IS 'Reference spectral signatures for known materials (validated with lab tests)';
COMMENT ON TABLE hyperspectral_lidar_fusion IS 'Fused hyperspectral + LiDAR data for 3D material quality mapping';
COMMENT ON TABLE progress_lidar_scans IS 'LiDAR point cloud scans for construction progress monitoring';
COMMENT ON TABLE bim_models IS 'BIM/IFC models (design intent) for comparison with reality';
COMMENT ON TABLE bim_lidar_alignments IS 'ICP-based alignment results between BIM models and LiDAR reality captures';
COMMENT ON TABLE progress_deviation_analysis IS 'Element-level deviation analysis (designed vs built)';
COMMENT ON TABLE progress_snapshots IS 'Time-series progress snapshots for historical tracking and trend analysis';
COMMENT ON TABLE progress_assurance_reports IS 'Automated progress assurance reports (one-click PDF generation in <10 minutes)';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
