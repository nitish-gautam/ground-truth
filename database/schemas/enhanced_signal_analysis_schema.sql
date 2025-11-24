-- ================================================================
-- ENHANCED GPR SIGNAL ANALYSIS SCHEMA
-- Underground Utility Detection Platform - Signal Processing Extension
-- ================================================================

-- Enhanced time-series signal analysis table
CREATE TABLE gpr_signal_timeseries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Time-series metadata
    trace_number INTEGER NOT NULL,
    total_traces INTEGER NOT NULL,
    sample_rate_mhz DECIMAL(8,3) NOT NULL,
    time_window_ns DECIMAL(8,3) NOT NULL,

    -- Signal envelope analysis
    signal_envelope vector(512), -- Hilbert transform envelope
    instantaneous_phase vector(512), -- Phase information
    instantaneous_frequency vector(512), -- Frequency variation over time

    -- Advanced signal characteristics
    signal_energy_distribution vector(64), -- Energy distribution across depth
    reflection_coefficients vector(128), -- Calculated reflection coefficients
    attenuation_profile vector(128), -- Signal attenuation with depth

    -- Velocity analysis
    velocity_model JSONB, -- Depth-velocity relationships
    two_way_travel_time_ns DECIMAL(8,3),
    estimated_dielectric_constant DECIMAL(6,3),

    -- Quality metrics
    signal_to_clutter_ratio DECIMAL(6,3),
    coherence_coefficient DECIMAL(5,4),
    signal_stability_index DECIMAL(5,4),

    -- Processing metadata
    processing_algorithm VARCHAR(100),
    processing_parameters JSONB,
    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(survey_id, trace_number)
);

-- Frequency domain analysis table
CREATE TABLE gpr_frequency_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_timeseries_id UUID NOT NULL REFERENCES gpr_signal_timeseries(id) ON DELETE CASCADE,

    -- FFT analysis results
    frequency_spectrum vector(256), -- Power spectral density
    phase_spectrum vector(256), -- Phase spectrum
    magnitude_spectrum vector(256), -- Magnitude spectrum

    -- Spectral features
    peak_frequency_mhz DECIMAL(6,2),
    bandwidth_3db_mhz DECIMAL(6,2),
    bandwidth_6db_mhz DECIMAL(6,2),
    spectral_centroid_mhz DECIMAL(6,2),
    spectral_spread DECIMAL(6,3),
    spectral_skewness DECIMAL(6,3),
    spectral_kurtosis DECIMAL(6,3),
    spectral_rolloff_85_mhz DECIMAL(6,2),
    spectral_rolloff_95_mhz DECIMAL(6,2),

    -- Wavelet analysis
    wavelet_coefficients vector(512), -- Continuous wavelet transform
    wavelet_energy_distribution vector(64), -- Energy across scales
    dominant_scale_factor DECIMAL(6,3),

    -- Spectral analysis metadata
    window_function VARCHAR(50) DEFAULT 'hamming',
    fft_size INTEGER DEFAULT 512,
    overlap_percentage DECIMAL(5,2) DEFAULT 50.0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Hyperbola detection and analysis
CREATE TABLE gpr_hyperbola_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    signal_timeseries_id UUID REFERENCES gpr_signal_timeseries(id),

    -- Hyperbola geometry
    apex_trace_position INTEGER NOT NULL,
    apex_time_ns DECIMAL(8,3) NOT NULL,
    apex_depth_estimate_m DECIMAL(6,3),

    -- Hyperbola characteristics
    curvature_coefficient DECIMAL(8,6),
    asymptote_angle_deg DECIMAL(6,2),
    hyperbola_width_traces INTEGER,
    symmetry_coefficient DECIMAL(5,4),

    -- Fitting parameters
    fitting_algorithm VARCHAR(100) DEFAULT 'least_squares',
    fitting_quality_r_squared DECIMAL(5,4),
    fitting_residual_rms DECIMAL(8,6),

    -- Velocity estimation from hyperbola
    estimated_velocity_m_ns DECIMAL(6,4),
    velocity_confidence DECIMAL(5,4),
    dielectric_constant_estimate DECIMAL(6,3),

    -- Classification confidence
    hyperbola_confidence_score DECIMAL(5,4),
    false_positive_probability DECIMAL(5,4),

    -- Detection metadata
    detection_algorithm VARCHAR(100),
    detection_parameters JSONB,
    manual_verification BOOLEAN DEFAULT FALSE,
    verified_by VARCHAR(255),
    verification_date TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Signal correlation analysis between traces
CREATE TABLE gpr_trace_correlations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Trace pair identification
    trace1_number INTEGER NOT NULL,
    trace2_number INTEGER NOT NULL,
    trace_separation_m DECIMAL(6,3),

    -- Correlation metrics
    cross_correlation_coefficient DECIMAL(6,4),
    normalized_cross_correlation DECIMAL(6,4),
    time_lag_ns DECIMAL(8,3),
    coherence_function vector(128), -- Frequency-dependent coherence

    -- Phase correlation
    phase_difference vector(128),
    phase_velocity_estimate_m_ns DECIMAL(6,4),

    -- Pattern similarity
    structural_similarity_index DECIMAL(5,4),
    pattern_matching_score DECIMAL(5,4),

    -- Analysis metadata
    correlation_algorithm VARCHAR(100),
    window_length_ns DECIMAL(6,2),
    overlap_percentage DECIMAL(5,2),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(survey_id, trace1_number, trace2_number),
    CHECK (trace1_number < trace2_number) -- Ensure ordered pairs
);

-- Advanced feature extraction for ML training
CREATE TABLE gpr_advanced_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,
    signal_timeseries_id UUID REFERENCES gpr_signal_timeseries(id),

    -- Statistical moments
    signal_mean DECIMAL(10,6),
    signal_variance DECIMAL(10,6),
    signal_skewness DECIMAL(8,4),
    signal_kurtosis DECIMAL(8,4),

    -- Texture features (Gray-Level Co-occurrence Matrix)
    glcm_contrast DECIMAL(8,4),
    glcm_dissimilarity DECIMAL(8,4),
    glcm_homogeneity DECIMAL(8,4),
    glcm_energy DECIMAL(8,4),
    glcm_correlation DECIMAL(8,4),

    -- Local Binary Pattern features
    lbp_uniform_patterns vector(59), -- Uniform LBP patterns
    lbp_histogram vector(256), -- Full LBP histogram

    -- Gabor filter responses
    gabor_responses vector(48), -- 6 orientations × 8 scales
    gabor_energy vector(48),
    gabor_mean_amplitude vector(48),

    -- Wavelet packet features
    wavelet_packet_energy vector(32), -- Energy in different frequency bands
    wavelet_packet_entropy vector(32), -- Entropy measures

    -- Morphological features
    morphological_opening vector(8), -- Different structuring elements
    morphological_closing vector(8),
    morphological_gradient vector(8),

    -- Edge and gradient features
    sobel_edge_magnitude vector(128),
    sobel_edge_direction vector(128),
    canny_edge_count INTEGER,
    gradient_histogram vector(36), -- 10-degree bins

    -- Fourier descriptor features
    fourier_descriptors vector(64),

    -- Feature extraction metadata
    extraction_method VARCHAR(100),
    feature_version VARCHAR(20),
    extraction_parameters JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Signal quality assessment
CREATE TABLE gpr_signal_quality (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    survey_id UUID NOT NULL REFERENCES gpr_surveys(id) ON DELETE CASCADE,

    -- Overall quality metrics
    overall_quality_score DECIMAL(5,4), -- 0-1 scale
    signal_clarity_index DECIMAL(5,4),
    noise_level_assessment VARCHAR(20) CHECK (noise_level_assessment IN ('low', 'medium', 'high', 'severe')),

    -- Specific quality indicators
    coupling_quality DECIMAL(5,4), -- Antenna-ground coupling
    signal_penetration_depth_m DECIMAL(6,3),
    resolution_vertical_cm DECIMAL(6,2),
    resolution_horizontal_cm DECIMAL(6,2),

    -- Noise characterization
    thermal_noise_level_db DECIMAL(6,2),
    electromagnetic_interference_db DECIMAL(6,2),
    clutter_level_db DECIMAL(6,2),
    multipath_artifacts_present BOOLEAN DEFAULT FALSE,

    -- Environmental impact on quality
    weather_impact_factor DECIMAL(5,4),
    ground_moisture_impact DECIMAL(5,4),
    surface_roughness_impact DECIMAL(5,4),

    -- Data completeness
    data_completeness_percentage DECIMAL(5,2),
    missing_traces_count INTEGER DEFAULT 0,
    corrupted_samples_count INTEGER DEFAULT 0,

    -- Quality assessment metadata
    assessment_algorithm VARCHAR(100),
    assessment_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assessed_by VARCHAR(255),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- PERFORMANCE INDEXES FOR SIGNAL ANALYSIS
-- ================================================================

-- Signal timeseries indexes
CREATE INDEX idx_signal_timeseries_survey ON gpr_signal_timeseries(survey_id);
CREATE INDEX idx_signal_timeseries_trace ON gpr_signal_timeseries(trace_number);
CREATE INDEX idx_signal_timeseries_energy ON gpr_signal_timeseries USING GIN(signal_energy_distribution);

-- Frequency analysis indexes
CREATE INDEX idx_frequency_analysis_signal ON gpr_frequency_analysis(signal_timeseries_id);
CREATE INDEX idx_frequency_peak ON gpr_frequency_analysis(peak_frequency_mhz);
CREATE INDEX idx_frequency_spectrum ON gpr_frequency_analysis USING GIN(frequency_spectrum);

-- Hyperbola analysis indexes
CREATE INDEX idx_hyperbola_survey ON gpr_hyperbola_analysis(survey_id);
CREATE INDEX idx_hyperbola_position ON gpr_hyperbola_analysis(apex_trace_position, apex_time_ns);
CREATE INDEX idx_hyperbola_confidence ON gpr_hyperbola_analysis(hyperbola_confidence_score);
CREATE INDEX idx_hyperbola_depth ON gpr_hyperbola_analysis(apex_depth_estimate_m);

-- Correlation analysis indexes
CREATE INDEX idx_trace_correlations_survey ON gpr_trace_correlations(survey_id);
CREATE INDEX idx_trace_correlations_pair ON gpr_trace_correlations(trace1_number, trace2_number);
CREATE INDEX idx_correlation_coefficient ON gpr_trace_correlations(cross_correlation_coefficient);

-- Advanced features indexes
CREATE INDEX idx_advanced_features_survey ON gpr_advanced_features(survey_id);
CREATE INDEX idx_advanced_features_signal ON gpr_advanced_features(signal_timeseries_id);
CREATE INDEX idx_gabor_responses ON gpr_advanced_features USING GIN(gabor_responses);
CREATE INDEX idx_wavelet_features ON gpr_advanced_features USING GIN(wavelet_packet_energy);

-- Signal quality indexes
CREATE INDEX idx_signal_quality_survey ON gpr_signal_quality(survey_id);
CREATE INDEX idx_signal_quality_score ON gpr_signal_quality(overall_quality_score);
CREATE INDEX idx_signal_quality_noise ON gpr_signal_quality(noise_level_assessment);

-- ================================================================
-- MATERIALIZED VIEWS FOR SIGNAL ANALYSIS
-- ================================================================

-- Signal quality summary per survey
CREATE MATERIALIZED VIEW signal_quality_summary AS
SELECT
    s.id as survey_id,
    s.location_id,
    sq.overall_quality_score,
    sq.signal_clarity_index,
    sq.noise_level_assessment,
    sq.signal_penetration_depth_m,
    COUNT(DISTINCT sts.id) as processed_traces_count,
    AVG(sts.signal_to_clutter_ratio) as avg_signal_clutter_ratio,
    COUNT(DISTINCT ha.id) as detected_hyperbolas_count,
    AVG(ha.hyperbola_confidence_score) as avg_hyperbola_confidence
FROM gpr_surveys s
LEFT JOIN gpr_signal_quality sq ON s.id = sq.survey_id
LEFT JOIN gpr_signal_timeseries sts ON s.id = sts.survey_id
LEFT JOIN gpr_hyperbola_analysis ha ON s.id = ha.survey_id
GROUP BY s.id, s.location_id, sq.overall_quality_score, sq.signal_clarity_index,
         sq.noise_level_assessment, sq.signal_penetration_depth_m;

-- Hyperbola detection performance summary
CREATE MATERIALIZED VIEW hyperbola_detection_performance AS
SELECT
    s.id as survey_id,
    s.location_id,
    COUNT(ha.id) as total_hyperbolas_detected,
    AVG(ha.hyperbola_confidence_score) as avg_confidence,
    COUNT(CASE WHEN ha.manual_verification = true THEN 1 END) as manually_verified_count,
    AVG(ha.curvature_coefficient) as avg_curvature,
    AVG(ha.fitting_quality_r_squared) as avg_fitting_quality,
    AVG(ha.estimated_velocity_m_ns) as avg_estimated_velocity
FROM gpr_surveys s
LEFT JOIN gpr_hyperbola_analysis ha ON s.id = ha.survey_id
GROUP BY s.id, s.location_id;

-- ================================================================
-- FUNCTIONS FOR SIGNAL PROCESSING
-- ================================================================

-- Function to calculate signal penetration depth
CREATE OR REPLACE FUNCTION calculate_signal_penetration_depth(
    survey_uuid UUID,
    noise_threshold_db DECIMAL DEFAULT -40.0
) RETURNS DECIMAL AS $$
DECLARE
    max_depth DECIMAL;
BEGIN
    SELECT MAX(depth_estimate_m) INTO max_depth
    FROM gpr_signal_data gsd
    JOIN gpr_signal_timeseries gst ON gsd.survey_id = gst.survey_id
    WHERE gsd.survey_id = survey_uuid
    AND gsd.signal_strength_db > noise_threshold_db
    AND gst.signal_to_clutter_ratio > 1.0;

    RETURN COALESCE(max_depth, 0.0);
END;
$$ LANGUAGE plpgsql;

-- Function to refresh signal analysis materialized views
CREATE OR REPLACE FUNCTION refresh_signal_analysis_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW signal_quality_summary;
    REFRESH MATERIALIZED VIEW hyperbola_detection_performance;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE gpr_signal_timeseries IS 'Enhanced time-series analysis of GPR signals with advanced signal processing features';
COMMENT ON TABLE gpr_frequency_analysis IS 'Frequency domain analysis including FFT, wavelets, and spectral features for ML training';
COMMENT ON TABLE gpr_hyperbola_analysis IS 'Comprehensive hyperbola detection and analysis for utility object identification';
COMMENT ON TABLE gpr_trace_correlations IS 'Cross-correlation analysis between GPR traces for pattern recognition';
COMMENT ON TABLE gpr_advanced_features IS 'Advanced feature extraction for machine learning model training and validation';
COMMENT ON TABLE gpr_signal_quality IS 'Signal quality assessment and environmental impact analysis';

COMMENT ON MATERIALIZED VIEW signal_quality_summary IS 'Summary of signal quality metrics per survey for performance monitoring';
COMMENT ON MATERIALIZED VIEW hyperbola_detection_performance IS 'Hyperbola detection performance metrics for algorithm validation';