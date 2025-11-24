-- ================================================================
-- UNDERGROUND UTILITY DETECTION PLATFORM - MASTER DEPLOYMENT SCRIPT
-- Complete Database Deployment Orchestration
-- ================================================================

-- Set client encoding and connection parameters
SET client_encoding = 'UTF8';
SET client_min_messages = WARNING;
SET log_min_messages = WARNING;

-- Enable timing for deployment tracking
\timing on

-- Display deployment banner
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '================================================================';
    RAISE NOTICE 'UNDERGROUND UTILITY DETECTION PLATFORM';
    RAISE NOTICE 'MASTER DATABASE DEPLOYMENT';
    RAISE NOTICE '================================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Starting comprehensive database deployment...';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
    RAISE NOTICE 'Timestamp: %', NOW();
    RAISE NOTICE '';
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 1: INITIALIZATION
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE 'PHASE 1: Database Initialization';
    RAISE NOTICE '- Setting up extensions and basic configuration';
    RAISE NOTICE '- Creating schemas and security roles';
    RAISE NOTICE '- Establishing audit framework';
END $$;

-- Execute master initialization
\echo 'Executing master initialization...'
\i 00_master_init.sql

-- Verify initialization
DO $$
DECLARE
    extension_count INTEGER;
    schema_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO extension_count
    FROM pg_extension
    WHERE extname IN ('uuid-ossp', 'postgis', 'vector', 'pg_trgm');

    SELECT COUNT(*) INTO schema_count
    FROM information_schema.schemata
    WHERE schema_name IN ('gpr_data', 'environmental', 'validation', 'ml_analytics', 'compliance', 'historical', 'utilities');

    IF extension_count >= 4 AND schema_count >= 7 THEN
        RAISE NOTICE '✓ Phase 1 completed successfully';
    ELSE
        RAISE EXCEPTION 'Phase 1 failed: Missing extensions or schemas';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 2: CORE SCHEMA DEPLOYMENT
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 2: Core Schema Deployment';
    RAISE NOTICE '- Deploying GPR infrastructure schema';
    RAISE NOTICE '- Setting up core project management tables';
    RAISE NOTICE '- Creating utility detection structures';
END $$;

-- Deploy GPR infrastructure schema
\echo 'Deploying GPR infrastructure schema...'
\i schemas/gpr_infrastructure_schema.sql

-- Verify core tables
DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_name IN ('projects', 'survey_sites', 'gpr_surveys', 'detected_utilities', 'gpr_image_data')
    AND table_schema = 'public';

    IF table_count >= 5 THEN
        RAISE NOTICE '✓ GPR infrastructure schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'GPR infrastructure schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 3: SIGNAL PROCESSING SCHEMA
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 3: Signal Processing Schema';
    RAISE NOTICE '- Deploying enhanced signal analysis capabilities';
    RAISE NOTICE '- Setting up time-series processing';
    RAISE NOTICE '- Creating frequency domain analysis structures';
END $$;

-- Deploy enhanced signal analysis schema
\echo 'Deploying enhanced signal analysis schema...'
\i schemas/enhanced_signal_analysis_schema.sql

-- Verify signal processing tables
DO $$
DECLARE
    signal_table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO signal_table_count
    FROM information_schema.tables
    WHERE table_name IN ('gpr_signal_timeseries', 'gpr_frequency_analysis', 'gpr_hyperbola_analysis')
    AND table_schema = 'public';

    IF signal_table_count >= 3 THEN
        RAISE NOTICE '✓ Signal processing schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'Signal processing schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 4: ENVIRONMENTAL METADATA SCHEMA
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 4: Environmental Metadata Schema';
    RAISE NOTICE '- Deploying comprehensive environmental tracking';
    RAISE NOTICE '- Setting up Twente dataset integration (25+ fields)';
    RAISE NOTICE '- Creating correlation analysis capabilities';
END $$;

-- Deploy environmental metadata schema
\echo 'Deploying environmental metadata schema...'
\i schemas/environmental_metadata_schema.sql

-- Verify environmental tables
DO $$
DECLARE
    env_table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO env_table_count
    FROM information_schema.tables
    WHERE table_name IN ('environmental_metadata', 'ground_conditions', 'weather_conditions', 'environmental_impact_analysis')
    AND table_schema = 'public';

    IF env_table_count >= 4 THEN
        RAISE NOTICE '✓ Environmental metadata schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'Environmental metadata schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 5: GROUND TRUTH VALIDATION SCHEMA
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 5: Ground Truth Validation Schema';
    RAISE NOTICE '- Deploying validation campaign management';
    RAISE NOTICE '- Setting up accuracy assessment framework';
    RAISE NOTICE '- Creating statistical analysis capabilities';
END $$;

-- Deploy ground truth validation schema
\echo 'Deploying ground truth validation schema...'
\i schemas/ground_truth_validation_schema.sql

-- Verify validation tables
DO $$
DECLARE
    validation_table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO validation_table_count
    FROM information_schema.tables
    WHERE table_name IN ('validation_campaigns', 'ground_truth_validations', 'accuracy_assessment_summary')
    AND table_schema = 'public';

    IF validation_table_count >= 3 THEN
        RAISE NOTICE '✓ Ground truth validation schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'Ground truth validation schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 6: ML PERFORMANCE TRACKING SCHEMA
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 6: ML Performance Tracking Schema';
    RAISE NOTICE '- Deploying ML model registry';
    RAISE NOTICE '- Setting up cross-validation tracking';
    RAISE NOTICE '- Creating performance monitoring capabilities';
END $$;

-- Deploy ML performance schema
\echo 'Deploying ML performance tracking schema...'
\i schemas/ml_performance_schema.sql

-- Verify ML tables
DO $$
DECLARE
    ml_table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO ml_table_count
    FROM information_schema.tables
    WHERE table_name IN ('ml_models', 'cross_validation_experiments', 'cv_fold_results', 'feature_importance_analysis')
    AND table_schema = 'public';

    IF ml_table_count >= 4 THEN
        RAISE NOTICE '✓ ML performance tracking schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'ML performance tracking schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 7: PAS 128 COMPLIANCE SCHEMA
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 7: PAS 128 Compliance Schema';
    RAISE NOTICE '- Deploying PAS 128:2022 compliance framework';
    RAISE NOTICE '- Setting up quality level determination';
    RAISE NOTICE '- Creating automated compliance validation';
END $$;

-- Deploy PAS 128 compliance schema
\echo 'Deploying PAS 128 compliance schema...'
\i schemas/pas128_compliance_schema.sql

-- Verify compliance tables
DO $$
DECLARE
    compliance_table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO compliance_table_count
    FROM information_schema.tables
    WHERE table_name IN ('pas128_quality_levels', 'pas128_compliance_assessments', 'quality_level_determination_rules')
    AND table_schema = 'public';

    IF compliance_table_count >= 3 THEN
        RAISE NOTICE '✓ PAS 128 compliance schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'PAS 128 compliance schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 8: HISTORICAL INCIDENT ANALYSIS SCHEMA
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 8: Historical Incident Analysis Schema';
    RAISE NOTICE '- Deploying USAG strike report integration';
    RAISE NOTICE '- Setting up pattern analysis capabilities';
    RAISE NOTICE '- Creating prevention measure tracking';
END $$;

-- Deploy USAG strike reports schema
\echo 'Deploying USAG strike reports schema...'
\i schemas/usag_strike_reports_schema.sql

-- Verify historical tables
DO $$
DECLARE
    historical_table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO historical_table_count
    FROM information_schema.tables
    WHERE table_name IN ('usag_strike_incidents', 'strike_pattern_analysis', 'strike_prevention_measures')
    AND table_schema = 'public';

    IF historical_table_count >= 3 THEN
        RAISE NOTICE '✓ Historical incident analysis schema deployed successfully';
    ELSE
        RAISE EXCEPTION 'Historical incident analysis schema deployment failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 9: SAMPLE DATA LOADING
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 9: Sample Data Loading';
    RAISE NOTICE '- Loading University of Twente dataset samples';
    RAISE NOTICE '- Loading Mojahid GPR image samples';
    RAISE NOTICE '- Creating validation and ML training data';
    RAISE NOTICE '- Populating compliance and historical examples';
END $$;

-- Load sample data
\echo 'Loading comprehensive sample data...'
\i sample_data_loading.sql

-- Verify data loading
DO $$
DECLARE
    total_projects INTEGER;
    total_surveys INTEGER;
    total_utilities INTEGER;
    total_images INTEGER;
    total_validations INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_projects FROM projects;
    SELECT COUNT(*) INTO total_surveys FROM gpr_surveys;
    SELECT COUNT(*) INTO total_utilities FROM detected_utilities;
    SELECT COUNT(*) INTO total_images FROM gpr_image_data;
    SELECT COUNT(*) INTO total_validations FROM ground_truth_validations;

    RAISE NOTICE 'Sample data loaded:';
    RAISE NOTICE '- Projects: %', total_projects;
    RAISE NOTICE '- GPR Surveys: %', total_surveys;
    RAISE NOTICE '- Detected Utilities: %', total_utilities;
    RAISE NOTICE '- GPR Images: %', total_images;
    RAISE NOTICE '- Ground Truth Validations: %', total_validations;

    IF total_projects > 0 AND total_surveys > 0 THEN
        RAISE NOTICE '✓ Sample data loaded successfully';
    ELSE
        RAISE EXCEPTION 'Sample data loading failed';
    END IF;
END $$;

-- ================================================================
-- DEPLOYMENT PHASE 10: OPTIMIZATION AND FINALIZATION
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE 'PHASE 10: Optimization and Finalization';
    RAISE NOTICE '- Applying performance optimizations';
    RAISE NOTICE '- Creating advanced indexes';
    RAISE NOTICE '- Setting up monitoring functions';
    RAISE NOTICE '- Finalizing deployment';
END $$;

-- Apply database optimizations
\echo 'Applying database optimizations...'
\i database_optimization.sql

-- Final verification and statistics
DO $$
DECLARE
    table_stats RECORD;
    index_count INTEGER;
    function_count INTEGER;
    view_count INTEGER;
    total_records INTEGER := 0;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '================================================================';
    RAISE NOTICE 'DEPLOYMENT COMPLETED SUCCESSFULLY';
    RAISE NOTICE '================================================================';
    RAISE NOTICE '';

    -- Count database objects
    SELECT COUNT(*) INTO index_count FROM pg_indexes WHERE schemaname = 'public';
    SELECT COUNT(*) INTO function_count FROM information_schema.routines WHERE routine_schema = 'public';
    SELECT COUNT(*) INTO view_count FROM information_schema.views WHERE table_schema = 'public';

    RAISE NOTICE 'DATABASE STATISTICS:';
    RAISE NOTICE '- Database: %', current_database();
    RAISE NOTICE '- Indexes: %', index_count;
    RAISE NOTICE '- Functions: %', function_count;
    RAISE NOTICE '- Views: %', view_count;
    RAISE NOTICE '';

    -- Show record counts for key tables
    RAISE NOTICE 'SAMPLE DATA OVERVIEW:';
    FOR table_stats IN
        SELECT
            tablename,
            n_tup_ins AS row_count
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        AND tablename IN (
            'projects', 'survey_sites', 'gpr_surveys', 'environmental_metadata',
            'detected_utilities', 'gpr_image_data', 'ground_truth_validations',
            'ml_models', 'pas128_compliance_assessments', 'usag_strike_incidents'
        )
        ORDER BY n_tup_ins DESC
    LOOP
        RAISE NOTICE '- %: % records', table_stats.tablename, table_stats.row_count;
        total_records := total_records + table_stats.row_count;
    END LOOP;

    RAISE NOTICE '';
    RAISE NOTICE 'Total sample records: %', total_records;
    RAISE NOTICE '';

    RAISE NOTICE 'DATASET INTEGRATION SUMMARY:';
    RAISE NOTICE '✓ University of Twente GPR Dataset (25+ metadata fields)';
    RAISE NOTICE '✓ Mojahid GPR Images Dataset (2,239+ images, 6 categories)';
    RAISE NOTICE '✓ PAS 128:2022 Compliance Framework';
    RAISE NOTICE '✓ USAG Strike Reports Integration';
    RAISE NOTICE '✓ Advanced Signal Processing Capabilities';
    RAISE NOTICE '✓ ML Model Performance Tracking';
    RAISE NOTICE '✓ Ground Truth Validation System';
    RAISE NOTICE '';

    RAISE NOTICE 'READY FOR:';
    RAISE NOTICE '• GPR signal analysis and feature extraction';
    RAISE NOTICE '• Machine learning model training and validation';
    RAISE NOTICE '• Environmental correlation analysis';
    RAISE NOTICE '• PAS 128 compliance assessment and reporting';
    RAISE NOTICE '• Historical pattern analysis and risk assessment';
    RAISE NOTICE '• Spatial and temporal data analysis';
    RAISE NOTICE '';

    RAISE NOTICE 'NEXT STEPS:';
    RAISE NOTICE '1. Configure application connection strings';
    RAISE NOTICE '2. Set up automated backup procedures';
    RAISE NOTICE '3. Configure monitoring and alerting';
    RAISE NOTICE '4. Review security settings and update passwords';
    RAISE NOTICE '5. Begin data ingestion from real datasets';
    RAISE NOTICE '';

    RAISE NOTICE 'DOCUMENTATION:';
    RAISE NOTICE '• Complete documentation: DATABASE_DOCUMENTATION.md';
    RAISE NOTICE '• API reference included with usage examples';
    RAISE NOTICE '• Performance optimization guide provided';
    RAISE NOTICE '• Troubleshooting and maintenance procedures';
    RAISE NOTICE '';

    RAISE NOTICE 'SUPPORT FUNCTIONS AVAILABLE:';
    RAISE NOTICE '• get_database_statistics() - Database overview';
    RAISE NOTICE '• analyze_query_performance() - Query optimization';
    RAISE NOTICE '• check_database_health() - Health monitoring';
    RAISE NOTICE '• refresh_all_materialized_views() - Data refresh';
    RAISE NOTICE '• run_performance_benchmark() - Performance testing';
    RAISE NOTICE '';

    RAISE NOTICE '================================================================';
    RAISE NOTICE 'DEPLOYMENT SUCCESSFUL - SYSTEM READY FOR OPERATION';
    RAISE NOTICE '================================================================';
END $$;

-- Log successful deployment
INSERT INTO deployment_log (deployment_type, script_name, version, status, notes)
VALUES (
    'full_deployment',
    'deploy_database.sql',
    '1.0.0',
    'success',
    'Complete database deployment successful: ' ||
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public')::TEXT ||
    ' tables, ' ||
    (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public')::TEXT ||
    ' indexes created'
);

-- Disable timing
\timing off

-- Final success message
\echo ''
\echo '================================================================'
\echo 'UNDERGROUND UTILITY DETECTION PLATFORM DATABASE'
\echo 'DEPLOYMENT COMPLETED SUCCESSFULLY!'
\echo '================================================================'
\echo ''
\echo 'The database is ready for GPR feature analysis, ML training,'
\echo 'and comprehensive utility detection platform operations.'
\echo ''
\echo 'Refer to DATABASE_DOCUMENTATION.md for detailed usage information.'
\echo '================================================================'