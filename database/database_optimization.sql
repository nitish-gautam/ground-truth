-- ================================================================
-- UNDERGROUND UTILITY DETECTION PLATFORM - DATABASE OPTIMIZATION
-- Performance Optimization, Index Enhancement, and Query Tuning
-- ================================================================

-- Check PostgreSQL version and capabilities
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL Version: %', version();
    RAISE NOTICE 'Starting database optimization process...';
END $$;

-- ================================================================
-- PERFORMANCE MONITORING SETUP
-- ================================================================

-- Enable query statistics tracking
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET track_functions = all;

-- Create performance monitoring views
CREATE OR REPLACE VIEW query_performance_summary AS
SELECT
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE calls > 10
ORDER BY total_time DESC;

CREATE OR REPLACE VIEW table_access_patterns AS
SELECT
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    CASE
        WHEN seq_scan > 0 THEN seq_tup_read / seq_scan
        ELSE 0
    END AS avg_seq_tup_read
FROM pg_stat_user_tables
ORDER BY seq_tup_read DESC;

-- ================================================================
-- ADVANCED INDEXING STRATEGY
-- ================================================================

-- Spatial indexing optimization for GPR surveys
DROP INDEX IF EXISTS idx_gpr_surveys_location_optimized;
CREATE INDEX CONCURRENTLY idx_gpr_surveys_location_optimized
ON gpr_surveys USING GIST(
    ST_Transform(
        (SELECT location FROM survey_sites WHERE id = gpr_surveys.site_id),
        3857
    )
) WHERE survey_date >= CURRENT_DATE - INTERVAL '2 years';

-- Multi-column indexes for common query patterns
DROP INDEX IF EXISTS idx_gpr_surveys_composite;
CREATE INDEX CONCURRENTLY idx_gpr_surveys_composite
ON gpr_surveys (project_id, survey_date, quality_level)
INCLUDE (confidence_score, amount_of_utilities);

DROP INDEX IF EXISTS idx_detected_utilities_composite;
CREATE INDEX CONCURRENTLY idx_detected_utilities_composite
ON detected_utilities (survey_id, utility_discipline, depth_m)
INCLUDE (confidence_score, detection_method);

DROP INDEX IF EXISTS idx_environmental_metadata_composite;
CREATE INDEX CONCURRENTLY idx_environmental_metadata_composite
ON environmental_metadata (survey_id, ground_relative_permittivity, amount_of_utilities)
INCLUDE (ground_condition_id, weather_condition_id);

-- Advanced indexes for ML and analytics workloads
DROP INDEX IF EXISTS idx_gpr_signal_depth_range;
CREATE INDEX CONCURRENTLY idx_gpr_signal_depth_range
ON gpr_signal_data (survey_id, depth_estimate_m)
WHERE depth_estimate_m BETWEEN 0.5 AND 3.0; -- Most common utility depth range

DROP INDEX IF EXISTS idx_model_performance_composite;
CREATE INDEX CONCURRENTLY idx_model_performance_composite
ON cross_validation_experiments (model_id, status, mean_f1_score)
INCLUDE (mean_accuracy, mean_precision, mean_recall);

-- Vector similarity search optimization
DROP INDEX IF EXISTS idx_gpr_image_vector_optimized;
CREATE INDEX CONCURRENTLY idx_gpr_image_vector_optimized
ON gpr_image_data USING ivfflat (feature_vector vector_cosine_ops)
WITH (lists = 100); -- Optimize for dataset size

-- Time-series optimized indexes for signal processing
DROP INDEX IF EXISTS idx_signal_timeseries_composite;
CREATE INDEX CONCURRENTLY idx_signal_timeseries_composite
ON gpr_signal_timeseries (survey_id, trace_number)
INCLUDE (signal_to_clutter_ratio, coherence_coefficient);

-- Ground truth validation optimization
DROP INDEX IF EXISTS idx_ground_truth_validation_composite;
CREATE INDEX CONCURRENTLY idx_ground_truth_validation_composite
ON ground_truth_validations (campaign_id, detection_result, validation_confidence)
INCLUDE (horizontal_position_error_mm, depth_error_mm);

-- USAG strike incident spatial-temporal index
DROP INDEX IF EXISTS idx_strike_incidents_spatio_temporal;
CREATE INDEX CONCURRENTLY idx_strike_incidents_spatio_temporal
ON usag_strike_incidents USING GIST (incident_location, incident_date);

-- PAS 128 compliance tracking
DROP INDEX IF EXISTS idx_pas128_compliance_composite;
CREATE INDEX CONCURRENTLY idx_pas128_compliance_composite
ON pas128_compliance_assessments (target_quality_level_id, compliant, compliance_score)
INCLUDE (assessment_date, horizontal_accuracy_achieved_mm);

-- ================================================================
-- ADVANCED STATISTICS AND QUERY OPTIMIZATION
-- ================================================================

-- Increase statistics target for frequently queried columns
ALTER TABLE gpr_surveys ALTER COLUMN confidence_score SET STATISTICS 1000;
ALTER TABLE detected_utilities ALTER COLUMN depth_m SET STATISTICS 1000;
ALTER TABLE environmental_metadata ALTER COLUMN ground_relative_permittivity SET STATISTICS 1000;
ALTER TABLE ground_truth_validations ALTER COLUMN horizontal_position_error_mm SET STATISTICS 1000;
ALTER TABLE cross_validation_experiments ALTER COLUMN mean_f1_score SET STATISTICS 1000;

-- Create extended statistics for correlated columns
CREATE STATISTICS IF NOT EXISTS stats_gpr_survey_environment
ON project_id, quality_level, confidence_score, amount_of_utilities
FROM gpr_surveys;

CREATE STATISTICS IF NOT EXISTS stats_detected_utilities_correlation
ON utility_discipline, depth_m, confidence_score, utility_diameter_mm
FROM detected_utilities;

CREATE STATISTICS IF NOT EXISTS stats_environmental_correlation
ON ground_relative_permittivity, amount_of_utilities, utility_crossing
FROM environmental_metadata;

-- Analyze tables to update statistics
ANALYZE gpr_surveys;
ANALYZE detected_utilities;
ANALYZE environmental_metadata;
ANALYZE ground_truth_validations;
ANALYZE cross_validation_experiments;

-- ================================================================
-- PARTITIONING STRATEGY FOR LARGE TABLES
-- ================================================================

-- Partition gpr_signal_data by survey_id for better performance
-- Note: This requires recreating the table in production with existing data

-- Create partitioned table structure for new installations
/*
CREATE TABLE gpr_signal_data_partitioned (
    LIKE gpr_signal_data INCLUDING ALL
) PARTITION BY HASH (survey_id);

-- Create 8 hash partitions
DO $$
DECLARE
    i INTEGER;
BEGIN
    FOR i IN 0..7 LOOP
        EXECUTE format('CREATE TABLE gpr_signal_data_p%s PARTITION OF gpr_signal_data_partitioned FOR VALUES WITH (modulus 8, remainder %s)', i, i);
    END LOOP;
END $$;
*/

-- ================================================================
-- MATERIALIZED VIEW OPTIMIZATION
-- ================================================================

-- Optimize materialized view refresh with concurrent refresh where possible
CREATE OR REPLACE FUNCTION refresh_critical_views()
RETURNS TEXT AS $$
DECLARE
    result TEXT := '';
BEGIN
    -- Refresh views that are frequently accessed
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY survey_statistics;
        result := result || 'survey_statistics refreshed' || chr(10);
    EXCEPTION WHEN OTHERS THEN
        result := result || 'survey_statistics failed: ' || SQLERRM || chr(10);
    END;

    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY signal_quality_summary;
        result := result || 'signal_quality_summary refreshed' || chr(10);
    EXCEPTION WHEN OTHERS THEN
        result := result || 'signal_quality_summary failed: ' || SQLERRM || chr(10);
    END;

    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY validation_performance_summary;
        result := result || 'validation_performance_summary refreshed' || chr(10);
    EXCEPTION WHEN OTHERS THEN
        result := result || 'validation_performance_summary failed: ' || SQLERRM || chr(10);
    END;

    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY model_performance_leaderboard;
        result := result || 'model_performance_leaderboard refreshed' || chr(10);
    EXCEPTION WHEN OTHERS THEN
        result := result || 'model_performance_leaderboard failed: ' || SQLERRM || chr(10);
    END;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic refresh of materialized views
CREATE OR REPLACE FUNCTION schedule_view_refresh()
RETURNS void AS $$
BEGIN
    -- This would be called by a cron job or scheduler
    PERFORM refresh_critical_views();

    -- Log the refresh
    INSERT INTO deployment_log (deployment_type, script_name, notes)
    VALUES ('maintenance', 'materialized_view_refresh', 'Automated materialized view refresh completed');
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- QUERY OPTIMIZATION FUNCTIONS
-- ================================================================

-- Function to analyze query performance
CREATE OR REPLACE FUNCTION analyze_query_performance(
    min_calls INTEGER DEFAULT 10,
    min_total_time DECIMAL DEFAULT 1000.0
)
RETURNS TABLE (
    query_snippet TEXT,
    calls BIGINT,
    total_time_ms DECIMAL,
    mean_time_ms DECIMAL,
    rows_returned BIGINT,
    optimization_suggestion TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        LEFT(qs.query, 100) || '...' AS query_snippet,
        qs.calls,
        ROUND(qs.total_time, 2) AS total_time_ms,
        ROUND(qs.mean_time, 2) AS mean_time_ms,
        qs.rows AS rows_returned,
        CASE
            WHEN qs.mean_time > 5000 THEN 'Consider adding indexes or query optimization'
            WHEN qs.calls > 1000 AND qs.mean_time > 100 THEN 'High frequency query - consider caching'
            WHEN qs.rows > 10000 AND qs.mean_time > 1000 THEN 'Large result set - consider pagination'
            ELSE 'Performance acceptable'
        END AS optimization_suggestion
    FROM pg_stat_statements qs
    WHERE qs.calls >= min_calls
    AND qs.total_time >= min_total_time
    ORDER BY qs.total_time DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to suggest missing indexes
CREATE OR REPLACE FUNCTION suggest_missing_indexes()
RETURNS TABLE (
    table_name TEXT,
    column_suggestions TEXT,
    reason TEXT,
    seq_scans BIGINT,
    seq_reads BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.tablename,
        'Consider index on frequently filtered columns',
        CASE
            WHEN t.seq_scan > 1000 AND t.seq_tup_read > 10000 THEN 'High sequential scan activity'
            WHEN t.seq_tup_read / GREATEST(t.seq_scan, 1) > 1000 THEN 'Large table scans detected'
            ELSE 'Monitor for optimization opportunities'
        END,
        t.seq_scan,
        t.seq_tup_read
    FROM pg_stat_user_tables t
    WHERE t.seq_scan > 100
    AND t.seq_tup_read > 1000
    ORDER BY t.seq_tup_read DESC;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- STORAGE OPTIMIZATION
-- ================================================================

-- Enable compression for large text fields
ALTER TABLE gpr_surveys ALTER COLUMN survey_map_path SET STORAGE EXTENDED;
ALTER TABLE environmental_metadata ALTER COLUMN other_ground_disturbances SET STORAGE EXTENDED;
ALTER TABLE ground_truth_validations ALTER COLUMN field_notes SET STORAGE EXTENDED;

-- Set FILLFACTOR for tables with frequent updates
ALTER TABLE gpr_surveys SET (fillfactor = 85);
ALTER TABLE environmental_metadata SET (fillfactor = 85);
ALTER TABLE ground_truth_validations SET (fillfactor = 90);

-- ================================================================
-- CONNECTION AND MEMORY OPTIMIZATION
-- ================================================================

-- Connection pooling recommendations (for application configuration)
CREATE OR REPLACE FUNCTION get_connection_recommendations()
RETURNS TABLE (
    setting_name TEXT,
    current_value TEXT,
    recommended_value TEXT,
    reason TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'max_connections',
        current_setting('max_connections'),
        '200',
        'Balanced for ML workloads and concurrent analytics'
    UNION ALL
    SELECT
        'shared_buffers',
        current_setting('shared_buffers'),
        '25% of RAM',
        'Optimize for GPR data processing'
    UNION ALL
    SELECT
        'work_mem',
        current_setting('work_mem'),
        '64MB',
        'Support complex spatial and vector queries'
    UNION ALL
    SELECT
        'maintenance_work_mem',
        current_setting('maintenance_work_mem'),
        '512MB',
        'Speed up index creation and VACUUM operations'
    UNION ALL
    SELECT
        'effective_cache_size',
        current_setting('effective_cache_size'),
        '75% of RAM',
        'Help query planner make better decisions';
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- VACUUM AND MAINTENANCE OPTIMIZATION
-- ================================================================

-- Custom VACUUM strategy for different table types
CREATE OR REPLACE FUNCTION smart_vacuum_analyze()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    result TEXT := '';
BEGIN
    -- Heavy transaction tables - more frequent VACUUM
    FOR table_record IN
        SELECT schemaname, tablename FROM pg_stat_user_tables
        WHERE n_tup_upd + n_tup_del > 1000
        ORDER BY n_tup_upd + n_tup_del DESC
    LOOP
        EXECUTE format('VACUUM (ANALYZE, SKIP_LOCKED) %I.%I', table_record.schemaname, table_record.tablename);
        result := result || format('VACUUM %s.%s' || chr(10), table_record.schemaname, table_record.tablename);
    END LOOP;

    -- Large tables - analyze only
    FOR table_record IN
        SELECT schemaname, tablename FROM pg_stat_user_tables
        WHERE n_tup_ins > 10000 AND (n_tup_upd + n_tup_del) < 100
        ORDER BY n_tup_ins DESC
    LOOP
        EXECUTE format('ANALYZE %I.%I', table_record.schemaname, table_record.tablename);
        result := result || format('ANALYZE %s.%s' || chr(10), table_record.schemaname, table_record.tablename);
    END LOOP;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- MONITORING AND ALERTING FUNCTIONS
-- ================================================================

-- Function to check database health
CREATE OR REPLACE FUNCTION check_database_health()
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    value TEXT,
    recommendation TEXT
) AS $$
BEGIN
    -- Check database size
    RETURN QUERY
    SELECT
        'Database Size',
        CASE WHEN pg_database_size(current_database()) > 100 * 1024 * 1024 * 1024 THEN 'WARNING' ELSE 'OK' END,
        pg_size_pretty(pg_database_size(current_database())),
        CASE WHEN pg_database_size(current_database()) > 100 * 1024 * 1024 * 1024 THEN 'Consider archiving old data' ELSE 'Size is manageable' END
    UNION ALL
    -- Check for unused indexes
    SELECT
        'Unused Indexes',
        CASE WHEN COUNT(*) > 5 THEN 'WARNING' ELSE 'OK' END,
        COUNT(*)::TEXT,
        CASE WHEN COUNT(*) > 5 THEN 'Review and drop unused indexes' ELSE 'Index usage is good' END
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0
    UNION ALL
    -- Check connection usage
    SELECT
        'Active Connections',
        CASE WHEN COUNT(*) > 150 THEN 'WARNING' ELSE 'OK' END,
        COUNT(*)::TEXT,
        CASE WHEN COUNT(*) > 150 THEN 'Consider connection pooling' ELSE 'Connection usage is normal' END
    FROM pg_stat_activity
    WHERE state = 'active';
END;
$$ LANGUAGE plpgsql;

-- Function to monitor long-running queries
CREATE OR REPLACE FUNCTION check_long_running_queries(
    duration_threshold INTERVAL DEFAULT '5 minutes'
)
RETURNS TABLE (
    pid INTEGER,
    duration INTERVAL,
    query_snippet TEXT,
    state TEXT,
    wait_event TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        sa.pid,
        NOW() - sa.query_start AS duration,
        LEFT(sa.query, 100) AS query_snippet,
        sa.state,
        sa.wait_event
    FROM pg_stat_activity sa
    WHERE sa.query_start IS NOT NULL
    AND NOW() - sa.query_start > duration_threshold
    AND sa.state != 'idle'
    ORDER BY duration DESC;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- PERFORMANCE TESTING FUNCTIONS
-- ================================================================

-- Function to run performance benchmarks
CREATE OR REPLACE FUNCTION run_performance_benchmark()
RETURNS TABLE (
    test_name TEXT,
    execution_time_ms DECIMAL,
    rows_processed INTEGER,
    performance_score TEXT
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    duration DECIMAL;
    row_count INTEGER;
BEGIN
    -- Test 1: Spatial query performance
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count
    FROM detected_utilities du
    JOIN survey_sites ss ON ST_DWithin(du.coordinates, ss.location, 0.001);
    end_time := clock_timestamp();
    duration := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

    RETURN QUERY SELECT
        'Spatial Query Test',
        duration,
        row_count,
        CASE WHEN duration < 100 THEN 'Excellent' WHEN duration < 500 THEN 'Good' ELSE 'Needs Optimization' END;

    -- Test 2: ML model query performance
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count
    FROM cross_validation_experiments cve
    JOIN ml_models m ON cve.model_id = m.id
    WHERE cve.mean_f1_score > 0.8;
    end_time := clock_timestamp();
    duration := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

    RETURN QUERY SELECT
        'ML Analytics Query Test',
        duration,
        row_count,
        CASE WHEN duration < 50 THEN 'Excellent' WHEN duration < 200 THEN 'Good' ELSE 'Needs Optimization' END;

    -- Test 3: Environmental correlation query
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count
    FROM environmental_metadata em
    JOIN gpr_surveys s ON em.survey_id = s.id
    WHERE em.ground_relative_permittivity BETWEEN 10 AND 20
    AND s.confidence_score > 0.8;
    end_time := clock_timestamp();
    duration := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

    RETURN QUERY SELECT
        'Environmental Correlation Test',
        duration,
        row_count,
        CASE WHEN duration < 75 THEN 'Excellent' WHEN duration < 300 THEN 'Good' ELSE 'Needs Optimization' END;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- OPTIMIZATION SUMMARY AND RECOMMENDATIONS
-- ================================================================

-- Function to provide optimization summary
CREATE OR REPLACE FUNCTION get_optimization_summary()
RETURNS TEXT AS $$
DECLARE
    result TEXT := '';
    index_count INTEGER;
    table_count INTEGER;
    total_size TEXT;
BEGIN
    SELECT COUNT(*) INTO index_count FROM pg_indexes WHERE schemaname = 'public';
    SELECT COUNT(*) INTO table_count FROM pg_tables WHERE schemaname = 'public';
    SELECT pg_size_pretty(pg_database_size(current_database())) INTO total_size;

    result := result || '================================================================' || chr(10);
    result := result || 'DATABASE OPTIMIZATION SUMMARY' || chr(10);
    result := result || '================================================================' || chr(10);
    result := result || '' || chr(10);
    result := result || 'Database: ' || current_database() || chr(10);
    result := result || 'Total Size: ' || total_size || chr(10);
    result := result || 'Tables: ' || table_count || chr(10);
    result := result || 'Indexes: ' || index_count || chr(10);
    result := result || '' || chr(10);
    result := result || 'OPTIMIZATIONS APPLIED:' || chr(10);
    result := result || '✓ Advanced composite indexes for common query patterns' || chr(10);
    result := result || '✓ Spatial indexing optimization for GPR data' || chr(10);
    result := result || '✓ Vector similarity search optimization' || chr(10);
    result := result || '✓ Extended statistics for correlated columns' || chr(10);
    result := result || '✓ Storage optimization for large fields' || chr(10);
    result := result || '✓ Performance monitoring views' || chr(10);
    result := result || '✓ Automated maintenance functions' || chr(10);
    result := result || '' || chr(10);
    result := result || 'MONITORING FUNCTIONS AVAILABLE:' || chr(10);
    result := result || '• analyze_query_performance() - Identify slow queries' || chr(10);
    result := result || '• suggest_missing_indexes() - Find indexing opportunities' || chr(10);
    result := result || '• check_database_health() - Overall health check' || chr(10);
    result := result || '• check_long_running_queries() - Monitor query performance' || chr(10);
    result := result || '• run_performance_benchmark() - Performance testing' || chr(10);
    result := result || '' || chr(10);
    result := result || 'RECOMMENDED MAINTENANCE SCHEDULE:' || chr(10);
    result := result || '• Daily: refresh_critical_views()' || chr(10);
    result := result || '• Weekly: smart_vacuum_analyze()' || chr(10);
    result := result || '• Monthly: check_database_health()' || chr(10);
    result := result || '• Quarterly: run_performance_benchmark()' || chr(10);
    result := result || '' || chr(10);
    result := result || 'For production deployment, consider:' || chr(10);
    result := result || '• Setting up connection pooling (PgBouncer)' || chr(10);
    result := result || '• Configuring automated backups' || chr(10);
    result := result || '• Implementing monitoring alerts' || chr(10);
    result := result || '• Regular performance review cycles' || chr(10);
    result := result || '================================================================' || chr(10);

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- FINALIZATION
-- ================================================================

-- Log optimization completion
INSERT INTO deployment_log (deployment_type, script_name, version, notes)
VALUES ('optimization', 'database_optimization.sql', '1.0.0', 'Database optimization and performance enhancements applied');

-- Run final analysis
ANALYZE;

-- Display optimization summary
SELECT get_optimization_summary() AS optimization_complete;