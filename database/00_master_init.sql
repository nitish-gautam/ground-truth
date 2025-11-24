-- ================================================================
-- UNDERGROUND UTILITY DETECTION PLATFORM - MASTER DATABASE INITIALIZATION
-- Comprehensive PostgreSQL Database Setup for GPR Feature Analysis
-- ================================================================

-- Required PostgreSQL version check
DO $$
BEGIN
    IF current_setting('server_version_num')::integer < 140000 THEN
        RAISE EXCEPTION 'PostgreSQL version 14.0 or higher required. Current version: %', version();
    END IF;
    RAISE NOTICE 'PostgreSQL version check passed: %', version();
END $$;

-- ================================================================
-- EXTENSION SETUP
-- ================================================================

-- Core PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"
    WITH SCHEMA public;

-- PostGIS for spatial data handling
CREATE EXTENSION IF NOT EXISTS "postgis"
    WITH SCHEMA public;

-- Vector extension for ML embeddings and similarity search
CREATE EXTENSION IF NOT EXISTS "vector"
    WITH SCHEMA public;

-- Full-text search and string matching
CREATE EXTENSION IF NOT EXISTS "pg_trgm"
    WITH SCHEMA public;

-- Additional useful extensions
CREATE EXTENSION IF NOT EXISTS "btree_gin"
    WITH SCHEMA public;

CREATE EXTENSION IF NOT EXISTS "btree_gist"
    WITH SCHEMA public;

-- Statistical extensions for advanced analytics
CREATE EXTENSION IF NOT EXISTS "tablefunc"
    WITH SCHEMA public;

-- Check extension installation
DO $$
DECLARE
    ext_record RECORD;
    extension_list TEXT[] := ARRAY['uuid-ossp', 'postgis', 'vector', 'pg_trgm', 'btree_gin', 'btree_gist', 'tablefunc'];
    ext_name TEXT;
BEGIN
    FOREACH ext_name IN ARRAY extension_list
    LOOP
        SELECT * INTO ext_record FROM pg_extension WHERE extname = ext_name;
        IF FOUND THEN
            RAISE NOTICE 'Extension % installed successfully', ext_name;
        ELSE
            RAISE WARNING 'Extension % not found', ext_name;
        END IF;
    END LOOP;
END $$;

-- ================================================================
-- DATABASE CONFIGURATION OPTIMIZATION
-- ================================================================

-- Set optimal configuration for GPR data processing
-- Note: These settings may require superuser privileges in production

-- Memory and performance settings
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Vector extension settings for ML workloads
ALTER SYSTEM SET max_connections = '200';
ALTER SYSTEM SET random_page_cost = 1.1;

-- Enable parallel processing for large datasets
ALTER SYSTEM SET max_parallel_workers = '8';
ALTER SYSTEM SET max_parallel_workers_per_gather = '4';

-- Logging configuration for monitoring
ALTER SYSTEM SET log_statement = 'ddl';
ALTER SYSTEM SET log_min_duration_statement = '1000'; -- Log queries > 1 second

RAISE NOTICE 'Database configuration optimized. Restart PostgreSQL to apply system-level changes.';

-- ================================================================
-- SCHEMA ORGANIZATION
-- ================================================================

-- Create schemas for logical organization
CREATE SCHEMA IF NOT EXISTS gpr_data
    AUTHORIZATION CURRENT_USER;

CREATE SCHEMA IF NOT EXISTS environmental
    AUTHORIZATION CURRENT_USER;

CREATE SCHEMA IF NOT EXISTS validation
    AUTHORIZATION CURRENT_USER;

CREATE SCHEMA IF NOT EXISTS ml_analytics
    AUTHORIZATION CURRENT_USER;

CREATE SCHEMA IF NOT EXISTS compliance
    AUTHORIZATION CURRENT_USER;

CREATE SCHEMA IF NOT EXISTS historical
    AUTHORIZATION CURRENT_USER;

CREATE SCHEMA IF NOT EXISTS utilities
    AUTHORIZATION CURRENT_USER;

-- Set search path to include all schemas
ALTER DATABASE CURRENT_DATABASE() SET search_path TO public, gpr_data, environmental, validation, ml_analytics, compliance, historical, utilities;

COMMENT ON SCHEMA gpr_data IS 'GPR survey data, signal processing, and feature extraction';
COMMENT ON SCHEMA environmental IS 'Environmental metadata and correlation analysis';
COMMENT ON SCHEMA validation IS 'Ground truth validation and accuracy assessment';
COMMENT ON SCHEMA ml_analytics IS 'Machine learning model performance and analytics';
COMMENT ON SCHEMA compliance IS 'PAS 128 compliance and quality assurance';
COMMENT ON SCHEMA historical IS 'Historical incident data and pattern analysis';
COMMENT ON SCHEMA utilities IS 'Utility functions and common procedures';

-- ================================================================
-- CUSTOM DATA TYPES
-- ================================================================

-- Quality level enumeration for PAS 128
CREATE TYPE quality_level_enum AS ENUM ('QL-A', 'QL-B', 'QL-C', 'QL-D');

-- Detection result enumeration
CREATE TYPE detection_result_enum AS ENUM ('true_positive', 'false_positive', 'false_negative', 'true_negative');

-- Model development stage enumeration
CREATE TYPE model_stage_enum AS ENUM ('development', 'training', 'validation', 'testing', 'production', 'deprecated');

-- Confidence level enumeration
CREATE TYPE confidence_level_enum AS ENUM ('very_low', 'low', 'medium', 'high', 'very_high');

-- Survey status enumeration
CREATE TYPE survey_status_enum AS ENUM ('planned', 'active', 'completed', 'validated', 'archived');

-- Utility discipline enumeration
CREATE TYPE utility_discipline_enum AS ENUM ('electricity', 'water', 'sewer', 'telecommunications', 'gas', 'oilGasChemicals', 'heating', 'other');

COMMENT ON TYPE quality_level_enum IS 'PAS 128:2022 quality levels for utility detection surveys';
COMMENT ON TYPE detection_result_enum IS 'Ground truth validation results classification';
COMMENT ON TYPE model_stage_enum IS 'ML model development lifecycle stages';

-- ================================================================
-- AUDIT TRAIL SETUP
-- ================================================================

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    row_id TEXT, -- Flexible to handle different ID types
    old_values JSONB,
    new_values JSONB,
    changed_by TEXT DEFAULT CURRENT_USER,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    application_name TEXT DEFAULT current_setting('application_name', true),
    client_addr INET DEFAULT inet_client_addr()
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (
            schema_name, table_name, operation, row_id, old_values
        ) VALUES (
            TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP,
            COALESCE(OLD.id::TEXT, 'unknown'),
            row_to_json(OLD)::JSONB
        );
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (
            schema_name, table_name, operation, row_id, old_values, new_values
        ) VALUES (
            TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP,
            COALESCE(NEW.id::TEXT, 'unknown'),
            row_to_json(OLD)::JSONB,
            row_to_json(NEW)::JSONB
        );
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (
            schema_name, table_name, operation, row_id, new_values
        ) VALUES (
            TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP,
            COALESCE(NEW.id::TEXT, 'unknown'),
            row_to_json(NEW)::JSONB
        );
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Index for audit log performance
CREATE INDEX IF NOT EXISTS idx_audit_log_table ON audit_log(schema_name, table_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);

-- ================================================================
-- UTILITY FUNCTIONS
-- ================================================================

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_statistics()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    row_count BIGINT,
    table_size_pretty TEXT,
    index_size_pretty TEXT,
    total_size_pretty TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname::TEXT,
        tablename::TEXT,
        n_tup_ins - n_tup_del AS row_count,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS table_size_pretty,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS index_size_pretty,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) + pg_indexes_size(schemaname||'.'||tablename)) AS total_size_pretty
    FROM pg_stat_user_tables
    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS TEXT AS $$
DECLARE
    view_record RECORD;
    result_text TEXT := '';
BEGIN
    FOR view_record IN
        SELECT schemaname, matviewname
        FROM pg_matviews
        WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
        ORDER BY schemaname, matviewname
    LOOP
        BEGIN
            EXECUTE format('REFRESH MATERIALIZED VIEW %I.%I', view_record.schemaname, view_record.matviewname);
            result_text := result_text || format('✓ Refreshed %s.%s' || chr(10), view_record.schemaname, view_record.matviewname);
        EXCEPTION WHEN OTHERS THEN
            result_text := result_text || format('✗ Failed to refresh %s.%s: %s' || chr(10), view_record.schemaname, view_record.matviewname, SQLERRM);
        END;
    END LOOP;

    IF result_text = '' THEN
        result_text := 'No materialized views found to refresh.';
    END IF;

    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze all tables for performance optimization
CREATE OR REPLACE FUNCTION analyze_all_tables()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    result_text TEXT := '';
BEGIN
    FOR table_record IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE schemaname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        ORDER BY schemaname, tablename
    LOOP
        BEGIN
            EXECUTE format('ANALYZE %I.%I', table_record.schemaname, table_record.tablename);
            result_text := result_text || format('✓ Analyzed %s.%s' || chr(10), table_record.schemaname, table_record.tablename);
        EXCEPTION WHEN OTHERS THEN
            result_text := result_text || format('✗ Failed to analyze %s.%s: %s' || chr(10), table_record.schemaname, table_record.tablename, SQLERRM);
        END;
    END LOOP;

    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- SECURITY SETUP
-- ================================================================

-- Create roles for different access levels
DO $$
BEGIN
    -- Database administrators
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gpr_admin') THEN
        CREATE ROLE gpr_admin WITH LOGIN PASSWORD 'change_me_admin_2024!';
        GRANT ALL PRIVILEGES ON DATABASE CURRENT_DATABASE() TO gpr_admin;
        RAISE NOTICE 'Created role: gpr_admin';
    END IF;

    -- Data analysts and scientists
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gpr_analyst') THEN
        CREATE ROLE gpr_analyst WITH LOGIN PASSWORD 'change_me_analyst_2024!';
        GRANT CONNECT ON DATABASE CURRENT_DATABASE() TO gpr_analyst;
        GRANT USAGE ON ALL SCHEMAS IN DATABASE CURRENT_DATABASE() TO gpr_analyst;
        GRANT SELECT ON ALL TABLES IN DATABASE CURRENT_DATABASE() TO gpr_analyst;
        GRANT EXECUTE ON ALL FUNCTIONS IN DATABASE CURRENT_DATABASE() TO gpr_analyst;
        RAISE NOTICE 'Created role: gpr_analyst';
    END IF;

    -- Application users
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gpr_app_user') THEN
        CREATE ROLE gpr_app_user WITH LOGIN PASSWORD 'change_me_app_2024!';
        GRANT CONNECT ON DATABASE CURRENT_DATABASE() TO gpr_app_user;
        GRANT USAGE ON ALL SCHEMAS IN DATABASE CURRENT_DATABASE() TO gpr_app_user;
        GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN DATABASE CURRENT_DATABASE() TO gpr_app_user;
        GRANT EXECUTE ON ALL FUNCTIONS IN DATABASE CURRENT_DATABASE() TO gpr_app_user;
        RAISE NOTICE 'Created role: gpr_app_user';
    END IF;

    -- Read-only users
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gpr_readonly') THEN
        CREATE ROLE gpr_readonly WITH LOGIN PASSWORD 'change_me_readonly_2024!';
        GRANT CONNECT ON DATABASE CURRENT_DATABASE() TO gpr_readonly;
        GRANT USAGE ON ALL SCHEMAS IN DATABASE CURRENT_DATABASE() TO gpr_readonly;
        GRANT SELECT ON ALL TABLES IN DATABASE CURRENT_DATABASE() TO gpr_readonly;
        RAISE NOTICE 'Created role: gpr_readonly';
    END IF;
END $$;

-- Row Level Security setup for multi-tenancy
ALTER TABLE IF EXISTS projects ENABLE ROW LEVEL SECURITY;

-- Create policy for project-based access control
CREATE POLICY IF NOT EXISTS project_access_policy ON projects
    FOR ALL TO gpr_app_user
    USING (client_organization = current_setting('app.current_client', true));

-- ================================================================
-- INITIALIZATION VALIDATION
-- ================================================================

-- Validate initialization
DO $$
DECLARE
    extension_count INTEGER;
    schema_count INTEGER;
    function_count INTEGER;
BEGIN
    -- Check extensions
    SELECT COUNT(*) INTO extension_count
    FROM pg_extension
    WHERE extname IN ('uuid-ossp', 'postgis', 'vector', 'pg_trgm');

    IF extension_count < 4 THEN
        RAISE WARNING 'Not all required extensions are installed. Expected 4, found %', extension_count;
    ELSE
        RAISE NOTICE 'All required extensions installed successfully';
    END IF;

    -- Check schemas
    SELECT COUNT(*) INTO schema_count
    FROM information_schema.schemata
    WHERE schema_name IN ('gpr_data', 'environmental', 'validation', 'ml_analytics', 'compliance', 'historical', 'utilities');

    IF schema_count < 7 THEN
        RAISE WARNING 'Not all schemas created. Expected 7, found %', schema_count;
    ELSE
        RAISE NOTICE 'All schemas created successfully';
    END IF;

    -- Check utility functions
    SELECT COUNT(*) INTO function_count
    FROM information_schema.routines
    WHERE routine_name IN ('get_database_statistics', 'refresh_all_materialized_views', 'analyze_all_tables');

    IF function_count < 3 THEN
        RAISE WARNING 'Not all utility functions created. Expected 3, found %', function_count;
    ELSE
        RAISE NOTICE 'All utility functions created successfully';
    END IF;

    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Next steps: 1) Deploy schema files 2) Load sample data 3) Configure application connections';
END $$;

-- ================================================================
-- DEPLOYMENT LOG
-- ================================================================

-- Create deployment tracking
CREATE TABLE IF NOT EXISTS deployment_log (
    id SERIAL PRIMARY KEY,
    deployment_type VARCHAR(100) NOT NULL,
    script_name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_by TEXT DEFAULT CURRENT_USER,
    status VARCHAR(50) DEFAULT 'success',
    notes TEXT
);

-- Log this initialization
INSERT INTO deployment_log (deployment_type, script_name, version, notes)
VALUES ('initialization', '00_master_init.sql', '1.0.0', 'Master database initialization completed');

-- ================================================================
-- FINAL NOTICES
-- ================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '================================================================';
    RAISE NOTICE 'UNDERGROUND UTILITY DETECTION PLATFORM - DATABASE INITIALIZED';
    RAISE NOTICE '================================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
    RAISE NOTICE 'Timestamp: %', NOW();
    RAISE NOTICE '';
    RAISE NOTICE 'IMPORTANT SECURITY NOTES:';
    RAISE NOTICE '- Change all default passwords immediately';
    RAISE NOTICE '- Review and adjust role permissions for production';
    RAISE NOTICE '- Configure SSL/TLS for database connections';
    RAISE NOTICE '- Set up regular backup procedures';
    RAISE NOTICE '';
    RAISE NOTICE 'NEXT STEPS:';
    RAISE NOTICE '1. Deploy individual schema files in order';
    RAISE NOTICE '2. Load sample data using data loading scripts';
    RAISE NOTICE '3. Configure application connection parameters';
    RAISE NOTICE '4. Set up monitoring and alerting';
    RAISE NOTICE '';
    RAISE NOTICE 'For support, refer to database documentation';
    RAISE NOTICE '================================================================';
END $$;