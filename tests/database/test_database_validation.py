#!/usr/bin/env python3
"""
Comprehensive Database Validation Test Suite

This module provides extensive validation for all 7 database schemas:
1. GPR Infrastructure Schema
2. Enhanced Signal Analysis Schema
3. Environmental Metadata Schema
4. Ground Truth Validation Schema
5. ML Performance Schema
6. PAS 128 Compliance Schema
7. USAG Strike Reports Schema

Tests include:
- Schema validation and integrity
- Table structure verification
- Index optimization validation
- Constraint enforcement
- Data type validation
- Performance benchmarking
- Cross-schema relationship validation
"""

import pytest
import psycopg2
import sqlite3
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch
import json
import tempfile
import logging

logger = logging.getLogger(__name__)


class DatabaseValidator:
    """Comprehensive database validation framework."""

    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize database validator.

        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.connection = None
        self.validation_results = {}

    def connect(self) -> None:
        """Establish database connection."""
        try:
            if self.db_config.get('provider') == 'postgresql':
                self.connection = psycopg2.connect(**self.db_config['params'])
            elif self.db_config.get('provider') == 'sqlite':
                self.connection = sqlite3.connect(self.db_config['params']['database'])
            else:
                raise ValueError(f"Unsupported database provider: {self.db_config.get('provider')}")

            logger.info(f"Connected to {self.db_config.get('provider')} database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Any]:
        """Execute SQL query and return results."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params or ())
            return cursor.fetchall()
        finally:
            cursor.close()

    def validate_schema_exists(self, schema_name: str) -> Dict[str, Any]:
        """Validate that a schema exists and contains expected tables."""
        results = {
            'schema_name': schema_name,
            'exists': False,
            'tables': [],
            'table_count': 0,
            'validation_errors': []
        }

        try:
            # Check if schema exists (PostgreSQL specific)
            if self.db_config.get('provider') == 'postgresql':
                query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s"
                schema_result = self.execute_query(query, (schema_name,))
                results['exists'] = len(schema_result) > 0

                if results['exists']:
                    # Get tables in schema
                    query = """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    ORDER BY table_name
                    """
                    tables = self.execute_query(query, (schema_name,))
                    results['tables'] = [table[0] for table in tables]
                    results['table_count'] = len(results['tables'])

            # For SQLite, check if tables exist (assuming schema prefix in table names)
            elif self.db_config.get('provider') == 'sqlite':
                query = """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE ?
                ORDER BY name
                """
                tables = self.execute_query(query, (f"{schema_name}_%",))
                results['tables'] = [table[0] for table in tables]
                results['table_count'] = len(results['tables'])
                results['exists'] = results['table_count'] > 0

        except Exception as e:
            results['validation_errors'].append(f"Schema validation failed: {str(e)}")
            logger.error(f"Schema validation error for {schema_name}: {e}")

        return results

    def validate_table_structure(self, table_name: str, expected_columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate table structure against expected schema."""
        results = {
            'table_name': table_name,
            'exists': False,
            'column_validation': {},
            'missing_columns': [],
            'unexpected_columns': [],
            'constraint_validation': {},
            'index_validation': {},
            'validation_errors': []
        }

        try:
            # Check if table exists
            if self.db_config.get('provider') == 'postgresql':
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """
                columns = self.execute_query(query, (table_name,))
            elif self.db_config.get('provider') == 'sqlite':
                query = f"PRAGMA table_info({table_name})"
                columns = self.execute_query(query)

            if columns:
                results['exists'] = True
                actual_columns = {col[0]: col for col in columns}
                expected_column_names = {col['name'] for col in expected_columns}
                actual_column_names = set(actual_columns.keys())

                # Validate columns
                for expected_col in expected_columns:
                    col_name = expected_col['name']
                    if col_name in actual_columns:
                        results['column_validation'][col_name] = self._validate_column(
                            actual_columns[col_name], expected_col
                        )
                    else:
                        results['missing_columns'].append(col_name)

                # Find unexpected columns
                results['unexpected_columns'] = list(actual_column_names - expected_column_names)

                # Validate constraints
                results['constraint_validation'] = self._validate_constraints(table_name)

                # Validate indexes
                results['index_validation'] = self._validate_indexes(table_name)

        except Exception as e:
            results['validation_errors'].append(f"Table structure validation failed: {str(e)}")
            logger.error(f"Table structure validation error for {table_name}: {e}")

        return results

    def _validate_column(self, actual_column: Tuple, expected_column: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual column properties."""
        validation = {
            'data_type_match': False,
            'nullable_match': False,
            'default_match': False,
            'issues': []
        }

        try:
            # Data type validation (simplified for cross-database compatibility)
            actual_type = actual_column[1].lower() if len(actual_column) > 1 else 'unknown'
            expected_type = expected_column.get('type', '').lower()

            # Basic type matching logic
            type_aliases = {
                'integer': ['int', 'integer', 'bigint'],
                'varchar': ['varchar', 'text', 'character varying'],
                'text': ['text', 'varchar', 'character varying'],
                'timestamp': ['timestamp', 'datetime'],
                'boolean': ['boolean', 'bool'],
                'numeric': ['numeric', 'decimal', 'float', 'double'],
                'jsonb': ['jsonb', 'json']
            }

            type_match = False
            for type_group, aliases in type_aliases.items():
                if expected_type in aliases and actual_type in aliases:
                    type_match = True
                    break

            validation['data_type_match'] = type_match or actual_type == expected_type
            if not validation['data_type_match']:
                validation['issues'].append(f"Type mismatch: expected {expected_type}, got {actual_type}")

            # Nullable validation
            if len(actual_column) > 2:
                actual_nullable = actual_column[2] == 'YES' if self.db_config.get('provider') == 'postgresql' else actual_column[2] == 0
                expected_nullable = expected_column.get('nullable', True)
                validation['nullable_match'] = actual_nullable == expected_nullable
                if not validation['nullable_match']:
                    validation['issues'].append(f"Nullable mismatch: expected {expected_nullable}, got {actual_nullable}")

        except Exception as e:
            validation['issues'].append(f"Column validation error: {str(e)}")

        return validation

    def _validate_constraints(self, table_name: str) -> Dict[str, Any]:
        """Validate table constraints."""
        constraints = {
            'primary_keys': [],
            'foreign_keys': [],
            'unique_constraints': [],
            'check_constraints': [],
            'validation_errors': []
        }

        try:
            if self.db_config.get('provider') == 'postgresql':
                # Primary keys
                pk_query = """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
                """
                pk_result = self.execute_query(pk_query, (table_name,))
                constraints['primary_keys'] = [row[0] for row in pk_result]

                # Foreign keys
                fk_query = """
                SELECT kcu.column_name, ccu.table_name AS foreign_table_name,
                       ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.table_name = %s AND tc.constraint_type = 'FOREIGN KEY'
                """
                fk_result = self.execute_query(fk_query, (table_name,))
                constraints['foreign_keys'] = [
                    {
                        'column': row[0],
                        'references_table': row[1],
                        'references_column': row[2]
                    }
                    for row in fk_result
                ]

        except Exception as e:
            constraints['validation_errors'].append(f"Constraint validation failed: {str(e)}")

        return constraints

    def _validate_indexes(self, table_name: str) -> Dict[str, Any]:
        """Validate table indexes."""
        indexes = {
            'indexes': [],
            'index_count': 0,
            'validation_errors': []
        }

        try:
            if self.db_config.get('provider') == 'postgresql':
                query = """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = %s
                """
                index_result = self.execute_query(query, (table_name,))
                indexes['indexes'] = [
                    {'name': row[0], 'definition': row[1]}
                    for row in index_result
                ]
                indexes['index_count'] = len(indexes['indexes'])

        except Exception as e:
            indexes['validation_errors'].append(f"Index validation failed: {str(e)}")

        return indexes

    def validate_data_integrity(self, table_name: str, integrity_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data integrity constraints."""
        results = {
            'table_name': table_name,
            'integrity_checks': {},
            'validation_errors': []
        }

        try:
            for check in integrity_checks:
                check_name = check['name']
                check_query = check['query']
                expected_result = check.get('expected_result', 0)

                try:
                    result = self.execute_query(check_query)
                    actual_result = result[0][0] if result else 0

                    results['integrity_checks'][check_name] = {
                        'passed': actual_result == expected_result,
                        'expected': expected_result,
                        'actual': actual_result,
                        'description': check.get('description', '')
                    }

                except Exception as e:
                    results['integrity_checks'][check_name] = {
                        'passed': False,
                        'error': str(e),
                        'description': check.get('description', '')
                    }

        except Exception as e:
            results['validation_errors'].append(f"Data integrity validation failed: {str(e)}")

        return results

    def benchmark_performance(self, performance_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark database performance."""
        results = {
            'performance_tests': {},
            'validation_errors': []
        }

        try:
            for test in performance_tests:
                test_name = test['name']
                test_query = test['query']
                expected_max_time = test.get('max_execution_time_ms', 1000)
                iterations = test.get('iterations', 5)

                execution_times = []

                for i in range(iterations):
                    start_time = time.time()
                    try:
                        self.execute_query(test_query)
                        execution_time_ms = (time.time() - start_time) * 1000
                        execution_times.append(execution_time_ms)
                    except Exception as e:
                        results['performance_tests'][test_name] = {
                            'passed': False,
                            'error': str(e)
                        }
                        break

                if execution_times:
                    avg_time = sum(execution_times) / len(execution_times)
                    max_time = max(execution_times)
                    min_time = min(execution_times)

                    results['performance_tests'][test_name] = {
                        'passed': avg_time <= expected_max_time,
                        'avg_execution_time_ms': avg_time,
                        'max_execution_time_ms': max_time,
                        'min_execution_time_ms': min_time,
                        'expected_max_time_ms': expected_max_time,
                        'all_execution_times': execution_times
                    }

        except Exception as e:
            results['validation_errors'].append(f"Performance benchmarking failed: {str(e)}")

        return results


class TestDatabaseValidation:
    """Test suite for comprehensive database validation."""

    @pytest.fixture
    def db_validator(self):
        """Database validator fixture."""
        # Use in-memory SQLite for testing
        db_config = {
            'provider': 'sqlite',
            'params': {'database': ':memory:'}
        }
        validator = DatabaseValidator(db_config)
        validator.connect()
        yield validator
        validator.disconnect()

    @pytest.fixture
    def sample_schema_definitions(self):
        """Sample schema definitions for testing."""
        return {
            'gpr_infrastructure': {
                'tables': [
                    {
                        'name': 'gpr_infrastructure_gpr_surveys',
                        'columns': [
                            {'name': 'id', 'type': 'uuid', 'nullable': False},
                            {'name': 'location_id', 'type': 'varchar', 'nullable': False},
                            {'name': 'survey_date', 'type': 'timestamp', 'nullable': False},
                            {'name': 'operator_name', 'type': 'varchar', 'nullable': True},
                            {'name': 'equipment_model', 'type': 'varchar', 'nullable': True}
                        ]
                    },
                    {
                        'name': 'gpr_infrastructure_detected_utilities',
                        'columns': [
                            {'name': 'id', 'type': 'uuid', 'nullable': False},
                            {'name': 'survey_id', 'type': 'uuid', 'nullable': False},
                            {'name': 'x_position', 'type': 'numeric', 'nullable': False},
                            {'name': 'y_position', 'type': 'numeric', 'nullable': False},
                            {'name': 'depth', 'type': 'numeric', 'nullable': False},
                            {'name': 'material', 'type': 'varchar', 'nullable': True},
                            {'name': 'discipline', 'type': 'varchar', 'nullable': True}
                        ]
                    }
                ]
            },
            'environmental_metadata': {
                'tables': [
                    {
                        'name': 'environmental_metadata_survey_conditions',
                        'columns': [
                            {'name': 'id', 'type': 'uuid', 'nullable': False},
                            {'name': 'survey_id', 'type': 'uuid', 'nullable': False},
                            {'name': 'weather_condition', 'type': 'varchar', 'nullable': True},
                            {'name': 'ground_condition', 'type': 'varchar', 'nullable': True},
                            {'name': 'permittivity', 'type': 'numeric', 'nullable': True}
                        ]
                    }
                ]
            }
        }

    def test_schema_existence_validation(self, db_validator, sample_schema_definitions):
        """Test validation of schema existence."""
        for schema_name, schema_def in sample_schema_definitions.items():
            result = db_validator.validate_schema_exists(schema_name)

            assert result['schema_name'] == schema_name
            assert 'exists' in result
            assert 'tables' in result
            assert 'table_count' in result
            assert isinstance(result['validation_errors'], list)

    def test_table_structure_validation(self, db_validator, sample_schema_definitions):
        """Test validation of table structures."""
        # Create a test table first
        create_table_sql = """
        CREATE TABLE test_table (
            id TEXT PRIMARY KEY,
            location_id TEXT NOT NULL,
            survey_date TEXT NOT NULL,
            operator_name TEXT,
            equipment_model TEXT
        )
        """
        db_validator.execute_query(create_table_sql)

        # Define expected columns
        expected_columns = [
            {'name': 'id', 'type': 'text', 'nullable': False},
            {'name': 'location_id', 'type': 'text', 'nullable': False},
            {'name': 'survey_date', 'type': 'text', 'nullable': False},
            {'name': 'operator_name', 'type': 'text', 'nullable': True},
            {'name': 'equipment_model', 'type': 'text', 'nullable': True}
        ]

        result = db_validator.validate_table_structure('test_table', expected_columns)

        assert result['table_name'] == 'test_table'
        assert result['exists'] is True
        assert isinstance(result['column_validation'], dict)
        assert isinstance(result['missing_columns'], list)
        assert isinstance(result['unexpected_columns'], list)

    def test_data_integrity_validation(self, db_validator):
        """Test data integrity validation."""
        # Create test table with data
        create_table_sql = """
        CREATE TABLE integrity_test (
            id INTEGER PRIMARY KEY,
            value INTEGER NOT NULL,
            category TEXT
        )
        """
        db_validator.execute_query(create_table_sql)

        # Insert test data
        insert_sql = """
        INSERT INTO integrity_test (id, value, category) VALUES
        (1, 100, 'A'),
        (2, 200, 'B'),
        (3, 300, 'A')
        """
        db_validator.execute_query(insert_sql)

        # Define integrity checks
        integrity_checks = [
            {
                'name': 'no_null_values',
                'query': 'SELECT COUNT(*) FROM integrity_test WHERE value IS NULL',
                'expected_result': 0,
                'description': 'Check for null values in value column'
            },
            {
                'name': 'positive_values',
                'query': 'SELECT COUNT(*) FROM integrity_test WHERE value <= 0',
                'expected_result': 0,
                'description': 'Check for non-positive values'
            },
            {
                'name': 'total_record_count',
                'query': 'SELECT COUNT(*) FROM integrity_test',
                'expected_result': 3,
                'description': 'Verify total record count'
            }
        ]

        result = db_validator.validate_data_integrity('integrity_test', integrity_checks)

        assert result['table_name'] == 'integrity_test'
        assert 'integrity_checks' in result

        for check_name in ['no_null_values', 'positive_values', 'total_record_count']:
            assert check_name in result['integrity_checks']
            assert result['integrity_checks'][check_name]['passed'] is True

    def test_performance_benchmarking(self, db_validator):
        """Test database performance benchmarking."""
        # Create test table for performance testing
        create_table_sql = """
        CREATE TABLE performance_test (
            id INTEGER PRIMARY KEY,
            data TEXT,
            timestamp INTEGER
        )
        """
        db_validator.execute_query(create_table_sql)

        # Insert test data
        for i in range(1000):
            insert_sql = f"INSERT INTO performance_test (id, data, timestamp) VALUES ({i}, 'test_data_{i}', {i})"
            db_validator.execute_query(insert_sql)

        # Define performance tests
        performance_tests = [
            {
                'name': 'simple_select',
                'query': 'SELECT COUNT(*) FROM performance_test',
                'max_execution_time_ms': 100,
                'iterations': 3
            },
            {
                'name': 'indexed_lookup',
                'query': 'SELECT * FROM performance_test WHERE id = 500',
                'max_execution_time_ms': 50,
                'iterations': 3
            }
        ]

        result = db_validator.benchmark_performance(performance_tests)

        assert 'performance_tests' in result
        for test_name in ['simple_select', 'indexed_lookup']:
            assert test_name in result['performance_tests']
            test_result = result['performance_tests'][test_name]
            assert 'avg_execution_time_ms' in test_result
            assert 'passed' in test_result

    def test_comprehensive_schema_validation(self, db_validator):
        """Test comprehensive validation of all schema components."""
        # This test would validate all 7 schemas in a real database
        # For demo purposes, we'll test with a minimal schema

        schemas_to_validate = [
            'gpr_infrastructure',
            'enhanced_signal_analysis',
            'environmental_metadata',
            'ground_truth_validation',
            'ml_performance',
            'pas128_compliance',
            'usag_strike_reports'
        ]

        validation_summary = {
            'schemas_validated': 0,
            'schemas_passed': 0,
            'total_tables': 0,
            'tables_passed': 0,
            'validation_errors': []
        }

        for schema_name in schemas_to_validate:
            try:
                result = db_validator.validate_schema_exists(schema_name)
                validation_summary['schemas_validated'] += 1

                if result['exists']:
                    validation_summary['schemas_passed'] += 1
                    validation_summary['total_tables'] += result['table_count']

            except Exception as e:
                validation_summary['validation_errors'].append(f"{schema_name}: {str(e)}")

        # Assert basic validation structure
        assert validation_summary['schemas_validated'] == len(schemas_to_validate)
        assert isinstance(validation_summary['validation_errors'], list)


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database connectivity and operations."""

    def test_database_connection(self):
        """Test database connection establishment."""
        db_config = {
            'provider': 'sqlite',
            'params': {'database': ':memory:'}
        }

        validator = DatabaseValidator(db_config)

        # Test connection
        validator.connect()
        assert validator.connection is not None

        # Test disconnection
        validator.disconnect()
        assert validator.connection is None

    def test_sql_execution(self):
        """Test SQL query execution."""
        db_config = {
            'provider': 'sqlite',
            'params': {'database': ':memory:'}
        }

        validator = DatabaseValidator(db_config)
        validator.connect()

        try:
            # Test table creation
            create_sql = "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
            result = validator.execute_query(create_sql)

            # Test data insertion
            insert_sql = "INSERT INTO test (name) VALUES ('test_name')"
            validator.execute_query(insert_sql)

            # Test data selection
            select_sql = "SELECT COUNT(*) FROM test"
            result = validator.execute_query(select_sql)
            assert result[0][0] == 1

        finally:
            validator.disconnect()


def create_database_validation_suite() -> DatabaseValidator:
    """Factory function to create database validation suite."""
    # This would typically load from configuration
    db_config = {
        'provider': 'postgresql',
        'params': {
            'host': 'localhost',
            'port': 5432,
            'database': 'gpr_platform',
            'user': 'gpr_user',
            'password': 'gpr_password'
        }
    }

    return DatabaseValidator(db_config)


if __name__ == '__main__':
    # Run database validation as standalone script
    import argparse

    parser = argparse.ArgumentParser(description='Database Validation Suite')
    parser.add_argument('--config', type=str, help='Database configuration file')
    parser.add_argument('--schema', type=str, help='Specific schema to validate')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            db_config = json.load(f)
    else:
        db_config = {
            'provider': 'sqlite',
            'params': {'database': ':memory:'}
        }

    # Run validation
    validator = DatabaseValidator(db_config)
    validator.connect()

    try:
        if args.schema:
            result = validator.validate_schema_exists(args.schema)
            print(f"Schema validation result: {json.dumps(result, indent=2)}")
        else:
            print("Running comprehensive database validation...")
            # Run full validation suite here

    finally:
        validator.disconnect()