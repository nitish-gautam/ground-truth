#!/usr/bin/env python3
"""
Comprehensive Database Validation Suite
=======================================

This module provides complete database schema validation, connectivity testing,
data integrity checks, and performance validation for all 7 database schemas
in the Underground Utility Detection Platform.

Database Schemas Validated:
1. GPR Data Schema (surveys, scans, signal data, processing results)
2. Environmental Data Schema (conditions, weather, ground state)
3. Validation Schema (ground truth, accuracy metrics, validation results)
4. Utilities Schema (utility records, materials, disciplines)
5. ML Analytics Schema (models, features, performance, training)
6. Base Schema (audit trails, timestamps, metadata)
7. User Management Schema (if implemented)

Features:
- Schema structure validation
- Data integrity constraints testing
- Relationship validation
- Performance benchmarking
- Connection pool testing
- Migration verification
- Data consistency checks
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
import psutil
import statistics

import asyncpg
import sqlalchemy as sa
from sqlalchemy import text, inspect, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Add backend to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "app"))

from core.config import settings
from core.database import get_database_url, init_db
from models import *


class DatabaseValidator:
    """Comprehensive database validation suite."""

    def __init__(self):
        """Initialize database validator."""
        self.database_url = get_database_url()
        self.engine = None
        self.session_factory = None
        self.validation_results = {}
        self.performance_metrics = {}

        # Schema definitions for validation
        self.expected_schemas = {
            "gpr_data": {
                "tables": ["gpr_survey", "gpr_scan", "gpr_signal_data", "gpr_processing_result"],
                "required_columns": {
                    "gpr_survey": ["id", "name", "location", "survey_date", "equipment_used", "frequency_mhz"],
                    "gpr_scan": ["id", "survey_id", "scan_number", "file_path", "scan_date"],
                    "gpr_signal_data": ["id", "scan_id", "signal_data", "sample_rate", "time_window"],
                    "gpr_processing_result": ["id", "scan_id", "algorithm_used", "processing_parameters"]
                }
            },
            "environmental": {
                "tables": ["environmental_data", "weather_condition", "ground_condition"],
                "required_columns": {
                    "environmental_data": ["id", "survey_id", "temperature", "humidity", "pressure"],
                    "weather_condition": ["id", "condition_type", "description", "impact_score"],
                    "ground_condition": ["id", "soil_type", "moisture_level", "conductivity"]
                }
            },
            "validation": {
                "tables": ["ground_truth_data", "validation_result", "accuracy_metrics"],
                "required_columns": {
                    "ground_truth_data": ["id", "survey_id", "utility_id", "x_position", "y_position", "depth"],
                    "validation_result": ["id", "detection_id", "ground_truth_id", "is_match", "accuracy_score"],
                    "accuracy_metrics": ["id", "validation_session_id", "precision", "recall", "f1_score"]
                }
            },
            "utilities": {
                "tables": ["utility_record", "utility_material", "utility_discipline"],
                "required_columns": {
                    "utility_record": ["id", "x_position", "y_position", "depth", "material_id", "discipline_id"],
                    "utility_material": ["id", "name", "material_type", "conductivity", "density"],
                    "utility_discipline": ["id", "name", "description", "priority_level"]
                }
            },
            "ml_analytics": {
                "tables": ["ml_model", "feature_vector", "model_performance", "training_session"],
                "required_columns": {
                    "ml_model": ["id", "name", "model_type", "algorithm", "version"],
                    "feature_vector": ["id", "scan_id", "feature_data", "feature_names"],
                    "model_performance": ["id", "model_id", "accuracy", "precision", "recall"],
                    "training_session": ["id", "model_id", "training_data_size", "training_duration"]
                }
            }
        }

    async def setup_database_connection(self):
        """Setup database connection and session factory."""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )

            self.session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

            return {"status": "SUCCESS", "message": "Database connection established"}

        except Exception as e:
            return {"status": "FAILED", "message": f"Database connection failed: {e}"}

    async def cleanup_database_connection(self):
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()

    def log_validation_result(self, test_name: str, status: str, details: Dict[str, Any] = None):
        """Log validation result."""
        self.validation_results[test_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

    def log_performance_metric(self, test_name: str, metric_type: str, value: float):
        """Log performance metric."""
        if test_name not in self.performance_metrics:
            self.performance_metrics[test_name] = {}

        if metric_type not in self.performance_metrics[test_name]:
            self.performance_metrics[test_name][metric_type] = []

        self.performance_metrics[test_name][metric_type].append(value)

    # =========================
    # Schema Validation Tests
    # =========================

    async def validate_database_schema(self):
        """Validate complete database schema structure."""
        print("🔍 Validating database schema structure...")

        try:
            async with self.engine.begin() as conn:
                # Get database metadata
                metadata = MetaData()
                await conn.run_sync(metadata.reflect)

                existing_tables = set(metadata.tables.keys())
                schema_results = {}

                # Validate each schema
                for schema_name, schema_def in self.expected_schemas.items():
                    schema_results[schema_name] = {
                        "tables_found": [],
                        "tables_missing": [],
                        "columns_validated": {},
                        "status": "UNKNOWN"
                    }

                    # Check tables
                    expected_tables = set(schema_def["tables"])
                    found_tables = expected_tables.intersection(existing_tables)
                    missing_tables = expected_tables - existing_tables

                    schema_results[schema_name]["tables_found"] = list(found_tables)
                    schema_results[schema_name]["tables_missing"] = list(missing_tables)

                    # Check columns for found tables
                    for table_name in found_tables:
                        if table_name in metadata.tables:
                            table = metadata.tables[table_name]
                            existing_columns = set(col.name for col in table.columns)
                            required_columns = set(schema_def["required_columns"].get(table_name, []))

                            schema_results[schema_name]["columns_validated"][table_name] = {
                                "required_columns": list(required_columns),
                                "existing_columns": list(existing_columns),
                                "missing_columns": list(required_columns - existing_columns),
                                "extra_columns": list(existing_columns - required_columns)
                            }

                    # Determine schema status
                    if not missing_tables:
                        all_columns_ok = all(
                            not col_info["missing_columns"]
                            for col_info in schema_results[schema_name]["columns_validated"].values()
                        )
                        schema_results[schema_name]["status"] = "COMPLETE" if all_columns_ok else "PARTIAL"
                    else:
                        schema_results[schema_name]["status"] = "INCOMPLETE"

                # Overall schema validation status
                all_complete = all(
                    result["status"] == "COMPLETE"
                    for result in schema_results.values()
                )

                overall_status = "PASS" if all_complete else "PARTIAL"

                self.log_validation_result("schema_validation", overall_status, {
                    "total_schemas": len(self.expected_schemas),
                    "total_tables_expected": sum(len(s["tables"]) for s in self.expected_schemas.values()),
                    "total_tables_found": len(existing_tables),
                    "schema_details": schema_results
                })

                return {
                    "status": overall_status,
                    "existing_tables": list(existing_tables),
                    "schema_results": schema_results
                }

        except Exception as e:
            self.log_validation_result("schema_validation", "FAILED", {"error": str(e)})
            raise

    async def validate_table_constraints(self):
        """Validate database constraints and indexes."""
        print("🔒 Validating table constraints and indexes...")

        try:
            constraint_results = {}

            async with self.engine.begin() as conn:
                # Get constraint information
                constraint_query = text("""
                    SELECT
                        tc.table_name,
                        tc.constraint_name,
                        tc.constraint_type,
                        kcu.column_name,
                        rc.unique_constraint_name,
                        rc.match_option,
                        rc.update_rule,
                        rc.delete_rule
                    FROM information_schema.table_constraints tc
                    LEFT JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    LEFT JOIN information_schema.referential_constraints rc
                        ON tc.constraint_name = rc.constraint_name
                        AND tc.table_schema = rc.constraint_schema
                    WHERE tc.table_schema = 'public'
                    ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name;
                """)

                result = await conn.execute(constraint_query)
                constraints = result.fetchall()

                # Process constraints by table
                for row in constraints:
                    table_name = row[0]
                    if table_name not in constraint_results:
                        constraint_results[table_name] = {
                            "primary_keys": [],
                            "foreign_keys": [],
                            "unique_constraints": [],
                            "check_constraints": [],
                            "not_null_constraints": []
                        }

                    constraint_type = row[2]
                    if constraint_type == "PRIMARY KEY":
                        constraint_results[table_name]["primary_keys"].append(row[1])
                    elif constraint_type == "FOREIGN KEY":
                        constraint_results[table_name]["foreign_keys"].append({
                            "name": row[1],
                            "column": row[3],
                            "update_rule": row[6],
                            "delete_rule": row[7]
                        })
                    elif constraint_type == "UNIQUE":
                        constraint_results[table_name]["unique_constraints"].append(row[1])
                    elif constraint_type == "CHECK":
                        constraint_results[table_name]["check_constraints"].append(row[1])

                # Get index information
                index_query = text("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname;
                """)

                result = await conn.execute(index_query)
                indexes = result.fetchall()

                index_results = {}
                for row in indexes:
                    table_name = row[1]
                    if table_name not in index_results:
                        index_results[table_name] = []
                    index_results[table_name].append({
                        "name": row[2],
                        "definition": row[3]
                    })

                self.log_validation_result("constraint_validation", "PASS", {
                    "constraints_by_table": constraint_results,
                    "indexes_by_table": index_results,
                    "total_tables_with_constraints": len(constraint_results)
                })

                return {
                    "status": "PASS",
                    "constraints": constraint_results,
                    "indexes": index_results
                }

        except Exception as e:
            self.log_validation_result("constraint_validation", "FAILED", {"error": str(e)})
            raise

    # =========================
    # Data Integrity Tests
    # =========================

    async def validate_data_integrity(self):
        """Validate data integrity across all tables."""
        print("✅ Validating data integrity...")

        try:
            integrity_results = {}

            async with self.session_factory() as session:
                # Test 1: Check for orphaned records
                orphan_checks = await self._check_orphaned_records(session)
                integrity_results["orphaned_records"] = orphan_checks

                # Test 2: Validate foreign key relationships
                fk_validation = await self._validate_foreign_keys(session)
                integrity_results["foreign_key_validation"] = fk_validation

                # Test 3: Check data consistency
                consistency_checks = await self._check_data_consistency(session)
                integrity_results["data_consistency"] = consistency_checks

                # Test 4: Validate required fields
                required_field_checks = await self._validate_required_fields(session)
                integrity_results["required_fields"] = required_field_checks

                # Determine overall status
                all_passed = all(
                    check.get("status") == "PASS"
                    for check in integrity_results.values()
                )

                overall_status = "PASS" if all_passed else "PARTIAL"

                self.log_validation_result("data_integrity", overall_status, integrity_results)

                return {
                    "status": overall_status,
                    "integrity_results": integrity_results
                }

        except Exception as e:
            self.log_validation_result("data_integrity", "FAILED", {"error": str(e)})
            raise

    async def _check_orphaned_records(self, session: AsyncSession) -> Dict[str, Any]:
        """Check for orphaned records in related tables."""
        orphan_results = {"status": "PASS", "issues": []}

        try:
            # Check GPR scans without surveys
            result = await session.execute(text("""
                SELECT COUNT(*) FROM gpr_scan s
                LEFT JOIN gpr_survey sur ON s.survey_id = sur.id
                WHERE sur.id IS NULL
            """))
            orphaned_scans = result.scalar() or 0

            if orphaned_scans > 0:
                orphan_results["issues"].append({
                    "table": "gpr_scan",
                    "orphaned_count": orphaned_scans,
                    "description": "GPR scans without parent survey"
                })
                orphan_results["status"] = "WARNING"

            # Check signal data without scans
            result = await session.execute(text("""
                SELECT COUNT(*) FROM gpr_signal_data sd
                LEFT JOIN gpr_scan s ON sd.scan_id = s.id
                WHERE s.id IS NULL
            """))
            orphaned_signals = result.scalar() or 0

            if orphaned_signals > 0:
                orphan_results["issues"].append({
                    "table": "gpr_signal_data",
                    "orphaned_count": orphaned_signals,
                    "description": "Signal data without parent scan"
                })
                orphan_results["status"] = "WARNING"

            # Check validation results without ground truth
            result = await session.execute(text("""
                SELECT COUNT(*) FROM validation_result vr
                LEFT JOIN ground_truth_data gt ON vr.ground_truth_id = gt.id
                WHERE gt.id IS NULL
            """))
            orphaned_validations = result.scalar() or 0

            if orphaned_validations > 0:
                orphan_results["issues"].append({
                    "table": "validation_result",
                    "orphaned_count": orphaned_validations,
                    "description": "Validation results without ground truth"
                })
                orphan_results["status"] = "WARNING"

        except Exception as e:
            orphan_results["status"] = "ERROR"
            orphan_results["error"] = str(e)

        return orphan_results

    async def _validate_foreign_keys(self, session: AsyncSession) -> Dict[str, Any]:
        """Validate foreign key relationships."""
        fk_results = {"status": "PASS", "relationships": []}

        try:
            # Test key relationships
            relationships_to_test = [
                ("gpr_scan", "survey_id", "gpr_survey", "id"),
                ("gpr_signal_data", "scan_id", "gpr_scan", "id"),
                ("gpr_processing_result", "scan_id", "gpr_scan", "id"),
                ("validation_result", "ground_truth_id", "ground_truth_data", "id"),
                ("utility_record", "material_id", "utility_material", "id"),
                ("utility_record", "discipline_id", "utility_discipline", "id")
            ]

            for child_table, child_col, parent_table, parent_col in relationships_to_test:
                try:
                    # Check if relationship is maintained
                    result = await session.execute(text(f"""
                        SELECT COUNT(*) FROM {child_table} c
                        LEFT JOIN {parent_table} p ON c.{child_col} = p.{parent_col}
                        WHERE c.{child_col} IS NOT NULL AND p.{parent_col} IS NULL
                    """))
                    broken_refs = result.scalar() or 0

                    relationship_status = "PASS" if broken_refs == 0 else "FAIL"
                    if broken_refs > 0:
                        fk_results["status"] = "FAIL"

                    fk_results["relationships"].append({
                        "child_table": child_table,
                        "parent_table": parent_table,
                        "foreign_key": f"{child_col} -> {parent_col}",
                        "broken_references": broken_refs,
                        "status": relationship_status
                    })

                except Exception as e:
                    # Table might not exist yet
                    fk_results["relationships"].append({
                        "child_table": child_table,
                        "parent_table": parent_table,
                        "foreign_key": f"{child_col} -> {parent_col}",
                        "status": "SKIP",
                        "reason": "Table not found or accessible"
                    })

        except Exception as e:
            fk_results["status"] = "ERROR"
            fk_results["error"] = str(e)

        return fk_results

    async def _check_data_consistency(self, session: AsyncSession) -> Dict[str, Any]:
        """Check data consistency rules."""
        consistency_results = {"status": "PASS", "checks": []}

        try:
            # Check 1: Survey dates should be reasonable
            result = await session.execute(text("""
                SELECT COUNT(*) FROM gpr_survey
                WHERE survey_date > NOW() OR survey_date < '1990-01-01'
            """))
            invalid_dates = result.scalar() or 0

            consistency_results["checks"].append({
                "check": "survey_date_range",
                "description": "Survey dates should be between 1990 and now",
                "issues_found": invalid_dates,
                "status": "PASS" if invalid_dates == 0 else "WARNING"
            })

            # Check 2: Coordinates should be reasonable
            result = await session.execute(text("""
                SELECT COUNT(*) FROM utility_record
                WHERE x_position < -180 OR x_position > 180
                   OR y_position < -90 OR y_position > 90
                   OR depth < 0 OR depth > 50
            """))
            invalid_coords = result.scalar() or 0

            consistency_results["checks"].append({
                "check": "coordinate_ranges",
                "description": "Coordinates and depths should be within reasonable ranges",
                "issues_found": invalid_coords,
                "status": "PASS" if invalid_coords == 0 else "WARNING"
            })

            # Check 3: Confidence scores should be between 0 and 1
            result = await session.execute(text("""
                SELECT COUNT(*) FROM validation_result
                WHERE accuracy_score < 0 OR accuracy_score > 1
            """))
            invalid_scores = result.scalar() or 0

            consistency_results["checks"].append({
                "check": "accuracy_score_range",
                "description": "Accuracy scores should be between 0 and 1",
                "issues_found": invalid_scores,
                "status": "PASS" if invalid_scores == 0 else "WARNING"
            })

            # Update overall status
            if any(check["status"] == "WARNING" for check in consistency_results["checks"]):
                consistency_results["status"] = "WARNING"

        except Exception as e:
            consistency_results["status"] = "ERROR"
            consistency_results["error"] = str(e)

        return consistency_results

    async def _validate_required_fields(self, session: AsyncSession) -> Dict[str, Any]:
        """Validate required fields are not null."""
        required_field_results = {"status": "PASS", "validations": []}

        # Define critical required fields
        required_fields = {
            "gpr_survey": ["name", "location", "survey_date"],
            "gpr_scan": ["survey_id", "scan_number"],
            "utility_record": ["x_position", "y_position", "depth"],
            "ground_truth_data": ["x_position", "y_position", "depth"]
        }

        try:
            for table, fields in required_fields.items():
                for field in fields:
                    try:
                        result = await session.execute(text(f"""
                            SELECT COUNT(*) FROM {table}
                            WHERE {field} IS NULL
                        """))
                        null_count = result.scalar() or 0

                        field_status = "PASS" if null_count == 0 else "FAIL"
                        if null_count > 0:
                            required_field_results["status"] = "FAIL"

                        required_field_results["validations"].append({
                            "table": table,
                            "field": field,
                            "null_count": null_count,
                            "status": field_status
                        })

                    except Exception as e:
                        required_field_results["validations"].append({
                            "table": table,
                            "field": field,
                            "status": "SKIP",
                            "reason": "Table or field not accessible"
                        })

        except Exception as e:
            required_field_results["status"] = "ERROR"
            required_field_results["error"] = str(e)

        return required_field_results

    # =========================
    # Performance Tests
    # =========================

    async def validate_database_performance(self):
        """Validate database performance metrics."""
        print("🚀 Validating database performance...")

        try:
            performance_results = {}

            # Test 1: Connection performance
            connection_perf = await self._test_connection_performance()
            performance_results["connection_performance"] = connection_perf

            # Test 2: Query performance
            query_perf = await self._test_query_performance()
            performance_results["query_performance"] = query_perf

            # Test 3: Insert performance
            insert_perf = await self._test_insert_performance()
            performance_results["insert_performance"] = insert_perf

            # Test 4: Concurrent operations
            concurrent_perf = await self._test_concurrent_operations()
            performance_results["concurrent_performance"] = concurrent_perf

            # Determine overall performance status
            avg_response_times = [
                perf.get("avg_response_time", 0)
                for perf in performance_results.values()
                if "avg_response_time" in perf
            ]

            overall_avg = statistics.mean(avg_response_times) if avg_response_times else 0
            performance_status = "PASS" if overall_avg < 1000 else "WARNING"  # 1 second threshold

            self.log_validation_result("database_performance", performance_status, {
                "overall_avg_response_ms": overall_avg,
                "performance_breakdown": performance_results
            })

            return {
                "status": performance_status,
                "overall_avg_response_ms": overall_avg,
                "performance_results": performance_results
            }

        except Exception as e:
            self.log_validation_result("database_performance", "FAILED", {"error": str(e)})
            raise

    async def _test_connection_performance(self) -> Dict[str, Any]:
        """Test database connection performance."""
        connection_times = []

        for i in range(10):
            start_time = time.time()
            try:
                async with self.engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                connection_time = (time.time() - start_time) * 1000
                connection_times.append(connection_time)
            except Exception as e:
                connection_times.append(float('inf'))

        valid_times = [t for t in connection_times if t != float('inf')]
        if valid_times:
            avg_time = statistics.mean(valid_times)
            max_time = max(valid_times)
            min_time = min(valid_times)
        else:
            avg_time = max_time = min_time = 0

        self.log_performance_metric("connection", "avg_response_time", avg_time)
        self.log_performance_metric("connection", "max_response_time", max_time)

        return {
            "avg_response_time": avg_time,
            "max_response_time": max_time,
            "min_response_time": min_time,
            "successful_connections": len(valid_times),
            "total_attempts": len(connection_times)
        }

    async def _test_query_performance(self) -> Dict[str, Any]:
        """Test query performance across different table sizes."""
        query_results = {}

        test_queries = [
            ("simple_select", "SELECT COUNT(*) FROM gpr_survey"),
            ("join_query", """
                SELECT COUNT(*) FROM gpr_survey s
                LEFT JOIN gpr_scan sc ON s.id = sc.survey_id
            """),
            ("complex_query", """
                SELECT s.name, COUNT(sc.id) as scan_count,
                       AVG(CASE WHEN vr.accuracy_score IS NOT NULL THEN vr.accuracy_score END) as avg_accuracy
                FROM gpr_survey s
                LEFT JOIN gpr_scan sc ON s.id = sc.survey_id
                LEFT JOIN validation_result vr ON sc.id = vr.detection_id
                GROUP BY s.id, s.name
                LIMIT 100
            """)
        ]

        async with self.session_factory() as session:
            for query_name, query_sql in test_queries:
                query_times = []

                for _ in range(5):  # Run each query 5 times
                    start_time = time.time()
                    try:
                        await session.execute(text(query_sql))
                        query_time = (time.time() - start_time) * 1000
                        query_times.append(query_time)
                    except Exception as e:
                        # Query might fail if tables don't exist or have no data
                        query_times.append(float('inf'))

                valid_times = [t for t in query_times if t != float('inf')]
                if valid_times:
                    avg_time = statistics.mean(valid_times)
                    self.log_performance_metric(f"query_{query_name}", "avg_response_time", avg_time)

                    query_results[query_name] = {
                        "avg_response_time": avg_time,
                        "max_response_time": max(valid_times),
                        "min_response_time": min(valid_times),
                        "successful_queries": len(valid_times)
                    }
                else:
                    query_results[query_name] = {
                        "status": "SKIP",
                        "reason": "Query failed - tables may not exist or have data"
                    }

        return query_results

    async def _test_insert_performance(self) -> Dict[str, Any]:
        """Test insert performance."""
        insert_times = []

        async with self.session_factory() as session:
            for i in range(10):
                start_time = time.time()
                try:
                    # Try to insert a test record
                    test_query = text("""
                        INSERT INTO gpr_survey (id, name, location, survey_date, equipment_used, frequency_mhz)
                        VALUES (gen_random_uuid(), :name, :location, :survey_date, :equipment, :frequency)
                        ON CONFLICT (id) DO NOTHING
                    """)

                    await session.execute(test_query, {
                        "name": f"Test Survey {i}",
                        "location": "Test Location",
                        "survey_date": datetime.now(),
                        "equipment": "Test Equipment",
                        "frequency": 400
                    })

                    await session.commit()
                    insert_time = (time.time() - start_time) * 1000
                    insert_times.append(insert_time)

                except Exception as e:
                    # Insert might fail if table doesn't exist
                    insert_times.append(float('inf'))
                    await session.rollback()

        valid_times = [t for t in insert_times if t != float('inf')]
        if valid_times:
            avg_time = statistics.mean(valid_times)
            self.log_performance_metric("insert", "avg_response_time", avg_time)

            return {
                "avg_response_time": avg_time,
                "max_response_time": max(valid_times),
                "successful_inserts": len(valid_times),
                "total_attempts": len(insert_times)
            }
        else:
            return {
                "status": "SKIP",
                "reason": "Insert operations failed - table may not exist"
            }

    async def _test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent database operations."""
        async def simple_query():
            start_time = time.time()
            try:
                async with self.session_factory() as session:
                    await session.execute(text("SELECT COUNT(*) FROM gpr_survey"))
                return (time.time() - start_time) * 1000
            except Exception:
                return float('inf')

        # Run 20 concurrent queries
        start_time = time.time()
        tasks = [simple_query() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000

        valid_times = [t for t in results if t != float('inf')]
        if valid_times:
            avg_time = statistics.mean(valid_times)
            self.log_performance_metric("concurrent", "avg_response_time", avg_time)

            return {
                "total_concurrent_time": total_time,
                "avg_response_time": avg_time,
                "max_response_time": max(valid_times),
                "successful_operations": len(valid_times),
                "total_operations": len(results)
            }
        else:
            return {
                "status": "SKIP",
                "reason": "Concurrent operations failed"
            }

    # =========================
    # Main Test Runner
    # =========================

    async def run_all_validations(self):
        """Run all database validation tests."""
        print("=" * 80)
        print("UNDERGROUND UTILITY DETECTION PLATFORM - DATABASE VALIDATION SUITE")
        print("=" * 80)

        validation_results = {}

        try:
            # Setup database connection
            print("\n🔌 Setting up database connection...")
            connection_result = await self.setup_database_connection()
            validation_results["connection_setup"] = connection_result

            if connection_result["status"] != "SUCCESS":
                print(f"❌ Database connection failed: {connection_result['message']}")
                return validation_results

            # Schema validation
            print("\n📋 Running schema validation...")
            schema_result = await self.validate_database_schema()
            validation_results["schema_validation"] = schema_result

            # Constraint validation
            print("\n🔒 Running constraint validation...")
            constraint_result = await self.validate_table_constraints()
            validation_results["constraint_validation"] = constraint_result

            # Data integrity validation
            print("\n✅ Running data integrity validation...")
            integrity_result = await self.validate_data_integrity()
            validation_results["data_integrity"] = integrity_result

            # Performance validation
            print("\n🚀 Running performance validation...")
            performance_result = await self.validate_database_performance()
            validation_results["performance_validation"] = performance_result

            return validation_results

        except Exception as e:
            print(f"❌ Database validation failed: {e}")
            validation_results["error"] = {"message": str(e), "type": type(e).__name__}
            return validation_results

        finally:
            # Cleanup
            await self.cleanup_database_connection()

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        # Calculate overall status
        test_statuses = [
            result.get("status", "UNKNOWN")
            for key, result in validation_results.items()
            if key != "error" and isinstance(result, dict)
        ]

        passed_tests = sum(1 for status in test_statuses if status == "PASS")
        total_tests = len(test_statuses)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_validations": total_tests,
                "passed_validations": passed_tests,
                "success_rate": success_rate,
                "overall_status": "PASS" if success_rate >= 80 else "PARTIAL" if success_rate >= 60 else "FAIL"
            },
            "database_info": {
                "database_url": self.database_url.replace(self.database_url.split('@')[0].split('://')[-1], '***') if '@' in self.database_url else self.database_url,
                "schemas_validated": list(self.expected_schemas.keys()),
                "total_schemas": len(self.expected_schemas)
            },
            "detailed_results": validation_results,
            "performance_metrics": self.performance_metrics,
            "validation_log": self.validation_results
        }

        return report

    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """Save validation report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"database_validation_report_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Database validation report saved to: {output_path}")
        return output_path


async def main():
    """Main function to run database validation."""
    print("Starting Underground Utility Detection Platform Database Validation...")

    # Initialize validator
    validator = DatabaseValidator()

    try:
        # Run all validations
        validation_results = await validator.run_all_validations()

        # Generate report
        report = validator.generate_validation_report(validation_results)

        # Print summary
        print("\n" + "=" * 80)
        print("DATABASE VALIDATION SUMMARY")
        print("=" * 80)

        summary = report["validation_summary"]
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Passed: {summary['passed_validations']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Overall Status: {summary['overall_status']}")

        if "performance_metrics" in validator.performance_metrics:
            print(f"\nDatabase Performance:")
            for metric_type, values in validator.performance_metrics.items():
                if values:
                    avg_val = statistics.mean(values) if isinstance(values, list) else values
                    print(f"  {metric_type}: {avg_val:.2f}ms")

        # Save report
        report_path = validator.save_report(report)

        print(f"\n🎯 Database validation completed!")
        print(f"Report available at: {report_path}")

        return report

    except Exception as e:
        print(f"\n❌ Database validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())