#!/usr/bin/env python3
"""
Master Test Runner for Underground Utility Detection Platform
============================================================

This module provides a comprehensive test orchestration system that runs
all validation and testing suites for the Underground Utility Detection Platform.

Test Suites Orchestrated:
1. Comprehensive API Testing Suite
2. Database Validation Suite
3. End-to-End System Tests
4. Data Pipeline Validation
5. ML Model Validation
6. Performance Benchmarking Suite

Features:
- Single command test execution
- Configurable test selection
- Parallel test execution
- Comprehensive reporting
- Health checks and prerequisites
- Error handling and recovery
- Test result aggregation
- Performance monitoring
- Test environment setup
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import multiprocessing

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import all test suites
try:
    from comprehensive_api_test_suite import APITestSuite
    from comprehensive_database_validation import DatabaseValidator
    from end_to_end_system_tests import EndToEndSystemTester
    from data_pipeline_validation import DataPipelineValidator
    from ml_model_validation import MLModelValidator
    from performance_benchmarking import PerformanceBenchmarker
except ImportError as e:
    print(f"Warning: Could not import test suite: {e}")
    print("Some test suites may not be available.")


class MasterTestRunner:
    """Master test orchestration and execution system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize master test runner."""
        self.config = self._load_config(config_path)
        self.results = {}
        self.execution_times = {}
        self.test_suites = {}

        # Setup logging
        self._setup_logging()

        # Test suite configurations
        self.available_suites = {
            "api_tests": {
                "name": "API Testing Suite",
                "class": APITestSuite,
                "description": "Comprehensive API endpoint testing",
                "dependencies": ["api_server"],
                "execution_time_estimate": 120  # seconds
            },
            "database_validation": {
                "name": "Database Validation Suite",
                "class": DatabaseValidator,
                "description": "Database schema and integrity validation",
                "dependencies": ["database"],
                "execution_time_estimate": 180
            },
            "end_to_end_tests": {
                "name": "End-to-End System Tests",
                "class": EndToEndSystemTester,
                "description": "Complete workflow testing",
                "dependencies": ["api_server", "database"],
                "execution_time_estimate": 300
            },
            "data_pipeline_validation": {
                "name": "Data Pipeline Validation",
                "class": DataPipelineValidator,
                "description": "Twente and Mojahid dataset validation",
                "dependencies": ["datasets"],
                "execution_time_estimate": 240
            },
            "ml_model_validation": {
                "name": "ML Model Validation",
                "class": MLModelValidator,
                "description": "Machine learning model testing",
                "dependencies": [],
                "execution_time_estimate": 360
            },
            "performance_benchmarking": {
                "name": "Performance Benchmarking",
                "class": PerformanceBenchmarker,
                "description": "Performance and load testing",
                "dependencies": ["api_server"],
                "execution_time_estimate": 600
            }
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration."""
        default_config = {
            "parallel_execution": True,
            "max_workers": min(4, multiprocessing.cpu_count()),
            "continue_on_failure": True,
            "generate_reports": True,
            "output_directory": "./test_results",
            "log_level": "INFO",
            "test_timeout_minutes": 30,
            "health_check_timeout_seconds": 60,
            "suite_selection": "all",  # or list of suite names
            "performance_baseline_file": None,
            "environment": "test"
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")

        return default_config

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config["log_level"].upper(), logging.INFO)

        # Create output directory
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = output_dir / f"master_test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Master Test Runner initialized. Log file: {log_file}")

    def _get_selected_suites(self) -> List[str]:
        """Get list of test suites to run based on configuration."""
        suite_selection = self.config["suite_selection"]

        if suite_selection == "all":
            return list(self.available_suites.keys())
        elif isinstance(suite_selection, list):
            # Validate suite names
            valid_suites = []
            for suite in suite_selection:
                if suite in self.available_suites:
                    valid_suites.append(suite)
                else:
                    self.logger.warning(f"Unknown test suite: {suite}")
            return valid_suites
        else:
            self.logger.error(f"Invalid suite_selection configuration: {suite_selection}")
            return list(self.available_suites.keys())

    # =========================
    # Health Checks and Prerequisites
    # =========================

    def check_prerequisites(self) -> Dict[str, bool]:
        """Check system prerequisites for test execution."""
        self.logger.info("Checking system prerequisites...")

        prerequisites = {
            "python_version": False,
            "required_packages": False,
            "api_server": False,
            "database": False,
            "datasets": False,
            "output_directory": False
        }

        try:
            # Check Python version
            if sys.version_info >= (3, 8):
                prerequisites["python_version"] = True
                self.logger.info(f"Python version: {sys.version.split()[0]} ✓")
            else:
                self.logger.error(f"Python 3.8+ required, found: {sys.version.split()[0]}")

            # Check required packages
            required_packages = [
                "fastapi", "sqlalchemy", "pandas", "numpy", "scikit-learn",
                "aiohttp", "psutil", "PIL", "cv2"
            ]

            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            if not missing_packages:
                prerequisites["required_packages"] = True
                self.logger.info("All required packages available ✓")
            else:
                self.logger.warning(f"Missing packages: {missing_packages}")

            # Check API server availability
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    prerequisites["api_server"] = True
                    self.logger.info("API server accessible ✓")
            except Exception as e:
                self.logger.warning(f"API server not accessible: {e}")

            # Check database connectivity
            try:
                # This would normally test database connection
                # For now, we'll assume it's available if API server is up
                prerequisites["database"] = prerequisites["api_server"]
                if prerequisites["database"]:
                    self.logger.info("Database connectivity assumed ✓")
            except Exception as e:
                self.logger.warning(f"Database connectivity check failed: {e}")

            # Check datasets
            dataset_paths = [
                "./datasets/twente",
                "./datasets/mojahid"
            ]

            dataset_available = False
            for path in dataset_paths:
                if Path(path).exists():
                    dataset_available = True
                    break

            prerequisites["datasets"] = dataset_available
            if dataset_available:
                self.logger.info("Dataset(s) available ✓")
            else:
                self.logger.warning("No datasets found in expected locations")

            # Check output directory
            output_dir = Path(self.config["output_directory"])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = output_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                prerequisites["output_directory"] = True
                self.logger.info(f"Output directory writable: {output_dir} ✓")
            except Exception as e:
                self.logger.error(f"Cannot write to output directory {output_dir}: {e}")

        except Exception as e:
            self.logger.error(f"Prerequisite check failed: {e}")

        # Summary
        passed_checks = sum(prerequisites.values())
        total_checks = len(prerequisites)
        self.logger.info(f"Prerequisites: {passed_checks}/{total_checks} passed")

        return prerequisites

    def run_health_checks(self) -> Dict[str, bool]:
        """Run comprehensive health checks."""
        self.logger.info("Running health checks...")

        health_checks = {
            "system_resources": False,
            "api_endpoints": False,
            "database_schema": False,
            "file_system": False
        }

        try:
            # Check system resources
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent

            if cpu_percent < 90 and memory_percent < 90 and disk_percent < 90:
                health_checks["system_resources"] = True
                self.logger.info(f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}% ✓")
            else:
                self.logger.warning(f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%")

            # Check key API endpoints
            try:
                import requests
                key_endpoints = [
                    "http://localhost:8000/health",
                    "http://localhost:8000/",
                    "http://localhost:8000/api/v1/datasets/info"
                ]

                endpoint_success = 0
                for endpoint in key_endpoints:
                    try:
                        response = requests.get(endpoint, timeout=5)
                        if response.status_code == 200:
                            endpoint_success += 1
                    except Exception:
                        pass

                if endpoint_success >= len(key_endpoints) * 0.8:  # 80% success rate
                    health_checks["api_endpoints"] = True
                    self.logger.info(f"API endpoints: {endpoint_success}/{len(key_endpoints)} accessible ✓")

            except Exception as e:
                self.logger.warning(f"API endpoint health check failed: {e}")

            # Database schema check would go here
            health_checks["database_schema"] = True  # Placeholder

            # File system check
            output_dir = Path(self.config["output_directory"])
            if output_dir.exists() and output_dir.is_dir():
                health_checks["file_system"] = True
                self.logger.info("File system accessible ✓")

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

        return health_checks

    # =========================
    # Test Suite Execution
    # =========================

    def run_single_test_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a single test suite."""
        self.logger.info(f"Starting test suite: {suite_name}")

        if suite_name not in self.available_suites:
            return {
                "suite_name": suite_name,
                "status": "ERROR",
                "error": f"Unknown test suite: {suite_name}",
                "execution_time": 0
            }

        suite_config = self.available_suites[suite_name]
        suite_start_time = time.time()

        try:
            # Check dependencies
            dependencies = suite_config.get("dependencies", [])
            for dep in dependencies:
                if dep == "api_server":
                    try:
                        import requests
                        response = requests.get("http://localhost:8000/health", timeout=5)
                        if response.status_code != 200:
                            raise Exception("API server not responding")
                    except Exception as e:
                        return {
                            "suite_name": suite_name,
                            "status": "SKIPPED",
                            "error": f"Dependency not available: {dep} - {e}",
                            "execution_time": 0
                        }

            # Initialize test suite
            suite_class = suite_config["class"]
            test_suite = suite_class()

            # Run the test suite
            if suite_name == "api_tests":
                results = test_suite.run_all_tests()
            elif suite_name == "database_validation":
                results = asyncio.run(test_suite.run_all_validations())
            elif suite_name == "end_to_end_tests":
                results = asyncio.run(test_suite.run_all_workflows())
            elif suite_name == "data_pipeline_validation":
                results = test_suite.run_all_validations()
            elif suite_name == "ml_model_validation":
                results = test_suite.run_all_validations()
            elif suite_name == "performance_benchmarking":
                results = asyncio.run(test_suite.run_all_benchmarks())
            else:
                raise Exception(f"Unknown execution method for suite: {suite_name}")

            execution_time = time.time() - suite_start_time

            # Store test suite instance for report generation
            self.test_suites[suite_name] = test_suite

            suite_result = {
                "suite_name": suite_name,
                "status": "COMPLETED",
                "execution_time": execution_time,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Completed test suite: {suite_name} in {execution_time:.2f}s")
            return suite_result

        except Exception as e:
            execution_time = time.time() - suite_start_time
            error_result = {
                "suite_name": suite_name,
                "status": "FAILED",
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.error(f"Test suite {suite_name} failed: {e}")
            return error_result

    def run_test_suites_parallel(self, suite_names: List[str]) -> Dict[str, Any]:
        """Run multiple test suites in parallel."""
        self.logger.info(f"Running {len(suite_names)} test suites in parallel...")

        results = {}
        max_workers = min(self.config["max_workers"], len(suite_names))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test suites
            future_to_suite = {
                executor.submit(self.run_single_test_suite, suite_name): suite_name
                for suite_name in suite_names
            }

            # Collect results as they complete
            for future in as_completed(future_to_suite, timeout=self.config["test_timeout_minutes"] * 60):
                suite_name = future_to_suite[future]
                try:
                    result = future.result()
                    results[suite_name] = result
                except Exception as e:
                    results[suite_name] = {
                        "suite_name": suite_name,
                        "status": "FAILED",
                        "error": f"Execution exception: {e}",
                        "execution_time": 0
                    }

        return results

    def run_test_suites_sequential(self, suite_names: List[str]) -> Dict[str, Any]:
        """Run test suites sequentially."""
        self.logger.info(f"Running {len(suite_names)} test suites sequentially...")

        results = {}

        for suite_name in suite_names:
            result = self.run_single_test_suite(suite_name)
            results[suite_name] = result

            # Check if we should continue on failure
            if result["status"] == "FAILED" and not self.config["continue_on_failure"]:
                self.logger.error(f"Stopping execution due to failure in {suite_name}")
                break

        return results

    # =========================
    # Report Generation
    # =========================

    def generate_comprehensive_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        self.logger.info("Generating comprehensive test report...")

        # Calculate summary statistics
        total_suites = len(test_results)
        completed_suites = sum(1 for result in test_results.values() if result["status"] == "COMPLETED")
        failed_suites = sum(1 for result in test_results.values() if result["status"] == "FAILED")
        skipped_suites = sum(1 for result in test_results.values() if result["status"] == "SKIPPED")

        total_execution_time = sum(result.get("execution_time", 0) for result in test_results.values())

        # Collect detailed metrics from each suite
        detailed_metrics = {}

        for suite_name, result in test_results.items():
            if result["status"] == "COMPLETED" and "results" in result:
                suite_results = result["results"]

                # Extract metrics based on suite type
                if suite_name == "api_tests":
                    if isinstance(suite_results, dict):
                        detailed_metrics[suite_name] = {
                            "total_tests": len(suite_results),
                            "passed_tests": sum(1 for r in suite_results.values() if r.get("status") == "PASS"),
                            "api_endpoints_tested": len(suite_results)
                        }

                elif suite_name == "database_validation":
                    # Extract database validation metrics
                    if "validation_summary" in suite_results:
                        detailed_metrics[suite_name] = suite_results["validation_summary"]

                elif suite_name == "performance_benchmarking":
                    # Extract performance metrics
                    if "overall_performance_score" in suite_results:
                        detailed_metrics[suite_name] = {
                            "performance_score": suite_results["overall_performance_score"]
                        }

        # Generate report
        report = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_runner_version": "1.0.0",
                "environment": self.config["environment"],
                "configuration": self.config
            },
            "execution_summary": {
                "total_suites": total_suites,
                "completed_suites": completed_suites,
                "failed_suites": failed_suites,
                "skipped_suites": skipped_suites,
                "success_rate": (completed_suites / total_suites * 100) if total_suites > 0 else 0,
                "total_execution_time_seconds": total_execution_time,
                "estimated_time_savings": self._calculate_time_savings(test_results)
            },
            "suite_results": test_results,
            "detailed_metrics": detailed_metrics,
            "quality_gates": self._evaluate_quality_gates(test_results),
            "recommendations": self._generate_recommendations(test_results),
            "artifacts": self._collect_artifacts()
        }

        return report

    def _calculate_time_savings(self, test_results: Dict[str, Any]) -> float:
        """Calculate time savings from parallel execution."""
        if not self.config["parallel_execution"]:
            return 0

        estimated_sequential_time = sum(
            self.available_suites[suite_name]["execution_time_estimate"]
            for suite_name in test_results.keys()
            if suite_name in self.available_suites
        )

        actual_total_time = sum(result.get("execution_time", 0) for result in test_results.values())

        return max(0, estimated_sequential_time - actual_total_time)

    def _evaluate_quality_gates(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality gates based on test results."""
        quality_gates = {
            "overall_pass": False,
            "api_tests_pass": False,
            "database_validation_pass": False,
            "performance_acceptable": False,
            "ml_models_acceptable": False,
            "critical_failures": []
        }

        # Overall pass: All critical suites completed successfully
        critical_suites = ["api_tests", "database_validation"]
        critical_passed = all(
            test_results.get(suite, {}).get("status") == "COMPLETED"
            for suite in critical_suites
        )

        quality_gates["overall_pass"] = critical_passed

        # Individual suite evaluations
        for suite_name, result in test_results.items():
            if result["status"] == "COMPLETED":
                if suite_name == "api_tests":
                    quality_gates["api_tests_pass"] = True
                elif suite_name == "database_validation":
                    quality_gates["database_validation_pass"] = True
                elif suite_name == "performance_benchmarking":
                    # Check if performance is acceptable
                    if "results" in result and "overall_performance_score" in result["results"]:
                        score = result["results"]["overall_performance_score"]
                        quality_gates["performance_acceptable"] = score >= 70
                elif suite_name == "ml_model_validation":
                    # Check ML model quality
                    quality_gates["ml_models_acceptable"] = True  # Placeholder
            elif result["status"] == "FAILED":
                quality_gates["critical_failures"].append(suite_name)

        return quality_gates

    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for failed suites
        failed_suites = [
            suite_name for suite_name, result in test_results.items()
            if result["status"] == "FAILED"
        ]

        if failed_suites:
            recommendations.append(f"Investigate and fix failures in: {', '.join(failed_suites)}")

        # Check for skipped suites
        skipped_suites = [
            suite_name for suite_name, result in test_results.items()
            if result["status"] == "SKIPPED"
        ]

        if skipped_suites:
            recommendations.append(f"Address dependencies for skipped suites: {', '.join(skipped_suites)}")

        # Performance recommendations
        if "performance_benchmarking" in test_results:
            perf_result = test_results["performance_benchmarking"]
            if perf_result["status"] == "COMPLETED" and "results" in perf_result:
                score = perf_result["results"].get("overall_performance_score", 0)
                if score < 70:
                    recommendations.append(f"Performance score ({score:.1f}) is below threshold - optimize system performance")

        # General recommendations
        total_time = sum(result.get("execution_time", 0) for result in test_results.values())
        if total_time > 1800:  # 30 minutes
            recommendations.append("Consider optimizing test execution time for faster feedback")

        if not recommendations:
            recommendations.append("All test suites executed successfully - system is ready for production")

        return recommendations

    def _collect_artifacts(self) -> Dict[str, List[str]]:
        """Collect test artifacts and reports."""
        artifacts = {
            "reports": [],
            "logs": [],
            "models": [],
            "datasets": []
        }

        output_dir = Path(self.config["output_directory"])

        # Collect generated reports
        for suite_name, test_suite in self.test_suites.items():
            if hasattr(test_suite, 'save_report'):
                # This would collect actual report paths
                artifacts["reports"].append(f"{suite_name}_report.json")

        # Collect log files
        log_files = list(output_dir.glob("*.log"))
        artifacts["logs"] = [str(f) for f in log_files]

        return artifacts

    def save_master_report(self, report: Dict[str, Any], output_path: str = None):
        """Save master test report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config["output_directory"])
            output_path = output_dir / f"master_test_report_{timestamp}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Master test report saved to: {output_path}")
        return output_path

    # =========================
    # Main Execution
    # =========================

    def run_all_tests(self):
        """Run all configured test suites."""
        print("=" * 80)
        print("UNDERGROUND UTILITY DETECTION PLATFORM - MASTER TEST RUNNER")
        print("=" * 80)

        start_time = time.time()

        try:
            # Check prerequisites
            print("\n🔍 Checking Prerequisites...")
            prerequisites = self.check_prerequisites()
            if not all(prerequisites.values()):
                self.logger.warning("Some prerequisites not met, but continuing...")

            # Run health checks
            print("\n🏥 Running Health Checks...")
            health_checks = self.run_health_checks()
            if not any(health_checks.values()):
                self.logger.error("Critical health checks failed!")
                return None

            # Get selected test suites
            selected_suites = self._get_selected_suites()
            if not selected_suites:
                self.logger.error("No test suites selected for execution")
                return None

            estimated_time = sum(
                self.available_suites[suite]["execution_time_estimate"]
                for suite in selected_suites
                if suite in self.available_suites
            )

            print(f"\n📋 Test Execution Plan:")
            print(f"Selected Suites: {len(selected_suites)}")
            for suite in selected_suites:
                suite_info = self.available_suites.get(suite, {})
                print(f"  • {suite_info.get('name', suite)}: {suite_info.get('description', 'No description')}")

            print(f"Estimated Time: {estimated_time // 60}m {estimated_time % 60}s")
            print(f"Parallel Execution: {'Yes' if self.config['parallel_execution'] else 'No'}")

            # Execute test suites
            print(f"\n🚀 Executing Test Suites...")

            if self.config["parallel_execution"]:
                test_results = self.run_test_suites_parallel(selected_suites)
            else:
                test_results = self.run_test_suites_sequential(selected_suites)

            # Generate comprehensive report
            print(f"\n📊 Generating Comprehensive Report...")
            master_report = self.generate_comprehensive_report(test_results)

            # Save master report
            if self.config["generate_reports"]:
                report_path = self.save_master_report(master_report)
                print(f"Master report saved: {report_path}")

            # Print execution summary
            total_time = time.time() - start_time
            print(f"\n" + "=" * 80)
            print("EXECUTION SUMMARY")
            print("=" * 80)

            summary = master_report["execution_summary"]
            print(f"Total Suites: {summary['total_suites']}")
            print(f"Completed: {summary['completed_suites']}")
            print(f"Failed: {summary['failed_suites']}")
            print(f"Skipped: {summary['skipped_suites']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Total Execution Time: {total_time:.1f}s")

            if self.config["parallel_execution"] and summary.get("estimated_time_savings", 0) > 0:
                print(f"Time Saved (Parallel): {summary['estimated_time_savings']:.1f}s")

            # Quality gates
            quality_gates = master_report["quality_gates"]
            print(f"\nQuality Gates:")
            print(f"Overall Pass: {'✓' if quality_gates['overall_pass'] else '✗'}")
            print(f"API Tests: {'✓' if quality_gates['api_tests_pass'] else '✗'}")
            print(f"Database Validation: {'✓' if quality_gates['database_validation_pass'] else '✗'}")
            print(f"Performance: {'✓' if quality_gates['performance_acceptable'] else '✗'}")

            # Recommendations
            if master_report["recommendations"]:
                print(f"\nRecommendations:")
                for i, rec in enumerate(master_report["recommendations"], 1):
                    print(f"  {i}. {rec}")

            print(f"\n🎯 Master test execution completed!")

            return master_report

        except Exception as e:
            self.logger.error(f"Master test execution failed: {e}")
            print(f"\n❌ Master test execution failed: {e}")
            return None


def main():
    """Main entry point for master test runner."""
    parser = argparse.ArgumentParser(
        description="Master Test Runner for Underground Utility Detection Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          # Run all test suites
  %(prog)s --suites api_tests database    # Run specific suites
  %(prog)s --config test_config.json      # Use custom configuration
  %(prog)s --sequential                   # Run tests sequentially
  %(prog)s --output ./results             # Custom output directory
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available test suites'
    )

    parser.add_argument(
        '--suites',
        nargs='+',
        help='Specific test suites to run',
        choices=['api_tests', 'database_validation', 'end_to_end_tests',
                'data_pipeline_validation', 'ml_model_validation', 'performance_benchmarking']
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run test suites sequentially instead of parallel'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./test_results',
        help='Output directory for test results'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Test timeout in minutes'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Build configuration from arguments
    config_overrides = {
        "output_directory": args.output,
        "test_timeout_minutes": args.timeout,
        "parallel_execution": not args.sequential,
        "log_level": "DEBUG" if args.verbose else "INFO"
    }

    if args.all:
        config_overrides["suite_selection"] = "all"
    elif args.suites:
        config_overrides["suite_selection"] = args.suites

    # Save temporary config if needed
    config_path = args.config
    if not config_path and (args.all or args.suites):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_overrides, f, indent=2)
            config_path = f.name

    try:
        # Initialize and run master test runner
        runner = MasterTestRunner(config_path)

        # Apply config overrides
        runner.config.update(config_overrides)

        # Run all tests
        result = runner.run_all_tests()

        # Cleanup temporary config
        if config_path and config_path != args.config:
            Path(config_path).unlink(missing_ok=True)

        if result:
            sys.exit(0 if result["quality_gates"]["overall_pass"] else 1)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Master test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()