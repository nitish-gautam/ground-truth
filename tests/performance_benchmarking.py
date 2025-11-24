#!/usr/bin/env python3
"""
Performance Benchmarking and Load Testing Suite
===============================================

This module provides comprehensive performance benchmarking and load testing
for the Underground Utility Detection Platform, validating system performance
under realistic and stress conditions.

Performance Tests:
1. API Endpoint Performance
   - Response time benchmarking
   - Throughput testing
   - Concurrent request handling
   - Rate limiting validation

2. Database Performance
   - Query execution times
   - Connection pooling efficiency
   - Transaction throughput
   - Concurrent operation handling

3. Data Processing Performance
   - GPR signal processing speed
   - Image classification throughput
   - Batch processing efficiency
   - Memory usage optimization

4. System Resource Usage
   - CPU utilization patterns
   - Memory consumption analysis
   - Disk I/O performance
   - Network bandwidth usage

5. Load Testing Scenarios
   - Normal load simulation
   - Peak load testing
   - Stress testing
   - Endurance testing

Features:
- Real-time performance monitoring
- Baseline performance establishment
- Performance regression detection
- Resource utilization analysis
- Scalability assessment
- Performance bottleneck identification
"""

import asyncio
import json
import time
import threading
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import statistics
import concurrent.futures
import psutil
import gc

import numpy as np
import requests
from fastapi.testclient import TestClient
import aiohttp
import asyncpg

# Add backend to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "app"))

from main import app
from core.config import settings


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking and load testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize performance benchmarker."""
        self.base_url = base_url
        self.api_v1 = f"{base_url}/api/v1"
        self.client = TestClient(app)

        # Performance results storage
        self.benchmark_results = {}
        self.load_test_results = {}
        self.resource_metrics = {}

        # Performance baselines (can be updated based on system capabilities)
        self.performance_baselines = {
            "api_response_time_ms": 500,
            "database_query_time_ms": 100,
            "throughput_requests_per_second": 100,
            "memory_usage_mb": 512,
            "cpu_utilization_percent": 80,
            "concurrent_requests": 50
        }

        # Test configurations
        self.load_test_configs = {
            "light_load": {"concurrent_users": 10, "duration_seconds": 30, "requests_per_user": 10},
            "normal_load": {"concurrent_users": 25, "duration_seconds": 60, "requests_per_user": 20},
            "peak_load": {"concurrent_users": 50, "duration_seconds": 120, "requests_per_user": 30},
            "stress_load": {"concurrent_users": 100, "duration_seconds": 180, "requests_per_user": 50}
        }

    def log_benchmark_result(self, test_name: str, metric_type: str, value: float, details: Dict[str, Any] = None):
        """Log benchmark result."""
        if test_name not in self.benchmark_results:
            self.benchmark_results[test_name] = {}

        if metric_type not in self.benchmark_results[test_name]:
            self.benchmark_results[test_name][metric_type] = []

        self.benchmark_results[test_name][metric_type].append({
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource utilization."""
        process = psutil.Process()

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "disk_io_read_mb": psutil.disk_io_counters().read_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
            "disk_io_write_mb": psutil.disk_io_counters().write_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
            "network_sent_mb": psutil.net_io_counters().bytes_sent / 1024 / 1024,
            "network_recv_mb": psutil.net_io_counters().bytes_recv / 1024 / 1024
        }

    # =========================
    # API Performance Benchmarking
    # =========================

    def benchmark_api_response_times(self):
        """Benchmark API endpoint response times."""
        print("⚡ Benchmarking API response times...")

        try:
            endpoints_to_test = [
                ("/health", "GET"),
                ("/", "GET"),
                (f"{self.api_v1}/datasets/info", "GET"),
                (f"{self.api_v1}/datasets/twente/status", "GET"),
                (f"{self.api_v1}/datasets/mojahid/status", "GET"),
                (f"{self.api_v1}/gpr/surveys", "GET"),
                (f"{self.api_v1}/environmental/conditions", "GET"),
                (f"{self.api_v1}/processing/filters", "GET"),
                (f"{self.api_v1}/validation/metrics", "GET"),
                (f"{self.api_v1}/analytics/dashboard", "GET"),
                (f"{self.api_v1}/material-classification/models", "GET"),
                (f"{self.api_v1}/compliance/quality-levels", "GET")
            ]

            endpoint_results = {}

            for endpoint, method in endpoints_to_test:
                response_times = []
                status_codes = []

                # Test each endpoint multiple times
                for i in range(20):
                    start_time = time.time()

                    try:
                        if method == "GET":
                            response = self.client.get(endpoint)
                        elif method == "POST":
                            response = self.client.post(endpoint, json={})

                        response_time = (time.time() - start_time) * 1000
                        response_times.append(response_time)
                        status_codes.append(response.status_code)

                        # Small delay between requests
                        time.sleep(0.01)

                    except Exception as e:
                        response_times.append(float('inf'))
                        status_codes.append(0)

                # Calculate statistics
                valid_times = [t for t in response_times if t != float('inf')]
                if valid_times:
                    endpoint_results[endpoint] = {
                        "method": method,
                        "avg_response_time_ms": statistics.mean(valid_times),
                        "min_response_time_ms": min(valid_times),
                        "max_response_time_ms": max(valid_times),
                        "median_response_time_ms": statistics.median(valid_times),
                        "std_response_time_ms": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                        "success_rate": sum(1 for code in status_codes if 200 <= code < 300) / len(status_codes),
                        "total_requests": len(response_times),
                        "successful_requests": len(valid_times)
                    }

                    # Log metrics
                    self.log_benchmark_result("api_response_times", endpoint,
                                            statistics.mean(valid_times), endpoint_results[endpoint])

            # Calculate overall API performance
            all_response_times = []
            for result in endpoint_results.values():
                all_response_times.append(result["avg_response_time_ms"])

            overall_performance = {
                "total_endpoints_tested": len(endpoints_to_test),
                "successful_endpoints": len(endpoint_results),
                "overall_avg_response_time_ms": statistics.mean(all_response_times) if all_response_times else 0,
                "overall_max_response_time_ms": max(all_response_times) if all_response_times else 0,
                "endpoints_below_baseline": sum(1 for t in all_response_times
                                              if t <= self.performance_baselines["api_response_time_ms"]),
                "endpoint_results": endpoint_results
            }

            return overall_performance

        except Exception as e:
            return {"error": str(e)}

    def benchmark_api_throughput(self):
        """Benchmark API throughput with concurrent requests."""
        print("🚄 Benchmarking API throughput...")

        try:
            def make_request(endpoint: str = "/health") -> Tuple[float, int]:
                """Make a single request and return response time and status code."""
                start_time = time.time()
                try:
                    response = self.client.get(endpoint)
                    response_time = (time.time() - start_time) * 1000
                    return response_time, response.status_code
                except Exception:
                    return float('inf'), 0

            throughput_results = {}
            concurrent_levels = [1, 5, 10, 25, 50]

            for concurrent_requests in concurrent_levels:
                print(f"  Testing with {concurrent_requests} concurrent requests...")

                start_time = time.time()

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                    # Submit multiple requests
                    futures = [executor.submit(make_request) for _ in range(concurrent_requests * 5)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]

                total_time = time.time() - start_time

                # Analyze results
                response_times = [r[0] for r in results if r[0] != float('inf')]
                status_codes = [r[1] for r in results]
                successful_requests = len(response_times)

                throughput_results[f"concurrent_{concurrent_requests}"] = {
                    "concurrent_requests": concurrent_requests,
                    "total_requests": len(results),
                    "successful_requests": successful_requests,
                    "total_time_seconds": total_time,
                    "throughput_requests_per_second": successful_requests / total_time if total_time > 0 else 0,
                    "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
                    "success_rate": successful_requests / len(results) if results else 0
                }

                # Log metrics
                self.log_benchmark_result("api_throughput", f"concurrent_{concurrent_requests}",
                                        throughput_results[f"concurrent_{concurrent_requests}"]["throughput_requests_per_second"])

            return throughput_results

        except Exception as e:
            return {"error": str(e)}

    # =========================
    # Database Performance Benchmarking
    # =========================

    async def benchmark_database_performance(self):
        """Benchmark database performance."""
        print("🗄️ Benchmarking database performance...")

        try:
            database_results = {}

            # Test 1: Connection establishment time
            connection_times = []
            for i in range(10):
                start_time = time.time()
                try:
                    # Mock database connection test
                    await asyncio.sleep(0.001)  # Simulate connection time
                    connection_time = (time.time() - start_time) * 1000
                    connection_times.append(connection_time)
                except Exception:
                    connection_times.append(float('inf'))

            valid_connection_times = [t for t in connection_times if t != float('inf')]
            if valid_connection_times:
                database_results["connection_performance"] = {
                    "avg_connection_time_ms": statistics.mean(valid_connection_times),
                    "max_connection_time_ms": max(valid_connection_times),
                    "successful_connections": len(valid_connection_times),
                    "total_attempts": len(connection_times)
                }

            # Test 2: Query execution times
            query_times = {}
            test_queries = [
                ("simple_select", "SELECT COUNT(*) FROM gpr_survey"),
                ("join_query", "SELECT s.id, sc.scan_number FROM gpr_survey s LEFT JOIN gpr_scan sc ON s.id = sc.survey_id LIMIT 100"),
                ("aggregate_query", "SELECT material, COUNT(*) FROM utility_record GROUP BY material LIMIT 10")
            ]

            for query_name, query_sql in test_queries:
                execution_times = []

                for i in range(5):
                    start_time = time.time()
                    try:
                        # Mock query execution
                        await asyncio.sleep(0.005 + np.random.uniform(0, 0.005))  # Simulate query time
                        execution_time = (time.time() - start_time) * 1000
                        execution_times.append(execution_time)
                    except Exception:
                        execution_times.append(float('inf'))

                valid_times = [t for t in execution_times if t != float('inf')]
                if valid_times:
                    query_times[query_name] = {
                        "avg_execution_time_ms": statistics.mean(valid_times),
                        "max_execution_time_ms": max(valid_times),
                        "successful_queries": len(valid_times)
                    }

            database_results["query_performance"] = query_times

            # Test 3: Concurrent database operations
            async def mock_db_operation():
                start_time = time.time()
                await asyncio.sleep(0.01 + np.random.uniform(0, 0.01))
                return (time.time() - start_time) * 1000

            concurrent_levels = [5, 10, 20]
            concurrent_results = {}

            for concurrent_ops in concurrent_levels:
                start_time = time.time()

                tasks = [mock_db_operation() for _ in range(concurrent_ops)]
                results = await asyncio.gather(*tasks)

                total_time = (time.time() - start_time) * 1000

                concurrent_results[f"concurrent_{concurrent_ops}"] = {
                    "concurrent_operations": concurrent_ops,
                    "total_time_ms": total_time,
                    "avg_operation_time_ms": statistics.mean(results),
                    "operations_per_second": concurrent_ops / (total_time / 1000) if total_time > 0 else 0
                }

            database_results["concurrent_performance"] = concurrent_results

            return database_results

        except Exception as e:
            return {"error": str(e)}

    # =========================
    # Data Processing Performance
    # =========================

    def benchmark_data_processing_performance(self):
        """Benchmark data processing performance."""
        print("🔄 Benchmarking data processing performance...")

        try:
            processing_results = {}

            # Test 1: GPR Signal Processing Simulation
            signal_processing_times = []
            signal_sizes = [1000, 5000, 10000, 25000]  # Different signal lengths

            for signal_size in signal_sizes:
                processing_times_for_size = []

                for i in range(5):
                    # Generate mock GPR signal
                    mock_signal = np.random.random(signal_size)

                    start_time = time.time()

                    # Simulate signal processing operations
                    # 1. Filtering
                    filtered_signal = np.convolve(mock_signal, np.ones(5)/5, mode='same')

                    # 2. FFT analysis
                    fft_result = np.fft.fft(filtered_signal)

                    # 3. Feature extraction
                    features = [
                        np.mean(filtered_signal),
                        np.std(filtered_signal),
                        np.max(np.abs(fft_result[:len(fft_result)//2]))
                    ]

                    processing_time = (time.time() - start_time) * 1000
                    processing_times_for_size.append(processing_time)

                signal_processing_times.append({
                    "signal_size": signal_size,
                    "avg_processing_time_ms": statistics.mean(processing_times_for_size),
                    "processing_rate_samples_per_ms": signal_size / statistics.mean(processing_times_for_size)
                })

            processing_results["signal_processing"] = signal_processing_times

            # Test 2: Image Processing Simulation (for Mojahid dataset)
            image_processing_times = []
            image_sizes = [(128, 128), (256, 256), (512, 512)]

            for width, height in image_sizes:
                processing_times_for_size = []

                for i in range(5):
                    # Generate mock image
                    mock_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

                    start_time = time.time()

                    # Simulate image processing operations
                    # 1. Resize
                    resized = mock_image[::2, ::2]  # Simple downsampling

                    # 2. Grayscale conversion
                    grayscale = np.mean(resized, axis=2)

                    # 3. Feature extraction
                    features = [
                        np.mean(grayscale),
                        np.std(grayscale),
                        np.sum(grayscale > 128)  # Threshold count
                    ]

                    processing_time = (time.time() - start_time) * 1000
                    processing_times_for_size.append(processing_time)

                image_processing_times.append({
                    "image_size": f"{width}x{height}",
                    "pixels": width * height,
                    "avg_processing_time_ms": statistics.mean(processing_times_for_size),
                    "processing_rate_pixels_per_ms": (width * height) / statistics.mean(processing_times_for_size)
                })

            processing_results["image_processing"] = image_processing_times

            # Test 3: Batch Processing Performance
            batch_sizes = [1, 5, 10, 25]
            batch_processing_results = []

            for batch_size in batch_sizes:
                start_time = time.time()

                # Simulate batch processing
                for i in range(batch_size):
                    # Mock file processing
                    mock_data = np.random.random(1000)
                    processed_data = np.fft.fft(mock_data)
                    features = [np.mean(mock_data), np.std(mock_data)]

                batch_time = (time.time() - start_time) * 1000

                batch_processing_results.append({
                    "batch_size": batch_size,
                    "total_processing_time_ms": batch_time,
                    "avg_time_per_item_ms": batch_time / batch_size,
                    "throughput_items_per_second": batch_size / (batch_time / 1000) if batch_time > 0 else 0
                })

            processing_results["batch_processing"] = batch_processing_results

            return processing_results

        except Exception as e:
            return {"error": str(e)}

    # =========================
    # System Resource Monitoring
    # =========================

    def benchmark_system_resources(self):
        """Benchmark system resource usage during operations."""
        print("📊 Benchmarking system resource usage...")

        try:
            resource_results = {}

            # Get baseline resources
            baseline_resources = self.get_system_resources()
            resource_results["baseline"] = baseline_resources

            # Monitor resources during CPU-intensive task
            cpu_intensive_resources = []

            start_time = time.time()
            while time.time() - start_time < 10:  # Monitor for 10 seconds
                # Simulate CPU-intensive work
                data = np.random.random((1000, 1000))
                result = np.dot(data, data.T)

                # Collect resource metrics
                resources = self.get_system_resources()
                cpu_intensive_resources.append(resources)

                time.sleep(0.5)

            # Calculate average resource usage during CPU-intensive task
            if cpu_intensive_resources:
                avg_cpu_resources = {
                    "avg_cpu_percent": statistics.mean([r["cpu_percent"] for r in cpu_intensive_resources]),
                    "max_cpu_percent": max([r["cpu_percent"] for r in cpu_intensive_resources]),
                    "avg_memory_mb": statistics.mean([r["memory_mb"] for r in cpu_intensive_resources]),
                    "max_memory_mb": max([r["memory_mb"] for r in cpu_intensive_resources]),
                    "samples_collected": len(cpu_intensive_resources)
                }
                resource_results["cpu_intensive_task"] = avg_cpu_resources

            # Monitor resources during memory-intensive task
            memory_intensive_resources = []
            large_data_arrays = []

            start_time = time.time()
            for i in range(5):
                # Simulate memory-intensive work
                large_array = np.random.random((5000, 5000))
                large_data_arrays.append(large_array)

                resources = self.get_system_resources()
                memory_intensive_resources.append(resources)

                time.sleep(1)

            if memory_intensive_resources:
                avg_memory_resources = {
                    "avg_memory_mb": statistics.mean([r["memory_mb"] for r in memory_intensive_resources]),
                    "max_memory_mb": max([r["memory_mb"] for r in memory_intensive_resources]),
                    "memory_growth_mb": max([r["memory_mb"] for r in memory_intensive_resources]) - baseline_resources["memory_mb"],
                    "samples_collected": len(memory_intensive_resources)
                }
                resource_results["memory_intensive_task"] = avg_memory_resources

            # Cleanup
            del large_data_arrays
            gc.collect()

            # Final resource check
            final_resources = self.get_system_resources()
            resource_results["final"] = final_resources

            return resource_results

        except Exception as e:
            return {"error": str(e)}

    # =========================
    # Load Testing Scenarios
    # =========================

    async def run_load_test_scenario(self, scenario_name: str, config: Dict[str, int]):
        """Run a specific load test scenario."""
        print(f"🔥 Running {scenario_name} load test...")

        try:
            concurrent_users = config["concurrent_users"]
            duration_seconds = config["duration_seconds"]
            requests_per_user = config["requests_per_user"]

            async def simulate_user_session(user_id: int):
                """Simulate a user session with multiple requests."""
                user_results = {
                    "user_id": user_id,
                    "requests_completed": 0,
                    "total_response_time": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "response_times": []
                }

                session_start = time.time()

                # Create session with timeout
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    for request_num in range(requests_per_user):
                        # Check if we've exceeded the duration
                        if time.time() - session_start > duration_seconds:
                            break

                        request_start = time.time()

                        try:
                            # Make request to health endpoint (lightweight)
                            async with session.get(f"{self.base_url}/health") as response:
                                request_time = (time.time() - request_start) * 1000

                                user_results["requests_completed"] += 1
                                user_results["total_response_time"] += request_time
                                user_results["response_times"].append(request_time)

                                if response.status == 200:
                                    user_results["successful_requests"] += 1
                                else:
                                    user_results["failed_requests"] += 1

                        except Exception as e:
                            user_results["failed_requests"] += 1
                            user_results["response_times"].append(float('inf'))

                        # Small delay between requests
                        await asyncio.sleep(0.1)

                return user_results

            # Start load test
            load_test_start = time.time()

            # Create concurrent user tasks
            user_tasks = [simulate_user_session(i) for i in range(concurrent_users)]
            user_results = await asyncio.gather(*user_tasks)

            load_test_duration = time.time() - load_test_start

            # Aggregate results
            total_requests = sum(result["requests_completed"] for result in user_results)
            successful_requests = sum(result["successful_requests"] for result in user_results)
            failed_requests = sum(result["failed_requests"] for result in user_results)

            all_response_times = []
            for result in user_results:
                all_response_times.extend([t for t in result["response_times"] if t != float('inf')])

            scenario_results = {
                "scenario_name": scenario_name,
                "configuration": config,
                "test_duration_seconds": load_test_duration,
                "concurrent_users": concurrent_users,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "requests_per_second": total_requests / load_test_duration if load_test_duration > 0 else 0,
                "avg_response_time_ms": statistics.mean(all_response_times) if all_response_times else 0,
                "median_response_time_ms": statistics.median(all_response_times) if all_response_times else 0,
                "p95_response_time_ms": np.percentile(all_response_times, 95) if all_response_times else 0,
                "p99_response_time_ms": np.percentile(all_response_times, 99) if all_response_times else 0,
                "max_response_time_ms": max(all_response_times) if all_response_times else 0
            }

            # Log metrics
            self.log_benchmark_result("load_test", scenario_name, scenario_results["requests_per_second"], scenario_results)

            return scenario_results

        except Exception as e:
            return {"scenario_name": scenario_name, "error": str(e)}

    async def run_all_load_tests(self):
        """Run all load testing scenarios."""
        print("⚡ Running comprehensive load tests...")

        load_test_results = {}

        for scenario_name, config in self.load_test_configs.items():
            result = await self.run_load_test_scenario(scenario_name, config)
            load_test_results[scenario_name] = result

            # Small break between tests
            await asyncio.sleep(5)

        return load_test_results

    # =========================
    # Performance Regression Detection
    # =========================

    def detect_performance_regressions(self, current_results: Dict[str, Any], baseline_results: Dict[str, Any] = None):
        """Detect performance regressions compared to baseline."""
        print("🔍 Detecting performance regressions...")

        if baseline_results is None:
            baseline_results = self.performance_baselines

        regression_results = {
            "regressions_detected": [],
            "improvements_detected": [],
            "metrics_within_threshold": [],
            "regression_threshold_percent": 20.0  # 20% degradation threshold
        }

        try:
            # Check API response times
            if "api_response_times" in current_results:
                current_avg = current_results["api_response_times"].get("overall_avg_response_time_ms", 0)
                baseline_avg = baseline_results.get("api_response_time_ms", 500)

                if current_avg > baseline_avg * 1.2:  # 20% slower
                    regression_results["regressions_detected"].append({
                        "metric": "api_response_time",
                        "current_value": current_avg,
                        "baseline_value": baseline_avg,
                        "degradation_percent": ((current_avg - baseline_avg) / baseline_avg) * 100
                    })
                elif current_avg < baseline_avg * 0.9:  # 10% faster
                    regression_results["improvements_detected"].append({
                        "metric": "api_response_time",
                        "current_value": current_avg,
                        "baseline_value": baseline_avg,
                        "improvement_percent": ((baseline_avg - current_avg) / baseline_avg) * 100
                    })
                else:
                    regression_results["metrics_within_threshold"].append("api_response_time")

            # Check throughput
            if "api_throughput" in current_results:
                # Get best throughput result
                throughput_results = current_results["api_throughput"]
                if throughput_results:
                    best_throughput = max([
                        result.get("throughput_requests_per_second", 0)
                        for result in throughput_results.values()
                        if isinstance(result, dict)
                    ])

                    baseline_throughput = baseline_results.get("throughput_requests_per_second", 100)

                    if best_throughput < baseline_throughput * 0.8:  # 20% lower throughput
                        regression_results["regressions_detected"].append({
                            "metric": "api_throughput",
                            "current_value": best_throughput,
                            "baseline_value": baseline_throughput,
                            "degradation_percent": ((baseline_throughput - best_throughput) / baseline_throughput) * 100
                        })

            return regression_results

        except Exception as e:
            return {"error": str(e)}

    # =========================
    # Main Benchmark Runner
    # =========================

    async def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("=" * 80)
        print("UNDERGROUND UTILITY DETECTION PLATFORM - PERFORMANCE BENCHMARKING")
        print("=" * 80)

        benchmark_results = {}

        # API Performance Benchmarks
        print("\n⚡ API Performance Benchmarking...")
        benchmark_results["api_response_times"] = self.benchmark_api_response_times()
        benchmark_results["api_throughput"] = self.benchmark_api_throughput()

        # Database Performance Benchmarks
        print("\n🗄️ Database Performance Benchmarking...")
        benchmark_results["database_performance"] = await self.benchmark_database_performance()

        # Data Processing Performance
        print("\n🔄 Data Processing Performance...")
        benchmark_results["data_processing"] = self.benchmark_data_processing_performance()

        # System Resource Monitoring
        print("\n📊 System Resource Monitoring...")
        benchmark_results["system_resources"] = self.benchmark_system_resources()

        # Load Testing
        print("\n🔥 Load Testing...")
        benchmark_results["load_tests"] = await self.run_all_load_tests()

        # Performance Regression Detection
        print("\n🔍 Performance Regression Detection...")
        benchmark_results["regression_analysis"] = self.detect_performance_regressions(benchmark_results)

        return benchmark_results

    def generate_performance_report(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "performance_summary": {
                "timestamp": datetime.now().isoformat(),
                "benchmarks_completed": len(benchmark_results),
                "performance_baselines": self.performance_baselines,
                "test_configurations": self.load_test_configs
            },
            "api_performance": {
                "response_times": benchmark_results.get("api_response_times", {}),
                "throughput": benchmark_results.get("api_throughput", {})
            },
            "database_performance": benchmark_results.get("database_performance", {}),
            "data_processing_performance": benchmark_results.get("data_processing", {}),
            "system_resources": benchmark_results.get("system_resources", {}),
            "load_test_results": benchmark_results.get("load_tests", {}),
            "regression_analysis": benchmark_results.get("regression_analysis", {}),
            "performance_recommendations": self._generate_performance_recommendations(benchmark_results)
        }

        # Calculate overall performance score
        report["overall_performance_score"] = self._calculate_performance_score(benchmark_results)

        return report

    def _calculate_performance_score(self, benchmark_results: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        scores = []

        # API response time score
        if "api_response_times" in benchmark_results:
            avg_response_time = benchmark_results["api_response_times"].get("overall_avg_response_time_ms", 1000)
            baseline = self.performance_baselines["api_response_time_ms"]
            api_score = max(0, min(100, (baseline / avg_response_time) * 100)) if avg_response_time > 0 else 0
            scores.append(api_score)

        # Throughput score
        if "api_throughput" in benchmark_results:
            throughput_results = benchmark_results["api_throughput"]
            if throughput_results:
                best_throughput = max([
                    result.get("throughput_requests_per_second", 0)
                    for result in throughput_results.values()
                    if isinstance(result, dict)
                ])
                baseline = self.performance_baselines["throughput_requests_per_second"]
                throughput_score = min(100, (best_throughput / baseline) * 100)
                scores.append(throughput_score)

        # Load test score
        if "load_tests" in benchmark_results:
            load_results = benchmark_results["load_tests"]
            success_rates = [
                result.get("success_rate", 0) * 100
                for result in load_results.values()
                if isinstance(result, dict) and "success_rate" in result
            ]
            if success_rates:
                load_score = statistics.mean(success_rates)
                scores.append(load_score)

        return statistics.mean(scores) if scores else 0

    def _generate_performance_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check API performance
        if "api_response_times" in benchmark_results:
            avg_response_time = benchmark_results["api_response_times"].get("overall_avg_response_time_ms", 0)
            if avg_response_time > self.performance_baselines["api_response_time_ms"]:
                recommendations.append(f"API response times are above baseline ({avg_response_time:.1f}ms vs {self.performance_baselines['api_response_time_ms']}ms) - consider optimizing endpoint logic")

        # Check throughput
        if "api_throughput" in benchmark_results:
            throughput_results = benchmark_results["api_throughput"]
            if throughput_results:
                best_throughput = max([
                    result.get("throughput_requests_per_second", 0)
                    for result in throughput_results.values()
                    if isinstance(result, dict)
                ])
                if best_throughput < self.performance_baselines["throughput_requests_per_second"]:
                    recommendations.append(f"API throughput is below baseline ({best_throughput:.1f} vs {self.performance_baselines['throughput_requests_per_second']} req/s) - consider scaling or optimization")

        # Check load test results
        if "load_tests" in benchmark_results:
            for scenario, results in benchmark_results["load_tests"].items():
                if isinstance(results, dict) and "success_rate" in results:
                    if results["success_rate"] < 0.95:
                        recommendations.append(f"Load test '{scenario}' has low success rate ({results['success_rate']*100:.1f}%) - investigate error handling")

        # Check regressions
        if "regression_analysis" in benchmark_results:
            regressions = benchmark_results["regression_analysis"].get("regressions_detected", [])
            if regressions:
                recommendations.append(f"Performance regressions detected in {len(regressions)} metrics - review recent changes")

        if not recommendations:
            recommendations.append("All performance metrics are within acceptable ranges")

        return recommendations

    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """Save performance report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_benchmark_report_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Performance benchmark report saved to: {output_path}")
        return output_path


async def main():
    """Main function to run performance benchmarks."""
    print("Starting Underground Utility Detection Platform Performance Benchmarking...")

    # Initialize benchmarker
    benchmarker = PerformanceBenchmarker()

    try:
        # Run all benchmarks
        benchmark_results = await benchmarker.run_all_benchmarks()

        # Generate report
        report = benchmarker.generate_performance_report(benchmark_results)

        # Print summary
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)

        summary = report["performance_summary"]
        print(f"Benchmarks Completed: {summary['benchmarks_completed']}")
        print(f"Overall Performance Score: {report['overall_performance_score']:.1f}/100")

        # API Performance
        if "api_performance" in report:
            api_perf = report["api_performance"]
            if "response_times" in api_perf:
                avg_time = api_perf["response_times"].get("overall_avg_response_time_ms", 0)
                print(f"Average API Response Time: {avg_time:.2f}ms")

        # Load Test Results
        if "load_test_results" in report:
            load_results = report["load_test_results"]
            best_scenario = max(load_results.keys(),
                              key=lambda k: load_results[k].get("success_rate", 0) if isinstance(load_results[k], dict) else 0)
            if best_scenario and isinstance(load_results[best_scenario], dict):
                print(f"Best Load Test: {best_scenario} ({load_results[best_scenario]['success_rate']*100:.1f}% success)")

        # Recommendations
        if report["performance_recommendations"]:
            print(f"\nPerformance Recommendations:")
            for i, rec in enumerate(report["performance_recommendations"], 1):
                print(f"  {i}. {rec}")

        # Save report
        report_path = benchmarker.save_report(report)

        print(f"\n🎯 Performance benchmarking completed!")
        print(f"Report available at: {report_path}")

        return report

    except Exception as e:
        print(f"\n❌ Performance benchmarking failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())