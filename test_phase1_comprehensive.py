#!/usr/bin/env python3
"""
Underground Utility Detection Platform - Phase 1 Comprehensive Test Suite
========================================================================

This script tests all Phase 1 endpoints with real data and displays responses.
Run this to validate your Phase 1 implementation.

Usage:
    python test_phase1_comprehensive.py
"""

import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8001"
TIMEOUT = 10.0

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def format_response(data: Any) -> str:
    """Format response data for display"""
    if isinstance(data, dict):
        if len(data) > 3:
            # Show first 3 keys for large objects
            preview = {k: v for i, (k, v) in enumerate(data.items()) if i < 3}
            return f"{json.dumps(preview, indent=2)}... ({len(data)} total fields)"
        return json.dumps(data, indent=2)
    elif isinstance(data, list):
        if len(data) > 2:
            return f"[{len(data)} items] First item: {json.dumps(data[0], indent=2) if data else 'None'}"
        return json.dumps(data, indent=2)
    else:
        return str(data)

def test_endpoint(method: str, endpoint: str, description: str, data: Dict = None) -> Dict[str, Any]:
    """Test a single endpoint"""
    print(f"{Colors.YELLOW}🔍 Testing: {description}{Colors.END}")
    print(f"   {method} {endpoint}")

    try:
        if method == 'GET':
            response = httpx.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
        elif method == 'POST':
            response = httpx.post(f"{BASE_URL}{endpoint}", json=data or {}, timeout=TIMEOUT)
        elif method == 'PUT':
            response = httpx.put(f"{BASE_URL}{endpoint}", json=data or {}, timeout=TIMEOUT)
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(f"   Status: {response.status_code}")

        if response.status_code < 400:
            print_success(f"Success: {description}")
            try:
                response_data = response.json()
                print(f"   Response: {format_response(response_data)}")
                return {"status": "success", "code": response.status_code, "data": response_data}
            except:
                print(f"   Response: {response.text[:200]}...")
                return {"status": "success", "code": response.status_code, "data": response.text}
        else:
            print_error(f"Failed: {description} (Status: {response.status_code})")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
                return {"status": "error", "code": response.status_code, "error": error_data}
            except:
                print(f"   Error: {response.text}")
                return {"status": "error", "code": response.status_code, "error": response.text}

    except Exception as e:
        print_error(f"Exception: {description} - {str(e)}")
        return {"status": "exception", "error": str(e)}

    print()

def main():
    """Run comprehensive Phase 1 tests"""
    print_header("🎉 Phase 1: Underground Utility Detection Platform Test Suite")

    print_info(f"Testing API at: {BASE_URL}")
    print_info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # =================================================================
    # 1. BASIC HEALTH CHECKS
    # =================================================================
    print_header("1. Basic Health Checks")

    basic_tests = [
        ('GET', '/', 'Root endpoint'),
        ('GET', '/health', 'Health check'),
        ('GET', '/api/v1/openapi.json', 'OpenAPI specification'),
    ]

    for method, endpoint, description in basic_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # =================================================================
    # 2. DATASET MANAGEMENT & REAL DATA
    # =================================================================
    print_header("2. Dataset Management & Real Data")

    dataset_tests = [
        ('GET', '/api/v1/datasets/info', 'Dataset information'),
        ('GET', '/api/v1/datasets/twente/status', 'Twente dataset status'),
        ('GET', '/api/v1/datasets/mojahid/status', 'Mojahid dataset status'),
        ('POST', '/api/v1/datasets/twente/load', 'Load Twente dataset'),
        ('POST', '/api/v1/datasets/mojahid/load', 'Load Mojahid dataset'),
    ]

    for method, endpoint, description in dataset_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # =================================================================
    # 3. GPR DATA PROCESSING
    # =================================================================
    print_header("3. GPR Data Processing")

    gpr_tests = [
        ('GET', '/api/v1/gpr/surveys', 'GPR surveys'),
        ('GET', '/api/v1/gpr/scans', 'GPR scans'),
        ('GET', '/api/v1/gpr/statistics', 'GPR statistics'),
    ]

    for method, endpoint, description in gpr_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # Create survey test data
    survey_data = {
        "name": "Test Survey",
        "location": "Test Location",
        "survey_date": "2024-01-15",
        "equipment": "400MHz GPR"
    }
    results["POST /api/v1/gpr/surveys"] = test_endpoint('POST', '/api/v1/gpr/surveys', 'Create GPR survey', survey_data)

    # =================================================================
    # 4. MATERIAL CLASSIFICATION (REAL DATA)
    # =================================================================
    print_header("4. Material Classification with Real Data")

    # Test material database
    results["GET materials"] = test_endpoint('GET', '/api/v1/material-classification/materials', 'Material database')

    # Test material prediction with realistic parameters
    material_predict_data = {
        "diameter_mm": 150,
        "depth_m": 1.2,
        "signal_frequency": 400,
        "ground_conditions": "clayey"
    }
    results["POST predict"] = test_endpoint('POST', '/api/v1/material-classification/predict',
                                          'Predict material type', material_predict_data)

    # Test material detectability analysis
    detectability_data = {
        "material_type": "cast_iron",
        "diameter_mm": 200,
        "environmental_factors": {
            "ground_type": "sandy",
            "moisture_level": "dry"
        }
    }
    results["POST analyze"] = test_endpoint('POST', '/api/v1/material-classification/analyze',
                                          'Analyze material detectability', detectability_data)

    # Test diameter correlation
    results["GET diameter-correlation"] = test_endpoint('GET',
        '/api/v1/material-classification/diameter-correlation?diameter_mm=150&include_optimal_parameters=true',
        'Diameter-material correlation')

    # Test discipline analysis
    results["GET discipline"] = test_endpoint('GET', '/api/v1/material-classification/discipline/water/analysis',
                                            'Water discipline analysis')

    # =================================================================
    # 5. PAS 128 COMPLIANCE AUTOMATION
    # =================================================================
    print_header("5. PAS 128 Compliance Automation")

    compliance_tests = [
        ('GET', '/api/v1/compliance/status', 'Compliance service status'),
        ('GET', '/api/v1/compliance/quality-levels/specifications', 'Quality level specifications'),
        ('GET', '/api/v1/compliance/benchmarks', 'Compliance benchmarks'),
    ]

    for method, endpoint, description in compliance_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # Test quality level determination
    quality_level_data = {
        "survey_data": {
            "detection_method": "gpr_400mhz",
            "verification_method": "em",
            "spatial_accuracy": 0.1,
            "coverage_completeness": 95.0
        }
    }
    results["POST quality-level"] = test_endpoint('POST', '/api/v1/compliance/quality-level/determine',
                                                'Determine quality level', quality_level_data)

    # =================================================================
    # 6. SIGNAL PROCESSING
    # =================================================================
    print_header("6. Signal Processing")

    # Test signal filtering
    filter_data = {
        "filter_type": "bandpass",
        "low_freq": 200,
        "high_freq": 800
    }
    results["POST filter"] = test_endpoint('POST', '/api/v1/processing/filter',
                                         'Apply signal filter', filter_data)

    # Test feature extraction
    feature_data = {
        "scan_id": "twente_scan_001",
        "feature_types": ["amplitude", "frequency", "depth"]
    }
    results["POST extract-features"] = test_endpoint('POST', '/api/v1/processing/extract-features',
                                                   'Extract features', feature_data)

    # =================================================================
    # 7. ENVIRONMENTAL ANALYSIS
    # =================================================================
    print_header("7. Environmental Analysis")

    environmental_tests = [
        ('GET', '/api/v1/environmental/conditions', 'Environmental conditions'),
        ('GET', '/api/v1/environmental/correlations', 'Environmental correlations'),
    ]

    for method, endpoint, description in environmental_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # =================================================================
    # 8. VALIDATION & GROUND TRUTH
    # =================================================================
    print_header("8. Validation & Ground Truth")

    validation_tests = [
        ('GET', '/api/v1/validation/ground-truth', 'Ground truth data'),
        ('GET', '/api/v1/validation/accuracy-metrics', 'Accuracy metrics'),
    ]

    for method, endpoint, description in validation_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # =================================================================
    # 9. ML ANALYTICS
    # =================================================================
    print_header("9. ML Analytics")

    analytics_tests = [
        ('GET', '/api/v1/analytics/models', 'ML models'),
        ('GET', '/api/v1/analytics/performance', 'Model performance'),
    ]

    for method, endpoint, description in analytics_tests:
        results[f"{method} {endpoint}"] = test_endpoint(method, endpoint, description)

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print_header("🎯 Phase 1 Test Results Summary")

    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    error_count = sum(1 for r in results.values() if r.get("status") == "error")
    exception_count = sum(1 for r in results.values() if r.get("status") == "exception")
    total_tests = len(results)

    print(f"📊 Total tests run: {total_tests}")
    print_success(f"Successful: {success_count}")
    if error_count > 0:
        print_error(f"HTTP Errors: {error_count}")
    if exception_count > 0:
        print_error(f"Exceptions: {exception_count}")

    success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n🏆 Success rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print_success("✅ Phase 1 validation PASSED! Your implementation is working excellently!")
    elif success_rate >= 60:
        print_info("⚠️  Phase 1 validation PARTIAL. Most components working, some need attention.")
    else:
        print_error("❌ Phase 1 validation FAILED. Significant issues found.")

    print_info(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =================================================================
    # DETAILED RESULTS & LOGGING
    # =================================================================
    print("\n" + "="*70)
    print("💾 Saving detailed results and logs...")

    # Create comprehensive log content
    log_content = []
    log_content.append("="*80)
    log_content.append("Phase 1 Underground Utility Detection Platform - Test Log")
    log_content.append("="*80)
    log_content.append(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"API Base URL: {BASE_URL}")
    log_content.append("")

    # Add summary to log
    log_content.append("SUMMARY:")
    log_content.append(f"Total tests: {total_tests}")
    log_content.append(f"Successful: {success_count}")
    log_content.append(f"HTTP Errors: {error_count}")
    log_content.append(f"Exceptions: {exception_count}")
    log_content.append(f"Success rate: {success_rate:.1f}%")
    log_content.append("")

    # Add detailed results to log
    log_content.append("DETAILED TEST RESULTS:")
    log_content.append("="*50)

    for endpoint, result in results.items():
        log_content.append(f"\n🔍 {endpoint}")
        log_content.append(f"Status: {result.get('status', 'unknown')}")

        if 'code' in result:
            log_content.append(f"HTTP Code: {result['code']}")

        if result.get('status') == 'success' and 'data' in result:
            log_content.append("Response Data:")
            try:
                if isinstance(result['data'], (dict, list)):
                    log_content.append(json.dumps(result['data'], indent=2, default=str))
                else:
                    log_content.append(str(result['data']))
            except:
                log_content.append(str(result['data']))

        elif 'error' in result:
            log_content.append("Error:")
            try:
                if isinstance(result['error'], dict):
                    log_content.append(json.dumps(result['error'], indent=2, default=str))
                else:
                    log_content.append(str(result['error']))
            except:
                log_content.append(str(result['error']))

        log_content.append("-" * 40)

    # Save JSON results
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "successful": success_count,
            "errors": error_count,
            "exceptions": exception_count,
            "success_rate": success_rate
        },
        "results": results
    }

    try:
        # Save JSON results
        with open("test_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print_success("✅ JSON results saved to test_results.json")

        # Save detailed log
        with open("test_phase1_detailed.log", "w") as f:
            f.write("\n".join(log_content))
        print_success("✅ Detailed log saved to test_phase1_detailed.log")

        # Save summary log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f"test_summary_{timestamp}.txt"
        with open(summary_filename, "w") as f:
            f.write(f"Phase 1 Test Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total tests: {total_tests}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"HTTP Errors: {error_count}\n")
            f.write(f"Exceptions: {exception_count}\n")
            f.write(f"Success rate: {success_rate:.1f}%\n\n")

            f.write("Working Endpoints:\n")
            for endpoint, result in results.items():
                if result.get('status') == 'success':
                    f.write(f"✅ {endpoint}\n")

            f.write("\nFailed Endpoints:\n")
            for endpoint, result in results.items():
                if result.get('status') != 'success':
                    f.write(f"❌ {endpoint} - {result.get('status')}\n")

        print_success(f"✅ Summary saved to {summary_filename}")

    except Exception as e:
        print_error(f"Failed to save results: {e}")

if __name__ == "__main__":
    print("🚀 Starting Phase 1 Underground Utility Detection Platform Test Suite...")
    print("⚡ Make sure your API is running on http://127.0.0.1:8001")

    # Quick connection test
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            print_success("✅ API connection successful!")
            main()
        else:
            print_error("❌ API not responding correctly. Please start the backend server.")
            sys.exit(1)
    except Exception as e:
        print_error(f"❌ Cannot connect to API: {e}")
        print_info("Please run: cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8001")
        sys.exit(1)