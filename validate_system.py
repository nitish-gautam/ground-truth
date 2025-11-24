#!/usr/bin/env python3
"""
Underground Utility Detection Platform - System Validation Script
================================================================

This script provides a simple way to validate that all components of the
Underground Utility Detection Platform are working correctly.

Usage:
    python validate_system.py                    # Quick health check
    python validate_system.py --full             # Complete validation
    python validate_system.py --component api    # Test specific component
    python validate_system.py --demo             # Interactive demo
"""

import sys
import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

class SystemValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }

    def check_prerequisites(self) -> bool:
        """Check if system prerequisites are met"""
        print_header("CHECKING SYSTEM PREREQUISITES")

        checks = []

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            print_success(f"Python version: {python_version.major}.{python_version.minor}")
            checks.append(True)
        else:
            print_error(f"Python version {python_version.major}.{python_version.minor} < 3.11 (required)")
            checks.append(False)

        # Check if tests directory exists
        if self.tests_dir.exists():
            print_success("Tests directory found")
            checks.append(True)
        else:
            print_error("Tests directory not found")
            checks.append(False)

        # Check if datasets exist
        datasets_dir = self.project_root / "datasets"
        if datasets_dir.exists():
            print_success("Datasets directory found")

            # Check for specific datasets
            twente_dir = datasets_dir / "raw" / "twente_gpr"
            mojahid_dir = datasets_dir / "raw" / "mojahid_images"

            if twente_dir.exists():
                print_success("Twente GPR dataset found")
            else:
                print_warning("Twente GPR dataset not found (optional for basic testing)")

            if mojahid_dir.exists():
                print_success("Mojahid images dataset found")
            else:
                print_warning("Mojahid images dataset not found (optional for basic testing)")

            checks.append(True)
        else:
            print_warning("Datasets directory not found (optional for basic testing)")
            checks.append(True)  # Not critical for basic validation

        # Check if backend directory exists
        backend_dir = self.project_root / "backend"
        if backend_dir.exists():
            print_success("Backend directory found")
            checks.append(True)
        else:
            print_error("Backend directory not found")
            checks.append(False)

        return all(checks)

    def run_quick_test(self) -> bool:
        """Run quick health check"""
        print_header("QUICK HEALTH CHECK")

        try:
            # Check if quick test script exists
            quick_test_script = self.tests_dir / "quick_test.py"
            if not quick_test_script.exists():
                print_error("Quick test script not found")
                return False

            print_info("Running quick health check...")
            result = subprocess.run([
                sys.executable, str(quick_test_script)
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print_success("Quick health check passed")
                self.results["tests"]["quick_check"] = {"status": "passed", "details": result.stdout}
                return True
            else:
                print_error("Quick health check failed")
                print(f"Error: {result.stderr}")
                self.results["tests"]["quick_check"] = {"status": "failed", "error": result.stderr}
                return False

        except subprocess.TimeoutExpired:
            print_error("Quick health check timed out")
            self.results["tests"]["quick_check"] = {"status": "failed", "error": "Timeout"}
            return False
        except Exception as e:
            print_error(f"Quick health check error: {str(e)}")
            self.results["tests"]["quick_check"] = {"status": "failed", "error": str(e)}
            return False

    def run_component_test(self, component: str) -> bool:
        """Run specific component test"""
        print_header(f"TESTING COMPONENT: {component.upper()}")

        component_scripts = {
            "database": "comprehensive_database_validation.py",
            "api": "comprehensive_api_test_suite.py",
            "data": "data_pipeline_validation.py",
            "ml": "ml_model_validation.py",
            "performance": "performance_benchmarking.py",
            "e2e": "end_to_end_system_tests.py"
        }

        if component not in component_scripts:
            print_error(f"Unknown component: {component}")
            print_info(f"Available components: {', '.join(component_scripts.keys())}")
            return False

        script_name = component_scripts[component]
        script_path = self.tests_dir / script_name

        if not script_path.exists():
            print_error(f"Component test script not found: {script_name}")
            return False

        try:
            print_info(f"Running {component} tests...")
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                print_success(f"{component} tests passed")
                self.results["tests"][component] = {"status": "passed", "details": result.stdout}
                return True
            else:
                print_error(f"{component} tests failed")
                print(f"Error: {result.stderr}")
                self.results["tests"][component] = {"status": "failed", "error": result.stderr}
                return False

        except subprocess.TimeoutExpired:
            print_error(f"{component} tests timed out")
            self.results["tests"][component] = {"status": "failed", "error": "Timeout"}
            return False
        except Exception as e:
            print_error(f"{component} test error: {str(e)}")
            self.results["tests"][component] = {"status": "failed", "error": str(e)}
            return False

    def run_full_validation(self) -> bool:
        """Run complete system validation"""
        print_header("FULL SYSTEM VALIDATION")

        try:
            master_script = self.tests_dir / "master_test_runner.py"
            if not master_script.exists():
                print_error("Master test runner not found")
                return False

            print_info("Running comprehensive validation (this may take 5-10 minutes)...")
            result = subprocess.run([
                sys.executable, str(master_script), "--all"
            ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout

            if result.returncode == 0:
                print_success("Full system validation passed")
                self.results["tests"]["full_validation"] = {"status": "passed", "details": result.stdout}
                return True
            else:
                print_error("Full system validation failed")
                print(f"Error: {result.stderr}")
                self.results["tests"]["full_validation"] = {"status": "failed", "error": result.stderr}
                return False

        except subprocess.TimeoutExpired:
            print_error("Full validation timed out")
            self.results["tests"]["full_validation"] = {"status": "failed", "error": "Timeout"}
            return False
        except Exception as e:
            print_error(f"Full validation error: {str(e)}")
            self.results["tests"]["full_validation"] = {"status": "failed", "error": str(e)}
            return False

    def run_interactive_demo(self):
        """Run interactive demonstration"""
        print_header("INTERACTIVE SYSTEM DEMO")

        print("This demo will guide you through validating the system step by step.\n")

        # Step 1: Prerequisites
        input("Press Enter to check system prerequisites...")
        if not self.check_prerequisites():
            print_error("Prerequisites check failed. Please fix issues before continuing.")
            return

        # Step 2: Quick test
        input("\nPress Enter to run quick health check...")
        if not self.run_quick_test():
            print_error("Quick health check failed.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return

        # Step 3: Component selection
        print("\nSelect components to test:")
        components = ["database", "api", "data", "ml", "performance", "e2e"]
        for i, comp in enumerate(components, 1):
            print(f"  {i}. {comp}")

        selection = input("\nEnter component numbers (comma-separated, or 'all'): ")

        if selection.lower() == 'all':
            selected_components = components
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_components = [components[i] for i in indices if 0 <= i < len(components)]
            except:
                print_error("Invalid selection")
                return

        # Run selected tests
        for component in selected_components:
            input(f"\nPress Enter to test {component}...")
            self.run_component_test(component)

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print validation summary"""
        print_header("VALIDATION SUMMARY")

        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() if test["status"] == "passed")

        print(f"Total tests run: {total_tests}")
        print(f"Tests passed: {passed_tests}")
        print(f"Tests failed: {total_tests - passed_tests}")

        if passed_tests == total_tests:
            print_success("🎉 ALL TESTS PASSED - SYSTEM IS WORKING CORRECTLY!")
        else:
            print_error("❌ SOME TESTS FAILED - CHECK LOGS FOR DETAILS")

        # Save results
        results_file = self.project_root / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print_info(f"Detailed results saved to: {results_file}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Underground Utility Detection Platform")
    parser.add_argument("--full", action="store_true", help="Run full validation suite")
    parser.add_argument("--component", choices=["database", "api", "data", "ml", "performance", "e2e"],
                       help="Test specific component")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")

    args = parser.parse_args()

    validator = SystemValidator()

    # Check prerequisites first
    if not validator.check_prerequisites():
        print_error("Prerequisites check failed. Please fix issues before running tests.")
        sys.exit(1)

    success = True

    if args.demo:
        validator.run_interactive_demo()
    elif args.full:
        success = validator.run_full_validation()
    elif args.component:
        success = validator.run_component_test(args.component)
    else:
        # Default: quick test
        success = validator.run_quick_test()

    validator.print_summary()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()