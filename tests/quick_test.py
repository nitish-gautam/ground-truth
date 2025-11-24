#!/usr/bin/env python3
"""
Quick Test Script for Underground Utility Detection Platform
============================================================

This script provides a quick health check and basic validation of the
Underground Utility Detection Platform. Use this for rapid development
cycle testing and basic system validation.

Features:
- Fast execution (< 2 minutes)
- Essential component testing
- Basic API endpoint validation
- Quick database connectivity check
- System resource validation
- Simple pass/fail reporting
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))


class QuickTestRunner:
    """Quick test runner for essential system validation."""

    def __init__(self):
        """Initialize quick test runner."""
        self.results = {}
        self.start_time = time.time()

    def test_api_health(self):
        """Test basic API health endpoints."""
        print("🔍 Testing API Health...")

        endpoints = [
            "http://localhost:8000/health",
            "http://localhost:8000/",
            "http://localhost:8000/api/v1/datasets/info"
        ]

        passed = 0
        total = len(endpoints)

        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    passed += 1
                    print(f"  ✅ {endpoint}")
                else:
                    print(f"  ❌ {endpoint} (HTTP {response.status_code})")
            except Exception as e:
                print(f"  ❌ {endpoint} (Error: {e})")

        self.results["api_health"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0
        }

        print(f"   API Health: {passed}/{total} endpoints OK\n")

    def test_system_resources(self):
        """Test system resource availability."""
        print("💻 Testing System Resources...")

        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Check thresholds
            cpu_ok = cpu_percent < 90
            memory_ok = memory.percent < 90
            disk_ok = disk.percent < 90

            print(f"  CPU Usage: {cpu_percent:.1f}% {'✅' if cpu_ok else '⚠️'}")
            print(f"  Memory Usage: {memory.percent:.1f}% {'✅' if memory_ok else '⚠️'}")
            print(f"  Disk Usage: {disk.percent:.1f}% {'✅' if disk_ok else '⚠️'}")

            self.results["system_resources"] = {
                "cpu_ok": cpu_ok,
                "memory_ok": memory_ok,
                "disk_ok": disk_ok,
                "all_ok": cpu_ok and memory_ok and disk_ok
            }

        except ImportError:
            print("  ⚠️ psutil not available - skipping resource check")
            self.results["system_resources"] = {"all_ok": True}

        print()

    def test_python_environment(self):
        """Test Python environment and key dependencies."""
        print("🐍 Testing Python Environment...")

        # Check Python version
        python_version_ok = sys.version_info >= (3, 8)
        print(f"  Python Version: {sys.version.split()[0]} {'✅' if python_version_ok else '❌'}")

        # Check key packages
        packages = [
            "fastapi",
            "sqlalchemy",
            "pandas",
            "numpy",
            "requests"
        ]

        packages_ok = 0
        for package in packages:
            try:
                __import__(package)
                print(f"  {package}: ✅")
                packages_ok += 1
            except ImportError:
                print(f"  {package}: ❌")

        self.results["python_environment"] = {
            "python_version_ok": python_version_ok,
            "packages_ok": packages_ok,
            "total_packages": len(packages),
            "all_ok": python_version_ok and packages_ok == len(packages)
        }

        print()

    def test_file_system(self):
        """Test file system access."""
        print("📁 Testing File System...")

        # Test write access
        try:
            test_dir = Path("./test_results")
            test_dir.mkdir(exist_ok=True)

            test_file = test_dir / "quick_test.tmp"
            test_file.write_text(f"Test file created at {datetime.now()}")

            # Read back
            content = test_file.read_text()

            # Clean up
            test_file.unlink()

            print("  Write/Read Access: ✅")
            filesystem_ok = True

        except Exception as e:
            print(f"  Write/Read Access: ❌ ({e})")
            filesystem_ok = False

        self.results["file_system"] = {
            "access_ok": filesystem_ok
        }

        print()

    def test_basic_imports(self):
        """Test basic application imports."""
        print("📦 Testing Application Imports...")

        imports_ok = 0
        total_imports = 0

        # Test backend imports
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "app"))
            from main import app
            print("  FastAPI App: ✅")
            imports_ok += 1
        except Exception as e:
            print(f"  FastAPI App: ❌ ({e})")
        total_imports += 1

        # Test test suite imports
        try:
            from comprehensive_api_test_suite import APITestSuite
            print("  API Test Suite: ✅")
            imports_ok += 1
        except Exception as e:
            print(f"  API Test Suite: ❌ ({e})")
        total_imports += 1

        try:
            from master_test_runner import MasterTestRunner
            print("  Master Test Runner: ✅")
            imports_ok += 1
        except Exception as e:
            print(f"  Master Test Runner: ❌ ({e})")
        total_imports += 1

        self.results["imports"] = {
            "imports_ok": imports_ok,
            "total_imports": total_imports,
            "all_ok": imports_ok == total_imports
        }

        print()

    def run_quick_tests(self):
        """Run all quick tests."""
        print("=" * 60)
        print("UNDERGROUND UTILITY DETECTION PLATFORM - QUICK TEST")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run all tests
        self.test_python_environment()
        self.test_system_resources()
        self.test_file_system()
        self.test_basic_imports()
        self.test_api_health()

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate test summary."""
        total_time = time.time() - self.start_time

        print("=" * 60)
        print("QUICK TEST SUMMARY")
        print("=" * 60)

        # Overall status
        all_passed = all(
            result.get("all_ok", result.get("access_ok", True))
            for result in self.results.values()
        )

        print(f"Overall Status: {'✅ PASS' if all_passed else '❌ FAIL'}")
        print(f"Execution Time: {total_time:.2f} seconds")
        print()

        # Detailed results
        print("Test Details:")

        # Python Environment
        py_env = self.results.get("python_environment", {})
        print(f"  Python Environment: {'✅' if py_env.get('all_ok') else '❌'}")

        # System Resources
        sys_res = self.results.get("system_resources", {})
        print(f"  System Resources: {'✅' if sys_res.get('all_ok') else '⚠️'}")

        # File System
        fs = self.results.get("file_system", {})
        print(f"  File System Access: {'✅' if fs.get('access_ok') else '❌'}")

        # Imports
        imports = self.results.get("imports", {})
        if imports:
            print(f"  Application Imports: {'✅' if imports.get('all_ok') else '❌'} ({imports.get('imports_ok', 0)}/{imports.get('total_imports', 0)})")

        # API Health
        api = self.results.get("api_health", {})
        if api:
            print(f"  API Health: {'✅' if api.get('success_rate', 0) > 0.5 else '❌'} ({api.get('passed', 0)}/{api.get('total', 0)} endpoints)")

        print()

        # Recommendations
        print("Recommendations:")
        if not all_passed:
            if not py_env.get("all_ok"):
                print("  • Check Python version and install missing packages")
            if not sys_res.get("all_ok"):
                print("  • Monitor system resources - high usage detected")
            if not fs.get("access_ok"):
                print("  • Check file system permissions")
            if not imports.get("all_ok"):
                print("  • Review application dependencies and imports")
            if api.get("success_rate", 0) <= 0.5:
                print("  • Start the API server or check connectivity")
        else:
            print("  • System ready for comprehensive testing!")

        print()
        print("Next Steps:")
        if all_passed:
            print("  • Run full test suite: python master_test_runner.py --all")
            print("  • Run specific tests: python master_test_runner.py --suites api_tests")
        else:
            print("  • Fix identified issues and re-run quick test")
            print("  • Check system prerequisites and dependencies")

        return all_passed


def main():
    """Main entry point for quick test."""
    runner = QuickTestRunner()

    try:
        success = runner.run_quick_tests()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Quick test interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()