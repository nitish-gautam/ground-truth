"""
API endpoint testing suite for the Underground Utility Detection Platform.

This module provides comprehensive testing for all FastAPI endpoints.
"""

from .test_api_endpoints import APIEndpointTester, create_api_endpoint_test_suite

__all__ = ['APIEndpointTester', 'create_api_endpoint_test_suite']