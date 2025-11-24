"""
Global pytest configuration and fixtures for GPR validation testing framework.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import logging

# Add the backend app to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def ground_truth_data_dir():
    """Path to ground truth datasets directory."""
    return Path(__file__).parent.parent / "datasets" / "raw"


@pytest.fixture(scope="session")
def twente_metadata_path(ground_truth_data_dir):
    """Path to Twente GPR metadata CSV file."""
    return ground_truth_data_dir / "twente_gpr" / "Metadata.csv"


@pytest.fixture(scope="session")
def pas128_spec_path(ground_truth_data_dir):
    """Path to PAS 128 specification JSON file."""
    return ground_truth_data_dir / "pas128_docs" / "quality_levels_specification.json"


@pytest.fixture
def sample_gpr_detection():
    """Sample GPR detection result for testing."""
    return {
        "location_id": "01.1",
        "detected_utilities": [
            {
                "x_position": 10.5,
                "y_position": 20.3,
                "depth": 1.2,
                "material": "steel",
                "diameter": 200,
                "discipline": "water",
                "confidence": 0.85
            },
            {
                "x_position": 11.2,
                "y_position": 21.1,
                "depth": 0.8,
                "material": "polyVinylChloride",
                "diameter": 125,
                "discipline": "sewer",
                "confidence": 0.92
            }
        ],
        "environmental_conditions": {
            "weather": "Dry",
            "ground_condition": "Sandy",
            "permittivity": 9.0,
            "land_cover": "Brick road concrete"
        },
        "survey_metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "operator": "test_operator",
            "equipment": "test_gpr_system"
        }
    }


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth data for testing."""
    return {
        "location_id": "01.1",
        "true_utilities": [
            {
                "x_position": 10.2,
                "y_position": 20.1,
                "depth": 1.15,
                "material": "steel",
                "diameter": 200,
                "discipline": "water"
            },
            {
                "x_position": 11.0,
                "y_position": 21.0,
                "depth": 0.85,
                "material": "polyVinylChloride",
                "diameter": 125,
                "discipline": "sewer"
            }
        ],
        "environmental_conditions": {
            "weather": "Dry",
            "ground_condition": "Sandy",
            "permittivity": 9.0,
            "land_cover": "Brick road concrete",
            "terrain_levelling": "Flat",
            "terrain_smoothness": "Smooth"
        }
    }


@pytest.fixture
def pas128_quality_levels():
    """PAS 128 quality level specifications."""
    return {
        "QL-A": {"horizontal_accuracy": 300, "depth_accuracy": 300},
        "QL-B": {"horizontal_accuracy": 500, "depth_accuracy": None},
        "QL-C": {"horizontal_accuracy": 1000, "depth_accuracy": None},
        "QL-D": {"horizontal_accuracy": 2000, "depth_accuracy": None}
    }


@pytest.fixture
def environmental_test_conditions():
    """Environmental conditions for testing environmental factor validation."""
    return [
        {
            "weather": "Dry",
            "ground_condition": "Sandy",
            "permittivity": 9.0,
            "expected_performance": "high"
        },
        {
            "weather": "Rainy",
            "ground_condition": "Sandy",
            "permittivity": 14.0,
            "expected_performance": "medium"
        },
        {
            "weather": "Dry",
            "ground_condition": "Clayey",
            "permittivity": 16.0,
            "expected_performance": "low"
        }
    ]


@pytest.fixture
def mock_gpr_processor():
    """Mock GPR signal processor for testing."""
    mock = Mock()
    mock.process_gpr_data.return_value = {
        "detected_utilities": [],
        "confidence_scores": [],
        "processing_metadata": {
            "algorithm_version": "test_v1.0",
            "processing_time": 0.5
        }
    }
    return mock


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock = MagicMock()
    mock.execute.return_value = []
    mock.fetchall.return_value = []
    mock.commit.return_value = None
    return mock


@pytest.fixture
def validation_metrics():
    """Standard validation metrics for testing."""
    return {
        "position_accuracy": {
            "horizontal_rmse": 0.0,
            "vertical_rmse": 0.0,
            "mean_horizontal_error": 0.0,
            "mean_vertical_error": 0.0
        },
        "detection_performance": {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": 0.0
        },
        "material_classification": {
            "confusion_matrix": None,
            "classification_accuracy": 0.0,
            "per_class_precision": {},
            "per_class_recall": {}
        },
        "depth_estimation": {
            "depth_rmse": 0.0,
            "depth_mae": 0.0,
            "depth_bias": 0.0
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Create test reports directory
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

    yield

    # Cleanup after test
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Setup test data fixtures at session start."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    data_dir = fixtures_dir / "data"
    data_dir.mkdir(exist_ok=True)

    logger.info("Test environment setup completed")

    yield

    logger.info("Test environment teardown completed")


# Custom pytest markers for better test organization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "validation: Ground truth validation tests"
    )
    config.addinivalue_line(
        "markers", "pas128: PAS 128 compliance tests"
    )
    config.addinivalue_line(
        "markers", "environmental: Environmental factor validation tests"
    )
    config.addinivalue_line(
        "markers", "statistical: Statistical analysis tests"
    )
    config.addinivalue_line(
        "markers", "accuracy: Accuracy assessment tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "regression: Regression tests for model changes"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file location
        test_path = str(item.fspath)

        if "/validation/" in test_path:
            item.add_marker(pytest.mark.validation)
        if "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        if "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)

        # Add slow marker for tests that might take longer
        if "benchmark" in test_path or "performance" in test_path:
            item.add_marker(pytest.mark.slow)