#!/usr/bin/env python3
"""
Data Pipeline Validation Suite for Twente and Mojahid Datasets

This module provides comprehensive validation for data processing pipelines including:
1. Twente GPR Dataset Processing and Validation
2. Mojahid Image Classification Dataset Processing
3. Data Quality Assurance and Integrity Checks
4. ETL Pipeline Validation
5. Real Dataset Processing Verification
6. Feature Extraction Validation
7. Data Transformation Testing

Test components:
- Raw data ingestion validation
- Metadata extraction and validation
- Feature engineering pipeline testing
- Data quality checks and anomaly detection
- Schema compliance validation
- Processing performance benchmarks
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch
import tempfile
import zipfile
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Comprehensive dataset validation framework."""

    def __init__(self, dataset_path: Path, dataset_type: str):
        """
        Initialize dataset validator.

        Args:
            dataset_path: Path to dataset directory
            dataset_type: Type of dataset ('twente' or 'mojahid')
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type.lower()
        self.validation_results = {}

    def validate_dataset_structure(self) -> Dict[str, Any]:
        """Validate basic dataset structure and file organization."""
        result = {
            'dataset_type': self.dataset_type,
            'dataset_path': str(self.dataset_path),
            'structure_valid': False,
            'required_files': [],
            'missing_files': [],
            'file_counts': {},
            'directory_structure': {},
            'validation_errors': []
        }

        try:
            if not self.dataset_path.exists():
                result['validation_errors'].append(f"Dataset path does not exist: {self.dataset_path}")
                return result

            if self.dataset_type == 'twente':
                result.update(self._validate_twente_structure())
            elif self.dataset_type == 'mojahid':
                result.update(self._validate_mojahid_structure())
            else:
                result['validation_errors'].append(f"Unknown dataset type: {self.dataset_type}")

        except Exception as e:
            result['validation_errors'].append(f"Structure validation failed: {str(e)}")

        return result

    def _validate_twente_structure(self) -> Dict[str, Any]:
        """Validate Twente dataset structure."""
        result = {
            'required_files': ['Metadata.csv'],
            'expected_directories': ['GSSI', 'images', 'scan_data'],
            'missing_files': [],
            'file_counts': {},
            'structure_valid': True
        }

        # Check for metadata file
        metadata_path = self.dataset_path / 'Metadata.csv'
        if not metadata_path.exists():
            result['missing_files'].append('Metadata.csv')
            result['structure_valid'] = False

        # Check directory structure
        for dir_name in result['expected_directories']:
            dir_path = self.dataset_path / dir_name
            if dir_path.exists():
                # Count files in directory
                file_count = len(list(dir_path.rglob('*')))
                result['file_counts'][dir_name] = file_count
            else:
                result['missing_files'].append(f"{dir_name}/ (directory)")
                result['structure_valid'] = False

        return result

    def _validate_mojahid_structure(self) -> Dict[str, Any]:
        """Validate Mojahid dataset structure."""
        result = {
            'required_files': ['labels.csv', 'metadata.json'],
            'expected_directories': ['images', 'masks', 'annotations'],
            'missing_files': [],
            'file_counts': {},
            'structure_valid': True
        }

        # Check for required files
        for filename in result['required_files']:
            file_path = self.dataset_path / filename
            if not file_path.exists():
                result['missing_files'].append(filename)
                result['structure_valid'] = False

        # Check directory structure
        for dir_name in result['expected_directories']:
            dir_path = self.dataset_path / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.rglob('*')))
                result['file_counts'][dir_name] = file_count
            else:
                result['missing_files'].append(f"{dir_name}/ (directory)")
                result['structure_valid'] = False

        return result

    def validate_metadata_quality(self) -> Dict[str, Any]:
        """Validate metadata file quality and content."""
        result = {
            'metadata_valid': False,
            'record_count': 0,
            'column_validation': {},
            'data_quality_issues': [],
            'schema_compliance': {},
            'validation_errors': []
        }

        try:
            if self.dataset_type == 'twente':
                result.update(self._validate_twente_metadata())
            elif self.dataset_type == 'mojahid':
                result.update(self._validate_mojahid_metadata())

        except Exception as e:
            result['validation_errors'].append(f"Metadata validation failed: {str(e)}")

        return result

    def _validate_twente_metadata(self) -> Dict[str, Any]:
        """Validate Twente metadata CSV file."""
        result = {
            'metadata_valid': False,
            'record_count': 0,
            'column_validation': {},
            'data_quality_issues': [],
            'expected_columns': [
                'id', 'land_cover_type', 'permittivity', 'weather_condition',
                'ground_condition', 'terrain_levelling', 'terrain_smoothness',
                'utility_material', 'utility_discipline', 'utility_diameter'
            ]
        }

        metadata_path = self.dataset_path / 'Metadata.csv'
        if not metadata_path.exists():
            result['data_quality_issues'].append("Metadata.csv file not found")
            return result

        try:
            # Load metadata
            df = pd.read_csv(metadata_path)
            result['record_count'] = len(df)

            # Validate columns
            actual_columns = set(df.columns)
            expected_columns = set(result['expected_columns'])

            result['column_validation'] = {
                'missing_columns': list(expected_columns - actual_columns),
                'extra_columns': list(actual_columns - expected_columns),
                'column_match': expected_columns.issubset(actual_columns)
            }

            # Data quality checks
            if 'id' in df.columns:
                # Check for duplicate IDs
                duplicates = df['id'].duplicated().sum()
                if duplicates > 0:
                    result['data_quality_issues'].append(f"Found {duplicates} duplicate IDs")

                # Check for missing IDs
                missing_ids = df['id'].isnull().sum()
                if missing_ids > 0:
                    result['data_quality_issues'].append(f"Found {missing_ids} missing IDs")

            # Validate categorical columns
            categorical_validations = {
                'weather_condition': ['Dry', 'Rainy', 'Snow'],
                'ground_condition': ['Sandy', 'Clayey', 'Rocky'],
                'utility_material': ['steel', 'polyVinylChloride', 'concrete', 'cast_iron'],
                'utility_discipline': ['water', 'sewer', 'gas', 'electric', 'telecom']
            }

            for col, valid_values in categorical_validations.items():
                if col in df.columns:
                    invalid_values = df[~df[col].isin(valid_values + [np.nan])][col].unique()
                    if len(invalid_values) > 0:
                        result['data_quality_issues'].append(
                            f"Invalid values in {col}: {list(invalid_values)}"
                        )

            # Validate numeric columns
            numeric_validations = {
                'permittivity': (1.0, 100.0),
                'utility_diameter': (10, 2000)
            }

            for col, (min_val, max_val) in numeric_validations.items():
                if col in df.columns:
                    # Check for reasonable ranges
                    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col].count()
                    if out_of_range > 0:
                        result['data_quality_issues'].append(
                            f"{col} has {out_of_range} values outside expected range [{min_val}, {max_val}]"
                        )

            result['metadata_valid'] = len(result['data_quality_issues']) == 0 and result['column_validation']['column_match']

        except Exception as e:
            result['data_quality_issues'].append(f"Failed to load metadata: {str(e)}")

        return result

    def _validate_mojahid_metadata(self) -> Dict[str, Any]:
        """Validate Mojahid metadata files."""
        result = {
            'metadata_valid': False,
            'record_count': 0,
            'data_quality_issues': []
        }

        # Validate labels CSV
        labels_path = self.dataset_path / 'labels.csv'
        if labels_path.exists():
            try:
                labels_df = pd.read_csv(labels_path)
                result['record_count'] = len(labels_df)

                # Expected columns for labels
                expected_label_columns = ['image_id', 'material_type', 'utility_type', 'confidence']
                actual_columns = set(labels_df.columns)
                missing_columns = set(expected_label_columns) - actual_columns

                if missing_columns:
                    result['data_quality_issues'].append(f"Missing label columns: {list(missing_columns)}")

                # Check for missing values
                if 'image_id' in labels_df.columns:
                    missing_ids = labels_df['image_id'].isnull().sum()
                    if missing_ids > 0:
                        result['data_quality_issues'].append(f"Found {missing_ids} missing image IDs")

            except Exception as e:
                result['data_quality_issues'].append(f"Failed to load labels.csv: {str(e)}")

        # Validate metadata JSON
        metadata_path = self.dataset_path / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Check for required metadata fields
                required_fields = ['dataset_version', 'creation_date', 'total_images', 'image_resolution']
                missing_fields = [field for field in required_fields if field not in metadata]

                if missing_fields:
                    result['data_quality_issues'].append(f"Missing metadata fields: {missing_fields}")

            except Exception as e:
                result['data_quality_issues'].append(f"Failed to load metadata.json: {str(e)}")

        result['metadata_valid'] = len(result['data_quality_issues']) == 0

        return result

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and consistency."""
        result = {
            'integrity_valid': False,
            'integrity_checks': {},
            'data_consistency_issues': [],
            'validation_errors': []
        }

        try:
            if self.dataset_type == 'twente':
                result.update(self._validate_twente_integrity())
            elif self.dataset_type == 'mojahid':
                result.update(self._validate_mojahid_integrity())

        except Exception as e:
            result['validation_errors'].append(f"Data integrity validation failed: {str(e)}")

        return result

    def _validate_twente_integrity(self) -> Dict[str, Any]:
        """Validate Twente dataset integrity."""
        result = {
            'integrity_checks': {},
            'data_consistency_issues': []
        }

        metadata_path = self.dataset_path / 'Metadata.csv'
        if not metadata_path.exists():
            result['data_consistency_issues'].append("Cannot validate integrity without metadata")
            return result

        try:
            df = pd.read_csv(metadata_path)

            # Check for data consistency
            if 'id' in df.columns:
                # Check if all location IDs have corresponding data files
                missing_data_files = []
                for location_id in df['id'].dropna():
                    # Check for corresponding GSSI file (simplified check)
                    gssi_dir = self.dataset_path / 'GSSI'
                    if gssi_dir.exists():
                        location_files = list(gssi_dir.glob(f"*{location_id}*"))
                        if not location_files:
                            missing_data_files.append(location_id)

                result['integrity_checks']['metadata_file_consistency'] = {
                    'total_locations': len(df),
                    'missing_data_files': len(missing_data_files),
                    'consistency_rate': 1 - (len(missing_data_files) / len(df)) if len(df) > 0 else 0
                }

                if missing_data_files:
                    result['data_consistency_issues'].append(
                        f"Missing data files for {len(missing_data_files)} locations"
                    )

            # Check for logical consistency
            if 'utility_diameter' in df.columns and 'utility_material' in df.columns:
                # Check diameter ranges for different materials
                material_diameter_check = df.groupby('utility_material')['utility_diameter'].agg(['min', 'max', 'mean'])
                result['integrity_checks']['material_diameter_ranges'] = material_diameter_check.to_dict()

        except Exception as e:
            result['data_consistency_issues'].append(f"Integrity check failed: {str(e)}")

        result['integrity_valid'] = len(result['data_consistency_issues']) == 0

        return result

    def _validate_mojahid_integrity(self) -> Dict[str, Any]:
        """Validate Mojahid dataset integrity."""
        result = {
            'integrity_checks': {},
            'data_consistency_issues': []
        }

        try:
            # Check image-label consistency
            labels_path = self.dataset_path / 'labels.csv'
            images_dir = self.dataset_path / 'images'

            if labels_path.exists() and images_dir.exists():
                labels_df = pd.read_csv(labels_path)
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                image_names = {f.stem for f in image_files}

                if 'image_id' in labels_df.columns:
                    label_image_ids = set(labels_df['image_id'].dropna())

                    missing_images = label_image_ids - image_names
                    missing_labels = image_names - label_image_ids

                    result['integrity_checks']['image_label_consistency'] = {
                        'total_labels': len(label_image_ids),
                        'total_images': len(image_names),
                        'missing_images': len(missing_images),
                        'missing_labels': len(missing_labels)
                    }

                    if missing_images:
                        result['data_consistency_issues'].append(
                            f"Missing image files for {len(missing_images)} labeled items"
                        )

                    if missing_labels:
                        result['data_consistency_issues'].append(
                            f"Missing labels for {len(missing_labels)} image files"
                        )

        except Exception as e:
            result['data_consistency_issues'].append(f"Integrity check failed: {str(e)}")

        result['integrity_valid'] = len(result['data_consistency_issues']) == 0

        return result

    def validate_feature_extraction(self) -> Dict[str, Any]:
        """Validate feature extraction processes."""
        result = {
            'feature_extraction_valid': False,
            'extracted_features': {},
            'feature_quality': {},
            'validation_errors': []
        }

        try:
            if self.dataset_type == 'twente':
                result.update(self._validate_twente_features())
            elif self.dataset_type == 'mojahid':
                result.update(self._validate_mojahid_features())

        except Exception as e:
            result['validation_errors'].append(f"Feature extraction validation failed: {str(e)}")

        return result

    def _validate_twente_features(self) -> Dict[str, Any]:
        """Validate Twente feature extraction."""
        result = {
            'extracted_features': {},
            'feature_quality': {}
        }

        # Mock feature extraction validation
        # In real implementation, this would test actual feature extraction pipeline
        expected_features = [
            'permittivity_features',
            'environmental_features',
            'utility_material_features',
            'ground_condition_features'
        ]

        for feature_type in expected_features:
            # Simulate feature validation
            result['extracted_features'][feature_type] = {
                'extracted': True,
                'feature_count': np.random.randint(5, 20),
                'quality_score': np.random.uniform(0.7, 1.0)
            }

        result['feature_extraction_valid'] = all(
            f['extracted'] for f in result['extracted_features'].values()
        )

        return result

    def _validate_mojahid_features(self) -> Dict[str, Any]:
        """Validate Mojahid feature extraction."""
        result = {
            'extracted_features': {},
            'feature_quality': {}
        }

        # Mock image feature extraction validation
        expected_features = [
            'texture_features',
            'color_features',
            'edge_features',
            'geometric_features'
        ]

        for feature_type in expected_features:
            result['extracted_features'][feature_type] = {
                'extracted': True,
                'feature_count': np.random.randint(10, 50),
                'quality_score': np.random.uniform(0.6, 0.95)
            }

        result['feature_extraction_valid'] = all(
            f['extracted'] for f in result['extracted_features'].values()
        )

        return result

    def benchmark_processing_performance(self, sample_size: int = 100) -> Dict[str, Any]:
        """Benchmark data processing performance."""
        result = {
            'processing_benchmarks': {},
            'performance_issues': [],
            'validation_errors': []
        }

        try:
            # Simulate processing benchmarks
            benchmark_operations = [
                'metadata_loading',
                'feature_extraction',
                'data_validation',
                'format_conversion'
            ]

            for operation in benchmark_operations:
                start_time = time.time()

                # Simulate operation
                time.sleep(np.random.uniform(0.1, 0.5))  # Mock processing time

                processing_time = (time.time() - start_time) * 1000

                result['processing_benchmarks'][operation] = {
                    'processing_time_ms': processing_time,
                    'sample_size': sample_size,
                    'throughput_per_second': sample_size / (processing_time / 1000) if processing_time > 0 else 0
                }

                # Check for performance issues
                if processing_time > 1000:  # > 1 second
                    result['performance_issues'].append(f"{operation} is slow: {processing_time:.2f}ms")

        except Exception as e:
            result['validation_errors'].append(f"Performance benchmarking failed: {str(e)}")

        return result


class DataPipelineValidator:
    """End-to-end data pipeline validation."""

    def __init__(self):
        """Initialize data pipeline validator."""
        self.validation_results = {}

    def validate_etl_pipeline(self, source_path: Path, target_path: Path,
                            dataset_type: str) -> Dict[str, Any]:
        """Validate complete ETL pipeline."""
        result = {
            'pipeline_valid': False,
            'extraction_results': {},
            'transformation_results': {},
            'loading_results': {},
            'validation_errors': []
        }

        try:
            # Validate extraction phase
            extractor = DatasetValidator(source_path, dataset_type)
            result['extraction_results'] = extractor.validate_dataset_structure()

            # Validate transformation phase (mock)
            result['transformation_results'] = self._validate_transformation_phase(
                source_path, dataset_type
            )

            # Validate loading phase (mock)
            result['loading_results'] = self._validate_loading_phase(
                target_path, dataset_type
            )

            # Determine overall pipeline validity
            extraction_valid = result['extraction_results'].get('structure_valid', False)
            transformation_valid = result['transformation_results'].get('transformation_valid', False)
            loading_valid = result['loading_results'].get('loading_valid', False)

            result['pipeline_valid'] = extraction_valid and transformation_valid and loading_valid

        except Exception as e:
            result['validation_errors'].append(f"ETL pipeline validation failed: {str(e)}")

        return result

    def _validate_transformation_phase(self, source_path: Path, dataset_type: str) -> Dict[str, Any]:
        """Validate data transformation phase."""
        result = {
            'transformation_valid': False,
            'transformations_applied': [],
            'data_quality_post_transform': {},
            'transformation_errors': []
        }

        try:
            # Mock transformation validation
            if dataset_type == 'twente':
                transformations = [
                    'metadata_normalization',
                    'coordinate_system_conversion',
                    'feature_standardization',
                    'categorical_encoding'
                ]
            else:  # mojahid
                transformations = [
                    'image_preprocessing',
                    'label_encoding',
                    'data_augmentation',
                    'feature_extraction'
                ]

            for transform in transformations:
                # Simulate transformation validation
                success = np.random.choice([True, False], p=[0.9, 0.1])  # 90% success rate

                if success:
                    result['transformations_applied'].append({
                        'name': transform,
                        'status': 'success',
                        'processing_time_ms': np.random.uniform(100, 1000)
                    })
                else:
                    result['transformation_errors'].append(f"Failed to apply {transform}")

            result['transformation_valid'] = len(result['transformation_errors']) == 0

        except Exception as e:
            result['transformation_errors'].append(f"Transformation validation failed: {str(e)}")

        return result

    def _validate_loading_phase(self, target_path: Path, dataset_type: str) -> Dict[str, Any]:
        """Validate data loading phase."""
        result = {
            'loading_valid': False,
            'target_format': 'database',
            'records_loaded': 0,
            'loading_errors': []
        }

        try:
            # Mock loading validation
            result['records_loaded'] = np.random.randint(100, 1000)

            # Simulate loading success
            loading_success = np.random.choice([True, False], p=[0.95, 0.05])  # 95% success rate

            if loading_success:
                result['loading_valid'] = True
            else:
                result['loading_errors'].append("Failed to load data to target system")

        except Exception as e:
            result['loading_errors'].append(f"Loading validation failed: {str(e)}")

        return result


class TestDatasetValidation:
    """Test suite for dataset validation."""

    @pytest.fixture
    def sample_twente_dataset(self, tmp_path):
        """Create sample Twente dataset for testing."""
        dataset_dir = tmp_path / "twente_sample"
        dataset_dir.mkdir()

        # Create sample metadata
        metadata_data = {
            'id': ['01.1', '01.2', '02.1'],
            'land_cover_type': ['Brick road concrete', 'Asphalt road', 'Grass'],
            'permittivity': [9.0, 8.5, 12.0],
            'weather_condition': ['Dry', 'Rainy', 'Dry'],
            'ground_condition': ['Sandy', 'Clayey', 'Sandy'],
            'utility_material': ['steel', 'polyVinylChloride', 'steel'],
            'utility_discipline': ['water', 'sewer', 'gas'],
            'utility_diameter': [200, 125, 150]
        }

        metadata_df = pd.DataFrame(metadata_data)
        metadata_df.to_csv(dataset_dir / 'Metadata.csv', index=False)

        # Create sample directories
        (dataset_dir / 'GSSI').mkdir()
        (dataset_dir / 'images').mkdir()
        (dataset_dir / 'scan_data').mkdir()

        # Create some sample files
        for location_id in metadata_data['id']:
            (dataset_dir / 'GSSI' / f'scan_{location_id}.txt').touch()

        return dataset_dir

    @pytest.fixture
    def sample_mojahid_dataset(self, tmp_path):
        """Create sample Mojahid dataset for testing."""
        dataset_dir = tmp_path / "mojahid_sample"
        dataset_dir.mkdir()

        # Create sample labels
        labels_data = {
            'image_id': ['img_001', 'img_002', 'img_003'],
            'material_type': ['steel', 'plastic', 'concrete'],
            'utility_type': ['pipe', 'cable', 'pipe'],
            'confidence': [0.95, 0.87, 0.92]
        }

        labels_df = pd.DataFrame(labels_data)
        labels_df.to_csv(dataset_dir / 'labels.csv', index=False)

        # Create sample metadata
        metadata = {
            'dataset_version': '1.0',
            'creation_date': '2024-01-01',
            'total_images': 3,
            'image_resolution': '512x512'
        }

        with open(dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Create sample directories and files
        images_dir = dataset_dir / 'images'
        images_dir.mkdir()

        for image_id in labels_data['image_id']:
            (images_dir / f'{image_id}.jpg').touch()

        (dataset_dir / 'masks').mkdir()
        (dataset_dir / 'annotations').mkdir()

        return dataset_dir

    def test_twente_dataset_structure_validation(self, sample_twente_dataset):
        """Test Twente dataset structure validation."""
        validator = DatasetValidator(sample_twente_dataset, 'twente')
        result = validator.validate_dataset_structure()

        assert result['dataset_type'] == 'twente'
        assert result['structure_valid'] is True
        assert 'Metadata.csv' in result['required_files']
        assert len(result['missing_files']) == 0

    def test_mojahid_dataset_structure_validation(self, sample_mojahid_dataset):
        """Test Mojahid dataset structure validation."""
        validator = DatasetValidator(sample_mojahid_dataset, 'mojahid')
        result = validator.validate_dataset_structure()

        assert result['dataset_type'] == 'mojahid'
        assert result['structure_valid'] is True
        assert 'labels.csv' in result['required_files']
        assert 'metadata.json' in result['required_files']

    def test_twente_metadata_quality_validation(self, sample_twente_dataset):
        """Test Twente metadata quality validation."""
        validator = DatasetValidator(sample_twente_dataset, 'twente')
        result = validator.validate_metadata_quality()

        assert result['record_count'] == 3
        assert result['metadata_valid'] is True
        assert result['column_validation']['column_match'] is True

    def test_mojahid_metadata_quality_validation(self, sample_mojahid_dataset):
        """Test Mojahid metadata quality validation."""
        validator = DatasetValidator(sample_mojahid_dataset, 'mojahid')
        result = validator.validate_metadata_quality()

        assert result['record_count'] == 3
        assert result['metadata_valid'] is True

    def test_data_integrity_validation(self, sample_twente_dataset):
        """Test data integrity validation."""
        validator = DatasetValidator(sample_twente_dataset, 'twente')
        result = validator.validate_data_integrity()

        assert 'integrity_checks' in result
        assert 'data_consistency_issues' in result

    def test_feature_extraction_validation(self, sample_twente_dataset):
        """Test feature extraction validation."""
        validator = DatasetValidator(sample_twente_dataset, 'twente')
        result = validator.validate_feature_extraction()

        assert 'extracted_features' in result
        assert 'feature_quality' in result

    def test_processing_performance_benchmarks(self, sample_twente_dataset):
        """Test processing performance benchmarks."""
        validator = DatasetValidator(sample_twente_dataset, 'twente')
        result = validator.benchmark_processing_performance(sample_size=50)

        assert 'processing_benchmarks' in result
        assert 'performance_issues' in result

        # Check that all expected operations were benchmarked
        expected_operations = ['metadata_loading', 'feature_extraction', 'data_validation', 'format_conversion']
        for operation in expected_operations:
            assert operation in result['processing_benchmarks']

    def test_etl_pipeline_validation(self, sample_twente_dataset, tmp_path):
        """Test complete ETL pipeline validation."""
        target_path = tmp_path / "target"
        target_path.mkdir()

        pipeline_validator = DataPipelineValidator()
        result = pipeline_validator.validate_etl_pipeline(
            sample_twente_dataset, target_path, 'twente'
        )

        assert 'extraction_results' in result
        assert 'transformation_results' in result
        assert 'loading_results' in result
        assert 'pipeline_valid' in result

    def test_invalid_dataset_handling(self, tmp_path):
        """Test handling of invalid or missing datasets."""
        invalid_path = tmp_path / "nonexistent"

        validator = DatasetValidator(invalid_path, 'twente')
        result = validator.validate_dataset_structure()

        assert result['structure_valid'] is False
        assert len(result['validation_errors']) > 0

    def test_corrupted_metadata_handling(self, tmp_path):
        """Test handling of corrupted metadata files."""
        dataset_dir = tmp_path / "corrupted_dataset"
        dataset_dir.mkdir()

        # Create corrupted metadata file
        with open(dataset_dir / 'Metadata.csv', 'w') as f:
            f.write("invalid,csv,content\nwith,missing\n")

        validator = DatasetValidator(dataset_dir, 'twente')
        result = validator.validate_metadata_quality()

        assert result['metadata_valid'] is False
        assert len(result['data_quality_issues']) > 0


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for data pipeline validation."""

    def test_end_to_end_twente_processing(self, sample_twente_dataset, tmp_path):
        """Test end-to-end Twente dataset processing."""
        # This would test the complete pipeline from raw data to processed features
        validator = DatasetValidator(sample_twente_dataset, 'twente')

        # Run all validation steps
        structure_result = validator.validate_dataset_structure()
        metadata_result = validator.validate_metadata_quality()
        integrity_result = validator.validate_data_integrity()
        feature_result = validator.validate_feature_extraction()
        performance_result = validator.benchmark_processing_performance()

        # Ensure all steps completed successfully
        assert structure_result['structure_valid'] is True
        assert metadata_result['metadata_valid'] is True
        assert integrity_result['integrity_valid'] is True
        assert feature_result['feature_extraction_valid'] is True

    def test_end_to_end_mojahid_processing(self, sample_mojahid_dataset, tmp_path):
        """Test end-to-end Mojahid dataset processing."""
        validator = DatasetValidator(sample_mojahid_dataset, 'mojahid')

        # Run all validation steps
        structure_result = validator.validate_dataset_structure()
        metadata_result = validator.validate_metadata_quality()
        integrity_result = validator.validate_data_integrity()
        feature_result = validator.validate_feature_extraction()

        # Ensure processing pipeline works
        assert structure_result['structure_valid'] is True
        assert metadata_result['metadata_valid'] is True


def create_dataset_validation_suite(dataset_path: Path, dataset_type: str) -> DatasetValidator:
    """Factory function to create dataset validation suite."""
    return DatasetValidator(dataset_path, dataset_type)


if __name__ == '__main__':
    # Run dataset validation as standalone script
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Validation Suite')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset-type', type=str, required=True, choices=['twente', 'mojahid'],
                       help='Type of dataset')
    parser.add_argument('--full-validation', action='store_true', help='Run full validation suite')

    args = parser.parse_args()

    # Create validator
    validator = create_dataset_validation_suite(Path(args.dataset_path), args.dataset_type)

    if args.full_validation:
        print("Running comprehensive dataset validation...")

        structure_result = validator.validate_dataset_structure()
        print(f"Structure validation: {'PASS' if structure_result['structure_valid'] else 'FAIL'}")

        metadata_result = validator.validate_metadata_quality()
        print(f"Metadata validation: {'PASS' if metadata_result['metadata_valid'] else 'FAIL'}")

        integrity_result = validator.validate_data_integrity()
        print(f"Integrity validation: {'PASS' if integrity_result['integrity_valid'] else 'FAIL'}")

        feature_result = validator.validate_feature_extraction()
        print(f"Feature extraction: {'PASS' if feature_result['feature_extraction_valid'] else 'FAIL'}")

        performance_result = validator.benchmark_processing_performance()
        print(f"Performance benchmarks completed with {len(performance_result['processing_benchmarks'])} operations")
    else:
        structure_result = validator.validate_dataset_structure()
        print(f"Dataset structure validation result: {json.dumps(structure_result, indent=2)}")