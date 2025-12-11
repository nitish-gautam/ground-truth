"""
Load Sample Data into Database
===============================

Loads LiDAR, Hyperspectral, and BIM sample data into the database.

Usage:
    python scripts/load_sample_data.py
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, date
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models.lidar import LidarDTMTile, LidarPointCloudCoverage
from app.models.hyperspectral import (
    HyperspectralMaterialSample,
    ConcreteStrengthCalibration,
    MaterialType,
    QualityLabel
)
from app.models.bim import BIMTestModel, IFCVersion, BIMPurpose, ComplexityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://gpr_app_user:change_me_app_2024!@localhost:5432/gpr_platform"
)

def load_lidar_samples(session):
    """Load sample LiDAR DTM tile metadata."""
    logger.info("Loading LiDAR DTM tile samples...")

    # Sample DTM tiles from datasets/raw/lidar/
    dtm_tiles = [
        {
            "tile_name": "SK23ne",
            "grid_reference": "SK23NE",
            "file_path": "/datasets/raw/lidar/dtm/SK23ne.tif",
            "file_size_mb": 42.5,
            "resolution_meters": 1.0,
            "min_elevation": 45.2,
            "max_elevation": 178.6,
            "mean_elevation": 112.3,
            "std_elevation": 28.4,
            "capture_year": 2022,
            "capture_date": date(2022, 6, 15),
            "source": "Environment Agency",
            "dataset_name": "LIDAR-DTM-1M-2022",
            "is_accessible": True,
            "is_processed": True,
            "metadata": {"tile_format": "GeoTIFF", "compression": "LZW"}
        },
        {
            "tile_name": "SK24ne",
            "grid_reference": "SK24NE",
            "file_path": "/datasets/raw/lidar/dtm/SK24ne.tif",
            "file_size_mb": 41.8,
            "resolution_meters": 1.0,
            "min_elevation": 52.1,
            "max_elevation": 165.3,
            "mean_elevation": 108.7,
            "std_elevation": 26.9,
            "capture_year": 2022,
            "capture_date": date(2022, 6, 16),
            "source": "Environment Agency",
            "dataset_name": "LIDAR-DTM-1M-2022",
            "is_accessible": True,
            "is_processed": True,
            "metadata": {"tile_format": "GeoTIFF", "compression": "LZW"}
        },
        {
            "tile_name": "SK33nw",
            "grid_reference": "SK33NW",
            "file_path": "/datasets/raw/lidar/dtm/SK33nw.tif",
            "file_size_mb": 43.2,
            "resolution_meters": 1.0,
            "min_elevation": 38.5,
            "max_elevation": 145.7,
            "mean_elevation": 92.1,
            "std_elevation": 22.3,
            "capture_year": 2022,
            "capture_date": date(2022, 6, 17),
            "source": "Environment Agency",
            "dataset_name": "LIDAR-DTM-1M-2022",
            "is_accessible": True,
            "is_processed": True,
            "metadata": {"tile_format": "GeoTIFF", "compression": "LZW"}
        }
    ]

    for tile_data in dtm_tiles:
        # Create bounds geometry (simplified - 10km x 10km tiles)
        # In production, parse from actual GeoTIFF
        tile_data["bounds"] = text(
            "ST_GeomFromText('POLYGON((400000 300000, 410000 300000, 410000 310000, 400000 310000, 400000 300000))', 27700)"
        )

        tile = LidarDTMTile(**{k: v for k, v in tile_data.items() if k != 'bounds'})
        session.add(tile)

    session.commit()
    logger.info(f"Loaded {len(dtm_tiles)} LiDAR DTM tiles")

    # Load historical coverage data
    logger.info("Loading historical LiDAR coverage data...")
    coverage_years = [2005, 2010, 2015, 2018, 2020, 2022]

    for year in coverage_years:
        coverage = LidarPointCloudCoverage(
            year=year,
            tile_reference="SK23",
            tile_name="SK23ne",
            data_available=True,
            data_quality="good" if year >= 2018 else "moderate",
            point_density=2.0 if year >= 2020 else 1.0,
            has_rgb=year >= 2020,
            has_intensity=True,
            has_classification=year >= 2018,
            provider="Environment Agency",
            capture_date_start=date(year, 5, 1),
            capture_date_end=date(year, 9, 30),
            source_format="LAZ",
            metadata={"coverage_notes": f"{year} National LiDAR Programme"}
        )
        session.add(coverage)

    session.commit()
    logger.info(f"Loaded {len(coverage_years)} historical coverage records")


def load_hyperspectral_samples(session):
    """Load sample hyperspectral training data (UMKC dataset simulation)."""
    logger.info("Loading UMKC hyperspectral training samples...")

    # Simulate UMKC concrete samples
    concrete_samples = []

    # Generate 150 base samples (will be augmented to 1,500+)
    for i in range(1, 51):  # Start with 50 samples
        # Generate realistic spectral signature (204 bands, 400-1000nm)
        spectral_signature = []
        for band in range(204):
            wavelength = 400 + (band * 600 / 203)
            # Simulate concrete spectral response with peaks/valleys
            if 500 <= wavelength <= 600:  # Cement hydration
                value = 0.55 + (i % 10) * 0.01
            elif 700 <= wavelength <= 850:  # Moisture
                value = 0.30 + (i % 15) * 0.005
            elif 900 <= wavelength <= 1000:  # Aggregate
                value = 0.50 + (i % 12) * 0.008
            else:
                value = 0.45 + (i % 8) * 0.01

            spectral_signature.append(round(value, 3))

        # Simulate concrete strength based on spectral features
        cement_quality = spectral_signature[int((550 - 400) / (600 / 203))]
        moisture = spectral_signature[int((775 - 400) / (600 / 203))]
        strength_mpa = 25.0 + (cement_quality * 40) - (moisture * 20) + (i % 5) * 2

        sample = HyperspectralMaterialSample(
            sample_id=f"UMKC_C_{i:03d}",
            sample_name=f"Concrete Sample {i}",
            material_type=MaterialType.CONCRETE,
            material_subtype=f"C{int(strength_mpa / 5) * 5}",  # C25, C30, C35, etc.
            surface_condition="good" if i % 3 == 0 else "moderate",
            surface_age=f"{(i % 10) + 1} months",
            moisture_level="low" if moisture < 0.35 else "moderate",
            image_path=f"/datasets/raw/hyperspectral/UMKC/concrete/sample_{i:03d}.jpg",
            image_format="JPEG",
            resolution="512x512",
            spectral_signature=spectral_signature,
            num_bands=204,
            wavelength_range_nm="400-1000",
            is_specim_compatible=True,
            spectral_resolution_nm=3.0,
            source="UMKC",
            dataset_name="UMKC Concrete Library",
            quality_label=QualityLabel.TRAINING if i <= 35 else (
                QualityLabel.VALIDATION if i <= 45 else QualityLabel.TEST
            ),
            is_augmented=False,
            ground_truth_strength_mpa=round(strength_mpa, 1),
            ground_truth_moisture_pct=round(moisture * 15, 1),
            ground_truth_defects=[],
            capture_date=date(2023, 1 + (i % 12), 1),
            metadata={
                "lab_tested": True,
                "curing_days": 28,
                "test_method": "compression_test"
            }
        )
        concrete_samples.append(sample)
        session.add(sample)

    session.commit()
    logger.info(f"Loaded {len(concrete_samples)} UMKC hyperspectral samples")

    # Create calibration model
    logger.info("Creating concrete strength calibration model...")

    calibration = ConcreteStrengthCalibration(
        calibration_name="UMKC Lab Calibration v1.0",
        description="Primary calibration model trained on UMKC concrete dataset",
        test_location="lab",
        num_samples=35,  # Training samples
        strength_range_min=25.0,
        strength_range_max=60.0,
        key_wavelengths=[550.0, 775.0, 950.0],  # Cement, moisture, aggregate
        spectral_coefficients={
            "cement_hydration_coef": 0.65,
            "moisture_coef": -0.45,
            "aggregate_coef": 0.25,
            "intercept": 28.5
        },
        model_type="multiple_linear_regression",
        model_parameters={
            "n_features": 3,
            "regularization": "ridge",
            "alpha": 0.1
        },
        r_squared=0.87,  # Target: 0.89
        mae=3.5,  # Target: 3.2 MPa
        rmse=4.2,
        is_validated=True,
        validation_date=date(2024, 1, 15),
        validated_by="UMKC Lab",
        is_active=True,
        notes="Initial calibration - to be refined with additional field data"
    )
    session.add(calibration)
    session.commit()

    logger.info("Calibration model created")


def load_bim_samples(session):
    """Load sample BIM/IFC test models."""
    logger.info("Loading BIM test models...")

    bim_models = [
        {
            "model_name": "IFC 4.3 Bridge Example",
            "description": "IFC 4.3 bridge/viaduct model demonstrating IfcBridge entity",
            "file_path": "/datasets/raw/bim/ifc-4.3/bridge_sample.ifc",
            "file_size_kb": 1250.5,
            "ifc_version": IFCVersion.IFC_4_3,
            "schema_name": "IFC4X3_ADD2",
            "num_elements": 456,
            "element_counts": {
                "IfcBridge": 1,
                "IfcBeam": 45,
                "IfcColumn": 32,
                "IfcSlab": 28,
                "IfcWall": 15
            },
            "has_alignment": True,
            "has_bridge": True,
            "has_tunnel": False,
            "has_earthworks": False,
            "project_name": "IFC 4.3 Bridge Sample",
            "purpose": BIMPurpose.TEST,
            "complexity_level": ComplexityLevel.MODERATE,
            "import_date": datetime.now(),
            "is_validated": True,
            "validation_errors": [],
            "validation_warnings": ["Minor geometry inconsistencies"],
            "metadata": {"source": "IFC 4.3 Sample Models", "category": "infrastructure"}
        },
        {
            "model_name": "IFC 4.3 Tunnel Example",
            "description": "IFC 4.3 tunnel model demonstrating IfcTunnel entity",
            "file_path": "/datasets/raw/bim/ifc-4.3/tunnel_sample.ifc",
            "file_size_kb": 2100.3,
            "ifc_version": IFCVersion.IFC_4_3,
            "schema_name": "IFC4X3_ADD2",
            "num_elements": 823,
            "element_counts": {
                "IfcTunnel": 1,
                "IfcEarthworksCut": 12,
                "IfcWall": 156,
                "IfcSlab": 45,
                "IfcSpace": 8
            },
            "has_alignment": True,
            "has_bridge": False,
            "has_tunnel": True,
            "has_earthworks": True,
            "project_name": "IFC 4.3 Tunnel Sample",
            "purpose": BIMPurpose.TEST,
            "complexity_level": ComplexityLevel.COMPLEX,
            "import_date": datetime.now(),
            "is_validated": True,
            "validation_errors": [],
            "validation_warnings": [],
            "metadata": {"source": "IFC 4.3 Sample Models", "category": "infrastructure"}
        },
        {
            "model_name": "IFC 4.0 Office Building",
            "description": "IFC 4.0 standard office building model",
            "file_path": "/datasets/raw/bim/ifc-4.0/office_building.ifc",
            "file_size_kb": 3450.8,
            "ifc_version": IFCVersion.IFC_4,
            "schema_name": "IFC4",
            "num_elements": 1234,
            "element_counts": {
                "IfcWall": 234,
                "IfcSlab": 78,
                "IfcColumn": 89,
                "IfcDoor": 45,
                "IfcWindow": 123
            },
            "has_alignment": False,
            "has_bridge": False,
            "has_tunnel": False,
            "has_earthworks": False,
            "project_name": "Office Building Sample",
            "purpose": BIMPurpose.TRAINING,
            "complexity_level": ComplexityLevel.MODERATE,
            "import_date": datetime.now(),
            "is_validated": True,
            "validation_errors": [],
            "validation_warnings": ["Missing some property sets"],
            "metadata": {"source": "IFC 4.0 Test Suite", "category": "building"}
        }
    ]

    for model_data in bim_models:
        model = BIMTestModel(**model_data)
        session.add(model)

    session.commit()
    logger.info(f"Loaded {len(bim_models)} BIM test models")


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("LOADING SAMPLE DATA INTO DATABASE")
    logger.info("=" * 80)

    # Create engine and session
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Load data
        load_lidar_samples(session)
        load_hyperspectral_samples(session)
        load_bim_samples(session)

        logger.info("=" * 80)
        logger.info("✅ SAMPLE DATA LOADING COMPLETE")
        logger.info("=" * 80)

        # Print summary
        logger.info("\nData Summary:")
        logger.info(f"  - LiDAR DTM Tiles: {session.query(LidarDTMTile).count()}")
        logger.info(f"  - LiDAR Coverage Records: {session.query(LidarPointCloudCoverage).count()}")
        logger.info(f"  - Hyperspectral Samples: {session.query(HyperspectralMaterialSample).count()}")
        logger.info(f"  - Calibration Models: {session.query(ConcreteStrengthCalibration).count()}")
        logger.info(f"  - BIM Models: {session.query(BIMTestModel).count()}")

    except Exception as e:
        logger.error(f"❌ Error loading sample data: {e}", exc_info=True)
        session.rollback()
        return 1

    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
