#!/usr/bin/env python3
"""
Create Synthetic Demo Data for LiDAR and Hyperspectral Features
================================================================

This script populates the database with synthetic demo data to showcase:
1. LiDAR elevation profile generation
2. Hyperspectral concrete quality assessment

Data is clearly marked as synthetic for demonstration purposes.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
from uuid import uuid4

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
# Import models after app initialization to avoid circular dependencies
import app.models  # This imports all models
from app.models.lidar import LidarDTMTile
from app.models.hyperspectral import HyperspectralAnalysis, HyperspectralMaterialSample, MaterialType
from app.models.base import Base
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon, box


def create_synthetic_lidar_tiles(session):
    """
    Create synthetic LiDAR DTM tiles covering the HS2 route area.

    Covers British National Grid area around:
    - Easting: 423000 - 425000 (2km span)
    - Northing: 338000 - 340000 (2km span)

    Creates 4 tiles (1km x 1km each) with realistic elevation data.
    """
    print("\n🗺️  Creating Synthetic LiDAR DTM Tiles...")

    tiles_data = [
        {
            "tile_name": "SK23ne_SYNTHETIC",
            "grid_reference": "SK23NE",
            "bounds": box(423000, 338000, 424000, 339000),  # 1km x 1km
            "min_elevation": 45.2,
            "max_elevation": 87.5,
            "mean_elevation": 65.3,
            "std_elevation": 8.7,
            "description": "Synthetic tile covering western section"
        },
        {
            "tile_name": "SK24nw_SYNTHETIC",
            "grid_reference": "SK24NW",
            "bounds": box(424000, 339000, 425000, 340000),  # 1km x 1km
            "min_elevation": 52.1,
            "max_elevation": 92.3,
            "mean_elevation": 71.2,
            "std_elevation": 9.4,
            "description": "Synthetic tile covering central section"
        },
        {
            "tile_name": "SK24ne_SYNTHETIC",
            "grid_reference": "SK24NE",
            "bounds": box(424000, 338000, 425000, 339000),  # 1km x 1km
            "min_elevation": 48.7,
            "max_elevation": 89.1,
            "mean_elevation": 68.4,
            "std_elevation": 8.2,
            "description": "Synthetic tile covering eastern section"
        },
        {
            "tile_name": "SK23nw_SYNTHETIC",
            "grid_reference": "SK23NW",
            "bounds": box(423000, 339000, 424000, 340000),  # 1km x 1km
            "min_elevation": 51.3,
            "max_elevation": 94.8,
            "mean_elevation": 73.1,
            "std_elevation": 10.1,
            "description": "Synthetic tile covering northern section"
        }
    ]

    created_tiles = []
    for tile_data in tiles_data:
        # Check if tile already exists
        existing = session.query(LidarDTMTile).filter_by(
            tile_name=tile_data["tile_name"]
        ).first()

        if existing:
            print(f"   ⏭️  Tile {tile_data['tile_name']} already exists, skipping")
            created_tiles.append(existing)
            continue

        tile = LidarDTMTile(
            id=uuid4(),
            tile_name=tile_data["tile_name"],
            grid_reference=tile_data["grid_reference"],
            file_path=f"/synthetic/lidar/{tile_data['tile_name']}.tif",
            file_size_mb=125.5,
            resolution_meters=1.0,
            bounds=from_shape(tile_data["bounds"], srid=27700),
            min_elevation=tile_data["min_elevation"],
            max_elevation=tile_data["max_elevation"],
            mean_elevation=tile_data["mean_elevation"],
            std_elevation=tile_data["std_elevation"],
            capture_year=2022,
            capture_date=datetime(2022, 6, 15),
            source="SYNTHETIC - UK Environment Agency (Demo)",
            dataset_name="lidar_composite_dtm-2022-1m-SYNTHETIC",
            tile_metadata={
                "is_synthetic": True,
                "purpose": "demonstration",
                "description": tile_data["description"],
                "coordinate_system": "EPSG:27700 (British National Grid)",
                "vertical_datum": "ODN (Ordnance Datum Newlyn)"
            },
            is_processed=True,
            is_accessible=True,
            created_at=datetime.utcnow()
        )

        session.add(tile)
        created_tiles.append(tile)
        print(f"   ✅ Created tile: {tile_data['tile_name']} "
              f"({tile_data['bounds'].bounds[0]:.0f}, {tile_data['bounds'].bounds[1]:.0f}) → "
              f"({tile_data['bounds'].bounds[2]:.0f}, {tile_data['bounds'].bounds[3]:.0f})")

    session.commit()
    print(f"\n✅ Created {len([t for t in created_tiles if t.tile_name.endswith('_SYNTHETIC')])} LiDAR DTM tiles")
    return created_tiles


def create_synthetic_hyperspectral_data(session):
    """
    Create synthetic hyperspectral scan and material sample data.

    Demonstrates:
    - Concrete strength prediction
    - Material classification
    - Defect detection
    """
    print("\n🔬 Creating Synthetic Hyperspectral Data...")

    # Create material samples (training data)
    samples_data = [
        {
            "sample_id": "CONCRETE_C40_001_SYNTH",
            "material_type": "concrete",
            "material_subtype": "C40/50",
            "strength_mpa": 45.2,
            "moisture_pct": 2.3,
            "description": "High-strength concrete sample - C40/50 grade"
        },
        {
            "sample_id": "CONCRETE_C40_002_SYNTH",
            "material_type": "concrete",
            "material_subtype": "C40/50",
            "strength_mpa": 48.7,
            "moisture_pct": 1.8,
            "description": "High-strength concrete sample - C40/50 grade"
        },
        {
            "sample_id": "CONCRETE_C30_001_SYNTH",
            "material_type": "concrete",
            "material_subtype": "C30/37",
            "strength_mpa": 35.4,
            "moisture_pct": 3.1,
            "description": "Standard concrete sample - C30/37 grade"
        }
    ]

    created_samples = []
    for sample_data in samples_data:
        existing = session.query(HyperspectralMaterialSample).filter_by(
            sample_id=sample_data["sample_id"]
        ).first()

        if existing:
            print(f"   ⏭️  Sample {sample_data['sample_id']} already exists, skipping")
            created_samples.append(existing)
            continue

        # Generate synthetic spectral data (204 bands for Specim IQ)
        spectral_data = [random.uniform(0.1, 0.9) for _ in range(204)]

        sample = HyperspectralMaterialSample(
            id=uuid4(),
            sample_id=sample_data["sample_id"],
            material_type=sample_data["material_type"],
            material_subtype=sample_data["material_subtype"],
            image_path=f"/synthetic/hyperspectral/{sample_data['sample_id']}.hdr",
            num_bands=204,
            wavelength_range_nm=[397.32, 1003.49],
            spectral_data=spectral_data,
            ground_truth_strength_mpa=sample_data["strength_mpa"],
            ground_truth_moisture_pct=sample_data["moisture_pct"],
            ground_truth_defects=[],
            sample_metadata={
                "is_synthetic": True,
                "purpose": "demonstration",
                "description": sample_data["description"],
                "camera": "Specim IQ (Synthetic)",
                "capture_conditions": "lab",
                "sample_age_days": 28
            },
            created_at=datetime.utcnow()
        )

        session.add(sample)
        created_samples.append(sample)
        print(f"   ✅ Created sample: {sample_data['sample_id']} "
              f"({sample_data['strength_mpa']} MPa)")

    # Create hyperspectral analyses (field measurements)
    analyses_data = [
        {
            "analysis_name": "HS2_VIADUCT_PIER_01_SYNTH",
            "description": "Phase 1 Viaduct - Pier Foundation",
            "strength_mpa": 43.2,
            "pass_fail": "PASS",
            "defects": []
        },
        {
            "analysis_name": "HS2_TUNNEL_LINING_02_SYNTH",
            "description": "Chiltern Tunnel - Segment 145",
            "strength_mpa": 38.9,
            "pass_fail": "FAIL",
            "defects": [{"type": "low_strength", "severity": "medium", "area_mm2": 1250}]
        },
        {
            "analysis_name": "HS2_BRIDGE_DECK_03_SYNTH",
            "description": "Colne Valley Viaduct - Deck Section",
            "strength_mpa": 46.8,
            "pass_fail": "PASS",
            "defects": []
        }
    ]

    created_analyses = []
    for analysis_data in analyses_data:
        existing = session.query(HyperspectralAnalysis).filter_by(
            analysis_name=analysis_data["analysis_name"]
        ).first()

        if existing:
            print(f"   ⏭️  Analysis {analysis_data['analysis_name']} already exists, skipping")
            created_analyses.append(existing)
            continue

        # Generate synthetic spectral signature
        spectral_signature = {
            "wavelengths": [397.32 + i * 3.0 for i in range(204)],  # 204 bands
            "reflectance": [random.uniform(0.1, 0.9) for _ in range(204)]
        }

        key_wavelengths = {
            "cement_hydration": {"wavelength": 550, "value": random.uniform(0.4, 0.6)},
            "moisture": {"wavelength": 780, "value": random.uniform(0.3, 0.5)},
            "aggregate": {"wavelength": 950, "value": random.uniform(0.5, 0.7)}
        }

        analysis = HyperspectralAnalysis(
            id=uuid4(),
            analysis_name=analysis_data["analysis_name"],
            description=analysis_data["description"],
            image_path=f"/synthetic/hyperspectral/scans/{analysis_data['analysis_name']}.hdr",
            predicted_material=MaterialType.CONCRETE,
            predicted_strength_mpa=analysis_data["strength_mpa"],
            strength_confidence=random.uniform(0.85, 0.95),
            strength_range_min=analysis_data["strength_mpa"] - 3.0,
            strength_range_max=analysis_data["strength_mpa"] + 3.0,
            confidence_score=random.uniform(0.85, 0.95),
            curing_quality_score=random.uniform(0.7, 0.95),
            moisture_content_pct=random.uniform(1.5, 3.5),
            aggregate_quality_score=random.uniform(0.75, 0.95),
            defects_detected=analysis_data["defects"],
            defect_locations=[],
            defect_severity="none" if not analysis_data["defects"] else "medium",
            spectral_signature=spectral_signature,
            key_wavelengths=key_wavelengths,
            model_version="synthetic_v1.0",
            model_accuracy=0.89,
            processing_time_ms=random.randint(500, 1500),
            processed_by="synthetic_processor",
            is_validated=False,
            image_metadata={
                "is_synthetic": True,
                "purpose": "demonstration",
                "camera": "Specim IQ (Synthetic)",
                "target_spec": "C40/50 (≥40 MPa)",
                "actual_strength": analysis_data["strength_mpa"],
                "pass_fail": analysis_data["pass_fail"],
                "bands": 204,
                "wavelength_range": [397.32, 1003.49]
            },
            created_at=datetime.utcnow(),
            analyzed_at=datetime.utcnow() - timedelta(days=random.randint(1, 30))
        )

        session.add(analysis)
        created_analyses.append(analysis)
        print(f"   ✅ Created analysis: {analysis_data['analysis_name']} "
              f"({analysis_data['strength_mpa']} MPa - {analysis_data['pass_fail']})")

    session.commit()
    print(f"\n✅ Created {len(created_samples)} material samples and {len(created_analyses)} analyses")
    return created_samples, created_analyses


def main():
    """Main execution function"""
    print("=" * 70)
    print("🎨 CREATING SYNTHETIC DEMO DATA FOR HS2 PLATFORM")
    print("=" * 70)
    print("\nThis script creates synthetic data for demonstration purposes:")
    print("  • LiDAR DTM tiles for elevation profile generation")
    print("  • Hyperspectral scans for concrete quality assessment")
    print("\n⚠️  All data is clearly marked as SYNTHETIC in the database\n")

    # Create database engine (convert async URL to sync)
    db_url = str(settings.DATABASE_URL).replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Create synthetic data
        lidar_tiles = create_synthetic_lidar_tiles(session)
        samples, scans = create_synthetic_hyperspectral_data(session)

        print("\n" + "=" * 70)
        print("✅ SYNTHETIC DEMO DATA CREATED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   • LiDAR DTM Tiles: {len(lidar_tiles)}")
        print(f"   • Hyperspectral Material Samples: {len(samples)}")
        print(f"   • Hyperspectral Analyses: {len(scans)}")
        print(f"\n🚀 Ready to test on frontend at http://localhost:3003/hs2")
        print(f"\n   LiDAR Test Coordinates (British National Grid):")
        print(f"     Start:  Easting 423000, Northing 338000")
        print(f"     End:    Easting 424000, Northing 339000")
        print(f"\n   Hyperspectral: Upload any image to trigger synthetic analysis")

    except Exception as e:
        print(f"\n❌ Error creating synthetic data: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return 1
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
