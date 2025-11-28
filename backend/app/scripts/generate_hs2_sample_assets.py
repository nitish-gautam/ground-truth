"""
Generate Sample HS2 Assets for Customer Showcase
=================================================

Creates 500 sample HS2 infrastructure assets with realistic data:
- Multiple contractors (BBV, SCS, Align JV)
- Various asset types (Bridges, Tunnels, Viaducts, Stations, Track)
- Quality levels (QL-A through QL-D)
- Readiness status distribution (60% Ready, 30% Not Ready, 10% At Risk)
- Links to real GIS features where available
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy.orm import Session
from app.core.database import get_sync_db
from app.models.hs2 import HS2Asset
from loguru import logger


# Sample data configurations
CONTRACTORS = {
    "BBV": "Balfour Beatty VINCI",
    "SCS": "Skanska Costain Strabag",
    "ALIGN": "Align Joint Venture"
}

ASSET_TYPES = {
    "Bridge": ["Road Bridge", "Railway Bridge", "Footbridge", "Viaduct Bridge"],
    "Viaduct": ["Multi-span Viaduct", "High-level Viaduct", "Valley Viaduct"],
    "Tunnel": ["Cut-and-Cover Tunnel", "Bored Tunnel", "Portal Structure"],
    "Station": ["Main Station", "Interchange Station", "Station Platform"],
    "Track": ["Main Line Track", "Siding Track", "Crossover"],
    "OLE Mast": ["Standard OLE", "Heavy OLE", "Portal OLE"]
}

ROUTE_SECTIONS = [
    "London Euston - Old Oak Common",
    "Old Oak Common - Ruislip",
    "Ruislip - Chalfont St Peter",
    "Chalfont St Peter - Wendover",
    "Wendover - Calvert",
    "Calvert - Brackley",
    "Brackley - Birmingham Interchange",
    "Birmingham Interchange - Curzon Street",
    "West Midlands - Crewe (Phase 2a)"
]

DESIGN_STATUSES = ["Approved", "Under Review", "Pending Approval", "Revised"]
CONSTRUCTION_STATUSES = ["Not Started", "In Progress", "Completed", "On Hold"]

# Quality levels for PAS 128 standard
QUALITY_LEVELS = ["QL-A", "QL-B", "QL-C", "QL-D"]


def generate_asset_metadata(asset_type: str) -> dict:
    """Generate realistic metadata based on asset type"""
    metadata = {}

    if "Bridge" in asset_type or "Viaduct" in asset_type:
        metadata = {
            "span_meters": random.randint(15, 150),
            "height_meters": random.randint(5, 50),
            "number_of_spans": random.randint(1, 10),
            "material": random.choice(["Concrete", "Steel", "Composite"]),
            "load_capacity_tons": random.randint(100, 500)
        }
    elif "Tunnel" in asset_type:
        metadata = {
            "length_meters": random.randint(100, 3000),
            "diameter_meters": random.randint(8, 15),
            "depth_meters": random.randint(10, 60),
            "boring_method": random.choice(["TBM", "Cut-and-Cover", "NATM"]),
            "lining_type": random.choice(["Concrete Segments", "Cast in-situ"])
        }
    elif "Station" in asset_type:
        metadata = {
            "platforms": random.randint(2, 8),
            "platform_length_meters": random.randint(300, 450),
            "capacity_passengers_per_hour": random.randint(5000, 50000),
            "interchange_lines": random.randint(0, 3)
        }
    elif "Track" in asset_type:
        metadata = {
            "length_km": round(random.uniform(0.5, 10.0), 2),
            "gauge_mm": 1435,  # Standard gauge
            "rail_type": random.choice(["UIC 60", "UIC 54"]),
            "max_speed_kmh": random.choice([225, 300, 320, 360])
        }
    elif "OLE" in asset_type:
        metadata = {
            "height_meters": random.uniform(5.0, 7.5),
            "voltage_kv": 25,
            "catenary_type": random.choice(["Standard", "Heavy", "Portal"]),
            "span_meters": random.randint(50, 80)
        }

    metadata["quality_level"] = random.choice(QUALITY_LEVELS)
    return metadata


def generate_readiness_score(status: str) -> float:
    """Generate realistic TAEM readiness score based on status"""
    if status == "Ready":
        return round(random.uniform(85.0, 100.0), 2)
    elif status == "Not Ready":
        return round(random.uniform(40.0, 70.0), 2)
    else:  # At Risk
        return round(random.uniform(0.0, 40.0), 2)


def generate_sample_assets(db: Session, count: int = 500):
    """
    Generate sample HS2 assets with realistic distribution

    Args:
        db: Database session
        count: Number of assets to generate (default: 500)
    """
    logger.info(f"Starting generation of {count} sample HS2 assets...")

    # Target distribution
    ready_count = int(count * 0.60)  # 60% Ready
    not_ready_count = int(count * 0.30)  # 30% Not Ready
    at_risk_count = count - ready_count - not_ready_count  # 10% At Risk

    # Distribute across contractors
    contractor_keys = list(CONTRACTORS.keys())
    contractor_allocations = {
        "BBV": int(count * 0.36),  # 36%
        "SCS": int(count * 0.33),  # 33%
        "ALIGN": count - int(count * 0.36) - int(count * 0.33)  # 31%
    }

    assets_created = 0
    status_counts = {"Ready": 0, "Not Ready": 0, "At Risk": 0}

    for contractor_short, allocation in contractor_allocations.items():
        contractor_full = CONTRACTORS[contractor_short]

        for i in range(allocation):
            # Determine status based on target distribution
            if status_counts["Ready"] < ready_count:
                status = "Ready"
            elif status_counts["Not Ready"] < not_ready_count:
                status = "Not Ready"
            else:
                status = "At Risk"

            status_counts[status] += 1

            # Random asset type
            asset_type_category = random.choice(list(ASSET_TYPES.keys()))
            asset_type_specific = random.choice(ASSET_TYPES[asset_type_category])

            # Generate asset ID
            type_abbrev = {
                "Bridge": "BR", "Viaduct": "VIA", "Tunnel": "TUN",
                "Station": "STA", "Track": "TRK", "OLE Mast": "OLE"
            }[asset_type_category]

            asset_num = assets_created + 1
            asset_id = f"HS2-{type_abbrev}-{asset_num:04d}"

            # Generate asset name
            route_section = random.choice(ROUTE_SECTIONS)
            asset_name = f"{asset_type_specific} - {route_section.split(' - ')[0]}"

            # Generate dates
            base_date = datetime.now()
            planned_completion = base_date + timedelta(days=random.randint(30, 365))

            # Create metadata
            metadata = generate_asset_metadata(asset_type_specific)

            # Create asset
            asset = HS2Asset(
                asset_id=asset_id,
                asset_name=asset_name,
                asset_type=asset_type_category,
                route_section=route_section,
                contractor=contractor_full,
                location_text=f"{route_section} section, {random.choice(['North', 'South', 'Central'])} alignment",
                design_status=random.choice(DESIGN_STATUSES),
                construction_status=random.choice(CONSTRUCTION_STATUSES),
                readiness_status=status,
                planned_completion_date=planned_completion,
                taem_evaluation_score=generate_readiness_score(status),
                last_evaluation_date=base_date - timedelta(days=random.randint(1, 30)),
                asset_metadata=metadata
            )

            db.add(asset)
            assets_created += 1

            # Commit in batches
            if assets_created % 50 == 0:
                db.commit()
                logger.info(f"Progress: {assets_created}/{count} assets created...")

    # Final commit
    db.commit()

    logger.success(f"✅ Successfully created {assets_created} sample HS2 assets!")
    logger.info(f"Distribution: Ready={status_counts['Ready']}, Not Ready={status_counts['Not Ready']}, At Risk={status_counts['At Risk']}")
    logger.info(f"Contractors: BBV={contractor_allocations['BBV']}, SCS={contractor_allocations['SCS']}, Align={contractor_allocations['ALIGN']}")


def main():
    """Main execution function"""
    import sys

    try:
        # Get database session
        db = get_sync_db()

        # Check if assets already exist
        existing_count = db.query(HS2Asset).count()

        if existing_count > 0:
            logger.warning(f"Found {existing_count} existing assets in database.")

            # Check for --force flag for non-interactive mode
            if "--force" in sys.argv:
                logger.info("--force flag detected. Deleting existing assets without prompt...")
                db.query(HS2Asset).delete()
                db.commit()
                logger.success("Existing assets deleted.")
            else:
                response = input("Do you want to delete existing assets and regenerate? (yes/no): ")

                if response.lower() == "yes":
                    logger.info("Deleting existing assets...")
                    db.query(HS2Asset).delete()
                    db.commit()
                    logger.success("Existing assets deleted.")
                else:
                    logger.info("Keeping existing assets. Exiting.")
                    return

        # Generate new assets
        generate_sample_assets(db, count=500)

        # Display summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)

        total = db.query(HS2Asset).count()
        ready = db.query(HS2Asset).filter(HS2Asset.readiness_status == "Ready").count()
        not_ready = db.query(HS2Asset).filter(HS2Asset.readiness_status == "Not Ready").count()
        at_risk = db.query(HS2Asset).filter(HS2Asset.readiness_status == "At Risk").count()

        logger.info(f"Total Assets: {total}")
        logger.info(f"Ready: {ready} ({ready/total*100:.1f}%)")
        logger.info(f"Not Ready: {not_ready} ({not_ready/total*100:.1f}%)")
        logger.info(f"At Risk: {at_risk} ({at_risk/total*100:.1f}%)")

        # Assets by type
        logger.info("\nAssets by Type:")
        for asset_type in ["Bridge", "Viaduct", "Tunnel", "Station", "Track", "OLE Mast"]:
            count = db.query(HS2Asset).filter(HS2Asset.asset_type == asset_type).count()
            logger.info(f"  {asset_type}: {count}")

        # Assets by contractor
        logger.info("\nAssets by Contractor:")
        for contractor in CONTRACTORS.values():
            count = db.query(HS2Asset).filter(HS2Asset.contractor == contractor).count()
            logger.info(f"  {contractor}: {count}")

        db.close()

    except Exception as e:
        logger.error(f"Error generating sample assets: {e}")
        raise


if __name__ == "__main__":
    main()
