"""
Generate Sample HS2 Deliverables for Customer Showcase
======================================================

Creates 2,000 sample deliverables linked to the 500 HS2 assets:
- 4 deliverables per asset on average
- Various deliverable types (Design, Safety, Quality, Compliance)
- Status distribution matching asset readiness
- Due dates and completion dates
- Links to sample document storage paths
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy.orm import Session
from app.core.database import get_sync_db
from app.models.hs2 import HS2Asset, HS2Deliverable
from loguru import logger


# Deliverable type configurations
DELIVERABLE_TYPES = {
    "Design": [
        "Detailed Design Report",
        "Design Risk Assessment",
        "Design Change Notice",
        "Design Verification Report",
        "Technical Drawing Package"
    ],
    "Safety": [
        "CDM Health & Safety File",
        "RAMS (Risk Assessment Method Statement)",
        "Site Safety Inspection Report",
        "Safety Certificate",
        "Emergency Response Plan"
    ],
    "Quality": [
        "Quality Inspection Report",
        "Material Test Certificates",
        "Non-Conformance Report",
        "Quality Assurance Plan",
        "ITP (Inspection Test Plan)"
    ],
    "Environmental": [
        "Environmental Impact Assessment",
        "Ecology Survey Report",
        "Noise Monitoring Report",
        "Dust Monitoring Report",
        "Environmental Compliance Certificate"
    ],
    "Compliance": [
        "PAS 128 Compliance Report",
        "Building Regulations Approval",
        "Technical Approval Notice",
        "Certification of Compliance",
        "Regulatory Submission"
    ],
    "Progress": [
        "Monthly Progress Report",
        "Milestone Completion Report",
        "Works Programme Update",
        "Site Progress Photographs",
        "Commissioning Report"
    ]
}

# Status distribution based on asset readiness
STATUS_DISTRIBUTION = {
    "Ready": {
        "Completed": 0.85,  # 85% completed
        "In Progress": 0.10,  # 10% in progress
        "Not Started": 0.05   # 5% not started
    },
    "Not Ready": {
        "Completed": 0.40,  # 40% completed
        "In Progress": 0.35,  # 35% in progress
        "Not Started": 0.25   # 25% not started
    },
    "At Risk": {
        "Completed": 0.15,  # 15% completed
        "In Progress": 0.30,  # 30% in progress
        "Not Started": 0.35,  # 35% not started
        "Overdue": 0.20       # 20% overdue
    }
}


def get_deliverable_status(asset_readiness: str) -> str:
    """Determine deliverable status based on asset readiness"""
    distribution = STATUS_DISTRIBUTION[asset_readiness]
    rand = random.random()

    cumulative = 0.0
    for status, probability in distribution.items():
        cumulative += probability
        if rand <= cumulative:
            return status

    return "Not Started"


def generate_deliverable_metadata(deliverable_type: str, status: str) -> dict:
    """Generate realistic metadata for deliverable"""
    metadata = {
        "document_format": random.choice(["PDF", "DOCX", "XLSX", "DWG", "ZIP"]),
        "file_size_mb": round(random.uniform(0.5, 50.0), 2),
        "revision": random.choice(["Rev A", "Rev B", "Rev C", "Rev D", "Rev 01", "Rev 02"]),
        "author": random.choice([
            "John Smith", "Sarah Johnson", "Michael Brown",
            "Emma Wilson", "David Taylor", "Lisa Anderson"
        ]),
        "reviewer": random.choice([
            "Project Manager", "Technical Lead", "Quality Assurance",
            "Senior Engineer", "Design Manager", "Compliance Officer"
        ])
    }

    if status == "Completed":
        metadata["approval_date"] = (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
        metadata["approval_status"] = "Approved"
    elif status == "In Progress":
        metadata["progress_percentage"] = random.randint(20, 80)
        metadata["expected_completion"] = (datetime.now() + timedelta(days=random.randint(7, 60))).isoformat()

    return metadata


def generate_sample_deliverables(db: Session, target_count: int = 2000):
    """
    Generate sample deliverables linked to existing assets

    Args:
        db: Database session
        target_count: Target number of deliverables to generate (default: 2000)
    """
    logger.info(f"Starting generation of {target_count} sample deliverables...")

    # Get all assets
    assets = db.query(HS2Asset).all()

    if not assets:
        logger.error("No assets found in database. Please run generate_hs2_sample_assets.py first.")
        return

    logger.info(f"Found {len(assets)} assets to link deliverables to")

    # Calculate deliverables per asset
    deliverables_per_asset = target_count // len(assets)
    remainder = target_count % len(assets)

    logger.info(f"Generating ~{deliverables_per_asset} deliverables per asset")

    deliverables_created = 0
    status_counts = {"Completed": 0, "In Progress": 0, "Not Started": 0, "Overdue": 0}

    for asset in assets:
        # Determine number of deliverables for this asset
        num_deliverables = deliverables_per_asset
        if remainder > 0:
            num_deliverables += 1
            remainder -= 1

        for i in range(num_deliverables):
            # Random deliverable type and specific deliverable
            deliverable_category = random.choice(list(DELIVERABLE_TYPES.keys()))
            deliverable_name = random.choice(DELIVERABLE_TYPES[deliverable_category])

            # Generate deliverable ID
            deliverable_id = f"{asset.asset_id}-DEL-{deliverables_created + 1:04d}"

            # Determine status based on asset readiness
            status = get_deliverable_status(asset.readiness_status)
            status_counts[status] += 1

            # Generate dates
            due_date = asset.planned_completion_date - timedelta(days=random.randint(30, 180))

            submission_date = None
            approval_date = None
            approval_status = None
            days_overdue = None

            if status == "Completed":
                submission_date = due_date - timedelta(days=random.randint(5, 30))
                approval_date = submission_date + timedelta(days=random.randint(1, 14))
                approval_status = "Approved"
                status = "Approved"  # Update status to Approved for completed deliverables
            elif status == "In Progress":
                status = "Pending"  # Map to valid status
            elif status == "Overdue":
                # Overdue: due date in the past, not submitted
                due_date = datetime.now() - timedelta(days=random.randint(1, 60))
                days_overdue = (datetime.now() - due_date).days

            # Generate metadata
            metadata = generate_deliverable_metadata(deliverable_name, status)

            # Create document reference
            document_reference = f"DOC-{asset.contractor.replace(' ', '').upper()[:3]}-{deliverable_id}"

            # Determine priority based on deliverable category and status
            if deliverable_category in ["Safety", "Compliance"]:
                priority = "Critical"
            elif deliverable_category in ["Design", "Quality"]:
                priority = "Major"
            else:
                priority = "Minor"

            # Determine responsible party
            responsible_party = asset.contractor

            # Create deliverable
            deliverable = HS2Deliverable(
                asset_id=asset.id,
                deliverable_id=deliverable_id,
                deliverable_name=deliverable_name,
                deliverable_type=f"{deliverable_category} - {deliverable_name}",
                status=status,
                approval_status=approval_status,
                due_date=due_date,
                submission_date=submission_date,
                approval_date=approval_date,
                responsible_party=responsible_party,
                document_reference=document_reference,
                days_overdue=days_overdue,
                priority=priority,
                notes=f"{deliverable_name} for {asset.asset_name} (Category: {deliverable_category})"
            )

            db.add(deliverable)
            deliverables_created += 1

            # Commit in batches
            if deliverables_created % 100 == 0:
                db.commit()
                logger.info(f"Progress: {deliverables_created}/{target_count} deliverables created...")

    # Final commit
    db.commit()

    logger.success(f"✅ Successfully created {deliverables_created} sample deliverables!")
    logger.info(f"Status Distribution:")
    logger.info(f"  Completed: {status_counts['Completed']} ({status_counts['Completed']/deliverables_created*100:.1f}%)")
    logger.info(f"  In Progress: {status_counts['In Progress']} ({status_counts['In Progress']/deliverables_created*100:.1f}%)")
    logger.info(f"  Not Started: {status_counts['Not Started']} ({status_counts['Not Started']/deliverables_created*100:.1f}%)")
    logger.info(f"  Overdue: {status_counts['Overdue']} ({status_counts['Overdue']/deliverables_created*100:.1f}%)")


def main():
    """Main execution function"""
    try:
        # Get database session
        db = get_sync_db()

        # Check if deliverables already exist
        existing_count = db.query(HS2Deliverable).count()

        if existing_count > 0:
            logger.warning(f"Found {existing_count} existing deliverables in database.")

            # Check for --force flag for non-interactive mode
            if "--force" in sys.argv:
                logger.info("--force flag detected. Deleting existing deliverables without prompt...")
                db.query(HS2Deliverable).delete()
                db.commit()
                logger.success("Existing deliverables deleted.")
            else:
                response = input("Do you want to delete existing deliverables and regenerate? (yes/no): ")

                if response.lower() == "yes":
                    logger.info("Deleting existing deliverables...")
                    db.query(HS2Deliverable).delete()
                    db.commit()
                    logger.success("Existing deliverables deleted.")
                else:
                    logger.info("Keeping existing deliverables. Exiting.")
                    return

        # Generate new deliverables
        generate_sample_deliverables(db, target_count=2000)

        # Display summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)

        total = db.query(HS2Deliverable).count()
        approved = db.query(HS2Deliverable).filter(HS2Deliverable.status == "Approved").count()
        pending = db.query(HS2Deliverable).filter(HS2Deliverable.status == "Pending").count()
        not_started = db.query(HS2Deliverable).filter(HS2Deliverable.status == "Not Started").count()
        overdue = db.query(HS2Deliverable).filter(HS2Deliverable.status == "Overdue").count()

        logger.info(f"Total Deliverables: {total}")
        logger.info(f"Approved: {approved} ({approved/total*100:.1f}%)")
        logger.info(f"Pending: {pending} ({pending/total*100:.1f}%)")
        logger.info(f"Not Started: {not_started} ({not_started/total*100:.1f}%)")
        logger.info(f"Overdue: {overdue} ({overdue/total*100:.1f}%)")

        # Deliverables by priority
        logger.info("\nDeliverables by Priority:")
        for priority in ["Critical", "Major", "Minor"]:
            count = db.query(HS2Deliverable).filter(HS2Deliverable.priority == priority).count()
            logger.info(f"  {priority}: {count}")

        # Count assets with deliverables
        assets_with_deliverables = db.query(HS2Asset).join(HS2Deliverable).distinct().count()
        total_assets = db.query(HS2Asset).count()
        logger.info(f"\nAssets with Deliverables: {assets_with_deliverables}/{total_assets}")

        db.close()

    except Exception as e:
        logger.error(f"Error generating sample deliverables: {e}")
        raise


if __name__ == "__main__":
    main()
