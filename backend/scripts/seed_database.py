"""
HS2 Database Seeding Script
===========================

Loads placeholder data from JSON files and inserts into PostgreSQL database.
Handles foreign key relationships and ensures idempotent operations.
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from uuid import UUID

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import db_manager
from app.models.hs2 import (
    HS2Asset,
    HS2Deliverable,
    HS2Cost,
    HS2Certificate,
)

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "placeholder_data"


class HS2DataSeeder:
    """Handles seeding of HS2 data into the database."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.asset_id_map = {}  # Maps asset_id (string) to UUID
    
    async def load_json_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        logger.info(f"Loading data from {filepath}")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} records from {filename}")
        return data
    
    async def seed_assets(self) -> int:
        """Seed assets table."""
        logger.info("=" * 80)
        logger.info("Seeding Assets...")
        logger.info("=" * 80)
        
        assets_data = await self.load_json_file("assets.json")
        
        inserted_count = 0
        updated_count = 0
        
        for asset_data in assets_data:
            asset_id = asset_data["asset_id"]
            
            # Check if asset already exists
            query = select(HS2Asset).where(HS2Asset.asset_id == asset_id)
            result = await self.db.execute(query)
            existing_asset = result.scalar_one_or_none()
            
            if existing_asset:
                # Update existing asset
                for key, value in asset_data.items():
                    if key in ["planned_completion_date", "last_evaluation_date"]:
                        value = datetime.fromisoformat(value) if value else None
                    setattr(existing_asset, key, value)
                
                self.asset_id_map[asset_id] = existing_asset.id
                updated_count += 1
                logger.debug(f"Updated asset: {asset_id}")
            else:
                # Create new asset
                # Convert date strings to datetime objects
                if asset_data.get("planned_completion_date"):
                    asset_data["planned_completion_date"] = datetime.fromisoformat(
                        asset_data["planned_completion_date"]
                    )
                if asset_data.get("last_evaluation_date"):
                    asset_data["last_evaluation_date"] = datetime.fromisoformat(
                        asset_data["last_evaluation_date"]
                    )
                
                asset = HS2Asset(**asset_data)
                self.db.add(asset)
                await self.db.flush()  # Get the generated UUID
                
                self.asset_id_map[asset_id] = asset.id
                inserted_count += 1
                logger.debug(f"Inserted asset: {asset_id}")
        
        await self.db.commit()
        
        logger.info(f"✓ Assets seeded: {inserted_count} inserted, {updated_count} updated")
        return inserted_count + updated_count
    
    async def seed_deliverables(self) -> int:
        """Seed deliverables table."""
        logger.info("=" * 80)
        logger.info("Seeding Deliverables...")
        logger.info("=" * 80)
        
        deliverables_data = await self.load_json_file("deliverables.json")
        
        inserted_count = 0
        updated_count = 0
        
        for deliv_data in deliverables_data:
            deliverable_id = deliv_data["deliverable_id"]
            asset_id_str = deliv_data["asset_id"]
            
            # Check if deliverable already exists
            query = select(HS2Deliverable).where(HS2Deliverable.deliverable_id == deliverable_id)
            result = await self.db.execute(query)
            existing_deliv = result.scalar_one_or_none()
            
            # Map asset_id from string to UUID
            if asset_id_str not in self.asset_id_map:
                logger.warning(f"Asset {asset_id_str} not found for deliverable {deliverable_id}, skipping")
                continue
            
            deliv_data["asset_id"] = self.asset_id_map[asset_id_str]
            
            # Convert date strings to datetime objects
            for date_field in ["due_date", "submission_date", "approval_date"]:
                if deliv_data.get(date_field):
                    deliv_data[date_field] = datetime.fromisoformat(deliv_data[date_field])
            
            if existing_deliv:
                # Update existing deliverable
                for key, value in deliv_data.items():
                    if key != "deliverable_id":  # Don't update the ID
                        setattr(existing_deliv, key, value)
                
                updated_count += 1
                logger.debug(f"Updated deliverable: {deliverable_id}")
            else:
                # Create new deliverable
                deliverable = HS2Deliverable(**deliv_data)
                self.db.add(deliverable)
                inserted_count += 1
                logger.debug(f"Inserted deliverable: {deliverable_id}")
        
        await self.db.commit()
        
        logger.info(f"✓ Deliverables seeded: {inserted_count} inserted, {updated_count} updated")
        return inserted_count + updated_count
    
    async def seed_costs(self) -> int:
        """Seed costs table."""
        logger.info("=" * 80)
        logger.info("Seeding Cost Data...")
        logger.info("=" * 80)
        
        costs_data = await self.load_json_file("costs.json")
        
        inserted_count = 0
        updated_count = 0
        
        for cost_data in costs_data:
            cost_line_id = cost_data["cost_line_id"]
            asset_id_str = cost_data["asset_id"]
            
            # Check if cost record already exists
            query = select(HS2Cost).where(HS2Cost.cost_line_id == cost_line_id)
            result = await self.db.execute(query)
            existing_cost = result.scalar_one_or_none()
            
            # Map asset_id from string to UUID
            if asset_id_str not in self.asset_id_map:
                logger.warning(f"Asset {asset_id_str} not found for cost {cost_line_id}, skipping")
                continue
            
            cost_data["asset_id"] = self.asset_id_map[asset_id_str]
            
            if existing_cost:
                # Update existing cost
                for key, value in cost_data.items():
                    if key != "cost_line_id":  # Don't update the ID
                        setattr(existing_cost, key, value)
                
                updated_count += 1
                logger.debug(f"Updated cost: {cost_line_id}")
            else:
                # Create new cost
                cost = HS2Cost(**cost_data)
                self.db.add(cost)
                inserted_count += 1
                logger.debug(f"Inserted cost: {cost_line_id}")
        
        await self.db.commit()
        
        logger.info(f"✓ Costs seeded: {inserted_count} inserted, {updated_count} updated")
        return inserted_count + updated_count
    
    async def seed_certificates(self) -> int:
        """Seed certificates table."""
        logger.info("=" * 80)
        logger.info("Seeding Certificates...")
        logger.info("=" * 80)
        
        certificates_data = await self.load_json_file("certificates.json")
        
        inserted_count = 0
        updated_count = 0
        
        for cert_data in certificates_data:
            certificate_id = cert_data["certificate_id"]
            asset_id_str = cert_data["asset_id"]
            
            # Check if certificate already exists
            query = select(HS2Certificate).where(HS2Certificate.certificate_id == certificate_id)
            result = await self.db.execute(query)
            existing_cert = result.scalar_one_or_none()
            
            # Map asset_id from string to UUID
            if asset_id_str not in self.asset_id_map:
                logger.warning(f"Asset {asset_id_str} not found for certificate {certificate_id}, skipping")
                continue
            
            cert_data["asset_id"] = self.asset_id_map[asset_id_str]
            
            # Convert date strings to datetime objects
            for date_field in ["issue_date", "expiry_date"]:
                if cert_data.get(date_field):
                    cert_data[date_field] = datetime.fromisoformat(cert_data[date_field])
            
            if existing_cert:
                # Update existing certificate
                for key, value in cert_data.items():
                    if key != "certificate_id":  # Don't update the ID
                        setattr(existing_cert, key, value)
                
                updated_count += 1
                logger.debug(f"Updated certificate: {certificate_id}")
            else:
                # Create new certificate
                certificate = HS2Certificate(**cert_data)
                self.db.add(certificate)
                inserted_count += 1
                logger.debug(f"Inserted certificate: {certificate_id}")
        
        await self.db.commit()
        
        logger.info(f"✓ Certificates seeded: {inserted_count} inserted, {updated_count} updated")
        return inserted_count + updated_count
    
    async def verify_data(self):
        """Verify seeded data."""
        logger.info("=" * 80)
        logger.info("Verifying Seeded Data...")
        logger.info("=" * 80)
        
        # Count assets by status
        asset_query = select(HS2Asset.readiness_status, select([HS2Asset.id]).count()).group_by(
            HS2Asset.readiness_status
        )
        
        # Simpler count query
        from sqlalchemy import func
        
        ready_query = select(func.count(HS2Asset.id)).where(HS2Asset.readiness_status == "Ready")
        not_ready_query = select(func.count(HS2Asset.id)).where(HS2Asset.readiness_status == "Not Ready")
        at_risk_query = select(func.count(HS2Asset.id)).where(HS2Asset.readiness_status == "At Risk")
        
        ready_count = (await self.db.execute(ready_query)).scalar()
        not_ready_count = (await self.db.execute(not_ready_query)).scalar()
        at_risk_count = (await self.db.execute(at_risk_query)).scalar()
        
        total_assets = ready_count + not_ready_count + at_risk_count
        
        logger.info(f"\nAsset Status Distribution:")
        logger.info(f"  Ready:     {ready_count:2d} assets ({ready_count/total_assets*100:5.1f}%)")
        logger.info(f"  Not Ready: {not_ready_count:2d} assets ({not_ready_count/total_assets*100:5.1f}%)")
        logger.info(f"  At Risk:   {at_risk_count:2d} assets ({at_risk_count/total_assets*100:5.1f}%)")
        logger.info(f"  Total:     {total_assets:2d} assets")
        
        # Count deliverables, costs, certificates
        deliv_count = (await self.db.execute(select(func.count(HS2Deliverable.id)))).scalar()
        cost_count = (await self.db.execute(select(func.count(HS2Cost.id)))).scalar()
        cert_count = (await self.db.execute(select(func.count(HS2Certificate.id)))).scalar()
        
        logger.info(f"\nRelated Data:")
        logger.info(f"  Deliverables:  {deliv_count:4d}")
        logger.info(f"  Cost Records:  {cost_count:4d}")
        logger.info(f"  Certificates:  {cert_count:4d}")
        
        logger.info("\n✓ Data verification complete")


async def main():
    """Main seeding function."""
    logger.info("=" * 80)
    logger.info("HS2 Database Seeding Script")
    logger.info("=" * 80)
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Database: {db_manager.sync_engine.url if db_manager.sync_engine else 'Not initialized'}")
    logger.info("=" * 80)
    
    # Check if data files exist
    required_files = ["assets.json", "deliverables.json", "costs.json", "certificates.json"]
    missing_files = [f for f in required_files if not (DATA_DIR / f).exists()]
    
    if missing_files:
        logger.error(f"Missing data files: {', '.join(missing_files)}")
        logger.error(f"Please run generate_placeholder_data.py first")
        return
    
    # Initialize database
    await db_manager.init_async_engine()
    
    # Create session
    async with db_manager.async_session_factory() as db:
        try:
            seeder = HS2DataSeeder(db)
            
            # Seed all data
            await seeder.seed_assets()
            await seeder.seed_deliverables()
            await seeder.seed_costs()
            await seeder.seed_certificates()
            
            # Verify data
            await seeder.verify_data()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ Database seeding complete!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Seeding failed: {str(e)}")
            await db.rollback()
            raise
        finally:
            await db.close()
    
    # Close connections
    await db_manager.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
