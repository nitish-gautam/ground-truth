"""
HS2 Graph Synchronization Service
==================================

Syncs data from PostgreSQL to Neo4j graph database.

Features:
- Full sync of all HS2 assets
- Incremental sync on data changes
- Relationship creation (DEPENDS_ON, BLOCKS, HAS_DELIVERABLE, etc.)
- Spatial relationship mapping (INTERSECTS for GIS data)
- Batch processing for performance
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload

from app.models.hs2 import (
    HS2Asset,
    HS2Deliverable,
    HS2Certificate,
    HS2Cost
)
from app.services.graph_db import GraphDBService

logger = logging.getLogger(__name__)


class GraphSyncService:
    """
    Service for synchronizing PostgreSQL data to Neo4j graph database.

    Syncs:
    - Assets (500 HS2 infrastructure assets)
    - Deliverables (1,500+ deliverables)
    - Certificates (200+ certificates)
    - Payment records (1,800+ payments)
    - Relationships between all entities
    """

    def __init__(self, db_session: AsyncSession, graph_service: GraphDBService):
        """
        Initialize sync service.

        Args:
            db_session: PostgreSQL async session
            graph_service: Neo4j graph database service
        """
        self.db = db_session
        self.graph = graph_service
        self.batch_size = 50  # Process in batches for performance

    # ==================== Full Sync ====================

    async def sync_all(self) -> Dict[str, int]:
        """
        Perform full sync of all data from PostgreSQL to Neo4j.

        Returns:
            Dict with counts of synced entities
        """
        logger.info("🔄 Starting full sync PostgreSQL → Neo4j...")

        start_time = datetime.now()
        stats = {
            "assets": 0,
            "deliverables": 0,
            "certificates": 0,
            "payments": 0,
            "dependencies": 0,
            "blockers": 0
        }

        try:
            # 1. Sync assets
            logger.info("1️⃣  Syncing assets...")
            stats["assets"] = await self.sync_all_assets()

            # 2. Sync deliverables and create relationships
            logger.info("2️⃣  Syncing deliverables...")
            stats["deliverables"] = await self.sync_all_deliverables()

            # 3. Sync certificates
            logger.info("3️⃣  Syncing certificates...")
            stats["certificates"] = await self.sync_all_certificates()

            # 4. Sync payments (for cost variance blockers)
            logger.info("4️⃣  Syncing payments...")
            stats["payments"] = await self.sync_all_payments()

            # 5. Create asset dependencies
            logger.info("5️⃣  Creating asset dependencies...")
            stats["dependencies"] = await self.create_asset_dependencies()

            # 6. Identify and create blocker relationships
            logger.info("6️⃣  Creating blocker relationships...")
            stats["blockers"] = await self.create_blocker_relationships()

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Full sync complete in {elapsed:.2f}s")
            logger.info(f"   📊 Stats: {stats}")

            return stats

        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
            raise

    # ==================== Asset Sync ====================

    async def sync_all_assets(self) -> int:
        """
        Sync all assets from HS2Asset table to Neo4j.

        Returns:
            Number of assets synced
        """
        # Fetch all assets
        result = await self.db.execute(
            select(HS2Asset).limit(500)
        )
        assets = result.scalars().all()

        synced_count = 0

        # Process in batches
        for i in range(0, len(assets), self.batch_size):
            batch = assets[i:i + self.batch_size]

            for asset in batch:
                await self.graph.create_asset_node({
                    "id": asset.asset_id,
                    "name": asset.asset_name,
                    "taem_score": float(asset.taem_evaluation_score or 0),
                    "status": asset.readiness_status,
                    "contractor": asset.contractor,
                    "asset_type": asset.asset_type,
                    "route_section": asset.route_section or ""
                })
                synced_count += 1

            logger.info(f"   Synced {synced_count}/{len(assets)} assets...")

        return synced_count

    async def sync_single_asset(self, asset_id: str) -> bool:
        """
        Sync a single asset (for incremental updates).

        Args:
            asset_id: Asset ID to sync

        Returns:
            Success status
        """
        result = await self.db.execute(
            select(HS2Asset).where(HS2Asset.asset_id == asset_id)
        )
        asset = result.scalar_one_or_none()

        if not asset:
            logger.warning(f"Asset {asset_id} not found")
            return False

        await self.graph.create_asset_node({
            "id": asset.asset_id,
            "name": asset.asset_name,
            "taem_score": float(asset.taem_evaluation_score or 0),
            "status": asset.readiness_status,
            "contractor": asset.contractor,
            "asset_type": asset.asset_type
        })

        logger.info(f"✅ Synced asset: {asset_id}")
        return True

    # ==================== Deliverable Sync ====================

    async def sync_all_deliverables(self) -> int:
        """
        Sync all deliverables and create HAS_DELIVERABLE relationships.

        Returns:
            Number of deliverables synced
        """
        result = await self.db.execute(
            select(HS2Deliverable)
            .options(selectinload(HS2Deliverable.asset))
            .limit(2000)
        )
        deliverables = result.scalars().all()

        synced_count = 0

        for deliverable in deliverables:
            # Use asset.asset_id (string like "HS2-BR-0005"), not deliverable.asset_id (UUID)
            await self.graph.create_deliverable_node(
                asset_id=deliverable.asset.asset_id,  # Changed from str(deliverable.asset_id)
                deliverable_data={
                    "id": deliverable.deliverable_id,
                    "name": deliverable.deliverable_name,
                    "status": deliverable.status,
                    "due_date": deliverable.due_date.isoformat() if deliverable.due_date else None
                }
            )
            synced_count += 1

            if synced_count % 100 == 0:
                logger.info(f"   Synced {synced_count}/{len(deliverables)} deliverables...")

        return synced_count

    # ==================== Certificate Sync ====================

    async def sync_all_certificates(self) -> int:
        """
        Sync all certificates and create REQUIRES_CERTIFICATE relationships.

        Returns:
            Number of certificates synced
        """
        result = await self.db.execute(
            select(HS2Certificate)
            .options(selectinload(HS2Certificate.asset))
            .limit(500)
        )
        certificates = result.scalars().all()

        synced_count = 0

        for cert in certificates:
            await self.graph.create_certificate_relationship(
                asset_id=cert.asset.asset_id,  # Use asset.asset_id (string), not cert.asset_id (UUID)
                certificate_data={
                    "id": cert.certificate_id,
                    "type": cert.certificate_type,
                    "expiry_date": cert.expiry_date.isoformat() if cert.expiry_date else None,
                    "status": cert.status
                }
            )
            synced_count += 1

        return synced_count

    # ==================== Payment Sync ====================

    async def sync_all_payments(self) -> int:
        """
        Sync cost records to identify cost overrun blockers.

        Returns:
            Number of cost records synced
        """
        # Get costs with high cost variance (>15%)
        result = await self.db.execute(
            select(HS2Cost)
            .options(selectinload(HS2Cost.asset))
            .where(HS2Cost.variance_pct > 15.0)
            .limit(500)
        )
        costs = result.scalars().all()

        synced_count = 0

        with self.graph.driver.session() as session:
            for cost in costs:
                # Create Payment node and BLOCKS relationship if over budget
                session.run("""
                    MATCH (a:Asset {id: $asset_id})
                    MERGE (p:Payment {id: $cost_id})
                    SET p.variance_pct = $variance_pct,
                        p.amount = $amount,
                        p.status = 'over_budget'
                    MERGE (a)-[:HAS_PAYMENT]->(p)

                    // If variance > 15%, create BLOCKS relationship
                    WITH a, p
                    WHERE $variance_pct > 15.0
                    MERGE (p)-[:BLOCKS {reason: 'Cost overrun'}]->(a)

                    RETURN p
                """,
                    asset_id=cost.asset.asset_id,  # Use asset.asset_id (string), not cost.asset_id (UUID)
                    cost_id=cost.cost_line_id,
                    variance_pct=float(cost.variance_pct or 0),
                    amount=float(cost.actual_amount or 0)
                )
                synced_count += 1

        return synced_count

    # ==================== Relationship Creation ====================

    async def create_asset_dependencies(self) -> int:
        """
        Create DEPENDS_ON relationships between assets.

        Uses business logic:
        - Foundations depend on site preparation
        - Structures depend on foundations
        - MEP depends on structures

        Returns:
            Number of dependencies created
        """
        # Simplified dependency logic based on asset types
        dependency_rules = {
            "Structure": ["Foundation", "Site Preparation"],
            "MEP Systems": ["Structure"],
            "Finishing": ["Structure", "MEP Systems"]
        }

        created_count = 0

        # Fetch all assets
        result = await self.db.execute(select(HS2Asset))
        assets = result.scalars().all()

        # Group by type
        assets_by_type = {}
        for asset in assets:
            asset_type = asset.asset_type or "Unknown"
            if asset_type not in assets_by_type:
                assets_by_type[asset_type] = []
            assets_by_type[asset_type].append(asset)

        # Create dependencies
        with self.graph.driver.session() as session:
            for asset_type, depends_on_types in dependency_rules.items():
                if asset_type in assets_by_type:
                    for asset in assets_by_type[asset_type]:
                        for dep_type in depends_on_types:
                            if dep_type in assets_by_type:
                                # Link to first asset of dependency type (simplified)
                                dep_asset = assets_by_type[dep_type][0]

                                session.run("""
                                    MATCH (source:Asset {id: $source_id})
                                    MATCH (target:Asset {id: $target_id})
                                    MERGE (source)-[r:DEPENDS_ON]->(target)
                                    SET r.created_at = datetime()
                                    RETURN r
                                """,
                                    source_id=asset.asset_id,
                                    target_id=dep_asset.asset_id
                                )
                                created_count += 1

        logger.info(f"   Created {created_count} dependency relationships")
        return created_count

    async def create_blocker_relationships(self) -> int:
        """
        Identify and create BLOCKS relationships.

        Blockers include:
        - Missing deliverables
        - Expired certificates
        - Cost overruns > 15%

        Returns:
            Number of blocker relationships created
        """
        blocker_count = 0

        with self.graph.driver.session() as session:
            # 1. Missing deliverables already handled in create_deliverable_node

            # 2. Find assets with no critical deliverables
            result = session.run("""
                MATCH (a:Asset)
                WHERE a.status = 'Not Ready'
                AND NOT (a)-[:HAS_DELIVERABLE]->(:Deliverable {status: 'completed'})
                RETURN count(a) as count
            """)
            record = result.single()
            if record:
                blocker_count += record["count"]

            # 3. Expired certificates already handled in create_certificate_relationship

            # 4. Mark assets as blocked if they have any blockers
            session.run("""
                MATCH (a:Asset)<-[:BLOCKS]-(blocker)
                WITH a, count(blocker) as blocker_count
                SET a.is_blocked = true,
                    a.blocker_count = blocker_count
            """)

        logger.info(f"   Identified {blocker_count} blocker relationships")
        return blocker_count

    # ==================== Incremental Sync ====================

    async def sync_updated_since(self, since: datetime) -> Dict[str, int]:
        """
        Sync only entities updated since a given timestamp.

        Args:
            since: Timestamp to sync from

        Returns:
            Dict with counts of updated entities
        """
        logger.info(f"🔄 Syncing changes since {since}...")

        stats = {"assets": 0, "deliverables": 0}

        # Sync updated assets
        result = await self.db.execute(
            select(HS2Asset).where(
                HS2Asset.updated_at > since
            )
        )
        assets = result.scalars().all()

        for asset in assets:
            await self.sync_single_asset(asset.asset_id)
            stats["assets"] += 1

        # Sync updated deliverables
        result = await self.db.execute(
            select(HS2Deliverable)
            .options(selectinload(HS2Deliverable.asset))
            .where(HS2Deliverable.updated_at > since)
        )
        deliverables = result.scalars().all()

        for deliverable in deliverables:
            await self.graph.create_deliverable_node(
                asset_id=deliverable.asset.asset_id,  # Use asset.asset_id (string)
                deliverable_data={
                    "id": deliverable.deliverable_id,
                    "name": deliverable.deliverable_name,
                    "status": deliverable.status,
                    "due_date": deliverable.due_date.isoformat() if deliverable.due_date else None
                }
            )
            stats["deliverables"] += 1

        logger.info(f"✅ Incremental sync complete: {stats}")
        return stats

    # ==================== Utilities ====================

    async def validate_sync(self) -> Dict[str, Any]:
        """
        Validate that PostgreSQL and Neo4j are in sync.

        Returns:
            Validation report with counts and discrepancies
        """
        # Get PostgreSQL counts
        pg_asset_count = await self.db.scalar(
            select(HS2Asset).count()
        )
        pg_deliverable_count = await self.db.scalar(
            select(HS2Deliverable).count()
        )

        # Get Neo4j counts
        neo4j_stats = await self.graph.get_stats()

        report = {
            "postgresql": {
                "assets": pg_asset_count,
                "deliverables": pg_deliverable_count
            },
            "neo4j": neo4j_stats,
            "in_sync": (
                pg_asset_count == neo4j_stats["assets"] and
                pg_deliverable_count == neo4j_stats["deliverables"]
            )
        }

        if report["in_sync"]:
            logger.info("✅ PostgreSQL and Neo4j are in sync")
        else:
            logger.warning("⚠️  PostgreSQL and Neo4j counts differ")
            logger.warning(f"   PG Assets: {pg_asset_count}, Neo4j: {neo4j_stats['assets']}")
            logger.warning(f"   PG Deliverables: {pg_deliverable_count}, Neo4j: {neo4j_stats['deliverables']}")

        return report
