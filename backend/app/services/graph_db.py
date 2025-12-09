"""
HS2 Graph Database Service
===========================

Neo4j graph database service for managing asset relationships,
dependency chains, and explainability queries.

Features:
- Asset node management
- Relationship creation (DEPENDS_ON, BLOCKS, HAS_DELIVERABLE, etc.)
- Explainability queries (Why Not Ready?)
- Impact analysis (What's affected if X changes?)
- Performance-optimized graph traversal
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
import os

logger = logging.getLogger(__name__)


class GraphDBService:
    """
    Service for interacting with Neo4j graph database.

    Manages:
    - Connection pooling
    - Node creation (Asset, Deliverable, Certificate, etc.)
    - Relationship management
    - Graph queries for explainability and impact analysis
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize Neo4j driver with connection pooling.

        Args:
            uri: Neo4j connection URI (default: from env NEO4J_URI)
            user: Neo4j username (default: from env NEO4J_USER)
            password: Neo4j password (default: from env NEO4J_PASSWORD)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "hs2_graph_2024")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )

            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"✅ Connected to Neo4j at {self.uri}")

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Node Management ====================

    async def create_asset_node(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update an Asset node in the graph.

        Args:
            asset_data: Asset properties (id, name, taem_score, status, etc.)

        Returns:
            Created/updated asset node properties
        """
        with self.driver.session() as session:
            result = session.run("""
                MERGE (a:Asset {id: $id})
                SET a.name = $name,
                    a.taem_score = $taem_score,
                    a.status = $status,
                    a.contractor = $contractor,
                    a.asset_type = $asset_type,
                    a.updated_at = datetime()
                RETURN a
            """, **asset_data)

            record = result.single()
            if record:
                logger.info(f"Created/updated Asset node: {asset_data.get('id')}")
                return dict(record["a"])
            return {}

    async def create_deliverable_node(
        self,
        asset_id: str,
        deliverable_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create Deliverable node and HAS_DELIVERABLE relationship.

        Args:
            asset_id: Parent asset ID
            deliverable_data: Deliverable properties

        Returns:
            Created deliverable node
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Asset {id: $asset_id})
                MERGE (d:Deliverable {id: $deliverable_id})
                SET d.name = $name,
                    d.status = $status,
                    d.due_date = $due_date,
                    d.updated_at = datetime()
                MERGE (a)-[r:HAS_DELIVERABLE]->(d)

                // If deliverable is missing, create BLOCKS relationship
                WITH a, d
                WHERE $status = 'missing'
                MERGE (d)-[:BLOCKS]->(a)

                RETURN d
            """,
                asset_id=asset_id,
                deliverable_id=deliverable_data.get('id'),
                name=deliverable_data.get('name'),
                status=deliverable_data.get('status'),
                due_date=deliverable_data.get('due_date')
            )

            record = result.single()
            if record:
                return dict(record["d"])
            return {}

    async def create_certificate_relationship(
        self,
        asset_id: str,
        certificate_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create Certificate node and REQUIRES_CERTIFICATE relationship.

        Args:
            asset_id: Asset requiring the certificate
            certificate_data: Certificate properties

        Returns:
            Created certificate node
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Asset {id: $asset_id})
                MERGE (c:Certificate {id: $cert_id})
                SET c.type = $cert_type,
                    c.expiry_date = $expiry_date,
                    c.status = $status,
                    c.updated_at = datetime()
                MERGE (a)-[:REQUIRES_CERTIFICATE]->(c)

                // If certificate is expired, create BLOCKS relationship
                WITH a, c
                WHERE datetime($expiry_date) < datetime()
                MERGE (c)-[:BLOCKS]->(a)

                RETURN c
            """,
                asset_id=asset_id,
                cert_id=certificate_data.get('id'),
                cert_type=certificate_data.get('type'),
                expiry_date=certificate_data.get('expiry_date'),
                status=certificate_data.get('status')
            )

            record = result.single()
            if record:
                return dict(record["c"])
            return {}

    async def create_dependency_relationship(
        self,
        source_asset_id: str,
        target_asset_id: str,
        relationship_type: str = "DEPENDS_ON"
    ) -> bool:
        """
        Create dependency relationship between two assets.

        Args:
            source_asset_id: Asset that depends
            target_asset_id: Asset being depended upon
            relationship_type: Type of dependency (DEPENDS_ON, IMPACTS, etc.)

        Returns:
            Success status
        """
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (source:Asset {{id: $source_id}})
                MATCH (target:Asset {{id: $target_id}})
                MERGE (source)-[r:{relationship_type}]->(target)
                SET r.created_at = datetime()
                RETURN r
            """,
                source_id=source_asset_id,
                target_id=target_asset_id
            )

            return result.single() is not None

    # ==================== Explainability Queries ====================

    async def get_why_not_ready(self, asset_id: str) -> Dict[str, Any]:
        """
        Query why an asset is not ready - returns all blockers.

        Args:
            asset_id: Asset to analyze

        Returns:
            Dict with asset info and list of blockers
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (asset:Asset {id: $asset_id})
                OPTIONAL MATCH (blocker)-[:BLOCKS]->(asset)
                RETURN
                    asset,
                    collect(DISTINCT {
                        id: blocker.id,
                        type: labels(blocker)[0],
                        name: blocker.name,
                        status: blocker.status,
                        reason: CASE labels(blocker)[0]
                            WHEN 'Deliverable' THEN 'Missing deliverable'
                            WHEN 'Certificate' THEN 'Expired certificate'
                            WHEN 'Payment' THEN 'Cost overrun'
                            ELSE 'Blocker'
                        END
                    }) as blockers
            """, asset_id=asset_id)

            record = result.single()
            if record:
                return {
                    "asset": dict(record["asset"]),
                    "blockers": [b for b in record["blockers"] if b['id'] is not None],
                    "blocker_count": len([b for b in record["blockers"] if b['id'] is not None])
                }
            return {"asset": None, "blockers": [], "blocker_count": 0}

    async def get_dependency_chain(self, asset_id: str, max_depth: int = 5) -> List[Dict]:
        """
        Get full dependency chain for an asset.

        Args:
            asset_id: Starting asset
            max_depth: Maximum traversal depth

        Returns:
            List of nodes in dependency chain
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (asset:Asset {id: $asset_id})-[:DEPENDS_ON*1..%d]->(dependency)
                RETURN nodes(path) as chain
                LIMIT 50
            """ % max_depth, asset_id=asset_id)

            chains = []
            for record in result:
                chain = [dict(node) for node in record["chain"]]
                chains.append(chain)

            return chains

    async def get_impact_analysis(self, asset_id: str) -> Dict[str, Any]:
        """
        Analyze what would be impacted if this asset changes.

        Args:
            asset_id: Asset to analyze

        Returns:
            Dict with impacted assets and counts
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (asset:Asset {id: $asset_id})
                OPTIONAL MATCH (asset)<-[:DEPENDS_ON*1..3]-(impacted)
                RETURN
                    asset,
                    collect(DISTINCT impacted) as impacted_assets,
                    count(DISTINCT impacted) as impact_count
            """, asset_id=asset_id)

            record = result.single()
            if record:
                return {
                    "asset": dict(record["asset"]),
                    "impacted_assets": [dict(a) for a in record["impacted_assets"]],
                    "impact_count": record["impact_count"]
                }
            return {"asset": None, "impacted_assets": [], "impact_count": 0}

    # ==================== Graph Visualization Data ====================

    async def get_graph_for_visualization(
        self,
        asset_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get graph data formatted for D3.js visualization.

        Args:
            asset_id: Center asset
            depth: Relationship depth to traverse

        Returns:
            Dict with nodes and links for graph visualization
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (center:Asset {id: $asset_id})
                OPTIONAL MATCH path = (center)-[r*1..%d]-(related)
                WITH center, collect(DISTINCT related) as related_nodes, collect(DISTINCT r) as rels

                // Build nodes array
                WITH center, related_nodes, rels,
                    [{
                        id: center.id,
                        label: center.name,
                        type: 'asset',
                        status: center.status,
                        taem_score: center.taem_score
                    }] +
                    [node IN related_nodes | {
                        id: node.id,
                        label: COALESCE(node.name, node.type, 'Unknown'),
                        type: toLower(labels(node)[0]),
                        status: node.status
                    }] as nodes

                // Build links array
                UNWIND rels as rel_list
                UNWIND rel_list as rel
                WITH nodes, collect(DISTINCT {
                    source: startNode(rel).id,
                    target: endNode(rel).id,
                    type: type(rel),
                    label: type(rel)
                }) as links

                RETURN nodes, links
            """ % depth, asset_id=asset_id)

            record = result.single()
            if record:
                return {
                    "nodes": record["nodes"],
                    "links": record["links"]
                }
            return {"nodes": [], "links": []}

    # ==================== Utilities ====================

    async def health_check(self) -> bool:
        """Check if Neo4j is accessible"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as health")
                return result.single()["health"] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Asset) WITH count(a) as asset_count
                MATCH (d:Deliverable) WITH asset_count, count(d) as deliverable_count
                MATCH (c:Certificate) WITH asset_count, deliverable_count, count(c) as cert_count
                MATCH ()-[r]->() WITH asset_count, deliverable_count, cert_count, count(r) as rel_count
                RETURN asset_count, deliverable_count, cert_count, rel_count
            """)

            record = result.single()
            if record:
                return {
                    "assets": record["asset_count"],
                    "deliverables": record["deliverable_count"],
                    "certificates": record["cert_count"],
                    "relationships": record["rel_count"]
                }
            return {"assets": 0, "deliverables": 0, "certificates": 0, "relationships": 0}


# Singleton instance
_graph_db_service = None

def get_graph_db() -> GraphDBService:
    """Get or create GraphDB service singleton"""
    global _graph_db_service
    if _graph_db_service is None:
        _graph_db_service = GraphDBService()
    return _graph_db_service
