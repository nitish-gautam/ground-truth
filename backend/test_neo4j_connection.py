"""
Neo4j Connection Test Script
=============================

Tests basic connectivity to Neo4j and creates sample nodes.
Run after starting docker-compose to verify Neo4j is working.

Usage:
    python test_neo4j_connection.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.graph_db import GraphDBService


async def test_neo4j_connection():
    """Test Neo4j connectivity and basic operations"""

    print("=" * 60)
    print("Neo4j Connection Test")
    print("=" * 60)

    try:
        # Initialize service
        print("\n1️⃣  Initializing GraphDB service...")
        graph_db = GraphDBService()
        print("   ✅ Service initialized")

        # Test connectivity
        print("\n2️⃣  Testing connectivity...")
        is_healthy = await graph_db.health_check()
        if is_healthy:
            print("   ✅ Neo4j is healthy and responding")
        else:
            print("   ❌ Neo4j health check failed")
            return False

        # Create sample asset
        print("\n3️⃣  Creating sample asset node...")
        sample_asset = await graph_db.create_asset_node({
            "id": "TEST-001",
            "name": "Test Viaduct",
            "taem_score": 85.5,
            "status": "Ready",
            "contractor": "Test Contractor",
            "asset_type": "Structure"
        })
        print(f"   ✅ Created asset: {sample_asset.get('name')}")

        # Create sample deliverable
        print("\n4️⃣  Creating sample deliverable...")
        sample_deliverable = await graph_db.create_deliverable_node(
            asset_id="TEST-001",
            deliverable_data={
                "id": "DEL-001",
                "name": "Design Certificate",
                "status": "completed",
                "due_date": "2024-12-31"
            }
        )
        print(f"   ✅ Created deliverable: {sample_deliverable.get('name')}")

        # Get stats
        print("\n5️⃣  Fetching database statistics...")
        stats = await graph_db.get_stats()
        print(f"   📊 Assets: {stats['assets']}")
        print(f"   📊 Deliverables: {stats['deliverables']}")
        print(f"   📊 Certificates: {stats['certificates']}")
        print(f"   📊 Relationships: {stats['relationships']}")

        # Test explainability query
        print("\n6️⃣  Testing explainability query...")
        result = await graph_db.get_why_not_ready("TEST-001")
        print(f"   ✅ Asset: {result['asset']['name']}")
        print(f"   ✅ Status: {result['asset']['status']}")
        print(f"   ✅ Blockers: {result['blocker_count']}")

        # Close connection
        graph_db.close()
        print("\n✅ All tests passed!")
        print("=" * 60)

        print("\n💡 Next steps:")
        print("   1. Access Neo4j Browser: http://localhost:7475")
        print("   2. Login: neo4j / hs2_graph_2024")
        print("   3. Run query: MATCH (n) RETURN n LIMIT 25")
        print("   4. Visualize the test nodes you just created!")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if Neo4j container is running: docker ps | grep neo4j")
        print("   2. Check Neo4j logs: docker logs hs2-neo4j")
        print("   3. Verify connection: docker exec -it hs2-neo4j cypher-shell")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_neo4j_connection())
    sys.exit(0 if success else 1)
