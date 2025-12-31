"""
PostgreSQL to Neo4j Sync Script
================================

Syncs all HS2 data from PostgreSQL to Neo4j graph database.

Usage:
    # Full sync
    python sync_to_graph.py

    # Incremental sync (last 24 hours)
    python sync_to_graph.py --incremental

    # Validate sync
    python sync_to_graph.py --validate
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.services.graph_db import GraphDBService
from app.services.graph_sync import GraphSyncService
from app.core.config import settings


async def run_full_sync():
    """Perform full sync of all data"""
    print("=" * 70)
    print("PostgreSQL → Neo4j Full Sync")
    print("=" * 70)

    # Create database session
    engine = create_async_engine(
        str(settings.DATABASE_URL),
        echo=False,
        pool_pre_ping=True
    )

    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        # Initialize services
        graph_service = GraphDBService()

        # Check Neo4j connectivity
        print("\n📡 Checking Neo4j connectivity...")
        is_healthy = await graph_service.health_check()
        if not is_healthy:
            print("❌ Neo4j is not accessible. Please start Neo4j container.")
            print("   Run: docker compose up -d neo4j")
            return False

        print("✅ Neo4j is healthy and ready")

        # Initialize sync service
        sync_service = GraphSyncService(session, graph_service)

        # Run full sync
        print("\n🔄 Starting full sync...")
        start_time = datetime.now()

        try:
            stats = await sync_service.sync_all()

            elapsed = (datetime.now() - start_time).total_seconds()

            print("\n" + "=" * 70)
            print("✅ Sync Complete!")
            print("=" * 70)
            print(f"⏱️  Time: {elapsed:.2f} seconds")
            print(f"\n📊 Synced Entities:")
            print(f"   • Assets:        {stats['assets']}")
            print(f"   • Deliverables:  {stats['deliverables']}")
            print(f"   • Certificates:  {stats['certificates']}")
            print(f"   • Payments:      {stats['payments']}")
            print(f"   • Dependencies:  {stats['dependencies']}")
            print(f"   • Blockers:      {stats['blockers']}")

            print(f"\n💡 Next Steps:")
            print(f"   1. Open Neo4j Browser: http://localhost:7475")
            print(f"   2. Login: neo4j / hs2_graph_2024")
            print(f"   3. Run query: MATCH (a:Asset) RETURN a LIMIT 25")
            print(f"   4. Visualize relationships!")

            return True

        except Exception as e:
            print(f"\n❌ Sync failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            graph_service.close()
            await engine.dispose()


async def run_incremental_sync():
    """Sync only recent changes"""
    print("=" * 70)
    print("PostgreSQL → Neo4j Incremental Sync")
    print("=" * 70)

    engine = create_async_engine(str(settings.DATABASE_URL), echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        graph_service = GraphDBService()
        sync_service = GraphSyncService(session, graph_service)

        # Sync changes from last 24 hours
        since = datetime.now() - timedelta(hours=24)
        print(f"\n🔄 Syncing changes since {since}...")

        try:
            stats = await sync_service.sync_updated_since(since)

            print("\n✅ Incremental sync complete!")
            print(f"   • Updated assets: {stats['assets']}")
            print(f"   • Updated deliverables: {stats['deliverables']}")

            return True

        except Exception as e:
            print(f"❌ Incremental sync failed: {e}")
            return False
        finally:
            graph_service.close()
            await engine.dispose()


async def validate_sync():
    """Validate PostgreSQL and Neo4j are in sync"""
    print("=" * 70)
    print("Sync Validation Report")
    print("=" * 70)

    engine = create_async_engine(str(settings.DATABASE_URL), echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        graph_service = GraphDBService()
        sync_service = GraphSyncService(session, graph_service)

        try:
            report = await sync_service.validate_sync()

            print("\n📊 Entity Counts:")
            print(f"\nPostgreSQL:")
            print(f"   • Assets:        {report['postgresql']['assets']}")
            print(f"   • Deliverables:  {report['postgresql']['deliverables']}")

            print(f"\nNeo4j:")
            print(f"   • Assets:        {report['neo4j']['assets']}")
            print(f"   • Deliverables:  {report['neo4j']['deliverables']}")
            print(f"   • Certificates:  {report['neo4j']['certificates']}")
            print(f"   • Relationships: {report['neo4j']['relationships']}")

            if report['in_sync']:
                print("\n✅ Databases are in sync!")
            else:
                print("\n⚠️  Databases have different counts")
                print("   Consider running: python sync_to_graph.py")

            return report['in_sync']

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
        finally:
            graph_service.close()
            await engine.dispose()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sync HS2 data from PostgreSQL to Neo4j"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Sync only changes from last 24 hours"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate sync without making changes"
    )

    args = parser.parse_args()

    if args.validate:
        success = asyncio.run(validate_sync())
    elif args.incremental:
        success = asyncio.run(run_incremental_sync())
    else:
        success = asyncio.run(run_full_sync())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
