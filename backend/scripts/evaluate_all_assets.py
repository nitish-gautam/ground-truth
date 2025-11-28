"""
HS2 Asset Evaluation Script
===========================

Runs TAEM evaluation on all assets and updates the database.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func, text
from loguru import logger

from app.core.database import db_manager
from app.models.hs2 import HS2Asset
from app.services.taem_engine import TAEMEngine


async def evaluate_all_assets(force_refresh: bool = False):
    """Evaluate all assets using TAEM engine."""
    
    logger.info("=" * 80)
    logger.info("HS2 Asset Evaluation Script")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Force Refresh: {force_refresh}")
    logger.info("=" * 80)
    
    # Initialize database
    await db_manager.init_async_engine()
    
    async with db_manager.async_session_factory() as db:
        try:
            # Get all assets
            logger.info("\nFetching assets...")
            query = select(HS2Asset).order_by(HS2Asset.asset_id)
            result = await db.execute(query)
            assets = result.scalars().all()
            
            if not assets:
                logger.warning("No assets found in database")
                return
            
            logger.info(f"Found {len(assets)} assets to evaluate\n")
            
            # Initialize TAEM engine
            taem_engine = TAEMEngine(db)
            await taem_engine.load_rules()
            
            # Evaluate each asset
            results = {
                "Ready": [],
                "Not Ready": [],
                "At Risk": [],
            }
            
            failed_evaluations = []
            
            for i, asset in enumerate(assets, 1):
                try:
                    logger.info(f"[{i}/{len(assets)}] Evaluating {asset.asset_id} ({asset.asset_name})...")
                    
                    evaluation_result = await taem_engine.evaluate_asset(
                        asset.id,
                        force_refresh=force_refresh
                    )
                    
                    status = evaluation_result["readiness_status"]
                    score = evaluation_result["overall_score"]
                    
                    results[status].append({
                        "asset_id": asset.asset_id,
                        "asset_name": asset.asset_name,
                        "score": score,
                        "rules_passed": evaluation_result["rules_passed"],
                        "rules_failed": evaluation_result["rules_failed"],
                    })
                    
                    logger.info(f"  ✓ {status} - Score: {score:.2f}/100")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to evaluate {asset.asset_id}: {str(e)}")
                    failed_evaluations.append(asset.asset_id)
            
            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("Evaluation Summary")
            logger.info("=" * 80)
            
            total_evaluated = len(assets) - len(failed_evaluations)
            
            logger.info(f"\nTotal Assets:       {len(assets)}")
            logger.info(f"Successfully Evaluated: {total_evaluated}")
            logger.info(f"Failed:             {len(failed_evaluations)}")
            
            logger.info(f"\nReadiness Status Distribution:")
            for status in ["Ready", "Not Ready", "At Risk"]:
                count = len(results[status])
                pct = (count / total_evaluated * 100) if total_evaluated > 0 else 0
                logger.info(f"  {status:12s}: {count:2d} assets ({pct:5.1f}%)")
            
            # Print detailed results by status
            for status in ["Ready", "At Risk", "Not Ready"]:
                if results[status]:
                    logger.info(f"\n{status} Assets:")
                    for asset in sorted(results[status], key=lambda x: x["score"], reverse=True):
                        logger.info(
                            f"  {asset['asset_id']:12s} - Score: {asset['score']:6.2f} "
                            f"(Pass: {asset['rules_passed']}, Fail: {asset['rules_failed']})"
                        )
            
            if failed_evaluations:
                logger.info(f"\nFailed Evaluations:")
                for asset_id in failed_evaluations:
                    logger.info(f"  {asset_id}")
            
            # Refresh materialized view (if it exists)
            logger.info("\nRefreshing materialized views...")
            try:
                await db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY hs2_readiness_summary"))
                await db.commit()
                logger.info("  ✓ Materialized view refreshed")
            except Exception as e:
                logger.warning(f"  Could not refresh materialized view: {str(e)}")
                logger.warning("  (This is expected if the view doesn't exist yet)")
            
            # Final statistics
            logger.info("\n" + "=" * 80)
            
            # Calculate average scores
            if total_evaluated > 0:
                all_scores = [
                    asset["score"] 
                    for status_list in results.values() 
                    for asset in status_list
                ]
                avg_score = sum(all_scores) / len(all_scores)
                min_score = min(all_scores)
                max_score = max(all_scores)
                
                logger.info(f"\nScore Statistics:")
                logger.info(f"  Average: {avg_score:.2f}")
                logger.info(f"  Minimum: {min_score:.2f}")
                logger.info(f"  Maximum: {max_score:.2f}")
            
            logger.info("\n✅ Evaluation complete!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
        finally:
            await db.close()
    
    # Close connections
    await db_manager.close_connections()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate all HS2 assets using TAEM rules"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if recent evaluations exist"
    )
    
    args = parser.parse_args()
    
    asyncio.run(evaluate_all_assets(force_refresh=args.force))


if __name__ == "__main__":
    main()
