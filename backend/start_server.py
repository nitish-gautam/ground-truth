#!/usr/bin/env python3
"""
Underground Utility Detection Platform - Server Startup Script
============================================================

Production-ready startup script with proper error handling,
logging configuration, and database initialization.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

import uvicorn
from loguru import logger

# Import application components
from app.core.config import settings
from app.core.database import init_db, check_database_health
from app.core.logging_config import setup_logging
from app.main import app


async def startup_checks():
    """Perform startup checks and initialization."""
    logger.info("Starting Underground Utility Detection Platform...")

    # Setup logging
    setup_logging()
    logger.info("Logging configuration initialized")

    # Check dataset paths
    if not settings.DATA_ROOT_PATH.exists():
        logger.warning(f"Dataset root path does not exist: {settings.DATA_ROOT_PATH}")
        logger.info("Creating dataset directory structure...")
        settings.DATA_ROOT_PATH.mkdir(parents=True, exist_ok=True)

    if not settings.GPR_TWENTE_PATH.exists():
        logger.warning(f"Twente GPR dataset path not found: {settings.GPR_TWENTE_PATH}")

    if not settings.GPR_MOJAHID_PATH.exists():
        logger.warning(f"Mojahid images path not found: {settings.GPR_MOJAHID_PATH}")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialization completed")

        # Check database health
        health_status = await check_database_health()
        if health_status["status"] == "healthy":
            logger.info(f"Database health check passed: {health_status}")
        else:
            logger.error(f"Database health check failed: {health_status}")
            return False

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

    # Validate critical settings
    if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-here-change-in-production":
        logger.warning("SECRET_KEY is not properly configured - using default value")

    logger.info("Startup checks completed successfully")
    return True


def main():
    """Main entry point for the application."""
    print("=" * 70)
    print("🔍 Underground Utility Detection Platform")
    print("   FastAPI Backend for GPR Data Processing")
    print("=" * 70)

    # Run startup checks
    startup_success = asyncio.run(startup_checks())

    if not startup_success:
        logger.error("Startup checks failed. Exiting...")
        sys.exit(1)

    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": settings.DEBUG,
        "log_level": "info" if not settings.DEBUG else "debug",
        "access_log": True,
    }

    # Production-specific settings
    if not settings.DEBUG:
        uvicorn_config.update({
            "workers": settings.MAX_WORKERS,
            "loop": "uvloop",
            "http": "httptools",
        })

    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"API documentation: http://{settings.HOST}:{settings.PORT}/docs")

    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()