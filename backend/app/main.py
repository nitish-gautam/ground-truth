"""
Underground Utility Detection Platform - FastAPI Backend
===================================================

Main application entry point for GPR data processing and analysis.
Provides comprehensive APIs for:
- GPR signal processing and feature extraction
- Environmental correlation analysis
- Ground truth validation and accuracy assessment
- Batch processing of Twente and Mojahid datasets
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.gzip import GZipMiddleware

# Add the backend directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from api.v1.router import api_router
from core.config import settings
from core.database import init_db
from core.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Underground Utility Detection Platform...")

    # Setup logging
    setup_logging()

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down Underground Utility Detection Platform...")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Underground Utility Detection Platform",
        description="""
        Comprehensive GPR data processing and analysis platform for underground utility detection.

        ## Features

        * **GPR Data Processing**: Load and process Twente GPR dataset (125 scans)
        * **Image Classification**: Process Mojahid dataset (2,239+ labeled images)
        * **Signal Processing**: Advanced filtering, noise removal, and feature extraction
        * **Environmental Analysis**: Correlation between environmental factors and detection accuracy
        * **Ground Truth Validation**: Accuracy assessment and performance metrics
        * **Batch Processing**: Scalable processing for large datasets

        ## Datasets Supported

        * University of Twente GPR Dataset (125 real scans with ground truth)
        * Mojahid GPR Images (2,239+ labeled images across 6 categories)
        * PAS 128 Compliance Documents
        * USAG Strike Reports

        """,
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)

    return app


# Create the FastAPI application instance
app = create_application()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging."""
    logger.error(f"HTTP {exc.status_code} error on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors."""
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Underground Utility Detection Platform",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Underground Utility Detection Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )