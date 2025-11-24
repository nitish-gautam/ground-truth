"""
Database configuration and connection management
==============================================

SQLAlchemy setup for the Underground Utility Detection Platform with
async support, connection pooling, and comprehensive error handling.
"""

from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from loguru import logger

from .config import settings


class DatabaseManager:
    """Database connection and session manager."""

    def __init__(self):
        """Initialize database manager with async and sync engines."""
        self.async_engine = None
        self.sync_engine = None
        self.async_session_factory = None
        self.sync_session_factory = None

    async def init_async_engine(self):
        """Initialize async database engine with connection pooling."""
        try:
            self.async_engine = create_async_engine(
                settings.database_url_async,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_timeout=settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,
                echo=settings.DEBUG,
                future=True
            )

            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )

            logger.info("Async database engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize async database engine: {e}")
            raise

    def init_sync_engine(self):
        """Initialize sync database engine for non-async operations."""
        try:
            self.sync_engine = create_engine(
                settings.database_url_sync,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_timeout=settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,
                echo=settings.DEBUG,
                future=True
            )

            self.sync_session_factory = sessionmaker(
                self.sync_engine,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )

            logger.info("Sync database engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize sync database engine: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            if self.async_engine is None:
                await self.init_async_engine()

            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

            logger.info("Database connection test successful")
            return True

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper cleanup."""
        if self.async_session_factory is None:
            await self.init_async_engine()

        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Database session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_session(self) -> Session:
        """Get sync database session."""
        if self.sync_session_factory is None:
            self.init_sync_engine()

        return self.sync_session_factory()

    async def close_connections(self):
        """Close all database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("Async database connections closed")

        if self.sync_engine:
            self.sync_engine.dispose()
            logger.info("Sync database connections closed")


# Global database manager instance
db_manager = DatabaseManager()

# SQLAlchemy base for model definitions
Base = declarative_base()


async def init_db():
    """Initialize database connections and test connectivity."""
    logger.info("Initializing database connections...")

    try:
        # Initialize async engine
        await db_manager.init_async_engine()

        # Initialize sync engine
        db_manager.init_sync_engine()

        # Test connection
        connection_ok = await db_manager.test_connection()
        if not connection_ok:
            logger.warning("Database connection test failed - running in development mode without database")
        else:
            logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e} - running in development mode without database")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session."""
    async for session in db_manager.get_async_session():
        yield session


def get_sync_db() -> Session:
    """Get synchronous database session for non-async operations."""
    return db_manager.get_sync_session()


async def close_db():
    """Close database connections."""
    await db_manager.close_connections()


# Database utility functions
async def execute_raw_sql(sql: str, params: dict = None) -> list:
    """Execute raw SQL query and return results."""
    async for session in db_manager.get_async_session():
        try:
            result = await session.execute(text(sql), params or {})
            return result.fetchall()
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise


async def check_database_health() -> dict:
    """Check database health and return status information."""
    try:
        # Test connection
        connection_ok = await db_manager.test_connection()

        # Get database info
        async for session in db_manager.get_async_session():
            # Check PostgreSQL version
            version_result = await session.execute(text("SELECT version()"))
            db_version = version_result.scalar()

            # Check active connections
            connections_result = await session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            active_connections = connections_result.scalar()

            # Check database size
            size_result = await session.execute(
                text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            )
            db_size = size_result.scalar()

            return {
                "status": "healthy" if connection_ok else "unhealthy",
                "version": db_version,
                "active_connections": active_connections,
                "database_size": db_size,
                "connection_pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW
            }

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


class DatabaseTransaction:
    """Context manager for database transactions."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def __aenter__(self):
        """Enter transaction context."""
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context with proper rollback/commit."""
        if exc_type is not None:
            await self.session.rollback()
            logger.error(f"Transaction rolled back due to: {exc_val}")
        else:
            await self.session.commit()
            logger.debug("Transaction committed successfully")


async def get_db_transaction() -> DatabaseTransaction:
    """Get database session wrapped in transaction context."""
    async for session in db_manager.get_async_session():
        yield DatabaseTransaction(session)