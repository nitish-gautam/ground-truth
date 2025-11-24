"""
Application configuration management
=====================================

Centralized configuration for the Underground Utility Detection Platform.
Handles environment variables, database settings, and application parameters.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and type hints."""

    # Application settings
    APP_NAME: str = "Underground Utility Detection Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Host to bind the server")
    PORT: int = Field(default=8000, description="Port to bind the server")

    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="JWT token expiry in minutes")

    # Security settings
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Allowed hosts for the application")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )

    # Database settings
    DATABASE_URL: Optional[PostgresDsn] = Field(
        default=None,
        description="Complete database URL"
    )
    DB_HOST: str = Field(default="localhost", description="Database host")
    DB_PORT: int = Field(default=5432, description="Database port")
    DB_USER: str = Field(default="gpr_app_user", description="Database user")
    DB_PASSWORD: str = Field(default="change_me_app_2024!", description="Database password")
    DB_NAME: str = Field(default="gpr_platform", description="Database name")

    # Connection pool settings
    DB_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, description="Database max overflow connections")
    DB_POOL_TIMEOUT: int = Field(default=30, description="Database pool timeout in seconds")

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(cls, v: Optional[str], values) -> str:
        """Assemble database URL from individual components if not provided."""
        if isinstance(v, str):
            return v

        # Access other fields using values.data if they exist
        if hasattr(values, 'data'):
            data = values.data
        else:
            data = values

        return PostgresDsn.build(
            scheme="postgresql",
            username=data.get("DB_USER", "gpr_app_user"),
            password=data.get("DB_PASSWORD", "change_me_app_2024!"),
            host=data.get("DB_HOST", "localhost"),
            port=data.get("DB_PORT", 5432),
            path=data.get("DB_NAME", "gpr_platform"),
        )

    # File storage settings
    DATA_ROOT_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "datasets",
        description="Root path for dataset storage"
    )
    UPLOAD_MAX_SIZE: int = Field(default=100 * 1024 * 1024, description="Max upload size in bytes (100MB)")
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=[".zip", ".csv", ".dt1", ".hd", ".gps", ".jpg", ".png", ".pdf"],
        description="Allowed file extensions"
    )

    # GPR processing settings
    GPR_TWENTE_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "datasets" / "raw" / "twente_gpr",
        description="Path to Twente GPR dataset"
    )
    GPR_MOJAHID_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "datasets" / "raw" / "mojahid_images",
        description="Path to Mojahid images dataset"
    )

    # Signal processing parameters
    SAMPLING_FREQUENCY: float = Field(default=1000.0, description="GPR sampling frequency in MHz")
    TIME_WINDOW: float = Field(default=100.0, description="Time window in nanoseconds")
    DEPTH_CALIBRATION: float = Field(default=0.1, description="Depth calibration factor in m/ns")

    # Batch processing settings
    BATCH_SIZE: int = Field(default=10, description="Number of files to process in one batch")
    MAX_WORKERS: int = Field(default=4, description="Maximum number of worker processes")
    PROCESSING_TIMEOUT: int = Field(default=300, description="Processing timeout in seconds")

    # Machine learning settings
    MODEL_CACHE_SIZE: int = Field(default=100, description="Number of models to cache")
    FEATURE_VECTOR_SIZE: int = Field(default=512, description="Size of feature vectors")
    ML_CONFIDENCE_THRESHOLD: float = Field(default=0.8, description="ML model confidence threshold")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format string"
    )
    LOG_ROTATION: str = Field(default="10 MB", description="Log file rotation size")
    LOG_RETENTION: str = Field(default="1 week", description="Log file retention period")

    # Redis cache settings (optional)
    REDIS_URL: Optional[str] = Field(default=None, description="Redis URL for caching")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")

    # Monitoring settings
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=9090, description="Port for metrics endpoint")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_twente_zip_files(self) -> List[Path]:
        """Get list of Twente GPR ZIP files."""
        if not self.GPR_TWENTE_PATH.exists():
            return []

        return sorted([
            f for f in self.GPR_TWENTE_PATH.glob("*.zip")
            if f.name.replace(".zip", "").isdigit() or
               f.name in ["010.zip", "011.zip", "012.zip", "013.zip"]
        ])

    def get_mojahid_categories(self) -> List[str]:
        """Get list of Mojahid image categories."""
        mojahid_data_path = self.GPR_MOJAHID_PATH / "GPR_data"
        if not mojahid_data_path.exists():
            return []

        return [d.name for d in mojahid_data_path.iterdir() if d.is_dir()]

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        return str(self.DATABASE_URL).replace("postgresql://", "postgresql://")

    @property
    def database_url_async(self) -> str:
        """Get asynchronous database URL."""
        return str(self.DATABASE_URL).replace("postgresql://", "postgresql+asyncpg://")


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency function to get settings instance."""
    return settings