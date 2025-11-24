"""
Logging configuration for the Underground Utility Detection Platform
==================================================================

Centralized logging setup with structured logging, performance monitoring,
and comprehensive error tracking for GPR data processing operations.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from .config import settings


def setup_logging():
    """Configure application logging with structured format and rotation."""

    # Remove default logger
    logger.remove()

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console logging with colors
    logger.add(
        sys.stdout,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # File logging with rotation
    logger.add(
        log_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        backtrace=True,
        diagnose=True
    )

    # Error-specific logging
    logger.add(
        log_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="2 weeks",
        compression="zip",
        backtrace=True,
        diagnose=True
    )

    # GPR processing specific logging
    logger.add(
        log_dir / "gpr_processing.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[operation]} | {message}",
        level="INFO",
        rotation="50 MB",
        retention="1 month",
        compression="zip",
        filter=lambda record: "gpr_processing" in record["extra"]
    )

    # Performance monitoring
    logger.add(
        log_dir / "performance.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {extra[duration]}ms | {extra[operation]} | {message}",
        level="INFO",
        rotation="20 MB",
        retention="2 weeks",
        compression="zip",
        filter=lambda record: "performance" in record["extra"]
    )

    logger.info("Logging configuration initialized")


def get_gpr_logger():
    """Get logger specifically configured for GPR processing operations."""
    return logger.bind(gpr_processing=True)


def get_performance_logger():
    """Get logger for performance monitoring."""
    return logger.bind(performance=True)


class LoggerMixin:
    """Mixin class to provide structured logging capabilities."""

    @property
    def logger(self):
        """Get logger with class context."""
        return logger.bind(class_name=self.__class__.__name__)

    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation with context."""
        context = {"operation": operation, **kwargs}
        self.logger.bind(**context).info(f"Starting {operation}")

    def log_operation_complete(self, operation: str, duration_ms: float, **kwargs):
        """Log the completion of an operation with performance metrics."""
        context = {
            "operation": operation,
            "duration": duration_ms,
            "performance": True,
            **kwargs
        }
        self.logger.bind(**context).info(f"Completed {operation} in {duration_ms:.2f}ms")

    def log_operation_error(self, operation: str, error: Exception, **kwargs):
        """Log operation errors with full context."""
        context = {"operation": operation, **kwargs}
        self.logger.bind(**context).error(f"Error in {operation}: {error}", exc_info=True)

    def log_data_processing(self, dataset: str, records_processed: int, **kwargs):
        """Log data processing metrics."""
        context = {
            "gpr_processing": True,
            "dataset": dataset,
            "records_processed": records_processed,
            **kwargs
        }
        self.logger.bind(**context).info(
            f"Processed {records_processed} records from {dataset}"
        )

    def log_file_processing(self, file_path: str, file_size: int, processing_time: float):
        """Log file processing with detailed metrics."""
        context = {
            "gpr_processing": True,
            "file_path": file_path,
            "file_size_mb": file_size / (1024 * 1024),
            "processing_time_s": processing_time,
            "throughput_mbps": (file_size / (1024 * 1024)) / processing_time if processing_time > 0 else 0
        }
        self.logger.bind(**context).info(
            f"Processed file {file_path} ({file_size / (1024 * 1024):.2f}MB) in {processing_time:.2f}s"
        )

    def log_signal_processing(self, operation: str, signal_length: int, **kwargs):
        """Log signal processing operations."""
        context = {
            "gpr_processing": True,
            "operation": f"signal_{operation}",
            "signal_length": signal_length,
            **kwargs
        }
        self.logger.bind(**context).info(
            f"Signal processing: {operation} on {signal_length} samples"
        )

    def log_environmental_correlation(self, factor: str, correlation_value: float, **kwargs):
        """Log environmental correlation analysis results."""
        context = {
            "gpr_processing": True,
            "operation": "environmental_correlation",
            "environmental_factor": factor,
            "correlation": correlation_value,
            **kwargs
        }
        self.logger.bind(**context).info(
            f"Environmental correlation: {factor} = {correlation_value:.3f}"
        )

    def log_accuracy_assessment(self, method: str, accuracy: float, **kwargs):
        """Log accuracy assessment results."""
        context = {
            "gpr_processing": True,
            "operation": "accuracy_assessment",
            "method": method,
            "accuracy": accuracy,
            **kwargs
        }
        self.logger.bind(**context).info(
            f"Accuracy assessment: {method} achieved {accuracy:.3f} accuracy"
        )


def log_api_request(endpoint: str, method: str, **kwargs):
    """Log API request with context."""
    context = {"operation": "api_request", "endpoint": endpoint, "method": method, **kwargs}
    logger.bind(**context).info(f"{method} {endpoint}")


def log_api_response(endpoint: str, status_code: int, duration_ms: float, **kwargs):
    """Log API response with performance metrics."""
    context = {
        "operation": "api_response",
        "endpoint": endpoint,
        "status_code": status_code,
        "duration": duration_ms,
        "performance": True,
        **kwargs
    }
    logger.bind(**context).info(
        f"Response {status_code} for {endpoint} in {duration_ms:.2f}ms"
    )


def log_database_operation(operation: str, table: str, records: int = None, **kwargs):
    """Log database operations with metrics."""
    context = {
        "operation": f"db_{operation}",
        "table": table,
        "records": records,
        **kwargs
    }
    message = f"Database {operation} on {table}"
    if records is not None:
        message += f" ({records} records)"

    logger.bind(**context).info(message)


def log_batch_processing(batch_id: str, total_items: int, processed_items: int, **kwargs):
    """Log batch processing progress."""
    context = {
        "gpr_processing": True,
        "operation": "batch_processing",
        "batch_id": batch_id,
        "total_items": total_items,
        "processed_items": processed_items,
        "progress_percent": (processed_items / total_items * 100) if total_items > 0 else 0,
        **kwargs
    }
    logger.bind(**context).info(
        f"Batch {batch_id}: {processed_items}/{total_items} items processed "
        f"({processed_items / total_items * 100:.1f}%)"
    )