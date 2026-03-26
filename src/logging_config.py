"""
Centralized logging configuration for the 2YP pipeline.

Provides consistent logging across all modules with configurable verbosity
and file output with rotation.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Global logger instance
_logger: logging.Logger | None = None


def get_logger(name: str = "2yp") -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (defaults to "2yp")

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    console: bool = True,
    file_level: str | None = None,
) -> logging.Logger:
    """
    Configure logging for the 2YP pipeline.

    Args:
        level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional). If provided, enables file logging.
        console: Whether to enable console logging (default: True)
        file_level: File logging level (defaults to same as console level)

    Returns:
        Configured root logger for "2yp"

    Example:
        >>> from src.logging_config import setup_logging
        >>> logger = setup_logging(level="DEBUG", log_file=Path("data/psc/out/logs/pipeline.log"))
        >>> logger.info("Pipeline started")
    """
    global _logger

    # Create root logger for 2yp
    logger = logging.getLogger("2yp")
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler: 10MB max, keep 5 backup files
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )

        file_handler.setLevel(
            getattr(logging, (file_level or level).upper())
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _logger = logger
    return logger


def set_level(level: str) -> None:
    """
    Change the logging level for all console handlers.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = get_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            handler.setLevel(getattr(logging, level.upper()))


def log_section(logger: logging.Logger, title: str) -> None:
    """
    Log a section header for visual separation in logs.

    Args:
        logger: Logger instance
        title: Section title

    Example:
        >>> log_section(logger, "Ingest Phase")
    """
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)
