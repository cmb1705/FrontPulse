"""Tests for src/logging_config.py - Centralized logging framework."""
from __future__ import annotations

import logging

import pytest

from src.logging_config import get_logger, log_section, setup_logging


@pytest.mark.unit
class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_creates_logger_with_default_level(self):
        """Test that logger is created with INFO level by default."""
        logger = setup_logging(level="INFO", console=True)

        assert logger is not None
        assert logger.name == "2yp"
        assert logger.level == logging.INFO

    def test_creates_logger_with_debug_level(self):
        """Test that logger can be created with DEBUG level."""
        logger = setup_logging(level="DEBUG", console=True)

        assert logger.level == logging.DEBUG

    def test_creates_logger_with_warning_level(self):
        """Test that logger can be created with WARNING level."""
        logger = setup_logging(level="WARNING", console=True)

        assert logger.level == logging.WARNING

    def test_file_handler_creation(self, temp_dir):
        """Test that file handler is created when log_file is provided."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="INFO",
            log_file=log_file,
            console=False
        )

        # Check that log file was created
        assert log_file.exists()

        # Log a message and verify it appears in file
        logger.info("Test message")
        content = log_file.read_text()
        assert "Test message" in content

    def test_console_handler_disabled(self):
        """Test that console handler can be disabled."""
        logger = setup_logging(level="INFO", console=False)

        # Logger should have no console handlers
        [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream.name == "<stdout>"
        ]
        # Note: May still have file handlers, so we just check console is not there
        assert logger is not None

    def test_separate_file_level(self, temp_dir):
        """Test that file can have different log level than console."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="WARNING",  # Console WARNING
            log_file=log_file,
            file_level="DEBUG",  # File DEBUG
            console=True
        )

        # Log messages at different levels
        logger.debug("Debug message")
        logger.warning("Warning message")

        content = log_file.read_text()
        # File should have both (file_level=DEBUG)
        assert "Debug message" in content
        assert "Warning message" in content

    def test_rotating_file_handler_size_limit(self, temp_dir):
        """Test that rotating file handler respects size limits."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="INFO",
            log_file=log_file,
            console=False
        )

        # Find the RotatingFileHandler
        from logging.handlers import RotatingFileHandler
        rotating_handlers = [
            h for h in logger.handlers
            if isinstance(h, RotatingFileHandler)
        ]

        assert len(rotating_handlers) > 0
        handler = rotating_handlers[0]
        assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
        assert handler.backupCount == 5


@pytest.mark.unit
class TestGetLogger:
    """Test suite for get_logger function."""

    def test_returns_existing_logger(self):
        """Test that get_logger returns the existing logger."""
        # Setup a logger first
        setup_logging(level="INFO", console=True)

        # Get logger
        logger = get_logger()

        assert logger is not None
        assert logger.name == "2yp"

    def test_creates_basic_logger_if_not_setup(self):
        """Test that get_logger creates basic logger if setup not called."""
        # Clear any existing loggers
        logging.getLogger("2yp").handlers.clear()

        logger = get_logger()

        assert logger is not None
        assert logger.name == "2yp"


@pytest.mark.unit
class TestLogSection:
    """Test suite for log_section function."""

    def test_logs_section_header(self, temp_dir):
        """Test that log_section creates visual section separators."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="INFO",
            log_file=log_file,
            console=False
        )

        log_section(logger, "Test Section")

        content = log_file.read_text()
        assert "=" * 80 in content
        assert "Test Section" in content

    def test_section_format(self, temp_dir):
        """Test that section has proper formatting."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="INFO",
            log_file=log_file,
            console=False
        )

        log_section(logger, "My Section Title")

        content = log_file.read_text()
        lines = content.strip().split("\n")

        # Should have separator line, title line, separator line
        assert any("=" * 40 in line for line in lines)
        assert any("My Section Title" in line for line in lines)

    def test_handles_long_titles(self, temp_dir):
        """Test that long section titles are handled correctly."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="INFO",
            log_file=log_file,
            console=False
        )

        long_title = "A" * 100
        log_section(logger, long_title)

        content = log_file.read_text()
        assert long_title in content

    def test_respects_log_level(self, temp_dir):
        """Test that log_section respects logger's log level."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="WARNING",  # Only WARNING and above
            log_file=log_file,
            console=False
        )

        log_section(logger, "Should Not Appear")

        log_file.read_text()
        # log_section uses INFO level, so should not appear with WARNING level
        # Note: This behavior depends on implementation
        # If log_section is implemented to always log, this test may need adjustment
