"""Unit tests for the utility modules.

Tests cover logging setup and behavior.
"""

import logging

from sentiment_trader_analytics.utils.logging_utils import setup_logging


class TestLoggingUtils:
    """Tests for :func:`setup_logging`."""

    def test_setup_logging_returns_logger(self) -> None:
        logger = setup_logging(__name__)
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__

    def test_setup_logging_level(self) -> None:
        logger = setup_logging("test_module")
        assert logger.level in (logging.DEBUG, logging.INFO, 0)

    def test_setup_logging_with_handler(self) -> None:
        logger = setup_logging("test_handler")
        assert len(logger.handlers) >= 1
        has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert has_stream, "Expected at least one StreamHandler"
