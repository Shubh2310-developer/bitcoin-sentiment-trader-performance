"""Logging utilities for the sentiment trader analytics pipeline.

Provides a single point of configuration for structured logging
across all pipeline stages. Follows the standards in §15.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a logger with structured output.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level (default ``logging.INFO``).
        log_file: Optional path to a log file. If provided, output is
            written both to the file and to stdout.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler: logging.Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_pipeline_logger(name: str = "pipeline") -> logging.Logger:
    """Get a pipeline logger configured for standard pipeline usage.

    Args:
        name: Logger name (default ``pipeline``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return setup_logging(name, log_file="logs/pipeline.log")
