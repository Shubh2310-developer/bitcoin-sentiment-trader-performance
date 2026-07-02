"""File I/O helper functions for the sentiment trader analytics pipeline.

Provides reusable utilities for reading, writing, and validating
files across the pipeline stages.
"""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Filesystem path to the directory.

    Returns:
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()
