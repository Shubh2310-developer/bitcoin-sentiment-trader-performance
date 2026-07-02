"""Shared pytest fixtures and configuration for the test suite."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR
