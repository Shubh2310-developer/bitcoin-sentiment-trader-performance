"""Unit tests for the ingestion layer.

Tests cover Fear & Greed loader, Trader History loader, and checksum
lineage generation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_trader_analytics.config import IngestionConfig
from sentiment_trader_analytics.ingestion.fear_greed_loader import load_fear_greed_index
from sentiment_trader_analytics.ingestion.trader_history_loader import (
    load_trader_history,
)

FIXTURES = Path("tests/fixtures")


def _fg_config(tmp_path: Path, filename: str) -> IngestionConfig:
    return IngestionConfig(
        fear_greed_path=str(FIXTURES / filename),
        lineage_output_dir=str(tmp_path / "lineage"),
    )


def _th_config(tmp_path: Path, filename: str, chunk_size: int = 1000) -> IngestionConfig:
    return IngestionConfig(
        trader_history_path=str(FIXTURES / filename),
        lineage_output_dir=str(tmp_path / "lineage"),
        chunk_size=chunk_size,
    )


class TestFearGreedLoader:
    """Tests for :func:`load_fear_greed_index`."""

    def test_fear_greed_loader_happy_path(self, tmp_path: Path) -> None:
        config = _fg_config(tmp_path, "fear_greed_sample.csv")
        df = load_fear_greed_index(config)

        assert len(df) == 30
        assert list(df.columns) == ["timestamp", "value", "classification", "date"]

        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

        assert df["value"].dtype == np.int64

        assert isinstance(df["classification"].dtype, pd.CategoricalDtype)
        assert df["classification"].cat.ordered

        assert "source_file" in df.attrs
        assert df.attrs["row_count"] == 30

    def test_fear_greed_loader_missing_file(self, tmp_path: Path) -> None:
        config = _fg_config(tmp_path, "nonexistent.csv")
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_fear_greed_index(config)

    def test_fear_greed_loader_wrong_columns(self, tmp_path: Path) -> None:
        config = _fg_config(tmp_path, "fear_greed_bad_columns.csv")
        with pytest.raises(ValueError, match="missing expected columns"):
            load_fear_greed_index(config)


class TestTraderHistoryLoader:
    """Tests for :func:`load_trader_history`."""

    def test_trader_history_loader_happy_path(self, tmp_path: Path) -> None:
        config = _th_config(tmp_path, "trader_history_sample.csv")
        df = load_trader_history(config)

        assert len(df) == 200
        assert "Trade ID" in df.columns
        assert "Timestamp" in df.columns

        assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
        assert df["Timestamp"].dt.tz is not None
        assert str(df["Timestamp"].dt.tz) == "UTC"

        assert isinstance(df["Side"].dtype, pd.CategoricalDtype)
        assert isinstance(df["Direction"].dtype, pd.CategoricalDtype)

        float_cols = ["Size USD", "Execution Price", "Closed PnL", "Fee"]
        for col in float_cols:
            assert df[col].dtype == np.float64, f"{col} is {df[col].dtype}"

        assert df["Account"].dtype == np.dtype("object")

        assert "source_file" in df.attrs
        assert df.attrs["row_count"] == 200
        assert df.attrs["chunk_count"] > 0

    def test_trader_history_loader_naive_datetime_rejected(self, tmp_path: Path) -> None:
        config = _th_config(tmp_path, "trader_history_sample.csv")
        df = load_trader_history(config)

        assert df["Timestamp"].dtype.kind == "M"
        assert df["Timestamp"].dt.tz is not None
        assert str(df["Timestamp"].dt.tz) == "UTC"

        sample = df["Timestamp"].iloc[0]
        assert sample.tz is not None
        assert str(sample.tz) == "UTC"

    def test_trader_history_loader_missing_file(self, tmp_path: Path) -> None:
        config = _th_config(tmp_path, "nonexistent.csv")
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_trader_history(config)


class TestChecksum:
    """Tests for checksum lineage file generation."""

    def test_checksum_written(self, tmp_path: Path) -> None:
        config = _fg_config(tmp_path, "fear_greed_sample.csv")
        load_fear_greed_index(config)

        lineage_dir = tmp_path / "lineage"
        assert lineage_dir.exists()

        json_files = list(lineage_dir.glob("fear_greed_*.json"))
        assert len(json_files) >= 1

        with open(json_files[0]) as f:
            lineage = json.load(f)

        assert "sha256" in lineage
        assert len(lineage["sha256"]) == 64
        assert lineage["dataset"] == "fear_greed"
        assert lineage["row_count"] == 30
        assert "source_file" in lineage

    def test_trader_history_checksum_written(self, tmp_path: Path) -> None:
        config = _th_config(tmp_path, "trader_history_sample.csv")
        load_trader_history(config)

        lineage_dir = tmp_path / "lineage"
        assert lineage_dir.exists()

        json_files = list(lineage_dir.glob("trader_history_*.json"))
        assert len(json_files) >= 1

        with open(json_files[0]) as f:
            lineage = json.load(f)

        assert "sha256" in lineage
        assert len(lineage["sha256"]) == 64
        assert lineage["dataset"] == "trader_history"
        assert lineage["row_count"] == 200
        assert "chunk_count" in lineage
