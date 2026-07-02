"""Unit tests for the EDA module.

Tests cover descriptive statistics, missingness reporting, outlier
detection, the sentiment palette, and plotting function I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_trader_analytics.eda.summary_stats import (
    OutlierReport,
    compute_descriptive_stats,
    compute_missingness_report,
    compute_outlier_summary,
)
from sentiment_trader_analytics.visualization.plots import (
    SENTIMENT_PALETTE,
    plot_pnl_by_sentiment_boxplot,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small deterministic test DataFrame."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "sentiment_value": rng.uniform(0, 100, size=n),
            "Closed PnL": rng.uniform(-500, 500, size=n),
            "Leverage": rng.uniform(0.9, 1.1, size=n),
            "trader_win_rate_7d": rng.uniform(0, 1, size=n),
            "trader_pnl_rolling_7d": rng.uniform(-300, 300, size=n),
            "sentiment_classification": pd.Categorical(
                rng.choice(
                    ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                    size=n,
                ),
                categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                ordered=True,
            ),
            "string_col": ["abc"] * n,
        }
    )


@pytest.fixture
def missing_df() -> pd.DataFrame:
    """Create a DataFrame with known missing values."""
    n = 100
    df = pd.DataFrame(
        {
            "col_a": [1.0] * n,
            "col_b": [float("nan")] * n,
            "col_c": list(range(n)),
        }
    )
    df.loc[0:9, "col_a"] = float("nan")  # 10 NaN
    df.loc[0:4, "col_c"] = float("nan")  # 5 NaN
    return df


# ── Tests: compute_descriptive_stats ──────────────────────────────────


class TestDescriptiveStats:
    """Tests for :func:`compute_descriptive_stats`."""

    def test_compute_descriptive_stats_shape(self, sample_df: pd.DataFrame) -> None:
        columns = ["sentiment_value", "Closed PnL", "Leverage"]
        result = compute_descriptive_stats(sample_df, columns)
        assert result.shape[1] == len(
            columns
        ), f"Expected {len(columns)} feature columns, got {result.shape[1]}"
        expected_stats = {
            "count",
            "mean",
            "std",
            "min",
            "p25",
            "p50",
            "p75",
            "max",
            "skew",
            "kurtosis",
            "missing",
            "missing_pct",
        }
        assert (
            set(result.index) >= expected_stats
        ), f"Missing expected statistics: {expected_stats - set(result.index)}"

    def test_descriptive_stats_raises_on_no_valid_columns(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="None of the requested columns"):
            compute_descriptive_stats(sample_df, ["nonexistent"])


# ── Tests: compute_missingness_report ────────────────────────────────


class TestMissingnessReport:
    """Tests for :func:`compute_missingness_report`."""

    def test_compute_missingness_report_sorted(self, missing_df: pd.DataFrame) -> None:
        report = compute_missingness_report(missing_df)
        missing_counts = report["missing_count"].values
        for i in range(len(missing_counts) - 1):
            assert (
                missing_counts[i] >= missing_counts[i + 1]
            ), "Missingness report should be sorted descending by missing_count"

    def test_missingness_report_columns(self, missing_df: pd.DataFrame) -> None:
        report = compute_missingness_report(missing_df)
        assert set(report.columns) == {"column", "missing_count", "missing_pct", "dtype"}

    def test_missingness_report_known_values(self, missing_df: pd.DataFrame) -> None:
        report = compute_missingness_report(missing_df)
        col_a = report[report["column"] == "col_a"].iloc[0]
        assert col_a["missing_count"] == 10
        col_b = report[report["column"] == "col_b"].iloc[0]
        assert col_b["missing_count"] == 100
        col_c = report[report["column"] == "col_c"].iloc[0]
        assert col_c["missing_count"] == 5


# ── Tests: compute_outlier_summary ──────────────────────────────────


class TestOutlierSummary:
    """Tests for :func:`compute_outlier_summary`."""

    def test_compute_outlier_summary_iqr(self, sample_df: pd.DataFrame) -> None:
        report = compute_outlier_summary(sample_df, "sentiment_value", method="iqr")
        assert isinstance(report, OutlierReport)
        assert report.column == "sentiment_value"
        assert report.method == "iqr"
        assert report.n_outliers >= 0
        assert report.lower_bound < report.upper_bound

    def test_outlier_summary_with_inline_outliers(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        report = compute_outlier_summary(df, "x", method="iqr", config={"iqr_multiplier": 1.5})
        assert report.n_outliers == 1
        assert 100 in df.iloc[report.outlier_indices]["x"].values

    def test_outlier_summary_raises_on_nonexistent_column(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Column 'missing' not found"):
            compute_outlier_summary(sample_df, "missing")

    def test_outlier_summary_raises_on_non_numeric(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not numeric"):
            compute_outlier_summary(sample_df, "string_col")

    def test_outlier_summary_raises_on_unsupported_method(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported outlier detection method"):
            compute_outlier_summary(sample_df, "sentiment_value", method="zscore")


# ── Tests: sentiment palette ──────────────────────────────────────────


class TestSentimentPalette:
    """Tests for the :data:`SENTIMENT_PALETTE` constant."""

    def test_sentiment_palette_complete(self) -> None:
        expected_regimes = {"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}
        assert (
            set(SENTIMENT_PALETTE.keys()) == expected_regimes
        ), f"Missing regimes: {expected_regimes - set(SENTIMENT_PALETTE.keys())}"

    def test_sentiment_palette_no_duplicates(self) -> None:
        assert len(SENTIMENT_PALETTE) == len(set(SENTIMENT_PALETTE.keys()))

    def test_sentiment_palette_hex_colors(self) -> None:
        for color in SENTIMENT_PALETTE.values():
            assert color.startswith("#") and len(color) == 7


# ── Tests: plotting functions ─────────────────────────────────────────


class TestPlottingFunctions:
    """Tests for plotting function I/O behavior."""

    def test_pnl_boxplot_saves_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        output_path = tmp_path / "pnl_boxplot_test.png"
        plot_pnl_by_sentiment_boxplot(sample_df, str(output_path))
        assert output_path.exists(), "Boxplot PNG file should exist after save"
        assert output_path.stat().st_size > 0, "Boxplot PNG should be non-empty"
