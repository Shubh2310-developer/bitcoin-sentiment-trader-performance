"""Unit tests for the preprocessing layer.

Tests cover cleaning, deduplication, null handling, date extraction,
and safe merging with fan-out detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from sentiment_trader_analytics.config import PreprocessingConfig
from sentiment_trader_analytics.preprocessing import DataQualityError
from sentiment_trader_analytics.preprocessing.cleaning import (
    clean_fear_greed,
    clean_trader_history,
)
from sentiment_trader_analytics.preprocessing.merging import (
    extract_trade_date,
    merge_sentiment_and_trades,
)

# ── helpers ──────────────────────────────────────────────────────────


def _default_config(**kwargs: Any) -> PreprocessingConfig:
    return PreprocessingConfig(**kwargs)


def _utc_timestamps(n: int, start: str = "2024-01-01", freq: str = "D") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _fg_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _utc_timestamps(n),
            "value": np.random.default_rng(42).integers(0, 101, size=n).astype(np.int64),
            "classification": pd.Categorical(
                np.random.default_rng(99).choice(
                    ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"], size=n
                ),
                categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                ordered=True,
            ),
        }
    )


# ── clean_fear_greed ────────────────────────────────────────────────


class TestCleanFearGreed:
    """Tests for :func:`clean_fear_greed`."""

    def test_clean_fear_greed_drops_nulls(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("WARNING")
        df = _fg_df(10)
        df.loc[0, "value"] = None
        df.loc[1, "classification"] = None
        df.loc[2, "value"] = None
        df.loc[2, "classification"] = None

        result = clean_fear_greed(df, _default_config())

        assert len(result) == 7
        assert "FG nulls" in caplog.text
        assert "3 rows" in caplog.text

    def test_clean_fear_greed_deduplicates(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("WARNING")
        df = _fg_df(5)
        df = pd.concat([df, df.iloc[[2]].copy()], ignore_index=True)

        result = clean_fear_greed(df, _default_config())

        assert len(result) == 5
        assert "FG dupes" in caplog.text
        assert "1 duplicate" in caplog.text

    def test_asserts_utc_timezone(self) -> None:
        df = _fg_df(3)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        with pytest.raises(AssertionError, match="timezone-aware"):
            clean_fear_greed(df, _default_config())


# ── clean_trader_history ────────────────────────────────────────────


class TestCleanTraderHistory:
    """Tests for :func:`clean_trader_history`."""

    TRADER_COLS = [
        "Trade ID",
        "Account",
        "Timestamp",
        "Side",
        "Direction",
        "Size USD",
        "Execution Price",
        "Closed PnL",
        "Fee",
    ]

    @staticmethod
    def _th_df(n: int = 10) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "Trade ID": [f"TID_{i:04d}" for i in range(n)],
                "Account": [f"acc_{i % 3}" for i in range(n)],
                "Timestamp": _utc_timestamps(n),
                "Side": pd.Categorical(
                    rng.choice(["Long", "Short"], size=n),
                    categories=["Long", "Short"],
                    ordered=False,
                ),
                "Direction": pd.Categorical(
                    rng.choice(["Open", "Close"], size=n),
                    categories=["Open", "Close"],
                    ordered=False,
                ),
                "Size USD": rng.uniform(100, 10000, size=n).astype(np.float64),
                "Execution Price": rng.uniform(1000, 100000, size=n).astype(np.float64),
                "Closed PnL": rng.uniform(-1000, 1000, size=n).astype(np.float64),
                "Fee": rng.uniform(0, 50, size=n).astype(np.float64),
            }
        )

    def test_clean_trader_history_drops_null_account(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level("WARNING")
        df = self._th_df(10)
        df.loc[3, "Account"] = None

        result = clean_trader_history(df, _default_config())

        assert len(result) == 9
        assert "TH nulls" in caplog.text
        assert "Account" in caplog.text

    def test_clean_trader_history_exact_duplicate_trade_id(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level("INFO")
        df = self._th_df(5)
        df = pd.concat([df, df.iloc[[2]].copy()], ignore_index=True)

        result = clean_trader_history(df, _default_config())

        assert len(result) == 5
        assert "TH dupes" in caplog.text

    def test_clean_trader_history_nonexact_duplicate_trade_id(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level("WARNING")
        df = self._th_df(5)
        dup = df.iloc[[2]].copy()
        dup["Size USD"] = 99999.0  # different value
        df = pd.concat([df, dup], ignore_index=True)

        result = clean_trader_history(df, _default_config())

        assert len(result) == 5  # deduplicated to 5
        assert "non-exact duplicate" in caplog.text

    def test_asserts_utc_timezone(self) -> None:
        df = self._th_df(3)
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
        with pytest.raises(AssertionError, match="timezone-aware"):
            clean_trader_history(df, _default_config())

    def test_logs_before_after_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        df = self._th_df(10)
        df.loc[0, "Account"] = None
        df.loc[1, "Size USD"] = None

        result = clean_trader_history(df, _default_config())

        assert len(result) == 8
        assert "Started with 10 rows" in caplog.text
        assert "Completed:" in caplog.text


# ── extract_trade_date ──────────────────────────────────────────────


class TestExtractTradeDate:
    """Tests for :func:`extract_trade_date`."""

    def test_extract_trade_date(self) -> None:
        timestamps = pd.to_datetime(
            ["2024-01-15 14:30:00", "2024-06-01 00:00:00", "2024-12-31 23:59:59"],
            utc=True,
        )
        df = pd.DataFrame(
            {
                "Trade ID": ["A", "B", "C"],
                "Timestamp": timestamps,
            }
        )

        result = extract_trade_date(df)

        assert "trade_date_utc" in result.columns
        expected_dates = pd.Series(
            [
                pd.Timestamp("2024-01-15").date(),
                pd.Timestamp("2024-06-01").date(),
                pd.Timestamp("2024-12-31").date(),
            ]
        )
        pd.testing.assert_series_equal(
            result["trade_date_utc"].reset_index(drop=True),
            expected_dates,
            check_dtype=False,
            check_names=False,
        )
        assert result["Trade ID"].tolist() == ["A", "B", "C"]


# ── merge_sentiment_and_trades ──────────────────────────────────────


class TestMergeSentimentAndTrades:
    """Tests for :func:`merge_sentiment_and_trades`."""

    @staticmethod
    def _trades(n_dates: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        timestamps = _utc_timestamps(n_dates, freq="D")
        df = pd.DataFrame(
            {
                "Trade ID": [f"TID_{i:04d}" for i in range(n_dates)],
                "Account": ["acc_1"] * n_dates,
                "Timestamp": timestamps,
                "Size USD": rng.uniform(100, 10000, size=n_dates).astype(np.float64),
                "Execution Price": rng.uniform(1000, 100000, size=n_dates).astype(np.float64),
            }
        )
        return extract_trade_date(df)

    @staticmethod
    def _sentiment(n_dates: int = 5) -> pd.DataFrame:
        timestamps = _utc_timestamps(n_dates, freq="D")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": np.arange(10, 10 + n_dates, dtype=np.int64),
                "classification": pd.Categorical(
                    ["Neutral"] * n_dates,
                    categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                    ordered=True,
                ),
            }
        )

    def test_merge_left_join_preserves_trades(self) -> None:
        trades = self._trades(5)
        sentiment = self._sentiment(5)

        merged = merge_sentiment_and_trades(trades, sentiment, _default_config())

        assert len(merged) == 5
        assert merged["sentiment_missing"].sum() == 0

    def test_merge_sentiment_missing_flagged(self) -> None:
        trades = self._trades(5)
        sentiment = self._sentiment(3)  # only 3 sentiment dates for 5 trade dates

        merged = merge_sentiment_and_trades(trades, sentiment, _default_config())

        assert len(merged) == 5
        missing_mask = merged["sentiment_missing"]
        assert missing_mask.sum() == 2
        assert not missing_mask.iloc[0]
        assert missing_mask.iloc[-1]

    def test_merge_fan_out_detection(self) -> None:
        trades = self._trades(3)
        sentiment = self._sentiment(2)
        # Add a duplicate sentiment date to trigger fan-out
        sentiment = pd.concat([sentiment, sentiment.iloc[[0]].copy()], ignore_index=True)

        with pytest.raises(DataQualityError, match="fan-out"):
            merge_sentiment_and_trades(trades, sentiment, _default_config())
