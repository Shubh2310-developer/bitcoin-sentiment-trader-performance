"""Unit tests for the feature engineering layer.

Tests cover sentiment features, trader features (with cold-start
handling and look-ahead bias checks), time features, and the full
feature composition.
"""

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

from sentiment_trader_analytics.config import FeatureConfig
from sentiment_trader_analytics.feature_engineering.sentiment_features import (
    add_sentiment_fear_greed_flags,
    add_sentiment_lag,
    add_sentiment_regime_encoding,
    add_sentiment_rolling_mean,
)
from sentiment_trader_analytics.feature_engineering.time_features import (
    add_day_of_week,
    add_month,
    add_time_of_day,
)
from sentiment_trader_analytics.feature_engineering.trader_features import (
    _ensure_leverage,
    add_trader_avg_size,
    add_trader_leverage_avg,
    add_trader_pnl_rolling,
    add_trader_pnl_volatility,
    add_trader_trade_count,
    add_trader_win_rate,
    flag_cold_start_rows,
)

# ── helpers ──────────────────────────────────────────────────────────


def _default_config(**kwargs: Any) -> FeatureConfig:
    return FeatureConfig(**kwargs)


def _utc_timestamps(n: int, start: str = "2024-01-01", freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _make_sentiment_df(n: int = 20) -> pd.DataFrame:
    """Create a simple sentiment-only test DataFrame."""
    regimes = pd.Categorical(
        np.random.default_rng(42).choice(
            ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
            size=n,
        ),
        categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
        ordered=True,
    )
    return pd.DataFrame(
        {
            "Timestamp": _utc_timestamps(n, freq="D"),
            "sentiment_value": np.random.default_rng(99).uniform(0, 100, size=n),
            "sentiment_classification": regimes,
            "Account": ["acc_1"] * n,
            "Direction": pd.Categorical(
                np.random.default_rng(1).choice(["Open", "Close"], size=n),
            ),
            "Closed PnL": np.random.default_rng(2).uniform(-500, 500, size=n),
            "Size USD": np.random.default_rng(3).uniform(100, 10000, size=n),
            "Size Tokens": np.random.default_rng(4).uniform(1, 100, size=n),
            "Execution Price": np.random.default_rng(5).uniform(1000, 100000, size=n),
        }
    )


def _make_trader_df(
    n_per_account: int = 14,
    n_accounts: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a multi-account trader test DataFrame.

    Each account gets sequential hourly timestamps with known patterns
    for win/loss and PnL.
    """
    rng = np.random.default_rng(seed)
    regimes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    rows = []
    for acct_idx in range(n_accounts):
        account = f"acc_{acct_idx}"
        timestamps = _utc_timestamps(n_per_account, freq="h")
        for i, ts in enumerate(timestamps):
            direction = "Close" if i % 2 == 0 else "Open"
            pnl = 100.0 if i % 2 == 0 else -50.0
            rows.append(
                {
                    "Account": account,
                    "Timestamp": ts,
                    "Direction": direction,
                    "Closed PnL": pnl if direction == "Close" else np.nan,
                    "Size USD": rng.uniform(100, 10000),
                    "Size Tokens": rng.uniform(1, 100),
                    "Execution Price": rng.uniform(1000, 100000),
                    "sentiment_value": rng.uniform(0, 100),
                    "sentiment_classification": rng.choice(regimes),
                }
            )
    df = pd.DataFrame(rows)
    df["sentiment_classification"] = pd.Categorical(
        df["sentiment_classification"],
        categories=regimes,
        ordered=True,
    )
    return df


# ── Sentiment Feature Tests ─────────────────────────────────────────


class TestSentimentLag:
    """Tests for :func:`add_sentiment_lag`."""

    def test_sentiment_lag_is_prior_day(self) -> None:
        df = _make_sentiment_df(10)
        config = _default_config()
        result = add_sentiment_lag(df, config)
        # Lag(1) for row i should equal sentiment_value at row i-1
        expected = df["sentiment_value"].shift(1)
        tm.assert_series_equal(
            result["sentiment_value_lag_1d"],
            expected,
            check_names=False,
            check_dtype=False,
        )


class TestSentimentEncoding:
    """Tests for :func:`add_sentiment_regime_encoding`."""

    def test_sentiment_encoding_maps_correctly(self) -> None:
        df = _make_sentiment_df(10)
        config = _default_config()
        result = add_sentiment_regime_encoding(df, config)
        expected_map = {"Extreme Fear": 0, "Fear": 1, "Neutral": 2, "Greed": 3, "Extreme Greed": 4}
        for regime, expected_val in expected_map.items():
            mask = result["sentiment_classification"] == regime
            if mask.any():
                assert (
                    result.loc[mask, "sentiment_regime_encoded"] == expected_val
                ).all(), f"Regime {regime} should map to {expected_val}"


class TestSentimentRollingMean:
    """Tests for :func:`add_sentiment_rolling_mean`."""

    def test_rolling_mean_uses_closed_left(self) -> None:
        df = _make_sentiment_df(10)
        config = _default_config()
        result = add_sentiment_rolling_mean(df, config)
        # With closed="left", the rolling mean at row 0 should be NaN
        # (no prior rows in the window)
        assert pd.isna(
            result["sentiment_value_rolling_7d"].iloc[0]
        ), "First row should be NaN with closed='left' (no prior rows)"
        # Row 1 should equal the sentiment_value at row 0 only
        expected_1 = df["sentiment_value"].iloc[0]
        assert np.isclose(
            result["sentiment_value_rolling_7d"].iloc[1], expected_1
        ), f"Row 1 expected {expected_1}, got {result['sentiment_value_rolling_7d'].iloc[1]}"


class TestFearGreedFlags:
    """Tests for :func:`add_sentiment_fear_greed_flags`."""

    def test_fear_flag_true_for_fear_and_extreme_fear(self) -> None:
        df = _make_sentiment_df(20)
        config = _default_config()
        result = add_sentiment_fear_greed_flags(df, config)

        fear_mask = result["sentiment_classification"].isin(["Fear", "Extreme Fear"])
        not_fear_mask = ~result["sentiment_classification"].isin(["Fear", "Extreme Fear"])

        if fear_mask.any():
            assert result.loc[
                fear_mask, "sentiment_is_fear"
            ].all(), "All Fear/Extreme Fear rows should have sentiment_is_fear=True"
        if not_fear_mask.any():
            assert not result.loc[
                not_fear_mask, "sentiment_is_fear"
            ].any(), "Non-Fear rows should have sentiment_is_fear=False"

    def test_greed_flag_true_for_greed_and_extreme_greed(self) -> None:
        df = _make_sentiment_df(20)
        config = _default_config()
        result = add_sentiment_fear_greed_flags(df, config)

        greed_mask = result["sentiment_classification"].isin(["Greed", "Extreme Greed"])
        not_greed_mask = ~result["sentiment_classification"].isin(["Greed", "Extreme Greed"])

        if greed_mask.any():
            assert result.loc[
                greed_mask, "sentiment_is_greed"
            ].all(), "All Greed/Extreme Greed rows should have sentiment_is_greed=True"
        if not_greed_mask.any():
            assert not result.loc[
                not_greed_mask, "sentiment_is_greed"
            ].any(), "Non-Greed rows should have sentiment_is_greed=False"


# ── Trader Feature Tests ────────────────────────────────────────────


class TestTraderWinRate:
    """Tests for :func:`add_trader_win_rate`."""

    def test_rolling_win_rate_does_not_include_current_row(self) -> None:
        df = _make_trader_df(n_per_account=10, n_accounts=1, seed=100)
        # Force known pattern: row 0 = close+win, row 1 = close+loss, row 2 = close+win, etc.
        df["Direction"] = "Close"
        df["Closed PnL"] = [100.0, -50.0, 100.0, -50.0, 100.0, -50.0, 100.0, -50.0, 100.0, -50.0]

        config = _default_config()
        result = add_trader_win_rate(df, config)

        # Row 0: no prior rows → NaN
        assert pd.isna(result["trader_win_rate_7d"].iloc[0])

        # Row 1: only row 0 is prior → win rate = 1.0 (row 0 was a win)
        assert np.isclose(
            result["trader_win_rate_7d"].iloc[1], 1.0
        ), f"Expected win rate 1.0, got {result['trader_win_rate_7d'].iloc[1]}"

        # Row 2: rows 0-1 prior → 1 win / 2 total = 0.5
        assert np.isclose(
            result["trader_win_rate_7d"].iloc[2], 0.5
        ), f"Row 2 win rate should be 0.5, got {result['trader_win_rate_7d'].iloc[2]}"

    def test_win_rate_range(self) -> None:
        df = _make_trader_df(n_per_account=14, n_accounts=3, seed=200)
        config = _default_config()
        result = add_trader_win_rate(df, config)
        valid = result["trader_win_rate_7d"].dropna()
        assert (valid >= 0.0).all(), "Win rate must be >= 0.0"
        assert (valid <= 1.0).all(), "Win rate must be <= 1.0"


class TestEnsureLeverage:
    """Tests for :func:`_ensure_leverage`."""

    def test_ensure_leverage_column_present(self) -> None:
        """Should return df unchanged if Leverage column already exists."""
        df = pd.DataFrame({"Leverage": [1.0, 2.0, 3.0]})
        result = _ensure_leverage(df)
        pd.testing.assert_frame_equal(result, df)

    def test_ensure_leverage_with_size_tokens(self) -> None:
        """Should compute leverage from Size Tokens and Execution Price."""
        df = pd.DataFrame(
            {
                "Size Tokens": [100.0, 200.0],
                "Execution Price": [10.0, 20.0],
                "Size USD": [500.0, 1000.0],
            }
        )
        result = _ensure_leverage(df)
        assert "Leverage" in result.columns
        expected = ((df["Size Tokens"] * df["Execution Price"]) / df["Size USD"]).values
        assert result["Leverage"].values == pytest.approx(expected)

    def test_ensure_leverage_without_size_tokens(self) -> None:
        """Should default to 1.0 if neither Leverage nor Size Tokens exist."""
        df = pd.DataFrame(
            {
                "Size USD": [500.0, 1000.0],
                "Closed PnL": [50.0, -20.0],
            }
        )
        result = _ensure_leverage(df)
        assert "Leverage" in result.columns
        assert (result["Leverage"] == 1.0).all()


class TestTraderColdStart:
    """Tests for cold-start row flagging."""

    def test_cold_start_flagged(self) -> None:
        df = _make_trader_df(n_per_account=14, n_accounts=2, seed=300)
        config = _default_config()

        # Apply rolling trader features
        df = add_trader_win_rate(df, config)
        df = add_trader_pnl_rolling(df, config)
        df = add_trader_pnl_volatility(df, config)
        df = add_trader_trade_count(df, config)
        df = add_trader_avg_size(df, config)

        result = flag_cold_start_rows(df, config)

        # The first row of each account should be cold-start (all rolling features are NaN)
        for account in result["Account"].unique():
            account_df = result[result["Account"] == account].reset_index(drop=True)
            # Row 0 should always be cold-start
            assert account_df["feature_cold_start"].iloc[
                0
            ], f"First row of account {account} should be cold-start"


# ── Time Feature Tests ──────────────────────────────────────────────


class TestTimeFeatures:
    """Tests for time-derived features."""

    def test_time_features_correct_utc(self) -> None:
        timestamps = pd.to_datetime(
            ["2024-01-15 14:30:00", "2024-06-01 00:00:00", "2024-12-31 23:59:59"],
            utc=True,
        )
        # Shift the second timestamp to a Saturday (2024-06-01 is a Saturday)
        df = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Account": ["acc_1"] * 3,
                "Direction": ["Close"] * 3,
                "Closed PnL": [100.0, -50.0, 25.0],
                "Size USD": [1000.0, 2000.0, 3000.0],
                "Size Tokens": [1.0, 2.0, 3.0],
                "Execution Price": [10000.0, 20000.0, 30000.0],
                "sentiment_value": [50.0, 60.0, 70.0],
                "sentiment_classification": pd.Categorical(
                    ["Neutral", "Greed", "Fear"],
                    categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                    ordered=True,
                ),
            }
        )

        config = _default_config()
        result = add_time_of_day(df, config)
        result = add_day_of_week(result, config)
        result = add_month(result, config)

        # Row 0: 2024-01-15 14:30 UTC → hour=14, month=1
        assert result["time_hour_utc"].iloc[0] == 14, "Hour should be 14 UTC"
        assert result["time_month"].iloc[0] == 1, "Month should be 1"

        # Check day of week: 2024-01-15 was a Monday → dayofweek=0
        assert result["time_day_of_week"].iloc[0] == 0, "2024-01-15 should be Monday (0)"

        # Row 1: 2024-06-01 was a Saturday → dayofweek=5, is_weekend=True
        assert result["time_day_of_week"].iloc[1] == 5, "2024-06-01 should be Saturday (5)"
        assert result["time_is_weekend"].iloc[1], "Saturday should be weekend"

        # Row 2: 2024-12-31 was a Tuesday → dayofweek=1, is_weekend=False
        assert result["time_day_of_week"].iloc[2] == 1, "2024-12-31 should be Tuesday (1)"
        assert not result["time_is_weekend"].iloc[2], "Tuesday should not be weekend"


# ── Composition Tests ────────────────────────────────────────────────


class TestFeatureComposition:
    """Tests for the full feature composition pipeline."""

    EXPECTED_FEATURE_COLUMNS = {
        "sentiment_value_lag_1d",
        "sentiment_regime_encoded",
        "sentiment_value_rolling_7d",
        "sentiment_is_fear",
        "sentiment_is_greed",
        "trader_win_rate_7d",
        "trader_pnl_rolling_7d",
        "trader_pnl_rolling_30d",
        "trader_leverage_avg_24h",
        "trader_pnl_volatility_14d",
        "trader_trade_count_7d",
        "trader_avg_size_usd_7d",
        "feature_cold_start",
        "time_hour_utc",
        "time_day_of_week",
        "time_is_weekend",
        "time_month",
    }

    def test_feature_composition_is_additive(self) -> None:
        df = _make_trader_df(n_per_account=20, n_accounts=2, seed=500)
        config = _default_config()

        feature_functions = [
            add_sentiment_lag,
            add_sentiment_regime_encoding,
            add_sentiment_rolling_mean,
            add_sentiment_fear_greed_flags,
            add_trader_win_rate,
            add_trader_pnl_rolling,
            add_trader_leverage_avg,
            add_trader_pnl_volatility,
            add_trader_trade_count,
            add_trader_avg_size,
            flag_cold_start_rows,
            add_time_of_day,
            add_day_of_week,
            add_month,
        ]

        result = reduce(lambda d, fn: fn(d, config), feature_functions, df)

        result_features = set(result.columns)
        missing = self.EXPECTED_FEATURE_COLUMNS - result_features
        extra = result_features - set(df.columns) - self.EXPECTED_FEATURE_COLUMNS

        assert not missing, f"Missing expected feature columns: {missing}"
        if extra:
            # _ensure_leverage may add "Leverage" column — that's expected
            allowed_extra = {"Leverage"}
            assert extra <= allowed_extra, f"Unexpected extra columns: {extra - allowed_extra}"
