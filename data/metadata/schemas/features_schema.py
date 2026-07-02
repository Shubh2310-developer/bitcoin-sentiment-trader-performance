"""Pandera DataFrameSchema for the engineered feature store.

Validates all features produced by the feature engineering pipeline:
sentiment-derived, trader-performance, and time-based features.

Numeric features are checked for valid ranges. Boolean features are
enforced as bool dtype. The ``sentiment_regime_encoded`` column is
constrained to the set {0, 1, 2, 3, 4}.
"""

import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema


def _in_range_or_nan(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Check that non-NaN values are within [lo, hi]."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return pd.Series(True, index=s.index)
    return non_null.between(lo, hi, inclusive="both")


def _ge_zero_or_nan(s: pd.Series) -> pd.Series:
    """Check that non-NaN values are >= 0."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return pd.Series(True, index=s.index)
    return non_null >= 0.0


def _gt_zero_or_nan(s: pd.Series) -> pd.Series:
    """Check that non-NaN values are > 0."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return pd.Series(True, index=s.index)
    return non_null > 0.0


features_schema = DataFrameSchema(  # type: ignore[no-untyped-call]
    columns={
        # ── Sentiment features ──────────────────────────────────────
        "sentiment_value_lag_1d": Column(
            np.float64,
            checks=Check(_in_range_or_nan, lo=0.0, hi=100.0, element_wise=False),
            nullable=True,
            description="Sentiment value from 1 day prior",
        ),
        "sentiment_regime_encoded": Column(
            np.int8,
            checks=Check.isin([0, 1, 2, 3, 4]),
            nullable=False,
            description="Ordinal encoding of sentiment regime",
        ),
        "sentiment_value_rolling_7d": Column(
            np.float64,
            checks=Check(_in_range_or_nan, lo=0.0, hi=100.0, element_wise=False),
            nullable=True,
            description="7-day rolling mean of sentiment value",
        ),
        "sentiment_is_fear": Column(
            bool,
            nullable=False,
            description="True when regime is Fear or Extreme Fear",
        ),
        "sentiment_is_greed": Column(
            bool,
            nullable=False,
            description="True when regime is Greed or Extreme Greed",
        ),
        # ── Trader features ─────────────────────────────────────────
        "trader_win_rate_7d": Column(
            np.float64,
            checks=Check(_in_range_or_nan, lo=0.0, hi=1.0, element_wise=False),
            nullable=True,
            description="7-day rolling win rate [0.0, 1.0]",
        ),
        "trader_pnl_rolling_7d": Column(
            np.float64,
            nullable=True,
            description="7-day rolling sum of Closed PnL (USD)",
        ),
        "trader_pnl_rolling_30d": Column(
            np.float64,
            nullable=True,
            description="30-day rolling sum of Closed PnL (USD)",
        ),
        "trader_leverage_avg_24h": Column(
            np.float64,
            checks=Check(_gt_zero_or_nan, element_wise=False),
            nullable=True,
            description="24-hour rolling average leverage",
        ),
        "trader_pnl_volatility_14d": Column(
            np.float64,
            checks=Check(_ge_zero_or_nan, element_wise=False),
            nullable=True,
            description="14-day rolling std dev of Closed PnL",
        ),
        "trader_trade_count_7d": Column(
            np.float64,
            checks=Check(_ge_zero_or_nan, element_wise=False),
            nullable=True,
            description="7-day rolling trade count",
        ),
        "trader_avg_size_usd_7d": Column(
            np.float64,
            checks=Check(_ge_zero_or_nan, element_wise=False),
            nullable=True,
            description="7-day rolling mean position size (USD)",
        ),
        # ── Cold-start flag ─────────────────────────────────────────
        "feature_cold_start": Column(
            bool,
            nullable=False,
            description="True when rolling trader features have not warmed up",
        ),
        # ── Time features ───────────────────────────────────────────
        "time_hour_utc": Column(
            np.int8,
            checks=Check.in_range(0, 23),
            nullable=False,
            description="UTC hour of trade [0, 23]",
        ),
        "time_day_of_week": Column(
            np.int8,
            checks=Check.in_range(0, 6),
            nullable=False,
            description="Day of week [0=Monday, 6=Sunday]",
        ),
        "time_is_weekend": Column(
            bool,
            nullable=False,
            description="True if trade occurred on weekend",
        ),
        "time_month": Column(
            np.int8,
            checks=Check.in_range(1, 12),
            nullable=False,
            description="Month of trade [1, 12]",
        ),
    },
)
