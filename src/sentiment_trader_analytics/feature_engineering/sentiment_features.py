"""Sentiment-derived feature engineering functions.

All functions are pure (no side effects, no I/O) and follow the
signature ``f(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame``.

Each function adds one or more columns to the input DataFrame and
returns a new DataFrame (no in-place mutation).
"""

import numpy as np
import pandas as pd

from sentiment_trader_analytics.config import FeatureConfig


def add_sentiment_lag(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add a 1-day lagged sentiment value column.

    Uses ``.shift(config.sentiment_lag_days)`` which is safe from
    look-ahead bias — the lagged value comes strictly from a prior row.

    Args:
        df: DataFrame with a ``sentiment_value`` column.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``sentiment_value_lag_1d`` column added.
    """
    result = df.copy()
    result["sentiment_value_lag_1d"] = result["sentiment_value"].shift(config.sentiment_lag_days)
    return result


def add_sentiment_regime_encoding(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Ordinal-encode the sentiment classification column.

    Mapping is defined in ``config.sentiment_regime_encoding``:
        Extreme Fear → 0, Fear → 1, Neutral → 2, Greed → 3, Extreme Greed → 4

    Rows with missing classification produce NaN in the encoding.

    Args:
        df: DataFrame with a ``sentiment_classification`` column.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``sentiment_regime_encoded`` column added.
    """
    result = df.copy()
    encoded = result["sentiment_classification"].map(config.sentiment_regime_encoding)
    # Rows with missing sentiment classification (e.g., no sentiment data
    # for that date) produce NaN. We fill with Neutral (2) as a neutral
    # default rather than leaving a nullable int column that pandas cannot
    # represent as int8.
    if encoded.isna().any():
        encoded = encoded.fillna(2.0)
    result["sentiment_regime_encoded"] = encoded.astype(np.int8)
    return result


def add_sentiment_rolling_mean(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add a 7-day rolling mean of the sentiment value.

    Uses ``closed="left"`` to exclude the current row from the window,
    preventing any look-ahead bias.

    Args:
        df: DataFrame with a ``sentiment_value`` column.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``sentiment_value_rolling_7d`` column added.
    """
    result = df.copy()
    result["sentiment_value_rolling_7d"] = (
        result["sentiment_value"]
        .rolling(window=config.sentiment_rolling_window, min_periods=1, closed="left")
        .mean()
    )
    return result


def add_sentiment_fear_greed_flags(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add boolean flags for Fear and Greed sentiment regimes.

    ``sentiment_is_fear`` is True when classification is Fear or Extreme Fear.
    ``sentiment_is_greed`` is True when classification is Greed or Extreme Greed.

    Args:
        df: DataFrame with a ``sentiment_classification`` column.
        config: Feature engineering configuration (unused, kept for signature).

    Returns:
        DataFrame with ``sentiment_is_fear`` and ``sentiment_is_greed`` columns added.
    """
    _ = config  # kept for uniform function signature
    result = df.copy()
    result["sentiment_is_fear"] = result["sentiment_classification"].isin(["Fear", "Extreme Fear"])
    result["sentiment_is_greed"] = result["sentiment_classification"].isin(
        ["Greed", "Extreme Greed"]
    )
    return result
