"""Time-derived feature engineering functions.

All features are derived directly from the ``Timestamp`` column with no
windowing required — there is no look-ahead risk.

All functions are pure: ``f(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame``.
"""

import numpy as np
import pandas as pd

from sentiment_trader_analytics.config import FeatureConfig


def add_time_of_day(df: pd.DataFrame, _config: FeatureConfig) -> pd.DataFrame:
    """Extract the UTC hour from the ``Timestamp`` column.

    Args:
        df: DataFrame with a UTC-aware ``Timestamp`` column.
        _config: Feature engineering configuration (unused).

    Returns:
        DataFrame with ``time_hour_utc`` column added (int8, [0, 23]).
    """
    result = df.copy()
    result["time_hour_utc"] = result["Timestamp"].dt.hour.astype(np.int8)
    return result


def add_day_of_week(df: pd.DataFrame, _config: FeatureConfig) -> pd.DataFrame:
    """Extract day-of-week and weekend flag from ``Timestamp``.

    ``time_day_of_week``: 0=Monday, 6=Sunday (int8).
    ``time_is_weekend``: True if day_of_week is 5 (Saturday) or 6 (Sunday).

    Args:
        df: DataFrame with a UTC-aware ``Timestamp`` column.
        _config: Feature engineering configuration (unused).

    Returns:
        DataFrame with ``time_day_of_week`` and ``time_is_weekend`` columns.
    """
    result = df.copy()
    result["time_day_of_week"] = result["Timestamp"].dt.dayofweek.astype(np.int8)
    result["time_is_weekend"] = result["time_day_of_week"].isin([5, 6])
    return result


def add_month(df: pd.DataFrame, _config: FeatureConfig) -> pd.DataFrame:
    """Extract the month from the ``Timestamp`` column.

    Args:
        df: DataFrame with a UTC-aware ``Timestamp`` column.
        _config: Feature engineering configuration (unused).

    Returns:
        DataFrame with ``time_month`` column added (int8, [1, 12]).
    """
    result = df.copy()
    result["time_month"] = result["Timestamp"].dt.month.astype(np.int8)
    return result
