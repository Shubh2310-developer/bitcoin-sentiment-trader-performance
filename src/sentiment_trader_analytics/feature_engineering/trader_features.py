"""Trader performance-derived feature engineering functions.

All functions operate per-account (``.groupby("Account")``) and use
strictly historical windows (``closed="left"``) to prevent look-ahead bias.

Rolling features naturally produce NaN for the first window of each
account's history — these rows are retained and flagged via the
``feature_cold_start`` column (added by :func:`flag_cold_start_rows`).
"""

import numpy as np
import pandas as pd

from sentiment_trader_analytics.config import FeatureConfig


def _ensure_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Leverage column if not present in the DataFrame.

    Leverage is approximated as:
        ``Leverage ≈ (Size Tokens * Execution Price) / Size USD``

    This is the standard way to back-compute leverage on perpetual
    futures exchanges where Size USD is the margin used and
    Size Tokens * Execution Price is the notional position value.

    If ``Size Tokens`` is not available either, defaults to 1.0.
    """
    if "Leverage" in df.columns:
        return df
    result = df.copy()
    if "Size Tokens" in result.columns and "Execution Price" in result.columns:
        notional = result["Size Tokens"] * result["Execution Price"]
        result["Leverage"] = np.where(
            result["Size USD"] > 0,
            notional / result["Size USD"],
            1.0,
        )
    else:
        result["Leverage"] = 1.0
    return result


def add_trader_win_rate(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add rolling 7-day win rate per account.

    Win rate is the fraction of closed trades (``Direction == "Close"``)
    with positive ``Closed PnL`` within the rolling window.
    Uses ``closed="left"`` to exclude current row.

    Args:
        df: DataFrame with ``Account``, ``Direction``, and ``Closed PnL`` columns.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``trader_win_rate_7d`` column added.
    """
    window = config.trader_rolling_windows["win_rate"]
    result = df.copy()
    is_close = result["Direction"].astype(str).str.contains("Close", case=False, na=False)
    result["_is_win"] = (is_close & (result["Closed PnL"] > 0)).astype(float)
    result["_is_trade"] = (is_close & (result["Closed PnL"].notna())).astype(float)
    result["trader_win_rate_7d"] = (
        result.groupby("Account")["_is_win"].transform(
            lambda s: s.rolling(window=window, min_periods=1, closed="left").sum()
        )
    ) / (
        result.groupby("Account")["_is_trade"].transform(
            lambda s: s.rolling(window=window, min_periods=1, closed="left").sum()
        )
    )
    result = result.drop(columns=["_is_win", "_is_trade"])
    return result


def add_trader_pnl_rolling(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add rolling total PnL per account at 7-day and 30-day windows.

    Args:
        df: DataFrame with ``Account`` and ``Closed PnL`` columns.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``trader_pnl_rolling_7d`` and ``trader_pnl_rolling_30d`` columns.
    """
    window_7d = config.trader_rolling_windows["pnl_7d"]
    window_30d = config.trader_rolling_windows["pnl_30d"]
    result = df.copy()
    result["trader_pnl_rolling_7d"] = result.groupby("Account")["Closed PnL"].transform(
        lambda s: s.rolling(window=window_7d, min_periods=1, closed="left").sum()
    )
    result["trader_pnl_rolling_30d"] = result.groupby("Account")["Closed PnL"].transform(
        lambda s: s.rolling(window=window_30d, min_periods=1, closed="left").sum()
    )
    return result


def add_trader_leverage_avg(df: pd.DataFrame, _config: FeatureConfig) -> pd.DataFrame:
    """Add rolling average leverage per account.

    If a ``Leverage`` column does not exist, it is derived from
    ``Size Tokens``, ``Execution Price``, and ``Size USD``.

    Uses a 5-row rolling window with ``closed="left"`` (per-account).
    The window size approximates a 24-hour lookback given typical
    trade density in the deduplicated dataset.

    Args:
        df: DataFrame with ``Account``, and trade size/price columns.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``trader_leverage_avg_24h`` column added.
    """
    window = 5
    result = _ensure_leverage(df.copy())
    result["trader_leverage_avg_24h"] = result.groupby("Account")["Leverage"].transform(
        lambda s: s.rolling(window=window, min_periods=1, closed="left").mean()
    )
    return result


def add_trader_pnl_volatility(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add rolling 14-day PnL volatility (standard deviation) per account.

    Args:
        df: DataFrame with ``Account`` and ``Closed PnL`` columns.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``trader_pnl_volatility_14d`` column added.
    """
    window = config.trader_rolling_windows["pnl_volatility"]
    result = df.copy()
    result["trader_pnl_volatility_14d"] = result.groupby("Account")["Closed PnL"].transform(
        lambda s: s.rolling(window=window, min_periods=1, closed="left").std()
    )
    return result


def add_trader_trade_count(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add rolling 7-day trade count per account.

    Args:
        df: DataFrame with ``Account`` column.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``trader_trade_count_7d`` column added.
    """
    window = config.trader_rolling_windows["trade_count"]
    result = df.copy()
    result["trader_trade_count_7d"] = result.groupby("Account")["Account"].transform(
        lambda s: s.rolling(window=window, min_periods=1, closed="left").count()
    )
    return result


def add_trader_avg_size(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add rolling 7-day mean position size per account.

    Args:
        df: DataFrame with ``Account`` and ``Size USD`` columns.
        config: Feature engineering configuration.

    Returns:
        DataFrame with ``trader_avg_size_usd_7d`` column added.
    """
    window = config.trader_rolling_windows["avg_size"]
    result = df.copy()
    result["trader_avg_size_usd_7d"] = result.groupby("Account")["Size USD"].transform(
        lambda s: s.rolling(window=window, min_periods=1, closed="left").mean()
    )
    return result


def flag_cold_start_rows(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Flag rows where rolling trader features have not yet warmed up.

    A row is flagged as cold-start if ANY of the rolling trader feature
    columns is NaN for that row. Cold-start rows are retained for
    completeness but should be excluded from downstream statistical
    analyses that require fully-warmed windows.

    Args:
        df: DataFrame with rolling trader feature columns.
        config: Feature engineering configuration (unused, kept for signature).

    Returns:
        DataFrame with ``feature_cold_start`` boolean column added.
    """
    _ = config
    result = df.copy()
    rolling_cols = [col for col in result.columns if col.startswith("trader_")]
    result["feature_cold_start"] = result[rolling_cols].isna().any(axis=1)
    return result
