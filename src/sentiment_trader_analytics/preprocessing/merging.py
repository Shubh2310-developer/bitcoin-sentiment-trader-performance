"""Dataset merging utilities for joining sentiment and trader datasets.

Provides a safe left-join with fan-out detection to prevent accidental
row multiplication from duplicate join keys.
"""

import logging

import pandas as pd

from sentiment_trader_analytics.config import PreprocessingConfig
from sentiment_trader_analytics.preprocessing import DataQualityError

logger = logging.getLogger(__name__)


def extract_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a UTC calendar date column from the ``Timestamp`` column.

    Adds ``trade_date_utc`` as a ``datetime.date`` column extracted from
    each row's ``Timestamp`` (UTC).

    Args:
        df: DataFrame with a UTC-aware ``Timestamp`` column.

    Returns:
        DataFrame with the appended ``trade_date_utc`` column.
    """
    df = df.copy()
    df["trade_date_utc"] = df["Timestamp"].dt.date
    return df


def merge_sentiment_and_trades(
    trades_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Left-join trades on UTC calendar date against sentiment records.

    The join is safe by construction — if duplicate sentiment dates
    cause a row-count increase beyond ``fan_out_tolerance_fraction`` the
    pipeline halts.

    Args:
        trades_df: Cleaned trades DataFrame with ``trade_date_utc`` column.
        sentiment_df: Cleaned sentiment DataFrame with ``timestamp`` column.
        config: Preprocessing configuration containing fan-out tolerance.

    Returns:
        Merged DataFrame with ``sentiment_missing`` boolean column.

    Raises:
        DataQualityError: If fan-out exceeds configured tolerance.
    """
    pre_join_count = len(trades_df)

    sentiment = sentiment_df.copy()
    sentiment["sentiment_date"] = sentiment["timestamp"].dt.date

    sentiment_map = sentiment[["sentiment_date", "value", "classification"]].rename(
        columns={"value": "sentiment_value", "classification": "sentiment_classification"}
    )

    merged = trades_df.merge(
        sentiment_map,
        left_on="trade_date_utc",
        right_on="sentiment_date",
        how="left",
    )

    merged["sentiment_missing"] = merged["sentiment_value"].isna()

    matched = int((~merged["sentiment_missing"]).sum())
    match_rate = (matched / pre_join_count * 100) if pre_join_count > 0 else 0.0
    unmatched = pre_join_count - matched
    logger.info("Merge | Match rate: %.2f%% (%d/%d)", match_rate, matched, pre_join_count)
    logger.info("Merge | Unmatched trades: %d", unmatched)

    max_allowed = int(pre_join_count * (1.0 + config.fan_out_tolerance_fraction))
    if len(merged) > max_allowed:
        logger.error(
            "Merge | Fan-out detected: %d merged rows from %d trade rows "
            "(tolerance=%.0f%%, max_allowed=%d)",
            len(merged),
            pre_join_count,
            config.fan_out_tolerance_fraction * 100,
            max_allowed,
        )
        raise DataQualityError(
            f"Merge fan-out: {len(merged)} rows from {pre_join_count} trade rows "
            f"(tolerance={config.fan_out_tolerance_fraction:.0%})"
        )

    return merged
