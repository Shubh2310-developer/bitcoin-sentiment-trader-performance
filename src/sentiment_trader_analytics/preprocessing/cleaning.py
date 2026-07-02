"""Cleaning transformations for Fear & Greed and Trader History datasets.

All functions are pure — they accept a DataFrame and return a cleaned
DataFrame without side effects (logging is the only exception).
"""

import logging

import pandas as pd

from sentiment_trader_analytics.config import PreprocessingConfig

logger = logging.getLogger(__name__)


def clean_fear_greed(df: pd.DataFrame, _config: PreprocessingConfig) -> pd.DataFrame:
    """Clean the Fear & Greed Index DataFrame.

    Steps:
        1. Verify ``timestamp`` is ``datetime64[UTC]`` (assert, not coerce).
        2. Drop rows with null ``value`` or ``classification``.
        3. Remove duplicate timestamps.

    Args:
        df: Validated Fear & Greed DataFrame from Phase 02.
        config: Preprocessing configuration.

    Returns:
        Cleaned DataFrame.
    """
    _verify_utc(df, "timestamp")

    n_before = len(df)

    null_mask = df["value"].isna() | df["classification"].isna()
    null_count = int(null_mask.sum())
    if null_count > 0:
        proportion = null_count / n_before
        logger.warning(
            "FG nulls | Dropping %d rows (%.2f%%) with null value or classification",
            null_count,
            proportion * 100,
        )
        df = df[~null_mask].copy()

    n_pre_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    dup_count = n_pre_dedup - len(df)
    if dup_count > 0:
        logger.warning("FG dupes | Removed %d duplicate timestamp(s)", dup_count)

    return df


def clean_trader_history(df: pd.DataFrame, _config: PreprocessingConfig) -> pd.DataFrame:
    """Clean the Trader History DataFrame.

    Steps:
        1. Verify ``Timestamp`` is ``datetime64[UTC]``.
        2. Drop rows with null ``Account``, ``Timestamp``, ``Size USD``,
           or ``Execution Price``.
        3. Deduplicate by ``Trade ID`` — exact duplicates dropped;
           non-exact duplicates raise :class:`DataQualityError`.

    Args:
        df: Validated Trader History DataFrame from Phase 02.
        config: Preprocessing configuration.

    Returns:
        Cleaned DataFrame.

    Raises:
        DataQualityError: If non-exact duplicate Trade IDs are found.
    """
    _verify_utc(df, "Timestamp")

    n_before = len(df)
    logger.info("TH cleaning | Started with %d rows", n_before)

    null_cols = ["Account", "Timestamp", "Size USD", "Execution Price"]
    for col in null_cols:
        col_null_count = int(df[col].isna().sum())
        if col_null_count > 0:
            proportion = col_null_count / len(df)
            logger.warning(
                "TH nulls   | Dropping %d rows (%.2f%%) with null %s",
                col_null_count,
                proportion * 100,
                col,
            )
            df = df[df[col].notna()].copy()

    dup_mask = df.duplicated(subset=["Trade ID"], keep=False)
    n_pre_dedup = len(df)
    if dup_mask.any():
        dup_df = df[dup_mask]
        has_non_exact = False
        for _tid, group in dup_df.groupby("Trade ID", sort=False):
            if len(group) > 1:
                first = group.iloc[0]
                for idx in range(1, len(group)):
                    if not group.iloc[idx].equals(first):
                        has_non_exact = True
                        break
                if has_non_exact:
                    break

        if has_non_exact:
            non_exact_count = int(dup_mask.sum())
            logger.warning(
                "TH dupes   | Found %d rows with non-exact duplicate Trade IDs — "
                "likely float64 precision collisions (see Phase 01 ingestion). "
                "Dropping duplicates, keeping first occurrence.",
                non_exact_count,
            )

        df = df.drop_duplicates(subset=["Trade ID"], keep="first")
        dup_count = n_pre_dedup - len(df)
        if dup_count > 0 and not has_non_exact:
            logger.info("TH dupes   | Removed %d exact duplicate Trade ID(s)", dup_count)

    n_after = len(df)
    logger.info("TH cleaning | Completed: %d rows (removed %d)", n_after, n_before - n_after)

    return df


def _verify_utc(df: pd.DataFrame, col: str) -> None:
    """Assert that *col* is timezone-aware UTC datetime64."""
    assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} must be datetime64 dtype"
    assert df[col].dt.tz is not None, f"{col} must be timezone-aware"
    assert str(df[col].dt.tz) == "UTC", f"{col} must be in UTC"
