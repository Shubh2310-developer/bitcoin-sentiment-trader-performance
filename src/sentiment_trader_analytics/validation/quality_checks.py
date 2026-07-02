"""Data quality check functions beyond schema enforcement.

Provides checks for temporal coverage, value-distribution anomalies,
and trader-account cardinality. Each check returns a ``QualityReport``
describing whether the check passed and why.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class QualityReport:
    """Report from a single data quality check.

    Attributes:
        check_name: Human-readable name of the quality check.
        passed: Whether the check succeeded.
        metric_value: The computed value that was tested (if applicable).
        threshold: The threshold it was compared against (if applicable).
        details: Free-text explanation or additional context.
    """

    check_name: str
    passed: bool
    metric_value: float | None = None
    threshold: float | None = None
    details: str = ""


def check_temporal_coverage(
    df: pd.DataFrame,
    config: Any,
    *,
    timestamp_col: str = "timestamp",
) -> QualityReport:
    """Verify that the DataFrame's date range meets minimum coverage.

    Computes the number of calendar days between the min and max
    timestamps and compares against ``config.validation.min_temporal_coverage_days``.

    Args:
        df: The DataFrame to check (must have a timezone-aware datetime column).
        config: Application config with a ``validation`` attribute.
        timestamp_col: Name of the datetime column (default ``"timestamp"``).

    Returns:
        A ``QualityReport`` indicating pass/fail with coverage details.
    """
    threshold_days = config.validation.min_temporal_coverage_days
    if timestamp_col not in df.columns or df[timestamp_col].isnull().all():
        return QualityReport(
            check_name="temporal_coverage",
            passed=False,
            details=f"Column '{timestamp_col}' missing or fully null",
        )

    date_min = df[timestamp_col].min()
    date_max = df[timestamp_col].max()
    coverage_days = (date_max - date_min).days

    passed = coverage_days >= threshold_days

    return QualityReport(
        check_name="temporal_coverage",
        passed=passed,
        metric_value=float(coverage_days),
        threshold=float(threshold_days),
        details=(
            f"Date range: {date_min.date()} to {date_max.date()} "
            f"({coverage_days} days, threshold: {threshold_days})"
        ),
    )


def check_value_distribution(
    df: pd.DataFrame,
    column: str,
    expected_range: tuple[float, float],
    config: Any,
) -> QualityReport:
    """Check for anomalies in a column's value distribution.

    Flags issues such as zero variance (all values identical) or all
    values falling outside the expected range.

    Args:
        df: The DataFrame to check.
        column: Column name to inspect.
        expected_range: The (min, max) inclusive range values should
            reasonably fall within.
        config: Application config (used for logging context).

    Returns:
        A ``QualityReport`` indicating pass/fail.
    """
    _ = config

    if column not in df.columns:
        return QualityReport(
            check_name="value_distribution",
            passed=False,
            details=f"Column '{column}' not found in DataFrame",
        )

    non_null = df[column].dropna()
    if len(non_null) == 0:
        return QualityReport(
            check_name="value_distribution",
            passed=False,
            details=f"Column '{column}' has no non-null values",
        )

    if non_null.nunique() <= 1:
        return QualityReport(
            check_name="value_distribution",
            passed=False,
            metric_value=float(non_null.nunique()),
            details=f"Column '{column}' has zero or near-zero variance "
            f"(unique values: {non_null.nunique()})",
        )

    lo, hi = expected_range
    out_of_range = ((non_null < lo) | (non_null > hi)).sum()
    if out_of_range > 0:
        return QualityReport(
            check_name="value_distribution",
            passed=False,
            metric_value=float(out_of_range),
            details=f"Column '{column}' has {out_of_range} values "
            f"outside expected range [{lo}, {hi}]",
        )

    return QualityReport(
        check_name="value_distribution",
        passed=True,
        metric_value=float(non_null.nunique()),
        details=f"Column '{column}' has {non_null.nunique()} unique values, "
        f"all within [{lo}, {hi}]",
    )


def check_trader_account_cardinality(
    df: pd.DataFrame,
    config: Any,
    *,
    account_col: str = "Account",
) -> QualityReport:
    """Verify the number of distinct trader accounts meets the minimum.

    Args:
        df: Trader history DataFrame.
        config: Application config with a ``validation`` attribute.
        account_col: Column name for trader account identifier
            (default ``"Account"``).

    Returns:
        A ``QualityReport`` indicating pass/fail.
    """
    threshold = config.validation.min_distinct_accounts

    if account_col not in df.columns:
        return QualityReport(
            check_name="account_cardinality",
            passed=False,
            details=f"Column '{account_col}' not found in DataFrame",
        )

    distinct_accounts = df[account_col].nunique()
    passed = distinct_accounts >= threshold

    return QualityReport(
        check_name="account_cardinality",
        passed=passed,
        metric_value=float(distinct_accounts),
        threshold=float(threshold),
        details=(f"Found {distinct_accounts} distinct accounts " f"(threshold: {threshold})"),
    )
