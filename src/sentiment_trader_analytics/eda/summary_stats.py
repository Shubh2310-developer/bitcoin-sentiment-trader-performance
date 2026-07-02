"""Summary statistics helpers for EDA.

Provides pure functions for computing descriptive statistics,
missingness reports, and outlier summaries from a DataFrame.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd


class OutlierReport(NamedTuple):
    """Container for outlier detection results.

    Attributes:
        column: The column that was analyzed.
        method: Detection method used (e.g., 'iqr').
        config: Configuration parameters used for detection.
        n_outliers: Number of detected outliers.
        outlier_indices: Indices of outlier rows.
        lower_bound: Lower threshold value.
        upper_bound: Upper threshold value.
    """

    column: str
    method: str
    config: dict[str, Any]
    n_outliers: int
    outlier_indices: np.ndarray
    lower_bound: float
    upper_bound: float


def compute_descriptive_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute descriptive statistics for specified columns.

    Computes count, mean, std, min, 25%, 50%, 75%, max, skew, and kurtosis
    for each requested column.

    Args:
        df: Input DataFrame.
        columns: List of column names to describe. Only numeric columns
            present in the DataFrame are included.

    Returns:
        A DataFrame where each row is a statistic and each column is one
        of the requested features.

    Raises:
        ValueError: If no numeric columns from ``columns`` are found in the
            DataFrame.
    """
    available = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not available:
        raise ValueError("None of the requested columns are numeric or present in the DataFrame.")

    stats = df[available].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats["skew"] = df[available].skew()
    stats["kurtosis"] = df[available].kurtosis()
    stats["missing"] = df[available].isna().sum()
    stats["missing_pct"] = (df[available].isna().sum() / len(df)) * 100

    stats = stats.rename(
        columns={
            "25%": "p25",
            "50%": "p50",
            "75%": "p75",
        }
    )

    return stats.T


def compute_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a missingness report, sorted by missing count descending.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with columns ``column``, ``missing_count``,
        ``missing_pct``, and ``dtype``, sorted by missing count descending.
    """
    total = len(df)
    records = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / total) * 100
        records.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "dtype": str(df[col].dtype),
            }
        )

    report = pd.DataFrame(records)
    report = report.sort_values("missing_count", ascending=False).reset_index(drop=True)
    return report


def compute_outlier_summary(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    config: dict[str, Any] | None = None,
) -> OutlierReport:
    """Compute outlier summary for a numeric column.

    Supports the IQR method by default. A config dict may contain
    ``iqr_multiplier`` (default 1.5).

    Args:
        df: Input DataFrame.
        column: Name of the numeric column to analyze.
        method: Detection method (currently only ``'iqr'`` is supported).
        config: Optional configuration dict (e.g., ``{'iqr_multiplier': 2.0}``).

    Returns:
        An OutlierReport named tuple.

    Raises:
        ValueError: If the column is not found, is not numeric, or if an
            unsupported method is requested.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")

    _config = dict(config or {})
    iqr_multiplier = _config.get("iqr_multiplier", 1.5)

    if method == "iqr":
        data = df[column].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.values
        n_outliers = len(outlier_indices)
    else:
        raise ValueError(f"Unsupported outlier detection method: '{method}'.")

    return OutlierReport(
        column=column,
        method=method,
        config=_config,
        n_outliers=n_outliers,
        outlier_indices=outlier_indices,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
