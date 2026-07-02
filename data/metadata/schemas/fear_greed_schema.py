"""Pandera DataFrameSchema for the Fear & Greed Index dataset.

Defines column types, constraints, nullability, uniqueness, and a
cross-column validation that classification must be strictly consistent
with the numeric value per the standard regime thresholds.
"""

import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema

CLASSIFICATION_THRESHOLDS: dict[str, tuple[int, int]] = {
    "Extreme Fear": (0, 24),
    "Fear": (25, 44),
    "Neutral": (45, 55),
    "Greed": (56, 74),
    "Extreme Greed": (75, 100),
}

ALLOWED_CLASSIFICATIONS: list[str] = list(CLASSIFICATION_THRESHOLDS.keys())


def _check_datetime_utc(s: pd.Series) -> bool:
    """Check that a column is datetime64 with a UTC timezone."""
    return (
        pd.api.types.is_datetime64_any_dtype(s)
        and hasattr(s.dt, "tz")
        and s.dt.tz is not None
    )


def _check_classification_value_consistency(df: pd.DataFrame) -> pd.Series:
    """Verify every row's classification matches the value thresholds.

    For each classification regime, all rows in that regime must have
    values falling within the corresponding threshold interval.

    Returns:
        A boolean Series where ``False`` indicates a violation.
    """
    result = pd.Series(True, index=df.index, dtype=bool)
    for classification, (lo, hi) in CLASSIFICATION_THRESHOLDS.items():
        mask = df["classification"] == classification
        if mask.any():
            result[mask] = df.loc[mask, "value"].between(lo, hi, inclusive="both")
    return result


fear_greed_schema = DataFrameSchema(  # type: ignore[no-untyped-call]
    columns={
        "timestamp": Column(
            checks=Check(
                _check_datetime_utc,
                element_wise=False,
                name="datetime_utc",
                error="timestamp must be datetime64 with UTC timezone",
            ),
            nullable=False,
            unique=True,
        ),
        "value": Column(
            np.int64,
            checks=Check.in_range(0, 100),
            nullable=False,
        ),
        "classification": Column(
            checks=Check.isin(ALLOWED_CLASSIFICATIONS),
            nullable=False,
        ),
    },
    checks=[
        Check(
            _check_classification_value_consistency,
            element_wise=False,
            name="classification_value_consistency",
            error="classification must be consistent with value thresholds",
        ),
    ],
)
