"""Pandera DataFrameSchema for the Hyperliquid Trader History dataset.

Defines column types, constraints, nullability, and uniqueness for
all trader trade-record columns. ``Closed PnL``, ``Fee``, and
``Leverage`` are nullable (null represents open positions or missing
fields); all other columns are required and non-nullable.
"""

import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema

ALLOWED_SIDES: list[str] = ["Long", "Short"]
ALLOWED_DIRECTIONS: list[str] = ["Open", "Close"]


def _check_datetime_utc(s: pd.Series) -> bool:
    """Check that a column is datetime64 with a UTC timezone."""
    return (
        pd.api.types.is_datetime64_any_dtype(s)
        and hasattr(s.dt, "tz")
        and s.dt.tz is not None
    )


trader_history_schema = DataFrameSchema(  # type: ignore[no-untyped-call]
    columns={
        "Trade ID": Column(
            nullable=False,
            unique=True,
        ),
        "Account": Column(
            checks=Check(
                lambda s: s.astype(str).str.len() > 0,
                element_wise=False,
                error="Account must be non-empty",
            ),
            nullable=False,
        ),
        "Timestamp": Column(
            checks=Check(
                _check_datetime_utc,
                element_wise=False,
                name="datetime_utc",
                error="Timestamp must be datetime64 with UTC timezone",
            ),
            nullable=False,
        ),
        "Side": Column(
            checks=Check.isin(ALLOWED_SIDES),
            nullable=False,
        ),
        "Direction": Column(
            checks=Check.isin(ALLOWED_DIRECTIONS),
            nullable=False,
        ),
        "Size USD": Column(
            np.float64,
            checks=Check.ge(0.0),
            nullable=False,
        ),
        "Execution Price": Column(
            np.float64,
            checks=Check.gt(0.0),
            nullable=False,
        ),
        "Closed PnL": Column(
            np.float64,
            nullable=True,
        ),
        "Fee": Column(
            np.float64,
            checks=Check.ge(0.0),
            nullable=True,
        ),
        "Leverage": Column(
            np.float64,
            checks=Check.gt(0.0),
            nullable=True,
            required=False,
        ),
    },
)
