"""Date and time handling utilities for the sentiment trader pipeline.

Provides helpers for timezone conversion, date range generation, and
temporal feature construction used across pipeline stages.
"""

from datetime import UTC, datetime


def now_utc() -> datetime:
    """Return the current UTC datetime with timezone awareness.

    Returns:
        Timezone-aware datetime in UTC.
    """
    return datetime.now(UTC)


def to_utc_timestamp(dt: datetime | None) -> datetime | None:
    """Convert a datetime to UTC, raising if it is naive.

    Args:
        dt: Input datetime (may be None).

    Returns:
        UTC-normalized datetime or None if input was None.

    Raises:
        ValueError: If the input is a naive datetime without timezone info.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError(f"Naive datetime encountered: {dt}. All datetimes must be timezone-aware.")
    return dt.astimezone(UTC)
