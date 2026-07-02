"""Preprocessing and cleaning transformations for the analytics pipeline.

Provides pure functions for cleaning, deduplication, null handling,
and safe dataset merging. See Phase 03 of the phased execution plan.
"""


class DataQualityError(Exception):
    """Raised when a data quality issue is detected during preprocessing.

    This halts the pipeline pending investigation.
    """

    pass
