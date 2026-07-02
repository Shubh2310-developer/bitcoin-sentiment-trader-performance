"""Data validation and schema enforcement module.

Validates ingested DataFrames against Pandera schemas defined in
``data/metadata/schemas/``, and runs domain-quality checks (temporal
coverage, value distribution, account cardinality).
"""

from sentiment_trader_analytics.validation.quality_checks import (
    QualityReport,
    check_temporal_coverage,
    check_trader_account_cardinality,
    check_value_distribution,
)
from sentiment_trader_analytics.validation.schema_checks import (
    ValidationResult,
    validate_fear_greed,
    validate_trader_history,
)

__all__ = [
    "QualityReport",
    "ValidationResult",
    "check_temporal_coverage",
    "check_trader_account_cardinality",
    "check_value_distribution",
    "validate_fear_greed",
    "validate_trader_history",
]
