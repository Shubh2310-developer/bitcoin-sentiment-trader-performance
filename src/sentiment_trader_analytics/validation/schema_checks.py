"""Schema enforcement wrappers for the validation pipeline.

Defines the ``ValidationResult`` model (Pydantic) and pure functions
that wrap ``pandera`` validation, catching ``SchemaErrors`` and
formatting violations into structured records.
"""

import pandas as pd
import pandera.errors
from pandera import DataFrameSchema
from pydantic import BaseModel, ConfigDict


class ValidationResult(BaseModel):
    """Result of a Pandera schema validation.

    Attributes:
        passed: Whether all schema checks succeeded.
        violation_count: Number of individual violations found.
        violations: Structured list of violation records, each containing
            column name, row index, offending value, and rule violated.
        validated_df: The validated DataFrame if validation passed,
            ``None`` otherwise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool
    violation_count: int = 0
    violations: list[dict[str, object]] = []
    validated_df: pd.DataFrame | None = None


def _extract_violations(
    err: pandera.errors.SchemaErrors,
) -> list[dict[str, object]]:
    """Parse a ``SchemaErrors`` exception into structured violation records.

    Args:
        err: The pandera ``SchemaErrors`` instance raised during validation.

    Returns:
        A list of dicts with keys: ``column``, ``index``, ``value``, ``rule``.
    """
    violations: list[dict[str, object]] = []
    try:
        fc = err.failure_cases
        for _, row in fc.iterrows():
            violations.append(
                {
                    "column": str(row.get("column", "")),
                    "index": row.get("index", None),
                    "value": row.get("failure_case", ""),
                    "rule": str(row.get("check", "")),
                }
            )
    except Exception:
        violations.append(
            {
                "column": "unknown",
                "index": None,
                "value": "unknown",
                "rule": "failed to parse SchemaErrors",
            }
        )
    return violations


def validate_fear_greed(
    df: pd.DataFrame,
    schema: DataFrameSchema,
) -> ValidationResult:
    """Validate a Fear & Greed DataFrame against its Pandera schema.

    Args:
        df: The ingested Fear & Greed DataFrame.
        schema: The ``fear_greed_schema`` instance.

    Returns:
        A ``ValidationResult`` with violation details if validation fails,
        or a passing result with the validated DataFrame.
    """
    try:
        validated = schema.validate(df, lazy=True)
        return ValidationResult(
            passed=True,
            violation_count=0,
            validated_df=validated,
        )
    except pandera.errors.SchemaErrors as err:
        violations = _extract_violations(err)
        return ValidationResult(
            passed=False,
            violation_count=len(violations),
            violations=violations,
            validated_df=None,
        )


def validate_trader_history(
    df: pd.DataFrame,
    schema: DataFrameSchema,
) -> ValidationResult:
    """Validate a Trader History DataFrame against its Pandera schema.

    Args:
        df: The ingested Trader History DataFrame.
        schema: The ``trader_history_schema`` instance.

    Returns:
        A ``ValidationResult`` with violation details if validation fails,
        or a passing result with the validated DataFrame.
    """
    try:
        validated = schema.validate(df, lazy=True)
        return ValidationResult(
            passed=True,
            violation_count=0,
            validated_df=validated,
        )
    except pandera.errors.SchemaErrors as err:
        violations = _extract_violations(err)
        return ValidationResult(
            passed=False,
            violation_count=len(violations),
            violations=violations,
            validated_df=None,
        )
