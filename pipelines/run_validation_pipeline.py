#!/usr/bin/env python3
"""Validation pipeline entry point.

Orchestrates schema and quality validation on ingested DataFrames from
Phase 01. Instantiates Pandera schemas, calls validation functions,
and halts (exit non-zero) on any schema violation.

Usage:
    python pipelines/run_validation_pipeline.py --config configs/base.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so that ``data.metadata.schemas`` is
# importable.  This is a standard pattern for pipeline entry-point scripts.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402

from data.metadata.pipelines.fear_and_greed_pipeline import (  # noqa: E402
    process_fear_greed,
)
from data.metadata.pipelines.trader_history_pipeline import (  # noqa: E402
    process_trader_history,
)
from data.metadata.schemas.fear_greed_schema import fear_greed_schema  # noqa: E402
from data.metadata.schemas.trader_history_schema import (  # noqa: E402
    trader_history_schema,
)
from sentiment_trader_analytics.config import load_config  # noqa: E402
from sentiment_trader_analytics.ingestion.fear_greed_loader import (  # noqa: E402
    load_fear_greed_index,
)
from sentiment_trader_analytics.ingestion.trader_history_loader import (  # noqa: E402
    load_trader_history,
)
from sentiment_trader_analytics.utils.logging_utils import setup_logging  # noqa: E402
from sentiment_trader_analytics.validation.schema_checks import (  # noqa: E402
    validate_fear_greed,
    validate_trader_history,
)

logger = setup_logging("validation", log_file="logs/validation.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run the data validation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def _run_schema_validation(
    df: pd.DataFrame,
    schema: Any,
    dataset_name: str,
    validate_fn: Any,
) -> None:
    """Run a single schema validation and handle failure.

    Args:
        df: DataFrame to validate.
        schema: Pandera ``DataFrameSchema`` instance.
        dataset_name: Human-readable dataset label for logging.
        validate_fn: One of ``validate_fear_greed`` / ``validate_trader_history``.

    Raises:
        SystemExit: If validation fails.
    """
    logger.info("Validating %s schema ...", dataset_name)
    result = validate_fn(df, schema)

    if result.passed:
        logger.info(
            "%s validation PASSED (%d rows)",
            dataset_name,
            len(result.validated_df) if result.validated_df is not None else 0,
        )
    else:
        logger.error(
            "%s validation FAILED — %d violation(s)",
            dataset_name,
            result.violation_count,
        )
        for v in result.violations:
            logger.error(
                "  VIOLATION | col=%(column)s | row=%(index)s | " "value=%(value)s | rule=%(rule)s",
                v,
            )
        sys.exit(1)


def main() -> None:
    """Execute the validation pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    start_time = time.time()
    logger.info("Validation pipeline started (config: %s)", config_path)

    try:
        fg_df = load_fear_greed_index(config.ingestion)
        th_df = load_trader_history(config.ingestion)

        # Process datasets through pipelines
        fg_df = process_fear_greed(fg_df)
        th_df = process_trader_history(th_df)
    except Exception:
        logger.exception("Ingestion step failed — cannot validate")
        sys.exit(1)

    _run_schema_validation(fg_df, fear_greed_schema, "Fear & Greed", validate_fear_greed)
    _run_schema_validation(th_df, trader_history_schema, "Trader History", validate_trader_history)

    elapsed = time.time() - start_time
    logger.info("Validation pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
