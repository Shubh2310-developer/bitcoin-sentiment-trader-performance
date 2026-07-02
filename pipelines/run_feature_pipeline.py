#!/usr/bin/env python3
"""Feature engineering pipeline entry point.

Orchestrates the computation of all engineered features (sentiment,
trader, and time domains) using a functional composition pattern.
Validates the final feature store against the Pandera schema and
writes versioned output to ``data/features/``.

Usage:
    python pipelines/run_feature_pipeline.py --config configs/base.yaml
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from functools import reduce
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402

from sentiment_trader_analytics.config import (  # noqa: E402
    FeatureConfig,
    load_config,
)
from sentiment_trader_analytics.feature_engineering.sentiment_features import (  # noqa: E402
    add_sentiment_fear_greed_flags,
    add_sentiment_lag,
    add_sentiment_regime_encoding,
    add_sentiment_rolling_mean,
)
from sentiment_trader_analytics.feature_engineering.time_features import (  # noqa: E402
    add_day_of_week,
    add_month,
    add_time_of_day,
)
from sentiment_trader_analytics.feature_engineering.trader_features import (  # noqa: E402
    add_trader_avg_size,
    add_trader_leverage_avg,
    add_trader_pnl_rolling,
    add_trader_pnl_volatility,
    add_trader_trade_count,
    add_trader_win_rate,
    flag_cold_start_rows,
)
from sentiment_trader_analytics.utils.logging_utils import setup_logging  # noqa: E402

logger = setup_logging("feature_engineering", log_file="logs/pipeline.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run the feature engineering pipeline (Phase 04).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def _find_latest_processed(processed_dir: Path) -> Path:
    """Find the most recently created parquet file in the processed directory.

    Args:
        processed_dir: Directory containing processed parquet files.

    Returns:
        Path to the latest parquet file.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    parquet_files = sorted(processed_dir.glob("sentiment_trader_merged_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No processed parquet files found in {processed_dir}. "
            "Run the preprocessing pipeline first."
        )
    return parquet_files[-1]


def _write_schema_snapshot(
    output_dir: Path, run_id: str, schema_type: str = "feature_store"
) -> None:
    """Write a schema snapshot JSON to metadata directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema": schema_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
    }
    filename = f"{schema_type}_{run_id}.json"
    with open(output_dir / filename, "w") as f:
        json.dump(snapshot, f, indent=2)


def _write_data_dictionary_snapshot(
    output_dir: Path, df: pd.DataFrame, dataset_name: str, run_id: str
) -> None:
    """Write a data-dictionary snapshot with row count, null counts, dtypes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = {}
    for col in df.columns:
        columns[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
        }
    snapshot = {
        "dataset": dataset_name,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "row_count": len(df),
        "columns": columns,
    }
    filename = f"{dataset_name}_{run_id}.json"
    with open(output_dir / filename, "w") as f:
        json.dump(snapshot, f, indent=2)


def _write_lineage(output_dir: Path, dataset_name: str, run_id: str, row_count: int) -> None:
    """Write a lineage record for the feature store."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "pipeline": "feature_engineering",
        "dataset": dataset_name,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_datasets": ["sentiment_trader_merged"],
        "transformations": [
            "add_sentiment_lag",
            "add_sentiment_regime_encoding",
            "add_sentiment_rolling_mean",
            "add_sentiment_fear_greed_flags",
            "add_trader_win_rate",
            "add_trader_pnl_rolling",
            "add_trader_leverage_avg",
            "add_trader_pnl_volatility",
            "add_trader_trade_count",
            "add_trader_avg_size",
            "flag_cold_start_rows",
            "add_time_of_day",
            "add_day_of_week",
            "add_month",
        ],
        "row_count": row_count,
    }
    filename = f"{dataset_name}_{run_id}.json"
    with open(output_dir / filename, "w") as f:
        json.dump(snapshot, f, indent=2)


def main() -> None:
    """Execute the feature engineering pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    fe_config: FeatureConfig = config.feature_engineering
    start_time = time.time()

    run_id = fe_config.run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logger.info("Feature engineering pipeline started (run_id=%s)", run_id)

    try:
        # 1. Load the processed dataset
        processed_dir = Path("data/processed")
        input_path = _find_latest_processed(processed_dir)
        logger.info("Loading processed data from: %s", input_path)
        df = pd.read_parquet(input_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # 2. Compose feature functions via functools.reduce
        feature_functions = [
            add_sentiment_lag,
            add_sentiment_regime_encoding,
            add_sentiment_rolling_mean,
            add_sentiment_fear_greed_flags,
            add_trader_win_rate,
            add_trader_pnl_rolling,
            add_trader_leverage_avg,
            add_trader_pnl_volatility,
            add_trader_trade_count,
            add_trader_avg_size,
            flag_cold_start_rows,
            add_time_of_day,
            add_day_of_week,
            add_month,
        ]

        features_df = reduce(lambda d, fn: fn(d, fe_config), feature_functions, df)
        logger.info(
            "Feature composition complete: %d rows, %d columns",
            len(features_df),
            len(features_df.columns),
        )

        # 3. Validate against Pandera schema
        from data.metadata.schemas.features_schema import features_schema  # noqa: E402

        validated_df = features_schema.validate(features_df, lazy=True)
        logger.info("Pandera schema validation passed (0 violations)")

        # 4. Write feature store (parquet, versioned by run_id)
        output_dir = fe_config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"feature_store_{run_id}.parquet"
        validated_df.to_parquet(output_file)
        logger.info("Feature store written: %s (%d rows)", output_file, len(validated_df))

        # 5. Write metadata
        schema_dir = Path("data/metadata/schemas")
        data_dict_dir = Path("data/metadata/data_dictionary")
        lineage_dir = Path("data/metadata/lineage")

        _write_schema_snapshot(schema_dir, run_id)
        _write_data_dictionary_snapshot(data_dict_dir, validated_df, "feature_store", run_id)
        _write_lineage(lineage_dir, "feature_store", run_id, len(validated_df))
        logger.info("Metadata artifacts written to data/metadata/")

    except Exception:
        logger.exception("Feature engineering pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Feature engineering pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
