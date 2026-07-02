#!/usr/bin/env python3
"""Preprocessing pipeline entry point.

Orchestrates cleaning, deduplication, and merging of Fear & Greed and
Trader History datasets. Writes interim cleaned data to ``data/interim/``
and the final joined dataset to ``data/processed/``.

Usage:
    python pipelines/run_preprocessing_pipeline.py --config configs/base.yaml
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is on sys.path for ``data.metadata.schemas`` access.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402

from sentiment_trader_analytics.config import load_config  # noqa: E402
from sentiment_trader_analytics.ingestion.fear_greed_loader import (  # noqa: E402
    load_fear_greed_index,
)
from sentiment_trader_analytics.ingestion.trader_history_loader import (  # noqa: E402
    load_trader_history,
)
from sentiment_trader_analytics.preprocessing.cleaning import (  # noqa: E402
    clean_fear_greed,
    clean_trader_history,
)
from sentiment_trader_analytics.preprocessing.merging import (  # noqa: E402
    extract_trade_date,
    merge_sentiment_and_trades,
)
from sentiment_trader_analytics.utils.logging_utils import setup_logging  # noqa: E402

logger = setup_logging("preprocessing", log_file="logs/pipeline.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline (Phase 03).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def _write_data_dictionary(df: pd.DataFrame, output_path: Path, dataset_name: str) -> None:
    """Write a data-dictionary JSON snapshot."""
    output_path.mkdir(parents=True, exist_ok=True)
    columns = {}
    for col in df.columns:
        columns[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
        }
    snapshot = {
        "dataset": dataset_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "row_count": len(df),
        "columns": columns,
    }
    filename = f"{dataset_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path / filename, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    logger.info("Data dictionary written: %s", output_path / filename)


def _write_lineage(
    output_path: Path,
    dataset_name: str,
    source_datasets: list[str],
    transformations: list[str],
    row_count: int,
) -> None:
    """Write a lineage snapshot JSON."""
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "pipeline": "preprocessing",
        "dataset": dataset_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_datasets": source_datasets,
        "transformations": transformations,
        "row_count": row_count,
    }
    filename = f"{dataset_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path / filename, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    logger.info("Lineage written: %s", output_path / filename)


def main() -> None:
    """Execute the preprocessing pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    pc = config.preprocessing
    start_time = time.time()
    logger.info("Preprocessing pipeline started (config: %s)", config_path)

    try:
        # 1. Load validated DataFrames from raw (ingestion)
        fg_raw = load_fear_greed_index(config.ingestion)
        th_raw = load_trader_history(config.ingestion)
        logger.info("Loaded: FG=%d rows, TH=%d rows", len(fg_raw), len(th_raw))

        # 2. Clean both datasets
        fg_clean = clean_fear_greed(fg_raw, pc)
        th_clean = clean_trader_history(th_raw, pc)
        logger.info("Cleaned: FG=%d rows, TH=%d rows", len(fg_clean), len(th_clean))

        # 3. Write interim outputs
        interim_dir = Path("data/interim")
        interim_dir.mkdir(parents=True, exist_ok=True)
        if pc.interim_output_format == "parquet":
            fg_clean.to_parquet(interim_dir / "fear_greed_cleaned.parquet")
            th_clean.to_parquet(interim_dir / "trader_history_cleaned.parquet")
        else:
            fg_clean.to_csv(interim_dir / "fear_greed_cleaned.csv", index=False)
            th_clean.to_csv(interim_dir / "trader_history_cleaned.csv", index=False)
        logger.info("Interim outputs written to %s", interim_dir)

        # 4. Extract trade date and merge
        th_with_date = extract_trade_date(th_clean)
        merged = merge_sentiment_and_trades(th_with_date, fg_clean, pc)
        logger.info("Merged dataset: %d rows, %d columns", len(merged), len(merged.columns))

        # 5. Write final processed dataset with versioned filename
        processed_dir = pc.processed_output_path
        processed_dir.mkdir(parents=True, exist_ok=True)
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_file = processed_dir / f"sentiment_trader_merged_{version}.parquet"
        merged.to_parquet(output_file)
        logger.info("Processed output written: %s", output_file)

        # 6. Write metadata
        data_dict_dir = Path("data/metadata/data_dictionary")
        lineage_dir = Path("data/metadata/lineage")
        _write_data_dictionary(merged, data_dict_dir, "sentiment_trader_merged")
        _write_lineage(
            lineage_dir,
            "sentiment_trader_merged",
            source_datasets=["fear_greed", "trader_history"],
            transformations=[
                "clean_fear_greed",
                "clean_trader_history",
                "extract_trade_date",
                "merge_sentiment_and_trades",
            ],
            row_count=len(merged),
        )

    except Exception:
        logger.exception("Preprocessing pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Preprocessing pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
