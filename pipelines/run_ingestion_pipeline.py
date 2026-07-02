#!/usr/bin/env python3
"""Ingestion pipeline entry point.

Orchestrates the loading of Fear & Greed Index and Trader History datasets
from raw CSV files into typed, metadata-annotated DataFrames.

Usage:
    python pipelines/run_ingestion_pipeline.py --config configs/base.yaml
"""

import argparse
import sys
import time
from pathlib import Path

from sentiment_trader_analytics.config import load_config
from sentiment_trader_analytics.ingestion.fear_greed_loader import load_fear_greed_index
from sentiment_trader_analytics.ingestion.trader_history_loader import (
    load_trader_history,
)
from sentiment_trader_analytics.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("ingestion")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run the data ingestion pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the ingestion pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    start_time = time.time()
    logger.info("Ingestion pipeline started (config: %s)", config_path)

    try:
        fg_df = load_fear_greed_index(config.ingestion)
        logger.info(
            "Fear & Greed loaded: %d rows, %d columns",
            len(fg_df),
            len(fg_df.columns),
        )

        th_df = load_trader_history(config.ingestion)
        logger.info(
            "Trader history loaded: %d rows, %d columns",
            len(th_df),
            len(th_df.columns),
        )

    except Exception:
        logger.exception("Ingestion pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Ingestion pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
