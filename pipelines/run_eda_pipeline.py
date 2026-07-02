#!/usr/bin/env python3
"""EDA pipeline entry point.

Orchestrates the generation of all mandatory EDA artifacts (figures and
summary tables). Loads the engineered feature store, computes descriptive
statistics, missingness reports, outlier summaries, and saves all figures
to ``outputs/figures/eda/`` and all tables to ``outputs/tables/eda/``.

Usage:
    python pipelines/run_eda_pipeline.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402

from sentiment_trader_analytics.config import (  # noqa: E402
    AppConfig,
    EDAConfig,
    load_config,
)
from sentiment_trader_analytics.eda.summary_stats import (  # noqa: E402
    compute_descriptive_stats,
    compute_missingness_report,
    compute_outlier_summary,
)
from sentiment_trader_analytics.utils.logging_utils import setup_logging  # noqa: E402
from sentiment_trader_analytics.visualization.plots import (  # noqa: E402
    plot_feature_correlation_heatmap,
    plot_leverage_distribution_histogram,
    plot_missingness_heatmap,
    plot_pnl_by_sentiment_boxplot,
    plot_pnl_timeseries,
    plot_sentiment_regime_frequency_barplot,
    plot_sentiment_value_histogram,
    plot_sentiment_value_timeseries,
    plot_trade_count_timeseries,
    plot_trader_pnl_distribution_histogram,
)

logger = setup_logging("eda", log_file="logs/pipeline.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run the EDA pipeline (Phase 05).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def _find_latest_features(features_dir: Path) -> Path:
    """Find the most recently created parquet file in the features directory.

    Args:
        features_dir: Directory containing feature store parquet files.

    Returns:
        Path to the latest parquet file.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    parquet_files = sorted(features_dir.glob("feature_store_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No feature store parquet files found in {features_dir}. "
            "Run the feature engineering pipeline first."
        )
    return parquet_files[-1]


def main() -> None:
    """Execute the EDA pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config: AppConfig = load_config(str(config_path))
    eda_config: EDAConfig = config.eda
    start_time = time.time()

    figures_dir = Path(eda_config.figures_dir)
    tables_dir = Path(eda_config.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("EDA pipeline started")
    logger.info("Figures output: %s", figures_dir)
    logger.info("Tables output: %s", tables_dir)

    try:
        # 1. Load the feature store
        features_dir = Path("data/features")
        input_path = _find_latest_features(features_dir)
        logger.info("Loading feature store from: %s", input_path)
        df = pd.read_parquet(input_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # 2. Generate figures
        logger.info("Generating EDA figures ...")

        plot_sentiment_value_histogram(df, figures_dir / "sentiment_value_histogram.png")
        logger.info("  -> sentiment_value_histogram.png")

        plot_sentiment_regime_frequency_barplot(
            df, figures_dir / "sentiment_regime_frequency_barplot.png"
        )
        logger.info("  -> sentiment_regime_frequency_barplot.png")

        plot_trader_pnl_distribution_histogram(
            df, figures_dir / "trader_pnl_distribution_histogram.png"
        )
        logger.info("  -> trader_pnl_distribution_histogram.png")

        plot_pnl_by_sentiment_boxplot(df, figures_dir / "pnl_by_sentiment_boxplot.png")
        logger.info("  -> pnl_by_sentiment_boxplot.png")

        plot_leverage_distribution_histogram(
            df, figures_dir / "leverage_distribution_histogram.png"
        )
        logger.info("  -> leverage_distribution_histogram.png")

        plot_missingness_heatmap(df, figures_dir / "missingness_heatmap.png")
        logger.info("  -> missingness_heatmap.png")

        plot_feature_correlation_heatmap(
            df,
            figures_dir / "feature_correlation_heatmap.png",
            numeric_features=eda_config.numeric_features,
            method=eda_config.correlation_method,
        )
        logger.info("  -> feature_correlation_heatmap.png")

        plot_sentiment_value_timeseries(df, figures_dir / "sentiment_value_timeseries.png")
        logger.info("  -> sentiment_value_timeseries.png")

        plot_trade_count_timeseries(df, figures_dir / "trade_count_timeseries.png")
        logger.info("  -> trade_count_timeseries.png")

        plot_pnl_timeseries(df, figures_dir / "pnl_timeseries.png")
        logger.info("  -> pnl_timeseries.png")

        # 3. Compute statistics and save tables
        logger.info("Generating EDA tables ...")

        stats_columns = [
            "sentiment_value",
            "Closed PnL",
            "Leverage",
            "trader_win_rate_7d",
            "trader_pnl_rolling_7d",
            "trader_pnl_rolling_30d",
            "trader_leverage_avg_24h",
            "trader_pnl_volatility_14d",
            "trader_trade_count_7d",
            "trader_avg_size_usd_7d",
        ]
        desc_stats = compute_descriptive_stats(df, stats_columns)
        desc_stats.to_csv(tables_dir / "descriptive_stats.csv")
        logger.info("  -> descriptive_stats.csv (%d x %d)", *desc_stats.shape)

        missing_report = compute_missingness_report(df)
        missing_report.to_csv(tables_dir / "missingness_report.csv", index=False)
        logger.info("  -> missingness_report.csv (%d rows)", len(missing_report))

        # Outlier summaries for key columns
        outlier_cols = ["sentiment_value", "Closed PnL", "Leverage"]
        outlier_records = []
        for col in outlier_cols:
            report = compute_outlier_summary(
                df,
                col,
                method=eda_config.outlier_method,
                config={"iqr_multiplier": eda_config.outlier_iqr_multiplier},
            )
            outlier_records.append(
                {
                    "column": report.column,
                    "method": report.method,
                    "n_outliers": report.n_outliers,
                    "lower_bound": round(report.lower_bound, 4),
                    "upper_bound": round(report.upper_bound, 4),
                    "total_rows": len(df),
                    "outlier_pct": round((report.n_outliers / len(df)) * 100, 2),
                }
            )
        outlier_df = pd.DataFrame(outlier_records)
        outlier_df.to_csv(tables_dir / "outlier_summary.csv", index=False)
        logger.info("  -> outlier_summary.csv (%d rows)", len(outlier_df))

        # PnL by regime statistics — observed=False ensures ALL 5 regimes appear
        all_regimes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        pnl_by_regime = (
            df.dropna(subset=["sentiment_classification", "Closed PnL"])
            .groupby("sentiment_classification", observed=False)["Closed PnL"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .round(4)
        )
        pnl_by_regime.columns = [
            "trade_count",
            "mean_pnl_usd",
            "std_pnl_usd",
            "min_pnl_usd",
            "median_pnl_usd",
            "max_pnl_usd",
        ]
        # Guarantee all 5 regimes present even if absent from data
        for regime in all_regimes:
            if regime not in pnl_by_regime.index:
                pnl_by_regime.loc[regime] = [
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ]
        pnl_by_regime = pnl_by_regime.reindex(all_regimes)
        pnl_by_regime.to_csv(tables_dir / "pnl_by_regime_stats.csv")
        logger.info("  -> pnl_by_regime_stats.csv (%d rows, all 5 regimes)", len(pnl_by_regime))

    except Exception:
        logger.exception("EDA pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("EDA pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
