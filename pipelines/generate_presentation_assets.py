#!/usr/bin/env python3
# ruff: noqa: E402
"""Presentation asset regeneration at 300 DPI.

Loads the engineered feature store and regenerates the mandatory report
figures at 300 DPI for inclusion in stakeholder presentations. Every figure
uses the shared SENTIMENT_PALETTE and includes title, axis labels, legend,
and source/generation-date footnote.

Usage:
    python pipelines/generate_presentation_assets.py --config configs/base.yaml

Pipeline stage: §11.8 — Reporting (sub-task: presentation asset generation)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sentiment_trader_analytics.config import AppConfig, load_config
from sentiment_trader_analytics.utils.logging_utils import setup_logging
from sentiment_trader_analytics.visualization.plots import (
    _add_footnote,
    plot_feature_correlation_heatmap,
    plot_pnl_by_sentiment_boxplot,
    plot_sentiment_regime_frequency_barplot,
    plot_sentiment_value_timeseries,
)

matplotlib.use("Agg")

logger = setup_logging("presentation_assets", log_file="logs/pipeline.log")

PRESENTATION_DPI = 300
PRESENTATION_OUTPUT_DIR = Path("outputs/presentation_assets")

REQUIRED_FIGURES: list[dict[str, Any]] = [
    {
        "name": "pnl_by_sentiment_boxplot.png",
        "generator": "boxplot",
        "description": "Box plot of Closed PnL grouped by sentiment regime",
    },
    {
        "name": "sentiment_value_timeseries.png",
        "generator": "timeseries",
        "description": "Line chart of sentiment values over the observation period",
    },
    {
        "name": "feature_correlation_heatmap.png",
        "generator": "correlation_heatmap",
        "description": "Pairwise Spearman correlation of numeric features",
    },
    {
        "name": "sentiment_regime_frequency_barplot.png",
        "generator": "regime_barplot",
        "description": "Bar chart of trade count per sentiment regime",
    },
    {
        "name": "ml_classification_feature_importance.png",
        "generator": "ml_importance",
        "source": "outputs/figures/ml/classification_feature_importance.png",
        "description": "Permutation feature importance for Random Forest classifier",
    },
    {
        "name": "ml_regression_feature_importance.png",
        "generator": "ml_importance",
        "source": "outputs/figures/ml/regression_feature_importance.png",
        "description": "Permutation feature importance for Random Forest regressor",
    },
]


def _find_latest_features(features_dir: Path) -> Path:
    """Find the most recent feature store parquet file."""
    parquet_files = sorted(features_dir.glob("feature_store_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No feature store parquet files found in {features_dir}. "
            "Run the feature engineering pipeline first."
        )
    return parquet_files[-1]


def _regenerate_ml_feature_importance(source_png: Path, output_path: Path, title: str) -> None:
    """Re-export an ML feature importance chart at 300 DPI.

    Reads the existing 150 DPI PNG, re-displays it on a new figure at the
    target DPI, and adds a footnote.
    """
    img = plt.imread(source_png)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)
    _add_footnote(ax, _dpi=300)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  -> %s (300 DPI)", output_path.name)


def generate_presentation_assets(config: AppConfig) -> list[Path]:
    """Regenerate all required report figures at 300 DPI.

    Args:
        config: Application configuration object.

    Returns:
        List of paths to generated presentation assets.
    """
    figures_dir = PRESENTATION_OUTPUT_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    logger.info("Loading feature store data for figure regeneration...")
    features_dir = Path("data/features")
    input_path = _find_latest_features(features_dir)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    logger.info("Regenerating presentation assets at %d DPI...", PRESENTATION_DPI)

    numeric_features = None
    correlation_method = "spearman"
    if hasattr(config, "eda"):
        cfg = config.eda
        numeric_features = cfg.numeric_features if hasattr(cfg, "numeric_features") else None
        correlation_method = (
            cfg.correlation_method if hasattr(cfg, "correlation_method") else "spearman"
        )

    for fig_spec in REQUIRED_FIGURES:
        name = fig_spec["name"]
        output_path = figures_dir / name
        logger.info("Generating %s...", name)

        try:
            if fig_spec["generator"] == "boxplot":
                plot_pnl_by_sentiment_boxplot(df, output_path, dpi=PRESENTATION_DPI)

            elif fig_spec["generator"] == "timeseries":
                plot_sentiment_value_timeseries(df, output_path, dpi=PRESENTATION_DPI)

            elif fig_spec["generator"] == "correlation_heatmap":
                plot_feature_correlation_heatmap(
                    df,
                    output_path,
                    numeric_features=numeric_features,
                    method=correlation_method,
                    dpi=PRESENTATION_DPI,
                )

            elif fig_spec["generator"] == "regime_barplot":
                plot_sentiment_regime_frequency_barplot(df, output_path, dpi=PRESENTATION_DPI)

            elif fig_spec["generator"] == "ml_importance":
                source_png = Path(fig_spec["source"])
                title = fig_spec["description"].split("Permutation ")[-1].capitalize()
                if source_png.exists():
                    _regenerate_ml_feature_importance(source_png, output_path, title)
                else:
                    logger.warning(
                        "ML feature importance source not found: %s — skipping", source_png
                    )
                    continue

            else:
                logger.warning("Unknown generator: %s — skipping", fig_spec["generator"])
                continue

            if output_path.exists():
                generated.append(output_path)

        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info(
        "Presentation asset generation complete: %d / %d figures generated",
        len(generated),
        len(REQUIRED_FIGURES),
    )
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate report figures at 300 DPI for presentation assets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config: AppConfig = load_config(str(config_path))
    start_time = time.time()
    try:
        generated = generate_presentation_assets(config)
        if not generated:
            logger.warning("No presentation assets were generated.")
    except Exception:
        logger.exception("Presentation asset generation failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Presentation asset generation completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
