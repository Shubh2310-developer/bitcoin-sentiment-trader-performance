"""Reusable plotting functions for the sentiment trader analytics pipeline.

All plotting functions use the shared ``SENTIMENT_PALETTE`` for sentiment
regime coloring. Every figure includes a title, axis labels with units,
a legend (where applicable), and a source/generation-date footnote.
Figures are saved at ≥150 DPI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

SENTIMENT_PALETTE: dict[str, str] = {
    "Extreme Fear": "#c0392b",
    "Fear": "#e74c3c",
    "Neutral": "#95a5a6",
    "Greed": "#27ae60",
    "Extreme Greed": "#145a32",
}

REGIME_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
DPI = 150


def _add_footnote(ax: matplotlib.axes.Axes, _dpi: int = DPI) -> None:
    """Add a generation-date footnote to the bottom-left of a figure axis.

    Args:
        ax: The matplotlib Axes to annotate.
        dpi: Resolution DPI (unused, kept for interface consistency).
    """
    generation_date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ax.text(
        0.0,
        -0.12,
        f"Source: sentiment-trader-analytics | Generated: {generation_date}",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        ha="left",
        va="top",
    )


def plot_sentiment_value_histogram(
    df: pd.DataFrame, output_path: str | Path, dpi: int = DPI
) -> None:
    """Plot a histogram with KDE overlay of sentiment values.

    Args:
        df: DataFrame with a ``sentiment_value`` column.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df["sentiment_value"].dropna()
    sns.histplot(data, kde=True, bins=30, color="#2980b9", edgecolor="white", ax=ax)
    ax.set_title("Distribution of Bitcoin Fear & Greed Index Values", fontsize=14)
    ax.set_xlabel("Sentiment Value (0–100)")
    ax.set_ylabel("Frequency")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_sentiment_regime_frequency_barplot(
    df: pd.DataFrame, output_path: str | Path, dpi: int = DPI
) -> None:
    """Plot a bar chart of sentiment regime frequencies using the sentiment palette.

    Args:
        df: DataFrame with a ``sentiment_classification`` column.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    counts = df["sentiment_classification"].value_counts()
    palette = {r: SENTIMENT_PALETTE.get(r, "#95a5a6") for r in REGIME_ORDER if r in counts.index}

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        list(palette.keys()),
        [counts.get(r, 0) for r in palette],
        color=list(palette.values()),
        edgecolor="white",
    )
    ax.set_title("Frequency of Sentiment Regimes", fontsize=14)
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Number of Trades")
    ax.set_xticks(range(len(palette)))
    ax.set_xticklabels(list(palette.keys()), rotation=0)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_trader_pnl_distribution_histogram(
    df: pd.DataFrame, output_path: str | Path, dpi: int = DPI
) -> None:
    """Plot a histogram with KDE overlay of trader Closed PnL.

    Args:
        df: DataFrame with a ``Closed PnL`` column.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df["Closed PnL"].dropna()
    sns.histplot(data, kde=True, bins=50, color="#8e44ad", edgecolor="white", ax=ax)
    ax.set_title("Distribution of Trader Closed PnL (USD)", fontsize=14)
    ax.set_xlabel("Closed PnL (USD)")
    ax.set_ylabel("Frequency")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_pnl_by_sentiment_boxplot(
    df: pd.DataFrame, output_path: str | Path, dpi: int = DPI
) -> None:
    """Plot a box plot of Closed PnL grouped by sentiment regime.

    Args:
        df: DataFrame with ``Closed PnL`` and ``sentiment_classification`` columns.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    palette = {r: SENTIMENT_PALETTE.get(r, "#95a5a6") for r in REGIME_ORDER}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df.dropna(subset=["sentiment_classification", "Closed PnL"]),
        x="sentiment_classification",
        y="Closed PnL",
        order=REGIME_ORDER,
        hue="sentiment_classification",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title("Trader Closed PnL by Sentiment Regime", fontsize=14)
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Closed PnL (USD)")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_leverage_distribution_histogram(
    df: pd.DataFrame, output_path: str | Path, dpi: int = DPI
) -> None:
    """Plot a histogram of trade leverage values.

    Args:
        df: DataFrame with a ``Leverage`` column.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df["Leverage"].dropna()
    sns.histplot(data, bins=50, color="#2c3e50", edgecolor="white", ax=ax)
    ax.set_title("Distribution of Trade Leverage", fontsize=14)
    ax.set_xlabel("Leverage (x)")
    ax.set_ylabel("Frequency")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_missingness_heatmap(df: pd.DataFrame, output_path: str | Path, dpi: int = DPI) -> None:
    """Plot a heatmap of missing values across columns.

    Args:
        df: DataFrame to evaluate.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    missing = df.isna()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        missing.T,
        cbar=False,
        cmap=["#2c3e50", "#f1c40f"],
        ax=ax,
    )
    ax.set_title("Missing Value Heatmap", fontsize=14)
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Column")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    numeric_features: list[str] | None = None,
    method: str = "spearman",
    dpi: int = DPI,
) -> None:
    """Plot a correlation heatmap of numeric features.

    Args:
        df: DataFrame to evaluate.
        output_path: Path to save the figure.
        numeric_features: List of numeric column names. If None, uses all numeric columns.
        method: Correlation method (pearson, spearman, or kendall).
        dpi: Resolution in DPI (default 150).
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    available = [c for c in numeric_features if c in df.columns]
    corr = df[available].corr(method=method)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Matrix ({method.capitalize()})", fontsize=14)
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_sentiment_value_timeseries(
    df: pd.DataFrame, output_path: str | Path, dpi: int = DPI
) -> None:
    """Plot sentiment values over time as a line chart.

    Args:
        df: DataFrame with ``sentiment_date`` and ``sentiment_value`` columns.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df.dropna(subset=["sentiment_date", "sentiment_value"])
    ax.plot(
        pd.to_datetime(ts["sentiment_date"]),
        ts["sentiment_value"],
        color="#2980b9",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.set_title("Bitcoin Fear & Greed Index Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Value (0–100)")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_trade_count_timeseries(df: pd.DataFrame, output_path: str | Path, dpi: int = DPI) -> None:
    """Plot trade count over time as a line chart.

    Args:
        df: DataFrame with ``sentiment_date`` and ``trader_trade_count_7d`` columns.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df.dropna(subset=["sentiment_date", "trader_trade_count_7d"])
    ax.plot(
        pd.to_datetime(ts["sentiment_date"]),
        ts["trader_trade_count_7d"],
        color="#27ae60",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.set_title("Trade Activity (7-Day Rolling Count) Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Trade Count (7-day rolling)")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_pnl_timeseries(df: pd.DataFrame, output_path: str | Path, dpi: int = DPI) -> None:
    """Plot trader PnL over time as a line chart.

    Args:
        df: DataFrame with ``sentiment_date`` and ``trader_pnl_rolling_7d`` columns.
        output_path: Path to save the figure.
        dpi: Resolution in DPI (default 150).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df.dropna(subset=["sentiment_date", "trader_pnl_rolling_7d"])
    ax.plot(
        pd.to_datetime(ts["sentiment_date"]),
        ts["trader_pnl_rolling_7d"],
        color="#8e44ad",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.set_title("Trader PnL (7-Day Rolling) Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL (USD, 7-day rolling)")
    _add_footnote(ax, dpi)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
