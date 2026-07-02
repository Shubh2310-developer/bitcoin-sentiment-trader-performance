"""Correlation and association analysis for the sentiment trader pipeline.

Provides functions for computing pairwise correlations with effect sizes,
confidence intervals, and full correlation matrices.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class CorrelationResult(NamedTuple):
    """Container for a single correlation test result.

    Attributes:
        coefficient: The correlation coefficient.
        p_value: P-value of the test.
        sample_size: Number of observations used.
        confidence_interval_95: 95% CI as (lower, upper).
        method: Correlation method used (pearson or spearman).
    """

    coefficient: float
    p_value: float
    sample_size: int
    confidence_interval_95: tuple[float, float]
    method: str


def _fisher_z_ci(r: float, n: int) -> tuple[float, float]:
    """Compute 95% CI for Pearson correlation via Fisher z-transformation.

    Args:
        r: Pearson correlation coefficient.
        n: Sample size.

    Returns:
        (lower, upper) 95% CI bounds.
    """
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.96
    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)
    return (float(ci_lower), float(ci_upper))


def _bootstrap_spearman_ci(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% CI for Spearman correlation via bootstrap.

    Args:
        x: First variable.
        y: Second variable.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed: Random seed for reproducibility (default 42).

    Returns:
        (lower, upper) 95% CI bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_x = x[idx]
        boot_y = y[idx]
        with np.errstate(invalid="ignore"):
            if np.std(boot_x) == 0 or np.std(boot_y) == 0:
                continue
            rho, _ = spearmanr(boot_x, boot_y)
            boot_stats.append(float(rho))
    if len(boot_stats) < 100:
        return (-1.0, 1.0)
    return (float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5)))


def compute_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
    method: str = "spearman",
    alpha: float = 0.05,  # noqa: ARG001
) -> CorrelationResult:
    """Compute correlation between two series with CI and effect size.

    Uses Spearman by default (non-parametric, robust to non-normality).
    Falls back to Pearson if explicitly requested.

    Args:
        series_x: First variable.
        series_y: Second variable.
        method: Correlation method ("pearson" or "spearman", default "spearman").
        alpha: Significance level (default 0.05). Used for CI calculation
            but currently fixed at 95% level.

    Returns:
        A CorrelationResult with coefficient, p-value, n, 95% CI, and method.

    Raises:
        ValueError: If fewer than 3 non-NA paired observations remain.
    """
    combined = pd.concat([series_x, series_y], axis=1, keys=["x", "y"]).dropna()
    if len(combined) < 3:
        raise ValueError(
            f"Correlation requires at least 3 paired observations; got {len(combined)}"
        )

    x = combined["x"].values.astype(np.float64)
    y = combined["y"].values.astype(np.float64)
    n = len(combined)

    if method == "pearson":
        coef, p_val = pearsonr(x, y)
        ci = _fisher_z_ci(coef, n)
        method_name = "pearson"
    else:
        coef, p_val = spearmanr(x, y)
        ci = _bootstrap_spearman_ci(x, y)
        method_name = "spearman"

    return CorrelationResult(
        coefficient=float(coef),
        p_value=float(p_val),
        sample_size=n,
        confidence_interval_95=ci,
        method=method_name,
    )


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    output_path: str | None = None,
) -> pd.DataFrame:
    """Compute Spearman correlation matrix for specified columns.

    Args:
        df: Input DataFrame.
        columns: List of column names to include.
        output_path: Optional file path to save the matrix as CSV.

    Returns:
        A DataFrame containing the Spearman correlation matrix.
    """
    available = [c for c in columns if c in df.columns]
    missing = set(columns) - set(available)
    if missing:
        logger.warning("Columns not found in DataFrame: %s", missing)

    if not available:
        raise ValueError("No valid columns provided for correlation matrix.")

    corr_matrix = df[available].corr(method="spearman")

    if output_path:
        import pathlib

        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(output_path)
        logger.info("Correlation matrix saved to: %s", output_path)

    return corr_matrix
