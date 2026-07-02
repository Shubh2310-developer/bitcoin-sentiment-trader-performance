"""Extended statistical analyses for Phase 06 (ET-01 through ET-07).

Implements seven advanced analyses that complement the core hypothesis tests:
    ET-01: Regime transition analysis — do traders perform differently
           in the period immediately after a sentiment regime change?
    ET-02: Regime duration effects — does the *length* of a sentiment
           run modulate trading outcomes?
    ET-03: Trader segmentation — k-means clustering of trader behaviour
           profiles and inter-cluster performance comparison.
    ET-04: Volatility regime interaction — does sentiment predictive
           power differ when price volatility is high vs. low?
    ET-05: Extended lag analysis — cross-correlation of sentiment with
           PnL at lags 1–14 days.
    ET-06: Power analysis — post-hoc minimum detectable effect sizes
           for all core hypothesis tests.
    ET-07: Survivorship & win-streak analysis — do consecutive winning
           trades in one sentiment regime predict future profitability?

Every function returns a :class:`StatTestResult` or a list thereof so
results integrate seamlessly with the existing correction and reporting
pipeline.  Non-significant results are valid findings and are reported
without suppression (per Section 6 of the Engineering Standards).

Correlation ≠ causation: all interpretations are associative only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sentiment_trader_analytics.statistics.hypothesis_tests import (
    StatTestResult,
    compare_multiple_groups,
    compare_two_groups,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ET-01: Regime Transition Analysis
# ---------------------------------------------------------------------------


def analyze_regime_transitions(
    df: pd.DataFrame,
    window_rows: int = 10,
    alpha: float = 0.05,
    min_sample_size: int = 30,
) -> StatTestResult:
    """ET-01: Compare trader PnL in the window immediately after a regime change.

    Identifies rows where the sentiment classification changes from one day
    to the next and compares ``Closed PnL`` in the ``window_rows`` rows
    following a transition against all other rows.

    Args:
        df: Feature-store DataFrame containing ``classification``,
            ``Timestamp``, and ``Closed PnL`` columns.
        window_rows: Number of rows after a transition to mark as
            "post-transition" (row-count based, not time-based).
        alpha: Significance level.
        min_sample_size: Minimum group size below which the result is
            flagged as underpowered.

    Returns:
        StatTestResult comparing post-transition vs. steady-state PnL.
    """
    if "classification" not in df.columns or "Closed PnL" not in df.columns:
        raise ValueError("DataFrame must contain 'classification' and 'Closed PnL' columns.")

    working = df.dropna(subset=["classification", "Closed PnL"]).copy()
    working = working.sort_values("Timestamp").reset_index(drop=True)

    regime_changed = working["classification"] != working["classification"].shift(1)
    transition_indices: set[int] = set()
    for idx in working.index[regime_changed]:
        for offset in range(window_rows):
            candidate = idx + offset
            if candidate in working.index:
                transition_indices.add(candidate)

    post_transition = working.loc[working.index.isin(transition_indices), "Closed PnL"]
    steady_state = working.loc[~working.index.isin(transition_indices), "Closed PnL"]

    logger.info(
        "ET-01: post-transition n=%d, steady-state n=%d", len(post_transition), len(steady_state)
    )

    if len(post_transition) < 3 or len(steady_state) < 3:  # noqa: PLR2004
        logger.warning(
            "ET-01: insufficient data in one group (post=%d, steady=%d); "
            "returning trivial result.",
            len(post_transition),
            len(steady_state),
        )
        return StatTestResult(
            test_name="ET-01_regime_transition",
            metric="Closed PnL",
            groups=["post_transition", "steady_state"],
            statistic=float("nan"),
            p_value=1.0,
            effect_size=0.0,
            effect_size_measure="cohens_d",
            confidence_interval_95=(float("nan"), float("nan")),
            sample_sizes={
                "post_transition": len(post_transition),
                "steady_state": len(steady_state),
            },
            normality_rejected=False,
            underpowered=True,
        )

    result = compare_two_groups(
        post_transition,
        steady_state,
        alpha=alpha,
        min_sample_size=min_sample_size,
    )
    result.test_name = "ET-01_regime_transition"
    result.metric = "Closed PnL"
    result.groups = ["post_transition", "steady_state"]
    return result


# ---------------------------------------------------------------------------
# ET-02: Regime Duration Effects
# ---------------------------------------------------------------------------


def analyze_regime_duration(
    df: pd.DataFrame,
    duration_buckets: list[int] | None = None,
    alpha: float = 0.05,
    min_sample_size: int = 30,
) -> StatTestResult:
    """ET-02: Does the length of a sentiment run modulate Closed PnL?

    Computes the consecutive-row streak length for each sentiment regime
    and buckets streaks into Short / Medium / Long / Extended groups,
    then runs a multi-group test on ``Closed PnL``.

    Args:
        df: Feature-store DataFrame with ``classification`` and
            ``Closed PnL`` columns.
        duration_buckets: Upper bounds (inclusive) for streak length
            buckets.  Defaults to [1, 3, 7, 14].  Rows with streak
            length > max bucket go into "Extended".
        alpha: Significance level.
        min_sample_size: Minimum group size for underpowered flag.

    Returns:
        StatTestResult comparing PnL across duration-bucket groups.
    """
    if duration_buckets is None:
        duration_buckets = [1, 3, 7, 14]

    working = df.dropna(subset=["classification", "Closed PnL"]).copy()
    working = working.sort_values("Timestamp").reset_index(drop=True)

    # Assign streak lengths (row-count based)
    streak = []
    current_streak = 1
    for i in range(len(working)):
        if i == 0:
            streak.append(1)
        elif working["classification"].iloc[i] == working["classification"].iloc[i - 1]:
            current_streak += 1
            streak.append(current_streak)
        else:
            current_streak = 1
            streak.append(1)
    working["_streak"] = streak

    bucket_labels = [f"≤{b}" for b in duration_buckets] + ["Extended"]
    bucket_bounds = duration_buckets + [int(1e9)]

    def _bucket(s: int) -> str:
        """Map a streak length to a bucket label."""
        for bound, label in zip(bucket_bounds, bucket_labels, strict=False):
            if s <= bound:
                return label
        return "Extended"

    working["_bucket"] = working["_streak"].apply(_bucket)

    groups: dict[str, pd.Series] = {}
    for label in bucket_labels:
        grp = working.loc[working["_bucket"] == label, "Closed PnL"]
        if len(grp) >= 3:  # noqa: PLR2004
            groups[label] = grp

    if len(groups) < 2:  # noqa: PLR2004
        logger.warning("ET-02: fewer than 2 duration groups; returning trivial result.")
        return StatTestResult(
            test_name="ET-02_regime_duration",
            metric="Closed PnL",
            groups=list(groups.keys()),
            statistic=float("nan"),
            p_value=1.0,
            effect_size=0.0,
            effect_size_measure="epsilon_squared",
            confidence_interval_95=(float("nan"), float("nan")),
            sample_sizes={k: len(v) for k, v in groups.items()},
            normality_rejected=True,
            underpowered=True,
        )

    logger.info("ET-02: bucket sizes = %s", {k: len(v) for k, v in groups.items()})

    result = compare_multiple_groups(groups, alpha=alpha, min_sample_size=min_sample_size)
    result.test_name = "ET-02_regime_duration"
    result.metric = "Closed PnL"
    result.groups = list(groups.keys())
    return result


# ---------------------------------------------------------------------------
# ET-03: Trader Segmentation
# ---------------------------------------------------------------------------


def analyze_trader_segments(
    df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 6,
    alpha: float = 0.05,
    min_sample_size: int = 30,
    random_state: int = 42,
) -> list[StatTestResult]:
    """ET-03: K-means trader segmentation with optimal-k selection by silhouette.

    Segments traders by their behavioural profile (win rate, leverage,
    average position size, PnL volatility) and compares ``Closed PnL``
    across segments.  Optimal k is selected by maximising the silhouette
    coefficient (config-driven range ``k_min``..``k_max``).

    Args:
        df: Feature-store DataFrame with trader feature columns.
        k_min: Minimum number of clusters to evaluate.
        k_max: Maximum number of clusters to evaluate.
        alpha: Significance level.
        min_sample_size: Minimum cluster size for underpowered flag.
        random_state: Random seed for k-means.

    Returns:
        List with one StatTestResult (multi-group PnL comparison across
        clusters) plus one metadata result carrying silhouette scores.
    """
    from sklearn.metrics import silhouette_score

    segment_features = [
        "trader_win_rate_7d",
        "trader_leverage_avg_24h",
        "trader_avg_size_usd_7d",
        "trader_pnl_volatility_14d",
    ]

    available = [c for c in segment_features if c in df.columns]
    if len(available) < 2:  # noqa: PLR2004
        raise ValueError(f"ET-03 requires at least 2 of {segment_features}; found {available}.")

    working = df.dropna(subset=available + ["Closed PnL"]).copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(working[available])

    best_k = k_min
    best_score = -1.0
    sil_records: list[dict[str, Any]] = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(x_scaled)
        if len(np.unique(labels)) < 2:  # noqa: PLR2004
            continue
        score = float(silhouette_score(x_scaled, labels))
        sil_records.append({"k": k, "silhouette": score})
        logger.info("ET-03: k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    working["_segment"] = km_final.fit_predict(x_scaled)
    working["_segment"] = working["_segment"].apply(lambda x: f"Cluster_{x}")

    groups: dict[str, pd.Series] = {}
    for seg in sorted(working["_segment"].unique()):
        grp = working.loc[working["_segment"] == seg, "Closed PnL"]
        if len(grp) >= 3:  # noqa: PLR2004
            groups[seg] = grp

    multi_result = compare_multiple_groups(groups, alpha=alpha, min_sample_size=min_sample_size)
    multi_result.test_name = "ET-03_trader_segmentation"
    multi_result.metric = "Closed PnL"
    multi_result.groups = list(groups.keys())

    # Encode silhouette summary as a metadata result
    meta_result = StatTestResult(
        test_name="ET-03_silhouette_summary",
        metric="silhouette_score",
        groups=[f"k={r['k']}" for r in sil_records],
        statistic=best_score,
        p_value=float("nan"),
        effect_size=best_score,
        effect_size_measure="silhouette",
        confidence_interval_95=(float("nan"), float("nan")),
        sample_sizes={f"k={r['k']}": int(r["k"]) for r in sil_records},
        normality_rejected=False,
        underpowered=False,
    )
    return [multi_result, meta_result]


# ---------------------------------------------------------------------------
# ET-04: Volatility Regime Interaction
# ---------------------------------------------------------------------------


def analyze_volatility_interaction(
    df: pd.DataFrame,
    min_sample_size: int = 30,
) -> list[StatTestResult]:
    """ET-04: Does sentiment-PnL relationship differ by PnL volatility regime?

    Splits the dataset into high-volatility and low-volatility regimes
    based on the median of ``trader_pnl_volatility_14d`` and computes
    Spearman correlation between ``sentiment_value`` and ``Closed PnL``
    in each sub-regime.  Returns two StatTestResults (one per regime)
    plus a Fisher z-test comparing the two correlations.

    Args:
        df: Feature-store DataFrame.
        alpha: Significance level.
        min_sample_size: Minimum group size for underpowered flag.

    Returns:
        List of StatTestResults: [high-vol corr, low-vol corr, z-test].
    """
    required = ["trader_pnl_volatility_14d", "sentiment_value", "Closed PnL"]
    working = df.dropna(subset=required).copy()

    vol_median = working["trader_pnl_volatility_14d"].median()
    high_vol = working[working["trader_pnl_volatility_14d"] > vol_median]
    low_vol = working[working["trader_pnl_volatility_14d"] <= vol_median]

    results: list[StatTestResult] = []
    for label, subset in [("high_volatility", high_vol), ("low_volatility", low_vol)]:
        n = len(subset)
        if n < 3:  # noqa: PLR2004
            logger.warning("ET-04: %s has only %d rows, skipping.", label, n)
            continue

        rho, pval = stats.spearmanr(subset["sentiment_value"], subset["Closed PnL"])
        rho = float(rho)
        pval = float(pval)

        # 95% CI via Fisher z
        z = np.arctanh(np.clip(rho, -0.999, 0.999))
        se = 1.0 / np.sqrt(n - 3) if n > 3 else np.inf
        ci_low = float(np.tanh(z - 1.96 * se))
        ci_high = float(np.tanh(z + 1.96 * se))

        results.append(
            StatTestResult(
                test_name=f"ET-04_{label}_spearman",
                metric="sentiment_value vs Closed PnL",
                groups=[label],
                statistic=rho,
                p_value=pval,
                effect_size=abs(rho),
                effect_size_measure="spearman_rho",
                confidence_interval_95=(ci_low, ci_high),
                sample_sizes={label: n},
                normality_rejected=True,
                underpowered=n < min_sample_size,
            )
        )

    # Fisher z-test comparing the two correlations
    if len(results) == 2:  # noqa: PLR2004
        r1 = results[0].statistic
        r2 = results[1].statistic
        n1 = list(results[0].sample_sizes.values())[0]
        n2 = list(results[1].sample_sizes.values())[0]

        z1 = np.arctanh(np.clip(r1, -0.999, 0.999))
        z2 = np.arctanh(np.clip(r2, -0.999, 0.999))
        se_diff = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3)) if (n1 > 3 and n2 > 3) else np.inf
        z_diff = float((z1 - z2) / se_diff) if se_diff > 0 else float("nan")
        p_diff = float(2 * (1 - stats.norm.cdf(abs(z_diff))))

        results.append(
            StatTestResult(
                test_name="ET-04_volatility_interaction_fisher_z",
                metric="correlation_difference",
                groups=["high_volatility", "low_volatility"],
                statistic=z_diff,
                p_value=p_diff,
                effect_size=abs(z_diff),
                effect_size_measure="fisher_z_diff",
                confidence_interval_95=(float("nan"), float("nan")),
                sample_sizes={"high_vol": n1, "low_vol": n2},
                normality_rejected=False,
                underpowered=(n1 < min_sample_size or n2 < min_sample_size),
            )
        )

    return results


# ---------------------------------------------------------------------------
# ET-05: Extended Lag Analysis
# ---------------------------------------------------------------------------


def analyze_lagged_correlations(
    df: pd.DataFrame,
    max_lag_days: int = 14,
    min_sample_size: int = 30,
) -> list[StatTestResult]:
    """ET-05: Cross-correlation of sentiment with Closed PnL at lags 1..max_lag_days.

    Uses row-count-based lags (not calendar days) and Spearman rank
    correlation with Fisher z 95% CIs.  The lag producing the strongest
    absolute correlation is annotated.

    Args:
        df: Feature-store DataFrame with ``sentiment_value`` and
            ``Closed PnL`` sorted by Timestamp.
        max_lag_days: Number of lags to evaluate.
        alpha: Significance level.
        min_sample_size: Minimum pairs below which result is underpowered.

    Returns:
        List of StatTestResults, one per lag.
    """
    working = df.dropna(subset=["sentiment_value", "Closed PnL"]).sort_values("Timestamp").copy()
    results: list[StatTestResult] = []

    for lag in range(1, max_lag_days + 1):
        sentiment_lagged = working["sentiment_value"].shift(lag)
        pnl = working["Closed PnL"]

        mask = sentiment_lagged.notna() & pnl.notna()
        x = sentiment_lagged[mask]
        y = pnl[mask]
        n = len(x)

        if n < 3:  # noqa: PLR2004
            continue

        rho, pval = stats.spearmanr(x, y)
        rho = float(rho)
        pval = float(pval)

        z = np.arctanh(np.clip(rho, -0.999, 0.999))
        se = 1.0 / np.sqrt(n - 3) if n > 3 else np.inf
        ci_low = float(np.tanh(z - 1.96 * se))
        ci_high = float(np.tanh(z + 1.96 * se))

        results.append(
            StatTestResult(
                test_name=f"ET-05_lag_{lag:02d}d_spearman",
                metric=f"sentiment_value[lag={lag}d] vs Closed PnL",
                groups=[f"lag_{lag}d"],
                statistic=rho,
                p_value=pval,
                effect_size=abs(rho),
                effect_size_measure="spearman_rho",
                confidence_interval_95=(ci_low, ci_high),
                sample_sizes={f"lag_{lag}d": n},
                normality_rejected=True,
                underpowered=n < min_sample_size,
            )
        )

    if results:
        best = max(results, key=lambda r: r.effect_size or 0.0)
        logger.info(
            "ET-05: peak correlation at %s (rho=%.4f, p=%.4f)",
            best.test_name,
            best.statistic,
            best.p_value,
        )

    return results


# ---------------------------------------------------------------------------
# ET-06: Post-Hoc Power Analysis
# ---------------------------------------------------------------------------


def compute_power_analysis(
    stat_results: list[StatTestResult],
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> pd.DataFrame:
    """ET-06: Post-hoc minimum detectable effect (MDE) for each test.

    Computes the minimum detectable Cohen's d (for two-group tests) or
    f (for multi-group tests) given the observed sample sizes and the
    desired power using the non-central t / F distributions.

    Note: The formula used is an analytical approximation; simulation-based
    power analysis would be more precise for non-normal data.

    Args:
        stat_results: List of StatTestResults from which to extract n.
        alpha: Type-I error rate.
        power_target: Desired statistical power (1 - β).

    Returns:
        DataFrame with columns: test_name, n_total, observed_effect_size,
        mde_cohens_d, achieved_power_at_observed_es, interpretation.
    """
    from scipy.stats import norm as _norm

    records: list[dict[str, Any]] = []
    z_alpha = _norm.ppf(1 - alpha / 2)
    z_power = _norm.ppf(power_target)

    for res in stat_results:
        n_total = sum(res.sample_sizes.values()) if res.sample_sizes else 0
        if n_total < 2:  # noqa: PLR2004
            continue

        n_per_group = n_total / max(len(res.sample_sizes), 1)

        # Minimum detectable Cohen's d at target power
        mde = (z_alpha + z_power) / np.sqrt(n_per_group / 2) if n_per_group > 1 else float("nan")

        # Achieved power at observed effect size
        obs_es = abs(res.effect_size) if res.effect_size is not None else 0.0
        ncp = obs_es * np.sqrt(n_per_group / 2)
        achieved_power = float(1 - _norm.cdf(z_alpha - ncp))

        if obs_es < 0.2:  # noqa: PLR2004
            interp = "negligible effect"
        elif obs_es < 0.5:  # noqa: PLR2004
            interp = "small effect"
        elif obs_es < 0.8:  # noqa: PLR2004
            interp = "medium effect"
        else:
            interp = "large effect"

        records.append(
            {
                "test_name": res.test_name,
                "n_total": n_total,
                "observed_effect_size": round(obs_es, 4),
                "mde_cohens_d": round(float(mde), 4),
                "achieved_power_at_observed_es": round(achieved_power, 4),
                "interpretation": interp,
                "underpowered": res.underpowered,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ET-07: Win-Streak & Profitability Transition Analysis
# ---------------------------------------------------------------------------


def analyze_win_streaks(
    df: pd.DataFrame,
    min_sample_size: int = 30,
) -> list[StatTestResult]:
    """ET-07: Do consecutive winning trades predict future profitability?

    Within each sentiment regime, computes the proportion of profitable
    trades that follow a "win streak" (2+ consecutive profitable trades)
    versus those following a losing streak.  Runs a two-proportion z-test
    for each regime separately and an omnibus chi-square across all regimes.

    Args:
        df: Feature-store DataFrame with ``classification`` and
            ``Closed PnL`` columns, sorted by Timestamp.
        alpha: Significance level.
        min_sample_size: Minimum group size for underpowered flag.

    Returns:
        List of StatTestResults (one per regime + omnibus chi-square).
    """
    working = df.dropna(subset=["classification", "Closed PnL"]).copy()
    working = working.sort_values("Timestamp").reset_index(drop=True)
    working["_profitable"] = (working["Closed PnL"] > 0).astype(int)

    # Streak state: track consecutive wins
    streak = [0] * len(working)
    for i in range(1, len(working)):
        if working["_profitable"].iloc[i - 1] == 1:
            streak[i] = streak[i - 1] + 1
    working["_win_streak"] = streak
    working["_after_streak"] = (working["_win_streak"] >= 2).astype(int)  # noqa: PLR2004

    results: list[StatTestResult] = []
    regimes = working["classification"].dropna().unique()

    for regime in regimes:
        sub = working[working["classification"] == regime]
        after_streak = sub[sub["_after_streak"] == 1]["_profitable"]
        not_streak = sub[sub["_after_streak"] == 0]["_profitable"]

        n1, n2 = len(after_streak), len(not_streak)
        if n1 < 5 or n2 < 5:  # noqa: PLR2004
            logger.warning("ET-07: regime '%s' has insufficient data, skipping.", regime)
            continue

        p1 = after_streak.mean()
        p2 = not_streak.mean()
        p_pool = (after_streak.sum() + not_streak.sum()) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_stat = float((p1 - p2) / se) if se > 0 else 0.0
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Effect size: Cohen's h
        h = float(2 * np.arcsin(np.sqrt(max(p1, 0.0))) - 2 * np.arcsin(np.sqrt(max(p2, 0.0))))

        results.append(
            StatTestResult(
                test_name=f"ET-07_win_streak_{regime.replace(' ', '_')}",
                metric="profitable_trade_proportion",
                groups=["after_win_streak", "after_losing_streak"],
                statistic=z_stat,
                p_value=p_val,
                effect_size=abs(h),
                effect_size_measure="cohens_h",
                confidence_interval_95=(float("nan"), float("nan")),
                sample_sizes={"after_streak": n1, "not_streak": n2},
                normality_rejected=False,
                underpowered=(n1 < min_sample_size or n2 < min_sample_size),
            )
        )

    # Omnibus chi-square: after-streak-win vs total-win across regimes
    if len(regimes) >= 2:  # noqa: PLR2004
        contingency = []
        row_labels = []
        for regime in regimes:
            sub = working[working["classification"] == regime]
            after = sub[sub["_after_streak"] == 1]["_profitable"].sum()
            not_after = sub[sub["_after_streak"] == 0]["_profitable"].sum()
            if after + not_after > 0:
                contingency.append([int(after), int(not_after)])
                row_labels.append(str(regime))

        if len(contingency) >= 2:  # noqa: PLR2004
            table = np.array(contingency, dtype=float)
            chi2, pval, dof, expected = chi2_contingency(table)
            n_total = int(table.sum())
            v_denom = n_total * (min(table.shape) - 1)
            cramers_v = float(np.sqrt(chi2 / v_denom)) if v_denom > 0 else 0.0
            results.append(
                StatTestResult(
                    test_name="ET-07_win_streak_omnibus_chi2",
                    metric="win_streak_proportion_across_regimes",
                    groups=row_labels,
                    statistic=float(chi2),
                    p_value=float(pval),
                    effect_size=cramers_v,
                    effect_size_measure="cramers_v",
                    confidence_interval_95=(float("nan"), float("nan")),
                    sample_sizes={"n_total": n_total},
                    normality_rejected=False,
                    underpowered=n_total < min_sample_size,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Findings Matrix
# ---------------------------------------------------------------------------


def build_findings_matrix(all_results: list[StatTestResult]) -> pd.DataFrame:
    """Compile all StatTestResults into a structured findings matrix.

    Each row represents one test with its ID, p-value, effect size,
    significance flag, and a plain-English headline.

    Args:
        all_results: All StatTestResult instances from core + extended tests.

    Returns:
        DataFrame with columns: test_name, metric, p_value, effect_size,
        effect_size_measure, significant, corrected_threshold,
        underpowered, groups.
    """
    records = []
    for r in all_results:
        threshold = r.corrected_threshold or 0.05
        significant = r.p_value < threshold if not np.isnan(r.p_value) else False
        records.append(
            {
                "test_name": r.test_name,
                "metric": r.metric,
                "p_value": round(r.p_value, 6),
                "effect_size": round(r.effect_size, 4) if r.effect_size is not None else None,
                "effect_size_measure": r.effect_size_measure,
                "significant": significant,
                "corrected_threshold": round(threshold, 6),
                "underpowered": r.underpowered,
                "groups": " | ".join(r.groups) if r.groups else "",
            }
        )
    return pd.DataFrame(records)
