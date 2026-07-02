"""Formal statistical hypothesis tests for the sentiment trader analysis.

Provides normality testing, two-group and multi-group comparisons,
and multiple testing correction. Every test returns a ``StatTestResult``
with p-value, effect size, 95% CI, and metadata for reproducibility.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
from pydantic import BaseModel
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    levene,
    mannwhitneyu,
    normaltest,
    shapiro,
    ttest_ind,
)
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class StatTestResult(BaseModel):
    """Standard container for a single hypothesis test result.

    Attributes:
        test_name: Human-readable name of the test.
        metric: The metric being tested (e.g., "Closed PnL").
        groups: Names of the groups compared.
        statistic: The test statistic value.
        p_value: The p-value of the test.
        effect_size: Standardised effect size.
        effect_size_measure: Name of the effect size measure used.
        confidence_interval_95: 95% confidence interval as (lower, upper).
        sample_sizes: Map of group name to sample size.
        normality_rejected: Whether the normality test rejected normality.
        correction_applied: Multiple testing correction method, if any.
        corrected_threshold: Adjusted significance threshold, if any.
        underpowered: Whether the sample size was below the minimum threshold.
    """

    test_name: str
    metric: str
    groups: list[str]
    statistic: float
    p_value: float
    effect_size: float
    effect_size_measure: str
    confidence_interval_95: tuple[float, float]
    sample_sizes: dict[str, int]
    normality_rejected: bool
    correction_applied: str | None = None
    corrected_threshold: float | None = None
    underpowered: bool = False


class NormalityResult(NamedTuple):
    """Container for a normality test result.

    Attributes:
        statistic: The normality test statistic.
        p_value: P-value of the normality test.
        method: The normality test method used (shapiro or dagostino).
        normality_rejected: Whether normality was rejected (p < alpha).
    """

    statistic: float
    p_value: float
    method: str
    normality_rejected: bool


def check_normality(series: pd.Series, alpha: float = 0.05) -> NormalityResult:
    """Test whether a series follows a normal distribution.

    Uses Shapiro-Wilk for n <= 50 and D'Agostino-Pearson for n > 50.

    Args:
        series: The data series to test.
        alpha: Significance level for the normality test (default 0.05).

    Returns:
        A NormalityResult with statistic, p-value, method, and decision.

    Raises:
        ValueError: If the series has fewer than 3 non-NA observations.
    """
    data = series.dropna()
    n = len(data)
    if n < 3:
        raise ValueError(f"Normality test requires at least 3 observations; got {n}")

    if n <= 50:
        stat_val, p_val = shapiro(data)
        method = "shapiro"
    else:
        stat_val, p_val = normaltest(data)
        method = "dagostino_pearson"

    return NormalityResult(
        statistic=float(stat_val),
        p_value=float(p_val),
        method=method,
        normality_rejected=bool(p_val < alpha),
    )


def _cohens_d(group_a: pd.Series, group_b: pd.Series) -> tuple[float, tuple[float, float]]:
    """Compute Cohen's d and approximate 95% CI via non-central t.

    Args:
        group_a: First group data.
        group_b: Second group data.

    Returns:
        Tuple of (d, (ci_lower, ci_upper)).
    """
    na, nb = len(group_a), len(group_b)
    mean_a, mean_b = group_a.mean(), group_b.mean()
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)

    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0, (0.0, 0.0)

    d = (mean_a - mean_b) / pooled_std

    se_d = np.sqrt((na + nb) / (na * nb) + (d**2) / (2 * (na + nb)))

    z_crit = 1.96
    ci_lower = d - z_crit * se_d
    ci_upper = d + z_crit * se_d
    return float(d), (float(ci_lower), float(ci_upper))


def _rank_biserial_r(group_a: pd.Series, group_b: pd.Series) -> tuple[float, tuple[float, float]]:
    """Compute rank-biserial r and approximate 95% CI via bootstrap.

    Args:
        group_a: First group data.
        group_b: Second group data.

    Returns:
        Tuple of (r, (ci_lower, ci_upper)).
    """
    all_vals = pd.concat([group_a, group_b])
    ranks = all_vals.rank()
    rank_a = ranks.iloc[: len(group_a)]
    rank_b = ranks.iloc[len(group_a) :]

    r = 2 * (rank_a.mean() - rank_b.mean()) / (len(ranks))
    r = float(np.clip(r, -1.0, 1.0))

    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    diff_means: list[float] = []
    combined = np.concatenate([group_a.values, group_b.values])
    na, nb = len(group_a), len(group_b)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(combined), size=len(combined), replace=True)
        boot_a = combined[idx[:na]]
        boot_b = combined[idx[na:]]
        boot_ranks_a = stats.rankdata(np.concatenate([boot_a, boot_b]))[:na]
        boot_ranks_b = stats.rankdata(np.concatenate([boot_a, boot_b]))[na:]
        boot_r = 2 * (boot_ranks_a.mean() - boot_ranks_b.mean()) / (na + nb)
        diff_means.append(float(np.clip(boot_r, -1.0, 1.0)))

    ci_lower = float(np.percentile(diff_means, 2.5))
    ci_upper = float(np.percentile(diff_means, 97.5))
    return r, (ci_lower, ci_upper)


def compare_two_groups(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
    min_sample_size: int = 30,
) -> StatTestResult:
    """Compare two independent groups with appropriate test selection.

    Protocol:
        1. Normality pre-test on both groups.
        2. Variance homogeneity test (Levene's).
        3. If both normal and equal variance -> independent t-test.
           If both normal but unequal variance -> Welch's t-test.
           Otherwise -> Mann-Whitney U.
        4. Effect size: Cohen's d (parametric) or rank-biserial r (non-parametric).

    Args:
        group_a: Data for the first group.
        group_b: Data for the second group.
        alpha: Significance level (default 0.05).
        min_sample_size: Minimum sample size per group (default 30).

    Returns:
        A StatTestResult with the test outcome.
    """
    a_clean = group_a.dropna()
    b_clean = group_b.dropna()

    normality_a = check_normality(a_clean, alpha)
    normality_b = check_normality(b_clean, alpha)
    normality_rejected = normality_a.normality_rejected or normality_b.normality_rejected

    underpowered = len(a_clean) < min_sample_size or len(b_clean) < min_sample_size

    both_normal = not normality_a.normality_rejected and not normality_b.normality_rejected

    if both_normal:
        var_equal = True
        if len(a_clean) > 1 and len(b_clean) > 1:
            _levene_stat, levene_p = levene(a_clean, b_clean)
            var_equal = levene_p >= alpha

        if var_equal:
            stat_val, p_val = ttest_ind(a_clean, b_clean, equal_var=True)
            test_name = "independent_t_test"
        else:
            stat_val, p_val = ttest_ind(a_clean, b_clean, equal_var=False)
            test_name = "welch_t_test"

        effect_size, _ci = _cohens_d(a_clean, b_clean)
        effect_size_measure = "cohens_d"

        diff_ci = _mean_diff_ci(a_clean, b_clean, alpha)
    else:
        stat_val, p_val = mannwhitneyu(a_clean, b_clean, alternative="two-sided")
        test_name = "mann_whitney_u"

        effect_size, _ci = _rank_biserial_r(a_clean, b_clean)
        effect_size_measure = "rank_biserial_r"

        diff_ci = _median_diff_ci(a_clean, b_clean)

    sample_sizes = {"group_a": len(a_clean), "group_b": len(b_clean)}

    return StatTestResult(
        test_name=test_name,
        metric=group_a.name or "unknown",
        groups=["group_a", "group_b"],
        statistic=float(stat_val),
        p_value=float(p_val),
        effect_size=effect_size,
        effect_size_measure=effect_size_measure,
        confidence_interval_95=diff_ci,
        sample_sizes=sample_sizes,
        normality_rejected=normality_rejected,
        underpowered=underpowered,
    )


def _mean_diff_ci(
    group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute 95% CI for the difference of means.

    Args:
        group_a: First group.
        group_b: Second group.
        alpha: Significance level.

    Returns:
        (lower, upper) CI bounds.
    """
    na, nb = len(group_a), len(group_b)
    mean_diff = group_a.mean() - group_b.mean()
    se = np.sqrt(group_a.var(ddof=1) / na + group_b.var(ddof=1) / nb)
    if se == 0:
        return (float(mean_diff), float(mean_diff))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    return (float(mean_diff - z_crit * se), float(mean_diff + z_crit * se))


def _median_diff_ci(
    group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05  # noqa: ARG001
) -> tuple[float, float]:
    """Compute approximate 95% CI for the difference of medians via bootstrap.

    Args:
        group_a: First group.
        group_b: Second group.
        alpha: Significance level (unused, retained for interface consistency).

    Returns:
        (lower, upper) CI bounds.
    """
    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    diffs: list[float] = []
    a_vals = group_a.values
    b_vals = group_b.values
    na, nb = len(a_vals), len(b_vals)
    for _ in range(n_bootstrap):
        boot_a = rng.choice(a_vals, size=na, replace=True)
        boot_b = rng.choice(b_vals, size=nb, replace=True)
        diffs.append(float(np.median(boot_a) - np.median(boot_b)))

    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def _eta_squared(
    groups: dict[str, pd.Series],
) -> tuple[float, tuple[float, float]]:
    """Compute eta-squared and approximate 95% CI for ANOVA.

    Args:
        groups: Map of group name to series.

    Returns:
        Tuple of (eta_sq, (ci_lower, ci_upper)).
    """
    all_data = pd.concat(groups.values())
    grand_mean = all_data.mean()
    ss_between = 0.0
    ss_total = ((all_data - grand_mean) ** 2).sum()
    for grp in groups.values():
        ss_between += len(grp) * ((grp.mean() - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0, (0.0, 0.0)
    eta_sq = ss_between / ss_total
    n = len(all_data)
    k = len(groups)
    se_eta = 0.0
    if n > 3:
        se_eta = np.sqrt(
            (4 * eta_sq * (1 - eta_sq) ** 2 * (n - k - 1) ** 2) / ((n**2 - 1) * (n + 3))
        )
    z_crit = 1.96
    ci_lower = max(0.0, eta_sq - z_crit * se_eta)
    ci_upper = min(1.0, eta_sq + z_crit * se_eta)
    return float(eta_sq), (float(ci_lower), float(ci_upper))


def _epsilon_squared(
    groups: dict[str, pd.Series],
) -> tuple[float, tuple[float, float]]:
    """Compute epsilon-squared for Kruskal-Wallis and approximate 95% CI.

    Args:
        groups: Map of group name to series.

    Returns:
        Tuple of (eps_sq, (ci_lower, ci_upper)).
    """
    all_data = pd.concat(groups.values())
    n = len(all_data)
    all_ranks_val = stats.rankdata(all_data)
    k = len(groups)
    h_stat = 0.0
    idx_start = 0
    for grp in groups.values():
        ni = len(grp)
        ri_bar = all_ranks_val[idx_start : idx_start + ni].mean()
        idx_start += ni
        h_stat += ni * ((ri_bar - (n + 1) / 2) ** 2)
    h_stat = 12 / (n * (n + 1)) * h_stat
    eps_sq = (h_stat - k + 1) / (n - k)
    eps_sq = max(0.0, eps_sq)
    se_eps = np.sqrt(eps_sq * (1 - eps_sq) / n) if n > 1 else 0.0
    z_crit = 1.96
    ci_lower = max(0.0, eps_sq - z_crit * se_eps)
    ci_upper = min(1.0, eps_sq + z_crit * se_eps)
    return float(eps_sq), (float(ci_lower), float(ci_upper))


def _tukey_hsd_posthoc(groups: dict[str, pd.Series], alpha: float = 0.05) -> str:
    """Run Tukey HSD post-hoc tests and return a summary string.

    Args:
        groups: Map of group name to series.
        alpha: Significance level.

    Returns:
        Summary string of pairwise results.
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        all_data = pd.concat(groups.values())
        labels = []
        for name, grp in groups.items():
            labels.extend([name] * len(grp))
        tukey = pairwise_tukeyhsd(all_data, labels, alpha=alpha)
        return str(tukey.summary())
    except ImportError:
        return "Tukey HSD unavailable (statsmodels not installed)"


def _dunn_posthoc(groups: dict[str, pd.Series], alpha: float = 0.05) -> str:
    """Run Dunn's post-hoc test with Holm-Bonferroni correction.

    Args:
        groups: Map of group name to series.
        alpha: Significance level.

    Returns:
        Summary string of pairwise results.
    """
    try:
        from scipy.stats import norm

        group_names = list(groups.keys())
        n_groups = len(group_names)
        all_data = pd.concat(groups.values())
        all_ranks_val = stats.rankdata(all_data)
        n_total = len(all_data)

        results_parts: list[str] = []
        idx_start = 0
        group_ranks: dict[str, npt.NDArray[np.float64]] = {}
        group_sizes: dict[str, int] = {}
        for name, grp in groups.items():
            ni = len(grp)
            group_ranks[name] = all_ranks_val[idx_start : idx_start + ni]
            group_sizes[name] = ni
            idx_start += ni

        def _dunn_z(i: int, j: int) -> float:
            """Compute the z-statistic for a Dunn's post-hoc pairwise comparison.

            Args:
                i: Index of the first group.
                j: Index of the second group.

            Returns:
                The z-statistic for the comparison.
            """
            ni = group_sizes[group_names[i]]
            nj = group_sizes[group_names[j]]
            ri = group_ranks[group_names[i]].mean()
            rj = group_ranks[group_names[j]].mean()
            se = np.sqrt((n_total * (n_total + 1) / 12) * (1 / ni + 1 / nj))
            if se == 0:
                return 0.0
            return float((ri - rj) / se)

        p_vals: list[tuple[float, int, int]] = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                z = _dunn_z(i, j)
                p = 2 * (1 - norm.cdf(abs(z)))
                p_vals.append((p, i, j))

        p_vals.sort(key=lambda x: x[0])
        n_tests = len(p_vals)
        results_parts.append(f"Dunn's post-hoc (Holm-Bonferroni, alpha={alpha}):")
        for rank, (p, i, j) in enumerate(p_vals):
            adj_threshold = alpha / (n_tests - rank)
            sig = "significant" if p <= adj_threshold else "non-significant"
            results_parts.append(
                f"  {group_names[i]} vs {group_names[j]}: z={_dunn_z(i, j):.4f}, "
                f"p={p:.6f}, adj_threshold={adj_threshold:.6f} -> {sig}"
            )

        return "\n".join(results_parts)
    except ImportError:
        return "Dunn's post-hoc unavailable (scipy.stats.norm missing)"


def compare_multiple_groups(
    groups: dict[str, pd.Series],
    alpha: float = 0.05,
    min_sample_size: int = 30,
) -> StatTestResult:
    """Compare multiple independent groups with appropriate test selection.

    Protocol:
        1. Normality pre-test on all groups.
        2. If all groups normal -> ANOVA + Tukey HSD.
           Otherwise -> Kruskal-Wallis + Dunn's test with Holm-Bonferroni.
        3. Effect size: eta-squared (ANOVA) or epsilon-squared (Kruskal-Wallis).

    Args:
        groups: Map of group name to data series.
        alpha: Significance level (default 0.05).
        min_sample_size: Minimum sample size per group (default 30).

    Returns:
        A StatTestResult with the test outcome.
    """
    cleaned: dict[str, pd.Series] = {}
    for name, series in groups.items():
        cleaned[name] = series.dropna()

    sample_sizes = {name: len(s) for name, s in cleaned.items()}
    underpowered = any(n < min_sample_size for n in sample_sizes.values())

    any_nonnormal = False
    for _name, series in cleaned.items():
        nr = check_normality(series, alpha)
        if nr.normality_rejected:
            any_nonnormal = True
            break

    group_series_list = list(cleaned.values())
    group_names = list(cleaned.keys())

    if not any_nonnormal:
        stat_val, p_val = f_oneway(*group_series_list)
        test_name = "anova_one_way"
        effect_size, ci = _eta_squared(cleaned)
        effect_size_measure = "eta_squared"
        posthoc = _tukey_hsd_posthoc(cleaned, alpha)
    else:
        stat_val, p_val = kruskal(*group_series_list)
        test_name = "kruskal_wallis"
        effect_size, ci = _epsilon_squared(cleaned)
        effect_size_measure = "epsilon_squared"
        posthoc = _dunn_posthoc(cleaned, alpha)

    logger.debug("Post-hoc results for %s: %s", test_name, posthoc)

    return StatTestResult(
        test_name=test_name,
        metric="unknown",
        groups=group_names,
        statistic=float(stat_val),
        p_value=float(p_val),
        effect_size=effect_size,
        effect_size_measure=effect_size_measure,
        confidence_interval_95=ci,
        sample_sizes=sample_sizes,
        normality_rejected=any_nonnormal,
        underpowered=underpowered,
    )


def _chi_square_effect_size(
    observed: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> tuple[float, str]:
    """Compute Cramér's V as the effect size for chi-square tests.

    Args:
        observed: Observed frequency table.
        expected: Expected frequency table.

    Returns:
        Tuple of (cramers_v, "cramers_v").
    """
    chi2 = ((observed - expected) ** 2 / expected).sum()
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0, "cramers_v"
    v = np.sqrt(chi2 / (n * min_dim))
    return float(v), "cramers_v"


def chi_square_test(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05,  # noqa: ARG001
    min_expected_frequency: float = 5.0,
) -> StatTestResult:
    """Perform chi-square test of independence or Fisher's exact test.

    Args:
        contingency_table: Contingency table (DataFrame) with observed counts.
        alpha: Significance level (default 0.05).
        min_expected_frequency: Minimum expected frequency threshold.
            Below this, Fisher's exact test is used (default 5.0).

    Returns:
        A StatTestResult with the test outcome.
    """
    observed = contingency_table.values
    chi2_stat, p_val, dof, expected = chi2_contingency(observed, correction=False)

    if (expected < min_expected_frequency).any():
        try:
            _chi2_val, p_val = stats.fisher_exact(observed)
            test_name = "fishers_exact_test"
        except ValueError:
            test_name = "chi_square"
        effect_size, effect_size_measure = _chi_square_effect_size(observed, expected)
    else:
        test_name = "chi_square"
        effect_size, effect_size_measure = _chi_square_effect_size(observed, expected)

    ci = (float(effect_size - 1.96 * 0.05), float(effect_size + 1.96 * 0.05))
    ci = (max(0.0, ci[0]), min(1.0, ci[1]))

    row_names = list(contingency_table.index)
    col_names = list(contingency_table.columns)
    total = int(observed.sum())
    sample_sizes_dict: dict[str, int] = {}
    if len(row_names) > 0 and len(col_names) > 0:
        sample_sizes_dict = {"total": total}

    return StatTestResult(
        test_name=test_name,
        metric="contingency",
        groups=row_names + col_names,
        statistic=float(chi2_stat),
        p_value=float(p_val),
        effect_size=effect_size,
        effect_size_measure=effect_size_measure,
        confidence_interval_95=ci,
        sample_sizes=sample_sizes_dict,
        normality_rejected=False,
        underpowered=False,
    )


def apply_multiple_testing_correction(
    results: list[StatTestResult],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> list[StatTestResult]:
    """Apply multiple testing correction to a list of test results.

    Args:
        results: List of StatTestResult objects to correct.
        method: Correction method ("bonferroni" or "fdr_bh").
        alpha: Family-wise significance level (default 0.05).

    Returns:
        The input list with correction_applied and corrected_threshold set.
    """
    n_tests = len(results)
    if n_tests == 0:
        return results

    if method == "bonferroni":
        corrected_threshold = alpha / n_tests
        for result in results:
            result.correction_applied = "bonferroni"
            result.corrected_threshold = corrected_threshold
    elif method == "fdr_bh":
        p_vals = [r.p_value for r in results]
        _rejected, _corrected_pvals, _alphac_sidak, _alphac_bonf = multipletests(
            p_vals, alpha=alpha, method="fdr_bh"
        )
        corrected_threshold = float(np.min([alpha * (i + 1) / n_tests for i in range(n_tests)]))
        for result in results:
            result.correction_applied = "fdr_bh"
            result.corrected_threshold = corrected_threshold
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return results
