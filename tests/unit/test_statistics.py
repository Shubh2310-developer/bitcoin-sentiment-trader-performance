"""Unit tests for the statistical analysis layer.

Tests cover normality testing, two-group and multi-group comparisons,
correlation analysis, multiple testing correction, and effect size
reporting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_trader_analytics.statistics.correlation_analysis import (
    compute_correlation,
    compute_correlation_matrix,
)
from sentiment_trader_analytics.statistics.hypothesis_tests import (
    StatTestResult,
    apply_multiple_testing_correction,
    check_normality,
    chi_square_test,
    compare_multiple_groups,
    compare_two_groups,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _normal_series(n: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, size=n), name="test_metric")


def _uniform_series(n: int, low: float = 0.0, high: float = 10.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(low, high, size=n), name="test_metric")


def _exponential_series(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.exponential(scale=2.0, size=n), name="test_metric")


# ── Normality Tests ──────────────────────────────────────────────────


class TestNormality:
    """Tests for the test_normality function."""

    def test_normality_shapiro_small_sample(self) -> None:
        """Shapiro-Wilk should be used for n <= 50."""
        data = _normal_series(30, seed=99)
        result = check_normality(data, alpha=0.05)
        assert result.method == "shapiro"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.normality_rejected, bool)

    def test_normality_dagostino_large_sample(self) -> None:
        """D'Agostino-Pearson should be used for n > 50."""
        data = _normal_series(100, seed=99)
        result = check_normality(data, alpha=0.05)
        assert result.method == "dagostino_pearson"
        assert isinstance(result.statistic, float)

    def test_normality_normal_data_not_rejected(self) -> None:
        """Normality should not be rejected for normal data."""
        data = _normal_series(50, seed=42)
        result = check_normality(data, alpha=0.05)
        # Normal data should not reject at alpha=0.05
        assert not result.normality_rejected

    def test_normality_nonnormal_data_rejected(self) -> None:
        """Normality should be rejected for clearly non-normal data."""
        data = _exponential_series(100, seed=42)
        result = check_normality(data, alpha=0.05)
        # Exponential data is clearly non-normal
        assert result.normality_rejected

    def test_normality_raises_on_too_few(self) -> None:
        """Normality test should raise on fewer than 3 observations."""
        data = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 3"):
            check_normality(data)


# ── Two-Group Comparison Tests ───────────────────────────────────────


class TestCompareTwoGroups:
    """Tests for the compare_two_groups function."""

    def test_compare_two_groups_ttest_on_normal(self) -> None:
        """Normal data with equal variance should trigger t-test."""
        group_a = _normal_series(50, mean=0.0, seed=100)
        group_b = _normal_series(50, mean=0.5, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05)
        assert result.test_name in ("independent_t_test", "welch_t_test")

    def test_compare_two_groups_mann_whitney_on_nonnormal(self) -> None:
        """Non-normal data should trigger Mann-Whitney U."""
        group_a = _exponential_series(50, seed=100)
        group_b = _exponential_series(50, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05)
        assert result.test_name == "mann_whitney_u"

    def test_compare_two_groups_effect_size_present(self) -> None:
        """Effect size should always be present in two-group results."""
        group_a = _normal_series(50, seed=100)
        group_b = _normal_series(50, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05)
        assert result.effect_size is not None
        assert result.effect_size_measure in ("cohens_d", "rank_biserial_r")

    def test_compare_two_groups_underpowered_flag(self) -> None:
        """Tests with n below threshold should be flagged underpowered."""
        group_a = _normal_series(5, seed=100)
        group_b = _normal_series(5, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05, min_sample_size=30)
        assert result.underpowered is True

    def test_compare_two_groups_confidence_interval(self) -> None:
        """95% CI should be a tuple of two floats."""
        group_a = _normal_series(50, seed=100)
        group_b = _normal_series(50, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05)
        assert isinstance(result.confidence_interval_95, tuple)
        assert len(result.confidence_interval_95) == 2
        assert isinstance(result.confidence_interval_95[0], float)
        assert isinstance(result.confidence_interval_95[1], float)


# ── Multi-Group Comparison Tests ─────────────────────────────────────


class TestCompareMultipleGroups:
    """Tests for the compare_multiple_groups function."""

    def test_compare_multiple_groups_kruskal_wallis(self) -> None:
        """Non-normal data should trigger Kruskal-Wallis."""
        groups = {
            "A": _exponential_series(30, seed=100),
            "B": _exponential_series(30, seed=101),
            "C": _exponential_series(30, seed=102),
        }
        result = compare_multiple_groups(groups, alpha=0.05)
        assert result.test_name == "kruskal_wallis"

    def test_compare_multiple_groups_anova(self) -> None:
        """Normal data should trigger ANOVA."""
        groups = {
            "A": _normal_series(30, mean=0.0, seed=100),
            "B": _normal_series(30, mean=0.5, seed=101),
            "C": _normal_series(30, mean=1.0, seed=102),
        }
        result = compare_multiple_groups(groups, alpha=0.05)
        assert result.test_name == "anova_one_way"

    def test_compare_multiple_groups_effect_size_present(self) -> None:
        """Effect size should always be present in multi-group results."""
        groups = {
            "A": _normal_series(30, seed=100),
            "B": _normal_series(30, seed=101),
            "C": _normal_series(30, seed=102),
        }
        result = compare_multiple_groups(groups, alpha=0.05)
        assert result.effect_size is not None
        assert result.effect_size_measure in ("eta_squared", "epsilon_squared")

    def test_compare_multiple_groups_with_near_normal(self) -> None:
        """Multi-group test should return valid result with various data."""
        rng = np.random.default_rng(42)
        groups = {
            "A": pd.Series(rng.normal(0, 1, 25), name="test"),
            "B": pd.Series(rng.normal(0.5, 1, 25), name="test"),
            "C": pd.Series(rng.normal(1.0, 1, 25), name="test"),
        }
        result = compare_multiple_groups(groups, alpha=0.05)
        assert result.statistic is not None
        assert result.p_value is not None


# ── Chi-Square Test ──────────────────────────────────────────────────


class TestChiSquare:
    """Tests for the chi_square_test function."""

    def test_chi_square_basic(self) -> None:
        """Chi-square should produce a valid result."""
        data = pd.DataFrame(
            {
                "A": [30, 10],
                "B": [10, 30],
            },
            index=["Group1", "Group2"],
        )
        result = chi_square_test(data, alpha=0.05)
        assert result.test_name == "chi_square"
        assert result.statistic > 0
        assert isinstance(result.effect_size, float)
        assert result.effect_size_measure == "cramers_v"

    def test_chi_square_effect_size_present(self) -> None:
        """Effect size should be present in chi-square results."""
        data = pd.DataFrame(
            {
                "Yes": [50, 20],
                "No": [30, 40],
            },
            index=["Fear", "Greed"],
        )
        result = chi_square_test(data, alpha=0.05)
        assert result.effect_size is not None
        assert result.effect_size_measure == "cramers_v"


# ── Multiple Testing Correction ──────────────────────────────────────


class TestMultipleTestingCorrection:
    """Tests for apply_multiple_testing_correction."""

    def test_bonferroni_correction(self) -> None:
        """Bonferroni corrected threshold should be alpha / n_tests."""
        results = [
            StatTestResult(
                test_name=f"HT-{i:02d}",
                metric="test",
                groups=["A", "B"],
                statistic=1.0,
                p_value=0.05,
                effect_size=0.5,
                effect_size_measure="cohens_d",
                confidence_interval_95=(0.0, 1.0),
                sample_sizes={"A": 50, "B": 50},
                normality_rejected=False,
            )
            for i in range(4)
        ]
        corrected = apply_multiple_testing_correction(results, method="bonferroni", alpha=0.05)
        for r in corrected:
            assert r.correction_applied == "bonferroni"
            assert r.corrected_threshold == pytest.approx(0.05 / 4)

    def test_fdr_bh_correction(self) -> None:
        """FDR-BH correction should set correction_applied."""
        results = [
            StatTestResult(
                test_name=f"HT-{i:02d}",
                metric="test",
                groups=["A", "B"],
                statistic=1.0,
                p_value=0.03,
                effect_size=0.5,
                effect_size_measure="cohens_d",
                confidence_interval_95=(0.0, 1.0),
                sample_sizes={"A": 50, "B": 50},
                normality_rejected=False,
            )
            for i in range(3)
        ]
        corrected = apply_multiple_testing_correction(results, method="fdr_bh", alpha=0.05)
        for r in corrected:
            assert r.correction_applied == "fdr_bh"
            assert r.corrected_threshold is not None

    def test_correction_empty_list(self) -> None:
        """Correction on empty list should return empty list."""
        corrected = apply_multiple_testing_correction([], method="bonferroni", alpha=0.05)
        assert corrected == []


# ── Effect Size Present ──────────────────────────────────────────────


class TestEffectSizePresence:
    """Tests that effect_size is present in all StatTestResult instances."""

    def test_effect_size_present_in_all_results(self) -> None:
        """StatTestResult.effect_size should never be None."""
        result = StatTestResult(
            test_name="test",
            metric="metric",
            groups=["A"],
            statistic=1.0,
            p_value=0.05,
            effect_size=0.5,
            effect_size_measure="cohens_d",
            confidence_interval_95=(0.0, 1.0),
            sample_sizes={"A": 50},
            normality_rejected=False,
        )
        assert result.effect_size is not None
        assert result.effect_size == 0.5


# ── Underpowered Flag ────────────────────────────────────────────────


class TestUnderpowered:
    """Tests for the underpowered flag."""

    def test_underpowered_flag_on_small_n(self) -> None:
        """underpowered should be True when n is below threshold."""
        group_a = _normal_series(5, seed=100)
        group_b = _normal_series(5, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05, min_sample_size=30)
        assert result.underpowered is True

    def test_not_underpowered_on_large_n(self) -> None:
        """underpowered should be False when n meets threshold."""
        group_a = _normal_series(100, seed=100)
        group_b = _normal_series(100, seed=101)
        result = compare_two_groups(group_a, group_b, alpha=0.05, min_sample_size=30)
        assert result.underpowered is False


# ── Correlation Tests ────────────────────────────────────────────────


class TestCorrelation:
    """Tests for compute_correlation."""

    def test_spearman_correlation_ci(self) -> None:
        """95% CI for Spearman should be a non-null tuple of floats."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 1, 100), name="x")
        y = pd.Series(rng.normal(0, 1, 100), name="y")
        result = compute_correlation(x, y, method="spearman")
        assert result.confidence_interval_95 is not None
        assert isinstance(result.confidence_interval_95, tuple)
        assert len(result.confidence_interval_95) == 2
        assert isinstance(result.confidence_interval_95[0], float)
        assert isinstance(result.confidence_interval_95[1], float)

    def test_pearson_correlation_ci(self) -> None:
        """95% CI for Pearson should use Fisher z."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 1, 100), name="x")
        y = pd.Series(rng.normal(0, 1, 100), name="y")
        result = compute_correlation(x, y, method="pearson")
        assert result.method == "pearson"
        assert len(result.confidence_interval_95) == 2

    def test_spearman_correlation_strong_positive(self) -> None:
        """Spearman should detect a strong positive monotonic relationship."""
        rng = np.random.default_rng(42)
        x = pd.Series(np.arange(100), name="x")
        y = pd.Series(x + rng.normal(0, 5, 100), name="y")
        result = compute_correlation(x, y, method="spearman")
        assert result.coefficient > 0.5
        assert result.p_value < 0.05

    def test_correlation_matrix_basic(self) -> None:
        """Correlation matrix should return a square DataFrame."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 50),
                "b": rng.normal(0, 1, 50),
                "c": rng.normal(0, 1, 50),
            }
        )
        matrix = compute_correlation_matrix(df, ["a", "b", "c"])
        assert matrix.shape == (3, 3)
        assert list(matrix.index) == ["a", "b", "c"]

    def test_correlation_matrix_with_missing_column(self) -> None:
        """Correlation matrix should warn on missing columns but still work."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 50),
                "b": rng.normal(0, 1, 50),
            }
        )
        matrix = compute_correlation_matrix(df, ["a", "b", "missing"])
        assert matrix.shape == (2, 2)

    def test_correlation_matrix_no_valid_columns_raises(self) -> None:
        """Correlation matrix should raise when no columns are found."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="No valid columns"):
            compute_correlation_matrix(df, ["a", "b"])

    def test_correlation_matrix_saves_file(self, tmp_path: Path) -> None:
        """Correlation matrix should save to CSV when output_path is given."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 50),
                "b": rng.normal(0, 1, 50),
            }
        )
        import pathlib

        output_path = str(tmp_path / "corr_matrix.csv")
        matrix = compute_correlation_matrix(df, ["a", "b"], output_path=output_path)
        assert pathlib.Path(output_path).exists()
        assert matrix.shape == (2, 2)

    def test_correlation_raises_on_too_few(self) -> None:
        """Correlation should raise on fewer than 3 pairs."""
        x = pd.Series([1.0, 2.0], name="x")
        y = pd.Series([3.0, 4.0], name="y")
        with pytest.raises(ValueError, match="at least 3"):
            compute_correlation(x, y)


# ── Chi-Square Effect Size ───────────────────────────────────────────


class TestChiSquareEffectSize:
    """Tests for effect size in chi-square."""

    def test_cramers_v_range(self) -> None:
        """Cramer's V should be between 0 and 1."""
        data = pd.DataFrame(
            {
                "Cat1": [50, 10, 5],
                "Cat2": [10, 50, 10],
                "Cat3": [5, 10, 50],
            },
            index=["Fear", "Greed", "Neutral"],
        )
        result = chi_square_test(data, alpha=0.05)
        assert 0.0 <= result.effect_size <= 1.0


# ── Full Result Integrity ────────────────────────────────────────────


class TestResultIntegrity:
    """Tests for StatTestResult integrity."""

    def test_stat_test_result_all_fields(self) -> None:
        """All required fields should be set on StatTestResult."""
        result = StatTestResult(
            test_name="test",
            metric="metric",
            groups=["A"],
            statistic=1.0,
            p_value=0.05,
            effect_size=0.5,
            effect_size_measure="cohens_d",
            confidence_interval_95=(0.0, 1.0),
            sample_sizes={"A": 50},
            normality_rejected=False,
        )
        assert result.test_name == "test"
        assert result.statistic == 1.0
        assert result.p_value == 0.05
        assert result.effect_size == 0.5
        assert isinstance(result.confidence_interval_95, tuple)
        assert result.sample_sizes == {"A": 50}
        assert result.normality_rejected is False
        assert result.correction_applied is None
        assert result.corrected_threshold is None
        assert result.underpowered is False
