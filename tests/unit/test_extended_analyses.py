"""Unit tests for the extended statistical analyses (ET-01 through ET-07).

Tests cover regime transition analysis, duration effects, trader
segmentation, volatility interaction, lagged correlations, power
analysis, win-streak analysis, and the findings matrix builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentiment_trader_analytics.statistics.extended_analyses import (
    analyze_lagged_correlations,
    analyze_regime_duration,
    analyze_regime_transitions,
    analyze_trader_segments,
    analyze_volatility_interaction,
    analyze_win_streaks,
    build_findings_matrix,
    compute_power_analysis,
)
from sentiment_trader_analytics.statistics.hypothesis_tests import StatTestResult

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_feature_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build a minimal synthetic feature-store DataFrame with regime blocks.

    Regimes are assigned in blocks of 10 rows to ensure meaningful
    steady-state (non-transition) windows for ET-01.
    """
    rng = np.random.default_rng(seed)
    regimes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # Build regime column with blocks of 10 consecutive rows per regime
    block_size = 10
    regime_col = [regimes[(i // block_size) % len(regimes)] for i in range(n)]

    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "classification": regime_col,
            "sentiment_value": rng.integers(0, 100, size=n).astype(float),
            "Closed PnL": rng.normal(0, 100, size=n),
            "Leverage": rng.uniform(1, 20, size=n),
            "trader_win_rate_7d": rng.uniform(0, 1, size=n),
            "trader_pnl_rolling_7d": rng.normal(0, 50, size=n),
            "trader_pnl_rolling_30d": rng.normal(0, 100, size=n),
            "trader_leverage_avg_24h": rng.uniform(1, 15, size=n),
            "trader_pnl_volatility_14d": rng.uniform(10, 200, size=n),
            "trader_trade_count_7d": rng.integers(1, 50, size=n).astype(float),
            "trader_avg_size_usd_7d": rng.uniform(100, 10000, size=n),
        }
    )
    return df


def _dummy_stat_result(test_name: str, p: float, es: float, n: int = 100) -> StatTestResult:
    return StatTestResult(
        test_name=test_name,
        metric="test_metric",
        groups=["A", "B"],
        statistic=1.0,
        p_value=p,
        effect_size=es,
        effect_size_measure="cohens_d",
        confidence_interval_95=(0.0, 1.0),
        sample_sizes={"A": n, "B": n},
        normality_rejected=False,
        underpowered=n < 30,
    )


# ── ET-01: Regime Transitions ─────────────────────────────────────────────


class TestRegimeTransitions:
    """Tests for analyze_regime_transitions."""

    def test_returns_stat_test_result(self) -> None:
        """ET-01 should return a valid StatTestResult."""
        df = _make_feature_df()
        result = analyze_regime_transitions(df, window_rows=3)
        assert isinstance(result, StatTestResult)
        assert result.test_name == "ET-01_regime_transition"

    def test_effect_size_present(self) -> None:
        """ET-01 result must always have an effect size."""
        df = _make_feature_df()
        result = analyze_regime_transitions(df, window_rows=3)
        assert result.effect_size is not None

    def test_groups_correct(self) -> None:
        """ET-01 groups should be post_transition and steady_state."""
        df = _make_feature_df()
        result = analyze_regime_transitions(df, window_rows=3)
        assert "post_transition" in result.groups
        assert "steady_state" in result.groups

    def test_raises_on_missing_columns(self) -> None:
        """ET-01 should raise ValueError for DataFrames missing required columns."""
        df = pd.DataFrame({"Timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC")})
        with pytest.raises(ValueError, match="classification"):
            analyze_regime_transitions(df)


# ── ET-02: Regime Duration ────────────────────────────────────────────────


class TestRegimeDuration:
    """Tests for analyze_regime_duration."""

    def test_returns_stat_test_result(self) -> None:
        """ET-02 should return a valid StatTestResult."""
        df = _make_feature_df()
        result = analyze_regime_duration(df, duration_buckets=[1, 3, 7, 14])
        assert isinstance(result, StatTestResult)
        assert result.test_name == "ET-02_regime_duration"

    def test_effect_size_present(self) -> None:
        """ET-02 effect size must be non-None."""
        df = _make_feature_df()
        result = analyze_regime_duration(df)
        assert result.effect_size is not None

    def test_default_buckets(self) -> None:
        """ET-02 should use [1, 3, 7, 14] as default buckets."""
        df = _make_feature_df()
        result = analyze_regime_duration(df)
        assert isinstance(result, StatTestResult)


# ── ET-03: Trader Segmentation ────────────────────────────────────────────


class TestTraderSegmentation:
    """Tests for analyze_trader_segments."""

    def test_returns_list_of_results(self) -> None:
        """ET-03 should return a list of StatTestResult objects."""
        df = _make_feature_df()
        results = analyze_trader_segments(df, k_min=2, k_max=4)
        assert isinstance(results, list)
        assert len(results) >= 2  # noqa: PLR2004

    def test_first_result_is_cluster_comparison(self) -> None:
        """First result should be the inter-cluster PnL comparison."""
        df = _make_feature_df()
        results = analyze_trader_segments(df, k_min=2, k_max=3)
        assert results[0].test_name == "ET-03_trader_segmentation"

    def test_silhouette_meta_result(self) -> None:
        """Second result should carry silhouette scores."""
        df = _make_feature_df()
        results = analyze_trader_segments(df, k_min=2, k_max=3)
        meta = results[1]
        assert "silhouette" in meta.test_name
        assert meta.effect_size_measure == "silhouette"

    def test_raises_on_missing_features(self) -> None:
        """ET-03 should raise if fewer than 2 required columns are available."""
        df = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2024-01-01", periods=10),
                "Closed PnL": [1.0] * 10,
            }
        )
        with pytest.raises(ValueError, match="at least 2"):
            analyze_trader_segments(df)


# ── ET-04: Volatility Interaction ─────────────────────────────────────────


class TestVolatilityInteraction:
    """Tests for analyze_volatility_interaction."""

    def test_returns_list(self) -> None:
        """ET-04 should return a list of StatTestResults."""
        df = _make_feature_df()
        results = analyze_volatility_interaction(df)
        assert isinstance(results, list)
        assert len(results) >= 2  # noqa: PLR2004

    def test_high_low_vol_results(self) -> None:
        """ET-04 should produce results for both high and low volatility."""
        df = _make_feature_df()
        results = analyze_volatility_interaction(df)
        names = [r.test_name for r in results]
        assert any("high_volatility" in n for n in names)
        assert any("low_volatility" in n for n in names)

    def test_fisher_z_result(self) -> None:
        """ET-04 should include a Fisher z-test comparing the two correlations."""
        df = _make_feature_df()
        results = analyze_volatility_interaction(df)
        names = [r.test_name for r in results]
        assert any("fisher_z" in n for n in names)


# ── ET-05: Extended Lag Analysis ──────────────────────────────────────────


class TestLaggedCorrelations:
    """Tests for analyze_lagged_correlations."""

    def test_returns_list_of_length_max_lag(self) -> None:
        """ET-05 should return one result per lag."""
        df = _make_feature_df(n=500)
        results = analyze_lagged_correlations(df, max_lag_days=5)
        assert len(results) == 5  # noqa: PLR2004

    def test_all_results_have_effect_size(self) -> None:
        """All lag results should have non-None effect sizes."""
        df = _make_feature_df(n=500)
        results = analyze_lagged_correlations(df, max_lag_days=3)
        for r in results:
            assert r.effect_size is not None
            assert r.effect_size_measure == "spearman_rho"

    def test_lag_names_are_sequential(self) -> None:
        """Test names should contain 'lag_01d' through 'lag_NNd'."""
        df = _make_feature_df(n=500)
        results = analyze_lagged_correlations(df, max_lag_days=3)
        for i, r in enumerate(results, start=1):
            assert f"lag_{i:02d}d" in r.test_name


# ── ET-06: Power Analysis ─────────────────────────────────────────────────


class TestPowerAnalysis:
    """Tests for compute_power_analysis."""

    def test_returns_dataframe(self) -> None:
        """Power analysis should return a DataFrame."""
        results = [
            _dummy_stat_result("T-01", 0.04, 0.3, 100),
            _dummy_stat_result("T-02", 0.20, 0.1, 50),
        ]
        df = compute_power_analysis(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # noqa: PLR2004

    def test_required_columns_present(self) -> None:
        """Power analysis DataFrame must have the required columns."""
        results = [_dummy_stat_result("T-01", 0.04, 0.3)]
        df = compute_power_analysis(results)
        required = {
            "test_name",
            "n_total",
            "observed_effect_size",
            "mde_cohens_d",
            "achieved_power_at_observed_es",
            "interpretation",
            "underpowered",
        }
        assert required.issubset(set(df.columns))

    def test_large_effect_high_power(self) -> None:
        """A large effect size with many observations should have high power."""
        results = [_dummy_stat_result("T-01", 0.001, 0.9, 500)]
        df = compute_power_analysis(results)
        assert float(df["achieved_power_at_observed_es"].iloc[0]) > 0.7  # noqa: PLR2004

    def test_empty_list_returns_empty_df(self) -> None:
        """An empty list of results should return an empty DataFrame."""
        df = compute_power_analysis([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ── ET-07: Win-Streak Analysis ────────────────────────────────────────────


class TestWinStreaks:
    """Tests for analyze_win_streaks."""

    def test_returns_list(self) -> None:
        """ET-07 should return a list of StatTestResults."""
        df = _make_feature_df(n=500)
        results = analyze_win_streaks(df)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_omnibus_chi2_present(self) -> None:
        """ET-07 should include an omnibus chi-square result."""
        df = _make_feature_df(n=500)
        results = analyze_win_streaks(df)
        names = [r.test_name for r in results]
        assert any("omnibus" in n for n in names)

    def test_effect_size_cohens_h(self) -> None:
        """Per-regime results should use Cohen's h as effect size measure."""
        df = _make_feature_df(n=500)
        results = analyze_win_streaks(df)
        per_regime = [r for r in results if "omnibus" not in r.test_name]
        for r in per_regime:
            assert r.effect_size_measure == "cohens_h"


# ── Findings Matrix ───────────────────────────────────────────────────────


class TestFindingsMatrix:
    """Tests for build_findings_matrix."""

    def test_returns_dataframe_with_required_columns(self) -> None:
        """Findings matrix should have all required columns."""
        results = [
            _dummy_stat_result("HT-01", 0.01, 0.4, 200),
            _dummy_stat_result("HT-02", 0.40, 0.1, 100),
        ]
        df = build_findings_matrix(results)
        required = {"test_name", "metric", "p_value", "effect_size", "significant"}
        assert required.issubset(set(df.columns))

    def test_significant_flag_correct(self) -> None:
        """Significant flag should be True for p < corrected_threshold."""
        r = _dummy_stat_result("HT-01", 0.001, 0.6, 200)
        r.corrected_threshold = 0.05
        df = build_findings_matrix([r])
        assert bool(df["significant"].iloc[0]) is True

    def test_not_significant_flag_correct(self) -> None:
        """Significant flag should be False for p >= corrected_threshold."""
        r = _dummy_stat_result("HT-02", 0.40, 0.1, 200)
        r.corrected_threshold = 0.05
        df = build_findings_matrix([r])
        assert bool(df["significant"].iloc[0]) is False

    def test_length_matches_input(self) -> None:
        """Findings matrix row count should match input list length."""
        results = [_dummy_stat_result(f"T-{i}", 0.05, 0.2) for i in range(5)]
        df = build_findings_matrix(results)
        assert len(df) == 5  # noqa: PLR2004
