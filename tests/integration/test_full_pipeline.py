"""Integration tests for the full sentiment trader analytics pipeline.

Tests cover the end-to-end data flow from ingestion through feature
engineering, statistical analysis, and data-leakage detection using
deterministic fixture files.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from data.metadata.schemas.fear_greed_schema import fear_greed_schema
from data.metadata.schemas.trader_history_schema import trader_history_schema
from sentiment_trader_analytics.config import (
    FeatureConfig,
    IngestionConfig,
    PreprocessingConfig,
    StatConfig,
)
from sentiment_trader_analytics.feature_engineering.sentiment_features import (
    add_sentiment_fear_greed_flags,
    add_sentiment_lag,
    add_sentiment_regime_encoding,
    add_sentiment_rolling_mean,
)
from sentiment_trader_analytics.feature_engineering.time_features import (
    add_day_of_week,
    add_month,
    add_time_of_day,
)
from sentiment_trader_analytics.feature_engineering.trader_features import (
    add_trader_avg_size,
    add_trader_leverage_avg,
    add_trader_pnl_rolling,
    add_trader_pnl_volatility,
    add_trader_trade_count,
    add_trader_win_rate,
    flag_cold_start_rows,
)
from sentiment_trader_analytics.ingestion.fear_greed_loader import load_fear_greed_index
from sentiment_trader_analytics.ingestion.trader_history_loader import (
    load_trader_history,
)
from sentiment_trader_analytics.preprocessing.cleaning import (
    clean_fear_greed,
    clean_trader_history,
)
from sentiment_trader_analytics.preprocessing.merging import (
    extract_trade_date,
    merge_sentiment_and_trades,
)
from sentiment_trader_analytics.statistics.hypothesis_tests import (
    check_normality,
    chi_square_test,
    compare_two_groups,
)
from sentiment_trader_analytics.validation.schema_checks import (
    validate_fear_greed,
    validate_trader_history,
)

FIXTURES = Path("tests/fixtures")


# ── Helpers ──────────────────────────────────────────────────────────────


def _ingestion_config(tmp_path: Path, fg_file: str, th_file: str) -> IngestionConfig:
    return IngestionConfig(
        fear_greed_path=str(FIXTURES / fg_file),
        trader_history_path=str(FIXTURES / th_file),
        lineage_output_dir=str(tmp_path / "lineage"),
        chunk_size=1000,
    )


def _preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig()


def _feature_config() -> FeatureConfig:
    return FeatureConfig()


def _stat_config() -> StatConfig:
    return StatConfig()


def _load_th_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.float64), unit="ms", utc=True)
    df["Account"] = df["Account"].astype(str)
    float_cols = ["Size USD", "Execution Price", "Closed PnL", "Fee"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    if "Leverage" in df.columns:
        df["Leverage"] = pd.to_numeric(df["Leverage"], errors="coerce").astype(np.float64)
    return df


def _load_fg_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["value"] = df["value"].astype(np.int64)
    df["classification"] = df["classification"].astype(
        pd.CategoricalDtype(
            categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
            ordered=True,
        )
    )
    return df


# ── Test: Ingestion to Features ──────────────────────────────────────────


class TestFullPipelineIngestionToFeatures:
    """End-to-end: ingestion → validation → preprocessing → feature engineering."""

    def test_full_pipeline_ingestion_to_features(self, tmp_path: Path) -> None:
        fg_config = _ingestion_config(tmp_path, "fear_greed_sample.csv", "trader_history_valid.csv")

        # Stage 1: Ingestion
        fg_df = load_fear_greed_index(fg_config)
        th_df = load_trader_history(fg_config)

        assert len(fg_df) == 30
        assert len(th_df) == 10

        # Stage 2: Validation
        fg_valid = validate_fear_greed(fg_df, fear_greed_schema)
        assert fg_valid.passed is True, f"FG validation failed: {fg_valid.violations}"

        th_valid = validate_trader_history(th_df, trader_history_schema)
        assert th_valid.passed is True, f"TH validation failed: {th_valid.violations}"

        # Stage 3: Preprocessing
        fg_clean = clean_fear_greed(fg_valid.validated_df, _preprocessing_config())
        th_clean = clean_trader_history(th_valid.validated_df, _preprocessing_config())

        th_clean = extract_trade_date(th_clean)
        merged = merge_sentiment_and_trades(th_clean, fg_clean, _preprocessing_config())

        assert len(merged) > 0
        assert "sentiment_missing" in merged.columns
        assert "trade_date_utc" in merged.columns

        # Stage 4: Feature Engineering
        config = _feature_config()
        feature_functions = [
            add_sentiment_lag,
            add_sentiment_regime_encoding,
            add_sentiment_rolling_mean,
            add_sentiment_fear_greed_flags,
            add_trader_win_rate,
            add_trader_pnl_rolling,
            add_trader_leverage_avg,
            add_trader_pnl_volatility,
            add_trader_trade_count,
            add_trader_avg_size,
            flag_cold_start_rows,
            add_time_of_day,
            add_day_of_week,
            add_month,
        ]

        features = reduce(lambda d, fn: fn(d, config), feature_functions, merged)

        expected_features = {
            "sentiment_value_lag_1d",
            "sentiment_regime_encoded",
            "sentiment_value_rolling_7d",
            "sentiment_is_fear",
            "sentiment_is_greed",
            "trader_win_rate_7d",
            "trader_pnl_rolling_7d",
            "trader_pnl_rolling_30d",
            "trader_leverage_avg_24h",
            "trader_pnl_volatility_14d",
            "trader_trade_count_7d",
            "trader_avg_size_usd_7d",
            "feature_cold_start",
            "time_hour_utc",
            "time_day_of_week",
            "time_is_weekend",
            "time_month",
        }

        present = set(features.columns)
        missing = expected_features - present
        assert not missing, f"Missing expected feature columns: {missing}"

        # Feature schema checks
        assert features["feature_cold_start"].dtype == bool
        win_rate = features["trader_win_rate_7d"].dropna()
        if len(win_rate) > 0:
            assert win_rate.between(0, 1, inclusive="both").all()


# ── Test: Statistics Pipeline ─────────────────────────────────────────────


class TestFullPipelineStatistics:
    """End-to-end: run statistical analysis on feature-engineered data."""

    def test_full_pipeline_statistics(self, tmp_path: Path) -> None:
        fg_config = _ingestion_config(tmp_path, "fear_greed_sample.csv", "trader_history_valid.csv")

        fg_df = load_fear_greed_index(fg_config)
        th_df = load_trader_history(fg_config)

        fg_valid = validate_fear_greed(fg_df, fear_greed_schema)
        th_valid = validate_trader_history(th_df, trader_history_schema)

        fg_clean = clean_fear_greed(fg_valid.validated_df, _preprocessing_config())
        th_clean = clean_trader_history(th_valid.validated_df, _preprocessing_config())
        th_clean = extract_trade_date(th_clean)
        merged = merge_sentiment_and_trades(th_clean, fg_clean, _preprocessing_config())

        config = _feature_config()
        feature_functions = [
            add_sentiment_lag,
            add_sentiment_regime_encoding,
            add_sentiment_rolling_mean,
            add_sentiment_fear_greed_flags,
            add_trader_win_rate,
            add_trader_pnl_rolling,
            add_trader_leverage_avg,
            add_trader_pnl_volatility,
            add_trader_trade_count,
            add_trader_avg_size,
            flag_cold_start_rows,
            add_time_of_day,
            add_day_of_week,
            add_month,
        ]
        features = reduce(lambda d, fn: fn(d, config), feature_functions, merged)

        # Normality check
        norm_result = check_normality(features["Closed PnL"].dropna())
        assert norm_result.statistic > 0
        assert norm_result.method in ("shapiro", "dagostino_pearson")

        # Two-group comparison
        fear_pnl = features.loc[features["sentiment_is_fear"], "Closed PnL"].dropna()
        greed_pnl = features.loc[features["sentiment_is_greed"], "Closed PnL"].dropna()

        if len(fear_pnl) >= 3 and len(greed_pnl) >= 3:
            two_group = compare_two_groups(fear_pnl, greed_pnl)
            assert two_group.test_name is not None
            assert two_group.statistic is not None
            assert two_group.p_value is not None
            assert two_group.effect_size is not None

        # Chi-square test
        if len(features) > 0:
            sentiment_cls = (
                features["sentiment_classification"].cat.add_categories("Unknown").fillna("Unknown")
            )
            direction = features["Direction"].cat.add_categories("Unknown").fillna("Unknown")
            contingency = pd.crosstab(sentiment_cls, direction)
            cs_result = chi_square_test(contingency)
            assert cs_result.test_name == "chi_square"
            assert cs_result.statistic >= 0


# ── Test: No Data Leakage ─────────────────────────────────────────────────


class TestFullPipelineNoDataLeakage:
    """Verify no future data leaks into training features."""

    def _make_ordered_sentiment(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "value": np.arange(10, 10 + 10, dtype=np.int64),
                "classification": pd.Categorical(
                    ["Neutral"] * 10,
                    categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                    ordered=True,
                ),
            }
        )

    def _make_ordered_trades(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "Trade ID": [f"TID_{i:04d}" for i in range(10)],
                "Account": ["acc_1"] * 10,
                "Timestamp": timestamps,
                "Side": pd.Categorical(
                    rng.choice(["Long", "Short"], size=10),
                    categories=["Long", "Short"],
                ),
                "Direction": pd.Categorical(
                    rng.choice(["Open", "Close"], size=10),
                    categories=["Open", "Close"],
                ),
                "Size USD": rng.uniform(100, 10000, size=10).astype(np.float64),
                "Execution Price": rng.uniform(1000, 100000, size=10).astype(np.float64),
                "Closed PnL": rng.uniform(-500, 500, size=10).astype(np.float64),
                "Fee": rng.uniform(0, 50, size=10).astype(np.float64),
            }
        )
        config = PreprocessingConfig()
        df = clean_trader_history(df, config)
        df = extract_trade_date(df)
        return df

    def test_full_pipeline_no_data_leakage(self) -> None:
        sentiment = self._make_ordered_sentiment()
        trades = self._make_ordered_trades()

        merged = merge_sentiment_and_trades(trades, sentiment, _preprocessing_config())

        config = _feature_config()
        features = add_sentiment_lag(merged, config)
        features = add_sentiment_rolling_mean(features, config)

        features = features.sort_values("Timestamp")

        # Lag(1) feature at row i must equal sentiment_value at row i-1
        for i in range(1, len(features)):
            if pd.notna(features["sentiment_value"].iloc[i - 1]) and pd.notna(
                features["sentiment_value_lag_1d"].iloc[i]
            ):
                assert (
                    features["sentiment_value_lag_1d"].iloc[i]
                    == features["sentiment_value"].iloc[i - 1]
                ), f"Leakage at row {i}: lag value does not match prior day's value"

        # Rolling mean with closed="left" should not include current row
        for i in range(2, len(features)):
            if pd.notna(features["sentiment_value_rolling_7d"].iloc[i]):
                current_val = features["sentiment_value"].iloc[i]
                rolling_mean = features["sentiment_value_rolling_7d"].iloc[i]
                assert (
                    rolling_mean != current_val or np.isnan(rolling_mean) or np.isnan(current_val)
                ), f"Leakage row {i}: rolling mean==current"
