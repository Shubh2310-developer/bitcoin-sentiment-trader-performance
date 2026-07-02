"""Unit tests for the configuration module.

Tests cover config model defaults, constrained fields, and the
load_config function.
"""

from pathlib import Path

import pytest
import yaml

from sentiment_trader_analytics.config import (
    AppConfig,
    EDAConfig,
    FeatureConfig,
    IngestionConfig,
    MLConfig,
    PreprocessingConfig,
    StatConfig,
    ValidationConfig,
    load_config,
)


class TestConfigDefaults:
    """Tests for config model default values."""

    def test_ingestion_config_defaults(self) -> None:
        cfg = IngestionConfig()
        assert cfg.chunk_size == 50000
        assert cfg.fear_greed_path == Path("data/raw/fear_greed/fear_greed_index.csv")
        assert cfg.trader_history_path == Path("data/raw/trader_history/historical_data.csv")

    def test_validation_config_defaults(self) -> None:
        cfg = ValidationConfig()
        assert cfg.min_temporal_coverage_days == 365
        assert cfg.min_distinct_accounts == 3
        assert cfg.expected_value_range == (0, 100)

    def test_preprocessing_config_defaults(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.interim_output_format == "parquet"
        assert cfg.fan_out_tolerance_fraction == 0.0
        assert cfg.null_strategy_fear_greed == "drop"
        assert cfg.null_strategy_trader == "drop"

    def test_feature_config_defaults(self) -> None:
        cfg = FeatureConfig()
        assert cfg.sentiment_lag_days == 1
        assert cfg.sentiment_rolling_window == 7
        assert len(cfg.trader_rolling_windows) == 6

    def test_eda_config_defaults(self) -> None:
        cfg = EDAConfig()
        assert cfg.outlier_method == "iqr"
        assert cfg.correlation_method == "spearman"
        assert len(cfg.numeric_features) == 10

    def test_stat_config_defaults(self) -> None:
        cfg = StatConfig()
        assert cfg.alpha == 0.05
        assert cfg.correction_method == "bonferroni"
        assert cfg.min_sample_size == 30

    def test_ml_config_defaults(self) -> None:
        cfg = MLConfig()
        assert cfg.random_seed == 42
        assert cfg.train_cutoff_date == "2025-04-01"
        assert cfg.cv_folds == 5

    def test_app_config_defaults(self) -> None:
        cfg = AppConfig()
        assert isinstance(cfg.ingestion, IngestionConfig)
        assert isinstance(cfg.validation, ValidationConfig)
        assert isinstance(cfg.preprocessing, PreprocessingConfig)
        assert isinstance(cfg.feature_engineering, FeatureConfig)
        assert isinstance(cfg.eda, EDAConfig)
        assert isinstance(cfg.statistics, StatConfig)
        assert isinstance(cfg.ml, MLConfig)


class TestLoadConfig:
    """Tests for :func:`load_config`."""

    def test_load_config_success(self, tmp_path: Path) -> None:
        config_path = tmp_path / "test_config.yaml"
        raw = {
            "ingestion": {"chunk_size": 1000},
            "validation": {"min_distinct_accounts": 5},
        }
        with open(config_path, "w") as f:
            yaml.dump(raw, f)

        cfg = load_config(config_path)
        assert cfg.ingestion.chunk_size == 1000
        assert cfg.validation.min_distinct_accounts == 5

    def test_load_config_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_config(Path("nonexistent.yaml"))

    def test_load_config_overrides(self, tmp_path: Path) -> None:
        config_path = tmp_path / "override.yaml"
        raw = {
            "feature_engineering": {"sentiment_lag_days": 3, "sentiment_rolling_window": 14},
            "statistics": {"alpha": 0.01, "correction_method": "fdr_bh"},
        }
        with open(config_path, "w") as f:
            yaml.dump(raw, f)

        cfg = load_config(config_path)
        assert cfg.feature_engineering.sentiment_lag_days == 3
        assert cfg.feature_engineering.sentiment_rolling_window == 14
        assert cfg.statistics.alpha == 0.01
        assert cfg.statistics.correction_method == "fdr_bh"
