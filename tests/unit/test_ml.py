"""Unit tests for the ML module (Phase 08).

Tests chronological split, baseline model training, time-series CV,
metrics computation, and reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import TimeSeriesSplit

from sentiment_trader_analytics.config import MLConfig
from sentiment_trader_analytics.ml.evaluation import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_feature_importance,
    evaluate_classification,
    evaluate_regression,
)
from sentiment_trader_analytics.ml.training import (
    FEATURE_COLUMNS,
    build_baseline_model,
    build_model_pipeline,
    build_train_test_split,
    prepare_features,
    run_cross_validation,
    train_model,
)


@pytest.fixture
def ml_config() -> MLConfig:
    """Return a default MLConfig for testing."""
    return MLConfig(
        random_seed=42,
        train_cutoff_date="2025-04-01",
        cv_folds=3,
        tracking_uri=Path("experiments/mlruns"),
        experiment_name="test_experiments",
    )


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create a synthetic feature store DataFrame for testing."""
    np.random.seed(42)
    n = 100

    # Generate sequential timestamps for chronological ordering
    timestamps = pd.date_range(start="2025-01-01", periods=n, freq="D", tz="UTC")

    data = {
        "Timestamp": timestamps,
        "Closed PnL": np.random.randn(n) * 100,
        "feature_cold_start": [False] * n,
    }

    for col in FEATURE_COLUMNS:
        data[col] = np.random.randn(n)

    df = pd.DataFrame(data)

    # Add some nulls to test imputer
    df.loc[0:5, "sentiment_value_lag_1d"] = np.nan
    df.loc[2:4, "sentiment_value_rolling_7d"] = np.nan

    return df


class TestChronologicalSplit:
    """Tests for the chronological train/test split."""

    def test_train_dates_before_cutoff(
        self, sample_feature_df: pd.DataFrame, ml_config: MLConfig
    ) -> None:
        """Train set timestamps must be strictly before the cutoff date."""
        x, y_reg, y_cls = prepare_features(sample_feature_df, ml_config)

        split = build_train_test_split(x, y_reg, y_cls, ml_config)
        cutoff = pd.Timestamp(ml_config.train_cutoff_date, tz="UTC")

        assert (split.train_indices < cutoff).all(), "Train has rows on/after cutoff"

    def test_test_dates_after_cutoff(
        self, sample_feature_df: pd.DataFrame, ml_config: MLConfig
    ) -> None:
        """Test set timestamps must be on or after the cutoff date."""
        x, y_reg, y_cls = prepare_features(sample_feature_df, ml_config)

        split = build_train_test_split(x, y_reg, y_cls, ml_config)
        cutoff = pd.Timestamp(ml_config.train_cutoff_date, tz="UTC")

        assert (split.test_indices >= cutoff).all(), "Test has rows before cutoff"

    def test_empty_train_raises(self, ml_config: MLConfig) -> None:
        """Splitting with all data after cutoff should raise ValueError."""
        df = pd.DataFrame(
            {
                "Timestamp": pd.date_range(start="2025-06-01", periods=10, freq="D", tz="UTC"),
                "Closed PnL": np.zeros(10),
                "feature_cold_start": [False] * 10,
            }
        )
        for col in FEATURE_COLUMNS:
            df[col] = np.zeros(10)

        x, y_reg, y_cls = prepare_features(df, ml_config)
        with pytest.raises(ValueError, match="Train set is empty"):
            build_train_test_split(x, y_reg, y_cls, ml_config)

    def test_empty_test_raises(self, ml_config: MLConfig) -> None:
        """Splitting with all data before cutoff should raise ValueError."""
        df = pd.DataFrame(
            {
                "Timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D", tz="UTC"),
                "Closed PnL": np.zeros(10),
                "feature_cold_start": [False] * 10,
            }
        )
        for col in FEATURE_COLUMNS:
            df[col] = np.zeros(10)

        x, y_reg, y_cls = prepare_features(df, ml_config)
        with pytest.raises(ValueError, match="Test set is empty"):
            build_train_test_split(x, y_reg, y_cls, ml_config)


class TestBaselineModelTraining:
    """Tests for baseline model training."""

    def test_dummy_regressor_fits(self, ml_config: MLConfig) -> None:
        """DummyRegressor with 'mean' strategy should fit and predict."""
        x = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        y = pd.Series([10.0, 20.0, 30.0])

        pipeline = build_baseline_model("regression", ml_config)
        fitted = train_model(pipeline, x, y, ml_config)
        preds = fitted.predict(x)

        assert isinstance(fitted.named_steps["estimator"], DummyRegressor)
        assert np.allclose(preds, 20.0)  # mean of [10, 20, 30]

    def test_dummy_classifier_fits(self, ml_config: MLConfig) -> None:
        """DummyClassifier with 'most_frequent' should predict majority class."""
        x = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 5.0, 6.0, 7.0]})
        y = pd.Series([0, 0, 1, 1])

        pipeline = build_baseline_model("classification", ml_config)
        fitted = train_model(pipeline, x, y, ml_config)
        preds = fitted.predict(x)

        assert isinstance(fitted.named_steps["estimator"], DummyClassifier)
        assert all(preds == 0)  # majority is 0 (tied, first wins)

    def test_invalid_task_raises(self, ml_config: MLConfig) -> None:
        """Unknown task should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            build_baseline_model("unknown", ml_config)


class TestTimeSeriesCV:
    """Tests for time-series cross-validation."""

    def test_correct_number_of_folds(self, ml_config: MLConfig) -> None:
        """TimeSeriesSplit should produce exactly config.cv_folds folds."""
        x = pd.DataFrame({"f1": np.random.randn(100)})

        tscv = TimeSeriesSplit(n_splits=ml_config.cv_folds)
        splits = list(tscv.split(x))
        assert len(splits) == ml_config.cv_folds

    def test_folds_are_chronological(self, ml_config: MLConfig) -> None:
        """Each fold should have train indices before val indices."""
        x = pd.DataFrame({"f1": np.random.randn(100)})

        tscv = TimeSeriesSplit(n_splits=ml_config.cv_folds)
        for train_idx, val_idx in tscv.split(x):
            assert max(train_idx) < min(val_idx)

    def test_run_cross_validation_returns_scores(
        self, sample_feature_df: pd.DataFrame, ml_config: MLConfig
    ) -> None:
        """run_cross_validation should return one score per fold."""
        x, y_reg, _ = prepare_features(sample_feature_df, ml_config)

        # Use a smaller config for speed
        fast_config = MLConfig(
            random_seed=42,
            train_cutoff_date="2025-04-01",
            cv_folds=2,
            tracking_uri=Path("experiments/mlruns"),
            experiment_name="test",
        )
        split = build_train_test_split(x, y_reg, y_reg, fast_config)

        pipeline = build_model_pipeline("regression", fast_config)
        scores = run_cross_validation(pipeline, split.X_train, split.y_train_reg, fast_config)

        assert len(scores) == fast_config.cv_folds
        assert all(isinstance(s, float) for s in scores)


class TestMetricsComputation:
    """Tests for evaluation metrics computation."""

    def test_regression_metrics_correct(self) -> None:
        """Regression metrics should compute correctly with known values."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])
        baseline_pred = np.array([25.0, 25.0, 25.0, 25.0])

        metrics = evaluate_regression(y_true, y_pred, baseline_pred)

        assert isinstance(metrics, RegressionMetrics)
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert isinstance(metrics.r2, float)
        assert metrics.baseline_rmse > 0

    def test_classification_metrics_correct(self) -> None:
        """Classification metrics should compute correctly with known values."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        baseline_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        metrics = evaluate_classification(y_true, y_pred, y_proba, baseline_pred)

        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.roc_auc > 0.5

    def test_classification_perfect_predictions(self) -> None:
        """Perfect predictions should yield accuracy=1.0 and ROC-AUC=1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        baseline_pred = np.array([0, 0, 0, 0])

        metrics = evaluate_classification(y_true, y_pred, y_proba, baseline_pred)
        assert metrics.accuracy == 1.0
        assert metrics.roc_auc == 1.0

    def test_regression_lift(self) -> None:
        """Lift should be positive when model beats baseline."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        baseline_pred = np.array([20.0, 20.0, 20.0])

        metrics = evaluate_regression(y_true, y_pred, baseline_pred)
        assert metrics.lift_rmse > 0  # model RMSE < baseline RMSE => positive lift

    def test_feature_importance_shape(self, ml_config: MLConfig) -> None:
        """Feature importance should return one entry per feature."""
        x = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
        y = pd.Series([0, 0, 1, 1])

        pipeline = build_model_pipeline("classification", ml_config)
        pipeline.fit(x, y)

        report = compute_feature_importance(pipeline, x, y, ["f1", "f2"], 42)
        assert len(report.feature_names) == 2
        assert len(report.native_importance) == 2
        assert len(report.permutation_importance_mean) == 2


class TestReproducibilitySeed:
    """Tests for reproducibility with fixed random seed."""

    def test_identical_regression_runs(self, sample_feature_df: pd.DataFrame) -> None:
        """Two training runs with the same seed should produce identical predictions."""
        config = MLConfig(random_seed=42, train_cutoff_date="2025-04-01", cv_folds=3)
        x, y_reg, _ = prepare_features(sample_feature_df, config)
        split = build_train_test_split(x, y_reg, y_reg, config)

        pipeline1 = build_model_pipeline("regression", config)
        fitted1 = train_model(pipeline1, split.X_train, split.y_train_reg, config)
        preds1 = fitted1.predict(split.X_test)

        pipeline2 = build_model_pipeline("regression", config)
        fitted2 = train_model(pipeline2, split.X_train, split.y_train_reg, config)
        preds2 = fitted2.predict(split.X_test)

        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_different_seed_different_predictions(self, sample_feature_df: pd.DataFrame) -> None:
        """Different seeds should (generally) produce different predictions."""
        config1 = MLConfig(random_seed=42, train_cutoff_date="2025-04-01", cv_folds=3)
        config2 = MLConfig(random_seed=99, train_cutoff_date="2025-04-01", cv_folds=3)

        x, y_reg, _ = prepare_features(sample_feature_df, config1)
        split = build_train_test_split(x, y_reg, y_reg, config1)

        pipeline1 = build_model_pipeline("regression", config1)
        fitted1 = train_model(pipeline1, split.X_train, split.y_train_reg, config1)
        preds1 = fitted1.predict(split.X_test)

        pipeline2 = build_model_pipeline("regression", config2)
        fitted2 = train_model(pipeline2, split.X_train, split.y_train_reg, config2)
        preds2 = fitted2.predict(split.X_test)

        assert not np.allclose(preds1, preds2)

    def test_prepare_features_reproducible(self) -> None:
        """prepare_features with the same seed and data should produce same output."""
        config = MLConfig(random_seed=42, train_cutoff_date="2025-04-01", cv_folds=3)
        df = pd.DataFrame(
            {
                "Timestamp": pd.date_range(start="2024-01-01", periods=20, freq="D", tz="UTC"),
                "Closed PnL": np.random.RandomState(42).randn(20) * 100,
                "feature_cold_start": [False] * 20,
            }
        )
        for col in FEATURE_COLUMNS:
            df[col] = np.random.RandomState(42).randn(20)

        x1, y_reg1, y_cls1 = prepare_features(df, config)
        x2, y_reg2, y_cls2 = prepare_features(df, config)

        pd.testing.assert_frame_equal(x1, x2)
        pd.testing.assert_series_equal(y_reg1, y_reg2)
        pd.testing.assert_series_equal(y_cls1, y_cls2)
