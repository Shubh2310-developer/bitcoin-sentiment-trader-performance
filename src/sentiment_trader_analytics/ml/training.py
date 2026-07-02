"""Training orchestration for baseline ML models.

Provides pure, reproducible functions for preparing features,
splitting data chronologically, constructing baseline and candidate
model pipelines, running time-series cross-validation, and training.

Supported candidate models:
    - Random Forest (rf) — ensemble of decision trees.
    - XGBoost (xgb) — gradient boosting with regularisation.
    - LightGBM (lgbm) — gradient boosting optimised for speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentiment_trader_analytics.config import MLConfig

FEATURE_COLUMNS: list[str] = [
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
    "time_hour_utc",
    "time_day_of_week",
    "time_is_weekend",
    "time_month",
]

REGRESSION_TARGET = "Closed PnL"
CLASSIFICATION_TARGET = "profitable_trade"


@dataclass
class TrainTestSplit:
    """Container for chronologically split train/test data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_reg: pd.Series
    y_test_reg: pd.Series
    y_train_cls: pd.Series
    y_test_cls: pd.Series
    train_indices: pd.Index
    test_indices: pd.Index


@dataclass
class CVResult:
    """Container for cross-validation results."""

    reg_scores: list[float] = field(default_factory=list)
    cls_scores: list[float] = field(default_factory=list)


def prepare_features(
    df: pd.DataFrame, config: MLConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract feature matrix X and target DataFrames, excluding cold-start rows.

    The returned X has its index set to the sorted Timestamp values so that
    ``build_train_test_split`` can perform chronological comparison.

    Args:
        df: Full feature-store DataFrame.
        config: ML configuration.

    Returns:
        Tuple of (X, y_regression_targets, y_classification_targets) where
        cold-start rows are excluded and rows are sorted by Timestamp.
    """
    np.random.seed(config.random_seed)

    working = df[~df["feature_cold_start"]].copy()
    working = working.sort_values("Timestamp")

    x = working[FEATURE_COLUMNS].copy()
    x.index = working["Timestamp"]
    y_reg = working[REGRESSION_TARGET].copy()
    y_cls = (working[REGRESSION_TARGET] > 0).astype(int)

    return x, y_reg, y_cls


def build_train_test_split(
    x: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    config: MLConfig,
) -> TrainTestSplit:
    """Perform a chronological train/test split at config.train_cutoff_date.

    Args:
        x: Feature matrix with Timestamp index (must be sorted before calling).
        y_reg: Regression target (Closed PnL).
        y_cls: Classification target (profitable indicator).
        config: ML configuration containing the cutoff date.

    Returns:
        TrainTestSplit with chronologically separated data.

    Raises:
        ValueError: If the cutoff date produces empty train or test sets.
    """
    cutoff = pd.Timestamp(config.train_cutoff_date, tz="UTC")

    train_mask = x.index < cutoff
    test_mask = x.index >= cutoff

    if train_mask.sum() == 0:
        raise ValueError(
            f"Train set is empty after cutoff {config.train_cutoff_date}. "
            "No data before this date."
        )
    if test_mask.sum() == 0:
        raise ValueError(
            f"Test set is empty after cutoff {config.train_cutoff_date}. "
            "No data on/after this date."
        )

    return TrainTestSplit(
        X_train=x.loc[train_mask],
        X_test=x.loc[test_mask],
        y_train_reg=y_reg.loc[train_mask],
        y_test_reg=y_reg.loc[test_mask],
        y_train_cls=y_cls.loc[train_mask],
        y_test_cls=y_cls.loc[test_mask],
        train_indices=x.index[train_mask],
        test_indices=x.index[test_mask],
    )


def build_baseline_model(task: str, config: MLConfig) -> Pipeline:
    """Build a scikit-learn Pipeline wrapping a DummyRegressor or DummyClassifier.

    Args:
        task: 'regression' or 'classification'.
        config: ML configuration (for random_seed).

    Returns:
        A Pipeline with imputer + dummy estimator.

    Raises:
        ValueError: If task is not 'regression' or 'classification'.
    """
    np.random.seed(config.random_seed)

    imputer = SimpleImputer(strategy="median")

    if task == "regression":
        estimator: Any = DummyRegressor(strategy="mean")
    elif task == "classification":
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")

    return Pipeline([("imputer", imputer), ("estimator", estimator)])


def build_model_pipeline(task: str, config: MLConfig) -> Pipeline:
    """Build a Random Forest scikit-learn Pipeline for the given task.

    Pipeline: SimpleImputer(median) -> StandardScaler -> RandomForest

    Args:
        task: 'regression' or 'classification'.
        config: ML configuration (for random_seed).

    Returns:
        A configured Pipeline ready for fitting.

    Raises:
        ValueError: If task is not 'regression' or 'classification'.
    """
    np.random.seed(config.random_seed)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if task == "regression":
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=config.random_seed,
            n_jobs=-1,
        )
    elif task == "classification":
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")

    return Pipeline([("imputer", imputer), ("scaler", scaler), ("estimator", estimator)])


def build_xgboost_pipeline(task: str, config: MLConfig) -> Pipeline:
    """Build an XGBoost Pipeline for the given task.

    Pipeline: SimpleImputer(median) -> XGBoostRegressor/Classifier

    XGBoost handles missing values natively; the imputer is included
    as a safety net for downstream compatibility.

    Args:
        task: 'regression' or 'classification'.
        config: ML configuration (for random_seed).

    Returns:
        A configured Pipeline ready for fitting.

    Raises:
        ValueError: If task is not 'regression' or 'classification'.
        ImportError: If xgboost is not installed.
    """
    import xgboost as xgb  # noqa: PLC0415

    np.random.seed(config.random_seed)
    imputer = SimpleImputer(strategy="median")

    if task == "regression":
        estimator: Any = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.random_seed,
            n_jobs=-1,
            verbosity=0,
        )
    elif task == "classification":
        estimator = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,
            eval_metric="logloss",
            random_state=config.random_seed,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")

    return Pipeline([("imputer", imputer), ("estimator", estimator)])


def build_lightgbm_pipeline(task: str, config: MLConfig) -> Pipeline:
    """Build a LightGBM Pipeline for the given task.

    Pipeline: SimpleImputer(median) -> LGBMRegressor/Classifier

    Args:
        task: 'regression' or 'classification'.
        config: ML configuration (for random_seed).

    Returns:
        A configured Pipeline ready for fitting.

    Raises:
        ValueError: If task is not 'regression' or 'classification'.
        ImportError: If lightgbm is not installed.
    """
    import lightgbm as lgb  # noqa: PLC0415

    np.random.seed(config.random_seed)
    imputer = SimpleImputer(strategy="median")

    if task == "regression":
        estimator: Any = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    elif task == "classification":
        estimator = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")

    return Pipeline([("imputer", imputer), ("estimator", estimator)])


def run_cross_validation(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: MLConfig,
) -> list[float]:
    """Run TimeSeriesSplit cross-validation on the training set.

    Args:
        pipeline: Unfitted scikit-learn Pipeline.
        x_train: Training feature matrix.
        y_train: Training target series.
        config: ML configuration (cv_folds, random_seed).

    Returns:
        List of R² (regression) or ROC-AUC (classification) scores per fold.
    """
    np.random.seed(config.random_seed)

    tscv = TimeSeriesSplit(n_splits=config.cv_folds)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(x_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipeline = _clone_pipeline(pipeline)
        fold_pipeline.fit(x_tr, y_tr)
        score = fold_pipeline.score(x_val, y_val)
        scores.append(score)

    return scores


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    """Clone a Pipeline by reconstructing it with the same parameters.

    Args:
        pipeline: Pipeline to clone.

    Returns:
        A new unfitted Pipeline with identical parameters.
    """
    from sklearn import clone as sklearn_clone

    return sklearn_clone(pipeline)


def train_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: MLConfig,
) -> Pipeline:
    """Fit a pipeline on the training data with the global random seed set.

    Args:
        pipeline: Unfitted scikit-learn Pipeline.
        x_train: Training feature matrix.
        y_train: Training target.
        config: ML configuration (for random_seed).

    Returns:
        Fitted Pipeline.
    """
    np.random.seed(config.random_seed)
    pipeline.fit(x_train, y_train)
    return pipeline
