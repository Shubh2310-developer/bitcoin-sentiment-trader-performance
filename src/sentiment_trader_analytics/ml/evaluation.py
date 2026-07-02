"""Model evaluation metrics and feature importance computation.

Provides regression and classification evaluation functions with
baseline comparison, and both model-native and permutation importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics."""

    rmse: float
    mae: float
    r2: float
    baseline_rmse: float
    baseline_mae: float
    baseline_r2: float
    lift_rmse: float
    lift_mae: float
    lift_r2: float


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    baseline_accuracy: float
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float
    lift_accuracy: float
    lift_precision: float
    lift_recall: float
    lift_f1: float


@dataclass
class FeatureImportanceReport:
    """Container for feature importance results."""

    feature_names: list[str]
    native_importance: list[float]
    permutation_importance_mean: list[float]
    permutation_importance_std: list[float]


def evaluate_regression(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    baseline_pred: np.ndarray[Any, Any],
) -> RegressionMetrics:
    """Compute regression metrics and lift above baseline.

    Args:
        y_true: Ground truth target values.
        y_pred: Candidate model predictions.
        baseline_pred: Baseline (dummy) model predictions.

    Returns:
        RegressionMetrics with RMSE, MAE, R² for both model and baseline.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
    baseline_r2 = float(r2_score(y_true, baseline_pred))

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        baseline_rmse=baseline_rmse,
        baseline_mae=baseline_mae,
        baseline_r2=baseline_r2,
        lift_rmse=(baseline_rmse - rmse) / baseline_rmse if baseline_rmse != 0 else 0.0,
        lift_mae=(baseline_mae - mae) / baseline_mae if baseline_mae != 0 else 0.0,
        lift_r2=r2 - baseline_r2,
    )


def evaluate_classification(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    y_proba: np.ndarray[Any, Any] | None,
    baseline_pred: np.ndarray[Any, Any],
) -> ClassificationMetrics:
    """Compute classification metrics and lift above baseline.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Candidate model class predictions.
        y_proba: Candidate model probability predictions (for ROC-AUC).
        baseline_pred: Baseline (dummy) model predictions.

    Returns:
        ClassificationMetrics with accuracy, precision, recall, F1, ROC-AUC.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc_auc_val = float(roc_auc_score(y_true, y_proba))
    else:
        roc_auc_val = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    baseline_precision = float(precision_score(y_true, baseline_pred, zero_division=0))
    baseline_recall = float(recall_score(y_true, baseline_pred, zero_division=0))
    baseline_f1 = float(f1_score(y_true, baseline_pred, zero_division=0))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc_val,
        confusion_matrix=cm,
        baseline_accuracy=baseline_accuracy,
        baseline_precision=baseline_precision,
        baseline_recall=baseline_recall,
        baseline_f1=baseline_f1,
        lift_accuracy=accuracy - baseline_accuracy,
        lift_precision=precision - baseline_precision,
        lift_recall=recall - baseline_recall,
        lift_f1=f1 - baseline_f1,
    )


def compute_feature_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    random_seed: int = 42,
) -> FeatureImportanceReport:
    """Compute both model-native (impurity-based) and permutation importance.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        x_test: Test feature matrix for permutation importance.
        y_test: Test target for permutation importance.
        feature_names: List of feature column names.
        random_seed: Random seed for permutation importance.

    Returns:
        FeatureImportanceReport with both importance types.
    """
    estimator = pipeline.named_steps["estimator"]

    if hasattr(estimator, "feature_importances_"):
        native = list(estimator.feature_importances_)
    else:
        native = [0.0] * len(feature_names)

    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=10,
        random_state=random_seed,
        n_jobs=-1,
    )

    return FeatureImportanceReport(
        feature_names=feature_names,
        native_importance=native,
        permutation_importance_mean=list(result.importances_mean),
        permutation_importance_std=list(result.importances_std),
    )
