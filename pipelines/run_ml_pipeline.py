#!/usr/bin/env python3
# ruff: noqa: E402
"""Machine learning pipeline entry point (Phase 08).

Trains and evaluates baseline (dummy) and candidate (Random Forest)
models for both regression (PnL prediction) and classification
(profitable vs. unprofitable) tasks using a chronological split.
Logs all metrics, artifacts, and models to MLflow.

Usage:
    python pipelines/run_ml_pipeline.py --config configs/base.yaml
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sentiment_trader_analytics.config import MLConfig, load_config
from sentiment_trader_analytics.ml.evaluation import (
    ClassificationMetrics,
    FeatureImportanceReport,
    RegressionMetrics,
    compute_feature_importance,
    evaluate_classification,
    evaluate_regression,
)
from sentiment_trader_analytics.ml.training import (
    FEATURE_COLUMNS,
    build_baseline_model,
    build_lightgbm_pipeline,
    build_model_pipeline,
    build_xgboost_pipeline,
    prepare_features,
    run_cross_validation,
    train_model,
)
from sentiment_trader_analytics.utils.logging_utils import setup_logging

logger = setup_logging("ml_pipeline", log_file="logs/pipeline.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the ML pipeline (Phase 08) — train and evaluate baseline models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def _find_latest_feature_store(features_dir: Path) -> Path:
    """Find the most recent feature store parquet file.

    Args:
        features_dir: Directory containing feature_store_*.parquet files.

    Returns:
        Path to the latest parquet file.

    Raises:
        FileNotFoundError: If no feature store files are found.
    """
    parquet_files = sorted(features_dir.glob("feature_store_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No feature store files found in {features_dir}. "
            "Run the feature engineering pipeline first."
        )
    return parquet_files[-1]


def _run_name(task: str, model_type: str) -> str:
    """Generate MLflow run name following naming convention.

    Args:
        task: 'regression' or 'classification'.
        model_type: 'baseline' or 'rf'.

    Returns:
        Formatted run name, e.g. 'ml_regression_baseline_20260701'.
    """
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    return f"ml_{task}_{model_type}_{date_str}"


def _compute_shap_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Compute SHAP values for a fitted pipeline and save a summary plot.

    Falls back to ``TreeExplainer`` for tree-based models and
    ``Explainer`` for others.  Saves both a PNG bar plot and a JSON
    file containing mean absolute SHAP values per feature.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        x_test: Test feature matrix used for SHAP computation.
        feature_names: List of feature column names.
        output_path: Base path (without extension) for output files.
    """
    try:
        import shap  # noqa: PLC0415

        estimator = pipeline.named_steps["estimator"]
        if "imputer" in pipeline.named_steps:
            x_transformed = pipeline.named_steps["imputer"].transform(x_test)
            if "scaler" in pipeline.named_steps:
                x_transformed = pipeline.named_steps["scaler"].transform(x_transformed)
        else:
            x_transformed = x_test.values

        x_df = pd.DataFrame(x_transformed, columns=feature_names)

        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(x_df)
        except Exception:  # noqa: BLE001
            explainer_fallback = shap.Explainer(estimator, x_df)
            shap_values = explainer_fallback(x_df).values

        # For classifiers, shap_values may be list[array]; use class-1 slice
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_dict = dict(zip(feature_names, mean_abs_shap.tolist(), strict=False))

        import json  # noqa: PLC0415

        import matplotlib.pyplot as plt  # noqa: PLC0415

        json_path = Path(str(output_path) + "_shap.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(shap_dict, f, indent=2)

        sorted_idx = np.argsort(mean_abs_shap)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in sorted_idx],
            [mean_abs_shap[i] for i in sorted_idx],
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance")
        plt.tight_layout()
        png_path = Path(str(output_path) + "_shap.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP importance saved to %s", output_path)
    except Exception:  # noqa: BLE001
        logger.warning("SHAP computation failed; skipping.", exc_info=True)


def _make_importance_chart(report: FeatureImportanceReport, output_path: Path) -> None:
    """Generate and save a feature importance bar chart.

    Args:
        report: FeatureImportanceReport with native + permutation importance.
        output_path: Path to save the PNG chart.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    idx = np.argsort(report.native_importance)
    ax1.barh([report.feature_names[i] for i in idx], [report.native_importance[i] for i in idx])
    ax1.set_title("Native (Impurity-Based) Importance")
    ax1.set_xlabel("Importance")

    idx2 = np.argsort(report.permutation_importance_mean)
    ax2.barh(
        [report.feature_names[i] for i in idx2],
        [report.permutation_importance_mean[i] for i in idx2],
        xerr=[report.permutation_importance_std[i] for i in idx2],
    )
    ax2.set_title("Permutation Importance")
    ax2.set_xlabel("Mean Importance (10 repeats)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance chart saved to %s", output_path)


def _log_regression_metrics(_run_name: str, metrics: RegressionMetrics) -> None:
    """Log regression metrics to MLflow.

    Args:
        run_name: Name of the MLflow run.
        metrics: RegressionMetrics to log.
    """
    mlflow.log_metric("rmse", metrics.rmse)
    mlflow.log_metric("mae", metrics.mae)
    mlflow.log_metric("r2", metrics.r2)
    mlflow.log_metric("baseline_rmse", metrics.baseline_rmse)
    mlflow.log_metric("baseline_mae", metrics.baseline_mae)
    mlflow.log_metric("baseline_r2", metrics.baseline_r2)
    mlflow.log_metric("lift_rmse", metrics.lift_rmse)
    mlflow.log_metric("lift_mae", metrics.lift_mae)
    mlflow.log_metric("lift_r2", metrics.lift_r2)


def _log_classification_metrics(_run_name: str, metrics: ClassificationMetrics) -> None:
    """Log classification metrics to MLflow.

    Args:
        run_name: Name of the MLflow run.
        metrics: ClassificationMetrics to log.
    """
    mlflow.log_metric("accuracy", metrics.accuracy)
    mlflow.log_metric("precision", metrics.precision)
    mlflow.log_metric("recall", metrics.recall)
    mlflow.log_metric("f1", metrics.f1)
    mlflow.log_metric("roc_auc", metrics.roc_auc)
    mlflow.log_metric("baseline_accuracy", metrics.baseline_accuracy)
    mlflow.log_metric("baseline_precision", metrics.baseline_precision)
    mlflow.log_metric("baseline_recall", metrics.baseline_recall)
    mlflow.log_metric("baseline_f1", metrics.baseline_f1)
    mlflow.log_metric("lift_accuracy", metrics.lift_accuracy)
    mlflow.log_metric("lift_precision", metrics.lift_precision)
    mlflow.log_metric("lift_recall", metrics.lift_recall)
    mlflow.log_metric("lift_f1", metrics.lift_f1)

    cm = metrics.confusion_matrix
    if cm:
        mlflow.log_text(json.dumps(cm), "confusion_matrix.json")


def _save_evaluation_summary(
    regression_metrics: RegressionMetrics,
    classification_metrics: ClassificationMetrics,
    output_path: Path,
    config: MLConfig,
) -> None:
    """Write the combined evaluation summary CSV.

    Args:
        regression_metrics: Metrics from the regression candidate model.
        classification_metrics: Metrics from the classification candidate model.
        output_path: Path for the output CSV.
        config: ML configuration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "task": "regression",
            "model": "rf",
            "random_seed": config.random_seed,
            "cutoff_date": config.train_cutoff_date,
            "rmse": regression_metrics.rmse,
            "mae": regression_metrics.mae,
            "r2": regression_metrics.r2,
            "baseline_rmse": regression_metrics.baseline_rmse,
            "baseline_mae": regression_metrics.baseline_mae,
            "baseline_r2": regression_metrics.baseline_r2,
            "lift_rmse": regression_metrics.lift_rmse,
            "lift_mae": regression_metrics.lift_mae,
            "lift_r2": regression_metrics.lift_r2,
        },
        {
            "task": "classification",
            "model": "rf",
            "random_seed": config.random_seed,
            "cutoff_date": config.train_cutoff_date,
            "accuracy": classification_metrics.accuracy,
            "precision": classification_metrics.precision,
            "recall": classification_metrics.recall,
            "f1": classification_metrics.f1,
            "roc_auc": classification_metrics.roc_auc,
            "baseline_accuracy": classification_metrics.baseline_accuracy,
            "baseline_precision": classification_metrics.baseline_precision,
            "baseline_recall": classification_metrics.baseline_recall,
            "baseline_f1": classification_metrics.baseline_f1,
            "lift_accuracy": classification_metrics.lift_accuracy,
            "lift_precision": classification_metrics.lift_precision,
            "lift_recall": classification_metrics.lift_recall,
            "lift_f1": classification_metrics.lift_f1,
        },
    ]

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Evaluation summary written to %s", output_path)


def main() -> None:
    """Execute the ML pipeline.

    Sequence:
        1. Load data/features/, exclude cold-start rows
        2. Prepare features and targets
        3. Build chronological train/test split
        4. Train and log baseline models (regression + classification)
        5. Train and log candidate models (Random Forest)
        6. Compare metrics; warn if candidate does not beat baseline
        7. Compute and log feature importance
        8. Write evaluation summary CSV
    """
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    app_config = load_config(str(config_path))
    ml_config: MLConfig = app_config.ml
    start_time = time.time()

    logger.info(
        "ML pipeline started (seed=%d, cutoff=%s, cv_folds=%d)",
        ml_config.random_seed,
        ml_config.train_cutoff_date,
        ml_config.cv_folds,
    )

    tracking_path = Path(ml_config.tracking_uri).resolve()
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment(ml_config.experiment_name)

    try:
        # 1. Load feature store
        features_dir = Path("data/features")
        input_path = _find_latest_feature_store(features_dir)
        logger.info("Loading feature store from: %s", input_path)
        df = pd.read_parquet(input_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # 2. Prepare features and targets
        np.random.seed(ml_config.random_seed)
        x, y_reg, y_cls = prepare_features(df, ml_config)
        logger.info(
            "Features prepared: x=%s, y_reg=%s, y_cls=%s (cold-start excluded)",
            x.shape,
            y_reg.shape,
            y_cls.shape,
        )

        # Need timestamp index for chronological split
        working = df[~df["feature_cold_start"]].copy()
        working = working.sort_values("Timestamp")

        # 3. Chronological split
        cutoff = pd.Timestamp(ml_config.train_cutoff_date, tz="UTC")
        train_mask = working["Timestamp"] < cutoff
        test_mask = working["Timestamp"] >= cutoff

        x_train, x_test = x[train_mask.values], x[test_mask.values]
        y_reg_train, y_reg_test = y_reg[train_mask.values], y_reg[test_mask.values]
        y_cls_train, y_cls_test = y_cls[train_mask.values], y_cls[test_mask.values]

        logger.info(
            "Chronological split: train=%d, test=%d (cutoff=%s)",
            len(x_train),
            len(x_test),
            ml_config.train_cutoff_date,
        )

        # 4. Train and log baseline models
        logger.info("=== Training baseline models ===")

        with mlflow.start_run(run_name=_run_name("regression", "baseline"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "regression",
                    "model_type": "baseline",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "cv_folds": ml_config.cv_folds,
                }
            )

            baseline_reg = build_baseline_model("regression", ml_config)
            baseline_reg = train_model(baseline_reg, x_train, y_reg_train, ml_config)
            baseline_reg_pred = baseline_reg.predict(x_test)

            baseline_cv = run_cross_validation(baseline_reg, x_train, y_reg_train, ml_config)
            for i, s in enumerate(baseline_cv):
                mlflow.log_metric(f"cv_fold_{i}_r2", s)

            mlflow.sklearn.log_model(
                baseline_reg, "baseline_regression_model", skops_trusted_types=["numpy.dtype"]
            )
            logger.info("Baseline regression model logged to MLflow (run=%s)", run.info.run_id)

        with mlflow.start_run(
            run_name=_run_name("classification", "baseline"), nested=False
        ) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "classification",
                    "model_type": "baseline",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "cv_folds": ml_config.cv_folds,
                }
            )

            baseline_cls = build_baseline_model("classification", ml_config)
            baseline_cls = train_model(baseline_cls, x_train, y_cls_train, ml_config)
            baseline_cls_pred = baseline_cls.predict(x_test)

            cls_cv = run_cross_validation(baseline_cls, x_train, y_cls_train, ml_config)
            for i, s in enumerate(cls_cv):
                mlflow.log_metric(f"cv_fold_{i}_accuracy", s)

            mlflow.sklearn.log_model(
                baseline_cls, "baseline_classification_model", skops_trusted_types=["numpy.dtype"]
            )
            logger.info("Baseline classification model logged to MLflow (run=%s)", run.info.run_id)

        # 5. Train and log candidate (RF) models
        logger.info("=== Training candidate (Random Forest) models ===")

        regression_metrics: RegressionMetrics | None = None
        classification_metrics: ClassificationMetrics | None = None

        with mlflow.start_run(run_name=_run_name("regression", "rf"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "regression",
                    "model_type": "rf",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "cv_folds": ml_config.cv_folds,
                }
            )

            rf_reg = build_model_pipeline("regression", ml_config)
            rf_reg = train_model(rf_reg, x_train, y_reg_train, ml_config)
            rf_reg_pred = rf_reg.predict(x_test)

            rf_cv = run_cross_validation(rf_reg, x_train, y_reg_train, ml_config)
            for i, s in enumerate(rf_cv):
                mlflow.log_metric(f"cv_fold_{i}_r2", s)

            regression_metrics = evaluate_regression(
                y_reg_test.values, rf_reg_pred, baseline_reg_pred
            )
            _log_regression_metrics("regression_rf", regression_metrics)

            # Warn if candidate does not beat baseline
            if regression_metrics.r2 <= regression_metrics.baseline_r2:
                logger.warning(
                    "Regression candidate R² (%.4f) <= baseline R² (%.4f). "
                    "Both metrics are reported.",
                    regression_metrics.r2,
                    regression_metrics.baseline_r2,
                )

            # Feature importance
            imp_report = compute_feature_importance(
                rf_reg, x_test, y_reg_test, FEATURE_COLUMNS, ml_config.random_seed
            )
            imp_chart_path = Path("outputs/figures/ml/regression_feature_importance.png")
            imp_chart_path.parent.mkdir(parents=True, exist_ok=True)
            _make_importance_chart(imp_report, imp_chart_path)
            mlflow.log_artifact(str(imp_chart_path), "feature_importance")
            mlflow.log_text(
                json.dumps(
                    {
                        "feature_names": imp_report.feature_names,
                        "native_importance": imp_report.native_importance,
                        "permutation_importance_mean": imp_report.permutation_importance_mean,
                        "permutation_importance_std": imp_report.permutation_importance_std,
                    }
                ),
                "feature_importance.json",
            )

            mlflow.sklearn.log_model(
                rf_reg, "regression_rf_model", skops_trusted_types=["numpy.dtype"]
            )
            logger.info("RF regression model logged to MLflow (run=%s)", run.info.run_id)

        with mlflow.start_run(run_name=_run_name("classification", "rf"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "classification",
                    "model_type": "rf",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "cv_folds": ml_config.cv_folds,
                }
            )

            rf_cls = build_model_pipeline("classification", ml_config)
            rf_cls = train_model(rf_cls, x_train, y_cls_train, ml_config)
            rf_cls_pred = rf_cls.predict(x_test)
            rf_cls_proba = rf_cls.predict_proba(x_test)[:, 1]

            rf_cls_cv = run_cross_validation(rf_cls, x_train, y_cls_train, ml_config)
            for i, s in enumerate(rf_cls_cv):
                mlflow.log_metric(f"cv_fold_{i}_roc_auc", s)

            classification_metrics = evaluate_classification(
                y_cls_test.values, rf_cls_pred, rf_cls_proba, baseline_cls_pred
            )
            _log_classification_metrics("classification_rf", classification_metrics)

            if classification_metrics.accuracy <= classification_metrics.baseline_accuracy:
                logger.warning(
                    "Classification candidate accuracy (%.4f) <= baseline accuracy (%.4f). "
                    "Both metrics are reported.",
                    classification_metrics.accuracy,
                    classification_metrics.baseline_accuracy,
                )

            imp_report = compute_feature_importance(
                rf_cls, x_test, y_cls_test, FEATURE_COLUMNS, ml_config.random_seed
            )
            imp_chart_path = Path("outputs/figures/ml/classification_feature_importance.png")
            imp_chart_path.parent.mkdir(parents=True, exist_ok=True)
            _make_importance_chart(imp_report, imp_chart_path)
            mlflow.log_artifact(str(imp_chart_path), "feature_importance")
            mlflow.log_text(
                json.dumps(
                    {
                        "feature_names": imp_report.feature_names,
                        "native_importance": imp_report.native_importance,
                        "permutation_importance_mean": imp_report.permutation_importance_mean,
                        "permutation_importance_std": imp_report.permutation_importance_std,
                    }
                ),
                "feature_importance.json",
            )

            mlflow.sklearn.log_model(
                rf_cls, "classification_rf_model", skops_trusted_types=["numpy.dtype"]
            )
            logger.info("RF classification model logged to MLflow (run=%s)", run.info.run_id)

        # 6. XGBoost candidate models
        logger.info("=== Training XGBoost candidate models ===")
        xgb_reg_metrics: RegressionMetrics | None = None
        xgb_cls_metrics: ClassificationMetrics | None = None

        with mlflow.start_run(run_name=_run_name("regression", "xgb"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "regression",
                    "model_type": "xgb",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                }
            )
            xgb_reg = build_xgboost_pipeline("regression", ml_config)
            xgb_reg = train_model(xgb_reg, x_train, y_reg_train, ml_config)
            xgb_reg_pred = xgb_reg.predict(x_test)
            xgb_reg_metrics = evaluate_regression(
                y_reg_test.values, xgb_reg_pred, baseline_reg_pred
            )
            _log_regression_metrics("regression_xgb", xgb_reg_metrics)
            shap_base = Path("outputs/figures/ml/xgb_regression")
            shap_base.parent.mkdir(parents=True, exist_ok=True)
            _compute_shap_importance(xgb_reg, x_test, FEATURE_COLUMNS, shap_base)
            _skops_trusted = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
            mlflow.sklearn.log_model(
                xgb_reg, "regression_xgb_model", skops_trusted_types=_skops_trusted
            )
            logger.info("XGBoost regression logged (run=%s)", run.info.run_id)

        with mlflow.start_run(run_name=_run_name("classification", "xgb"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "classification",
                    "model_type": "xgb",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                }
            )
            xgb_cls = build_xgboost_pipeline("classification", ml_config)
            xgb_cls = train_model(xgb_cls, x_train, y_cls_train, ml_config)
            xgb_cls_pred = xgb_cls.predict(x_test)
            xgb_cls_proba = xgb_cls.predict_proba(x_test)[:, 1]
            xgb_cls_metrics = evaluate_classification(
                y_cls_test.values, xgb_cls_pred, xgb_cls_proba, baseline_cls_pred
            )
            _log_classification_metrics("classification_xgb", xgb_cls_metrics)
            shap_base = Path("outputs/figures/ml/xgb_classification")
            _compute_shap_importance(xgb_cls, x_test, FEATURE_COLUMNS, shap_base)
            _skops_trusted = [
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ]
            mlflow.sklearn.log_model(
                xgb_cls, "classification_xgb_model", skops_trusted_types=_skops_trusted
            )
            logger.info("XGBoost classification logged (run=%s)", run.info.run_id)

        # 7. LightGBM candidate models
        logger.info("=== Training LightGBM candidate models ===")
        lgbm_reg_metrics: RegressionMetrics | None = None
        lgbm_cls_metrics: ClassificationMetrics | None = None

        with mlflow.start_run(run_name=_run_name("regression", "lgbm"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "regression",
                    "model_type": "lgbm",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                }
            )
            lgbm_reg = build_lightgbm_pipeline("regression", ml_config)
            lgbm_reg = train_model(lgbm_reg, x_train, y_reg_train, ml_config)
            lgbm_reg_pred = lgbm_reg.predict(x_test)
            lgbm_reg_metrics = evaluate_regression(
                y_reg_test.values, lgbm_reg_pred, baseline_reg_pred
            )
            _log_regression_metrics("regression_lgbm", lgbm_reg_metrics)
            shap_base = Path("outputs/figures/ml/lgbm_regression")
            _compute_shap_importance(lgbm_reg, x_test, FEATURE_COLUMNS, shap_base)
            _skops_trusted = [
                "numpy.dtype",
                "lightgbm.sklearn.LGBMRegressor",
                "lightgbm.basic.Booster",
                "collections.OrderedDict",
            ]
            mlflow.sklearn.log_model(
                lgbm_reg, "regression_lgbm_model", skops_trusted_types=_skops_trusted
            )
            logger.info("LightGBM regression logged (run=%s)", run.info.run_id)

        with mlflow.start_run(run_name=_run_name("classification", "lgbm"), nested=False) as run:
            mlflow.log_params(
                {
                    "random_seed": ml_config.random_seed,
                    "task": "classification",
                    "model_type": "lgbm",
                    "cutoff_date": ml_config.train_cutoff_date,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                }
            )
            lgbm_cls = build_lightgbm_pipeline("classification", ml_config)
            lgbm_cls = train_model(lgbm_cls, x_train, y_cls_train, ml_config)
            lgbm_cls_pred = lgbm_cls.predict(x_test)
            lgbm_cls_proba = lgbm_cls.predict_proba(x_test)[:, 1]
            lgbm_cls_metrics = evaluate_classification(
                y_cls_test.values, lgbm_cls_pred, lgbm_cls_proba, baseline_cls_pred
            )
            _log_classification_metrics("classification_lgbm", lgbm_cls_metrics)
            shap_base = Path("outputs/figures/ml/lgbm_classification")
            _compute_shap_importance(lgbm_cls, x_test, FEATURE_COLUMNS, shap_base)
            _skops_trusted = [
                "numpy.dtype",
                "lightgbm.sklearn.LGBMClassifier",
                "lightgbm.basic.Booster",
                "collections.OrderedDict",
            ]
            mlflow.sklearn.log_model(
                lgbm_cls, "classification_lgbm_model", skops_trusted_types=_skops_trusted
            )
            logger.info("LightGBM classification logged (run=%s)", run.info.run_id)

        # 8. Multi-model comparison summary
        summary_records = []
        model_reg_pairs = [
            ("rf", regression_metrics),
            ("xgb", xgb_reg_metrics),
            ("lgbm", lgbm_reg_metrics),
        ]
        for model_name, m in model_reg_pairs:
            if m is not None:
                summary_records.append(
                    {
                        "task": "regression",
                        "model": model_name,
                        "random_seed": ml_config.random_seed,
                        "cutoff_date": ml_config.train_cutoff_date,
                        "rmse": m.rmse,
                        "mae": m.mae,
                        "r2": m.r2,
                        "baseline_rmse": m.baseline_rmse,
                        "baseline_mae": m.baseline_mae,
                        "baseline_r2": m.baseline_r2,
                        "lift_rmse": m.lift_rmse,
                        "lift_mae": m.lift_mae,
                        "lift_r2": m.lift_r2,
                    }
                )

        model_cls_pairs = [
            ("rf", classification_metrics),
            ("xgb", xgb_cls_metrics),
            ("lgbm", lgbm_cls_metrics),
        ]
        for model_name, m in model_cls_pairs:
            if m is not None:
                summary_records.append(
                    {
                        "task": "classification",
                        "model": model_name,
                        "random_seed": ml_config.random_seed,
                        "cutoff_date": ml_config.train_cutoff_date,
                        "accuracy": m.accuracy,
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "roc_auc": m.roc_auc,
                        "baseline_accuracy": m.baseline_accuracy,
                        "baseline_precision": m.baseline_precision,
                        "baseline_recall": m.baseline_recall,
                        "baseline_f1": m.baseline_f1,
                        "lift_accuracy": m.lift_accuracy,
                        "lift_precision": m.lift_precision,
                        "lift_recall": m.lift_recall,
                        "lift_f1": m.lift_f1,
                    }
                )

        if summary_records:
            summary_path = Path("outputs/tables/statistics/ml_evaluation_summary.csv")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary_records).to_csv(summary_path, index=False)
            logger.info("Multi-model evaluation summary saved to %s", summary_path)

        # 9. Config artifact
        config_snapshot = ml_config.model_dump_json(indent=2)
        mlflow.log_text(config_snapshot, "ml_config.json")

    except Exception:
        logger.exception("ML pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("ML pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
