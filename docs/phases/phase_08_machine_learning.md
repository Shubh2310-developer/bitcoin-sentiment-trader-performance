# Phase 08: Machine Learning (Optional)
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Train and evaluate baseline predictive models on the engineered feature store, using time-aware splits and rigorous evaluation. This phase is **optional** and is invoked **only** if Phase 06 establishes a statistically defensible signal worth modeling (at least one hypothesis test with p < α and non-negligible effect size).

**Pipeline stage:** §11.7 — Machine Learning (`run_ml_pipeline.py`)

---

## 2. Prerequisites

- [ ] Phase 07 complete: at least one insight with `report_ready: True` and non-negligible effect size
- [ ] Principal Engineer decision to invoke ML module (explicit sign-off required)
- [ ] `configs/pipelines/ml.yaml` fully configured: `random_seed`, `train_cutoff_date`, `cv_folds`, `baseline_strategy`
- [ ] MLflow tracking configured (`experiments/mlruns/`)

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | `data/features/` |
| **Output** | Model artifacts + metrics logged to `experiments/mlruns/` |
| **Responsibility** | Train/evaluate per §10 |
| **Failure condition** | Any metric worse than baseline must still be reported, never discarded |

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `machine-learning-engineer` | `/agent:machine-learning-engineer` |
| **Supporting** | `model-evaluator` | `/agent:model-evaluator` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-ml-engineer` | `/skill:senior-ml-engineer` | MLOps patterns, time-series CV, leakage prevention |
| `scikit-learn` | `/skill:scikit-learn` | Pipeline construction, cross-validation, metrics |
| `mlops-weights-and-biases` | `/skill:mlops-weights-and-biases` | Experiment tracking, artifact registry |
| `ml-engineer` | `/skill:ml-engineer` | Model serving, feature store integration |

---

## 6. Implementation Tasks

### Task 8.1 — Training Module (`src/sentiment_trader_analytics/ml/training.py`)

**Functions:**
- `prepare_features(df: pd.DataFrame, config: MLConfig) -> tuple[pd.DataFrame, pd.Series]` — returns feature matrix X and target y, with cold-start rows excluded
- `build_train_test_split(X, y, config: MLConfig) -> tuple[...]` — chronological split at `config.train_cutoff_date`
- `build_baseline_model(strategy: str, config: MLConfig) -> BaseEstimator` — mean predictor (regression) or majority class (classification)
- `build_model_pipeline(config: MLConfig) -> Pipeline` — scikit-learn Pipeline with preprocessing + model
- `run_cross_validation(pipeline, X_train, y_train, config: MLConfig) -> CVResult` — `TimeSeriesSplit` with configured folds
- `train_model(pipeline, X_train, y_train, config: MLConfig) -> FittedModel`

**Reproducibility:** `np.random.seed(config.random_seed)` called at top of training run. Seed logged to MLflow.

### Task 8.2 — Evaluation Module (`src/sentiment_trader_analytics/ml/evaluation.py`)

**Task types and required metrics:**

Regression (PnL prediction):
- RMSE, MAE, R²
- Lift above baseline (delta on each metric)

Classification (profitable vs. unprofitable trade):
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix
- Class imbalance explicitly addressed (class weights or SMOTE — documented in `docs/methodology.md`)
- Lift above majority-class baseline

**Functions:**
- `evaluate_regression(y_true, y_pred, baseline_pred) -> RegressionMetrics`
- `evaluate_classification(y_true, y_pred, y_proba, baseline_pred) -> ClassificationMetrics`
- `compute_feature_importance(model, X, config: MLConfig) -> FeatureImportanceReport` — model-native + permutation importance

### Task 8.3 — MLflow Logging (`src/sentiment_trader_analytics/ml/training.py`)

Every training run logs:
- Config snapshot (as JSON artifact)
- Training and test metrics
- Feature importance chart
- Training data version (git SHA + processed dataset path)
- Model artifact (serialized via `mlflow.sklearn.log_model`)
- Random seed used

**Run naming convention:** `ml_<task>_<model_type>_<date>` e.g., `ml_pnl_regression_rf_20260701`

### Task 8.4 — Pipeline Orchestrator (`pipelines/run_ml_pipeline.py`)

Sequence:
1. Load `data/features/`, exclude cold-start rows
2. Prepare features and target
3. Build chronological train/test split
4. Train and log baseline model
5. Train and log candidate model(s)
6. Compare metrics: if candidate does not beat baseline, log at `WARNING` but do **not** discard — both metrics are reported
7. Compute and log feature importance
8. Write evaluation summary to `outputs/tables/statistics/ml_evaluation_summary.csv`

---

## 7. Model Scope (Initial)

| Task | Target Variable | Model Type | Baseline |
|---|---|---|---|
| Regression | `Closed PnL` | Random Forest Regressor | Predict mean training PnL |
| Classification | `profitable_trade` (PnL > 0) | Random Forest Classifier | Majority class predictor |

A single model family (Random Forest) is used first for interpretability and feature importance. Complexity is added only if the baseline is beaten and the business case warrants it.

---

## 8. Verification Commands

```bash
python pipelines/run_ml_pipeline.py --config configs/base.yaml

# Inspect MLflow runs
mlflow ui --backend-store-uri experiments/mlruns

# Verify baseline was logged
mlflow runs list --experiment-name "ml_experiments" | grep baseline

# Verify metrics present
python -c "
import mlflow
runs = mlflow.search_runs()
assert 'metrics.rmse' in runs.columns or 'metrics.roc_auc' in runs.columns
"

pytest tests/unit/test_ml.py -v
ruff check src/sentiment_trader_analytics/ml/
mypy src/sentiment_trader_analytics/ml/
```

---

## 9. Go / No-Go Gate

| Check | Verification |
|---|---|
| Baseline model logged in MLflow | Run with "baseline" in name exists |
| All required metrics present | No null values in metrics columns |
| Metrics worse than baseline are reported, not discarded | Both baseline and model metrics in evaluation CSV |
| Feature importance computed | Importance report artifact present in MLflow run |
| Random seed logged | `params.random_seed` in MLflow run |
| All unit tests pass | `pytest tests/unit/test_ml.py` exits 0 |

---

## 10. Limitations Disclosure (Mandatory in Final Report)

The following limitations must appear in the final report alongside any ML findings:
- Dataset temporal coverage (date range)
- Hyperliquid-only training data (venue-specific, not generalizable)
- Feature store covers only available trades (open positions excluded)
- Out-of-time test set may not represent future market regimes

---

## 11. Enhanced ML Requirements (v2.0 — Hiring Assignment Optimization)

The current Random Forest baseline provides a starting point. The following enhancements are required to demonstrate depth and competitiveness:

### 11.1 Model Diversity

| Model | Task | Why | Expected Gain |
|-------|------|-----|--------------|
| XGBoost | Both | Gradient boosting, handles non-linearity well | Best chance at lift |
| LightGBM | Both | Faster, leaf-wise growth, handles categoricals | Speed + potential accuracy |
| Logistic Regression | Classification | Interpretable baseline for comparison | Coefficients as feature importance |
| Ridge Regression | Regression | Handles multicollinearity in features | Stability vs RF variance |
| Ensemble (Voting) | Both | Combine RF + XGBoost predictions | Variance reduction |

### 11.2 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# RF params
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# XGBoost params
xgb_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}
```

### 11.3 Feature Engineering for ML (Additional)

Create a second feature set optimized for ML (not for statistical analysis):
- **Interaction features**: sentiment × leverage, sentiment × size, sentiment × day_of_week
- **Polynomial features**: sentiment_value², sentiment_value³ (non-linear effects)
- **Ratio features**: trader_win_rate_7d / market_avg_win_rate, trader_size / account_avg_size
- **Rolling rank features**: percentile rank of PnL within account's own history (14d, 30d)
- **Market context features**: regime change flags, days_since_regime_change, consecutive_days_in_regime

### 11.4 Feature Selection & Importance Analysis

```python
# Recursive Feature Elimination (RFE) for stability
from sklearn.feature_selection import RFECV

# Permutation importance on best model for each task
from sklearn.inspection import permutation_importance

# SHAP values for interpretability
import shap
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### 11.5 Learning Curve Analysis

- Plot training vs validation score as training size increases
- Identify whether the model suffers from high bias (underfitting) or high variance (overfitting)
- This is critical for the interview — it shows you know how to diagnose model failures

### 11.6 Error Analysis (Interview Gold)

- For classification: analyze which trades are misclassified (False Positives vs False Negatives)
  - Are we over-predicting profitable trades in Greed regimes?
  - Are we missing unprofitable trades in Fear regimes?
- For regression: plot residuals vs. predicted values, check for heteroscedasticity
- Segment error by trader archetype (from ET-03): does the model perform differently for different clusters?
- **Deliverable**: `ml_error_analysis.csv` with error rates per archetype

### 11.7 Failure Explanation (Critical for Interview)

Create a section in the final report titled **"Why Sentiment Doesn't Predict PnL"** that covers:
1. **Signal-to-noise ratio**: PnL variance is dominated by market moves, not sentiment
2. **Temporal misalignment**: daily sentiment granularity misses intraday trader decisions
3. **Missing confounders**: no account-level risk management data, no market microstructure data
4. **Population bias**: Hyperliquid traders may not represent the broader market
5. **Feature saturation**: with 16 features at the current level, we may have hit the information ceiling for this dataset

### 11.8 Updated Deliverables

```
outputs/tables/ml/
├── model_comparison.csv                    # All models, all metrics side-by-side
├── hyperparameter_tuning_results.csv       # Best params per model
├── feature_importance_ranked.csv           # SHAP + permutation + native importance
├── learning_curves.png                     # Train/val scores vs sample size
├── classification_error_analysis.csv       # FP/FN breakdown by regime & archetype
├── residual_analysis.png                   # Regression residual plots
└── ml_failure_analysis.md                  # Written explanation of model limitations
```

### 11.9 Updated Go/No-Go Gate

| Check | Verification |
|---|---|
| 3+ model types compared | `model_comparison.csv` has 3+ rows |
| Hyperparameter tuning logged | `hyperparameter_tuning_results.csv` exists |
| Feature importance with SHAP | `feature_importance_ranked.csv` exists |
| Learning curve analysis | `learning_curves.png` exists |
| Error analysis complete | `classification_error_analysis.csv` exists |
| Failure analysis written | `ml_failure_analysis.md` exists |
| MLflow runs with metrics | `experiments/mlruns/` has 6+ runs |

---

*Governed by §10. Proceed to [Phase 09](phase_09_reporting.md) upon gate clearance.*
