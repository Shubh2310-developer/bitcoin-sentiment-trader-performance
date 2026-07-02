# Phase 04: Feature Engineering
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Construct all engineered features using pure, stateless functions. Features cover three domains: sentiment-derived features, trader performance metrics, and time-based features. All features are schema-validated via Pandera before being written to `data/features/`. Every feature is documented in `docs/data_dictionary.md` before merge.

**Pipeline stage:** §11.4 — Feature Engineering (`run_feature_pipeline.py`)

---

## 2. Prerequisites

- [ ] Phase 03 complete (Go/No-Go gate cleared)
- [ ] `data/processed/` contains the canonical merged dataset
- [ ] `configs/pipelines/feature_engineering.yaml` populated with all window sizes, lag values, and encoding maps
- [ ] `docs/data_dictionary.md §4` populated with feature definitions before code is written
- [ ] Feature schema definitions written to `data/metadata/schemas/` before pipeline runs

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | `data/processed/` |
| **Output** | `data/features/` |
| **Responsibility** | Compute all engineered features per §6 |
| **Failure condition** | Feature schema validation failure; detected look-ahead bias in a new feature |

---

## 4. Agent Assignment

| Role | Agent | Activation Command |
|---|---|---|
| **Primary** | `machine-learning-engineer` | `/agent:machine-learning-engineer` |
| **Supporting** | `python-pro` | `/agent:python-pro` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-ml-engineer` | `/skill:senior-ml-engineer` | Feature store design, look-ahead bias review |
| `ml-engineer` | `/skill:ml-engineer` | Feature pipeline wiring, Pandera feature validation |
| `scikit-learn` | `/skill:scikit-learn` | Feature preprocessing patterns (encoding, scaling references) |

---

## 6. Command Invocations

| Command | Activation | Purpose |
|---|---|---|
| `update-docs` | `/update-docs` | Sync `docs/data_dictionary.md` after new features added |

---

## 7. Implementation Tasks

### Task 4.1 — Sentiment Features (`src/sentiment_trader_analytics/feature_engineering/sentiment_features.py`)

**All functions have signature:** `f(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame`

| Function | Feature(s) Produced | Look-Ahead Check |
|---|---|---|
| `add_sentiment_lag` | `sentiment_value_lag_1d` | Uses `.shift(1)` — safe |
| `add_sentiment_regime_encoding` | `sentiment_regime_encoded` | Point-in-time classification — safe |
| `add_sentiment_rolling_mean` | `sentiment_value_rolling_7d` | `closed=left` window excludes current row — verified |
| `add_sentiment_fear_greed_flags` | `sentiment_is_fear`, `sentiment_is_greed` | Point-in-time — safe |

Window sizes sourced from `configs/pipelines/feature_engineering.yaml`. Encoding map for `sentiment_regime_encoded` sourced from same config.

### Task 4.2 — Trader Features (`src/sentiment_trader_analytics/feature_engineering/trader_features.py`)

**All functions operate per-account (`.groupby("Account")`) and use historical-only windows.**

| Function | Feature(s) Produced | Window | Look-Ahead Check |
|---|---|---|---|
| `add_trader_win_rate` | `trader_win_rate_7d` | 7 days | `min_periods=1`, `closed="left"` — safe |
| `add_trader_pnl_rolling` | `trader_pnl_rolling_7d`, `trader_pnl_rolling_30d` | 7d, 30d | `closed="left"` — safe |
| `add_trader_leverage_avg` | `trader_leverage_avg_24h` | 24h | Time-based rolling with `closed="left"` — safe |
| `add_trader_pnl_volatility` | `trader_pnl_volatility_14d` | 14 days | `closed="left"` — safe |
| `add_trader_trade_count` | `trader_trade_count_7d` | 7 days | `closed="left"` — safe |
| `add_trader_avg_size` | `trader_avg_size_usd_7d` | 7 days | `closed="left"` — safe |

**Cold-start handling:** Rows within the first full window of an account's history will produce `NaN` for rolling features. These rows are **not dropped** — they are retained and flagged with a boolean `feature_cold_start` column. Statistical analysis stages exclude cold-start rows when computing regime comparisons.

### Task 4.3 — Time Features (`src/sentiment_trader_analytics/feature_engineering/time_features.py`)

| Function | Feature(s) Produced |
|---|---|
| `add_time_of_day` | `time_hour_utc` |
| `add_day_of_week` | `time_day_of_week`, `time_is_weekend` |
| `add_month` | `time_month` |

All derived directly from `Timestamp` — no windowing, no look-ahead risk.

### Task 4.4 — Feature Validation Schema (`data/metadata/schemas/features_schema.py`)

Pandera schema covering all engineered features:
- Numeric features: correct dtype, valid range (e.g., `trader_win_rate_7d` in [0.0, 1.0])
- Boolean features: dtype `bool`
- Categorical encoding: `sentiment_regime_encoded` in {0, 1, 2, 3, 4}
- `feature_cold_start`: dtype `bool`, non-null

### Task 4.5 — Pipeline Orchestrator (`pipelines/run_feature_pipeline.py`)

Sequence:
1. Load `data/processed/` dataset
2. Apply all sentiment feature functions (compose via `functools.reduce`)
3. Apply all trader feature functions
4. Apply all time feature functions
5. Validate against `features_schema` — halt on any violation
6. Write feature store to `data/features/` (parquet, versioned by run_id)
7. Write schema snapshot and metadata to `data/metadata/`

**Composability pattern:** Feature functions are applied in a pipeline:
```python
from functools import reduce

feature_functions = [
    add_sentiment_lag,
    add_sentiment_regime_encoding,
    add_trader_win_rate,
    # ...
]
features_df = reduce(lambda df, fn: fn(df, config), feature_functions, processed_df)
```

---

## 8. Verification Commands

```bash
# Run feature pipeline
python pipelines/run_feature_pipeline.py --config configs/base.yaml

# Inspect feature store
python -c "
import pandas as pd
df = pd.read_parquet('data/features/')
print('Shape:', df.shape)
print('Columns:', sorted(df.columns.tolist()))
print('Null counts:\n', df.isnull().sum())
"

# Run unit tests
pytest tests/unit/test_feature_engineering.py -v

# Look-ahead bias check: confirm no future rows leak into window
python -c "
import pandas as pd
df = pd.read_parquet('data/features/')
# Verify lag feature is strictly prior to current sentiment
assert (df['sentiment_value_lag_1d'].shift(-1) != df['sentiment_value']).any(), 'Lag looks valid'
"

# Linting
ruff check src/sentiment_trader_analytics/feature_engineering/
mypy src/sentiment_trader_analytics/feature_engineering/
```

---

## 9. Go / No-Go Gate

**Proceed to Phase 05 only when ALL of the following are true:**

| Check | Verification |
|---|---|
| Feature schema validation passes with zero violations | Pandera validation exits clean |
| Zero look-ahead bias detected in code review | Code review checklist signed off |
| All features documented in `docs/data_dictionary.md` | Every feature name in §4 of data dictionary |
| Feature store written | Files present in `data/features/` |
| Cold-start flag column present | `"feature_cold_start"` in feature store columns |
| All unit tests pass | `pytest tests/unit/test_feature_engineering.py` exits 0 |

---

## 10. Unit Test Requirements (`tests/unit/test_feature_engineering.py`)

- `test_sentiment_lag_is_prior_day` — lag column equals previous row's sentiment value
- `test_sentiment_encoding_maps_correctly` — all five regime values map to correct ordinal
- `test_rolling_win_rate_does_not_include_current_row` — window excludes `t=0`
- `test_cold_start_flagged` — first N rows of an account have `feature_cold_start == True`
- `test_win_rate_range` — `trader_win_rate_7d` always in [0.0, 1.0]
- `test_fear_flag_true_for_fear_and_extreme_fear` — `sentiment_is_fear` correctly set
- `test_time_features_correct_utc` — `time_hour_utc` matches UTC timestamp hour
- `test_feature_composition_is_additive` — composed feature set has all expected columns

---

*Governed by §6. Proceed to [Phase 05](phase_05_eda.md) and [Phase 06](phase_06_statistical_analysis.md) in parallel upon gate clearance.*
