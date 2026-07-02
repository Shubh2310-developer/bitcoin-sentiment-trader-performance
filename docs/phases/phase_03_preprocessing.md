# Phase 03: Preprocessing & Cleaning
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Apply all cleaning transformations to the validated DataFrames from Phase 02: handle missing values (with logged counts), deduplicate by primary key, verify UTC timezone normalization (already enforced at ingestion, verified here), and join the sentiment and trader datasets on UTC calendar date. Write the analysis-ready output to `data/processed/`.

**Pipeline stage:** §11.3 — Cleaning / Preprocessing (`run_preprocessing_pipeline.py`)

---

## 2. Prerequisites

- [ ] Phase 02 complete (Go/No-Go gate cleared)
- [ ] `configs/pipelines/preprocessing.yaml` populated with join strategy, null strategies, and fan-out tolerance
- [ ] `docs/methodology.md` updated with null handling decisions before this phase runs
- [ ] `data/interim/` directory exists (auto-created by pipeline if absent)
- [ ] `data/processed/` directory exists

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | Validated DataFrames from Phase 02 |
| **Output** | Cleaned data written to `data/interim/`, then merged output to `data/processed/` |
| **Responsibility** | Handle missing values, duplicates, timezone normalization, join sentiment and trader datasets on date |
| **Failure condition** | Post-merge row count anomalies (e.g., unexpected fan-out from the join) halt the pipeline pending investigation |

---

## 4. Agent Assignment

| Role | Agent | Activation Command |
|---|---|---|
| **Primary** | `python-pro` | `/agent:python-pro` |
| **Supporting** | `data-scientist` | `/agent:data-scientist` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-engineer` | `/skill:senior-data-engineer` | ETL join design, data quality remediation patterns |
| `python-patterns` | `/skill:python-patterns` | Pure function design for cleaning transformations |

---

## 6. Command Invocations

| Command | Activation | Purpose |
|---|---|---|
| `update-docs` | `/update-docs` | Update `docs/methodology.md` with null handling strategy decisions |

---

## 7. Implementation Tasks

### Task 3.1 — Cleaning Module (`src/sentiment_trader_analytics/preprocessing/cleaning.py`)

**Functions:**

`clean_fear_greed(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame`
- Verify `timestamp` is `datetime64[UTC]` (assert, do not coerce — coercion happened at ingestion)
- Drop rows with null `value` or `classification`; log dropped count and proportion at `WARNING`
- Remove duplicate timestamps; log removed count at `WARNING`
- Return cleaned DataFrame

`clean_trader_history(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame`
- Verify `Timestamp` is `datetime64[UTC]`
- Drop rows with null `Account`, `Timestamp`, `Size USD`, or `Execution Price`; log each column's drop count and proportion at `WARNING`
- Deduplicate by `Trade ID` (exact duplicates only; non-exact duplicates with same ID → `ERROR` + halt)
- Log total row count before and after cleaning at `INFO`
- Return cleaned DataFrame

**Critical rule:** Null strategy choices (drop vs. impute) must be documented in `docs/methodology.md` §2.4 **before** this code is merged.

### Task 3.2 — Merging Module (`src/sentiment_trader_analytics/preprocessing/merging.py`)

**Functions:**

`extract_trade_date(df: pd.DataFrame) -> pd.DataFrame`
- Derives `trade_date_utc` column: `df["Timestamp"].dt.date` (UTC calendar date)
- Returns DataFrame with new column added

`merge_sentiment_and_trades(trades_df: pd.DataFrame, sentiment_df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame`
- Left join: trades (left) on `trade_date_utc` against sentiment (right) on `timestamp` date
- Post-merge, rows where `sentiment_value` is null are flagged with a boolean column `sentiment_missing`
- Log: match rate (% of trades with a sentiment record), unmatched trade count at `INFO`
- **Fan-out check**: if merged row count exceeds pre-join trade row count by more than the configured tolerance (default 0%), log at `ERROR` and halt — this indicates a non-unique sentiment date (which should have been caught in validation)
- Return merged DataFrame

### Task 3.3 — Pipeline Orchestrator (`pipelines/run_preprocessing_pipeline.py`)

Sequence:
1. Load validated DataFrames (from in-memory handoff or `data/interim/`)
2. Call `clean_fear_greed` and `clean_trader_history`
3. Write interim outputs to `data/interim/` (parquet preferred, configurable)
4. Call `extract_trade_date` and `merge_sentiment_and_trades`
5. Write final processed dataset to `data/processed/` with versioned filename
6. Write metadata snapshot to `data/metadata/data_dictionary/` and `data/metadata/lineage/`
7. Log total pipeline duration at `INFO`

**Config parameters (all from `configs/pipelines/preprocessing.yaml`):**
- `interim_output_format` (parquet | csv)
- `processed_output_path`
- `fan_out_tolerance_fraction` (default 0.0)
- `null_strategy_fear_greed` (drop | flag)
- `null_strategy_trader` (drop | flag | halt)

---

## 8. Verification Commands

```bash
# Run preprocessing pipeline
python pipelines/run_preprocessing_pipeline.py --config configs/base.yaml

# Inspect processed output
python -c "import pandas as pd; df = pd.read_parquet('data/processed/'); print(df.shape, df.dtypes)"

# Verify no naive timestamps
python -c "import pandas as pd; df = pd.read_parquet('data/processed/'); assert df['Timestamp'].dt.tz is not None"

# Verify sentiment match rate in logs
grep "match_rate" logs/pipeline.log

# Run unit tests
pytest tests/unit/test_preprocessing.py -v

# Linting
ruff check src/sentiment_trader_analytics/preprocessing/
mypy src/sentiment_trader_analytics/preprocessing/
```

---

## 9. Go / No-Go Gate

**Proceed to Phase 04 only when ALL of the following are true:**

| Check | Verification |
|---|---|
| Post-merge row count within fan-out tolerance | No `ERROR`-level fan-out log entries |
| All timestamps UTC-aware in processed output | Assertion passes: `df["Timestamp"].dt.tz is not None` |
| Null handling strategy documented | `docs/methodology.md §2.4` table is populated with current strategies |
| Processed dataset written | File present in `data/processed/` |
| Metadata records written | Files present in `data/metadata/data_dictionary/` and `data/metadata/lineage/` |
| All unit tests pass | `pytest tests/unit/test_preprocessing.py` exits 0 |

---

## 10. Unit Test Requirements (`tests/unit/test_preprocessing.py`)

- `test_clean_fear_greed_drops_nulls` — null rows dropped; drop count logged
- `test_clean_fear_greed_deduplicates` — duplicate timestamps removed; count logged
- `test_clean_trader_history_drops_null_account` — null Account rows dropped with log
- `test_clean_trader_history_exact_duplicate_trade_id` — exact duplicate removed with log
- `test_clean_trader_history_nonexact_duplicate_trade_id` — non-exact duplicate Trade ID raises `DataQualityError`
- `test_merge_left_join_preserves_trades` — all trade rows present in merged output
- `test_merge_sentiment_missing_flagged` — trades without sentiment get `sentiment_missing == True`
- `test_merge_fan_out_detection` — duplicate sentiment dates trigger fan-out halt
- `test_extract_trade_date` — UTC date correctly extracted from timestamps

---

*Governed by §5.4. Proceed to [Phase 04](phase_04_feature_engineering.md) upon gate clearance.*
