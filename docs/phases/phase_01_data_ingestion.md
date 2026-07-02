# Phase 01: Data Ingestion
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Establish validated, production-grade file-reading pipelines for both raw datasets. Produce standardized, dtype-coerced, timezone-aware DataFrames with source metadata attached. Record checksums to `data/metadata/lineage/` for full traceability.

**Pipeline stage:** §11.1 — Ingestion (`run_ingestion_pipeline.py`)

---

## 2. Prerequisites

- [ ] Raw data files present at expected paths:
  - `data/raw/fear_greed/fear_greed_index.csv` (~90 KB)
  - `data/raw/trader_history/historical_data.csv` (~45 MB)
- [ ] Conda environment activated: `conda activate bst`
- [ ] `configs/pipelines/ingestion.yaml` populated with file paths and dtype mappings
- [ ] `configs/data/fear_greed.yaml` and `configs/data/trader_history.yaml` configured
- [ ] `configs/logging.yaml` configured
- [ ] `src/sentiment_trader_analytics/utils/logging_utils.py` implemented

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | Raw files in `data/raw/` |
| **Output** | Standardized, loaded in-memory DataFrames with consistent dtypes |
| **Responsibility** | Read files, enforce initial dtype coercion, attach source metadata |
| **Failure condition** | Missing file, unreadable format, or unexpected column set → hard failure, logged with file path |

---

## 4. Agent Assignment

| Role | Agent | Activation Command |
|---|---|---|
| **Primary** | `python-pro` | `/agent:python-pro` |
| **Supporting** | `debugger` | `/agent:debugger` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-engineer` | `/skill:senior-data-engineer` | ETL pipeline design, checksum verification, metadata generation |
| `python-patterns` | `/skill:python-patterns` | Production-grade Python architecture, type-safe loader design |

---

## 6. Command Invocations

| Command | Activation | Purpose |
|---|---|---|
| `docs-maintenance` | `/docs-maintenance` | Audit docs against code after ingestion modules are written |
| `update-docs` | `/update-docs` | Update `docs/data_dictionary.md` with ingested schema snapshot |

---

## 7. Implementation Tasks

### Task 1.1 — Fear & Greed Loader (`src/sentiment_trader_analytics/ingestion/fear_greed_loader.py`)

**Specification:**
- Pure function: `load_fear_greed_index(config: IngestionConfig) -> pd.DataFrame`
- Reads `data/raw/fear_greed/fear_greed_index.csv` using path from `configs/data/fear_greed.yaml`
- Enforces initial dtype coercion: `timestamp` → `datetime64[UTC]`, `value` → `int64`, `classification` → `pd.CategoricalDtype`
- Attaches source metadata: file path, row count, ingestion timestamp (UTC)
- Computes SHA-256 checksum of the file and writes it to `data/metadata/lineage/fear_greed_<run_id>.json`
- Logs: file path, row count, column names, dtypes — at `INFO` level
- Raises `FileNotFoundError` on missing file; `ValueError` on unexpected column set — both logged at `ERROR`

**Acceptance Criteria:**
- Function is fully typed (`pd.DataFrame` return, `IngestionConfig` parameter)
- Google-style docstring present
- Checksum file written on every call
- All edge cases (missing file, wrong columns, bad dtypes) raise explicitly named exceptions

### Task 1.2 — Trader History Loader (`src/sentiment_trader_analytics/ingestion/trader_history_loader.py`)

**Specification:**
- Pure function: `load_trader_history(config: IngestionConfig) -> pd.DataFrame`
- Reads `data/raw/trader_history/historical_data.csv` (chunked for ~45 MB) using path from `configs/data/trader_history.yaml`
- Enforces dtype coercion: `Timestamp` → `datetime64[UTC]`, `Side`/`Direction` → `pd.CategoricalDtype`, `Size USD`/`Execution Price`/`Closed PnL`/`Fee`/`Leverage` → `float64`, `Account` → `str`
- Attaches source metadata: file path, row count, ingestion timestamp, chunk count if chunked
- Computes SHA-256 checksum and writes to `data/metadata/lineage/trader_history_<run_id>.json`
- Logs: file path, row count, column names, memory usage estimate — at `INFO` level

**Acceptance Criteria:**
- Handles large file (45 MB) via chunked reading (`chunksize` from config)
- No naive datetimes persist past this function
- `Trade ID` column dtype is preserved faithfully

### Task 1.3 — Pipeline Orchestrator (`pipelines/run_ingestion_pipeline.py`)

**Specification:**
- Calls `load_fear_greed_index` and `load_trader_history`
- Logs pipeline start/end with wall-clock duration at `INFO`
- On failure: logs the exception at `ERROR`, exits with non-zero code
- Returns loaded DataFrames to subsequent stages via an in-memory handoff or serialized to `data/interim/` if configured

**Config parameters (all from `configs/pipelines/ingestion.yaml`):**
- `fear_greed_path`
- `trader_history_path`
- `chunk_size`
- `lineage_output_dir`

---

## 8. Verification Commands

```bash
# Run ingestion pipeline standalone
python pipelines/run_ingestion_pipeline.py --config configs/base.yaml

# Verify checksums written
ls -la data/metadata/lineage/

# Run unit tests for ingestion
pytest tests/unit/test_ingestion.py -v

# Check logging output
cat logs/pipeline.log | grep "ingestion"

# Static analysis
ruff check src/sentiment_trader_analytics/ingestion/
mypy src/sentiment_trader_analytics/ingestion/
black --check src/sentiment_trader_analytics/ingestion/
```

---

## 9. Go / No-Go Gate

**Proceed to Phase 02 only when ALL of the following are true:**

| Check | Verification |
|---|---|
| Both loaders execute without exception | `python pipelines/run_ingestion_pipeline.py` exits 0 |
| Checksums written for both datasets | Files present in `data/metadata/lineage/` |
| Initial dtype coercion passes | No `CastError` or `ValueError` in `logs/pipeline.log` |
| Column names match expected schema | Logged column list matches `configs/data/*.yaml` definitions |
| All unit tests pass | `pytest tests/unit/test_ingestion.py` exits 0 |
| Zero ruff warnings | `ruff check src/sentiment_trader_analytics/ingestion/` produces no output |

---

## 10. Unit Test Requirements (`tests/unit/test_ingestion.py`)

- `test_fear_greed_loader_happy_path` — loads fixture CSV, verifies dtypes and row count
- `test_fear_greed_loader_missing_file` — asserts `FileNotFoundError` raised with file path in message
- `test_fear_greed_loader_wrong_columns` — asserts `ValueError` raised on unexpected column set
- `test_trader_history_loader_happy_path` — loads fixture CSV, verifies dtypes and row count
- `test_trader_history_loader_naive_datetime_rejected` — verifies UTC timestamp enforcement
- `test_checksum_written` — verifies lineage file created with correct SHA-256

**Fixture files required (`tests/fixtures/`):**
- `fear_greed_sample.csv` — 10-row valid sample
- `fear_greed_bad_columns.csv` — missing required column
- `trader_history_sample.csv` — 50-row valid sample with all column types

---

*Governed by §5.1, §5.4, §15. Proceed to [Phase 02](phase_02_data_validation.md) upon gate clearance.*
