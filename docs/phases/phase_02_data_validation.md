# Phase 02: Data Validation
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Apply Pandera-based schema enforcement against the ingested DataFrames from Phase 01. Validate all domain constraints for both datasets. Any schema violation beyond the configured tolerance halts the pipeline; violations are logged in full to `logs/validation.log`.

**Pipeline stage:** §11.2 — Validation (`run_validation_pipeline.py`)

---

## 2. Prerequisites

- [ ] Phase 01 complete (Go/No-Go gate cleared)
- [ ] Pandera installed and listed in `pyproject.toml`
- [ ] Schema definitions present in `data/metadata/schemas/`
- [ ] `configs/pipelines/validation.yaml` populated with violation tolerance thresholds
- [ ] `logs/validation.log` handler configured in `configs/logging.yaml`

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | Ingested DataFrames from Phase 01 |
| **Output** | Pass/fail validation report + validated DataFrames |
| **Responsibility** | Enforce schemas defined in `data/metadata/schemas/` |
| **Failure condition** | Any schema violation beyond configured tolerance halts the pipeline; violations logged in full to `logs/validation.log` |

---

## 4. Agent Assignment

| Role | Agent | Activation Command |
|---|---|---|
| **Primary** | `data-scientist` | `/agent:data-scientist` |
| **Supporting** | `python-pro` | `/agent:python-pro` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-scientist` | `/skill:senior-data-scientist` | Schema design, constraint specification, quality assurance |
| `senior-data-engineer` | `/skill:senior-data-engineer` | Pandera schema implementation, validation pipeline wiring |

---

## 6. Implementation Tasks

### Task 2.1 — Fear & Greed Schema Definition (`data/metadata/schemas/fear_greed_schema.py`)

Define a Pandera `DataFrameSchema` enforcing:

| Column | Constraint |
|---|---|
| `timestamp` | `datetime64[UTC]` dtype; unique across rows; no nulls |
| `value` | `int64`; range [0, 100] inclusive; no nulls |
| `classification` | One of `{"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}`; no nulls |
| Cross-column | `classification` must be consistent with `value` per regime thresholds |

**Cross-column validation (custom Pandera check):**
- `value in [0, 24]` → `classification == "Extreme Fear"`
- `value in [25, 44]` → `classification == "Fear"`
- `value in [45, 55]` → `classification == "Neutral"`
- `value in [56, 74]` → `classification == "Greed"`
- `value in [75, 100]` → `classification == "Extreme Greed"`

### Task 2.2 — Trader History Schema Definition (`data/metadata/schemas/trader_history_schema.py`)

Define a Pandera `DataFrameSchema` enforcing:

| Column | Constraint |
|---|---|
| `Trade ID` | Non-null; unique across entire DataFrame |
| `Account` | Non-null; non-empty string |
| `Timestamp` | `datetime64[UTC]` dtype; no nulls |
| `Side` | One of `{"Long", "Short"}`; no nulls |
| `Direction` | One of `{"Open", "Close"}`; no nulls |
| `Size USD` | `float64`; ≥ 0.0; no nulls |
| `Execution Price` | `float64`; > 0.0; no nulls |
| `Closed PnL` | `float64`; nullable (null for open positions) |
| `Fee` | `float64`; ≥ 0.0; nullable |
| `Leverage` | `float64`; > 0.0; nullable |

### Task 2.3 — Schema Checks Module (`src/sentiment_trader_analytics/validation/schema_checks.py`)

**Functions:**
- `validate_fear_greed(df: pd.DataFrame, schema: DataFrameSchema) -> ValidationResult`
- `validate_trader_history(df: pd.DataFrame, schema: DataFrameSchema) -> ValidationResult`

`ValidationResult` is a Pydantic model containing:
- `passed: bool`
- `violation_count: int`
- `violations: list[str]` — one entry per violation with column name, row index, value, and rule violated
- `validated_df: Optional[pd.DataFrame]` — the DataFrame if passed, None otherwise

### Task 2.4 — Quality Checks Module (`src/sentiment_trader_analytics/validation/quality_checks.py`)

Additional data quality checks beyond strict schema:
- `check_temporal_coverage(df, config) -> QualityReport` — verifies date range meets minimum coverage requirement
- `check_value_distribution(df, column, expected_range, config) -> QualityReport` — flags unexpected distributional anomalies (e.g., entire month of Extreme Fear with zero variation)
- `check_trader_account_cardinality(df, config) -> QualityReport` — verifies minimum number of distinct accounts for statistical validity

### Task 2.5 — Pipeline Orchestrator (`pipelines/run_validation_pipeline.py`)

- Loads schemas from `data/metadata/schemas/`
- Calls `validate_fear_greed` and `validate_trader_history`
- On validation failure: writes full `ValidationResult.violations` to `logs/validation.log` at `ERROR` level; exits non-zero
- On success: logs validation summary (row count, violation count = 0) at `INFO`; writes schema snapshot to `data/metadata/schemas/`

---

## 7. Verification Commands

```bash
# Run validation pipeline
python pipelines/run_validation_pipeline.py --config configs/base.yaml

# Inspect validation log
cat logs/validation.log

# Run unit tests
pytest tests/unit/test_validation.py -v

# Type check
mypy src/sentiment_trader_analytics/validation/
ruff check src/sentiment_trader_analytics/validation/
```

---

## 8. Go / No-Go Gate

**Proceed to Phase 03 only when ALL of the following are true:**

| Check | Verification |
|---|---|
| Zero schema violations in both datasets | `ValidationResult.passed == True` for both |
| Validation report written to `logs/validation.log` | File present and non-empty |
| Schema snapshot written | Files present in `data/metadata/schemas/` |
| All unit tests pass | `pytest tests/unit/test_validation.py` exits 0 |
| Zero ruff / mypy warnings | Linting and type checking clean |

---

## 9. Unit Test Requirements (`tests/unit/test_validation.py`)

- `test_fear_greed_schema_valid_fixture` — valid fixture passes schema
- `test_fear_greed_schema_value_out_of_range` — value 101 triggers violation
- `test_fear_greed_schema_classification_mismatch` — value 10 with "Greed" triggers cross-column violation
- `test_fear_greed_schema_duplicate_timestamp` — duplicate date triggers uniqueness violation
- `test_trader_schema_null_account` — null Account triggers violation
- `test_trader_schema_duplicate_trade_id` — duplicate Trade ID triggers violation
- `test_trader_schema_negative_size_usd` — negative Size USD triggers violation
- `test_validation_result_structure` — `ValidationResult` fields present and typed correctly

---

*Governed by §5.3. Proceed to [Phase 03](phase_03_preprocessing.md) upon gate clearance.*
