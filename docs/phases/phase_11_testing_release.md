# Phase 11: Testing, CI Validation & Release
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Execute the full automated test suite, verify ≥85% line coverage on `src/`, confirm all CI checks are green, tag a semantic version release, and freeze the reproducible state of data + config + code that produced the final report artifacts.

**Pipeline stage:** Testing, CI Validation & Release

---

## 2. Prerequisites

- [ ] Phase 10 complete (Go/No-Go gate cleared)
- [ ] All unit and integration tests written for every implemented module
- [ ] `tests/fixtures/` populated with deterministic, hand-crafted fixture files
- [ ] `.github/workflows/ci.yml` configured with all CI checks
- [ ] All prior phases' Go/No-Go gates cleared

---

## 3. Standards Alignment

| Standard | Reference |
|---|---|
| Unit tests mirror `src/` structure | §14 |
| Minimum 85% line coverage on `src/` | §14 |
| Integration tests with fixture data (no live data) | §14 |
| Bug fixes ship with regression test | §14 |
| Branch naming and commit message format | §13 |
| Semantic versioning for releases | §13 |
| Pre-commit hooks: black, isort, ruff, mypy, nbstripout | §4 |

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `code-reviewer` | `/agent:code-reviewer` |
| **Supporting** | `debugger`, `python-pro` | `/agent:debugger`, `/agent:python-pro` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `code-reviewer` | `/skill:code-reviewer` | PR analysis, quality gate enforcement, security review |
| `python-patterns` | `/skill:python-patterns` | Test architecture review, async/sync patterns |

---

## 6. Test Suite Requirements

### 6.1 Unit Tests (`tests/unit/`)

One test module per source module, minimum:

| Source Module | Test Module | Min Tests |
|---|---|---|
| `ingestion/fear_greed_loader.py` | `test_ingestion.py` | 6 |
| `ingestion/trader_history_loader.py` | `test_ingestion.py` | 5 |
| `validation/schema_checks.py` | `test_validation.py` | 8 |
| `validation/quality_checks.py` | `test_validation.py` | 3 |
| `preprocessing/cleaning.py` | `test_preprocessing.py` | 6 |
| `preprocessing/merging.py` | `test_preprocessing.py` | 5 |
| `feature_engineering/sentiment_features.py` | `test_feature_engineering.py` | 5 |
| `feature_engineering/trader_features.py` | `test_feature_engineering.py` | 6 |
| `feature_engineering/time_features.py` | `test_feature_engineering.py` | 3 |
| `statistics/hypothesis_tests.py` | `test_statistics.py` | 9 |
| `statistics/correlation_analysis.py` | `test_statistics.py` | 3 |
| `eda/summary_stats.py` | `test_eda.py` | 5 |
| `visualization/plots.py` | `test_eda.py` | 2 |
| `business/insight_generator.py` | `test_business.py` | 4 |
| `ml/training.py` (if Phase 08) | `test_ml.py` | 5 |
| `ml/evaluation.py` (if Phase 08) | `test_ml.py` | 4 |

### 6.2 Integration Tests (`tests/integration/test_full_pipeline.py`)

Full end-to-end pipeline test on small fixture dataset:
- `test_full_pipeline_ingestion_to_features` — runs ingestion → validation → preprocessing → feature engineering on fixture data; verifies feature store schema
- `test_full_pipeline_statistics` — runs statistical pipeline on fixture feature store; verifies all 8 test results present
- `test_full_pipeline_no_data_leakage` — verifies no future data leaks into training features using a synthetic ordered fixture

### 6.3 Fixture Requirements (`tests/fixtures/`)

| Fixture File | Content | Purpose |
|---|---|---|
| `fear_greed_sample.csv` | 30 rows, valid data | Happy path testing |
| `fear_greed_bad_columns.csv` | Missing required column | Error path testing |
| `fear_greed_duplicate_dates.csv` | 2 rows with same date | Deduplication testing |
| `trader_history_sample.csv` | 200 rows, all column types, multiple accounts | Happy path testing |
| `trader_history_null_account.csv` | 1 row with null Account | Validation error testing |
| `trader_history_duplicate_trade_id.csv` | 2 rows with same Trade ID | Deduplication testing |

All fixture files are deterministic, hand-crafted, and never contain production data.

---

## 7. CI Pipeline Verification (`.github/workflows/ci.yml`)

The CI pipeline must run on every push and pull request and enforce:

```bash
# Format check
black --check src/ tests/ pipelines/

# Import sort check
isort --check-only src/ tests/ pipelines/

# Linting (zero warnings)
ruff check src/ tests/ pipelines/

# Type checking (strict on src/)
mypy --strict src/sentiment_trader_analytics/
mypy tests/ pipelines/

# Test suite with coverage
pytest tests/ \
    --cov=src/sentiment_trader_analytics \
    --cov-report=term-missing \
    --cov-fail-under=85

# Notebook output stripped (pre-commit)
nbstripout --check notebooks/**/*.ipynb
```

**All checks must pass before any merge to `main`.**

---

## 8. Release Procedure

### Step 1: Pre-Release Checklist

- [ ] `outputs/reports/final_report.pdf` — exists and renders correctly
- [ ] `outputs/reports/executive_summary.pdf` — exists, ≤ 2 pages
- [ ] `outputs/figures/` — all mandatory figures present
- [ ] `outputs/tables/` — all statistical and summary tables present
- [ ] All notebooks with outputs stripped (`nbstripout` verified)
- [ ] `README.md` — quickstart and directory overview current
- [ ] `outputs/presentation_assets/` — 300 DPI visuals present
- [ ] All CI checks green on release commit

### Step 2: Git Tagging

```bash
# Ensure on main branch with all changes committed
git checkout main
git pull origin main

# Create annotated tag with semantic version
git tag -a v1.0.0 -m "Release v1.0.0: Initial full analysis report

- Complete ingestion through reporting pipeline
- 8 hypothesis tests across 5 sentiment regimes
- Final report and executive summary generated
- All CI checks green"

git push origin v1.0.0
```

### Step 3: Freeze Configuration State

```bash
# Record exact config hash in release notes
git log -1 --format="%H" configs/base.yaml > outputs/reports/config_version.txt
git log -1 --format="%H" > outputs/reports/code_version.txt
```

---

## 9. Verification Commands

```bash
# Full test suite
pytest tests/ --cov=src/sentiment_trader_analytics --cov-report=term-missing -v

# Coverage check
pytest tests/ --cov=src/sentiment_trader_analytics --cov-fail-under=85

# Full CI simulation
black --check src/ tests/ pipelines/
isort --check-only src/ tests/ pipelines/
ruff check src/ tests/ pipelines/
mypy --strict src/sentiment_trader_analytics/
mypy tests/ pipelines/

# Pre-release checklist verification
python -c "
from pathlib import Path
required = [
    'outputs/reports/final_report.pdf',
    'outputs/reports/executive_summary.pdf',
    'outputs/figures/eda/pnl_by_sentiment_boxplot.png',
    'outputs/tables/statistics/all_hypothesis_tests.csv',
]
missing = [p for p in required if not Path(p).exists()]
print('MISSING:', missing if missing else 'None — all present')
"
```

---

## 10. Go / No-Go Gate (Release)

**Tag and release only when ALL of the following are true:**

| Check | Verification |
|---|---|
| All unit tests pass | `pytest tests/unit/` exits 0 |
| All integration tests pass | `pytest tests/integration/` exits 0 |
| Coverage ≥ 85% on `src/` | `--cov-fail-under=85` passes |
| Zero ruff warnings | `ruff check src/` produces no output |
| Zero mypy errors on `src/` | `mypy --strict src/` exits 0 |
| Black and isort clean | Both check commands exit 0 |
| Final report PDF exists | File present, opens correctly |
| Executive summary PDF exists | File present, ≤ 2 pages |
| Notebook outputs stripped | `nbstripout --check` exits 0 |
| Git tag applied | `git tag -l "v*"` shows release tag |

---

*Governed by §13, §19. This is the final phase. Upon gate clearance, the repository is in a released, reproducible state.*
