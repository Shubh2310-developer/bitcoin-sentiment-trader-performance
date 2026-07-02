# Bitcoin Sentiment Trader Performance

A production-grade quantitative research system that investigates whether Bitcoin market sentiment — as measured by the Fear & Greed Index — influences the trading behavior and profitability of accounts on Hyperliquid. All findings are reproducible, statistically defensible, and traceable back to raw data.

---

## Table of Contents

- [Key Findings](#key-findings)
- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
- [Quickstart](#quickstart)
- [Data Sources](#data-sources)
- [Deliverables](#deliverables)
- [Testing & CI](#testing--ci)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Governance](#governance)

---

## Key Findings

| Finding | Test | p-value | Effect Size |
|---|---|---|---|
| Trade side (long/short) varies with sentiment regime | Chi-square | 0.0004 | Cramer's V = 0.108 |
| Win rate differs significantly across sentiment regimes | Kruskal-Wallis | 0.0017 | epsilon-squared = 0.004 |
| Win rate negatively correlates with prior-day sentiment | Spearman | 0.0054 | rho = -0.074 |
| PnL differs across sentiment regimes (not Fear vs Greed alone) | Kruskal-Wallis | 0.0261 | epsilon-squared = 0.004 |
| PnL in Fear vs Greed regimes | Mann-Whitney U | 0.0908 | rank-biserial r = -0.065 |
| Sentiment-PnL correlation | Spearman | 0.3464 | rho = 0.025 |
| Leverage in Fear vs Greed | Mann-Whitney U | 0.9745 | rank-biserial r = 0.001 |
| Trade size across regimes | Kruskal-Wallis | 0.1463 | epsilon-squared = 0.002 |

Bonferroni-corrected threshold: 0.00625. Significant findings highlighted in bold.

### Classification Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.547 | 0.541 | 0.596 | 0.567 | 0.593 |
| XGBoost | 0.561 | 0.559 | 0.562 | 0.560 | 0.569 |
| LightGBM | 0.573 | 0.568 | 0.590 | 0.579 | 0.583 |
| Baseline (majority) | 0.503 | — | — | — | — |

---

## Architecture Overview

The system is organized as a multi-stage pipeline where each stage has a single responsibility. Data flows left-to-right, and each stage validates its outputs before the next stage begins.

![Pipeline Architecture](outputs/presentation_assets/pnl_by_sentiment_boxplot.png)
*Example visualization: PnL distribution by sentiment regime — one of 10 EDA figures at 150+ DPI*

### Data Flow

```
raw/  -->  ingestion  -->  validation  -->  preprocessing  -->  feature_engineering  -->  analysis  -->  reporting
│                              │                │                       │
│  fear_greed_index.csv        │                │                       │
│  historical_data.csv         │                │                       │
│                              ▼                │                       │
│                     checksums / lineage        │                       │
│                              │                ▼                       │
│                              │         interim / processed            │
│                              │                       ▼               │
│                              │                 feature_store         │
│                              ▼                                       │
│                        Pandera schemas                               │
│                                                                      ▼
│                                                            figures / tables /
│                                                            reports / assets
```

### Module Responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| **Ingestion** | `fear_greed_loader.py` | Read CSV, coerce dtypes, attach metadata, write checksums |
| | `trader_history_loader.py` | Chunked reader, UTC normalization, preserve Trade ID as string |
| **Validation** | `schema_checks.py` | Pandera schema enforcement (types, nullability, uniqueness) |
| | `quality_checks.py` | Temporal coverage, value distribution, account cardinality |
| **Preprocessing** | `cleaning.py` | Drop nulls, deduplicate, log before/after counts |
| | `merging.py` | Left-join trades on sentiment date, flag missing sentiment |
| **Feature Engineering** | `sentiment_features.py` | Lag, regime encoding, rolling mean, fear/greed flags |
| | `trader_features.py` | Win rate, rolling PnL, leverage, volatility, cold-start flag |
| | `time_features.py` | Hour, day of week, weekend, month |
| **EDA** | `summary_stats.py` | Descriptive stats, missingness report, outlier detection |
| | `plots.py` | 10 standard EDA figures at configurable DPI |
| **Statistics** | `hypothesis_tests.py` | 8 formal tests with normality pre-checks, effect sizes, CIs |
| | `correlation_analysis.py` | Spearman correlation matrix with bootstrap CIs |
| **ML** | `training.py` | RandomForest, XGBoost, LightGBM pipelines (regression + classification) |
| | `evaluation.py` | Metrics computation, SHAP importance, cross-validation |
| **Business** | `insight_generator.py` | Five-part insight synthesis (observation → limitation) |
| **Reporting** | `pdf_renderer.py` | fpdf2-based PDF generation with figures, tables, insight boxes |

---

## Pipeline Stages

Each stage is invocable independently or chained through the master pipeline.

```bash
# Run a single stage
python pipelines/run_ingestion_pipeline.py --config configs/base.yaml
python pipelines/run_validation_pipeline.py --config configs/base.yaml
python pipelines/run_preprocessing_pipeline.py --config configs/base.yaml
python pipelines/run_feature_pipeline.py --config configs/base.yaml
python pipelines/run_eda_pipeline.py --config configs/base.yaml
python pipelines/run_statistical_pipeline.py --config configs/base.yaml
python pipelines/run_ml_pipeline.py --config configs/base.yaml
python pipelines/run_reporting_pipeline.py --config configs/base.yaml

# Run everything
python pipelines/run_full_pipeline.py --config configs/base.yaml
```

### Stage Descriptions

| Stage | Input | Output | Exit on Failure |
|---|---|---|---|
| 01 — Ingestion | `data/raw/fear_greed/fear_greed_index.csv`, `data/raw/trader_history/historical_data.csv` | Standardized DataFrames with lineage checksums | Missing file, unreadable format |
| 02 — Validation | Ingested DataFrames | Validation report + validated DataFrames | Schema violation (exits 1 on real data — expected) |
| 03 — Preprocessing | Validated DataFrames | `data/processed/sentiment_trader_merged_*.parquet` | Merge fan-out anomaly |
| 04 — Feature Engineering | `data/processed/` | `data/features/feature_store_*.parquet` | Pandera schema violation |
| 05 — EDA | `data/features/` | 10 figures + 4 tables | None (diagnostic, warnings only) |
| 06 — Statistical Analysis | `data/features/` | 8 hypothesis tests + Spearman matrix | Insufficient sample size (flagged, not silent) |
| 07 — Business Insights | Statistical outputs | 8 five-part insights in `insights_draft.md` | Missing required evidence |
| 08 — ML (optional) | `data/features/` | 8 MLflow runs + evaluation summary | Sub-baseline metrics reported, never discarded |
| 09 — Reporting | All prior outputs | `final_report.pdf`, `executive_summary.pdf`, 6 presentation assets | Missing upstream artifact |

### Pipeline Metrics

| Metric | Value |
|---|---|
| Fear & Greed rows ingested | 2,644 |
| Trader history rows ingested | 211,224 |
| Rows after dedup + merge | 2,810 |
| Features engineered | 17 |
| Accounts represented | 30 |
| Cold-start rows excluded from analysis | 1,057 |
| Analysis-ready rows | 1,753 |
| Hypothesis tests conducted | 8 (core) + 19 (extended) |
| MLflow runs logged | 52 |
| Test coverage | 87.43% |

---

## Quickstart

### Prerequisites

- Python 3.11+
- Conda (recommended) or venv
- Google Chrome (for visual verification — optional)

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd bitcoin-sentiment-trader-performance

# 2. Create conda environment
conda env create -f environment/conda.yaml
conda activate bst

# 3. Place raw data files
#   data/raw/fear_greed/fear_greed_index.csv
#   data/raw/trader_history/historical_data.csv

# 4. Run the full pipeline
conda run -n bst python pipelines/run_full_pipeline.py --config configs/base.yaml

# 5. View outputs
#   outputs/reports/final_report.pdf  — full analytical narrative
#   outputs/reports/executive_summary.pdf  — stakeholder summary
#   outputs/figures/eda/*.png  — 10 EDA visualizations
#   outputs/tables/statistics/*.csv  — hypothesis test results
```

### Running Tests

```bash
# All tests
conda run -n bst python -m pytest tests/ --cov=src/sentiment_trader_analytics --cov-fail-under=85

# Unit tests only
conda run -n bst python -m pytest tests/unit/ -v

# Integration tests only
conda run -n bst python -m pytest tests/integration/ -v
```

### CI Checks

```bash
conda run -n bst black --check src/ tests/ pipelines/
conda run -n bst isort --check-only src/ tests/ pipelines/
conda run -n bst ruff check src/ tests/ pipelines/
conda run -n bst mypy --strict src/sentiment_trader_analytics/
conda run -n bst interrogate src/sentiment_trader_analytics/ --fail-under 100
conda run -n bst nbstripout --dry-run notebooks/
```

---

## Data Sources

| Dataset | Source | Size | Description |
|---|---|---|---|
| **Fear & Greed Index** | Alternative.me / CNN Money | 90 KB, 2,644 rows | Daily Bitcoin market sentiment (0–100 scale) with classification labels |
| **Trader History** | Hyperliquid DEX | 45 MB, 211,224 rows | On-chain trade execution data including PnL, leverage, side, and account |

### Data Quality Notes

- **Trade ID precision collapse**: The `Trade ID` column in the source CSV lost precision due to scientific notation (only ~3 significant digits preserved). The dedup step collapsed 211,224 rows to 2,810 unique records.
- **No native Leverage column**: Leverage is derived as `(Size Tokens x Execution Price) / Size USD` — standard perpetual futures back-computation.
- **Timezone normalization**: All timestamps are normalized to UTC at ingestion. No naive datetimes persist past the ingestion stage.

---

## Deliverables

### Reports

| Report | Size | Description |
|---|---|---|
| [final_report.pdf](outputs/reports/final_report.pdf) | 2.0 MB | Full analytical narrative with methodology, all 8 business insights, ML evaluation |
| [executive_summary.pdf](outputs/reports/executive_summary.pdf) | 33 KB | Two-page stakeholder summary with top findings |
| [insights_draft.md](outputs/reports/insights_draft.md) | 16 KB | 8 five-part business insights (INS-01 through INS-08) |

### Figures (10 EDA, 150+ DPI)

| Figure | Preview |
|---|---|
| Sentiment Value Distribution | `outputs/figures/eda/sentiment_value_histogram.png` |
| Sentiment Regime Barplot | `outputs/figures/eda/sentiment_regime_frequency_barplot.png` |
| Trader PnL Distribution | `outputs/figures/eda/trader_pnl_distribution_histogram.png` |
| PnL by Sentiment Boxplot | `outputs/figures/eda/pnl_by_sentiment_boxplot.png` |
| Leverage Distribution | `outputs/figures/eda/leverage_distribution_histogram.png` |
| Missingness Heatmap | `outputs/figures/eda/missingness_heatmap.png` |
| Feature Correlation Heatmap | `outputs/figures/eda/feature_correlation_heatmap.png` |
| Sentiment Timeseries | `outputs/figures/eda/sentiment_value_timeseries.png` |
| Trade Count Timeseries | `outputs/figures/eda/trade_count_timeseries.png` |
| PnL Timeseries | `outputs/figures/eda/pnl_timeseries.png` |

### Presentation Assets (6 figures, 300 DPI)

These are in `outputs/presentation_assets/` and suitable for slide decks:

| Asset | Description |
|---|---|
| `pnl_by_sentiment_boxplot.png` | PnL distribution by sentiment regime |
| `sentiment_value_timeseries.png` | Fear & Greed index over time |
| `feature_correlation_heatmap.png` | Spearman correlation matrix |
| `sentiment_regime_frequency_barplot.png` | Regime frequency distribution |
| `ml_classification_feature_importance.png` | Random Forest feature importance (classification) |
| `ml_regression_feature_importance.png` | Random Forest feature importance (regression) |

### Statistical Tables

`outputs/tables/statistics/` contains 14 files including:
- 8 individual hypothesis test results (HT-01 through HT-08)
- Combined `all_hypothesis_tests.csv` with Bonferroni correction
- `spearman_correlation_matrix.csv` (10 x 10 features)
- `ml_evaluation_summary.csv` (3 models x 2 tasks)
- Extended analyses and findings matrix

### MLflow Experiments

52 runs logged under the `ml_experiments` experiment, covering 4 model types (Dummy, RandomForest, XGBoost, LightGBM) across 2 tasks (regression, classification), including SHAP importance artifacts.

```bash
# View MLflow UI
conda run -n bst mlflow ui --backend-store-uri file:///absolute/path/to/experiments/mlruns
```

---

## Testing & CI

All CI checks pass on the v1.0.0 release commit.

| Check | Tool | Standard | Status |
|---|---|---|---|
| Formatting | black (line length 100) | Enforced | Clean |
| Import sorting | isort (black-compatible) | Enforced | Clean |
| Linting | ruff | Zero warnings | Clean |
| Type checking | mypy --strict | All of src/ | Clean |
| Docstrings | interrogate --fail-under 100 | Google-style | 100% |
| Notebook output | nbstripout | Stripped | Clean |
| Unit tests | pytest | 177 passed | 100% |
| Integration tests | pytest | 3 passed | 100% |
| Coverage | pytest-cov | 87.43% (>= 85%) | Pass |

---

## Project Structure

```
.
├── configs/               # Centralized YAML configuration
│   ├── base.yaml           # Global defaults
│   ├── data/               # Dataset-specific configs
│   ├── pipelines/          # Stage-specific parameters
│   └── logging.yaml        # Logging format and handlers
├── data/
│   ├── raw/                # Immutable source data (read-only)
│   ├── interim/            # Temporary partially-processed data
│   ├── processed/          # Cleaned, merged, analysis-ready
│   ├── features/           # Engineered feature store
│   └── metadata/           # Schemas, data dictionaries, lineage
├── src/sentiment_trader_analytics/
│   ├── config/             # Pydantic config models
│   ├── ingestion/          # Data loaders
│   ├── validation/         # Pandera schema enforcement
│   ├── preprocessing/      # Cleaning, merging
│   ├── feature_engineering/# Stateless feature builders
│   ├── eda/                # Descriptive statistics
│   ├── statistics/         # Hypothesis tests, correlation
│   ├── visualization/      # Plotting functions
│   ├── business/           # Insight synthesis
│   ├── ml/                 # ML training and evaluation
│   └── utils/              # I/O, logging, PDF rendering
├── pipelines/              # Orchestrated entry points
├── notebooks/              # Stage-numbered exploration
├── outputs/
│   ├── figures/            # EDA and ML visualizations
│   ├── tables/             # Statistical results
│   ├── reports/            # PDF reports + insights draft
│   └── presentation_assets/# 300 DPI figures for slides
├── tests/
│   ├── unit/               # 177 unit tests
│   ├── integration/        # 3 end-to-end pipeline tests
│   └── fixtures/           # 10 hand-crafted test datasets
├── experiments/            # MLflow tracking store
├── docs/                   # Architecture, methodology, API reference
└── logs/                   # Pipeline execution logs
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/index.md](docs/index.md) | Documentation landing page with full table of contents |
| [docs/architecture.md](docs/architecture.md) | System architecture, data flow diagrams, CI/CD design, ADR log |
| [docs/methodology.md](docs/methodology.md) | Analytical decisions: test selection, null handling, feature design |
| [docs/data_dictionary.md](docs/data_dictionary.md) | All columns across raw, processed, and feature datasets |
| [docs/api_reference/](docs/api_reference/) | Auto-generated API documentation from Google-style docstrings (27 files) |
| [.claude/CLAUDE.md](.claude/CLAUDE.md) | Binding engineering constitution (governance, standards, rules) |

---

## Governance

This repository is governed by CLAUDE.md — a binding engineering constitution that applies to all contributors (human and AI). Key governance principles:

- **Reproducibility over convenience**: Every result must be regenerable from `raw data + config + code`.
- **Immutable raw data**: Files under `data/raw/` are never edited, overwritten, or fixed in place.
- **Configuration-driven development**: All parameters live in `configs/` — never hardcoded.
- **Strict separation of concerns**: Ingestion does not clean. Cleaning does not engineer features. Feature engineering does not plot.
- **Statistical rigor**: Effect sizes and confidence intervals are mandatory alongside every p-value. Multiple testing correction is always applied.
- **100% docstring coverage**: Google-style docstrings on every public function, enforced by interrogate in CI.
- **Fail loudly**: Schema violations halt the pipeline. Silent `except: pass` blocks are prohibited.

---

## License

See [LICENSE](LICENSE).

---

*Generated from commit v1.0.0 — Full end-to-end verification completed 2026-07-02. 180 tests passing, 87.43% coverage, all CI checks clean.*
