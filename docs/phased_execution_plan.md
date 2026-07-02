# Phased Execution Plan
## Repository: bitcoin-sentiment-trader-performance
### Classification: Internal Engineering Document

---

## 0. Document Purpose

This document is the master roadmap for executing all pipeline stages defined  Section 11. It maps every pipeline stage into a sequenced set of execution phases, defines agent and skill assignments, establishes go/no-go gates between phases, and provides the dependency graph for the full research pipeline.

This document operationalizes the repository engineering standards.

---

## 1. Phase Overview Table

| Phase | Name | Primary Pipeline Stage(s) | Primary Agent | Status |
|---|---|---|---|---|
| 01 | Data Ingestion | §11.1 Ingestion | `python-pro` | ⬜ Pending |
| 02 | Data Validation | §11.2 Validation | `data-scientist` | ⬜ Pending |
| 03 | Preprocessing & Cleaning | §11.3 Cleaning / Preprocessing | `python-pro` | ⬜ Pending |
| 04 | Feature Engineering | §11.4 Feature Engineering | `machine-learning-engineer` | ⬜ Pending |
| 05 | Exploratory Data Analysis | §11.5 EDA | `data-scientist` | ⬜ Pending |
| 06 | Statistical Analysis | §11.6 Statistical Analysis | `data-scientist` | ⬜ Pending |
| 07 | Business Insight Synthesis | §18 Business Insight Standards | `data-scientist` | ⬜ Pending |
| 08 | Machine Learning (Optional) | §11.7 Machine Learning | `machine-learning-engineer` | ⬜ Optional |
| 09 | Visualization & Reporting | §11.8 Reporting | `technical-writer` | ⬜ Pending |
| 10 | Documentation & Architecture | §12 Documentation Standards | `technical-writer` | ⬜ Pending |
| 11 | Testing, CI Validation & Release | §14 Testing Standards | `code-reviewer` | ⬜ Pending |

---

## 2. Dependency Graph

```
Phase 01: Data Ingestion
    │
    ▼
Phase 02: Data Validation
    │
    ▼
Phase 03: Preprocessing & Cleaning
    │
    ▼
Phase 04: Feature Engineering
    │
    ├──────────────────┐
    ▼                  ▼
Phase 05: EDA     Phase 06: Statistical Analysis
    │                  │
    └──────┬───────────┘
           ▼
    Phase 07: Business Insight Synthesis
           │
           ├──────────────────────────────────┐
           ▼                                  ▼
    Phase 08: Machine Learning (Optional)   Phase 09: Visualization & Reporting
           │                                  │
           └──────────┬───────────────────────┘
                      ▼
           Phase 10: Documentation & Architecture
                      │
                      ▼
           Phase 11: Testing, CI Validation & Release
```

**Critical path (non-optional):** 01 → 02 → 03 → 04 → 05 + 06 → 07 → 09 → 10 → 11

---

## 3. Phase Descriptions

### Phase 01 — Data Ingestion
Establish validated file-reading pipelines for both raw datasets (`fear_greed_index.csv` and `historical_data.csv`). Produce standardized, dtype-coerced, metadata-logged DataFrames. Hardening point: checksum verification written to `data/metadata/lineage/`.

**Go/No-Go Gate:** Both loaders execute without error; checksums recorded; initial dtype coercion passes; column names match schema definitions.

### Phase 02 — Data Validation
Apply Pandera-based schema enforcement against the ingested DataFrames. Validate domain constraints (Section 5.3): date parseability, value ranges, uniqueness of Trade ID, non-negative numerics. Any violation halts the pipeline.

**Go/No-Go Gate:** Zero schema violations in the validation report; validation report written to `logs/validation.log`.

### Phase 03 — Preprocessing & Cleaning
Handle missing values (logged by count/proportion per Section 5.4), deduplicate by primary key, normalize all timestamps to UTC, and join sentiment and trader datasets on date. Output written to `data/interim/` then `data/processed/`.

**Go/No-Go Gate:** Post-merge row count falls within ±5% of expected (no fan-out); all timestamps are UTC-aware; null strategy is documented in `docs/methodology.md`.

### Phase 04 — Feature Engineering
Construct all engineered features using pure, stateless functions per Section 6. Features cover: trader metrics (win rate, leverage, PnL rolling windows), sentiment features (lagged values, regime indicators), and time features (hour-of-day, day-of-week, rolling windows). Output written to `data/features/`; schema-validated via Pandera.

**Go/No-Go Gate:** Feature schema validation passes; zero look-ahead bias detected in code review; all features documented in `docs/data_dictionary.md`.

### Phase 05 — Exploratory Data Analysis
Execute standard EDA artifact suite per Section 7: distributions, missingness, outliers, correlations, time-series plots. Outputs to `outputs/figures/eda/` and `outputs/tables/eda/`. Stable analytical logic promoted from notebooks to `src/sentiment_trader_analytics/eda/`.

**Go/No-Go Gate:** All mandatory figure types generated at ≥150 DPI; summary statistics CSVs present in `outputs/tables/eda/`.

### Phase 06 — Statistical Analysis
Execute hypothesis test suite per Section 8: normality pre-tests, two-group and multi-group comparisons across sentiment regimes, correlation analyses, chi-square tests for categorical associations. All results with p-values, effect sizes, and 95% CIs saved to `outputs/tables/statistics/`. Multiple testing corrections applied.

**Go/No-Go Gate:** Every test result includes: test name, statistic, p-value, effect size, CI, sample size; no result missing effect size; corrections documented.

### Phase 07 — Business Insight Synthesis
Translate statistical findings into structured business insights per Section 18's five-part format: Observation, Statistical Evidence, Business Interpretation, Practical Recommendation, Limitation. Synthesized in `notebooks/07_business_insights/` calling into `src/sentiment_trader_analytics/business/insight_generator.py`.

**Go/No-Go Gate:** Every insight adheres to the five-part structure; no insight overstates statistical significance; causation/correlation caveats present.

### Phase 08 — Machine Learning (Optional)
Triggered only after Phase 06 establishes a defensible signal worth modeling. Time-aware train/test split, TimeSeriesSplit CV, baseline-first evaluation, MLflow experiment logging. Model artifacts logged to `experiments/mlruns/`.

**Go/No-Go Gate:** Baseline metric established and documented; no model artifact saved without MLflow record; evaluation metrics match Section 10 requirements.

### Phase 09 — Visualization & Reporting
Assemble final report (`outputs/reports/final_report.pdf`) and executive summary (`outputs/reports/executive_summary.pdf`). All charts at required DPI. Color palette consistent via `SENTIMENT_PALETTE`. No chart manually altered post-export.

**Go/No-Go Gate:** All required figures and tables referenced in the report exist; all charts have title, axis labels, legend, source footnote; PDFs render correctly.

### Phase 10 — Documentation & Architecture
Populate and update `docs/architecture.md`, `docs/methodology.md`, `docs/data_dictionary.md`, `docs/index.md`, and `docs/api_reference/`. Run `generate-api-documentation` command. Ensure README quickstart is current.

**Go/No-Go Gate:** All docs files non-empty and current; API reference generated from docstrings; `docs/methodology.md` records every non-obvious analytical decision.

### Phase 11 — Testing, CI Validation & Release
Execute full test suite (unit + integration). Verify ≥85% line coverage on `src/`. All CI checks green. Tag release with semantic version. Freeze config + data + code state per Section 13.

**Go/No-Go Gate:** All tests pass; coverage ≥85%; ruff/mypy/black/isort green; release tagged; `CHANGELOG` updated.

---

## 4. Agent & Skill Assignment Matrix

| Phase | Primary Agent | Supporting Agent(s) | Primary Skill(s) |
|---|---|---|---|
| 01 | `python-pro` | `debugger` | `senior-data-engineer`, `python-patterns` |
| 02 | `data-scientist` | `python-pro` | `senior-data-scientist`, `senior-data-engineer` |
| 03 | `python-pro` | `data-scientist` | `senior-data-engineer`, `python-patterns` |
| 04 | `machine-learning-engineer` | `python-pro` | `senior-ml-engineer`, `ml-engineer` |
| 05 | `data-scientist` | `python-pro` | `senior-data-scientist` |
| 06 | `data-scientist` | `code-reviewer` | `senior-data-scientist` |
| 07 | `data-scientist` | `technical-writer` | `senior-data-scientist` |
| 08 | `machine-learning-engineer` | `model-evaluator` | `senior-ml-engineer`, `scikit-learn`, `mlops-weights-and-biases` |
| 09 | `technical-writer` | `data-scientist` | `senior-data-scientist` |
| 10 | `technical-writer` | `code-reviewer` | `code-reviewer` |
| 11 | `code-reviewer` | `debugger`, `python-pro` | `code-reviewer`, `python-patterns` |

---

## 5. Configuration Reference

All pipeline parameters are sourced exclusively from `configs/`:

| Pipeline Stage | Config File |
|---|---|
| Ingestion | `configs/pipelines/ingestion.yaml` |
| Validation | `configs/pipelines/validation.yaml` |
| Preprocessing | `configs/pipelines/preprocessing.yaml` |
| Feature Engineering | `configs/pipelines/feature_engineering.yaml` |
| EDA | `configs/pipelines/eda.yaml` |
| Statistical Analysis | `configs/pipelines/statistics.yaml` |
| Machine Learning | `configs/pipelines/ml.yaml` |
| Data Sources | `configs/data/fear_greed.yaml`, `configs/data/trader_history.yaml` |
| Logging | `configs/logging.yaml` |

---

## 6. Output Artifact Registry

| Phase | Output Location | Artifact Type |
|---|---|---|
| 01 | `data/metadata/lineage/` | Checksums, source metadata |
| 02 | `logs/validation.log` | Validation report |
| 03 | `data/interim/`, `data/processed/` | Cleaned / merged datasets |
| 04 | `data/features/`, `data/metadata/schemas/` | Feature store, schema snapshots |
| 05 | `outputs/figures/eda/`, `outputs/tables/eda/` | EDA charts, summary tables |
| 06 | `outputs/tables/statistics/` | Hypothesis test results |
| 07 | `notebooks/07_business_insights/` | Business insight notebooks |
| 08 | `experiments/mlruns/`, `experiments/experiment_configs/` | Model artifacts, configs |
| 09 | `outputs/reports/`, `outputs/figures/`, `outputs/presentation_assets/` | Final reports, charts |
| 10 | `docs/` | Documentation files |
| 11 | `tests/`, `.github/workflows/` | Test suite, CI artifacts |

---

## 7. Release Checklist (Pre-Tag)

Per the project release checklist:

- [ ] `outputs/reports/final_report.pdf` — complete analytical narrative
- [ ] `outputs/reports/executive_summary.pdf` — stakeholder summary
- [ ] `outputs/figures/` — all supporting charts, named per convention
- [ ] `outputs/tables/` — all statistical and summary tables
- [ ] Validated notebooks under `notebooks/` (outputs stripped via `nbstripout`)
- [ ] `README.md` — current with quickstart and directory overview
- [ ] `outputs/presentation_assets/` — stakeholder-facing visuals
- [ ] All CI checks green on release commit
- [ ] Git tag applied with semantic version (e.g., `v1.0.0`)

---

## 8. Related Documents

- [Phase 01: Data Ingestion](phases/phase_01_data_ingestion.md)
- [Phase 02: Data Validation](phases/phase_02_data_validation.md)
- [Phase 03: Preprocessing & Cleaning](phases/phase_03_preprocessing.md)
- [Phase 04: Feature Engineering](phases/phase_04_feature_engineering.md)
- [Phase 05: Exploratory Data Analysis](phases/phase_05_eda.md)
- [Phase 06: Statistical Analysis](phases/phase_06_statistical_analysis.md)
- [Phase 07: Business Insight Synthesis](phases/phase_07_business_insights.md)
- [Phase 08: Machine Learning](phases/phase_08_machine_learning.md)
- [Phase 09: Visualization & Reporting](phases/phase_09_reporting.md)
- [Phase 10: Documentation & Architecture](phases/phase_10_documentation.md)
- [Phase 11: Testing, CI & Release](phases/phase_11_testing_release.md)
