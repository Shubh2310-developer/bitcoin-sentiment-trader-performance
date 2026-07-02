# Phase 05: Exploratory Data Analysis
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Generate the mandatory EDA artifact suite covering distributions, missingness, outliers, correlations, and time-series patterns. All figures saved to `outputs/figures/eda/` at ≥150 DPI; summary statistics tables saved to `outputs/tables/eda/` as CSV.

**Pipeline stage:** §11.5 — EDA (`run_eda_pipeline.py`)

**Note:** Phase 05 runs concurrently with Phase 06. Both require Phase 04 completion.

---

## 2. Prerequisites

- [ ] Phase 04 complete (Go/No-Go gate cleared)
- [ ] `data/features/` populated
- [ ] `outputs/figures/eda/` and `outputs/tables/eda/` directories exist
- [ ] `configs/pipelines/eda.yaml` populated
- [ ] `SENTIMENT_PALETTE` defined in `src/sentiment_trader_analytics/visualization/plots.py`

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | `data/features/` and/or `data/processed/` |
| **Output** | `outputs/figures/eda/`, `outputs/tables/eda/` |
| **Responsibility** | Generate the standard EDA artifact set per §7 |
| **Failure condition** | None blocking (EDA is diagnostic); empty/malformed output logged as warning |

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `data-scientist` | `/agent:data-scientist` |
| **Supporting** | `python-pro` | `/agent:python-pro` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-scientist` | `/skill:senior-data-scientist` | EDA design, distributional analysis, outlier identification |

---

## 6. Implementation Tasks

### Task 5.1 — Sentiment Palette (`src/sentiment_trader_analytics/visualization/plots.py`)

```python
SENTIMENT_PALETTE: dict[str, str] = {
    "Extreme Fear": "#c0392b",
    "Fear": "#e74c3c",
    "Neutral": "#95a5a6",
    "Greed": "#27ae60",
    "Extreme Greed": "#145a32",
}
```

This palette is imported by every plotting function. No ad-hoc color values elsewhere.

### Task 5.2 — Summary Stats Module (`src/sentiment_trader_analytics/eda/summary_stats.py`)

- `compute_descriptive_stats(df, columns) -> pd.DataFrame`
- `compute_missingness_report(df) -> pd.DataFrame`
- `compute_outlier_summary(df, column, method, config) -> OutlierReport`

### Task 5.3 — Mandatory EDA Artifacts

All figures: ≥150 DPI, saved to `outputs/figures/eda/`. All tables: CSV, saved to `outputs/tables/eda/`.

| Artifact | Type | Filename |
|---|---|---|
| Sentiment value distribution | Histogram + KDE | `sentiment_value_histogram.png` |
| Sentiment regime frequency | Bar chart | `sentiment_regime_frequency_barplot.png` |
| Trader PnL distribution | Histogram + KDE | `trader_pnl_distribution_histogram.png` |
| PnL by sentiment regime | Box plot | `pnl_by_sentiment_boxplot.png` |
| Leverage distribution | Histogram | `leverage_distribution_histogram.png` |
| Missingness heatmap | Heatmap | `missingness_heatmap.png` |
| Outlier summary | CSV | `outlier_summary.csv` |
| Feature correlation matrix | Heatmap | `feature_correlation_heatmap.png` |
| Sentiment over time | Line chart | `sentiment_value_timeseries.png` |
| Trade activity over time | Line chart | `trade_count_timeseries.png` |
| PnL over time | Line chart | `pnl_timeseries.png` |
| Descriptive stats | CSV | `descriptive_stats.csv` |
| PnL by regime stats | CSV | `pnl_by_regime_stats.csv` |

### Task 5.4 — EDA Notebooks

- `notebooks/04_eda/04_01_sentiment_distribution.ipynb`
- `notebooks/04_eda/04_02_trader_pnl_distribution.ipynb`
- `notebooks/04_eda/04_03_correlation_analysis.ipynb`
- `notebooks/04_eda/04_04_time_series_analysis.ipynb`

Each notebook calls into `src/` functions only. No business logic in notebooks.

### Task 5.5 — Pipeline Orchestrator (`pipelines/run_eda_pipeline.py`)

Calls all plotting and statistics functions programmatically without requiring notebook execution.

---

## 7. Visualization Standards Compliance

Every chart must have: title, axis labels with units, legend where applicable, source/generation-date footnote, `SENTIMENT_PALETTE` used for regime coloring, saved at ≥150 DPI, not manually edited post-export.

---

## 8. Verification Commands

```bash
python pipelines/run_eda_pipeline.py --config configs/base.yaml
ls outputs/figures/eda/
pytest tests/unit/test_eda.py -v
ruff check src/sentiment_trader_analytics/eda/
ruff check src/sentiment_trader_analytics/visualization/
```

---

## 9. Go / No-Go Gate

| Check | Verification |
|---|---|
| All 11 required figures generated | `ls outputs/figures/eda/` lists all |
| All figures at ≥150 DPI | PIL DPI check passes |
| `pnl_by_regime_stats.csv` present | File exists, non-empty |
| `SENTIMENT_PALETTE` used consistently | Code review confirms |
| All unit tests pass | `pytest tests/unit/test_eda.py` exits 0 |

---

## 10. Unit Test Requirements (`tests/unit/test_eda.py`)

- `test_compute_descriptive_stats_shape`
- `test_compute_missingness_report_sorted`
- `test_compute_outlier_summary_iqr`
- `test_sentiment_palette_complete` — all five regimes present
- `test_pnl_boxplot_saves_file` — file created at expected path (mocked I/O)

---

*Governed by §7, §9. Proceed to [Phase 07](phase_07_business_insights.md) when both Phase 05 and Phase 06 are complete.*
