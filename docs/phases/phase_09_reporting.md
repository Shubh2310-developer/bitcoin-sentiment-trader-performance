# Phase 09: Visualization & Reporting
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Assemble all upstream outputs — EDA figures, statistical tables, business insights, and optional ML evaluation — into the two final deliverables: `outputs/reports/final_report.pdf` and `outputs/reports/executive_summary.pdf`. Every figure must meet the DPI, labeling, and palette standards of The final report must contain all insights in the §18 five-part structure.

**Pipeline stage:** §11.8 — Reporting (final stage of `run_full_pipeline.py`)

---

## 2. Prerequisites

- [ ] Phase 07 complete: all insights `report_ready: True`
- [ ] Phase 08 complete (if ML module invoked)
- [ ] All mandatory EDA figures present in `outputs/figures/eda/`
- [ ] All statistical results present in `outputs/tables/statistics/`
- [ ] Presentation assets directory `outputs/presentation_assets/` exists
- [ ] Reporting tool configured (pandoc + LaTeX for PDF, or equivalent)

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | All prior outputs |
| **Output** | `outputs/reports/final_report.pdf`, `outputs/reports/executive_summary.pdf` |
| **Responsibility** | Assemble business insights per §18 into deliverable format |
| **Failure condition** | Missing required upstream artifact (figure/table) halts report generation with a clear list of what's missing |

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `technical-writer` | `/agent:technical-writer` |
| **Supporting** | `data-scientist` | `/agent:data-scientist` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-scientist` | `/skill:senior-data-scientist` | Statistical communication review, insight accuracy verification |

---

## 6. Command Invocations

| Command | Activation | Purpose |
|---|---|---|
| `generate-api-documentation` | `/generate-api-documentation` | Ensure API docs built before final documentation release |
| `docs-maintenance` | `/docs-maintenance` | Final documentation audit before release |

---

## 7. Implementation Tasks

### Task 9.1 — Reporting Figure Generation

For each figure destined for the final report, produce a **300 DPI** version (presentation quality) saved to `outputs/presentation_assets/`. The 150 DPI versions in `outputs/figures/` remain as standard analysis outputs.

**Mandatory report figures:**

| Figure | Source | Report Section |
|---|---|---|
| PnL by sentiment regime boxplot | `outputs/figures/eda/pnl_by_sentiment_boxplot.png` | Results: PnL Analysis |
| Sentiment value time series | `outputs/figures/eda/sentiment_value_timeseries.png` | Data: Sentiment Overview |
| Feature correlation heatmap | `outputs/figures/eda/feature_correlation_heatmap.png` | Results: Feature Relationships |
| Sentiment regime frequency | `outputs/figures/eda/sentiment_regime_frequency_barplot.png` | Data: Regime Distribution |
| ML feature importance (if Phase 08) | `experiments/mlruns/` | ML Results |

### Task 9.2 — Final Report Structure (`outputs/reports/final_report.md` → PDF)

```
1. Executive Summary
2. Introduction & Business Objective
3. Data Sources & Methodology
   3.1 Fear & Greed Index Dataset
   3.2 Hyperliquid Trader History Dataset
   3.3 Data Pipeline & Validation
   3.4 Feature Engineering
   3.5 Statistical Methodology
4. Exploratory Data Analysis
   4.1 Sentiment Regime Distribution
   4.2 Trader PnL Distribution
   4.3 Leverage & Position Sizing
   4.4 Time Series Overview
5. Statistical Analysis Results
   5.1 Hypothesis Test Results (HT-01 through HT-08)
   5.2 Correlation Analysis
   5.3 Multiple Testing Correction Applied
6. Business Insights & Recommendations
   [INS-01 through INS-08, each in §18 five-part structure]
7. Machine Learning Results (if Phase 08)
   7.1 Model Performance vs. Baseline
   7.2 Feature Importance
   7.3 Limitations
8. Limitations & Caveats
9. Appendix: Full Statistical Tables
```

### Task 9.3 — Executive Summary Structure (`outputs/reports/executive_summary.md` → PDF)

Maximum 2 pages. Contains:
- Business question and answer (1 paragraph)
- Top 3 actionable insights with recommendations (bullet points)
- Key statistical findings table (condensed)
- Limitations (2–3 sentences)

### Task 9.4 — Missing Artifact Pre-Check

Before report generation begins, the orchestrator verifies all required artifacts exist:

```python
REQUIRED_ARTIFACTS = [
    "outputs/figures/eda/pnl_by_sentiment_boxplot.png",
    "outputs/tables/statistics/all_hypothesis_tests.csv",
    "outputs/reports/insights_draft.md",
    # ... full list
]

missing = [p for p in REQUIRED_ARTIFACTS if not Path(p).exists()]
if missing:
    logger.error("Missing required artifacts: %s", missing)
    raise ReportGenerationError(f"Cannot generate report. Missing: {missing}")
```

---

## 8. Visualization Standards Final Compliance

Every chart in the final report must pass this checklist:
- [ ] Title present and descriptive
- [ ] X and Y axis labels with units
- [ ] Legend where applicable
- [ ] Source and generation-date footnote
- [ ] `SENTIMENT_PALETTE` used for regime coloring (no deviations)
- [ ] 300 DPI for presentation assets
- [ ] Not manually edited post-export

---

## 9. Verification Commands

```bash
# Pre-check: verify all required artifacts exist
python -c "
from pathlib import Path
required = [
    'outputs/figures/eda/pnl_by_sentiment_boxplot.png',
    'outputs/tables/statistics/all_hypothesis_tests.csv',
]
missing = [p for p in required if not Path(p).exists()]
if missing: print('MISSING:', missing)
else: print('All artifacts present')
"

# Generate final report (pandoc example)
pandoc outputs/reports/final_report.md \
    --pdf-engine=xelatex \
    -o outputs/reports/final_report.pdf

# Verify PDF created
ls -lh outputs/reports/final_report.pdf
ls -lh outputs/reports/executive_summary.pdf

# Verify presentation assets at 300 DPI
python -c "
from PIL import Image
img = Image.open('outputs/presentation_assets/pnl_by_sentiment_boxplot.png')
assert img.info.get('dpi', (0,0))[0] >= 300
print('DPI check passed')
"
```

---

## 10. Go / No-Go Gate

| Check | Verification |
|---|---|
| `final_report.pdf` exists | File present in `outputs/reports/` |
| `executive_summary.pdf` exists | File present in `outputs/reports/` |
| All insights in five-part structure | Every `BusinessInsight.report_ready == True` |
| All report figures at 300 DPI | PIL DPI check for all presentation assets |
| No missing artifact errors | Pre-check script exits clean |
| Executive summary ≤ 2 pages | Word count or page count verified |

---

## 11. Enhanced Narrative Structure (v2.0 — Hiring Assignment Optimization)

### 11.1 Final Report — 3-Act Narrative Structure

The final report must tell a story, not just present findings. Structure as follows:

**Executive Summary** (1 page, ≤400 words)
- One bold claim: "Sentiment alone won't predict your PnL — but ignoring it entirely is also wrong."
- The three most impactful findings (lead with the counterintuitive one)
- One specific, implementable recommendation
- Key stats box: sample size, time span, number of traders, significance count

**Introduction** (Why this matters)
- Hook: "Every trader has a story about 'the market being fearful' or 'greedy.' We tested whether the Fear & Greed Index actually predicts anything real."
- Business question restatement in plain language

**Data & Methodology** (1 page)
- Concise description (reference full methodology doc)
- Highlight: we use proper statistical corrections, effect sizes, and non-parametric tests where normality fails

**EDA Highlights** (1 page)
- Don't show all 11 figures — show the 3 most telling ones:
  1. Sentiment regime frequency (shows imbalance: more Fear than Greed)
  2. PnL by regime boxplot (shows the null result visually)
  3. Missingness heatmap (shows data quality rigor)
- Commentary on each, not just caption

**Statistical Results — "The Story"** (2-3 pages)
- **Section A: What doesn't change** (HT-01, HT-03, HT-04, HT-06)
  - Narrative: "Core trading parameters are sentiment-resistant"
  - Best visual: PnL-by-regime boxplot with overlaid means
- **Section B: Where sentiment sneaks in** (HT-05, HT-07, HT-08, ET-01, ET-05)
  - Narrative: "Behavior shifts in narrow, specific ways"
  - Best visual: Regime transition PnL delta chart
- **Section C: The hidden heterogeneity** (ET-03)
  - Narrative: "Not all traders are created equal — archetypes matter"
  - Best visual: Archetype × Sentiment heatmap
- **Findings Matrix**: single table ranking all findings by effect size

**Business Insights** (2 pages)
- Not 8 disconnected paragraphs — a **narrative flow**:
  1. **Insight 1-2** (the null): Core metrics don't change across regimes
  2. **Insight 3-4** (the nuance): But Side and Win Rate do shift slightly
  3. **Insight 5-6** (the surprise): Transitions and archetypes reveal hidden patterns
  4. **Insight 7-8** (the action): So here's what you should actually do
- Each insight uses the five-part format but written in **plain English first**, with stats in a supporting table

**ML Results — "Why Models Fail"** (1 page)
- Not just metrics — a **failure diagnosis**
- "The best model achieves 54.7% accuracy — here's why that's the ceiling for this dataset"
- Learning curve (shows underfitting, not data shortage)
- Archetype-stratified error rates (shows model works better for some traders)
- **Honesty earns trust** in an interview: explain *why* the problem is hard

**Limitations** (1 page)
- Not a throwaway — a **sophisticated assessment** of what this analysis can and cannot claim
- Each limitation paired with a suggestion for improvement

**Appendix** (reference only)
- Full hypothesis test table
- Model comparison table
- Figures not shown in main body

### 11.2 Executive Summary — High Impact

Structure (≤2 pages):

```
Page 1:
┌─────────────────────────────────────────┐
│  HEADLINE: One bold sentence             │
│  The 3 Key Findings (with $ amounts)     │
│  Key Stats Box                           │
│  Trader Archetype: surprise finding      │
└─────────────────────────────────────────┘
Page 2:
┌─────────────────────────────────────────┐
│  One-Page Visual Summary (the 4 best    │
│  figures arranged in a grid)            │
│  1-sentence caption under each          │
│  "So What?" box with recommendation     │
└─────────────────────────────────────────┘
```

### 11.3 Updated Verification Commands

```bash
# Verify 3-act structure
grep -c "What doesn't change\|Where sentiment sneaks\|hidden heterogeneity" outputs/reports/final_report.pdf 2>/dev/null || \
python -c "import PyPDF2; r=PyPDF2.PdfReader('outputs/reports/final_report.pdf'); print(f'{len(r.pages)} pages')"

# Verify executive summary ≤ 2 pages
python -c "import PyPDF2; r=PyPDF2.PdfReader('outputs/reports/executive_summary.pdf'); assert len(r.pages) <= 2, f'{len(r.pages)} pages'; print(f'{len(r.pages)} pages - OK')"

# Verify one-page visual summary exists
ls outputs/figures/business/one_page_visual_summary.png 2>/dev/null || echo "WARN: one-page visual not found"
```

### 11.4 Updated Go/No-Go Gate

| Check | Verification |
|---|---|
| Final report follows 3-act narrative structure | Section headings match "What doesn't change", "Where sentiment sneaks", "Hidden heterogeneity" |
| Executive summary ≤ 2 pages | PDF page count verified |
| One-page visual summary exists | `outputs/figures/business/one_page_visual_summary.png` |
| ML failure diagnosis included | Report contains "Why Models Fail" section |
| All 6 presentation assets at 300 DPI | PIL DPI check ≥ 300 |
| Trader archetype table in report | Report contains "Archetype" section |
| Limitations are sophisticated (not throwaway) | Each limitation paired with improvement suggestion |

---

*Governed by §9, §19. Proceed to [Phase 10](phase_10_documentation.md) upon gate clearance.*
