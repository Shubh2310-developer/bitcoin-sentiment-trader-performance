# Phase 07: Business Insight Synthesis
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Translate statistical findings from Phase 06 into structured business insights that follow the mandatory five-part format Each insight connects a statistical result to a concrete, actionable recommendation for a trading desk or product team, while explicitly acknowledging limitations.

**Pipeline stage:** Business Insight Synthesis (calls into `src/sentiment_trader_analytics/business/`)

---

## 2. Prerequisites

- [ ] Phase 05 complete (Go/No-Go gate cleared)
- [ ] Phase 06 complete (Go/No-Go gate cleared) — all hypothesis test results in `outputs/tables/statistics/`
- [ ] `outputs/tables/eda/pnl_by_regime_stats.csv` present

---

## 3. Five-Part Structure (Binding)

Every insight promoted to the final report must contain all five components:

1. **Observation** — the data pattern stated plainly (no statistical jargon)
2. **Statistical Evidence** — test name, statistic, p-value, effect size, CI
3. **Business Interpretation** — what this means in trading/product terms
4. **Practical Recommendation** — concrete, actionable suggestion
5. **Limitation** — confounders, sample size caveats, correlation vs. causation

Insights missing any component are **not report-ready** and must be returned to Phase 06.

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `data-scientist` | `/agent:data-scientist` |
| **Supporting** | `technical-writer` | `/agent:technical-writer` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-scientist` | `/skill:senior-data-scientist` | Insight synthesis, statistical communication, causation/correlation discipline |

---

## 6. Command Invocations

| Command | Activation | Purpose |
|---|---|---|
| `update-docs` | `/update-docs` | Sync methodology doc if analytical decisions made during insight synthesis |

---

## 7. Implementation Tasks

### Task 7.1 — Insight Generator (`src/sentiment_trader_analytics/business/insight_generator.py`)

**Data model** (`BusinessInsight` Pydantic model):
```
insight_id: str
observation: str
statistical_evidence: StatTestResult  # from Phase 06
business_interpretation: str
practical_recommendation: str
limitation: str
report_ready: bool  # True only if all 5 components are non-empty
```

**Functions:**
- `build_insight(evidence: StatTestResult, interpretation: str, recommendation: str, limitation: str) -> BusinessInsight`
- `validate_insight(insight: BusinessInsight) -> bool` — checks all five components non-empty and `effect_size` is not null
- `export_insights_to_markdown(insights: list[BusinessInsight], output_path: Path) -> None`

### Task 7.2 — Business Insights Notebook

**Notebook:** `notebooks/07_business_insights/07_01_sentiment_pnl_insights.ipynb`

For each of the 8 hypothesis tests (HT-01 through HT-08), produce one `BusinessInsight` object. Draft all five components inline in the notebook. Call `validate_insight` on each. Only report-ready insights proceed to Phase 09.

**Expected insights from Phase 06 results (to be finalized after actual data analysis):**

| Insight ID | Based on Test | Topic |
|---|---|---|
| INS-01 | HT-01 | PnL differential: Fear vs. Greed |
| INS-02 | HT-02 | PnL pattern across all 5 regimes |
| INS-03 | HT-03 | Leverage behavior: Fear vs. Greed |
| INS-04 | HT-04 | Position sizing across regimes |
| INS-05 | HT-05 | Trade direction preference by regime |
| INS-06 | HT-06 | Sentiment-PnL correlation magnitude |
| INS-07 | HT-07 | Win rate predictability from lagged sentiment |
| INS-08 | HT-08 | Win rate stability across regimes |

### Task 7.3 — Non-Significant Results Protocol

If a test result is not statistically significant (p ≥ α after correction) or has a negligible effect size:
- The null result is reported as an insight with `report_ready: True` — it is a valid finding
- Observation: "No statistically significant difference was detected in [metric] across [groups]."
- Practical recommendation: framed as "no evidence to support regime-based adjustment to [behavior]"
- This follows the explicit mandate in §8 and §18: a null result must not be discarded or understated

### Task 7.4 — Causation Disclaimer Template

Every insight must include a variation of the following in its `limitation` field:

> "This analysis establishes a statistical association, not a causal relationship. The observed pattern may be influenced by confounders including overall BTC price trend, market-wide volatility, and the composition of the Hyperliquid trader population during the observation period."

---

## 8. Verification Commands

```bash
# Validate all insights are report-ready
python -c "
from src.sentiment_trader_analytics.business.insight_generator import validate_insight
# Load insights from notebook export and validate each
"

# Run notebook (papermill for reproducibility)
papermill notebooks/07_business_insights/07_01_sentiment_pnl_insights.ipynb \
    outputs/notebooks/07_01_executed.ipynb \
    -p config_path configs/base.yaml

# Check all five components present in exported markdown
grep -c "## Observation" outputs/reports/insights_draft.md
grep -c "## Limitation" outputs/reports/insights_draft.md
```

---

## 9. Go / No-Go Gate

| Check | Verification |
|---|---|
| All 8 insights have `report_ready: True` | `validate_insight` returns True for all |
| No insight overstates significance | Effect size checked — negligible effect noted in business interpretation |
| Every insight contains causation/correlation caveat | `limitation` field non-empty and contains disclaimer |
| Null results reported, not hidden | Insignificant tests produce insights flagged as null results |
| Insights exported to markdown | `outputs/reports/insights_draft.md` exists |

---

## 11. Enhanced Narrative Requirements (v2.0 — Hiring Assignment Optimization)

The core 8 insights establish statistical findings. The following enhancements are required to build a **compelling, interview-winning narrative:**

### 11.1 Narrative Arc (The 3-Act Structure)

Organize the 8 insights + 7 extended findings into a coherent story:

**Act I — "The Setup" (Context & Baseline)**
- INS-03 (Leverage): "Traders don't change leverage based on sentiment"
- INS-04 (Size): "Position sizing is sentiment-independent"
- **Narrative**: Traders are consistent in their core execution parameters regardless of market mood

**Act II — "The Conflict" (Where Sentiment Matters)**
- INS-05 (Side): "Trade direction shifts with regime — more Longs in Greed"
- INS-07 (Win Rate): "Win rate slightly drops when sentiment was high yesterday"
- ET-01 (Transitions): "PnL drops in 24h after a Fear→Greed flip"
- ET-03 (Segments): "One trader cluster is 3x more sentiment-sensitive than others"
- **Narrative**: Sentiment DOES affect behavior, but only in specific, narrow ways and for specific trader types

**Act III — "The Resolution" (Actionable Takeaways)**
- INS-01/02 (PnL Null): "Despite behavioral shifts, overall PnL doesn't change"
- ET-07 (Directional): "Short traders actually benefit from Extreme Fear"
- **Narrative**: The actionable insight is NOT "trade based on sentiment" but rather "know which traders are sentiment-sensitive and manage them accordingly"

### 11.2 Key Narrative Devices

**The "Surprising Null" Hook**
- The most interesting finding is what DIDN'T happen: despite popular belief, traders don't materially change PnL across sentiment regimes
- Use this to establish credibility — the analysis doesn't cherry-pick positive results

**The Archetype Revelation**
- ET-03 trader segmentation is the most visually compelling insight
- Create a table showing each archetype's response to sentiment regimes
- Highlight the "Reactive" archetype vs the "Steady" archetype

**The "One Weird Finding"**
- ET-01 or ET-07 that shows a counterintuitive result (e.g., "Short traders profit during Extreme Fear")
- This becomes the memorable takeaway for non-technical stakeholders

### 11.3 Writing Standards for Interview Impact

| Element | Before (Avoid) | After (Use) |
|---------|---------------|-------------|
| Opening | "A not statistically significant difference was observed..." | "When fear grips the market, most traders keep trading exactly as they always do — and that consistency may be their undoing." |
| Technical depth | "Kruskal-Wallis H=9.25, p=0.026" | "The statistical test found that PnL differences across regimes are small enough that random noise could explain them. This is a genuine null result, not a lack of data." |
| Recommendation | "No evidence to support regime-based adjustment" | "Instead of sentiment-triggered trading rules, prioritize position-sizing discipline — it matters more than market mood." |
| Limitation | "May be influenced by confounders" | "This dataset covers Hyperliquid traders only. On-chain behavior may differ from CEX traders, and the 2024-2025 bull market context limits generalizability to bear markets." |

### 11.4 Additional Deliverables

1. **Executive Summary Narrative** (1 page, non-technical)
   - One sentence: "Trader performance is surprisingly resilient to market sentiment — but the right traders at the right moments show meaningful patterns."
   - Top 3 findings with real dollar figures
   - One action: "Monitor trader archetype, not market mood"

2. **One-Page Visual Summary**
   - A single figure/montage showing the 3-4 most impactful results
   - Suitable for a trading desk slide deck

3. **Trader Archetype Profiles** (supplemental output)
   - Profile each k-means cluster as a character: "The Whale" (high size, low frequency), "The Scaler" (high frequency, low size), "The Sentiment Trader" (high sensitivity)
   - Give each a sentiment response score

### 11.5 Updated Verification Commands

```bash
# Verify narrative completeness
grep -c "The Setup\|The Conflict\|The Resolution" outputs/reports/insights_draft.md
grep -c "archetype\|Archetype\|ARchetype" outputs/reports/insights_draft.md
grep -c "surprising\|Surprising\|counterintuitive" outputs/reports/insights_draft.md

# Verify one-page visual exists
ls outputs/figures/business/one_page_summary.png

# Verify trader archetypes output
ls outputs/tables/statistics/et03_trader_archetypes.csv
```

---

*Governed by Proceed to [Phase 08](phase_08_machine_learning.md) (if signal found) and [Phase 09](phase_09_reporting.md) upon gate clearance.*
