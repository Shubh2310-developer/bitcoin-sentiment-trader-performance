# Phase 06: Statistical Analysis
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Execute the full hypothesis test suite.: normality pre-tests, two-group and multi-group comparisons across sentiment regimes, correlation analyses, and chi-square tests for categorical associations. All results — with p-values, effect sizes, and 95% CIs — saved to `outputs/tables/statistics/`. Multiple testing corrections applied wherever more than one related hypothesis is tested.

**Pipeline stage:** §11.6 — Statistical Analysis (`run_statistical_pipeline.py`)

**Note:** Phase 06 runs concurrently with Phase 05. Both require Phase 04 completion.

---

## 2. Prerequisites

- [ ] Phase 04 complete (Go/No-Go gate cleared)
- [ ] `data/features/` populated
- [ ] `outputs/tables/statistics/` directory exists
- [ ] `configs/pipelines/statistics.yaml` populated with α, correction method, minimum sample size
- [ ] `docs/methodology.md §4` reviewed and confirmed

---

## 3. Pipeline Stage Alignment

| Attribute | Specification |
|---|---|
| **Input** | `data/features/` |
| **Output** | `outputs/tables/statistics/` (test results with p-values, effect sizes, CIs) |
| **Responsibility** | Execute the hypothesis test suite per §8 |
| **Failure condition** | Insufficient sample size for a requested test must fail gracefully with an explicit "underpowered" flag, not a silent skip |

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `data-scientist` | `/agent:data-scientist` |
| **Supporting** | `code-reviewer` | `/agent:code-reviewer` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `senior-data-scientist` | `/skill:senior-data-scientist` | Statistical test selection, effect size interpretation, multiple testing correction |

---

## 6. Implementation Tasks

### Task 6.1 — Hypothesis Tests Module (`src/sentiment_trader_analytics/statistics/hypothesis_tests.py`)

**Standard result type** (`StatTestResult` Pydantic model):
```
test_name: str
metric: str
groups: list[str]
statistic: float
p_value: float
effect_size: float
effect_size_measure: str
confidence_interval_95: tuple[float, float]
sample_sizes: dict[str, int]
normality_rejected: bool
correction_applied: Optional[str]
corrected_threshold: Optional[float]
underpowered: bool
```

**Functions to implement:**

`test_normality(series: pd.Series, config: StatConfig) -> NormalityResult`
- Shapiro-Wilk for n ≤ 50, D'Agostino-Pearson for n > 50
- Returns: statistic, p-value, method used, normality_rejected flag

`compare_two_groups(group_a: pd.Series, group_b: pd.Series, config: StatConfig) -> StatTestResult`
- Protocol per `docs/methodology.md §4.3`: normality → variance homogeneity → t-test / Welch / Mann-Whitney U
- Effect size: Cohen's d (parametric) or rank-biserial r (non-parametric)
- 95% CI on the difference

`compare_multiple_groups(groups: dict[str, pd.Series], config: StatConfig) -> StatTestResult`
- Protocol per `docs/methodology.md §4.4`: ANOVA or Kruskal-Wallis + post-hoc
- Post-hoc: Tukey HSD (ANOVA) or Dunn's test with Holm-Bonferroni (Kruskal-Wallis)
- Effect size: η² (ANOVA) or ε² (Kruskal-Wallis)

`apply_multiple_testing_correction(results: list[StatTestResult], method: str, config: StatConfig) -> list[StatTestResult]`
- Methods: "bonferroni" or "fdr_bh" (Benjamini-Hochberg)
- Adds `correction_applied` and `corrected_threshold` to each result

### Task 6.2 — Correlation Analysis Module (`src/sentiment_trader_analytics/statistics/correlation_analysis.py`)

`compute_correlation(series_x: pd.Series, series_y: pd.Series, config: StatConfig) -> CorrelationResult`
- Pearson if both approximately normal and linear relationship expected; Spearman otherwise (default Spearman)
- Returns: coefficient, p-value, sample size, 95% CI (Fisher z for Pearson; bootstrap for Spearman), method used

`compute_correlation_matrix(df: pd.DataFrame, columns: list[str], config: StatConfig) -> pd.DataFrame`
- Spearman correlation matrix for all specified columns
- Saved to `outputs/tables/statistics/spearman_correlation_matrix.csv`

### Task 6.3 — Hypothesis Test Suite (Primary Research Questions)

The following are the mandatory test runs aligned to the business objectives (§1.2):

| Test ID | Question | Test Applied | Groups | Output File |
|---|---|---|---|---|
| HT-01 | Does median PnL differ between Fear and Greed? | Two-group comparison | Fear vs. Greed | `ht01_pnl_fear_vs_greed.csv` |
| HT-02 | Does PnL differ across all 5 sentiment regimes? | Multi-group comparison (omnibus) | All 5 regimes | `ht02_pnl_across_regimes.csv` |
| HT-03 | Does leverage differ between Fear and Greed? | Two-group comparison | Fear vs. Greed | `ht03_leverage_fear_vs_greed.csv` |
| HT-04 | Does trade size differ across regimes? | Multi-group comparison | All 5 regimes | `ht04_size_across_regimes.csv` |
| HT-05 | Is trade Side distribution associated with regime? | Chi-square test | Side vs. regime | `ht05_side_vs_regime.csv` |
| HT-06 | Is sentiment value correlated with PnL? | Spearman correlation | sentiment_value vs. Closed PnL | `ht06_sentiment_pnl_correlation.csv` |
| HT-07 | Is rolling win rate correlated with sentiment lag? | Spearman correlation | trader_win_rate_7d vs. sentiment_value_lag_1d | `ht07_winrate_sentiment_correlation.csv` |
| HT-08 | Does win rate differ across regimes? | Multi-group comparison | All 5 regimes | `ht08_winrate_across_regimes.csv` |

Multiple testing correction (Bonferroni for HT-01 to HT-08 as pre-specified family) applied after all tests run.

### Task 6.4 — Pipeline Orchestrator (`pipelines/run_statistical_pipeline.py`)

Sequence:
1. Load `data/features/`; exclude rows with `feature_cold_start == True`
2. Run all normality pre-tests and log results
3. Execute HT-01 through HT-08
4. Apply multiple testing correction to the full result set
5. Save each result to its output CSV in `outputs/tables/statistics/`
6. Save combined results table to `outputs/tables/statistics/all_hypothesis_tests.csv`
7. Log underpowered flags at `WARNING` level

---

## 7. Reporting Standards Compliance

Every test result output must include:
- [ ] Test name
- [ ] Statistic value
- [ ] p-value
- [ ] Effect size AND effect size measure name
- [ ] 95% confidence interval
- [ ] Sample sizes per group
- [ ] Correction applied (if any) and corrected threshold
- [ ] `underpowered` flag if n below minimum

---

## 8. Verification Commands

```bash
python pipelines/run_statistical_pipeline.py --config configs/base.yaml
ls outputs/tables/statistics/
python -c "
import pandas as pd
df = pd.read_csv('outputs/tables/statistics/all_hypothesis_tests.csv')
required = ['test_name','statistic','p_value','effect_size','effect_size_measure','confidence_interval_95']
assert all(c in df.columns for c in required), f'Missing columns: {set(required)-set(df.columns)}'
print('All required columns present')
"
pytest tests/unit/test_statistics.py -v
ruff check src/sentiment_trader_analytics/statistics/
mypy src/sentiment_trader_analytics/statistics/
```

---

## 9. Go / No-Go Gate

| Check | Verification |
|---|---|
| All 8 hypothesis tests completed | `all_hypothesis_tests.csv` has 8 rows |
| Every result has effect size | No null `effect_size` values |
| Multiple testing correction applied | `correction_applied` column non-null |
| Underpowered tests flagged, not silently skipped | `underpowered` column present |
| All output CSVs written | `ls outputs/tables/statistics/` lists 9+ files |
| All unit tests pass | `pytest tests/unit/test_statistics.py` exits 0 |

---

## 10. Unit Test Requirements (`tests/unit/test_statistics.py`)

- `test_normality_shapiro_small_sample` — uses n=30 fixture, returns correct method
- `test_normality_dagostino_large_sample` — uses n=100 fixture, returns correct method
- `test_compare_two_groups_mann_whitney_on_nonnormal` — non-normal data → Mann-Whitney used
- `test_compare_two_groups_ttest_on_normal` — normal + equal variance → t-test used
- `test_compare_multiple_groups_kruskal_wallis` — non-normal data → Kruskal-Wallis used
- `test_effect_size_present_in_all_results` — `StatTestResult.effect_size` always non-null
- `test_underpowered_flag_on_small_n` — n below threshold → `underpowered == True`
- `test_bonferroni_correction` — corrected threshold = α / n_tests
- `test_spearman_correlation_ci` — 95% CI is not null and is a tuple of two floats

---

## 11. Enhanced Analysis: Deeper Pattern Discovery (v2.0)

The core 8 tests establish baseline findings. The following extended analyses are required to build a compelling, nuanced narrative for the hiring assignment:

### ET-01: Regime Transition Analysis
- Identify days where sentiment regime *changes* (e.g., Fear→Greed, Greed→Fear)
- Compare trader PnL, trade frequency, and position sizing in the 24/48/72 hours *before* vs *after* a regime transition
- Test: paired Wilcoxon signed-rank for within-trader behavior
- Effect size: median PnL difference, trade frequency delta
- Output: `et01_regime_transition_analysis.csv`

### ET-02: Regime Duration Effects
- Group trading days by current regime persistence (1–3 days, 4–7 days, 8+ days)
- Test whether prolonged regimes amplify behavioral changes
- Test: Kruskal-Wallis across duration buckets + Dunn's post-hoc
- Output: `et02_regime_duration_effects.csv`

### ET-03: Trader Segmentation (Behavioral Archetypes)
- Cluster traders via k-means on: win rate, avg leverage, trade frequency, avg position size, PnL variability
- Determine optimal k via silhouette score (k=3 to k=6)
- For each cluster, test sentiment sensitivity (PnL in Fear vs Greed per segment)
- Output: `et03_trader_archetypes.csv` + `et03_archetype_sentiment_sensitivity.csv`

### ET-04: Interaction Effects
- Test sentiment × leverage (low/high): does leverage amplify PnL differences?
- Test sentiment × position size (small/large)
- Test sentiment × day type (weekday vs weekend)
- Method: aligned rank transform (ART) or two-way ANOVA
- Output: `et04_interaction_effects.csv`

### ET-05: Extended Lagged Effects
- Cross-correlation of sentiment value with PnL at lags 1d, 3d, 7d, 14d
- Identify optimal prediction horizon per trader segment
- Output: `et05_lagged_correlation_matrix.csv`

### ET-06: Volatility Regime Analysis
- Split traders into high/low PnL-volatility groups
- Test whether high-volatility traders are more sentiment-responsive
- Output: `et06_volatility_sensitivity.csv`

### ET-07: Directional Asymmetry (Long vs Short)
- Does sentiment affect Long PnL differently than Short PnL?
- Does Extreme Greed drive more Long openings?
- Does Extreme Fear drive more Close events?
- Test: chi-square on contingency tables stratified by Side
- Output: `et07_directional_asymmetry.csv`

### Power Analysis & Findings Matrix
- Add post-hoc power calculations for all tests (HT-01–HT-08, ET-01–ET-07)
- Create `findings_matrix.csv`: rows = all tests, columns = significance | effect_size | powered | actionable_yn | narrative_label
- Sort by effect size magnitude (largest first) to identify the most operationally relevant findings

### Updated Verification Commands

```bash
# Verify all extended analysis outputs
ls outputs/tables/statistics/et*.csv
python -c "
import pandas as pd
# Check findings matrix exists
pd.read_csv('outputs/tables/statistics/findings_matrix.csv')
print('Findings matrix OK')
"
```

### Updated Go/No-Go Gate

| Check | Verification |
|---|---|
| All 8 hypothesis tests completed | `all_hypothesis_tests.csv` has 8 rows |
| All 7 extended analyses completed | `et*.csv` files present |
| Findings matrix generated | `findings_matrix.csv` exists |
| Power analysis completed | `power_analysis.csv` exists |
| Every result has effect size | No null `effect_size` values |
| Multiple testing correction applied | `correction_applied` column non-null |
| Underpowered tests flagged | `underpowered` column present |
| All unit tests pass | `pytest tests/unit/test_statistics.py` exits 0 |

---

*Governed by §8. Proceed to [Phase 07](phase_07_business_insights.md) when both Phase 05 and Phase 06 are complete.*
